//! Shared Candle memory-strategy request lifecycle.

use crate::candle_core::Device;
use crate::gen_core::{
    Error, GenerationMemory, GenerationRequest, MemoryGeometry, MemoryPhase, MemoryRequestScope,
    MemoryRunOutcome, Result,
};

type DecodeValidator = Box<dyn Fn(bool, u32, u32) -> Result<()> + 'static>;
type CleanupAction = Box<dyn FnMut() -> Result<()> + 'static>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LifecycleState {
    Active,
    CleanupPending,
    Complete,
}

/// Provider-owned parameters for [`CandleRequestScopeCore`].
pub struct CandleRequestScopeConfig {
    pub provider_id: &'static str,
    pub device: Device,
    pub geometry: MemoryGeometry,
    /// Provider default used when [`GenerationRequest::frames`] is omitted. Image providers retain
    /// the historical value `1`; video providers set their real default so an omitted request value
    /// is still bound to the exact admitted frame geometry.
    pub default_frames: u32,
    pub memory: Option<GenerationMemory>,
    pub use_pid: bool,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window: Option<u32>,
    /// The layer count read from the provider's model configuration.
    pub transformer_blocks: u32,
    pub decode_validator: DecodeValidator,
}

impl CandleRequestScopeConfig {
    pub fn new(
        provider_id: &'static str,
        device: Device,
        geometry: MemoryGeometry,
        memory: Option<GenerationMemory>,
        use_pid: bool,
        transformer_blocks: usize,
        decode_validator: impl Fn(bool, u32, u32) -> Result<()> + 'static,
    ) -> Result<Self> {
        let transformer_blocks = u32::try_from(transformer_blocks).map_err(|_| {
            Error::Unsupported(format!(
                "{provider_id}: transformer layer count {transformer_blocks} exceeds u32"
            ))
        })?;
        Ok(Self {
            provider_id,
            device,
            geometry,
            default_frames: 1,
            memory,
            use_pid,
            attention_chunk_size: None,
            transformer_window: None,
            transformer_blocks,
            decode_validator: Box::new(decode_validator),
        })
    }
}

/// The Candle twin of MLX's shared request scope. Candle cleanup is a device synchronization; its
/// allocator does not expose MLX's explicit cache-eviction operation.
pub struct CandleRequestScopeCore {
    config: CandleRequestScopeConfig,
    cleanup: CleanupAction,
    state: LifecycleState,
}

impl CandleRequestScopeCore {
    pub fn new(config: CandleRequestScopeConfig) -> Self {
        let device = config.device.clone();
        Self::with_cleanup_action(
            config,
            Box::new(move || device.synchronize().map_err(Error::backend)),
        )
    }

    fn with_cleanup_action(config: CandleRequestScopeConfig, cleanup: CleanupAction) -> Self {
        Self {
            config,
            cleanup,
            state: LifecycleState::Active,
        }
    }

    fn ensure_active(&self) -> Result<()> {
        if self.state != LifecycleState::Active {
            Err(Error::Msg(format!(
                "{}: memory-strategy request scope is already finished",
                self.config.provider_id
            )))
        } else {
            Ok(())
        }
    }

    fn cleanup_pending(&mut self) -> Result<()> {
        debug_assert_eq!(self.state, LifecycleState::CleanupPending);
        match (self.cleanup)() {
            Ok(()) => {
                self.state = LifecycleState::Complete;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn validate_geometry(&self, geometry: MemoryGeometry) -> Result<()> {
        if geometry.width == self.config.geometry.width
            && geometry.height == self.config.geometry.height
            && geometry.frames == self.config.geometry.frames
            && geometry.reference_count == self.config.geometry.reference_count
            && geometry.batch > 0
            && geometry.batch <= self.config.geometry.batch
        {
            return Ok(());
        }
        Err(Error::Unsupported(format!(
            "{}: hook geometry {}x{}x{} frames={} references={} does not fit admitted {}x{}x{} frames={} references={}",
            self.config.provider_id,
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
            geometry.reference_count,
            self.config.geometry.width,
            self.config.geometry.height,
            self.config.geometry.batch,
            self.config.geometry.frames,
            self.config.geometry.reference_count
        )))
    }
}

impl MemoryRequestScope for CandleRequestScopeCore {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.config.use_pid {
            return Err(Error::Unsupported(format!(
                "{}: request use_pid={} does not match admitted use_pid={}",
                self.config.provider_id, request.use_pid, self.config.use_pid
            )));
        }
        // `MemoryGeometry::batch` is the admitted maximum; a request may use any non-zero prefix.
        if request.width != self.config.geometry.width
            || request.height != self.config.geometry.height
            || request.count == 0
            || request.count > self.config.geometry.batch
            || request.frames.unwrap_or(self.config.default_frames) != self.config.geometry.frames
            || request.image_reference_count() != self.config.geometry.reference_count
        {
            return Err(Error::Unsupported(format!(
                "{}: request geometry {}x{}x{} frames={} references={} does not fit admitted {}x{}x{} frames={} references={}",
                self.config.provider_id,
                request.width,
                request.height,
                request.count,
                request.frames.unwrap_or(self.config.default_frames),
                request.image_reference_count(),
                self.config.geometry.width,
                self.config.geometry.height,
                self.config.geometry.batch,
                self.config.geometry.frames,
                self.config.geometry.reference_count
            )));
        }
        request.memory = self.config.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> Result<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> Result<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()> {
        self.ensure_active()?;
        self.validate_geometry(geometry)?;
        (self.config.decode_validator)(self.config.use_pid, tile_edge, overlap)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> Result<()> {
        self.ensure_active()?;
        if self.config.attention_chunk_size == Some(chunk_size) {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "{}: attention chunk size {chunk_size} was not admitted",
                self.config.provider_id
            )))
        }
    }

    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()> {
        self.ensure_active()?;
        let Some(window) = self.config.transformer_window else {
            return Err(Error::Unsupported(format!(
                "{}: bounded transformer residency was not selected",
                self.config.provider_id
            )));
        };
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(Error::Unsupported(format!(
                "{}: transformer window {window} requires a non-zero block count and aligned start, got {block_count} blocks at {first_block}",
                self.config.provider_id
            )));
        }
        if first_block >= self.config.transformer_blocks {
            return Err(Error::Unsupported(format!(
                "{}: transformer window starts at block {first_block}, past the {}-block stack",
                self.config.provider_id, self.config.transformer_blocks
            )));
        }
        let expected = window.min(self.config.transformer_blocks - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
                self.config.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> Result<()> {
        self.ensure_active()?;
        self.state = LifecycleState::CleanupPending;
        self.cleanup_pending()
    }
}

impl Drop for CandleRequestScopeCore {
    fn drop(&mut self) {
        if self.state == LifecycleState::Active {
            self.state = LifecycleState::CleanupPending;
        }
        if self.state == LifecycleState::CleanupPending {
            let _ = self.cleanup_pending();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_core::{Conditioning, Image};
    use std::cell::Cell;
    use std::rc::Rc;

    fn config(blocks: usize) -> CandleRequestScopeConfig {
        CandleRequestScopeConfig::new(
            "fixture",
            Device::Cpu,
            MemoryGeometry {
                width: 64,
                height: 64,
                batch: 3,
                frames: 1,
                reference_count: 0,
            },
            None,
            false,
            blocks,
            |_use_pid, edge, overlap| {
                (edge == 32 && overlap == 4)
                    .then_some(())
                    .ok_or_else(|| Error::Unsupported("wrong decode route".into()))
            },
        )
        .unwrap()
    }

    fn reference() -> Conditioning {
        Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0],
            },
            strength: None,
        }
    }

    #[test]
    fn configured_request_must_match_exact_reference_cardinality() {
        let mut cfg = config(7);
        cfg.geometry.reference_count = 2;
        let mut scope = CandleRequestScopeCore::with_cleanup_action(cfg, Box::new(|| Ok(())));
        let mut exact = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            conditioning: vec![reference(), reference()],
            ..Default::default()
        };
        scope.configure_request(&mut exact).unwrap();

        for conditioning in [
            vec![],
            vec![reference()],
            vec![reference(), reference(), reference(), reference()],
        ] {
            let mut mismatch = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                conditioning,
                ..Default::default()
            };
            assert!(scope.configure_request(&mut mismatch).is_err());
        }
    }

    #[test]
    fn configured_request_resolves_frames_through_the_provider_default() {
        let mut cfg = config(7);
        cfg.geometry.frames = 81;
        cfg.default_frames = 81;
        let mut scope = CandleRequestScopeCore::with_cleanup_action(cfg, Box::new(|| Ok(())));

        for frames in [None, Some(81)] {
            let mut exact = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                frames,
                ..Default::default()
            };
            scope.configure_request(&mut exact).unwrap();
        }
        for frames in [Some(1), Some(77), Some(85)] {
            let mut mismatch = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                frames,
                ..Default::default()
            };
            assert!(scope.configure_request(&mut mismatch).is_err());
        }
    }

    #[test]
    fn configured_scope_rejects_geometry_and_schedule_mutations() {
        let mut cfg = config(7);
        cfg.transformer_window = Some(3);
        let admitted = cfg.geometry;
        let mut scope = CandleRequestScopeCore::with_cleanup_action(cfg, Box::new(|| Ok(())));
        scope
            .configure_decode(
                32,
                4,
                MemoryGeometry {
                    batch: 1,
                    ..admitted
                },
            )
            .unwrap();
        for geometry in [
            MemoryGeometry {
                width: 32,
                ..admitted
            },
            MemoryGeometry {
                height: 32,
                ..admitted
            },
            MemoryGeometry {
                frames: 2,
                ..admitted
            },
            MemoryGeometry {
                reference_count: 1,
                ..admitted
            },
            MemoryGeometry {
                batch: 0,
                ..admitted
            },
            MemoryGeometry {
                batch: 4,
                ..admitted
            },
        ] {
            assert!(scope.configure_decode(32, 4, geometry).is_err());
        }
        scope.materialize_transformer_window(0, 3).unwrap();
        scope.materialize_transformer_window(3, 3).unwrap();
        scope.materialize_transformer_window(6, 1).unwrap();
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        assert!(scope.materialize_transformer_window(1, 3).is_err());

        let mut zero = config(7);
        zero.transformer_window = Some(0);
        let mut zero = CandleRequestScopeCore::with_cleanup_action(zero, Box::new(|| Ok(())));
        assert!(zero.materialize_transformer_window(0, 1).is_err());
    }

    #[test]
    fn cleanup_failure_is_terminal_and_drop_retries_only_cleanup() {
        let calls = Rc::new(Cell::new(0));
        let completed = Rc::new(Cell::new(false));
        let cleanup_calls = Rc::clone(&calls);
        let cleanup_completed = Rc::clone(&completed);
        let mut cfg = config(7);
        cfg.attention_chunk_size = Some(4);
        cfg.transformer_window = Some(3);
        let admitted = cfg.geometry;
        let mut scope = CandleRequestScopeCore::with_cleanup_action(
            cfg,
            Box::new(move || {
                let call = cleanup_calls.get() + 1;
                cleanup_calls.set(call);
                if call == 1 {
                    Err(Error::Msg("injected cleanup failure".into()))
                } else {
                    cleanup_completed.set(true);
                    Ok(())
                }
            }),
        );
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());

        let mut request = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.leave_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.configure_decode(32, 4, admitted).is_err());
        assert!(scope.configure_attention(4).is_err());
        assert!(scope.materialize_transformer_window(0, 3).is_err());
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
        assert_eq!(calls.get(), 1, "hooks must not retry cleanup");

        drop(scope);
        assert_eq!(calls.get(), 2);
        assert!(
            completed.get(),
            "Drop must complete the pending cleanup retry"
        );
    }

    #[test]
    fn z_image_constructs_the_candle_twin_from_model_config() {
        let source = include_str!("../../candle-gen-z-image/src/memory_strategy.rs");
        assert!(source.contains("request_scope::CandleRequestScopeCore::new"));
        assert!(source.contains(
            "candle_transformers::models::z_image::transformer::Config::z_image_turbo().n_layers"
        ));
        assert!(!source.contains("struct ZImageMemoryScope"));
        assert!(!source.contains("const BLOCKS: u32 = 30"));
    }
}
