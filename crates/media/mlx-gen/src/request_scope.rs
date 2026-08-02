//! Shared MLX memory-strategy request lifecycle.
//!
//! Provider families configure their admitted route and rung parameters here; this core owns the
//! invariants which must not drift between them: request geometry/route identity, exact attention
//! and transformer-window parameters (including ragged tails), terminal evaluation/cache eviction,
//! and the finished-state guard for every lifecycle hook.

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

/// Provider-owned parameters for [`MlxRequestScopeCore`].
pub struct MlxRequestScopeConfig {
    pub provider_id: &'static str,
    pub geometry: MemoryGeometry,
    pub memory: Option<GenerationMemory>,
    pub use_pid: bool,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window: Option<u32>,
    /// The layer count read from the provider's model configuration.
    pub transformer_blocks: u32,
    pub decode_validator: DecodeValidator,
}

impl MlxRequestScopeConfig {
    pub fn new(
        provider_id: &'static str,
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
            geometry,
            memory,
            use_pid,
            attention_chunk_size: None,
            transformer_window: None,
            transformer_blocks,
            decode_validator: Box::new(decode_validator),
        })
    }
}

/// Terminal-cleanup policy. `None` is used only by weights-free behavior probes.
#[derive(Clone, Copy)]
pub enum MlxScopeCleanup {
    Device,
    None,
}

/// The single MLX request-scope implementation shared by adopting provider families.
pub struct MlxRequestScopeCore {
    config: MlxRequestScopeConfig,
    cleanup: CleanupAction,
    state: LifecycleState,
}

impl MlxRequestScopeCore {
    pub fn new(config: MlxRequestScopeConfig) -> Self {
        Self::with_cleanup(config, MlxScopeCleanup::Device)
    }

    pub fn with_cleanup(config: MlxRequestScopeConfig, cleanup: MlxScopeCleanup) -> Self {
        let cleanup: CleanupAction = match cleanup {
            MlxScopeCleanup::Device => Box::new(|| {
                let barrier = mlx_rs::Array::from(0.0_f32);
                barrier.eval().map_err(crate::Error::from)?;
                drop(barrier);
                mlx_rs::memory::clear_cache();
                Ok(())
            }),
            MlxScopeCleanup::None => Box::new(|| Ok(())),
        };
        Self::with_cleanup_action(config, cleanup)
    }

    fn with_cleanup_action(config: MlxRequestScopeConfig, cleanup: CleanupAction) -> Self {
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
            && geometry.batch > 0
            && geometry.batch <= self.config.geometry.batch
        {
            return Ok(());
        }
        Err(Error::Unsupported(format!(
            "{}: hook geometry {}x{}x{} frames={} does not fit admitted {}x{}x{} frames={}",
            self.config.provider_id,
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
            self.config.geometry.width,
            self.config.geometry.height,
            self.config.geometry.batch,
            self.config.geometry.frames
        )))
    }
}

impl MemoryRequestScope for MlxRequestScopeCore {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.config.use_pid {
            return Err(Error::Unsupported(format!(
                "{}: request use_pid={} does not match admitted use_pid={}",
                self.config.provider_id, request.use_pid, self.config.use_pid
            )));
        }
        if request.width != self.config.geometry.width
            || request.height != self.config.geometry.height
            || request.count == 0
            || request.count > self.config.geometry.batch
        {
            return Err(Error::Unsupported(format!(
                "{}: request geometry {}x{}x{} exceeds admitted {}x{}x{}",
                self.config.provider_id,
                request.width,
                request.height,
                request.count,
                self.config.geometry.width,
                self.config.geometry.height,
                self.config.geometry.batch
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

impl Drop for MlxRequestScopeCore {
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
    use crate::gen_core::MemoryRunOutcome;
    use std::cell::Cell;
    use std::rc::Rc;

    fn config(blocks: usize) -> MlxRequestScopeConfig {
        MlxRequestScopeConfig::new(
            "fixture",
            MemoryGeometry {
                width: 64,
                height: 64,
                batch: 3,
                frames: 1,
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

    #[test]
    fn lifecycle_rejects_double_finish_and_every_post_finish_hook() {
        let mut scope = MlxRequestScopeCore::with_cleanup(config(7), MlxScopeCleanup::None);
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
        let mut request = GenerationRequest {
            prompt: "x".into(),
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.leave_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.configure_decode(32, 4, config(7).geometry).is_err());
        assert!(scope.configure_attention(4).is_err());
        assert!(scope.materialize_transformer_window(0, 1).is_err());
    }

    #[test]
    fn transformer_windows_accept_only_full_windows_and_the_ragged_tail() {
        let mut cfg = config(7);
        cfg.transformer_window = Some(3);
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
        scope.materialize_transformer_window(0, 3).unwrap();
        scope.materialize_transformer_window(3, 3).unwrap();
        scope.materialize_transformer_window(6, 1).unwrap();
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        assert!(scope.materialize_transformer_window(1, 3).is_err());
        assert!(scope.materialize_transformer_window(6, 3).is_err());
        assert!(scope.materialize_transformer_window(7, 1).is_err());

        let mut zero = config(7);
        zero.transformer_window = Some(0);
        let mut zero = MlxRequestScopeCore::with_cleanup(zero, MlxScopeCleanup::None);
        assert!(zero.materialize_transformer_window(0, 1).is_err());
    }

    #[test]
    fn decode_geometry_must_match_the_admitted_shape_and_batch_maximum() {
        let mut scope = MlxRequestScopeCore::with_cleanup(config(7), MlxScopeCleanup::None);
        let admitted = config(7).geometry;
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
        let mut scope = MlxRequestScopeCore::with_cleanup_action(
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
            prompt: "x".into(),
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.leave_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.configure_decode(32, 4, config(7).geometry).is_err());
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
    fn every_adopting_provider_constructs_the_shared_core() {
        for (name, source, config_marker) in [
            (
                "z-image",
                include_str!("../mlx-gen-z-image/src/memory_strategy.rs"),
                "ZImageTransformerConfig::turbo().n_layers",
            ),
            (
                "qwen-image",
                include_str!("../mlx-gen-qwen-image/src/memory_strategy.rs"),
                "QwenTransformerConfig::qwen_image_edit().num_layers",
            ),
            (
                "krea-base",
                include_str!("../mlx-gen-krea/src/block_memory_strategy.rs"),
                "Krea2Config::turbo().num_layers",
            ),
            (
                "krea-control",
                include_str!("../mlx-gen-krea/src/memory_strategy.rs"),
                "Krea2Config::turbo().num_layers",
            ),
            (
                "lens",
                include_str!("../mlx-gen-lens/src/memory_strategy.rs"),
                "GptOssConfig::lens().num_layers",
            ),
            (
                "mage",
                include_str!("../mlx-gen-mage/src/model.rs"),
                "MageFlowConfig::mage_flow().depth",
            ),
        ] {
            assert!(
                source.contains("request_scope::MlxRequestScopeCore::"),
                "{name} bypassed MlxRequestScopeCore"
            );
            assert!(
                !source.contains("struct ZImageMemoryScope")
                    && !source.contains("struct QwenMemoryScope")
                    && !source.contains("struct KreaMemoryScope")
                    && !source.contains("struct KreaControlMemoryScope")
                    && !source.contains("struct LensMemoryScope")
                    && !source.contains("struct MageMemoryScope"),
                "{name} retained a duplicate provider scope"
            );
            assert!(
                source.contains(config_marker),
                "{name} stopped deriving its block count from model config"
            );
        }
    }
}
