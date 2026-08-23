//! Provider-owned MLX lifecycle for the structurally admitted Wan2.2 I2V routes.

use std::cell::RefCell;

use mlx_gen::gen_core::{
    self, GenerationRequest, LoadSpec, MemoryGeometry, MemoryPhase, MemoryRequestScope,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
};

pub use gen_core::wan_i2v_memory::{PreparedWanI2vMemory, WanI2vBackend};

pub fn prepare_load_spec(spec: &mut LoadSpec, provider_id: &str) -> gen_core::Result<()> {
    gen_core::wan_i2v_memory::prepare_load_spec(spec, WanI2vBackend::Mlx, provider_id)
}

pub fn prepare(spec: &LoadSpec, provider_id: &str) -> gen_core::Result<PreparedWanI2vMemory> {
    PreparedWanI2vMemory::prepare(spec, WanI2vBackend::Mlx, provider_id)
}

pub fn request_evidence_revision(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<String> {
    gen_core::wan_i2v_memory::request_evidence_revision(prepared, request)
}

thread_local! {
    static ACTIVE_EVIDENCE: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct ActiveEvidenceGuard {
    armed: bool,
}

impl ActiveEvidenceGuard {
    fn arm(&mut self, evidence: String) -> gen_core::Result<()> {
        ACTIVE_EVIDENCE.with(|active| {
            if active.borrow().is_some() {
                return Err(gen_core::Error::Unsupported(
                    "another Wan video request is active on this thread".to_owned(),
                ));
            }
            *active.borrow_mut() = Some(evidence);
            Ok(())
        })?;
        self.armed = true;
        Ok(())
    }

    fn clear(&mut self) {
        if self.armed {
            ACTIVE_EVIDENCE.with(|active| *active.borrow_mut() = None);
            self.armed = false;
        }
    }
}

impl Drop for ActiveEvidenceGuard {
    fn drop(&mut self) {
        self.clear();
    }
}

pub fn validate_active_request(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    if request.memory.is_none() {
        return Ok(());
    }
    let expected = request_evidence_revision(prepared, request)?;
    ACTIVE_EVIDENCE.with(|active| {
        if active.borrow().as_deref() == Some(expected.as_str()) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: request does not match its admitted physical/request identity",
                prepared.route.provider_id()
            )))
        }
    })
}

pub fn staged(request: &GenerationRequest) -> bool {
    request.memory.is_some_and(|memory| memory.stage_residency)
}

pub fn decode_tiling(
    request: &GenerationRequest,
    width: u32,
    height: u32,
    frames: u32,
) -> mlx_gen::Result<Option<mlx_gen::TilingConfig>> {
    let Some(memory) = request.memory else {
        return crate::pipeline::auto_tiling_budgeted(
            height as i32,
            width as i32,
            frames as i32,
            true,
        );
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(mlx_gen::Error::Unsupported(
                "Wan decode parameters require BoundedDecode".to_owned(),
            ));
        }
        return crate::pipeline::auto_tiling_budgeted(
            height as i32,
            width as i32,
            frames as i32,
            true,
        );
    }
    if memory.chunk_attention
        || memory.stream_transformer_blocks
        || memory.attention_chunk_size.is_some()
        || memory.transformer_window_size.is_some()
        || memory.transformer_window_component.is_some()
    {
        return Err(mlx_gen::Error::Unsupported(
            "Wan Attention/Transformer rungs are Missing".to_owned(),
        ));
    }
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        mlx_gen::Error::Unsupported("Wan BoundedDecode requires a tile edge".to_owned())
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        mlx_gen::Error::Unsupported("Wan BoundedDecode requires an overlap".to_owned())
    })?;
    if !gen_core::wan_i2v_memory::DECODE_TILE_EDGES.contains(&edge)
        || !gen_core::wan_i2v_memory::DECODE_OVERLAPS.contains(&overlap)
    {
        return Err(mlx_gen::Error::Unsupported(format!(
            "Wan decode pair {edge}/{overlap} is outside the sealed domain"
        )));
    }
    Ok(Some(mlx_gen::TilingConfig {
        spatial: Some(mlx_gen::tiling::SpatialTiling {
            tile_px: edge as i32,
            overlap_px: overlap as i32,
        }),
        temporal: None,
    }))
}

pub fn safety_check(
    prepared: &PreparedWanI2vMemory,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match gen_core::wan_i2v_memory::validate_context(prepared, context) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

struct WanI2vRequestScope {
    core: mlx_gen::request_scope::MlxRequestScopeCore,
    prepared: PreparedWanI2vMemory,
    selection: gen_core::MemorySelection,
    evidence_revision: String,
    active: ActiveEvidenceGuard,
}

impl MemoryRequestScope for WanI2vRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        let actual = gen_core::wan_i2v_memory::validate_request_evidence(
            &self.prepared,
            request,
            &self.selection,
            &self.evidence_revision,
        )?;
        self.core.configure_request(request)?;
        self.active.arm(actual)?;
        Ok(())
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        self.core.configure_attention(size)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.core.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        let result = self.core.finish(outcome);
        self.active.clear();
        result
    }
}

pub fn begin_request<'a>(
    prepared: &'a PreparedWanI2vMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    gen_core::wan_i2v_memory::validate_context(prepared, context)?;
    let request_contract =
        gen_core::wan_i2v_memory::contract_for_mode_key(prepared, context.mode.as_key())?;
    let memory = request_contract.generation_memory(&context.selection);
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        prepared.route.provider_id(),
        context.geometry,
        memory,
        false,
        if prepared.route == gen_core::wan_i2v_memory::WanI2vRoute::Ti2v5b {
            30
        } else {
            40
        },
        |_pid, edge, overlap| {
            if gen_core::wan_i2v_memory::DECODE_TILE_EDGES.contains(&edge)
                && gen_core::wan_i2v_memory::DECODE_OVERLAPS.contains(&overlap)
            {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(
                    "Wan decode parameters crossed the sealed domain".to_owned(),
                ))
            }
        },
    )?;
    config.default_frames = context.geometry.frames;
    config.load_shape = prepared.contract.load_shape;
    Ok(Some(Box::new(WanI2vRequestScope {
        core: mlx_gen::request_scope::MlxRequestScopeCore::new(config),
        prepared: prepared.clone(),
        selection: context.selection,
        evidence_revision: context.evidence_revision.clone(),
        active: ActiveEvidenceGuard::default(),
    })))
}

pub fn selected_strategy(request: &GenerationRequest) -> MemoryStrategy {
    request.memory.map_or(MemoryStrategy::Resident, |memory| {
        if memory.tile_vae_decode {
            MemoryStrategy::BoundedDecode
        } else if memory.stage_residency {
            MemoryStrategy::StagedResidency
        } else {
            MemoryStrategy::Resident
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_first_last_receipt_clears_on_finish_and_drop() {
        ACTIVE_EVIDENCE.with(|active| *active.borrow_mut() = None);
        {
            let mut guard = ActiveEvidenceGuard::default();
            guard.arm("flf-finish".to_owned()).unwrap();
            assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_some()));
            guard.clear();
            assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_none()));
        }
        {
            let mut guard = ActiveEvidenceGuard::default();
            guard.arm("flf-cancel-or-error".to_owned()).unwrap();
        }
        assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_none()));

        let panic = std::panic::catch_unwind(|| {
            let mut guard = ActiveEvidenceGuard::default();
            guard.arm("replace-person-panic".to_owned()).unwrap();
            panic!("exercise request panic cleanup");
        });
        assert!(panic.is_err());
        assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_none()));
    }

    #[test]
    fn vace_fun_selected_staged_and_decode_controls_are_request_authoritative() {
        let staged_request = GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(staged(&staged_request));
        assert_eq!(
            selected_strategy(&staged_request),
            MemoryStrategy::StagedResidency
        );

        let decode_request = GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(192),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(staged(&decode_request));
        assert_eq!(
            selected_strategy(&decode_request),
            MemoryStrategy::BoundedDecode
        );
        assert!(decode_tiling(&decode_request, 832, 480, 45)
            .unwrap()
            .is_some());
    }

    #[test]
    fn active_receipts_reject_same_thread_overlap_and_isolate_concurrent_threads() {
        ACTIVE_EVIDENCE.with(|active| *active.borrow_mut() = None);
        let mut first = ActiveEvidenceGuard::default();
        first.arm("warm-request".to_owned()).unwrap();
        let mut crossed = ActiveEvidenceGuard::default();
        assert!(crossed.arm("crossed-request".to_owned()).is_err());
        let concurrent = std::thread::spawn(|| {
            let mut independent = ActiveEvidenceGuard::default();
            independent.arm("concurrent-request".to_owned()).unwrap();
            assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_some()));
        });
        concurrent.join().unwrap();
        first.clear();
        assert!(ACTIVE_EVIDENCE.with(|active| active.borrow().is_none()));
    }
}
