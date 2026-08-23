//! Provider-owned Candle lifecycle for structurally admitted Wan2.2 I2V requests.

use std::cell::RefCell;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationRequest, LoadSpec, MemoryGeometry, MemoryPhase, MemoryRequestScope,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, OffloadPolicy,
};

pub use gen_core::wan_i2v_memory::{PreparedWanI2vMemory, WanI2vBackend};

pub fn prepare_load_spec(spec: &mut LoadSpec, provider_id: &str) -> gen_core::Result<()> {
    gen_core::wan_i2v_memory::prepare_load_spec(spec, WanI2vBackend::Candle, provider_id)
}

pub fn prepare(spec: &LoadSpec, provider_id: &str) -> gen_core::Result<PreparedWanI2vMemory> {
    PreparedWanI2vMemory::prepare(spec, WanI2vBackend::Candle, provider_id)
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

pub fn selected_decode_cap(request: &GenerationRequest) -> gen_core::Result<Option<u32>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(gen_core::Error::Unsupported(
                "Wan decode parameters require BoundedDecode".to_owned(),
            ));
        }
        return Ok(None);
    }
    if memory.chunk_attention
        || memory.stream_transformer_blocks
        || memory.attention_chunk_size.is_some()
        || memory.transformer_window_size.is_some()
        || memory.transformer_window_component.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "Wan Attention/Transformer rungs are Missing".to_owned(),
        ));
    }
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        gen_core::Error::Unsupported("Wan BoundedDecode requires a tile edge".to_owned())
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("Wan BoundedDecode requires an overlap".to_owned())
    })?;
    if !gen_core::wan_i2v_memory::DECODE_TILE_EDGES.contains(&edge)
        || !gen_core::wan_i2v_memory::DECODE_OVERLAPS.contains(&overlap)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "Wan decode pair {edge}/{overlap} is outside the sealed domain"
        )));
    }
    Ok(Some(edge))
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
    core: candle_gen::request_scope::CandleRequestScopeCore,
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
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    gen_core::wan_i2v_memory::validate_context(prepared, context)?;
    let request_contract =
        gen_core::wan_i2v_memory::contract_for_mode_key(prepared, context.mode.as_key())?;
    let memory = request_contract.generation_memory(&context.selection);
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        prepared.route.provider_id(),
        device,
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
    Ok(Some(Box::new(WanI2vRequestScope {
        core: candle_gen::request_scope::CandleRequestScopeCore::new(config),
        prepared: prepared.clone(),
        selection: context.selection,
        evidence_revision: context.evidence_revision.clone(),
        active: ActiveEvidenceGuard::default(),
    })))
}

pub fn selected_strategy(request: &GenerationRequest) -> Option<MemoryStrategy> {
    request.memory.map(|memory| {
        if memory.tile_vae_decode {
            MemoryStrategy::BoundedDecode
        } else if memory.stage_residency {
            MemoryStrategy::StagedResidency
        } else {
            MemoryStrategy::Resident
        }
    })
}

/// Resolve the actual component-residency policy for this request.
///
/// SceneWorks loads Candle Wan with `Sequential` because that is the conservative historical
/// default. An admitted Resident rung is represented by an explicit all-disabled memory block, so
/// it must override that load default and use the cached resident renderer. Staged and the
/// cumulative BoundedDecode rung keep the sequential renderer. An unadmitted request has no memory
/// block and therefore preserves the load-time policy.
pub fn selected_offload_policy(
    loaded: OffloadPolicy,
    explicit_resident: bool,
    request: &GenerationRequest,
) -> OffloadPolicy {
    match request.memory {
        Some(memory) if memory.stage_residency => OffloadPolicy::Sequential,
        Some(_) if explicit_resident => OffloadPolicy::Resident,
        None => loaded,
        Some(_) => loaded,
    }
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
    fn selected_resident_and_staged_policies_are_operationally_distinct_and_truthful() {
        let mut resident = GenerationRequest {
            memory: Some(gen_core::GenerationMemory::default()),
            ..Default::default()
        };
        assert_eq!(selected_strategy(&resident), Some(MemoryStrategy::Resident));
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Sequential, true, &resident),
            OffloadPolicy::Resident,
            "an explicit Resident rung must override the production Sequential load default"
        );

        resident.memory = Some(gen_core::GenerationMemory {
            stage_residency: true,
            ..Default::default()
        });
        assert_eq!(
            selected_strategy(&resident),
            Some(MemoryStrategy::StagedResidency)
        );
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Sequential, true, &resident),
            OffloadPolicy::Sequential
        );

        resident.memory = None;
        assert_eq!(selected_strategy(&resident), None);
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Sequential, true, &resident),
            OffloadPolicy::Sequential,
            "an unadmitted request must preserve the load policy instead of claiming Resident"
        );

        resident.memory = Some(gen_core::GenerationMemory::default());
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Sequential, false, &resident),
            OffloadPolicy::Sequential,
            "a provider whose contract preserves load defaults must not inherit Wan I2V's override"
        );
    }
}
