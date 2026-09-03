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

/// Seal the Candle receipt for `provider_id` and stamp this crate's architecture axes onto it.
///
/// The sealed `prepared.contract` is exactly what every Wan I2V generator returns from
/// `Generator::memory_strategy_contract()`, so the axes must be stamped here rather than only on
/// the per-request contract [`request_contract_for_mode`] builds — otherwise the loaded path
/// publishes `gen_core::wan_i2v_memory`'s empty default (epic SC-22657, E2).
pub fn prepare(spec: &LoadSpec, provider_id: &str) -> gen_core::Result<PreparedWanI2vMemory> {
    let prepared = PreparedWanI2vMemory::prepare(spec, WanI2vBackend::Candle, provider_id)?;
    let facts = architecture_facts(prepared.route);
    Ok(prepared.with_architecture_facts(facts))
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

/// The trunk and autoencoder geometry each Wan I2V route actually runs (epic SC-22657, E2).
///
/// `gen_core::wan_i2v_memory` builds one shared contract for all four routes and publishes
/// [`gen_core::MemoryArchitectureFacts::default`] there, because it is backend-neutral and holds no
/// model config. This crate does, and — unlike the registry's weights-free contract surface — a
/// [`PreparedWanI2vMemory`] only exists for a snapshot that has already been resolved, sealed and
/// digested on disk. So no axis here is inferred from a provider id: the route was resolved from a
/// real snapshot before this is reachable, and the presets named below are the ones this crate's
/// loader instantiates for it.
///
/// `Ti2v5b` is the dense 5B over the z48 autoencoder; the three 14B routes — plain I2V, VACE and
/// VACE-Fun — are the A14B trunk over the z16 one.
///
/// `transformer_blocks` is ONE expert's depth on the dual-expert 14B routes, exactly as the preset
/// states it: a denoise step traverses the low-noise expert or the high-noise one, never both.
pub fn architecture_facts(
    route: gen_core::wan_i2v_memory::WanI2vRoute,
) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let (dit, latent_channels, tiling) = match route {
        gen_core::wan_i2v_memory::WanI2vRoute::Ti2v5b => (
            crate::config::TransformerConfig::ti2v_5b(),
            crate::config::VaeConfig::ti2v_5b().z_dim,
            crate::WAN_Z48_VAE_TILING,
        ),
        gen_core::wan_i2v_memory::WanI2vRoute::I2v14b
        | gen_core::wan_i2v_memory::WanI2vRoute::Vace
        | gen_core::wan_i2v_memory::WanI2vRoute::VaceFun => (
            crate::config::TransformerConfig::i2v_14b(),
            crate::config::Vae16Config::wan21().z_dim,
            crate::WAN_Z16_VAE_TILING,
        ),
    };
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(dit.num_heads),
        // Declared by the preset itself (`dim == num_heads * head_dim`), so it is read rather than
        // re-derived from the product.
        head_dim: af::declared(dit.head_dim),
        transformer_blocks: af::declared(dit.num_layers),
        // `patch` is `(p_t, p_h, p_w)`; the spatial entry is the axis this fact names.
        patch_size: af::declared(dit.patch.1),
        // The autoencoder's own `z_dim` — what it produces and consumes. VACE's wider
        // `vace_in_channels` is a *control* latent the trunk concatenates, not a VAE width.
        latent_channels: af::declared(latent_channels),
        vae_spatial_scale: u32::try_from(tiling.spatial_scale)
            .ok()
            .filter(|scale| *scale != 0),
        vae_temporal_scale: u32::try_from(tiling.temporal_scale)
            .ok()
            .filter(|scale| *scale != 0),
        // Both Wan trunks load and run bf16 (`lib.rs: DIT_DTYPE`).
        activation_dtype_width: af::dtype_width(crate::DIT_DTYPE),
    }
}

/// The per-request contract for one public mode, with this crate's architecture axes published over
/// the backend-neutral default `gen_core::wan_i2v_memory` can only supply.
pub fn request_contract_for_mode(
    prepared: &PreparedWanI2vMemory,
    mode: &str,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    let mut contract = gen_core::wan_i2v_memory::contract_for_mode_key(prepared, mode)?;
    contract.architecture_facts = architecture_facts(prepared.route);
    Ok(contract)
}

pub fn begin_request<'a>(
    prepared: &'a PreparedWanI2vMemory,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    gen_core::wan_i2v_memory::validate_context(prepared, context)?;
    let request_contract = request_contract_for_mode(prepared, context.mode.as_key())?;
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
///
/// The rung→policy mapping itself is **not** restated here: it is
/// [`gen_core::wan_i2v_memory::load_policy_for_selection`], the shared declaration both backends
/// read. A hand-written `match` on the memory block drifted from it — `stage_residency` was routed
/// to `Sequential` but `tile_vae_decode` fell through to the `Resident` arm, so an admitted
/// BoundedDecode request ran the resident renderer while the doc above and the shared declaration
/// both said sequential.
pub fn selected_offload_policy(
    loaded: OffloadPolicy,
    explicit_resident: bool,
    request: &GenerationRequest,
) -> OffloadPolicy {
    let Some(strategy) = selected_strategy(request) else {
        return loaded;
    };
    match gen_core::wan_i2v_memory::load_policy_for_selection(strategy) {
        OffloadPolicy::Sequential => OffloadPolicy::Sequential,
        // The shared declaration says this rung is resident; only an explicitly admitted Resident
        // route may override the conservative load default to say so.
        OffloadPolicy::Resident if explicit_resident => OffloadPolicy::Resident,
        OffloadPolicy::Resident => loaded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC (SC-22662, review follow-up): the Candle I2V routes publish their own trunk and VAE axes
    /// rather than `MemoryArchitectureFacts::default()`. `gen_core::wan_i2v_memory` has no model
    /// config, so its shared contract can only publish the empty default; this crate overlays the
    /// geometry its loader instantiates.
    ///
    /// The scale pair is asserted against each route's assigned VAE tiling rather than against
    /// literals, so a VAE reassignment cannot leave the published axes describing the old
    /// autoencoder.
    /// Feature-end review (SC-22667, blocker): the contract a **loaded** generator publishes is the
    /// sealed `prepared.contract`, and `gen_core::wan_i2v_memory` can only seal the empty default
    /// axes into it. Every route's `prepare` must therefore hand back a receipt whose contract
    /// already carries this crate's facts — the per-request overlay in `request_contract_for_mode`
    /// is not enough, because `Generator::memory_strategy_contract()` never goes through it.
    ///
    /// Mutation that fails this: `prepare` returning `PreparedWanI2vMemory::prepare(..)` unstamped
    /// (the shape under review) — every route then publishes `is_empty()` facts.
    #[test]
    fn the_sealed_contract_every_loaded_route_publishes_carries_the_trunk_and_vae_axes() {
        use gen_core::wan_i2v_memory::WanI2vRoute;

        for route in [
            WanI2vRoute::Ti2v5b,
            WanI2vRoute::I2v14b,
            WanI2vRoute::Vace,
            WanI2vRoute::VaceFun,
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let spec = gen_core_testkit::wan_i2v::write_candle_snapshot(tmp.path(), route);
            let prepared = prepare(&spec, route.provider_id())
                .unwrap_or_else(|error| panic!("{}: {error}", route.provider_id()));
            let sealed = &prepared.contract;
            assert_eq!(
                sealed.architecture_facts,
                architecture_facts(route),
                "{}: the sealed contract must publish this crate's axes",
                route.provider_id()
            );
            assert!(
                sealed.architecture_facts.has_declared_architecture_axis(),
                "{}: a loaded route must not publish the backend-neutral empty default",
                route.provider_id()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(sealed);
            // The per-request overlay agrees with the seal: neither path may drift from the other.
            let mode = match route {
                WanI2vRoute::Vace => "extend_clip",
                _ => "image_to_video",
            };
            assert_eq!(
                request_contract_for_mode(&prepared, mode)
                    .unwrap()
                    .architecture_facts,
                sealed.architecture_facts
            );
        }
    }

    #[test]
    fn every_i2v_route_publishes_its_trunk_and_vae_axes() {
        use gen_core::wan_i2v_memory::WanI2vRoute;

        let ti2v = architecture_facts(WanI2vRoute::Ti2v5b);
        assert_eq!(
            ti2v,
            gen_core::MemoryArchitectureFacts {
                attention_heads: Some(24),
                head_dim: Some(128),
                transformer_blocks: Some(30),
                patch_size: Some(2),
                latent_channels: Some(48),
                vae_spatial_scale: Some(16),
                vae_temporal_scale: Some(4),
                activation_dtype_width: Some(2),
            }
        );
        assert_eq!(
            (ti2v.vae_spatial_scale, ti2v.vae_temporal_scale),
            (
                Some(crate::WAN_Z48_VAE_TILING.spatial_scale as u32),
                Some(crate::WAN_Z48_VAE_TILING.temporal_scale as u32)
            )
        );

        for route in [WanI2vRoute::I2v14b, WanI2vRoute::Vace, WanI2vRoute::VaceFun] {
            let facts = architecture_facts(route);
            assert_eq!(
                facts,
                gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(40),
                    head_dim: Some(128),
                    // ONE expert's depth: a denoise step traverses one expert, never both.
                    transformer_blocks: Some(40),
                    patch_size: Some(2),
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    vae_temporal_scale: Some(4),
                    activation_dtype_width: Some(2),
                },
                "{} architecture facts",
                route.provider_id()
            );
            assert_eq!(
                (facts.vae_spatial_scale, facts.vae_temporal_scale),
                (
                    Some(crate::WAN_Z16_VAE_TILING.spatial_scale as u32),
                    Some(crate::WAN_Z16_VAE_TILING.temporal_scale as u32)
                )
            );
            assert!(facts.has_declared_architecture_axis());
            assert_ne!(facts, gen_core::MemoryArchitectureFacts::default());
        }
    }

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

        resident.memory = Some(gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(gen_core::wan_i2v_memory::DECODE_TILE_EDGES[0]),
            decode_overlap: Some(gen_core::wan_i2v_memory::DECODE_OVERLAPS[0]),
            ..Default::default()
        });
        assert_eq!(
            selected_strategy(&resident),
            Some(MemoryStrategy::BoundedDecode)
        );
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Sequential, true, &resident),
            OffloadPolicy::Sequential,
            "the cumulative BoundedDecode rung keeps the sequential renderer even on an explicitly \
             Resident-capable route — it must not inherit the Resident override"
        );
        assert_eq!(
            selected_offload_policy(OffloadPolicy::Resident, true, &resident),
            OffloadPolicy::Sequential,
            "BoundedDecode overrides a Resident load default rather than preserving it"
        );
        for strategy in MemoryStrategy::ALL {
            let memory = match strategy {
                MemoryStrategy::Resident => gen_core::GenerationMemory::default(),
                MemoryStrategy::StagedResidency => gen_core::GenerationMemory {
                    stage_residency: true,
                    ..Default::default()
                },
                MemoryStrategy::BoundedDecode => gen_core::GenerationMemory {
                    tile_vae_decode: true,
                    ..Default::default()
                },
                // Wan declares these rungs Missing; `selected_strategy` cannot report them.
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    continue
                }
            };
            let request = GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            };
            assert_eq!(
                selected_offload_policy(OffloadPolicy::Sequential, true, &request),
                gen_core::wan_i2v_memory::load_policy_for_selection(strategy),
                "{strategy:?} must agree with the shared rung -> load-policy declaration"
            );
        }

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
