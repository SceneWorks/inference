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

/// Seal the MLX receipt for `provider_id` and stamp this crate's architecture axes onto it.
///
/// The sealed `prepared.contract` is exactly what every Wan I2V generator returns from
/// `Generator::memory_strategy_contract()`, so the axes must be stamped here rather than only on
/// the per-request contract [`request_contract_for_mode`] builds — otherwise the loaded path
/// publishes `gen_core::wan_i2v_memory`'s empty default (epic SC-22657, E2). The axes are read
/// from the sealed root's own config, the same parse the loader runs.
pub fn prepare(spec: &LoadSpec, provider_id: &str) -> gen_core::Result<PreparedWanI2vMemory> {
    let prepared = PreparedWanI2vMemory::prepare(spec, WanI2vBackend::Mlx, provider_id)?;
    let facts = architecture_facts(prepared.route, Some(prepared.root()));
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

/// The trunk and autoencoder geometry each Wan I2V route actually runs (epic SC-22657, E2).
///
/// `gen_core::wan_i2v_memory` builds one shared contract for all four routes and publishes
/// [`gen_core::MemoryArchitectureFacts::default`] there, because it is backend-neutral and holds no
/// model config. This crate does: the routes are the same two Wan trunks the rest of the crate is
/// built from, so the axes are read off the same presets rather than restated as literals.
///
/// `Ti2v5b` is the dense 5B over the z48 autoencoder; the three 14B routes — plain I2V, VACE and
/// VACE-Fun — are the A14B trunk over the z16 one. VACE's extra control channels do not move any
/// axis here: `vace_in_channels` widens the *control* latent the trunk concatenates, while
/// `vae_z_dim` is what the autoencoder itself produces and consumes.
///
/// `transformer_blocks` is ONE expert's depth on the dual-expert 14B routes, exactly as the
/// preset's `num_layers` states it: a denoise step traverses the low-noise expert or the high-noise
/// one, never both.
///
/// When `root` names the sealed snapshot directory this re-runs the **loader's own parse** for the
/// route — `WanModelConfig::from_model_dir` for TI2V-5B and I2V-14B,
/// `WanVaceConfig::from_model_dir` / `vace_fun_from_model_dir` for the two VACE routes — so the
/// published axes are the snapshot's rather than the preset's. A snapshot that ships no native
/// `config.json` on the two plain routes publishes the route preset: `from_model_dir` would
/// otherwise silently fall back to the TI2V-5B preset for a 14B root, which the 14B loader rejects
/// rather than runs. Without a root (the weights-free surface) the preset is what the loader would
/// start from anyway.
pub fn architecture_facts(
    route: gen_core::wan_i2v_memory::WanI2vRoute,
    root: Option<&std::path::Path>,
) -> gen_core::MemoryArchitectureFacts {
    let wan = loader_config(route, root);
    let (_, patch_h, patch_w) = wan.patch_size;
    let (temporal_stride, spatial_stride, _) = wan.vae_stride;
    gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(wan.num_heads),
        // The exactness-gated helper, NOT `axis(wan.head_dim())` (SC-22667): `head_dim()` is a
        // plain `dim / num_heads`, which rounds a non-uniform stack into a fabricated width and
        // panics on a `"num_heads": 0` snapshot key before `axis` can decline it.
        head_dim: mlx_gen::architecture_facts::head_dim(wan.dim, wan.num_heads),
        transformer_blocks: mlx_gen::architecture_facts::axis(wan.num_layers),
        // A single scalar can only describe a square patch; an anisotropic one has no honest value.
        patch_size: (patch_h == patch_w)
            .then(|| mlx_gen::architecture_facts::axis(patch_h))
            .flatten(),
        latent_channels: mlx_gen::architecture_facts::axis(wan.vae_z_dim),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(spatial_stride),
        vae_temporal_scale: mlx_gen::architecture_facts::axis(temporal_stride),
        // SC-22667 (E2). This declared `HALF_ACTIVATION_WIDTH` for the DiT alone, but gen-core
        // carries ONE scalar for the whole contract — "bytes per element of the activation dtype" —
        // and this contract declares Conditioning and Decode as phases alongside Denoise. Two of
        // those three run f32: `vae.rs` says outright that everything in the autoencoder runs f32
        // (the reference upcasts it, and f32 also sidesteps the bf16 NAX kernel history), and
        // `text_encoder.rs` runs the whole UMT5-XXL with f32 activations, promoting
        // `matmul(f32, bf16)` to an f32 GEMM. The honest single scalar is therefore the widest
        // activation dtype any declared phase runs. Under-declaring it halves the estimate for two
        // real phases, and an under-declared floor admits a render that then OOMs — the failure the
        // ladder exists to prevent. The bf16-native denoise matmuls are unchanged; only their
        // f32 residual stream and the two f32 phases are now described.
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

/// The trunk config the loader for `route` builds from `root`, or the route preset without one.
fn loader_config(
    route: gen_core::wan_i2v_memory::WanI2vRoute,
    root: Option<&std::path::Path>,
) -> crate::config::WanModelConfig {
    use gen_core::wan_i2v_memory::WanI2vRoute;

    let preset = match route {
        WanI2vRoute::Ti2v5b => crate::config::WanModelConfig::wan22_ti2v_5b(),
        WanI2vRoute::I2v14b | WanI2vRoute::Vace | WanI2vRoute::VaceFun => {
            crate::config::WanModelConfig::wan22_i2v_14b()
        }
    };
    let Some(root) = root.filter(|root| root.is_dir()) else {
        return preset;
    };
    let parsed = match route {
        // `model.rs`: `WanModelConfig::from_model_dir(&root)`. Its absent-file fallback is the
        // TI2V-5B preset regardless of route, so only an actually-present `config.json` counts.
        WanI2vRoute::Ti2v5b | WanI2vRoute::I2v14b => root
            .join("config.json")
            .is_file()
            .then(|| crate::config::WanModelConfig::from_model_dir(root).ok())
            .flatten(),
        // `model_vace.rs`: the diffusers `transformer/config.json`, else the native `config.json`.
        WanI2vRoute::Vace => crate::config::WanVaceConfig::from_model_dir(root)
            .ok()
            .map(|vace| vace.base),
        WanI2vRoute::VaceFun => crate::config::WanVaceConfig::vace_fun_from_model_dir(root)
            .ok()
            .map(|vace| vace.base),
    };
    parsed.unwrap_or(preset)
}

/// The per-request contract for one public mode, with this crate's architecture axes published over
/// the backend-neutral default `gen_core::wan_i2v_memory` can only supply.
pub fn request_contract_for_mode(
    prepared: &PreparedWanI2vMemory,
    mode: &str,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    let mut contract = gen_core::wan_i2v_memory::contract_for_mode_key(prepared, mode)?;
    contract.architecture_facts = architecture_facts(prepared.route, Some(prepared.root()));
    Ok(contract)
}

pub fn begin_request<'a>(
    prepared: &'a PreparedWanI2vMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    gen_core::wan_i2v_memory::validate_context(prepared, context)?;
    let request_contract = request_contract_for_mode(prepared, context.mode.as_key())?;
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

    /// AC (SC-22662, review follow-up): the three 14B I2V routes publish the A14B trunk's axes, not
    /// `MemoryArchitectureFacts::default()`. `gen_core::wan_i2v_memory` has no model config, so its
    /// shared contract can only publish the empty default; this crate overlays the real geometry.
    ///
    /// The scale pair is asserted against the provider's own assigned VAE tiling rather than
    /// against literals, exactly as `memory_strategy.rs` does for the z48 route — so a VAE
    /// reassignment cannot leave the published axes describing the old autoencoder.
    /// Feature-end review (SC-22667, blocker): the contract a **loaded** generator publishes is the
    /// sealed `prepared.contract`, and `gen_core::wan_i2v_memory` can only seal the empty default
    /// axes into it. Every route's `prepare` must therefore hand back a receipt whose contract
    /// already carries this crate's facts — the per-request overlay in `request_contract_for_mode`
    /// is not enough, because `Generator::memory_strategy_contract()` never goes through it.
    ///
    /// Mutation that fails this: `prepare` returning `PreparedWanI2vMemory::prepare(..)` unstamped
    /// (the shape under review) — every route then publishes `is_empty()` facts and the
    /// `has_declared_architecture_axis` assertion fires for all four.
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
            let spec = gen_core_testkit::wan_i2v::write_mlx_snapshot(tmp.path(), route);
            let prepared = prepare(&spec, route.provider_id())
                .unwrap_or_else(|error| panic!("{}: {error}", route.provider_id()));
            let sealed = &prepared.contract;
            // The fixture's route-honest `config.json` resolves the same preset the loader would,
            // so the sealed axes equal the preset axes AND the snapshot-parsed axes.
            assert_eq!(
                sealed.architecture_facts,
                architecture_facts(route, None),
                "{}: the sealed contract must publish this crate's axes",
                route.provider_id()
            );
            assert_eq!(
                sealed.architecture_facts,
                architecture_facts(route, Some(prepared.root()))
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

    /// Feature-end review (SC-22667, minor): the I2V axes are read through the loader's own parse
    /// of the materialized root rather than restated from the preset, exactly as
    /// `memory_strategy.rs` does for the T2V route. A root whose native `config.json` declares a
    /// different depth publishes that depth; a root that ships no config on the plain routes keeps
    /// the route preset rather than `from_model_dir`'s route-blind TI2V-5B fallback.
    ///
    /// Mutation that fails this: `architecture_facts` ignoring `root` (the preset-only shape under
    /// review) — the mutated depths below then read back as the preset's 30 / 40.
    #[test]
    fn materialized_i2v_axes_come_from_the_loader_parse_of_the_root() {
        use gen_core::wan_i2v_memory::WanI2vRoute;

        // TI2V-5B / I2V-14B: the native `config.json` `WanModelConfig::from_model_dir` reads.
        for (route, model_type, dim, mutated_layers) in [
            (WanI2vRoute::Ti2v5b, "ti2v", 3072, 7_usize),
            (WanI2vRoute::I2v14b, "i2v", 5120, 11_usize),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            std::fs::write(
                tmp.path().join("config.json"),
                format!(
                    r#"{{"model_type":"{model_type}","model_version":"2.2","dim":{dim},"num_layers":{mutated_layers}}}"#
                ),
            )
            .unwrap();
            let facts = architecture_facts(route, Some(tmp.path()));
            assert_eq!(
                facts.transformer_blocks,
                Some(mutated_layers as u32),
                "{}: the materialized path must publish the snapshot's depth",
                route.provider_id()
            );
            // Every other axis is the preset's, because the snapshot only moved the depth.
            let preset = architecture_facts(route, None);
            assert_eq!(facts.attention_heads, preset.attention_heads);
            assert_eq!(facts.latent_channels, preset.latent_channels);

            // No config at all: the route preset, never `from_model_dir`'s TI2V-5B fallback.
            let bare = tempfile::tempdir().unwrap();
            assert_eq!(architecture_facts(route, Some(bare.path())), preset);
        }

        // VACE / VACE-Fun: the diffusers `transformer/config.json` `WanVaceConfig::from_model_dir`
        // prefers, with the trunk width read off `num_attention_heads * attention_head_dim`.
        for route in [WanI2vRoute::Vace, WanI2vRoute::VaceFun] {
            let tmp = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(tmp.path().join("transformer")).unwrap();
            std::fs::write(
                tmp.path().join("transformer/config.json"),
                r#"{"num_attention_heads":40,"attention_head_dim":128,"num_layers":13}"#,
            )
            .unwrap();
            let facts = architecture_facts(route, Some(tmp.path()));
            assert_eq!(
                (
                    facts.transformer_blocks,
                    facts.attention_heads,
                    facts.head_dim
                ),
                (Some(13), Some(40), Some(128)),
                "{}: the materialized path must publish the snapshot's geometry",
                route.provider_id()
            );
        }
    }

    #[test]
    fn every_i2v_route_publishes_its_trunk_and_vae_axes() {
        use gen_core::wan_i2v_memory::WanI2vRoute;

        let ti2v = architecture_facts(WanI2vRoute::Ti2v5b, None);
        assert_eq!(
            ti2v,
            gen_core::MemoryArchitectureFacts {
                attention_heads: Some(24),
                // 3072 / 24.
                head_dim: Some(128),
                transformer_blocks: Some(30),
                patch_size: Some(2),
                latent_channels: Some(48),
                vae_spatial_scale: Some(16),
                vae_temporal_scale: Some(4),
                activation_dtype_width: Some(4),
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
            let facts = architecture_facts(route, None);
            assert_eq!(
                facts,
                gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(40),
                    // 5120 / 40.
                    head_dim: Some(128),
                    // ONE expert's depth: a denoise step traverses one expert, never both.
                    transformer_blocks: Some(40),
                    patch_size: Some(2),
                    // `vae_z_dim`, not VACE's wider control-latent `vace_in_channels`.
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    vae_temporal_scale: Some(4),
                    activation_dtype_width: Some(4),
                },
                "{} architecture facts",
                route.provider_id()
            );
            // The published pair IS the z16 tiling geometry these routes' VAE assignment declares.
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
