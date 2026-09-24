//! Qwen-Image 2.1 Candle request-memory contract (sc-24112) — the backend twin of
//! `mlx_gen_qwen_image_2_1::memory_strategy`.
//!
//! | rung | strategy | this provider |
//! |---|---|---|
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — `Residency::from_policy` loads the Qwen3 tower, encodes, drops it, then loads the DiT + VAE |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | **implemented** — [`crate::pipeline::decode_tiling`] → `QwenImage21Vae::decode_rgba_tiled` (head once, up-sampling tail tiled and trapezoidally blended) |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | not applicable — the DiT's joint attention is block-causal per segment; the shared chunked-score kernel has no block-causal variant |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | not applicable — no block-streaming loader on this route |
//!
//! # Derived, never measured
//!
//! The **derived** tier and activation tables are owned once, in the MLX crate's
//! `memory_strategy::derived` module, and are not duplicated here: parameter counts and joint-token
//! arithmetic are properties of the model, not of the backend. What *is* backend-specific is the
//! byte width each component is materialized at, and that is priced from the snapshot's own tensor
//! headers in `asset_facts`. As on MLX, nothing here is an on-device observation.

use std::path::Path;

use candle_gen::gen_core::{
    self, LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization, ResidentRequestMemory,
    WeightsSource,
};

use crate::config::MAX_REFERENCE_IMAGES;
use crate::pipeline::DECODE_OVERLAP;
use crate::MODEL_ID;

/// Content fingerprint of this contract. Distinct from the MLX twin's: the two backends never share
/// calibration evidence, and the candle route materializes its components at different widths.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "qwen-image-2-1-candle-derived-2026-09-22-v1";

/// The published decode-tile domain — **derived defaults, pending the epic-end campaign**, not a
/// measured ladder. The crate's shipped default (512) is bracketed by its neighbours at the same
/// 64-px overlap; 768 and 640 are published on the strength of the 2512 route's measured ladder
/// (same 16x autoencoder family, same overlap), not on a run of this VAE. The terminal story's
/// measurement campaign is what turns these into a measured ladder, and nothing here claims
/// otherwise.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 256];

/// Pixels one latent token covers on each axis (the VAE's 16× spatial scale; `patch_size == 1`).
pub const PIXELS_PER_TOKEN: u64 = 16;

// ================================================================================================
// Admission geometry.
// ================================================================================================

/// The geometry envelope this route reports to its consumer contract so SceneWorks can admit or
/// refuse a request **without silently shrinking it**. Identical in meaning to the MLX twin's
/// `AdmissionGeometry`; both are derived from the same frozen preset table and joint-layout limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionGeometry {
    /// Longest side any preset uses (`Capabilities::max_size`).
    pub max_side: u32,
    /// Largest preset **area** in pixels. Not `max_side²`, and not the square default: 2752×1536 is
    /// the widest and 2048×2048 the default, but 2400×1792 is the largest by area. A consumer that
    /// budgeted from `max_size` alone would size the wrong worst case.
    pub max_preset_area: u64,
    /// Latent tokens the largest-area preset contributes.
    pub max_target_image_tokens: u64,
    /// Most reference images the joint layout accepts.
    pub max_reference_images: u32,
    /// Latent tokens **one** reference adds — [`REFERENCE_FIT_TOKENS`], the fitted 1024² grid every
    /// condition image is resized to, **not** the target's own token count. References are extra
    /// image-token blocks, so there is no separate reference budget.
    pub tokens_per_max_reference: u64,
    /// Worst-case joint sequence: the largest-area target plus [`Self::max_reference_images`]
    /// fitted references, plus the table's conditioning tokens.
    pub max_joint_tokens: u64,
    /// Pixels one latent token covers on each axis.
    pub pixels_per_token: u64,
    /// Images one request may ask for; they render sequentially, so this multiplies time, not peak.
    pub max_batch: u32,
}

/// Conditioning tokens the published envelope assumes — a full 256-token prompt through the T2I
/// template, matching the MLX twin's table.
pub const TABLE_CONDITIONING_TOKENS: u64 = 256;

/// Latent tokens **one reference image** contributes to the joint sequence.
///
/// A reference is **not** admitted at the request's own geometry: `reference::prepare` fits every
/// condition image to [`crate::config::OUTPUT_RESOLUTION`] (upstream's `output_resolution` default)
/// before it reaches the vision tower and the VAE, so its latent grid is `(1024/16)² = 4096` tokens
/// whatever the target size. Pricing references at the target's own token count overstates the
/// worst-case joint sequence by more than 3x at the largest preset — a consumer refusing requests
/// that would in fact have fitted.
pub const REFERENCE_FIT_TOKENS: u64 = (crate::config::OUTPUT_RESOLUTION as u64 / PIXELS_PER_TOKEN)
    * (crate::config::OUTPUT_RESOLUTION as u64 / PIXELS_PER_TOKEN);

/// Latent tokens one image of `width × height` contributes.
pub const fn image_tokens(width: u32, height: u32) -> u64 {
    (width as u64 / PIXELS_PER_TOKEN) * (height as u64 / PIXELS_PER_TOKEN)
}

/// The route's admission envelope, derived from the frozen presets and the joint-layout limit.
pub fn admission_geometry() -> AdmissionGeometry {
    let presets = crate::config::PRESETS;
    let max_side = presets
        .iter()
        .map(|p| p.width.max(p.height))
        .max()
        .unwrap_or(0);
    let widest = presets
        .iter()
        .copied()
        .max_by_key(|p| p.width as u64 * p.height as u64)
        .expect("the preset table is non-empty");
    let target_tokens = image_tokens(widest.width, widest.height);
    AdmissionGeometry {
        max_side,
        max_preset_area: widest.width as u64 * widest.height as u64,
        max_target_image_tokens: target_tokens,
        max_reference_images: MAX_REFERENCE_IMAGES as u32,
        tokens_per_max_reference: REFERENCE_FIT_TOKENS,
        max_joint_tokens: TABLE_CONDITIONING_TOKENS
            + target_tokens
            + MAX_REFERENCE_IMAGES as u64 * REFERENCE_FIT_TOKENS,
        pixels_per_token: PIXELS_PER_TOKEN,
        max_batch: crate::descriptor().capabilities.max_count,
    }
}

// ================================================================================================
// Contract.
// ================================================================================================

/// Bytes one resolved component occupies once loaded.
///
/// Each float tensor is priced at the width **its own** loader materializes it at, and each packed
/// (integer) tensor at its stored width — a packed tier is already in resident form on disk, so its
/// u32 codes and its bf16 scale/bias tables price as stored. `keep` selects the tensors the loader
/// actually opens; everything else contributes zero.
fn component_bytes(
    path: &Path,
    float_width: u64,
    keep: &dyn Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|header| keep(&header.name))
        .try_fold(0_u64, |sum, header| {
            let bytes = if header.is_float() {
                header.materialized_bytes(float_width)?
            } else {
                header.data_bytes
            };
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg(format!("{MODEL_ID}: component byte sum overflow"))
            })
        })
}

/// Width `loader::compute_dtype` materializes every float component at on this build's production
/// device (`candle_gen::default_device`) — the DiT, the VAE and the text encoder alike (sc-24114).
/// Public so a test can hold the priced width to the loaded tower's dtype.
pub fn compute_width() -> u64 {
    crate::loader::compute_dtype().size_in_bytes() as u64
}

/// The load-exact component inventory.
///
/// * `transformer/` at [`compute_width`] — bf16 on a GPU build, f32 on the CPU parity lane;
/// * `text_encoder/` over the **loaded** `model.language_model.*` prefix only, at
///   [`compute_width`] too: `loader::load_text_encoder_from` builds the tower's `VarBuilder` at
///   `compute_dtype_on(device)`, which is `compute_dtype()` on the production (GPU) device, so a
///   CUDA/Metal build holds the tower at bf16 exactly as upstream does and as the SceneWorks
///   floors were derived (before sc-24114 it was read at f32 on every backend and
///   priced at 4 B/param — a ~14 GiB overstatement of the resident bf16 footprint). The
///   checkpoint's untied `lm_head` and its whole `model.visual.*` tower are on disk but
///   materialized by nothing on this route, so they contribute zero rather than ~2.4 GB of weights
///   no render touches;
/// * `vae/` at [`compute_width`].
///
/// A packed tier's u32 code tensors are integers and therefore price at their stored width through
/// [`component_bytes`], independent of the float width — which is what makes one function correct
/// for all three tiers.
fn asset_facts(root: &Path) -> gen_core::Result<MemoryAssetFacts> {
    asset_facts_at(root, compute_width())
}

/// `asset_facts` at an explicit float `width` (bytes per element). The production inventory is
/// `asset_facts` at [`compute_width`]; this seam exists so the pricing rule — **every** float
/// component at ONE width, the tower included — can be held on the CPU lane, where the compute
/// width happens to equal the 4 B/param the tower used to be mispriced at.
pub fn asset_facts_at(root: &Path, width: u64) -> gen_core::Result<MemoryAssetFacts> {
    let all = |_: &str| true;
    let transformer = component_bytes(&root.join("transformer"), width, &all)?;
    let conditioning = component_bytes(&root.join("text_encoder"), width, &|name: &str| {
        name.starts_with(crate::loader::TEXT_ENCODER_PREFIX)
    })?;
    let decoder = component_bytes(&root.join("vae"), width, &all)?;
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

/// Architecture axes for one resolved load, read from the snapshot's **own** component configs.
///
/// Gated on a materialized root: a weights-free contract surface names a sentinel path that is
/// deliberately not on disk, so no pipeline is resolved there, no axis is knowable, and every one
/// stays `None` (`MemoryArchitectureFacts::default()`). That is the Candle rule the registry-wide
/// surface walk enforces — an axis on an unresolved surface could only have been inferred from the
/// provider id.
///
/// Every axis is read from `transformer/config.json` and `vae/config.json` rather than from this
/// crate's `production()` constants, so an imported or converted snapshot with a different geometry
/// publishes *its* geometry. The one literal is `patch_size`: 2.1's latents are **unpatched**
/// (`patch_size == 1` in the released config), so a latent token is a latent cell.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let Some(root) = af::snapshot_root(spec) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let dit = af::component_config(root, "transformer");
    let vae = af::component_config(root, "vae");
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::axis_of(dit.as_ref(), &["num_attention_heads"]),
        head_dim: af::axis_of(dit.as_ref(), &["attention_head_dim"]),
        transformer_blocks: af::axis_of(dit.as_ref(), &["num_layers"]),
        patch_size: af::axis_of(dit.as_ref(), &["patch_size"]),
        latent_channels: af::axis_of(dit.as_ref(), &["out_channels"]),
        vae_spatial_scale: af::axis_of(vae.as_ref(), &["scale_factor_spatial"]),
        // Structurally absent: a still-image autoencoder with no temporal axis on this route. The
        // released `vae/config.json` does carry a `scale_factor_temporal`, inherited from the Wan
        // video autoencoder this architecture descends from, but nothing on this route compresses
        // time — the latent has no frame axis — so reading it would publish a frames-per-latent
        // scale for a pipeline that has no frames.
        vae_temporal_scale: None,
        activation_dtype_width: u32::try_from(compute_width()).ok(),
    }
}

/// The numeric tier this generator actually runs.
///
/// `component_precision_floors` is **empty and stays empty**: a tier is a whole-pipeline contract
/// here, so no component is resident above the selected width.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The executable contract for a real load — [`weights_free_memory_strategy_contract`] plus the on-disk inventory.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = weights_free_memory_strategy_contract(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: memory facts require a snapshot directory"
        )));
    };
    // Refuse a tier/request disagreement here too, so a contract is never built for a load that
    // would be served at a different tier than the one it claims.
    crate::quant::resolve_requested_tier(root, spec.quantize)?;
    contract.asset_facts = asset_facts(root)?;
    Ok(contract)
}

/// Declaration-equivalent contract with no asset facts — the registry conformance surface.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if provider_id != MODEL_ID {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: not a Qwen-Image 2.1 Candle memory provider"
        )));
    }
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            // Rung 4 is not implemented here, so nothing materializes a window; the field has no
            // default and must still state what one WOULD do. This route's loader is a plain
            // `VarBuilder` read of already-device-format tensors.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(spec);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedAttention => MemoryStrategySupport::StructurallyNotApplicable {
                reason: "qwen_image_2_1's joint attention is block-causal over per-segment spans; \
                         the shared chunked-score kernel has no block-causal variant on this route"
                    .to_owned(),
            },
            MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "qwen_image_2_1 has no block-streaming loader; every DiT block is \
                             materialized by `loader::load_transformer`"
                        .to_owned(),
                }
            }
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP],
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    Ok(contract)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires an explicit tile edge"
        ))
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires an explicit overlap"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode geometry {edge}/{overlap} is outside the published domain \
             {DECODE_TILE_EDGES:?} at overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

/// The provider safety check: the shared handshake, then this route's own gate.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if !matches!(context.mode, MemoryMode::TextToImage) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: this story's route is text-to-image; got {:?}",
                context.mode
            )));
        }
        // References are PRICED, not refused: sc-24110 advertises Reference/MultiReference on this
        // route, and each condition image is an extra joint-sequence block worth
        // [`REFERENCE_FIT_TOKENS`]. The only reference gate here is the count the joint layout can
        // express; per-image validation belongs to `crate::reference`, so a bad reference stays a
        // request refusal rather than becoming a memory refusal.
        if context.geometry.reference_count as usize > MAX_REFERENCE_IMAGES {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} reference images exceed the {MAX_REFERENCE_IMAGES} the joint layout \
                 can express",
                context.geometry.reference_count
            )));
        }
        if context.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: there is no PiD decoder on this route"
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    Ok(vec![MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            loaded_tier(spec),
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )?,
    )])
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        MODEL_ID,
        candle_gen::default_device()?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::TransformerConfig::production().num_layers,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: there is no PiD decoder on this route"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
        },
    )?;
    Ok(Some(Box::new(
        candle_gen::request_scope::CandleRequestScopeCore::new(config),
    )))
}

/// The shared-ladder registration pair this provider publishes.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: |spec| memory_strategy_contract(MODEL_ID, spec),
    safety_check,
};

/// Behaviour registration: the provider-owned valid fixtures and the executable admission seam.
pub const MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Quant;

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()))
    }

    #[test]
    fn the_contract_declares_the_three_implemented_rungs_and_classifies_the_rest() {
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec()).unwrap();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(
                matches!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::StructurallyNotApplicable { .. }
                ),
                "{strategy:?} must be classified, not implied by the ladder order"
            );
        }
        assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
        assert_ne!(
            contract.calibration.as_ref().unwrap().fingerprint,
            "qwen-image-2-1-mlx-derived-2026-09-22-v1",
            "the two backends must never share calibration evidence"
        );
        // Facts conformance needs a RESOLVED snapshot: this surface names a path that is not on
        // disk, so it correctly publishes no architecture axis at all and the facts check would
        // (rightly) refuse it. `architecture_facts_are_read_from_the_snapshot_and_absent_without_one`
        // runs that check over the real miniature snapshot.
        assert!(contract.architecture_facts.is_empty());
        assert!(weights_free_memory_strategy_contract("qwen_image", &spec()).is_err());
    }

    /// The axes come from the snapshot's own component configs, and a weights-free surface — the
    /// registry's sentinel path — publishes none at all.
    #[test]
    fn architecture_facts_are_read_from_the_snapshot_and_absent_without_one() {
        let tiny = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/tiny-snapshot");
        let contract = weights_free_memory_strategy_contract(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(tiny)),
        )
        .unwrap();
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // The MINIATURE geometry, read from its own `transformer/config.json` — proof the
                // axes are read rather than restated from `production()` (32 heads x 128, 32 layers).
                attention_heads: Some(2),
                head_dim: Some(16),
                transformer_blocks: Some(2),
                // Unpatched latents.
                patch_size: Some(1),
                latent_channels: Some(8),
                // `vae/config.json`'s `scale_factor_spatial`.
                vae_spatial_scale: Some(16),
                vae_temporal_scale: None,
                activation_dtype_width: u32::try_from(compute_width()).ok(),
            }
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The registry's contract surface names a sentinel that is not on disk: nothing is resolved
        // there, so every axis stays undeclared.
        let surface = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert!(weights_free_memory_strategy_contract(MODEL_ID, &surface)
            .unwrap()
            .architecture_facts
            .is_empty());
    }

    /// No tier promotes any component, so no tier's receipt carries a floor.
    ///
    /// *Mutation that reds this:* returning a non-empty `component_precision_floors` from
    /// `loaded_tier` for any selection.
    #[test]
    fn no_tier_records_a_component_precision_floor() {
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut spec = spec();
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            assert!(
                loaded_tier(&spec).component_precision_floors.is_empty(),
                "{quant:?}: a whole-pipeline tier promotes nothing"
            );
        }
    }

    #[test]
    fn references_are_extra_image_tokens_and_the_envelope_reports_them() {
        let g = admission_geometry();
        assert_eq!(g.max_side, 2752);
        // The 4:3 2400x1792 preset, NOT the 2752-wide one and NOT the 2048 square default.
        assert_eq!(g.max_preset_area, 2400 * 1792);
        assert!(g.max_preset_area > 2752 * 1536);
        assert!(g.max_preset_area > 2048 * 2048);
        assert_eq!(g.max_target_image_tokens, 150 * 112);
        assert_eq!(g.max_reference_images, 10);
        // A reference is FITTED to 1024² before it reaches the tower, so it is 64x64 tokens — NOT
        // the target's 150x112. *Mutation that reds this:* `tokens_per_max_reference: target_tokens`.
        assert_eq!(g.tokens_per_max_reference, 64 * 64);
        assert_eq!(g.pixels_per_token, 16);
        assert_eq!(g.max_batch, 8);
        assert_eq!(
            g.max_joint_tokens,
            TABLE_CONDITIONING_TOKENS + 150 * 112 + 10 * 64 * 64
        );
        assert_eq!(g.max_joint_tokens, 58_016, "256 + 16_800 + 10 x 4_096");
    }

    /// The Candle envelope must agree with the MLX twin's on every axis: the geometry a consumer
    /// gates on is a property of the model, not of the backend.
    #[test]
    fn the_admission_envelope_matches_the_mlx_twin() {
        let g = admission_geometry();
        assert_eq!(g.max_side, 2752);
        assert_eq!(g.pixels_per_token, PIXELS_PER_TOKEN);
        assert_eq!(image_tokens(2048, 2048), 128 * 128);
        assert_eq!(image_tokens(2752, 1536), 172 * 96);
    }

    #[test]
    fn references_are_priced_and_only_pid_tier_and_over_count_are_refused() {
        let spec = spec();
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        );

        // References are priced, not refused — up to the count the joint layout can express.
        for count in [1_u32, MAX_REFERENCE_IMAGES as u32] {
            let mut referenced = fixture.context.clone();
            referenced.geometry.reference_count = count;
            referenced.has_reference = true;
            assert_eq!(
                safety_check(&spec, &contract, &referenced),
                MemorySafetyDecision::Accept,
                "{count} references must be admitted and priced"
            );
        }
        let mut too_many = fixture.context.clone();
        too_many.geometry.reference_count = MAX_REFERENCE_IMAGES as u32 + 1;
        too_many.has_reference = true;
        assert!(matches!(
            safety_check(&spec, &contract, &too_many),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut pid = fixture.context.clone();
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&spec, &contract, &pid),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut crossed = fixture.context;
        crossed.selection.tier.quant = Some(Quant::Q4);
        assert!(matches!(
            safety_check(&spec, &contract, &crossed),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// A **transparent** reference must survive the admission -> execution round trip and be
    /// priced like an opaque one (sc-24111).
    ///
    /// Two halves, both load-bearing:
    ///
    /// * `geometry_from_request` (admission) and `memory_reference_count` (execution) must both
    ///   score the request at ONE reference, so the scope's geometry gate lets it through. When
    ///   `image_reference_count` scored `ReferenceRgba` at zero, admission at `reference_count: 1`
    ///   was followed by a hard refusal here — "references=0 does not fit admitted ...
    ///   references=1" — and an admission taken FROM the request priced the edit with zero
    ///   reference tokens.
    /// * the joint-token pricing must add exactly one `REFERENCE_FIT_TOKENS` block, the same block
    ///   an RGB reference adds: a transparent reference is fitted to `OUTPUT_RESOLUTION` like any
    ///   other.
    ///
    /// *Mutation that reds this:* drop `Conditioning::ReferenceRgba` from the `=> 1` arm of
    /// `gen_core::GenerationRequest::image_reference_count` (i.e. restore the `_ => 0` wildcard).
    #[test]
    fn a_transparent_reference_passes_the_geometry_gate_and_is_priced_like_an_opaque_one() {
        use gen_core::{Conditioning, GenerationRequest, Image, RgbaImage};

        let rgb_reference = || Conditioning::Reference {
            image: Image {
                width: 64,
                height: 64,
                pixels: vec![0; 64 * 64 * 3],
            },
            strength: None,
        };
        let rgba_reference = || Conditioning::ReferenceRgba {
            image: RgbaImage {
                width: 64,
                height: 64,
                pixels: vec![0; 64 * 64 * 4],
            },
            strength: None,
        };

        let request_with = |conditioning: Vec<Conditioning>| GenerationRequest {
            prompt: "a red fox".into(),
            width: 2048,
            height: 2048,
            conditioning,
            ..Default::default()
        };

        // 1. Admission and execution agree, and agree with the RGB carrier.
        let transparent = request_with(vec![rgba_reference()]);
        let opaque = request_with(vec![rgb_reference()]);
        assert_eq!(
            gen_core::wan_i2v_memory::geometry_from_request(&transparent).reference_count,
            1,
            "admission must see one reference"
        );
        assert_eq!(
            transparent.memory_reference_count(),
            1,
            "execution must see the same one"
        );
        assert_eq!(
            gen_core::wan_i2v_memory::geometry_from_request(&transparent),
            gen_core::wan_i2v_memory::geometry_from_request(&opaque),
            "a transparent reference admits identically to an opaque one"
        );

        // 2. The scope's geometry gate accepts it, admitted from the request itself.
        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap();
        let mut context = fixture.context.clone();
        let admitted = gen_core::wan_i2v_memory::geometry_from_request(&transparent);
        context.geometry.width = admitted.width;
        context.geometry.height = admitted.height;
        context.geometry.batch = admitted.batch;
        // `geometry_from_request` reports `frames: 0` for a still-image request while the scope
        // compares against its own `default_frames`; that axis is not what this test is about, so
        // the fixture's value stands and the reference axis is the only one under test.
        context.geometry.reference_count = admitted.reference_count;
        context.has_reference = true;
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept,
            "a one-reference transparent edit must be admitted"
        );

        let mut scope = registered_begin_request(&spec, &contract, &context)
            .unwrap()
            .expect("this provider publishes a request scope");
        let mut executing = transparent.clone();
        scope.configure_request(&mut executing).expect(
            "the admitted transparent-reference request must pass the scope's geometry gate",
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();

        // 3. ...and it is priced with exactly one fitted reference block. This crate publishes the
        // per-reference cost on the admission geometry rather than through a `joint_tokens`
        // helper, so the claim is stated against that: the request's reference count times the
        // published per-reference token cost.
        let geometry = admission_geometry();
        assert_eq!(geometry.tokens_per_max_reference, REFERENCE_FIT_TOKENS);
        assert_eq!(
            transparent.memory_reference_count() as u64 * geometry.tokens_per_max_reference,
            REFERENCE_FIT_TOKENS,
            "a transparent reference costs one fitted block, like every other reference"
        );
        assert_eq!(
            opaque.memory_reference_count(),
            transparent.memory_reference_count(),
            "and exactly what the opaque carrier costs"
        );
    }

    #[test]
    fn bounded_decode_publishes_a_domain_and_refuses_everything_outside_it() {
        assert!(DECODE_TILE_EDGES.contains(&crate::pipeline::DECODE_TILE_EDGE));
        for edge in DECODE_TILE_EDGES {
            assert!(validate_decode(Some(*edge), Some(DECODE_OVERLAP)).is_ok());
        }
        assert!(validate_decode(Some(1024), Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(crate::pipeline::DECODE_TILE_EDGE), Some(96)).is_err());
        assert!(validate_decode(None, Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(crate::pipeline::DECODE_TILE_EDGE), None).is_err());
    }

    /// The derived tier ladder is owned by the MLX crate and is not duplicated here, but the
    /// *shape* it predicts must hold on this backend's pricing too: the text encoder is priced over
    /// the loaded prefix only, and a packed tier's integer codes price at their stored width.
    #[test]
    fn the_text_encoder_is_priced_over_the_loaded_prefix_only() {
        use candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().unwrap();
        let te = tmp.path().join("text_encoder");
        std::fs::create_dir_all(&te).unwrap();
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "model.language_model.layers.0.mlp.up_proj.weight".into(),
            Tensor::zeros((8, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            // Never loaded on this route.
            "model.visual.blocks.0.attn.qkv.weight".into(),
            Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "lm_head.weight".into(),
            Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&map, te.join("model.safetensors")).unwrap();

        let loaded = component_bytes(&te, 4, &|name: &str| {
            name.starts_with(crate::loader::TEXT_ENCODER_PREFIX)
        })
        .unwrap();
        let everything = component_bytes(&te, 4, &|_| true).unwrap();
        assert_eq!(loaded, 8 * 4 * 4, "only the language tower is materialized");
        assert!(
            loaded < everything,
            "the vision tower and lm_head must not be charged"
        );
    }

    /// A packed tier's u32 codes are integers: they price at their stored width whatever float
    /// width the build materializes at, which is what makes one pricing function serve all tiers.
    #[test]
    fn packed_codes_price_at_their_stored_width() {
        use candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "img_in.weight".into(),
            Tensor::zeros((8, 8), DType::U32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "img_in.scales".into(),
            Tensor::zeros((8, 1), DType::BF16, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&map, dir.join("model.safetensors")).unwrap();

        let at_bf16 = component_bytes(&dir, 2, &|_| true).unwrap();
        let at_f32 = component_bytes(&dir, 4, &|_| true).unwrap();
        assert_eq!(at_bf16, 8 * 8 * 4 + 8 * 2, "codes u32, scales bf16");
        assert_eq!(
            at_f32,
            8 * 8 * 4 + 8 * 4,
            "only the float scales follow the compute width"
        );
        // A tier is one width: the q4 tier's tower is q4, not a promoted q8.
        assert_eq!(crate::quant::Tier::Q4.text_encoder_bits(), Some(4));
        assert_eq!(
            crate::quant::Tier::Q4.text_encoder_bits(),
            crate::quant::Tier::Q4.transformer_bits()
        );
    }

    /// **Every float component is priced at ONE width — the tower included** (sc-24114). Before
    /// this the tower was read and priced at f32 on every backend, which put the candle bf16
    /// resident set ~14 GiB above the derivation the SceneWorks floors were taken from.
    ///
    /// Stated at an explicit width on the all-float dense tiny snapshot, where halving the width
    /// must exactly halve every component — the CPU lane's own compute width is 4, which is the
    /// very number the tower used to be hard-wired to, so `asset_facts()` alone could not tell the
    /// two apart there. The production inventory is then held to the compute width.
    ///
    /// *Mutation that reds this:* pricing `text_encoder/` at a literal `4` again inside
    /// `asset_facts_at` (the tower's bf16 bytes stop being half its f32 bytes).
    #[test]
    fn the_tower_is_priced_at_the_compute_width_like_every_other_component() {
        let tiny = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/tiny-snapshot");
        let at_bf16 = asset_facts_at(&tiny, 2).unwrap();
        let at_f32 = asset_facts_at(&tiny, 4).unwrap();
        assert!(at_bf16.conditioning_bytes > 0);
        assert_eq!(
            at_bf16.conditioning_bytes * 2,
            at_f32.conditioning_bytes,
            "the tower follows the float width like the DiT and the VAE"
        );
        assert_eq!(at_bf16.transformer_bytes * 2, at_f32.transformer_bytes);
        assert_eq!(at_bf16.decoder_bytes * 2, at_f32.decoder_bytes);
        assert_eq!(at_bf16.base_bytes * 2, at_f32.base_bytes);

        // The production inventory is the compute-width one, and the compute width is the
        // loaded tower's dtype.
        let spec = LoadSpec::new(WeightsSource::Dir(tiny.clone()));
        let contract = memory_strategy_contract(MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract.asset_facts,
            asset_facts_at(&tiny, compute_width()).unwrap()
        );
        assert_eq!(
            compute_width(),
            crate::loader::compute_dtype().size_in_bytes() as u64
        );
        // On the production device: a GPU build prices its GPU's bf16, while a CPU device is held
        // at f32 on every build (`loader::compute_dtype_on`; candle's CPU backend has no bf16
        // matmul), so loading onto `Device::Cpu` here would compare the wrong pair on CUDA/Metal.
        let device = candle_gen::default_device().unwrap();
        let tower = crate::loader::load_text_encoder(&tiny, &device).unwrap();
        assert_eq!(
            tower.dtype().size_in_bytes() as u64,
            compute_width(),
            "the priced width is the width the tower is actually materialized at"
        );
    }
}
