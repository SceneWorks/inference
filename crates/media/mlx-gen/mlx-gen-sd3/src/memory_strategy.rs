//! Shared image-memory ladder for SD3.5 Large, Large Turbo, and Medium on MLX/Metal.
//!
//! # Per-variant rung audit (SC-18606)
//!
//! Every rung below was audited against each variant's own engine code, not inferred family-wide
//! from Large:
//!
//! * **StagedResidency** — `model::build_residency` routes all three variants through the same
//!   [`mlx_gen::residency::Residency::from_policy`] seam, so `Sequential` bounds peak to
//!   `max(triple-TE, MMDiT+VAE)` identically for Large, Large-Turbo, and Medium.
//! * **BoundedDecode** — the 16-channel VAE is loaded by `loader::load_vae` with no variant
//!   parameter at all; the decode tile domain is therefore literally the same object for all three.
//! * **BoundedAttention** — `attention_plan` threads one [`AttentionPlan`] into
//!   `JointBlock::forward`, which applies it to the joint double-stream attention **and** to
//!   Medium's MMDiT-X image-stream-only `attn2` branch. SD3.5 has no RoPE, and its per-head
//!   qk-RMSNorm is applied after the BHSD reshape inside the attention itself, so chunking the
//!   score computation does not move any normalization out of its per-head domain.
//! * **BoundedTransformerResidency** — `block_stream::Sd3BlockStream` is arch-parametric: it
//!   materializes `arch.num_layers` blocks (38 Large/Turbo, 24 Medium) at `arch.num_heads`
//!   (38 / 38 / 24) and `arch.head_dim`, and asks `arch.is_dual_attention_block` per index so
//!   Medium's leading 13 dual-attention blocks rebuild with their `attn2` + 9-chunk `norm1`. Its
//!   per-index construction is the same `JointBlock::from_weights` call the resident
//!   `Sd3Transformer::from_weights` makes, so a streamed window is byte-identical to the resident
//!   block for every variant. The mechanism was never Large-specific; only the snapshot admission
//!   was, which `artifact_inventory::stream_inventory` now resolves per variant.
//!
//! What stays Large-only is one **string**, not implementation: the exact content-pinned SC-15522
//! artifact remains the sole carrier of [`MEMORY_CALIBRATION_FINGERPRINT`].
//!
//! # Production calibration identity (sc-22730, epic sc-22723 E1/E4)
//!
//! Every clean base load of a SHIPPED SceneWorks turnkey publishes a production calibration identity
//! keyed on the route — see [`production_calibration_fingerprint`] for the table and
//! `production_calibration_identity` for the binding. Until sc-22730 the identity existed only for
//! that one content-pinned Large BF16 snapshot, so all nine `<route>:<tier>:mlx` cells the worker can
//! actually load published none and no memory anchor could bind to them: the SceneWorks capture arm
//! reads `contract.calibration` off the LOADED generator and refuses a contract without one.
//!
//! The identity is independent of `OffloadPolicy` and `LoadShape` — the worker's still-image path
//! loads `Resident + EagerMaterialization` — and it is bound to the artifact, never to the request
//! knob: `artifact_inventory::shipped_turnkey_tier` proves the pinned repository, revision and tier
//! subdir and then cross-checks the transformer's packed triples against the `quantization` manifest
//! the converter wrote, and a dense snapshot opened with `quantize = Some(_)` (a genuine runtime
//! requantization, since this engine quantizes dense at load) publishes nothing. A load that is not
//! a shipped turnkey — an anonymous structurally admitted snapshot — still publishes no calibration
//! and is admitted only under explicit
//! [`mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated`] authority (SC-18093). Weights-free
//! registry conformance keeps its own [`STATIC_BEHAVIOR_FINGERPRINT`] family, which never equals a
//! production string.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy};

use crate::config::{Sd3Variant, LARGE_NUM_LAYERS, MEDIUM_NUM_LAYERS};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sd3-5-large-bf16-mlx-shared-ladder-2026-08-03-v1";
pub const CALIBRATED_REVISION: &str = crate::artifact_inventory::CALIBRATED_REVISION;
/// Static behavior identity of the weights-free registry surface.
///
/// Advanced to `v2` by SC-18606: the surface itself changed (Large-Turbo and Medium moved from
/// resident-only to the full published ladder), so a consumer holding a `v1` handshake must be
/// refused rather than silently graded against a different declaration. This is a source-owned
/// behavior identity and is deliberately distinct from the measured
/// [`MEMORY_CALIBRATION_FINGERPRINT`].
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "sd3-5-mlx-registry-behavior-v2";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

fn variant_for(provider_id: &str) -> mlx_gen::gen_core::Result<Sd3Variant> {
    match provider_id {
        crate::MODEL_ID => Ok(Sd3Variant::Large),
        crate::TURBO_MODEL_ID => Ok(Sd3Variant::LargeTurbo),
        crate::MEDIUM_MODEL_ID => Ok(Sd3Variant::Medium),
        _ => Err(CoreError::Unsupported(format!(
            "unknown SD3.5 memory provider {provider_id}"
        ))),
    }
}

/// Architecture axes for one SD3.5 variant (epic SC-22657, E2).
///
/// This crate mirrors the reference `transformer/config.json` as [`Sd3Arch`] — resolved per variant
/// by [`Sd3Variant::arch`] — and the reference `vae/config.json` as the `crate::vae` constants, so
/// those constants are the config every route here is built from. Large and Large-Turbo share the
/// 38-block MMDiT; Medium is the 24-block MMDiT-X.
///
/// `vae_temporal_scale` stays `None`: SD3.5's AutoencoderKL is an image autoencoder with no
/// temporal axis, and a structurally absent axis is declared absent, never zero.
fn architecture_facts(variant: Sd3Variant) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let arch = variant.arch();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(arch.num_heads),
        // Declared directly by the config rather than divided out: `hidden()` IS
        // `num_heads * head_dim` here, so the quotient would only restate the input.
        head_dim: mlx_gen::architecture_facts::axis(arch.head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(arch.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(arch.patch_size),
        latent_channels: mlx_gen::architecture_facts::axis(crate::vae::SD3_VAE_LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::vae::SD3_VAE_SCALE_FACTOR),
        vae_temporal_scale: None,
        // The MMDiT computes f32 activations over its bf16 weights (`transformer.rs`), so 4 is the
        // activation width even though the checkpoint is 16-bit.
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

fn provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    Ok(variant_for(provider_id)?.id())
}

fn clean(spec: &LoadSpec) -> bool {
    spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
}

pub(crate) fn structurally_streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && clean(spec)
        && matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
}

/// Build one variant's contract from the *structural* rung predicates.
///
/// `streamable` is the only artifact-dependent input; every other rung is decided by the load
/// selector alone, because the engine implements them identically for all three variants (see the
/// module audit). Calibration is passed in separately so evidence never gates implementation.
fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
    footprint: mlx_gen::PerComponentBytes,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let variant = variant_for(provider_id)?;
    let staged = spec.offload_policy == OffloadPolicy::Sequential;
    let clean = clean(spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(variant);
    contract.calibration = calibration;
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: staged,
        decode_tiling: clean,
        attention_chunking: clean,
        transformer_window_materialization: streamable,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if clean => {
                capability.parameters.decode_tile_edges = DECODE_TILE_EDGES.to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if clean => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

/// Production calibration identity table of the clean SD3.5 base routes, keyed on the route
/// (sc-22730, epic sc-22723 E1/E4).
///
/// This is the TABLE, not the binding: only `production_calibration_identity` — which proves the
/// load against the artifact on disk first — may turn one of these strings into a contract identity.
///
/// The measured [`MEMORY_CALIBRATION_FINGERPRINT`] stays exactly where its evidence is: the exact
/// content-pinned SC-15522 Large BF16 snapshot, and it is returned by the binding rather than by this
/// table. Every shipped SceneWorks turnkey publishes `sd3-5-<route>-mlx-shared-ladder-v1` instead.
/// Offload policy, load shape and tier are deliberately NOT inputs: one turnkey family per route
/// carries all three tiers of the same ladder, `MemoryCalibrationIdentity::load_shape` carries the
/// materialization axis separately, and the tier is what the binding proves before publishing at all.
pub fn production_calibration_fingerprint(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        crate::MODEL_ID => Some("sd3-5-large-mlx-shared-ladder-v1"),
        crate::TURBO_MODEL_ID => Some("sd3-5-large-turbo-mlx-shared-ladder-v1"),
        crate::MEDIUM_MODEL_ID => Some("sd3-5-medium-mlx-shared-ladder-v1"),
        _ => None,
    }
}

/// The identity a loaded SD3.5 route publishes, bound to the artifact it opens.
///
/// Before sc-22730 this was `calibrated.then(…)`: only the exact SC-15522 Large BF16 snapshot ever
/// carried an identity, so the SceneWorks turnkeys the worker actually loads — every tier of Large,
/// and every tier of Large-Turbo and Medium — published none, and a memory anchor had nothing to
/// bind (the capture arm reads `contract.calibration` off the loaded generator and refuses a
/// contract without one).
///
/// Two gates, both fail-closed:
///
/// * the load must be a clean base load ([`clean`]) at bf16 execution precision — an overlay or an
///   external text encoder is a different resident set that no anchor measured; and
/// * `spec.quantize` must be the tier the artifact on disk already is
///   (`artifact_inventory::shipped_turnkey_tier`). This engine quantizes DENSE AT LOAD
///   (`crate::model::load_heavy`), so a dense snapshot opened with `quantize = Some(_)` is a genuine
///   runtime requantization whose peak is nobody's anchor, while a packed turnkey opened at its own
///   tier is the checked no-op the worker performs (`resolve_quant` → `reconcile_resolved_tier_quant`
///   settle the load quant onto the resolved tier subdir).
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
    calibrated: bool,
) -> mlx_gen::gen_core::Result<Option<MemoryCalibrationIdentity>> {
    if !clean(spec) || spec.precision != mlx_gen::Precision::Bf16 {
        return Ok(None);
    }
    if calibrated {
        return Ok(Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )));
    }
    let Some(artifact_tier) = crate::artifact_inventory::shipped_turnkey_tier(provider_id, spec)?
    else {
        return Ok(None);
    };
    if artifact_tier != spec.quantize {
        return Ok(None);
    }
    Ok(production_calibration_fingerprint(provider_id)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)))
}

/// Production contract for a loaded (or loadable) SD3.5 route.
///
/// Rungs 1-3 need no artifact admission at all — they are residency policy, VAE tiling, and an
/// attention budget — so all three variants publish them from the load selector alone. Rung 4 needs
/// a stable transformer subtree and is resolved through
/// [`crate::artifact_inventory::stream_inventory`], the same seam
/// [`crate::model::load_heavy`](crate::model) uses to attach the block stream, so the declared rung
/// and the engaged rung cannot disagree.
///
/// Calibration is decided separately by `production_calibration_identity`, which is bound to the
/// artifact rather than to the rung: rung 4 stays gated on `streamable`, the identity does not.
pub(crate) fn contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    variant_for(provider_id)?;
    let stream = crate::artifact_inventory::stream_inventory(provider_id, spec)
        .map_err(mlx_gen::gen_core::Error::from)?;
    // Read the admitting inventory's OWN predicate rather than "some lookup returned something":
    // binding the measured fingerprint to anything but a content-pinned artifact is the failure
    // this guards. A Resident or Eager load cannot stream, but the exact Large artifact still
    // carries the measured identity for the rungs it does reach, so fall back to the direct query.
    let calibrated = match &stream {
        Some(inventory) => inventory.is_calibrated(),
        None => crate::artifact_inventory::calibrated_inventory(provider_id, spec)
            .map_err(mlx_gen::gen_core::Error::from)?
            .is_some_and(|inventory| inventory.is_calibrated()),
    };
    let streamable = stream.is_some();
    let calibration = production_calibration_identity(provider_id, spec, calibrated)?;
    build_contract(
        provider_id,
        spec,
        streamable,
        calibration,
        crate::model::component_footprint(spec)?,
    )
}

/// Weights-free registry-surface contract: the declaration downstream capability facts read.
///
/// It has no filesystem to consult, so rung 4 is published from the structural selector predicate
/// alone — identical for all three variants, since the block-stream mechanism is arch-parametric.
/// The calibration identity here is the source-owned [`STATIC_BEHAVIOR_FINGERPRINT`], made distinct
/// per provider so one variant's handshake can never be replayed against another's.
pub(crate) fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    variant_for(provider_id)?;
    build_contract(
        provider_id,
        spec,
        structurally_streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            format!(
                "{STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
            spec.load_shape,
        )),
        Default::default(),
    )
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let route_ok = matches!(context.mode, MemoryMode::TextToImage)
            && context.geometry.reference_count == 0
            || matches!(context.mode, MemoryMode::ImageToImage)
                && context.geometry.reference_count == 1;
        if !route_ok {
            return Err(CoreError::Unsupported(
                "sd3 memory route must be text-to-image or single-reference img2img".to_owned(),
            ));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(
                "sd3 does not implement PiD decode".to_owned(),
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let edge = context.selection.parameters.decode_tile_edge;
            let overlap = context.selection.parameters.decode_overlap;
            if !edge.is_some_and(|edge| DECODE_TILE_EDGES.contains(&edge))
                || overlap != Some(DECODE_OVERLAP)
            {
                return Err(CoreError::Unsupported(
                    "sd3 decode tile geometry is outside the calibrated domain".to_owned(),
                ));
            }
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) && (!context.has_phases || !contract.lifecycle.transformer_window_materialization)
        {
            return Err(CoreError::Unsupported(
                "sd3 transformer streaming requires a staged Sequential + DeferredMaterialization \
                 route over an admitted, unchanging transformer snapshot subtree"
                    .to_owned(),
            ));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    if contract
        .capability(strategy)
        .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    [(MemoryMode::TextToImage, 0), (MemoryMode::ImageToImage, 1)]
        .into_iter()
        .map(|(mode, reference_count)| {
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                mlx_gen::gen_core::MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: spec.offload_policy == OffloadPolicy::Sequential,
                    overlay: None,
                },
            )
            .map(mlx_gen::gen_core::MemoryBehaviorFixture::new)
        })
        .collect()
}

fn begin_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let provider = provider_static(&contract.provider_id)?;
    let blocks = match variant_for(provider)? {
        Sd3Variant::Large | Sd3Variant::LargeTurbo => LARGE_NUM_LAYERS,
        Sd3Variant::Medium => MEDIUM_NUM_LAYERS,
    };
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        blocks,
        |_use_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(CoreError::Unsupported(
                    "sd3 decode tile geometry is outside the calibrated domain".to_owned(),
                ))
            }
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

pub(crate) fn transformer_window(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if memory
        .transformer_window_component
        .unwrap_or(TransformerComponent::Dit)
        != TransformerComponent::Dit
    {
        return Err(mlx_gen::Error::Unsupported(
            "sd3 supports only the DiT transformer window component".to_owned(),
        ));
    }
    Ok(Some(
        memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
    ))
}

pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    Ok(Some(TilingConfig::spatial_only(
        memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
        memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{MemoryOptimizationAuthority, MemoryStrategySupport};

    /// Every registered SD3.5 route, so no test can silently cover only the calibrated variant.
    pub(crate) const PROVIDERS: [&str; 3] = [
        crate::MODEL_ID,
        crate::TURBO_MODEL_ID,
        crate::MEDIUM_MODEL_ID,
    ];

    /// One named, independently applied mutation of an otherwise-accepted run context.
    type LadderPointMutation = (&'static str, fn(&mut MemoryRunContext));

    fn streamable_spec(root: &std::path::Path) -> LoadSpec {
        LoadSpec::new(mlx_gen::WeightsSource::Dir(root.to_path_buf()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    /// A snapshot whose `transformer/` subtree exists and is stable. Deliberately NOT a real
    /// checkpoint: the structural admission pins identity, never content, so this is the exact
    /// shape the Turbo/Medium production route admits.
    fn transformer_snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("transformer")).unwrap();
        std::fs::write(
            dir.path()
                .join("transformer/diffusion_pytorch_model.safetensors"),
            b"sd3-structural-admission-fixture",
        )
        .unwrap();
        std::fs::write(dir.path().join("transformer/config.json"), b"{}").unwrap();
        dir
    }

    /// AC (SC-22662): each registered SD3.5 route publishes the axes of the MMDiT it actually runs —
    /// Large and Large-Turbo the 38-block plain MMDiT, Medium the 24-block MMDiT-X — and each
    /// contract passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_each_variants_own_arch_constants() {
        let spec = streamable_spec(std::path::Path::new("/nonexistent"));
        let large = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(38),
            head_dim: Some(64),
            transformer_blocks: Some(38),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            // f32 activations over bf16 weights.
            activation_dtype_width: Some(4),
        };
        let medium = mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(24),
            transformer_blocks: Some(24),
            ..large
        };
        for (provider, expected) in [
            (crate::MODEL_ID, large),
            (crate::TURBO_MODEL_ID, large),
            (crate::MEDIUM_MODEL_ID, medium),
        ] {
            let contract = weights_free_contract(provider, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts, expected,
                "{provider} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    #[test]
    fn every_variant_publishes_the_same_structural_ladder_on_the_registry_surface() {
        let spec = streamable_spec(std::path::Path::new("/nonexistent"));
        let mut fingerprints = std::collections::BTreeSet::new();
        for provider in PROVIDERS {
            let contract = weights_free_contract(provider, &spec).unwrap();
            for strategy in MemoryStrategy::ALL {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{provider} must publish the full ladder on the streamable selector"
                );
            }
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .parameters
                    .transformer_window_components,
                vec![TransformerComponent::Dit],
                "{provider}"
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .parameters
                    .transformer_window_sizes,
                vec![TRANSFORMER_WINDOW_SIZE],
                "{provider}"
            );
            assert!(contract.conformance_errors().is_empty(), "{provider}");
            let identity = contract.calibration.as_ref().unwrap();
            assert!(
                identity
                    .fingerprint
                    .starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                "{provider} must publish the source-owned behavior identity, got {}",
                identity.fingerprint
            );
            assert_ne!(
                identity.fingerprint, MEMORY_CALIBRATION_FINGERPRINT,
                "{provider} must not present the measured fingerprint weights-free"
            );
            assert!(
                fingerprints.insert(identity.fingerprint.clone()),
                "{provider} repeats another variant's behavior identity"
            );
        }
        assert_eq!(fingerprints.len(), PROVIDERS.len());
    }

    #[test]
    fn rung_four_is_declared_only_on_the_sequential_deferred_dense_selector() {
        for provider in PROVIDERS {
            for offload_policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                for load_shape in [
                    LoadShape::EagerMaterialization,
                    LoadShape::DeferredMaterialization,
                ] {
                    for quant in [None, Some(mlx_gen::Quant::Q4), Some(mlx_gen::Quant::Q8)] {
                        let mut spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/none".into()))
                            .with_offload_policy(offload_policy)
                            .with_load_shape(load_shape);
                        spec.quantize = quant;
                        let contract = weights_free_contract(provider, &spec).unwrap();
                        // Block windows materialize dense on-disk tensors; SD3.5 has no packed
                        // per-block path, so a Q4/Q8 load can never stream.
                        let expected = offload_policy == OffloadPolicy::Sequential
                            && load_shape == LoadShape::DeferredMaterialization
                            && quant.is_none();
                        assert_eq!(
                            contract
                                .capability(MemoryStrategy::BoundedTransformerResidency)
                                .unwrap()
                                .support,
                            if expected {
                                MemoryStrategySupport::Implemented
                            } else {
                                MemoryStrategySupport::Missing
                            },
                            "{provider} {offload_policy:?} {load_shape:?} {quant:?}"
                        );
                        assert_eq!(
                            contract
                                .capability(MemoryStrategy::StagedResidency)
                                .unwrap()
                                .support,
                            if offload_policy == OffloadPolicy::Sequential {
                                MemoryStrategySupport::Implemented
                            } else {
                                MemoryStrategySupport::Missing
                            },
                            "{provider} {offload_policy:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn production_rung_four_is_reachable_for_every_variant_under_estimate_authority() {
        let snapshot = transformer_snapshot();
        let spec = streamable_spec(snapshot.path());
        for provider in PROVIDERS {
            let production = contract_for(provider, &spec).unwrap();
            assert_eq!(
                production
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "{provider} production route must admit rung 4 over a stable transformer subtree"
            );
            assert!(
                production.calibration.is_none(),
                "{provider}: a structurally admitted snapshot carries no measured evidence"
            );

            // The behavior fixture is built from the registry surface (which owns the static
            // handshake) and then re-graded against the PRODUCTION contract under the estimate
            // authority SC-18093 requires of an uncalibrated optimized rung.
            let surface = weights_free_contract(provider, &spec).unwrap();
            let mut fixture =
                registered_fixture(&spec, &surface, MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .remove(0);
            assert_eq!(fixture.context.mode, MemoryMode::TextToImage, "{provider}");
            fixture.context.optimization_authority = MemoryOptimizationAuthority::Estimated;
            assert_eq!(
                safety_check(&spec, &production, &fixture.context),
                MemorySafetyDecision::Accept,
                "{provider}"
            );

            let mut scope = registered_begin_request(&spec, &production, &fixture.context)
                .unwrap()
                .unwrap_or_else(|| panic!("{provider} must open a rung-4 request scope"));
            scope.configure_request(&mut fixture.request).unwrap();

            // Reachability, not declaration: the request the scope wrote must drive the engine's
            // own window/tiling/attention consumers.
            let memory = fixture.request.memory.expect("configured memory controls");
            assert!(memory.stream_transformer_blocks, "{provider}");
            assert!(memory.stage_residency, "{provider}");
            assert_eq!(
                transformer_window(&fixture.request).unwrap(),
                Some(TRANSFORMER_WINDOW_SIZE as usize),
                "{provider}"
            );
            assert!(
                decode_tiling(&fixture.request).unwrap().is_some(),
                "{provider}"
            );
            assert_ne!(
                attention_plan(&fixture.request).budget,
                AttentionBudget::UNBOUNDED,
                "{provider} must bound attention when the scope engages rung 3"
            );
            scope
                .finish(mlx_gen::gen_core::MemoryRunOutcome::Complete)
                .unwrap();
        }
    }

    /// A header-valid safetensors file with no weight body: `safetensors_header_shapes` reads the
    /// header alone, so the tensor names are all a tier proof needs.
    fn write_header_only_safetensors(path: &std::path::Path, names: &[&str]) {
        let mut header = serde_json::Map::new();
        for name in names {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({ "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 0] }),
            );
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        std::fs::write(path, bytes).unwrap();
    }

    /// One shipped SceneWorks turnkey tier directory, at the exact HF-cache path
    /// `artifact_inventory::SHIPPED_TURNKEYS` pins for `provider_id`.
    fn turnkey_fixture(provider_id: &str, tier: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let (_, repository, revision) = crate::artifact_inventory::SHIPPED_TURNKEYS
            .iter()
            .find(|(id, _, _)| *id == provider_id)
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .join(format!("models--SceneWorks--{repository}"))
            .join("snapshots")
            .join(revision)
            .join(tier);
        for component in [
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        ] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        let packed = matches!(tier, "q4" | "q8");
        let names: &[&str] = if packed {
            &[
                "blocks.0.proj.weight",
                "blocks.0.proj.scales",
                "blocks.0.proj.biases",
            ]
        } else {
            &["blocks.0.proj.weight"]
        };
        write_header_only_safetensors(
            &root.join("transformer/diffusion_pytorch_model.safetensors"),
            names,
        );
        let config = if packed {
            serde_json::json!({ "quantization": { "bits": if tier == "q4" { 4 } else { 8 }, "group_size": 64 } })
        } else {
            serde_json::json!({})
        };
        std::fs::write(
            root.join("transformer/config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        (temp, root)
    }

    fn turnkey_spec(root: &std::path::Path, quant: Option<mlx_gen::Quant>) -> LoadSpec {
        // The worker's MLX still-image shape: Resident + EagerMaterialization are the `LoadSpec`
        // defaults and it never overrides either (`image_jobs/base.rs`, no `memory_route_registry`
        // rule and no `mlx.memoryStrategyContract` for sd3), with the load quant settled onto the
        // resolved tier subdir.
        let mut spec = LoadSpec::new(mlx_gen::WeightsSource::Dir(root.to_path_buf()));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        spec
    }

    /// sc-22730 (epic sc-22723, E1 measurable / E4 production loader): every shipped SD3.5 turnkey
    /// tier the worker can load publishes its route's production calibration identity, under the
    /// worker's own load shape and under the staged/deferred one, so a memory anchor has something
    /// to bind. Nine cells (three routes x three tiers) had none before.
    ///
    /// *Mutations this kills:* restoring `calibrated.then(…)` (every turnkey goes back to `None`);
    /// keying the string on the tier, the offload policy or the load shape (the six specs below
    /// would disagree); replaying one route's string on another; publishing the weights-free
    /// registry string in production; and dropping the artifact proof — the last two blocks load a
    /// dense turnkey with a quant knob and a packed turnkey at the wrong tier, both of which are
    /// runtime requantizations no anchor measured.
    #[test]
    fn every_shipped_turnkey_tier_publishes_its_routes_production_identity() {
        let mut published = std::collections::BTreeSet::new();
        for provider in PROVIDERS {
            let expected = production_calibration_fingerprint(provider).unwrap();
            let weights_free = format!(
                "{STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider.replace('_', "-")
            );
            assert_ne!(expected, weights_free);
            assert_ne!(expected, MEMORY_CALIBRATION_FINGERPRINT);
            for (tier, quant) in [
                ("bf16", None),
                ("q4", Some(mlx_gen::Quant::Q4)),
                ("q8", Some(mlx_gen::Quant::Q8)),
            ] {
                let (_temp, root) = turnkey_fixture(provider, tier);
                assert_eq!(
                    crate::artifact_inventory::shipped_turnkey_tier(
                        provider,
                        &turnkey_spec(&root, quant)
                    )
                    .unwrap(),
                    Some(quant),
                );
                for (offload, load_shape) in [
                    (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
                    (
                        OffloadPolicy::Sequential,
                        LoadShape::DeferredMaterialization,
                    ),
                ] {
                    let spec = turnkey_spec(&root, quant)
                        .with_offload_policy(offload)
                        .with_load_shape(load_shape);
                    let contract = contract_for(provider, &spec).unwrap();
                    let identity = contract.calibration.as_ref().unwrap_or_else(|| {
                        panic!("{provider} {tier} {load_shape:?} publishes none")
                    });
                    assert_eq!(identity.fingerprint, expected, "{provider} {tier}");
                    assert_eq!(identity.load_shape, load_shape);
                    assert!(contract.conformance_errors().is_empty());
                }

                // The request knob never outranks the artifact: every tier the snapshot is NOT.
                for wrong in [None, Some(mlx_gen::Quant::Q4), Some(mlx_gen::Quant::Q8)] {
                    if wrong == quant {
                        continue;
                    }
                    assert!(
                        contract_for(provider, &turnkey_spec(&root, wrong))
                            .unwrap()
                            .calibration
                            .is_none(),
                        "{provider} {tier} must publish nothing for a {wrong:?} load quant"
                    );
                }

                // Another route's artifact is not this route's cell.
                for other in PROVIDERS.into_iter().filter(|other| *other != provider) {
                    assert!(
                        crate::artifact_inventory::shipped_turnkey_tier(
                            other,
                            &turnkey_spec(&root, quant)
                        )
                        .unwrap()
                        .is_none(),
                        "{other} must not claim {provider}'s turnkey"
                    );
                }
            }
            published.insert(expected);
        }
        assert_eq!(published.len(), PROVIDERS.len());
    }

    #[test]
    fn only_the_content_pinned_large_artifact_binds_the_measured_fingerprint() {
        let snapshot = transformer_snapshot();
        let spec = streamable_spec(snapshot.path());
        for provider in PROVIDERS {
            let contract = contract_for(provider, &spec).unwrap();
            assert!(
                contract.calibration.is_none(),
                "{provider} must not present measured calibration for an unpinned snapshot"
            );
        }
        // The exact admission is keyed on the SC-15522 revision component, so no temp directory can
        // ever satisfy it — including one that names the Large provider.
        assert!(
            crate::artifact_inventory::calibrated_inventory(crate::MODEL_ID, &spec)
                .unwrap()
                .is_none()
        );
        for provider in [crate::TURBO_MODEL_ID, crate::MEDIUM_MODEL_ID] {
            assert!(
                crate::artifact_inventory::calibrated_inventory(provider, &spec)
                    .unwrap()
                    .is_none(),
                "{provider} must never reach the Large content pin"
            );
        }
    }

    #[test]
    fn structural_admission_fails_closed_on_every_unstreamable_axis_and_on_mutation() {
        let snapshot = transformer_snapshot();
        let good = streamable_spec(snapshot.path());
        let inventory = crate::artifact_inventory::stream_inventory(crate::MEDIUM_MODEL_ID, &good)
            .unwrap()
            .expect("a stable transformer subtree is admissible");
        assert!(!inventory.is_calibrated());
        inventory.ensure_unchanged().unwrap();
        std::fs::write(
            snapshot
                .path()
                .join("transformer/diffusion_pytorch_model.safetensors"),
            b"a different checkpoint entirely",
        )
        .unwrap();
        assert!(
            inventory.ensure_unchanged().is_err(),
            "a transformer file replaced under an admitted window must fail closed"
        );

        let empty = tempfile::tempdir().unwrap();
        let no_transformer = streamable_spec(empty.path());
        // A single-file source is refused earlier still: SD3.5 needs the diffusers multi-component
        // tree, so the per-component footprint rejects it before any rung is considered.
        let single_file = LoadSpec::new(mlx_gen::WeightsSource::File(
            snapshot.path().join("transformer/config.json"),
        ))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
        for provider in PROVIDERS {
            assert!(
                crate::artifact_inventory::stream_inventory(provider, &single_file)
                    .unwrap()
                    .is_none(),
                "{provider}: a single-file source must not admit the block window"
            );
            assert!(
                contract_for(provider, &single_file).is_err(),
                "{provider}: a single-file source is not a loadable SD3.5 route at all"
            );
        }

        let mut cases = vec![
            ("no transformer subtree", no_transformer),
            (
                "resident policy",
                streamable_spec(snapshot.path()).with_offload_policy(OffloadPolicy::Resident),
            ),
            (
                "eager shape",
                streamable_spec(snapshot.path()).with_load_shape(LoadShape::EagerMaterialization),
            ),
            (
                "quantized load",
                streamable_spec(snapshot.path()).with_quant(mlx_gen::Quant::Q4),
            ),
        ];
        let mut control = streamable_spec(snapshot.path());
        control.control = Some(mlx_gen::WeightsSource::File("/control.safetensors".into()));
        cases.push(("control overlay", control));
        let mut adapters = streamable_spec(snapshot.path());
        adapters.adapters.push(mlx_gen::AdapterSpec {
            path: "/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        });
        cases.push(("adapter overlay", adapters));

        for (label, spec) in cases {
            for provider in PROVIDERS {
                assert!(
                    crate::artifact_inventory::stream_inventory(provider, &spec)
                        .unwrap()
                        .is_none(),
                    "{provider}: {label} must not admit the block window"
                );
                let contract = contract_for(provider, &spec).unwrap();
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Missing,
                    "{provider}: {label}"
                );
            }
        }
    }

    #[test]
    fn the_block_window_materializes_each_variants_own_topology() {
        use crate::config::Sd3Variant;

        let snapshot = transformer_snapshot();
        let spec = streamable_spec(snapshot.path());
        for (variant, blocks, heads, dual_blocks) in [
            (Sd3Variant::Large, LARGE_NUM_LAYERS, 38, 0),
            (Sd3Variant::LargeTurbo, LARGE_NUM_LAYERS, 38, 0),
            (Sd3Variant::Medium, MEDIUM_NUM_LAYERS, 24, 13),
        ] {
            let inventory = crate::artifact_inventory::stream_inventory(variant.id(), &spec)
                .unwrap()
                .unwrap();
            let arch = variant.arch();
            let stream = crate::block_stream::Sd3BlockStream::new(inventory, arch);
            assert_eq!(stream.blocks(), blocks, "{}", variant.id());
            assert_eq!(arch.num_heads, heads, "{}", variant.id());
            assert_eq!(
                (0..blocks)
                    .filter(|index| arch.is_dual_attention_block(*index))
                    .count(),
                dual_blocks,
                "{} must rebuild its own MMDiT-X dual-attention blocks",
                variant.id()
            );
            // SD3.5 has no RoPE: the window carries only per-head qk-RMSNorm, which is applied
            // inside the attention after the BHSD reshape and so needs the variant's own head
            // count to reconstruct a block.
            assert_eq!(arch.hidden(), heads * arch.head_dim, "{}", variant.id());
        }
    }

    #[test]
    fn unsupported_ladder_points_are_refused_for_every_variant() {
        let snapshot = transformer_snapshot();
        let spec = streamable_spec(snapshot.path());
        for provider in PROVIDERS {
            let production = contract_for(provider, &spec).unwrap();
            let surface = weights_free_contract(provider, &spec).unwrap();
            let base =
                registered_fixture(&spec, &surface, MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .remove(0)
                    .context;

            // Without explicit estimate authority an uncalibrated optimized rung is inadmissible.
            assert!(
                matches!(
                    safety_check(&spec, &production, &base),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{provider}: calibrated authority must not survive against an uncalibrated contract"
            );

            let accepted = {
                let mut context = base.clone();
                context.optimization_authority = MemoryOptimizationAuthority::Estimated;
                context
            };
            assert_eq!(
                safety_check(&spec, &production, &accepted),
                MemorySafetyDecision::Accept,
                "{provider}"
            );

            // Mutate each ladder point INDIVIDUALLY: an all-at-once mutation would only prove the
            // set is guarded, not each member.
            let mutations: [LadderPointMutation; 6] = [
                ("decode tile edge", |context| {
                    context.selection.parameters.decode_tile_edge = Some(1024);
                }),
                ("decode overlap", |context| {
                    context.selection.parameters.decode_overlap = Some(64);
                }),
                ("attention chunk size", |context| {
                    context.selection.parameters.attention_chunk_size = Some(7);
                }),
                ("transformer window size", |context| {
                    context.selection.parameters.transformer_window_size = Some(2);
                }),
                ("transformer window component", |context| {
                    context.selection.parameters.transformer_window_component =
                        Some(TransformerComponent::TextEncoder);
                }),
                ("PiD decode", |context| {
                    context.use_pid = true;
                }),
            ];
            for (label, mutate) in mutations {
                let mut context = accepted.clone();
                mutate(&mut context);
                assert!(
                    matches!(
                        safety_check(&spec, &production, &context),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{provider}: {label} outside the declared domain must be refused"
                );
                assert!(
                    registered_begin_request(&spec, &production, &context).is_err(),
                    "{provider}: {label} must not open a request scope"
                );
            }
        }
    }

    #[test]
    fn unknown_providers_are_refused_by_both_contract_factories() {
        let spec = streamable_spec(std::path::Path::new("/nonexistent"));
        assert!(weights_free_contract("sd3_5_imaginary", &spec).is_err());
        assert!(contract_for("sd3_5_imaginary", &spec).is_err());
    }

    #[test]
    fn overlays_fail_closed_for_quality_sensitive_rungs() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_control(mlx_gen::WeightsSource::File("/nonexistent/control".into()));
        let contract = weights_free_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }
}
