//! MLX LTX-2.3 video memory-provider contract (SC-19109).
//!
//! The declaration follows the SC-18813 source survey rather than projecting the image ladder onto
//! video:
//!
//! - rung 1 is the shipped, unconditional Gemma -> AvDiT -> decode phase staging;
//! - rung 2 is the shipped budgeted LTX VAE tiler;
//! - rung 3 is missing because inference attention remains one monolithic SDPA call;
//! - rung 4 is missing because the 48-block AvDiT has no block-window materialization path.
//!
//! `MemoryStrategy::Resident` is gen-core's mandatory protocol baseline: it means "apply no new
//! request controls", not "keep every model component co-resident". LTX's historical baseline
//! already performs rung-1 phase staging, so `ResidentRequestMemory::PreserveLoadDefaults` leaves
//! that physical path unchanged while an explicit `StagedResidency` selection emits
//! `stage_residency=true`. The current contract vocabulary cannot separately say "unconditional"
//! versus "selectable"; SC-18816 owns that descriptor-level tri-state. Until then, resident
//! calibration must measure the already-staged baseline rather than synthesize a co-resident peak.
//!
//! The production factory sizes the exact component files this loader can materialize. The separate
//! weights-free factory exists only for registry conformance and deliberately injects zero asset
//! facts; production registry resolution never consults it.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    self, AdapterKind, AdapterResidencyMode, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategyPrerequisite,
    MemoryStrategySupport, Quant, ResidentRequestMemory, WeightsSource,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, Result};

use crate::config::{LtxConfig, SplitModel};
use crate::gemma::GemmaQuant;

/// Contract/execution identity introduced by SC-19109. The earlier SC-18808 capture predates this
/// shared-contract carrier and therefore cannot be reused as if its calibration semantics matched.
pub const CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-mlx-memory-ladder-v2";
const STATIC_CALIBRATION_FINGERPRINT: &str = "sc-20772-ltx-2-3-mlx-registry-behavior-v2";

pub const DECODE_OVERLAP: u32 = 64;

const SHIPPED_GEOMETRIES: &[(u32, u32)] =
    &[(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
const SHIPPED_FRAME_COUNTS: &[u32] = &[
    97, 121, 145, 153, 177, 193, 201, 241, 249, 289, 297, 361, 377, 449,
];
const I2V_STRENGTH_BITS: u32 = 1.0_f32.to_bits();

fn frame_count_matches_fps(fps: u32, frames: u32) -> bool {
    match fps {
        24 => [97, 145, 193, 241, 289, 361].contains(&frames),
        25 => [97, 153, 201, 249, 297, 377].contains(&frames),
        30 => [121, 177, 241, 297, 361, 449].contains(&frames),
        _ => false,
    }
}

fn decode_tile_edges() -> Vec<u32> {
    crate::pipeline::LTX_VAE_SPATIAL_PX
        .iter()
        .map(|&edge| edge as u32)
        .collect()
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            mlx_gen::Error::Msg(format!(
                "ltx_2_3: {label} safetensors byte total overflows u64"
            ))
        })
    })
}

fn required_projected_safetensors_bytes(
    path: &std::path::Path,
    label: &str,
    projection: ResidentProjection,
) -> Result<u64> {
    let bytes = projected_safetensors_bytes(path, |_| projection).map_err(mlx_gen::Error::from)?;
    if bytes == 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3: {label} has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn uncensored_enhancer_resident_bytes(spec: &LoadSpec) -> Result<u64> {
    let Some(source) = spec.components.get("uncensored_enhancer") else {
        return Ok(0);
    };
    let WeightsSource::Dir(path) = source else {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored_enhancer must be a directory source".into(),
        ));
    };
    // The amoral Gemma loader retains packed affine tensors, but `load_embedding` dequantizes the
    // packed token table to dense bf16. MLX keeps the packed source graph alive until that lazy
    // dequantization is evaluated, so the conditioning load peak is the complete stored tensor
    // inventory plus the expanded dense table. Its autoregressive KV work is bounded separately by
    // the request-scope MAX_ENHANCE_TOKENS ceiling.
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(mlx_gen::Error::from)?;
    let quant = crate::model::resolve_gemma_quant(path)?.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma must declare affine quantization geometry".into(),
        )
    })?;
    if quant.group <= 0 || !matches!(quant.bits, 4 | 8) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma quantization must be q4/q8 with a positive group (got {}-bit group-{})",
            quant.bits, quant.group
        )));
    }
    let embedding_weights = headers
        .iter()
        .filter(|header| header.name.ends_with("embed_tokens.weight"))
        .collect::<Vec<_>>();
    let [weight] = embedding_weights.as_slice() else {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma must contain exactly one packed embed_tokens.weight (got {})",
            embedding_weights.len()
        )));
    };
    let base = weight
        .name
        .strip_suffix(".weight")
        .expect("the filtered embedding name has a weight suffix");
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    let scales = headers
        .iter()
        .find(|header| header.name == scales_name)
        .ok_or_else(|| {
            mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: uncensored Gemma packed embedding is missing {scales_name}"
            ))
        })?;
    let biases = headers
        .iter()
        .find(|header| header.name == biases_name)
        .ok_or_else(|| {
            mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: uncensored Gemma packed embedding is missing {biases_name}"
            ))
        })?;
    if weight.dtype != gen_core::weightsmeta::Dtype::U32
        || !matches!(
            scales.dtype,
            gen_core::weightsmeta::Dtype::F16
                | gen_core::weightsmeta::Dtype::BF16
                | gen_core::weightsmeta::Dtype::F32
        )
        || biases.dtype != scales.dtype
        || biases.shape != scales.shape
    {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma packed embedding has invalid weight/scales/biases dtypes or shapes"
                .to_string(),
        ));
    }
    let [out, groups] = scales.shape.as_slice() else {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: uncensored Gemma packed embedding scales must be rank two".to_string(),
        ));
    };
    let group = usize::try_from(quant.group).map_err(|_| {
        mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma group is unrepresentable".into())
    })?;
    let input = groups.checked_mul(group).ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma embedding input width overflows".into())
    })?;
    let packed_bits = input
        .checked_mul(usize::try_from(quant.bits).map_err(|_| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma bit width is unrepresentable".into())
        })?)
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma packed width overflows".into())
        })?;
    if !packed_bits.is_multiple_of(32) || weight.shape.as_slice() != [*out, packed_bits / 32] {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: uncensored Gemma packed embedding geometry disagrees with {}-bit group-{} config",
            quant.bits, quant.group
        )));
    }
    let dense_embedding_bytes = u64::try_from(*out)
        .ok()
        .and_then(|out| {
            u64::try_from(input)
                .ok()
                .and_then(|input| out.checked_mul(input))
        })
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma dense embedding bytes overflow".into())
        })?;
    let stored_bytes = headers.iter().try_fold(0_u64, |total, header| {
        total.checked_add(header.data_bytes).ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma stored bytes overflow".into())
        })
    })?;
    let bytes = stored_bytes
        .checked_add(dense_embedding_bytes)
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_3: uncensored Gemma load peak overflows u64".into())
        })?;
    if bytes == 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3: uncensored Gemma enhancer has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

struct AssetDeclaration {
    facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
}

fn adapters_have_load_exact_additive_accounting(spec: &LoadSpec) -> Result<bool> {
    for adapter in &spec.adapters {
        if adapter.kind == AdapterKind::Lokr {
            return Ok(false);
        }
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)
            .map_err(mlx_gen::Error::from)?;
        let reconstructs_dense_delta = headers.iter().any(|tensor| {
            gen_core::weightsmeta::LOKR_TP_SUFFIXES
                .iter()
                .chain(gen_core::weightsmeta::LOHA_TP_SUFFIXES.iter())
                .any(|suffix| tensor.name.ends_with(suffix))
        });
        if reconstructs_dense_delta {
            return Ok(false);
        }
    }
    Ok(true)
}

fn production_asset_declaration(
    spec: &LoadSpec,
    gemma_dir: &std::path::Path,
) -> Result<AssetDeclaration> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires the split model directory used by the loader".into(),
        ));
    };

    // The conditioning phase includes the canonical dense Gemma snapshot, the LTX connector, the
    // request-scoped VAE encoder used by I2V, and the optional uncensored Gemma artifact when that
    // exact overlay is loaded. The encoder is released after its fitted image latents materialize;
    // keeping it in this phase floor accounts its widest instant without carrying it into denoise.
    let conditioning_bytes = checked_sum(
        "conditioning",
        [
            required_projected_safetensors_bytes(
                gemma_dir,
                "Gemma text encoder",
                ResidentProjection::Bfloat16,
            )?,
            required_projected_safetensors_bytes(
                &root.join("connector.safetensors"),
                "connector",
                ResidentProjection::Bfloat16,
            )?,
            required_projected_safetensors_bytes(
                &root.join("vae_encoder.safetensors"),
                "video VAE encoder",
                ResidentProjection::Float32,
            )?,
            uncensored_enhancer_resident_bytes(spec)?,
        ],
    )?;

    // The upsampler runs between the two denoise passes while the AvDiT is live, so it belongs to
    // the denoise-phase base footprint rather than to decode.
    let transformer_bytes = checked_sum(
        "denoise",
        [
            required_projected_safetensors_bytes(
                &root.join("transformer.safetensors"),
                "AudioVideo transformer",
                ResidentProjection::Stored,
            )?,
            required_projected_safetensors_bytes(
                &root.join("upsampler.safetensors"),
                "latent upsampler",
                ResidentProjection::Stored,
            )?,
        ],
    )?;

    // The decoder phase excludes the request-scoped encoder charged above.
    let decoder_bytes = checked_sum(
        "decode",
        [
            required_projected_safetensors_bytes(
                &root.join("vae_decoder.safetensors"),
                "video VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("audio_vae.safetensors"),
                "audio VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("vocoder.safetensors"),
                "vocoder",
                ResidentProjection::Float32,
            )?,
        ],
    )?;

    let overlay_bytes = match gen_core::adapter_stack_resident_bytes(
        &spec.adapters,
        AdapterResidencyMode::Additive,
    ) {
        Some(bytes) => bytes,
        None => {
            return Err(mlx_gen::Error::Msg(
                "ltx_2_3: every additive adapter must have a non-zero load-exact safetensors size"
                    .into(),
            ));
        }
    };
    let resident_components = (overlay_bytes > 0)
        .then(|| MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        })
        .into_iter()
        .collect();

    let base_bytes = checked_sum(
        "base model",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    Ok(AssetDeclaration {
        facts: MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    })
}

fn quant_from_split(split: &SplitModel) -> Result<Option<Quant>> {
    if !split.quantized {
        return Ok(None);
    }
    match split.bits {
        4 => Ok(Some(Quant::Q4)),
        8 => Ok(Some(Quant::Q8)),
        bits => Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: split_model.json declares unsupported {bits}-bit transformer weights"
        ))),
    }
}

pub(crate) fn numeric_tier_from_split(
    spec: &LoadSpec,
    split: &SplitModel,
) -> Result<MemoryNumericTier> {
    let quant = quant_from_split(split)?;
    if let Some(requested) = spec.quantize {
        if quant != Some(requested) {
            return Err(mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: requested {requested:?} does not match the checkpoint tier {quant:?}"
            )));
        }
    }
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> Result<MemoryNumericTier> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 numeric tier requires a split model directory".into(),
        ));
    };
    numeric_tier_from_split(spec, &SplitModel::from_model_dir(root)?)
}

/// Canonical request overlay spelling shared with the SceneWorks admission identity.
///
/// Adapter count is part of the identity (the auto-distill adapter is therefore not allowed to
/// borrow a plain-T2V cell), enhancer is a load-bearing asset axis, and the provider mode is a
/// request-only axis.  The latter is appended by SceneWorks because `LoadSpec` does not carry the
/// per-request `video_mode` field.
pub(crate) fn canonical_overlay(
    adapter_count: usize,
    enhancer: Option<&str>,
    video_mode: Option<&str>,
) -> Option<String> {
    let mut axes = Vec::new();
    if adapter_count > 0 {
        axes.push(format!("adapters:{adapter_count}"));
    }
    if let Some(enhancer) = enhancer {
        axes.push(format!("enhancer:{enhancer}"));
    }
    if let Some(video_mode) = video_mode {
        axes.push(format!("provider_video_mode:{video_mode}"));
    }
    (!axes.is_empty()).then(|| axes.join("+"))
}

pub(crate) fn route_overlay(spec: &LoadSpec) -> Option<String> {
    canonical_overlay(
        spec.adapters.len(),
        spec.components
            .contains_key("uncensored_enhancer")
            .then_some("uncensored"),
        None,
    )
}

/// Compare a request overlay to the loaded route without dropping request-only axes.
///
/// The provider contract is created from `LoadSpec`, so it can only know the loaded adapter and
/// enhancer axes.  `provider_video_mode:*` is carried by the request identity and is accepted only
/// as an additional, explicitly named axis.  Unknown or missing load axes still fail closed.
fn overlay_matches_loaded_route(actual: Option<&str>, expected: Option<&str>) -> bool {
    let load_axes = |overlay: Option<&str>| {
        overlay
            .into_iter()
            .flat_map(|value| value.split('+'))
            .filter(|axis| {
                !axis.starts_with("provider_video_mode:")
                    && !axis.starts_with("reference:image:")
                    && axis != &"enhancer:standard"
            })
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let actual_axes = actual.into_iter().flat_map(|value| value.split('+'));
    if actual_axes.clone().any(|axis| {
        axis.starts_with("provider_video_mode:") && !matches!(axis, "provider_video_mode:no_audio")
    }) {
        return false;
    }
    load_axes(actual) == load_axes(expected)
}

fn reference_axis(width: u32, height: u32) -> String {
    format!("reference:image:{width}x{height}:strength:{I2V_STRENGTH_BITS:08x}")
}

fn admitted_reference_axis(overlay: Option<&str>) -> Option<&str> {
    overlay
        .into_iter()
        .flat_map(|value| value.split('+'))
        .find(|axis| axis.starts_with("reference:image:"))
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: decode_tile_edges(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

fn build_contract(
    spec: &LoadSpec,
    asset_declaration: AssetDeclaration,
    calibration_fingerprint: &str,
) -> Result<MemoryProviderContract> {
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3 has no deferred/block-window loader; use EagerMaterialization".into(),
        ));
    }
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::FrameCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    let formula = if asset_declaration.resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope {
            phases: phases.clone(),
            variables,
        }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases.clone(),
            variables,
            resident_components: asset_declaration.resident_components,
        }
    };
    Ok(MemoryProviderContract {
        provider_id: crate::MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(),
        // LTX declares no decode-quality geometry policy table, so this route carries no semantic
        // decode authority — the fail-closed default every non-declaring provider contract uses.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        // LTX stages Gemma before the AvDiT for every render. A selected decode rung therefore
        // co-engages rung 1 even though the shared cost-order default intentionally does not.
        additional_prerequisites: vec![(
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )],
        default_engagement_exclusions: Vec::new(),
        // Gen-core requires Resident as the no-new-control baseline. It does not assert literal
        // co-residency: preserving defaults keeps LTX's historical always-staged phase order and
        // automatic decode guard. SC-18816 will make that unconditional staging separately visible.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula,
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: asset_declaration.facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
    split: &SplitModel,
    gemma_dir: &std::path::Path,
    gemma_quant: Option<GemmaQuant>,
) -> Result<Option<(MemoryProviderContract, MemoryNumericTier, Option<String>)>> {
    // The first production evidence campaign is calibrated against the canonical dense-bf16 Gemma
    // route. Quantized Gemma remains a supported generator input, but must fail open until it has a
    // separately identifiable and measured contract rather than borrowing the canonical evidence.
    if gemma_quant.is_some() || !adapters_have_load_exact_additive_accounting(spec)? {
        return Ok(None);
    }
    let tier = numeric_tier_from_split(spec, split)?;
    let contract = build_contract(
        spec,
        production_asset_declaration(spec, gemma_dir)?,
        CALIBRATION_FINGERPRINT,
    )?;
    Ok(Some((contract, tier, route_overlay(spec))))
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires a split model directory".into(),
        ));
    };
    let split = SplitModel::from_model_dir(root)?;
    let _tier = numeric_tier_from_split(spec, &split)?;
    let gemma_dir = crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    if let Some(quant) = crate::model::resolve_gemma_quant(&gemma_dir)? {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: the calibrated memory contract requires canonical dense-bf16 Gemma; \
             the requested Gemma snapshot declares {}-bit group-{} quantization",
            quant.bits, quant.group
        )));
    }
    if !adapters_have_load_exact_additive_accounting(spec)? {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: calibrated memory admission supports additive LoRA factors but not \
             LoKr/LoHa routes that reconstruct dense bf16 deltas"
                .into(),
        ));
    }
    build_contract(
        spec,
        production_asset_declaration(spec, &gemma_dir)?,
        CALIBRATION_FINGERPRINT,
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        AssetDeclaration {
            facts: MemoryAssetFacts::default(),
            resident_components: Vec::new(),
        },
        STATIC_CALIBRATION_FINGERPRINT,
    )
    .map_err(Into::into)
}

/// LTX witnesses the shared MLX tiers under both offload policies, but only the eager half of the
/// materialization axis. `build_contract` rejects `DeferredMaterialization` outright — LTX has no
/// deferred/block-window loader — so publishing the deferred selectors would advertise a load
/// surface no contract can be built for, and the registry conformance walk (which constructs every
/// published selector) would fail the whole MLX catalog. The witness set is the provider's own
/// finite inventory, not the shared default.
pub(crate) fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

fn fixture_contract(contract: &MemoryProviderContract) -> bool {
    contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION_FINGERPRINT)
}

fn registered_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> Result<MemoryNumericTier> {
    if fixture_contract(contract) {
        return Ok(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        });
    }
    resolved_numeric_tier(spec)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile edge".into())
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile overlap".into())
    })?;
    let edges = decode_tile_edges();
    if !edges.contains(&edge) {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode tile edge {edge} is outside the production domain {edges:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode overlap {overlap} is not the production overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let t2v = context.mode.as_key() == "text_to_video"
            && context.geometry.reference_count == 0
            && !context.has_reference;
        let i2v = context.mode.as_key() == "image_to_video"
            && context.geometry.reference_count == 1
            && context.has_reference;
        if (!t2v && !i2v) || context.use_pid || context.has_phases {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: memory admission requires exact text_to_video/no-reference or image_to_video/one-reference identity without PiD/phases"
                    .into(),
            ));
        }
        if !overlay_matches_loaded_route(context.overlay.as_deref(), expected_overlay) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: memory overlay {:?} does not match the loaded route {:?}",
                context.overlay, expected_overlay
            )));
        }
        let geometry = context.geometry;
        if geometry.batch != 1
            || !SHIPPED_GEOMETRIES.contains(&(geometry.width, geometry.height))
            || !SHIPPED_FRAME_COUNTS.contains(&geometry.frames)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: unsupported memory geometry {}x{}x{} frames={}",
                geometry.width, geometry.height, geometry.batch, geometry.frames
            )));
        }
        let reference = admitted_reference_axis(context.overlay.as_deref());
        if t2v && reference.is_some() {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: text_to_video admission cannot carry an image reference receipt".into(),
            ));
        }
        if i2v && reference != Some(reference_axis(geometry.width, geometry.height).as_str()) {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: image_to_video admission requires the exact fitted image shape and fixed strength receipt"
                    .into(),
            ));
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
        Some(loaded_tier),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier(spec, contract) {
        Ok(tier) => safety_check(contract, tier, route_overlay(spec).as_deref(), context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    let mut t2v_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: route_overlay(spec),
        },
    )?;
    t2v_context.geometry.width = 768;
    t2v_context.geometry.height = 512;
    t2v_context.geometry.frames = 153;
    let mut t2v = MemoryBehaviorFixture::new(t2v_context);
    t2v.request.width = 768;
    t2v.request.height = 512;
    t2v.request.frames = Some(153);
    t2v.request.fps = Some(25);

    let reference_receipt = reference_axis(768, 512);
    let i2v_overlay = match route_overlay(spec) {
        Some(load) => format!("{load}+{reference_receipt}"),
        None => reference_receipt,
    };
    let mut i2v_context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("image_to_video".into()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some(i2v_overlay),
        },
    )?;
    i2v_context.geometry.width = 768;
    i2v_context.geometry.height = 512;
    i2v_context.geometry.frames = 153;
    let mut i2v = MemoryBehaviorFixture::new(i2v_context);
    i2v.request.width = 768;
    i2v.request.height = 512;
    i2v.request.frames = Some(153);
    i2v.request.fps = Some(25);
    i2v.request.conditioning = vec![gen_core::Conditioning::Reference {
        image: gen_core::Image {
            width: 768,
            height: 512,
            pixels: vec![0; 768 * 512 * 3],
        },
        strength: Some(1.0),
    }];
    Ok(vec![t2v, i2v])
}

fn begin_with_cleanup(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, loaded_tier, expected_overlay, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        crate::MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    Ok(Some(Box::new(LtxMemoryRequestScope {
        inner: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
        expected_overlay: expected_overlay.map(str::to_owned),
        admitted_overlay: context.overlay.clone(),
        admitted_mode: context.mode.as_key().to_owned(),
    })))
}

/// LTX's calibrated evidence is deliberately narrower than the generator's complete capability
/// surface. Bind the admitted context to the actual request before the shared core installs any
/// selected controls, including axes that [`GenerationRequest::image_reference_count`] does not
/// represent (temporal conditioning and prompt enhancement).
struct LtxMemoryRequestScope {
    inner: mlx_gen::request_scope::MlxRequestScopeCore,
    expected_overlay: Option<String>,
    admitted_overlay: Option<String>,
    admitted_mode: String,
}

impl LtxMemoryRequestScope {
    fn validate_request(
        request: &GenerationRequest,
        expected_overlay: Option<&str>,
        admitted_overlay: Option<&str>,
        admitted_mode: &str,
    ) -> gen_core::Result<()> {
        let mut reference = None;
        for conditioning in &request.conditioning {
            let gen_core::Conditioning::Reference { image, strength } = conditioning else {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: this memory route accepts only the single fitted image Reference carrier"
                        .into(),
                ));
            };
            if reference.is_some() {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: image_to_video memory admission requires exactly one Reference"
                        .into(),
                ));
            }
            if image.width != request.width || image.height != request.height {
                return Err(gen_core::Error::Unsupported(format!(
                    "ltx_2_3: fitted reference shape {}x{} does not match output {}x{}",
                    image.width, image.height, request.width, request.height
                )));
            }
            let strength = strength.or(request.strength).unwrap_or(1.0);
            if strength.to_bits() != I2V_STRENGTH_BITS {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: image_to_video memory admission requires fixed strength 1.0".into(),
                ));
            }
            reference = Some(reference_axis(image.width, image.height));
        }
        match (admitted_mode, reference.as_deref()) {
            ("text_to_video", None) => {}
            ("image_to_video", Some(actual))
                if Some(actual) == admitted_reference_axis(admitted_overlay) => {}
            ("image_to_video", None) => {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: image_to_video admission requires one fitted Reference".into(),
                ));
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "ltx_2_3: request mode/reference receipt crossed the admitted memory identity"
                        .into(),
                ));
            }
        }
        // These controls are part of the exact request identity.  They are safe to carry through
        // a future variant-specific evidence cell; they must not be silently erased or borrow the
        // plain cell.  `SceneWorks` keeps them fail-open until that cell is packaged.
        let expected_has_enhancer = expected_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .any(|axis| axis == "enhancer:uncensored");
        let request_enhancer = if request.use_uncensored_enhancer {
            Some("uncensored")
        } else if request.enhance_prompt {
            Some("standard")
        } else {
            None
        };
        let admitted_enhancer = admitted_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .find_map(|axis| axis.strip_prefix("enhancer:"));
        if request_enhancer != admitted_enhancer {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: enhancer flavor does not match admitted overlay".into(),
            ));
        }
        if request.use_uncensored_enhancer && !expected_has_enhancer {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: uncensored enhancer does not match the loaded overlay".into(),
            ));
        }
        let admitted_video_mode = admitted_overlay
            .into_iter()
            .flat_map(|overlay| overlay.split('+'))
            .find_map(|axis| axis.strip_prefix("provider_video_mode:"));
        if let Some(video_mode) = request.video_mode.as_deref() {
            if video_mode != "no_audio" {
                return Err(gen_core::Error::Unsupported(format!(
                    "ltx_2_3: unsupported video_mode variant {video_mode:?}"
                )));
            }
        }
        if request.video_mode.as_deref() != admitted_video_mode {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: provider video mode does not match admitted overlay".into(),
            ));
        }
        if let Some(tokens) = request.enhance_max_tokens {
            if tokens == 0 || tokens > gen_core::generator::MAX_ENHANCE_TOKENS {
                return Err(gen_core::Error::Unsupported(format!(
                    "ltx_2_3: enhance_max_tokens must be in 1..={}, got {tokens}",
                    gen_core::generator::MAX_ENHANCE_TOKENS
                )));
            }
        }
        let fps = request.fps.unwrap_or(24);
        if !matches!(fps, 24 | 25 | 30) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: calibrated memory route requires public fps 24, 25, or 30, got {fps}"
            )));
        }
        if !SHIPPED_GEOMETRIES.contains(&(request.width, request.height))
            || !frame_count_matches_fps(fps, request.frames.unwrap_or(1))
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: request geometry/frame count is outside the shipped LTX ladder".into(),
            ));
        }
        if request
            .sampler
            .as_deref()
            .is_some_and(|sampler| sampler != "euler")
            || request.scheduler.is_some()
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: conditioned memory admission preserves the native Euler/fixed schedule"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl MemoryRequestScope for LtxMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        Self::validate_request(
            request,
            self.expected_overlay.as_deref(),
            self.admitted_overlay.as_deref(),
            &self.admitted_mode,
        )?;
        self.inner.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        contract,
        loaded_tier,
        expected_overlay,
        transformer_blocks,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    begin_with_cleanup(
        contract,
        tier,
        route_overlay(spec).as_deref(),
        LtxConfig::video_only_defaults().num_layers as usize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn decode_tiling(request: &GenerationRequest) -> Result<Option<TilingConfig>> {
    let Some(memory) = request.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_tile_edge".into(),
        )
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_overlap".into(),
        )
    })?;
    validate_decode(Some(edge), Some(overlap)).map_err(mlx_gen::Error::from)?;
    crate::pipeline::selected_tiling_budgeted_ltx(
        request.height as i32,
        request.width as i32,
        request.frames.unwrap_or(1) as i32,
        edge as i32,
        overlap as i32,
    )
    .map(Some)
}

pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: |spec| memory_strategy_contract(spec).map_err(Into::into),
    safety_check: registered_safety_check,
};

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        Conditioning, GenerationMemory, Image, MemoryStrategyParameters, Precision, ReplacementMode,
    };

    fn fixture_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into())).with_quant(Quant::Q8)
    }

    /// `ProviderRegistry::memory_contract_surfaces` constructs a contract for **every** selector the
    /// fixture publishes and fails the entire MLX catalog when one errors, so the published witness
    /// set must be exactly the set this provider can build. Asserting it here localizes the failure
    /// to LTX instead of surfacing it as eight red `mlx-gen-catalog` tests.
    #[test]
    fn every_published_contract_surface_builds_and_no_deferred_surface_is_published() {
        let surfaces = memory_contract_surface_specs();
        assert_eq!(
            surfaces.len(),
            gen_core::mlx_memory_contract_surface_specs().len() / 2,
            "the witness set is the eager half of the shared MLX surface"
        );
        for surface in &surfaces {
            assert_eq!(
                surface.selector.load_shape,
                LoadShape::EagerMaterialization,
                "{} has no deferred/block-window loader",
                surface.selector.id()
            );
            weights_free_memory_strategy_contract(&surface.spec).unwrap_or_else(|error| {
                panic!("surface {} must build: {error}", surface.selector.id())
            });
        }
        assert!(
            gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .filter(|surface| surface.selector.load_shape == LoadShape::DeferredMaterialization)
                .all(|surface| weights_free_memory_strategy_contract(&surface.spec).is_err()),
            "a deferred surface that now builds must be published, not filtered out"
        );
    }

    #[test]
    fn survey_rungs_and_video_formula_are_declared_without_a_default_standin() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert!(matches!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        for missing in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(matches!(
                contract.capability(missing).unwrap().support,
                MemoryStrategySupport::Missing
            ));
        }
        assert!(contract.formula.uses(MemoryFormulaVariable::FrameCount));
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ]
        );

        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, tier, false)
            .unwrap();
        let staged = contract
            .representative_selection(MemoryStrategy::StagedResidency, tier, false)
            .unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::Resident),
            [MemoryStrategy::Resident]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::StagedResidency),
            [MemoryStrategy::Resident, MemoryStrategy::StagedResidency]
        );
        assert_eq!(contract.generation_memory(&resident), None);
        assert_eq!(
            contract.generation_memory(&staged),
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn selected_decode_parameters_reach_the_real_tiling_carrier_and_mutations_fail() {
        let mut request = GenerationRequest {
            // Keep the carrier assertion below the smallest supported CI host's live MLX budget.
            // Long-clip temporal-budget behavior is covered by the injected-budget pipeline test;
            // this test owns only request-memory propagation and domain validation.
            width: 256,
            height: 256,
            frames: Some(25),
            count: 1,
            memory: Some(GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tiling = decode_tiling(&request).unwrap().unwrap();
        let spatial = tiling.spatial.unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));

        request.memory.as_mut().unwrap().decode_tile_edge = Some(255);
        assert!(decode_tiling(&request).is_err());
        request.memory.as_mut().unwrap().decode_tile_edge = None;
        assert!(decode_tiling(&request).is_err());

        let contract = weights_free_memory_strategy_contract(&fixture_spec()).unwrap();
        let mut selection = contract
            .representative_selection(
                MemoryStrategy::BoundedDecode,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
                false,
            )
            .unwrap();
        selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(511),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(contract.validate_selection(&selection).is_err());
    }

    fn calibrated_t2v_scope_and_request() -> (Box<dyn MemoryRequestScope>, GenerationRequest) {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
            .unwrap();
        let scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        (scope, fixture.request)
    }

    #[test]
    fn calibrated_scope_binds_empty_t2v_request_and_exact_fps_frame_pair_before_controls() {
        let (mut scope, request) = calibrated_t2v_scope_and_request();
        let mut accepted = request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let mut crossed_pair = GenerationRequest {
            fps: Some(24),
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut crossed_pair)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the shipped LTX ladder"));
        assert_eq!(crossed_pair.memory, None);

        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        for conditioning in [
            Conditioning::Keyframe {
                image: image.clone(),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::VideoClip {
                frames: vec![image.clone()],
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::ControlClip {
                frames: vec![image.clone()],
                mask: vec![image],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            },
        ] {
            let mut temporal_conditioning = GenerationRequest {
                conditioning: vec![conditioning],
                ..request.clone()
            };
            assert_eq!(temporal_conditioning.image_reference_count(), 0);
            let error = scope
                .configure_request(&mut temporal_conditioning)
                .unwrap_err()
                .to_string();
            assert!(error.contains("accepts only the single fitted image Reference carrier"));
            assert_eq!(temporal_conditioning.memory, None);
        }

        for variant in [
            GenerationRequest {
                enhance_prompt: true,
                ..request.clone()
            },
            GenerationRequest {
                use_uncensored_enhancer: true,
                ..request.clone()
            },
        ] {
            let mut variant = variant;
            let error = scope
                .configure_request(&mut variant)
                .unwrap_err()
                .to_string();
            assert!(error.contains("enhancer flavor does not match admitted overlay"));
            assert_eq!(variant.memory, None);
        }

        let mut enhancer_spec = fixture_spec();
        enhancer_spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir("/nonexistent-enhancer-fixture".into()),
        );
        let enhancer_contract = weights_free_memory_strategy_contract(&enhancer_spec).unwrap();
        let enhancer_fixture = registered_valid_fixtures(
            &enhancer_spec,
            &enhancer_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap();
        let mut enhancer_scope = registered_begin_request(
            &enhancer_spec,
            &enhancer_contract,
            &enhancer_fixture.context,
        )
        .unwrap()
        .unwrap();
        let mut crossed_standard = GenerationRequest {
            enhance_prompt: true,
            ..enhancer_fixture.request.clone()
        };
        let error = enhancer_scope
            .configure_request(&mut crossed_standard)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhancer flavor does not match admitted overlay"));
        let mut admitted_uncensored = GenerationRequest {
            enhance_prompt: true,
            use_uncensored_enhancer: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS),
            ..enhancer_fixture.request.clone()
        };
        enhancer_scope
            .configure_request(&mut admitted_uncensored)
            .unwrap();
        assert!(admitted_uncensored.memory.unwrap().stage_residency);
        let mut crossed_token_budget = GenerationRequest {
            enhance_prompt: true,
            use_uncensored_enhancer: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS + 1),
            ..enhancer_fixture.request.clone()
        };
        let error = enhancer_scope
            .configure_request(&mut crossed_token_budget)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhance_max_tokens must be in 1..="));
        assert_eq!(crossed_token_budget.memory, None);

        let standard_spec = fixture_spec();
        let standard_contract = weights_free_memory_strategy_contract(&standard_spec).unwrap();
        let mut standard_context = registered_valid_fixtures(
            &standard_spec,
            &standard_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap()
        .context;
        standard_context.overlay = Some("enhancer:standard".into());
        let mut standard_scope =
            registered_begin_request(&standard_spec, &standard_contract, &standard_context)
                .unwrap()
                .unwrap();
        let mut admitted_standard = GenerationRequest {
            enhance_prompt: true,
            enhance_max_tokens: Some(gen_core::generator::MAX_ENHANCE_TOKENS),
            ..request.clone()
        };
        standard_scope
            .configure_request(&mut admitted_standard)
            .unwrap();
        assert!(admitted_standard.memory.unwrap().stage_residency);
        let mut crossed_uncensored = GenerationRequest {
            use_uncensored_enhancer: true,
            ..request.clone()
        };
        let error = standard_scope
            .configure_request(&mut crossed_uncensored)
            .unwrap_err()
            .to_string();
        assert!(error.contains("enhancer flavor does not match admitted overlay"));

        let mut no_audio = GenerationRequest {
            video_mode: Some("no_audio".into()),
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut no_audio)
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider video mode does not match admitted overlay"));
        assert_eq!(no_audio.memory, None);

        let no_audio_spec = fixture_spec();
        let no_audio_contract = weights_free_memory_strategy_contract(&no_audio_spec).unwrap();
        let mut no_audio_context = registered_valid_fixtures(
            &no_audio_spec,
            &no_audio_contract,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .into_iter()
        .find(|fixture| fixture.context.mode.as_key() == "text_to_video")
        .unwrap()
        .context;
        no_audio_context.overlay = Some("provider_video_mode:no_audio".into());
        let mut no_audio_scope =
            registered_begin_request(&no_audio_spec, &no_audio_contract, &no_audio_context)
                .unwrap()
                .unwrap();
        let mut admitted_no_audio = GenerationRequest {
            video_mode: Some("no_audio".into()),
            ..request.clone()
        };
        no_audio_scope
            .configure_request(&mut admitted_no_audio)
            .unwrap();
        assert!(admitted_no_audio.memory.unwrap().stage_residency);
        let mut crossed_plain = request.clone();
        let error = no_audio_scope
            .configure_request(&mut crossed_plain)
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider video mode does not match admitted overlay"));

        for video_mode in ["video_only", "image_to_video"] {
            let mut variant = GenerationRequest {
                video_mode: Some(video_mode.into()),
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut variant)
                .unwrap_err()
                .to_string();
            assert!(error.contains("unsupported video_mode variant"));
            assert_eq!(variant.memory, None);
        }

        for fps in [Some(0), Some(23), Some(26), Some(29), Some(31)] {
            let mut outside_envelope = GenerationRequest {
                fps,
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut outside_envelope)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires public fps 24, 25, or 30"));
            assert_eq!(outside_envelope.memory, None);
        }

        // Provider-route rejection happens before the shared carrier mutates the request, and it
        // leaves the inner scope active so the caller can report the terminal error and run cleanup.
        scope
            .finish(MemoryRunOutcome::Error {
                message: "request route rejected".into(),
            })
            .unwrap();
        let mut after_finish = request;
        assert!(scope.configure_request(&mut after_finish).is_err());
    }

    #[test]
    fn calibrated_i2v_scope_binds_fitted_reference_strength_schedule_and_frame_identity() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .into_iter()
            .find(|fixture| fixture.context.mode.as_key() == "image_to_video")
            .unwrap();
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();

        let mut accepted = fixture.request.clone();
        scope.configure_request(&mut accepted).unwrap();
        assert!(accepted.memory.unwrap().stage_residency);

        let reference = fixture.request.conditioning[0].clone();
        for (mut crossed, expected) in [
            (
                GenerationRequest {
                    conditioning: Vec::new(),
                    ..fixture.request.clone()
                },
                "requires one fitted Reference",
            ),
            (
                GenerationRequest {
                    conditioning: vec![reference.clone(), reference.clone()],
                    ..fixture.request.clone()
                },
                "requires exactly one Reference",
            ),
            (
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: Image {
                            width: 512,
                            height: 768,
                            pixels: vec![0; 512 * 768 * 3],
                        },
                        strength: Some(1.0),
                    }],
                    ..fixture.request.clone()
                },
                "fitted reference shape",
            ),
            (
                GenerationRequest {
                    conditioning: vec![Conditioning::Reference {
                        image: Image {
                            width: 768,
                            height: 512,
                            pixels: vec![0; 768 * 512 * 3],
                        },
                        strength: Some(0.5),
                    }],
                    ..fixture.request.clone()
                },
                "requires fixed strength 1.0",
            ),
            (
                GenerationRequest {
                    fps: Some(24),
                    ..fixture.request.clone()
                },
                "outside the shipped LTX ladder",
            ),
            (
                GenerationRequest {
                    sampler: Some("heun".into()),
                    ..fixture.request.clone()
                },
                "native Euler/fixed schedule",
            ),
            (
                GenerationRequest {
                    scheduler: Some("linear".into()),
                    ..fixture.request.clone()
                },
                "native Euler/fixed schedule",
            ),
        ] {
            let error = scope
                .configure_request(&mut crossed)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "unexpected refusal: {error}");
            assert_eq!(crossed.memory, None);
        }

        let (mut t2v_scope, mut crossed_i2v) = calibrated_t2v_scope_and_request();
        crossed_i2v.conditioning = fixture.request.conditioning.clone();
        let error = t2v_scope
            .configure_request(&mut crossed_i2v)
            .unwrap_err()
            .to_string();
        assert!(error.contains("crossed the admitted memory identity"));

        let mut crossed_t2v = GenerationRequest {
            conditioning: Vec::new(),
            ..fixture.request
        };
        let error = scope
            .configure_request(&mut crossed_t2v)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires one fitted Reference"));
    }

    #[test]
    fn canonical_overlay_preserves_adapter_count_enhancer_and_provider_mode() {
        assert_eq!(canonical_overlay(0, None, None), None);
        assert_eq!(
            canonical_overlay(2, Some("uncensored"), Some("no_audio")).as_deref(),
            Some("adapters:2+enhancer:uncensored+provider_video_mode:no_audio")
        );
        assert!(overlay_matches_loaded_route(
            Some("adapters:2+enhancer:uncensored+provider_video_mode:no_audio"),
            Some("adapters:2+enhancer:uncensored")
        ));
        assert!(!overlay_matches_loaded_route(
            Some("adapters:1+enhancer:uncensored+provider_video_mode:no_audio"),
            Some("adapters:2+enhancer:uncensored")
        ));
        assert!(!overlay_matches_loaded_route(
            Some("provider_video_mode:video_only"),
            None
        ));
    }

    #[test]
    fn uncensored_enhancer_artifact_is_charged_to_conditioning_floor() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"quantization":{"mode":"affine","group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let header = br#"{"model.embed_tokens.weight":{"dtype":"U32","shape":[2,16],"data_offsets":[0,128]},"model.embed_tokens.scales":{"dtype":"F16","shape":[2,2],"data_offsets":[128,136]},"model.embed_tokens.biases":{"dtype":"F16","shape":[2,2],"data_offsets":[136,144]},"model.layer.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[144,152]}}"#;
        let mut padded = header.to_vec();
        while !(padded.len() + 8).is_multiple_of(8) {
            padded.push(b' ');
        }
        let mut file = (padded.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&padded);
        file.extend_from_slice(&[0; 152]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();

        let mut spec = fixture_spec();
        spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        let enhancer_bytes = uncensored_enhancer_resident_bytes(&spec).unwrap();
        // Stored peak: 152 B. Expanded embedding: [2, 2*64] bf16 = 512 B. Both coexist while MLX
        // evaluates the lazy dequantization graph, so the conditioning floor is their exact sum.
        assert_eq!(enhancer_bytes, 664);

        let contract = build_contract(
            &spec,
            AssetDeclaration {
                facts: MemoryAssetFacts {
                    base_bytes: enhancer_bytes,
                    conditioning_bytes: enhancer_bytes,
                    ..Default::default()
                },
                resident_components: Vec::new(),
            },
            STATIC_CALIBRATION_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, enhancer_bytes);
        assert_eq!(contract.total_resident_bytes(), enhancer_bytes);
    }

    #[test]
    fn uncensored_enhancer_rejects_config_that_disagrees_with_packed_embedding() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"quantization":{"mode":"affine","group_size":64,"bits":8}}"#,
        )
        .unwrap();
        let header = br#"{"model.embed_tokens.weight":{"dtype":"U32","shape":[2,16],"data_offsets":[0,128]},"model.embed_tokens.scales":{"dtype":"F16","shape":[2,2],"data_offsets":[128,136]},"model.embed_tokens.biases":{"dtype":"F16","shape":[2,2],"data_offsets":[136,144]}}"#;
        let mut padded = header.to_vec();
        while !(padded.len() + 8).is_multiple_of(8) {
            padded.push(b' ');
        }
        let mut file = (padded.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&padded);
        file.extend_from_slice(&[0; 144]);
        std::fs::write(dir.path().join("model.safetensors"), file).unwrap();
        let mut spec = fixture_spec();
        spec.components.insert(
            "uncensored_enhancer".into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        let error = uncensored_enhancer_resident_bytes(&spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("geometry disagrees"), "{error}");
    }
}
