//! The **LTX-2.5** MLX memory-strategy registration (sc-18797, epic 18755 R9).
//!
//! This is a SECOND registration in this crate, not a replacement.
//! [`crate::memory_strategy`] keeps declaring `ltx_2_3` exactly as it did; this module declares the
//! 2.5 engine id, whose ladder is genuinely different because 2.5 is the generation that got a
//! deferred block loader ([`crate::block_stream`]) and a bounded attention seam.
//!
//! It lives in its own file rather than as more arms inside `memory_strategy.rs` so the 2.3 contract
//! is not re-shaped by a 2.5 change, and so concurrent 2.5 work on the shared file does not collide
//! here.
//!
//! # What is declared, and what backs each declaration
//!
//! | rung | support | backed by |
//! |---|---|---|
//! | 1 `StagedResidency` | Implemented | LTX's unconditional Wan-style TE staging (epic 10975) |
//! | 2 `BoundedDecode` | Implemented | the shared tiled VAE decode, as 2.3 |
//! | 3 `BoundedAttention` | Implemented | `gen_core::attention_budget` via `mlx_gen::attention::sdpa_budgeted_bhsd` |
//! | 4 `BoundedTransformerResidency` | Implemented **iff streamable** | `gen_core::block_window` via [`crate::block_stream`] |
//!
//! Rungs 3 and 4 are declared here and `Missing` on 2.3 because the code that executes them is
//! reached only through the 2.5 loader. R9 forbids declaring a rung no route can execute, so the
//! support value is computed from the load spec rather than asserted.
//!
//! # Reachability
//!
//! The `ltx_2_5` generator descriptor and split loader are registered beside this contract. The
//! registry, loaded generator, shared request scope, and production transformer constructor all
//! consume this same declaration; removing any one of those links makes the reachability tests fail.

use gen_core::{AdapterKind, AdapterResidencyMode, OffloadPolicy};
use gen_core::{
    GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyPrerequisite, MemoryStrategySupport, ResidentRequestMemory, TransformerComponent,
    WeightsSource,
};
use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core;

use crate::memory_strategy::{decode_tile_edges, DECODE_OVERLAP};
use mlx_gen::gen_core::ltx_checkpoint::LtxComponent;

/// The LTX-2.5 MLX engine id.
///
/// A distinct id from [`crate::MODEL_ID`] (`ltx_2_3`) rather than a route overlay on it: the two
/// generations have different transformer deltas, different text encoders and — the reason it
/// matters here — different ladders. Sharing an id would make the 2.3 contract's `Missing` rungs and
/// this one's `Implemented` rungs two answers to the same question.
pub const LTX_2_5_MODEL_ID: &str = "ltx_2_5";

/// Rung 4's published window cadences over the 48-block AvDiT trunk.
///
/// **Distinguishable by construction.** Peak block residency is `window x per-block bytes` (linear —
/// that is the rung's contract), so each of these bounds twice the previous one: consecutive
/// candidates differ by a factor of 2 in exactly the quantity the rung claims to control, and a
/// calibration sweep across them records signal rather than noise. This is the Z-Image
/// `TRANSFORMER_WINDOW_SIZES = [1]` lesson applied: a domain whose members are indistinguishable in
/// the bounded quantity should have one member, and one whose members are distinguishable should say
/// so arithmetically rather than by taste.
///
/// Every value divides 48 exactly (48 = 2^4 x 3), so no candidate carries a ragged tail window whose
/// peak differs from its nominal size — a tail is not wrong, but it makes the sweep's independent
/// variable inexact.
///
/// `48` (fully resident) is deliberately absent: [`gen_core::block_window::BlockPlan::is_bounded`]
/// is false for an all-covering window, so publishing it would advertise a rung-4 selection that
/// bounds nothing while paying the windowing machinery's cost.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 16];

/// The default cadence — the tightest bound, which is what the rung exists for.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// The only component scope LTX-2.5 windows, and the AV-specific justification the story asks for.
///
/// LTX's audio branch is **not** a second transformer: each `transformer_blocks.{n}` entry is an
/// `AvBlock` carrying the video stack, the audio stack and both cross-modal attentions, so one
/// window over the block axis already bounds both modalities. `Dit` therefore describes what this
/// code does.
///
/// [`TransformerComponent::Both`] would additionally claim the **Gemma-4 text encoder** streams.
/// It does not: the encoder is a separate component, [`crate::block_stream`] does not window it, and
/// nothing is measured for it on this family. Declaring it would be an unreachable declaration —
/// precisely the defect class R9 exists to prevent — so the contract publishes `Dit` alone and a
/// `TextEncoder`/`Both` selection is refused rather than silently served as `Dit`.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Rung 3's published score budgets, in attention-score ELEMENTS per chunk.
///
/// Both engage at production video geometries and neither engages at the small end, which is what
/// makes them a domain rather than a pair of numbers. The arithmetic, from the shipped config
/// (`num_attention_heads = 32`) and a single batch:
///
/// - `768x512x145` -> roughly `24 x 16 x 18 = 6912` video self-attention tokens, so
///   `32 x 6912^2 ~= 1.4 Gi` scores. 64 Mi chunks that ~23 ways; 16 Mi chunks it ~91 ways.
/// - `256x256x9` -> roughly `8 x 8 x 2 = 128` tokens, so `32 x 128^2 = 0.5 Mi` scores. **Neither**
///   budget chunks it, deliberately: 2 MiB of f32 scratch is not what this rung exists to bound.
///
/// 64 Mi is [`gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET`], the shared family
/// operating point, kept first so LTX's evidence is comparable with its siblings'. 16 Mi is a 4x
/// tighter bound for hosts where the operating point is still too loose.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[67_108_864, 16_777_216];

/// The default budget — the tightest published bound.
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;

/// Calibration identity for the 2.5 contract. Distinct from the 2.3 fingerprint: a contract that
/// declares two more rungs is not the same measurement subject, and sharing a fingerprint would let
/// 2.3's evidence authorize a 2.5 selection.
const CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-mlx-ladder-v1";
const STATIC_CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-mlx-registry-v1";

/// Whether this load can execute rung 4.
///
/// Three conditions, each of which is a real refusal in [`crate::block_stream::LtxBlockStream::new`]
/// rather than a policy invented here — the declaration and the loader must agree, or the contract
/// promises something the loader will reject at request time:
///
/// 1. **Deferred materialization.** Rung 4's shared prerequisite. A block window over an
///    already-materialized trunk adds a copy and bounds nothing.
/// 2. **No adapters.** LTX installs LoRA onto loaded block objects, so a per-window rebuild from the
///    base component would silently carry none of them.
/// 3. **A directory bundle.** The stream reopens the resolved *transformer component* file; a
///    single-file spec is not a 2.5 split bundle.
fn streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn uses_diffusion_decoder(spec: &LoadSpec) -> bool {
    spec.components
        .contains_key(LtxComponent::DiffusionVideoVae.id())
}

fn strategies(spec: &LoadSpec) -> Vec<MemoryStrategyCapability> {
    let windowed = streamable(spec);
    let bounded_decode = !uses_diffusion_decoder(spec);
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedDecode if bounded_decode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedDecode => MemoryStrategySupport::Missing,
                MemoryStrategy::BoundedTransformerResidency if windowed => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode if bounded_decode => MemoryParameterRanges {
                    decode_tile_edges: decode_tile_edges(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if windowed => MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

struct AssetDeclaration {
    facts: gen_core::MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> gen_core::Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "ltx_2_5: {label} resident-byte total overflows u64"
            ))
        })
    })
}

fn required_projected_bytes(
    path: &std::path::Path,
    label: &str,
    projection: ResidentProjection,
) -> gen_core::Result<u64> {
    let bytes = projected_safetensors_bytes(path, |_| projection)?;
    if bytes == 0 {
        return Err(gen_core::Error::Msg(format!(
            "ltx_2_5: {label} has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn optional_projected_bytes(
    path: &std::path::Path,
    projection: ResidentProjection,
) -> gen_core::Result<u64> {
    if path.exists() {
        projected_safetensors_bytes(path, |_| projection)
    } else {
        Ok(0)
    }
}

fn adapters_have_load_exact_additive_accounting(spec: &LoadSpec) -> gen_core::Result<bool> {
    for adapter in &spec.adapters {
        if adapter.kind == AdapterKind::Lokr {
            return Ok(false);
        }
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)?;
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

fn production_asset_declaration(spec: &LoadSpec) -> gen_core::Result<AssetDeclaration> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "ltx_2_5: memory contract requires the split bundle directory used by the loader"
                .to_owned(),
        ));
    };
    let bundle = crate::bundle::resolve_split_bundle(spec)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
    let video_component = if uses_diffusion_decoder(spec) {
        LtxComponent::DiffusionVideoVae
    } else {
        LtxComponent::ConvVideoVae
    };
    let video_path = bundle
        .require(video_component)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?
        .path()
        .to_path_buf();
    let encoder_path = crate::model::ltx25_encoder_path(root, video_component, &video_path);
    let transformer_path = bundle
        .require(LtxComponent::Transformer)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?
        .path()
        .to_path_buf();
    let connector_path = transformer_path.with_file_name("connector.safetensors");
    let enhancer_path = crate::model::ltx25_enhancer_dir(spec)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;

    // Every component is projected to the dtype retained by its concrete constructor. Packed
    // `.weight` tensors are automatically kept at their stored width by `asset_facts`, even when
    // the surrounding dense component is promoted to f32.
    let conditioning_bytes = checked_sum(
        "conditioning",
        [
            required_projected_bytes(
                bundle
                    .require(LtxComponent::TextEncoder)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .path(),
                "Gemma-4 text encoder",
                ResidentProjection::Stored,
            )?,
            required_projected_bytes(
                &connector_path,
                "audio/video connector",
                ResidentProjection::Stored,
            )?,
            required_projected_bytes(
                &encoder_path,
                "selected video VAE encoder",
                ResidentProjection::Float32,
            )?,
            required_projected_bytes(
                bundle
                    .require(LtxComponent::DurationHead)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .path(),
                "duration head",
                ResidentProjection::Float32,
            )?,
            // The provider resolves this staged Gemma-4 snapshot at load and enhancement runs it
            // while the ordinary packed text encoder is still in the conditioning scope. Charge
            // it whenever present, matching the established LTX-2.3 enhancer accounting rule.
            optional_projected_bytes(&enhancer_path, ResidentProjection::Bfloat16)?,
        ],
    )?;
    let transformer_bytes = checked_sum(
        "denoise",
        [
            required_projected_bytes(
                &transformer_path,
                "audio/video transformer",
                ResidentProjection::Stored,
            )?,
            required_projected_bytes(
                bundle
                    .require(LtxComponent::SpatialUpsampler)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .path(),
                "spatial latent upsampler",
                ResidentProjection::Stored,
            )?,
            required_projected_bytes(
                bundle
                    .require(LtxComponent::TemporalUpsampler)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .path(),
                "temporal latent upsampler",
                ResidentProjection::Stored,
            )?,
        ],
    )?;
    let decoder_bytes = checked_sum(
        "decode",
        [
            required_projected_bytes(
                &video_path,
                "selected video VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_bytes(
                bundle
                    .require(LtxComponent::AudioVae)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .path(),
                "audio VAE and vocoder",
                ResidentProjection::Float32,
            )?,
        ],
    )?;
    if !adapters_have_load_exact_additive_accounting(spec)? {
        return Err(gen_core::Error::Unsupported(
            "ltx_2_5: calibrated memory admission supports additive LoRA factors but not LoKr/LoHa routes that reconstruct dense deltas".to_owned(),
        ));
    }
    let overlay_bytes =
        gen_core::adapter_stack_resident_bytes(&spec.adapters, AdapterResidencyMode::Additive)
            .ok_or_else(|| {
                gen_core::Error::Msg(
            "ltx_2_5: every additive adapter must have a non-zero load-exact safetensors size"
                .to_owned(),
        )
            })?;
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
        facts: gen_core::MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    })
}

fn build_contract(
    spec: &LoadSpec,
    assets: AssetDeclaration,
    calibration_fingerprint: &str,
) -> gen_core::Result<MemoryProviderContract> {
    let windowed = streamable(spec);
    let bounded_decode = !uses_diffusion_decoder(spec);
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut additional_prerequisites = vec![
        // Inherited from 2.3: LTX stages Gemma before the AvDiT on every render, so every selected
        // optimized rung below co-engages rung 1 even though the shared cost-order default does
        // not. Keep these edges on the provider: staging is LTX's shipped floor, not a universal
        // prerequisite for the same shared scratch-bounding rungs on other families.
        (
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ),
        (
            MemoryStrategy::BoundedAttention,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ),
    ];
    if windowed {
        // **Gate streaming to `OffloadPolicy::Sequential`** (the story's explicit design decision).
        //
        // Rung 4 bounds the 48-block trunk. Under a resident phase policy the Gemma-4 text encoder
        // and the VAE stay co-resident for the whole request, so the request PEAK is set by them and
        // bounding the trunk moves it by nothing — while the streamed stack still re-materializes
        // every block once per step and pays roughly 2x the latency, for identical output. That is a
        // pure loss, and the honest place to say so is the contract.
        //
        // `MemoryStrategy::engages` cannot supply this edge: rung 4 does NOT engage rung 1
        // universally (SC-15998 removed exactly that assumption, because a resident phase policy can
        // legitimately coexist with deferred block materialization on families whose other
        // components are small). LTX's components are not small, which is why this is declared here
        // as a provider-specific prerequisite — the sanctioned seam — rather than argued into the
        // shared graph.
        additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(MemoryProviderContract {
        // Shared with the 2.3 route: the measured 2.3 to 2.5 delta is two booleans, not a
        // dimension, so both routes publish the same axes from one derivation (SC-22662).
        architecture_facts: crate::memory_strategy::architecture_facts(),
        provider_id: LTX_2_5_MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            // Also what makes the MLX realization answer `DeviceFormatTransfer` for rung 4's
            // per-window cost obligation: a window maps lazy safetensors handles and reads the
            // block's own packed triples, with no host-side repack in the per-window path.
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(spec),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites,
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: bounded_decode,
            // Rung 3 is wired through `Attention::attn_budget` on all six attentions of every block.
            attention_chunking: true,
            // Tracks the rung-4 support value exactly: conformance rejects an `Implemented` rung 4
            // whose lifecycle does not claim the capability, and claiming it on a non-streamable
            // load would be the inverse lie.
            transformer_window_materialization: windowed,
        },
        formula: if assets.resident_components.is_empty() {
            MemoryFormulaKind::PhaseEnvelope {
                phases,
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::FrameCount,
                    MemoryFormulaVariable::BatchCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    MemoryFormulaVariable::OverlayBytes,
                    MemoryFormulaVariable::DecodeTileArea,
                ],
            }
        } else {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::FrameCount,
                    MemoryFormulaVariable::BatchCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    MemoryFormulaVariable::OverlayBytes,
                    MemoryFormulaVariable::DecodeTileArea,
                ],
                resident_components: assets.resident_components,
            }
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: assets.facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// Build the production LTX-2.5 memory contract from the exact files the loader resolves.
pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        production_asset_declaration(spec)?,
        CALIBRATION_FINGERPRINT,
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        AssetDeclaration {
            facts: gen_core::MemoryAssetFacts::default(),
            resident_components: Vec::new(),
        },
        STATIC_CALIBRATION_FINGERPRINT,
    )
}

/// Resolve the load-exact numeric tier from the converted checkpoint's own packing manifest.
///
/// LTX tiers are physically pre-packed; `LoadSpec::quantize` is only an optional assertion. Reading
/// the manifest here prevents an omitted assertion from pricing a q4/q8 tree as dense bf16.
pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(path) => {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_5: numeric-tier resolution requires a split bundle directory, got {}",
                path.display()
            )))
        }
    };
    let manifest = root.join("split_model.json");
    let quant = if manifest.is_file() {
        let split = crate::config::SplitModel::from_model_dir(root)
            .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
        let physical = if split.quantized {
            Some(match split.bits {
                4 => gen_core::Quant::Q4,
                8 => gen_core::Quant::Q8,
                bits => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "ltx_2_5: split_model.json declares unsupported {bits}-bit packing"
                    )))
                }
            })
        } else {
            None
        };
        if spec.quantize != physical && spec.quantize.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_5: requested quant tier {:?} disagrees with split_model.json tier {physical:?}",
                spec.quantize
            )));
        }
        physical
    } else {
        // Weights-free registry fixtures intentionally point at nonexistent paths. Their explicit
        // quant selector is the synthetic tier axis; production converted tiers always carry the
        // manifest above, so this branch cannot override a real artifact's physical identity.
        spec.quantize
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported(
            "ltx_2_5: bounded decode requires a selected tile edge".to_owned(),
        )
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported(
            "ltx_2_5: bounded decode requires a selected overlap".to_owned(),
        )
    })?;
    if !decode_tile_edges().contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_5: decode tile {edge}/{overlap} is outside the published domain"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransformerExecution {
    pub stream_blocks: bool,
    pub window_size: Option<u32>,
    pub attention_chunk_size: Option<u32>,
}

fn require_implemented(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<()> {
    if matches!(
        contract
            .capability(strategy)
            .map(|capability| &capability.support),
        Some(MemoryStrategySupport::Implemented)
    ) {
        Ok(())
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "ltx_2_5: request selects {strategy:?}, but the loaded contract does not implement it"
        )))
    }
}

/// Validate a direct request against the exact loaded contract.
///
/// Request scopes populate every selected parameter. Direct callers can bypass that scope, so this
/// production-path check rejects missing, stray, and out-of-domain controls instead of inventing
/// defaults that no admission decision selected.
pub(crate) fn validate_request_memory(
    contract: &MemoryProviderContract,
    memory: &GenerationMemory,
) -> gen_core::Result<TransformerExecution> {
    if memory.tile_vae_decode {
        require_implemented(contract, MemoryStrategy::BoundedDecode)?;
        validate_decode(memory.decode_tile_edge, memory.decode_overlap)?;
    } else if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
        return Err(gen_core::Error::Unsupported(
            "ltx_2_5: decode tile parameters require tile_vae_decode=true".to_owned(),
        ));
    }

    let attention_chunk_size = if memory.chunk_attention {
        require_implemented(contract, MemoryStrategy::BoundedAttention)?;
        let size = memory.attention_chunk_size.ok_or_else(|| {
            gen_core::Error::Unsupported(
                "ltx_2_5: chunk_attention requires attention_chunk_size".to_owned(),
            )
        })?;
        if !ATTENTION_CHUNK_SIZES.contains(&size) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_5: attention chunk {size} is outside the published domain {ATTENTION_CHUNK_SIZES:?}"
            )));
        }
        Some(size)
    } else {
        if memory.attention_chunk_size.is_some() {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_5: attention_chunk_size requires chunk_attention=true".to_owned(),
            ));
        }
        None
    };

    let window_size = if memory.stream_transformer_blocks {
        require_implemented(contract, MemoryStrategy::BoundedTransformerResidency)?;
        if !memory.stage_residency {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_5: streamed transformer blocks require staged residency in the same request"
                    .to_owned(),
            ));
        }
        let window = memory.transformer_window_size.ok_or_else(|| {
            gen_core::Error::Unsupported(
                "ltx_2_5: stream_transformer_blocks requires transformer_window_size".to_owned(),
            )
        })?;
        if !TRANSFORMER_WINDOW_SIZES.contains(&window) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_5: transformer window {window} is outside the published domain {TRANSFORMER_WINDOW_SIZES:?}"
            )));
        }
        let component = memory.transformer_window_component.ok_or_else(|| {
            gen_core::Error::Unsupported(
                "ltx_2_5: stream_transformer_blocks requires transformer_window_component"
                    .to_owned(),
            )
        })?;
        if component != TRANSFORMER_WINDOW_COMPONENT {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_5: transformer streaming is declared for {TRANSFORMER_WINDOW_COMPONENT:?}, got {component:?}"
            )));
        }
        Some(window)
    } else {
        if memory.transformer_window_size.is_some() || memory.transformer_window_component.is_some()
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_5: transformer window parameters require stream_transformer_blocks=true"
                    .to_owned(),
            ));
        }
        None
    };

    Ok(TransformerExecution {
        stream_blocks: memory.stream_transformer_blocks,
        window_size,
        attention_chunk_size,
    })
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode) {
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

fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let fixture = contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION_FINGERPRINT);
    let expected = if fixture {
        weights_free_memory_strategy_contract(spec)
    } else {
        memory_strategy_contract(spec)
    };
    match expected {
        Ok(expected) if expected == *contract => match resolved_numeric_tier(spec) {
            Ok(tier) => safety_check(contract, tier, context),
            Err(error) => MemorySafetyDecision::Reject {
                reason: error.to_string(),
            },
        },
        Ok(_) => MemorySafetyDecision::Reject {
            reason: "ltx_2_5: caller contract differs from the exact registered load contract"
                .to_owned(),
        },
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

fn begin_with_cleanup(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, loaded_tier, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        LTX_2_5_MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        48,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.load_shape = context.load_shape;
    config.default_frames = context.geometry.frames;
    if contract.engages_selection(&context.selection, MemoryStrategy::BoundedAttention) {
        config.attention_chunk_size = context.selection.parameters.attention_chunk_size;
    }
    if contract.engages_selection(
        &context.selection,
        MemoryStrategy::BoundedTransformerResidency,
    ) {
        config.transformer_window = context.selection.parameters.transformer_window_size;
    }
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        contract,
        loaded_tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        contract,
        resolved_numeric_tier(spec)?,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

fn registered_valid_fixtures(
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
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        resolved_numeric_tier(spec)?,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".to_owned()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 768;
    context.geometry.height = 512;
    context.geometry.frames = 153;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request = GenerationRequest {
        prompt: "weights-free LTX-2.5 memory behavior".to_owned(),
        width: 768,
        height: 512,
        frames: Some(153),
        ..Default::default()
    };
    Ok(vec![fixture.with_load_spec(spec.clone())])
}

/// The production LTX-2.5 MLX memory registration.
pub const LTX_2_5_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: LTX_2_5_MODEL_ID,
        contract: memory_strategy_contract,
        safety_check: registered_safety_check,
    };

pub const LTX_2_5_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: LTX_2_5_MODEL_ID,
        contract: weights_free_memory_strategy_contract,
        surface_specs: gen_core::mlx_memory_contract_surface_specs,
    };

pub const LTX_2_5_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: LTX_2_5_MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::OffloadPolicy;
    use std::path::Path;

    fn spec(load_shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-2-5".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(load_shape)
    }

    fn support(contract: &MemoryProviderContract, rung: MemoryStrategy) -> MemoryStrategySupport {
        contract
            .capability(rung)
            .expect("every rung declared")
            .support
            .clone()
    }

    fn write_one_tensor(path: &Path) {
        let mut header = serde_json::to_vec(&serde_json::json!({
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        }))
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(0_f32.to_le_bytes());
        std::fs::write(path, file).unwrap();
    }

    /// AC (SC-22662): the LTX-2.5 route publishes the same axes as the 2.3 route — the measured
    /// delta between them is two booleans, not a dimension — and passes the shared facts check.
    #[test]
    fn architecture_facts_match_the_shared_ltx_derivation() {
        let contract =
            weights_free_memory_strategy_contract(&spec(LoadShape::EagerMaterialization)).unwrap();
        assert_eq!(
            contract.architecture_facts,
            mlx_gen::gen_core::MemoryArchitectureFacts {
                attention_heads: Some(32),
                head_dim: Some(128),
                transformer_blocks: Some(48),
                // The AvDiT has no patchify: `patchify_proj` is a plain Linear over the
                // 128-channel latent token, and every patch factor lives inside the VAE.
                patch_size: None,
                latent_channels: Some(128),
                vae_spatial_scale: Some(32),
                // A video autoencoder: eight frames per latent unit.
                vae_temporal_scale: Some(8),
                activation_dtype_width: Some(2),
            },
            "ltx_2_5 architecture facts"
        );
        assert_eq!(
            contract.architecture_facts,
            crate::memory_strategy::architecture_facts()
        );
        assert!(contract.architecture_facts.has_snapshot_read_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
    }

    #[test]
    fn production_contract_prices_every_loaded_component() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        for name in [
            "transformer.safetensors",
            "connector.safetensors",
            "text_encoder.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "duration.safetensors",
            "spatial.safetensors",
            "temporal.safetensors",
        ] {
            write_one_tensor(&root.join(name));
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_component(
                LtxComponent::Transformer.id(),
                WeightsSource::File(root.join("transformer.safetensors")),
            )
            .with_component(
                LtxComponent::TextEncoder.id(),
                WeightsSource::File(root.join("text_encoder.safetensors")),
            )
            .with_component(
                LtxComponent::ConvVideoVae.id(),
                WeightsSource::File(root.join("vae_decoder.safetensors")),
            )
            .with_component(
                LtxComponent::AudioVae.id(),
                WeightsSource::File(root.join("audio_vae.safetensors")),
            )
            .with_component(
                LtxComponent::DurationHead.id(),
                WeightsSource::File(root.join("duration.safetensors")),
            )
            .with_component(
                LtxComponent::SpatialUpsampler.id(),
                WeightsSource::File(root.join("spatial.safetensors")),
            )
            .with_component(
                LtxComponent::TemporalUpsampler.id(),
                WeightsSource::File(root.join("temporal.safetensors")),
            );
        let contract = memory_strategy_contract(&spec).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 16);
        assert_eq!(contract.asset_facts.transformer_bytes, 12);
        assert_eq!(contract.asset_facts.decoder_bytes, 8);
        assert_eq!(contract.asset_facts.base_bytes, 36);
        assert_eq!(contract.asset_facts.overlay_bytes, 0);

        std::fs::create_dir_all(root.join("enhancer")).unwrap();
        write_one_tensor(&root.join("enhancer/model.safetensors"));
        let enhanced_contract = memory_strategy_contract(&spec).unwrap();
        assert_eq!(enhanced_contract.asset_facts.conditioning_bytes, 18);
        assert_eq!(enhanced_contract.asset_facts.base_bytes, 38);

        write_one_tensor(&root.join("adapter.safetensors"));
        let adapter_bytes = std::fs::metadata(root.join("adapter.safetensors"))
            .unwrap()
            .len();
        let mut adapted = spec;
        adapted.adapters.push(mlx_gen::AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        let adapted_contract = memory_strategy_contract(&adapted).unwrap();
        assert_eq!(adapted_contract.asset_facts.base_bytes, 38);
        assert_eq!(adapted_contract.asset_facts.overlay_bytes, adapter_bytes);
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = &adapted_contract.formula
        else {
            panic!("an additive adapter must switch to a component-aware formula")
        };
        assert_eq!(resident_components.len(), 1);
        assert_eq!(resident_components[0].resident_bytes, adapter_bytes);
    }

    #[test]
    fn diffusion_decoder_does_not_claim_or_accept_conv_tiling() {
        let spec = spec(LoadShape::EagerMaterialization).with_component(
            LtxComponent::DiffusionVideoVae.id(),
            WeightsSource::File("/nonexistent/diffusion-vae.safetensors".into()),
        );
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.decode_tiling);
        assert!(validate_request_memory(
            &contract,
            &GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(decode_tile_edges()[0]),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn direct_request_controls_fail_closed_against_loaded_contract() {
        let contract =
            weights_free_memory_strategy_contract(&spec(LoadShape::DeferredMaterialization))
                .unwrap();
        let valid = GenerationMemory {
            stage_residency: true,
            chunk_attention: true,
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            stream_transformer_blocks: true,
            transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
            ..Default::default()
        };
        let execution = validate_request_memory(&contract, &valid).unwrap();
        assert_eq!(execution.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
        assert_eq!(execution.window_size, Some(TRANSFORMER_WINDOW_SIZE));

        let mut missing_parameter = valid;
        missing_parameter.attention_chunk_size = None;
        assert!(validate_request_memory(&contract, &missing_parameter).is_err());

        let stray_parameter = GenerationMemory {
            transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
            ..Default::default()
        };
        assert!(validate_request_memory(&contract, &stray_parameter).is_err());

        let eager =
            weights_free_memory_strategy_contract(&spec(LoadShape::EagerMaterialization)).unwrap();
        assert!(validate_request_memory(&eager, &valid).is_err());
    }

    /// The registration must name the 2.5 id, not 2.3's. A copy-paste that left `crate::MODEL_ID` in
    /// place would register a *duplicate* `ltx_2_3` strategy, which `build()` rejects — but only
    /// once a generator exists, so the failure would surface in sc-18778 rather than here.
    #[test]
    fn the_registration_names_the_two_five_engine_id() {
        assert_eq!(LTX_2_5_MEMORY_REGISTRATION.provider_id, "ltx_2_5");
        assert_ne!(LTX_2_5_MEMORY_REGISTRATION.provider_id, crate::MODEL_ID);
        let contract =
            weights_free_memory_strategy_contract(&spec(LoadShape::DeferredMaterialization))
                .unwrap();
        assert_eq!(contract.provider_id, "ltx_2_5");
    }

    /// R9's core clause, on the axis this story owns: rung 4 is declared **only** where the loader
    /// can execute it. An eager load must not advertise it.
    #[test]
    fn rung_four_is_declared_only_for_a_deferred_load() {
        let deferred =
            weights_free_memory_strategy_contract(&spec(LoadShape::DeferredMaterialization))
                .unwrap();
        assert_eq!(
            support(&deferred, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Implemented
        );
        assert!(deferred.lifecycle.transformer_window_materialization);

        let eager =
            weights_free_memory_strategy_contract(&spec(LoadShape::EagerMaterialization)).unwrap();
        assert_eq!(
            support(&eager, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing,
            "an eager load has no block window to bound"
        );
        assert!(
            !eager.lifecycle.transformer_window_materialization,
            "the lifecycle capability must track the support value, not be pinned true"
        );
    }

    /// An adapted load cannot stream — `LtxBlockStream::new` refuses it — so the contract must not
    /// declare the rung for one. Declaration and loader must give the same answer.
    #[test]
    fn an_adapted_load_does_not_declare_rung_four() {
        let mut adapted = spec(LoadShape::DeferredMaterialization);
        adapted.adapters = vec![mlx_gen::AdapterSpec {
            path: "/nonexistent/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let contract = weights_free_memory_strategy_contract(&adapted).unwrap();
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    /// Rung 3 is unconditional: the budget threads through `Attention` on every load shape, so it is
    /// declared on both. This is the arm that would fail if rung 3 were copy-pasted onto rung 4's
    /// `streamable` gate.
    #[test]
    fn rung_three_is_declared_on_every_load_shape() {
        for shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            let contract = weights_free_memory_strategy_contract(&spec(shape)).unwrap();
            assert_eq!(
                support(&contract, MemoryStrategy::BoundedAttention),
                MemoryStrategySupport::Implemented,
                "rung 3 is a scratch bound, independent of materialization shape ({shape:?})"
            );
            assert!(contract.lifecycle.attention_chunking);
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                ATTENTION_CHUNK_SIZES,
            );
        }
    }

    #[test]
    fn bounded_attention_compositions_layer_on_the_unconditional_staging_floor() {
        let conv_spec = spec(LoadShape::EagerMaterialization);
        let conv_contract = weights_free_memory_strategy_contract(&conv_spec).unwrap();
        let conv_selection =
            registered_valid_fixtures(&conv_spec, &conv_contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .pop()
                .unwrap()
                .context
                .selection;
        assert_eq!(
            conv_contract.engaged_composition_for_selection(&conv_selection),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ]
        );

        let diffvae_spec = spec(LoadShape::EagerMaterialization).with_component(
            LtxComponent::DiffusionVideoVae.id(),
            WeightsSource::File("/nonexistent/diffusion-vae.safetensors".into()),
        );
        let diffvae_contract = weights_free_memory_strategy_contract(&diffvae_spec).unwrap();
        let diffvae_selection = registered_valid_fixtures(
            &diffvae_spec,
            &diffvae_contract,
            MemoryStrategy::BoundedAttention,
        )
        .unwrap()
        .pop()
        .unwrap()
        .context
        .selection;
        assert_eq!(
            diffvae_contract.engaged_composition_for_selection(&diffvae_selection),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedAttention,
            ]
        );
    }

    /// `DeferredMaterialization` supplies the re-openable source, but it is not itself an
    /// execution policy. Under `Resident`, re-materializing the trunk changes neither the peak nor
    /// output, so the rung must remain unavailable.
    #[test]
    fn resident_policy_does_not_declare_rung_four() {
        let mut resident = spec(LoadShape::DeferredMaterialization);
        resident.offload_policy = OffloadPolicy::Resident;
        let contract = weights_free_memory_strategy_contract(&resident).unwrap();
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    /// The story's "gate streaming to `Sequential`" decision, asserted as the contract edge that
    /// implements it. Without this prerequisite a rung-4 selection under a resident phase policy is
    /// admitted and buys nothing.
    #[test]
    fn rung_four_requires_staged_residency_in_the_same_request() {
        let contract =
            weights_free_memory_strategy_contract(&spec(LoadShape::DeferredMaterialization))
                .unwrap();
        assert!(
            contract.additional_prerequisites.contains(&(
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )),
            "rung 4 under a resident phase policy bounds nothing and costs ~2x; the contract must \
             say so"
        );
        // And the shared graph must NOT have been edited to say it universally.
        assert!(
            !MemoryStrategy::BoundedTransformerResidency
                .requires()
                .iter()
                .any(|prerequisite| matches!(
                    prerequisite,
                    MemoryStrategyPrerequisite::Rung {
                        rung: MemoryStrategy::StagedResidency,
                        ..
                    }
                )),
            "this is a provider-specific edge; SC-15998 removed it from the shared graph"
        );
    }

    /// The published window domain must be a real domain: every candidate distinguishable in the
    /// bounded quantity, every candidate exactly executable over 48 blocks, and none of them the
    /// degenerate all-covering window.
    #[test]
    fn published_window_candidates_are_distinguishable_and_exact() {
        let n_blocks = crate::config::LtxConfig::video_only_defaults().num_layers as usize;
        assert_eq!(n_blocks, 48);
        for pair in TRANSFORMER_WINDOW_SIZES.windows(2) {
            assert_eq!(
                pair[1],
                pair[0] * 2,
                "consecutive candidates must differ by 2x in peak block residency, which is linear \
                 in window size — anything closer records sweep noise"
            );
        }
        for &window in TRANSFORMER_WINDOW_SIZES {
            let plan = gen_core::block_window::BlockPlan::new(n_blocks, window as usize).unwrap();
            assert!(
                plan.is_bounded(),
                "window {window} must bound something; an all-covering window pays the machinery \
                 for zero saving"
            );
            assert_eq!(
                plan.window_count(),
                n_blocks / window as usize,
                "window {window} must divide 48 exactly, or the sweep's independent variable is \
                 inexact"
            );
        }
        assert!(TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE));
        assert!(ATTENTION_CHUNK_SIZES.contains(&ATTENTION_CHUNK_SIZE));
    }

    /// The contract must satisfy gen-core's own conformance rules on every load shape it accepts.
    /// An error here is contract-level — a caller that bails loses rungs 0-4, not just the offender.
    #[test]
    fn the_contract_conforms_on_every_accepted_load_shape() {
        for shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            let contract = weights_free_memory_strategy_contract(&spec(shape)).unwrap();
            let errors = contract.conformance_errors();
            assert!(
                errors.is_empty(),
                "{shape:?} contract is non-conforming: {errors:?}"
            );
        }
    }

    /// The AV component-scope decision, pinned so a later edit to `Both` has to argue with it.
    #[test]
    fn the_declared_window_scope_is_the_dit_alone() {
        assert_eq!(TRANSFORMER_WINDOW_COMPONENT, TransformerComponent::Dit);
        assert!(TRANSFORMER_WINDOW_COMPONENT.includes_dit());
        assert!(
            !TRANSFORMER_WINDOW_COMPONENT.includes_text_encoder(),
            "the Gemma-4 encoder is a separate component that block_stream does not window; \
             claiming it would be an unreachable declaration"
        );
        let contract =
            weights_free_memory_strategy_contract(&spec(LoadShape::DeferredMaterialization))
                .unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }

    #[test]
    fn registered_scope_installs_the_selected_attention_and_window_controls() {
        let spec = spec(LoadShape::DeferredMaterialization).with_quant(gen_core::Quant::Q8);
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();

        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let fixture = registered_valid_fixtures(&spec, &contract, strategy)
                .unwrap()
                .into_iter()
                .next()
                .expect("implemented rung must expose a behavior fixture");
            assert!(matches!(
                registered_safety_check(&spec, &contract, &fixture.context),
                MemorySafetyDecision::Accept
            ));
            let mut request = fixture.request;
            let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .expect("MLX memory rung must open a request scope");
            scope.configure_request(&mut request).unwrap();
            let memory = request.memory.expect("scope installs generation memory");
            match strategy {
                MemoryStrategy::BoundedAttention => {
                    assert!(memory.chunk_attention);
                    assert!(ATTENTION_CHUNK_SIZES
                        .contains(&memory.attention_chunk_size.expect("selected chunk")));
                }
                MemoryStrategy::BoundedTransformerResidency => {
                    assert!(memory.stream_transformer_blocks);
                    assert!(TRANSFORMER_WINDOW_SIZES
                        .contains(&memory.transformer_window_size.expect("selected window")));
                    assert_eq!(
                        memory.transformer_window_component,
                        Some(TRANSFORMER_WINDOW_COMPONENT)
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn registered_safety_rejects_a_contract_not_built_for_the_load() {
        let spec = spec(LoadShape::DeferredMaterialization);
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let mut forged = contract.clone();
        forged.provider_id = crate::MODEL_ID.to_owned();
        assert!(matches!(
            registered_safety_check(&spec, &forged, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
