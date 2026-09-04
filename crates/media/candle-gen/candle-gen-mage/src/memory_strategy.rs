//! Shared Candle/CUDA image-memory ladder for the six Mage-Flow routes (SC-15813).
//!
//! Provider mechanics are shared, while the calibration identity remains route-local: an Edit
//! measurement must never authorize the text-to-image route (or a sibling checkpoint) merely because
//! the architecture and implementation are shared.

use crate::config;
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
    MemoryWindowMaterialization, Precision, PrecisionFloorComponent, Quant,
    SafetensorsTensorHeader, TransformerComponent, WeightsSource,
};
use candle_gen::quant::PackedConfig;
use std::sync::{Arc, Mutex};

pub const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub const TRANSFORMER_BLOCKS: u32 = config::DEPTH as u32;

const PROVIDER_IDS: &[&str] = &[
    config::MODEL_ID,
    config::BASE_MODEL_ID,
    config::TURBO_MODEL_ID,
    config::EDIT_MODEL_ID,
    config::EDIT_BASE_MODEL_ID,
    config::EDIT_TURBO_MODEL_ID,
];

fn is_edit(provider_id: &str) -> bool {
    matches!(
        provider_id,
        config::EDIT_MODEL_ID | config::EDIT_BASE_MODEL_ID | config::EDIT_TURBO_MODEL_ID
    )
}

fn fingerprint(provider_id: &str) -> gen_core::Result<String> {
    if PROVIDER_IDS.contains(&provider_id) {
        Ok(format!(
            "mage-flow-cuda-shared-ladder-provider-abi-v2-{}",
            provider_id.replace('_', "-")
        ))
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "unknown Mage-Flow memory provider {provider_id}"
        )))
    }
}

fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
}

fn transformer_has_device_format(spec: &LoadSpec) -> gen_core::Result<bool> {
    if resolved_quant(spec)?.is_none() {
        return Ok(true);
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let path = root.join("transformer").join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(gen_core::Error::Msg(format!(
                "mage-flow: read {} while resolving streamed weight format: {error}",
                path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        gen_core::Error::Msg(format!(
            "mage-flow: parse {} while resolving streamed weight format: {error}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&config).is_some())
}

/// Activation dtype every Mage-Flow component computes in — the DiT, the CoD decoder (`vae.rs`)
/// and the text encoder (`text_encoder.rs`) are all materialized bf16, so this is the provider's
/// real activation width rather than a memory-model literal.
///
/// Under `MemoryArchitectureFacts::activation_dtype_width`'s SC-22667 definition this is the
/// **denoise** width, and Mage is the easy case: every component opens at the same dtype, so the
/// per-phase distinction the contract draws does not split this provider.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;

/// Bytes per element of [`ACTIVATION_DTYPE`] — the width `Weights::from_dir` opens **every** Mage
/// component at (`text_encoder.rs`, `transformer.rs` and `vae.rs` all pass `DType::BF16`).
///
/// Two leaf families leave that width *after* the open and are priced at [`F32_WIDTH`] instead
/// (SC-22667 review): the language stack's norms, which `candle_gen_boogu::text_encoder` reads
/// through `get_f32` ([`TEXT_ENCODER_F32_LEAVES`]), and the `.bias` of any projection a request-time
/// fold quantizes — `QLinear::fold` promotes the bias to f32 for the post-matmul add.
const COMPONENT_WIDTH: u64 = 2;

/// Bytes per element of the leaves the loaders hold f32 (see [`COMPONENT_WIDTH`]).
const F32_WIDTH: u64 = 4;

/// The language-stack leaves `candle_gen_boogu::text_encoder` loads through `Weights::get_f32`
/// regardless of the store dtype (sc-12828): the per-layer RMSNorm weights, the per-head q/k norms
/// and the final norm. Everything else in the stack opens at [`COMPONENT_WIDTH`] (or folds).
const TEXT_ENCODER_F32_LEAVES: [&str; 5] = [
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    "language_model.norm.weight",
];

/// The Qwen3-VL vision tower inside the shared text-encoder directory.
///
/// `MageTextEncoder::load_inner` builds it only when `multimodal` is set, which
/// `MagePipeline::load_components` and `load_text` pass as `false`: a generation route materializes
/// the language stack alone and never touches these tensors.
const VISION_TOWER_PREFIX: &str = "model.visual.";

/// The CoD autoencoder's **encoder** half inside the shared VAE directory (`MageVaeEncoder::load`'s
/// own `PREFIX`). `MageVae::load` — the generation-route constructor — passes `with_encoder = false`
/// and leaves every one of these tensors on disk; only `load_full`, which the Edit provider calls,
/// materializes them.
const VAE_ENCODER_PREFIX: &str = "student.dconv_encoder.";

/// The language stack's quantized projections (`candle_gen_boogu::text_encoder`): the four attention
/// projections and the three MLP projections of each decoder layer, and nothing else — the token
/// embedding stays a dense `QEmbedding` and the norms load f32.
const TEXT_ENCODER_PACKED_LEAVES: [&str; 7] = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
];

/// The DiT projection whose resident tier is floored by
/// [`PrecisionFloorComponent::TransformerHead`] — `MageTransformer::place` folds this one at
/// `component_quant(TransformerHead, selected)` while every other projection takes `selected`.
const TRANSFORMER_HEAD_KEY: &str = "norm_out.linear.weight";

/// Headers of one component directory, or none when the component is not on disk.
///
/// An absent component contributing `0` is the behaviour
/// [`gen_core::safetensors_path_bytes`] already had here, and several admission fixtures depend on
/// it; gating on that sum keeps it while still failing closed on a malformed shard.
fn component_headers(path: &std::path::Path) -> gen_core::Result<Vec<SafetensorsTensorHeader>> {
    if gen_core::safetensors_path_bytes(path) == 0 {
        return Ok(Vec::new());
    }
    gen_core::safetensors_path_tensor_headers(path)
}

/// Resident bytes of one projection folded to a GGUF block tier.
///
/// `crate::quant::quantize_onto` routes every fold through `QLinear::quantize_dequant_onto`, whose
/// block type is `candle_gen::quant::ggml_dtype(quant)` — `Q4_0` (18 B per 32 values) or `Q8_0`
/// (34 B per 32 values). The dense `[out, in]` weight is gone once the fold completes; the bias,
/// which the fold promotes to f32, is not.
fn folded_projection_bytes(
    header: &SafetensorsTensorHeader,
    quant: Quant,
) -> gen_core::Result<u64> {
    let dtype = candle_gen::quant::ggml_dtype(quant)
        .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?;
    let block = dtype.block_size() as u64;
    let elements = header
        .element_count()
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
    if block == 0 || !elements.is_multiple_of(block) {
        return Err(gen_core::Error::Unsupported(format!(
            "mage-flow: projection {} has {elements} elements, not a whole number of {block}-wide \
             {dtype:?} blocks",
            header.name
        )));
    }
    (elements / block)
        .checked_mul(dtype.type_size() as u64)
        .ok_or_else(|| gen_core::Error::Msg("mage-flow: folded projection bytes overflow".into()))
}

/// The MLX affine group size a physically packed component declares in its `config.json`
/// `quantization` block, or `None` for a dense component (no config, or no block).
///
/// This is the same `PackedConfig` `Weights::packed()` hands `crate::quant::linear`, read the same
/// way `transformer_has_device_format` reads it; a present-but-malformed file is an error there and
/// an error here.
fn packed_group_size(dir: &std::path::Path) -> gen_core::Result<Option<usize>> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(gen_core::Error::Msg(format!(
                "mage-flow: read {} while pricing a packed component: {error}",
                path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        gen_core::Error::Msg(format!(
            "mage-flow: parse {} while pricing a packed component: {error}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&config).map(|config| config.group_size as usize))
}

/// Load-exact bytes of a component whose linears fold to a GGUF tier at load.
///
/// `keep` selects the tensors this route materializes at all, `folds` names the ones the loader
/// quantizes and at which tier (`None` leaves the tensor dense at [`COMPONENT_WIDTH`]), and
/// `f32_leaf` names the leaves the loader holds f32 whatever the store width.
///
/// A **physically packed** projection — a `.weight` with `.scales` and `.biases` siblings — is
/// priced as what `crate::quant::linear` builds from it (SC-22667 review): `QLinear::from_packed_gs`
/// hands the MLX affine triple to `repack_packed_weight`, which lands a GGUF `Q4_1` (20 B per 32
/// values) or `Q8_0` (34 B per 32 values) tensor on the device and drops all three source planes.
/// The resident form is therefore neither the `U32` code plane at its stored width nor the
/// `scales`/`biases` sidecars, and `candle_gen::quant::mlx_packed_qtensor_resident_bytes` is the
/// shared pricing for exactly that conversion; the sidecars are priced with their weight, never on
/// their own. The `.bias` such a projection may carry stays at [`COMPONENT_WIDTH`]: `QLinear::fold`
/// returns early on an already-quantized base, so the request-time promotion below never runs.
///
/// A projection the request-time fold quantizes (`folds` is `Some`) loses its dense weight to the
/// GGUF block tensor and keeps its `.bias` **promoted to f32** — `QLinear::fold` moves the bias with
/// the weight and casts it for the post-matmul add — so that `.bias` is priced at [`F32_WIDTH`].
fn component_bytes(
    dir: &std::path::Path,
    keep: impl Fn(&str) -> bool,
    folds: impl Fn(&SafetensorsTensorHeader) -> Option<Quant>,
    f32_leaf: impl Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    let headers = component_headers(dir)?;
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<std::collections::BTreeMap<_, _>>();
    let packed_triple =
        |base: &str| -> Option<(&SafetensorsTensorHeader, &SafetensorsTensorHeader)> {
            Some((
                *by_name.get(format!("{base}.scales").as_str())?,
                *by_name.get(format!("{base}.biases").as_str())?,
            ))
        };
    // `.weight` headers whose stored form is a packed triple, keyed by their base name.
    let is_packed_weight = |base: &str| -> bool {
        by_name
            .get(format!("{base}.weight").as_str())
            .is_some_and(|weight| weight.dtype == gen_core::weightsmeta::Dtype::U32)
            && packed_triple(base).is_some()
    };
    let group = packed_group_size(dir)?;
    let overflow = || gen_core::Error::Msg("mage-flow: component byte overflow".into());
    let mut dense = Vec::new();
    let mut f32_leaves = Vec::new();
    let mut total = 0_u64;
    for header in &headers {
        if !keep(&header.name) {
            continue;
        }
        // The sidecar planes of a packed triple are consumed by the repack: priced with the weight.
        if let Some(base) = header
            .name
            .strip_suffix(".scales")
            .or_else(|| header.name.strip_suffix(".biases"))
        {
            if is_packed_weight(base) {
                continue;
            }
        }
        if let Some(base) = header.name.strip_suffix(".weight") {
            if let (Some((scales, biases)), true) = (packed_triple(base), is_packed_weight(base)) {
                let group = group.ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "mage-flow: {} carries a packed affine triple but {} declares no \
                         `quantization` group size",
                        header.name,
                        dir.join("config.json").display()
                    ))
                })?;
                total = total
                    .checked_add(candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                        header, scales, biases, group,
                    )?)
                    .ok_or_else(overflow)?;
                continue;
            }
        }
        if let Some(quant) = folds(header) {
            total = total
                .checked_add(folded_projection_bytes(header, quant)?)
                .ok_or_else(overflow)?;
            continue;
        }
        // A `.bias` whose sibling `.weight` the fold quantizes is promoted to f32 by that fold.
        let promoted_bias = header.name.strip_suffix(".bias").is_some_and(|base| {
            by_name
                .get(format!("{base}.weight").as_str())
                .is_some_and(|weight| keep(&weight.name) && folds(weight).is_some())
        });
        if promoted_bias || f32_leaf(&header.name) {
            f32_leaves.push(header.clone());
        } else {
            dense.push(header.clone());
        }
    }
    total
        .checked_add(gen_core::materialized_header_bytes(
            &dense,
            COMPONENT_WIDTH,
            dir,
        )?)
        .ok_or_else(overflow)?
        .checked_add(gen_core::materialized_header_bytes(
            &f32_leaves,
            F32_WIDTH,
            dir,
        )?)
        .ok_or_else(overflow)
}

/// The bytes each Mage component **materializes** for `provider_id`'s route (epic SC-22657, E1;
/// feature-end sweep SC-22667).
///
/// This is deliberately not `crate::component_footprint`, which is the registry's on-disk size seam
/// and is documented as an upper bound. Three things separate the two, and each of them moved a
/// contract's numbers:
///
/// * **The generation routes do not load the whole text-encoder or VAE directory.** The Qwen3-VL
///   vision tower ([`VISION_TOWER_PREFIX`]) and the CoD encoder ([`VAE_ENCODER_PREFIX`]) are Edit-only,
///   and an on-disk sum charged every text-to-image render for both.
/// * **A quantized load never holds the dense projections.** `MageTransformer::load_with_quant`
///   stages on CPU and folds each of the 174 live projections onto the device as a `Q4_0`/`Q8_0`
///   tensor, and `MageTextEncoder::load_inner` folds the language stack's seven per-layer
///   projections at the [`PrecisionFloorComponent::TextEncoder`] floor. Charging the bf16 source
///   over-declared the two largest components by roughly the whole quantization win.
/// * **A physical q4/q8 tier is already packed**, and `crate::quant::linear` repacks each affine
///   triple into a device-format GGUF tensor (`Q4_1` / `Q8_0`) that is neither the `U32` code plane
///   at its stored width nor the `scales`/`biases` sidecars — see [`component_bytes`].
///
/// Two smaller leaf families are priced f32 rather than at [`COMPONENT_WIDTH`] because that is
/// how their loaders hold them: the language stack's norms ([`TEXT_ENCODER_F32_LEAVES`], read via
/// `get_f32`) and the `.bias` of every projection a request-time fold quantizes (`QLinear::fold`
/// promotes it).
pub(crate) fn loaded_component_bytes(
    provider_id: &str,
    spec: &LoadSpec,
    dirs: &crate::MageComponentDirs,
) -> gen_core::Result<gen_core::PerComponentBytes> {
    let multimodal = is_edit(provider_id);
    let quant = resolved_quant(spec)?;
    let text_encoder_quant = quant.map(|selected| {
        crate::quant::component_quant(PrecisionFloorComponent::TextEncoder, selected)
    });
    let head_quant = quant.map(|selected| {
        crate::quant::component_quant(PrecisionFloorComponent::TransformerHead, selected)
    });
    Ok(gen_core::PerComponentBytes {
        text_encoder: component_bytes(
            &dirs.text_encoder,
            |name| multimodal || !name.starts_with(VISION_TOWER_PREFIX),
            |header| {
                text_encoder_quant.filter(|_| {
                    TEXT_ENCODER_PACKED_LEAVES
                        .iter()
                        .any(|leaf| header.name.ends_with(leaf))
                        && header.is_float()
                })
            },
            |name| {
                !name.starts_with(VISION_TOWER_PREFIX)
                    && TEXT_ENCODER_F32_LEAVES
                        .iter()
                        .any(|leaf| name.ends_with(leaf))
            },
        )?,
        dit: component_bytes(
            &dirs.transformer,
            |_| true,
            |header| {
                // Every live projection is a rank-2 float `.weight`; the norms this DiT carries are
                // rank-1 tensors and it has no convolutions at all, so the shape is the predicate.
                let projection = header.name.ends_with(".weight")
                    && header.shape.len() == 2
                    && header.is_float();
                if !projection {
                    return None;
                }
                if header.name.ends_with(TRANSFORMER_HEAD_KEY) {
                    head_quant
                } else {
                    quant
                }
            },
            // The DiT's norms are read at the store width; only a folded projection's bias leaves
            // it, and `component_bytes` prices that from the fold itself.
            |_| false,
        )?,
        vae: component_bytes(
            &dirs.vae,
            |name| multimodal || !name.starts_with(VAE_ENCODER_PREFIX),
            |_| None,
            |_| false,
        )?,
    })
}

/// Snapshot-read architecture axes for the six Mage-Flow routes (epic SC-22657, E2).
///
/// The DiT axes are read from the exact file the loader parses: `pipeline::load_components` reads
/// `<transformer dir>/config.json` and hands it to [`config::MageConfig::from_json`], whose keys
/// are `num_heads`, `depth` and `patch_size`. Reading those same keys here publishes what the
/// pipeline will construct; nothing is inferred from the model id.
///
/// `head_dim` comes from the same source. Mage's config carries no head-dim key, but it does carry
/// `hidden_size` — [`config::MageConfig::from_json`] reads it — so the honest per-head width is the
/// quotient `hidden_size / num_heads`, which on the reference snapshot is 3072 / 24 = the crate
/// constant `config::HEAD_DIM`. Publishing that constant unconditionally while the heads and
/// blocks beside it came from the config would make a snapshot with a different `hidden_size`
/// report a width its own trunk does not have. `af::head_dim` declines a quotient that does not
/// divide evenly rather than rounding one, so a non-uniform snapshot publishes no width at all.
///
/// **There is no preset fallback for a config the loader refuses** (SC-22667, the
/// `MemoryArchitectureFacts` preset-fallback rule, and its review round). `MageConfig::from_json`
/// reads every geometry key through its required-integer accessor and **errors** on a partial file,
/// then `validate`s the parsed geometry against the frozen RL geometry and errors on drift. A config
/// it refuses has **no loadable geometry at all**, so the whole DiT axis block is gated on that same
/// parse: the four trunk axes are read off the parsed [`config::MageConfig`] when it loads, and
/// every one of them is `None` when it does not. Publishing `num_heads` or `depth` off a JSON the
/// loader rejects — as the previous `head_dim`-only gate did — described a render that cannot
/// happen, which is exactly the shape the rule forbids.
///
/// The decoder axes are the crate constants [`config::LATENT_CHANNELS`] and
/// [`config::VAE_DOWNSAMPLE`]: Mage's CoD decoder is built from code, not from a `vae/config.json`,
/// so they are preset axes unconditionally and are published whether or not the DiT config loads.
///
/// A weights-free contract (the registry's sentinel surface path, or a single-file import)
/// publishes `MemoryArchitectureFacts::default()`.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let Some(root) = af::snapshot_root(spec) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    // The exact reader the pipeline uses, over the exact file it reads. `Err` — a missing file, a
    // partial config, a drifted geometry — means the loader would refuse this snapshot, and a trunk
    // that never constructs has no axes to declare.
    let dit = std::fs::read_to_string(root.join("transformer").join("config.json"))
        .ok()
        .and_then(|text| config::MageConfig::from_json(&text).ok());
    let (attention_heads, head_dim, transformer_blocks, patch_size) = match &dit {
        Some(cfg) => {
            let heads = af::declared(cfg.num_heads);
            (
                heads,
                // `hidden_size / num_heads` from the parsed config — `af::head_dim` declines a
                // quotient that does not divide rather than rounding one.
                af::head_dim(af::declared(cfg.hidden_size), heads),
                af::declared(cfg.depth),
                af::declared(cfg.patch_size),
            )
        }
        None => (None, None, None, None),
    };
    gen_core::MemoryArchitectureFacts {
        attention_heads,
        head_dim,
        transformer_blocks,
        patch_size,
        latent_channels: af::declared(config::LATENT_CHANNELS),
        vae_spatial_scale: af::declared(config::VAE_DOWNSAMPLE),
        // Structurally absent: the CoD decoder is a still-image decoder with no temporal axis, so
        // there is no frames-per-latent scale to declare (absent is `None`, never `Some(0)`).
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

pub fn provider_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_with_dirs(provider_id, spec, &crate::resolved_component_dirs(spec)?)
}

/// The production contract for an already-resolved component layout — the entry point the loader
/// itself uses, so the generator and the registry price the identical directories.
pub(crate) fn provider_contract_with_dirs(
    provider_id: &str,
    spec: &LoadSpec,
    dirs: &crate::MageComponentDirs,
) -> gen_core::Result<MemoryProviderContract> {
    let components = loaded_component_bytes(provider_id, spec, dirs)?;
    provider_contract_with_components(provider_id, spec, components)
}

pub(crate) fn provider_contract_with_components(
    provider_id: &str,
    spec: &LoadSpec,
    components: gen_core::PerComponentBytes,
) -> gen_core::Result<MemoryProviderContract> {
    let calibration_fingerprint = fingerprint(provider_id)?;
    let streamable = streamable(spec) && transformer_has_device_format(spec)?;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                // Mage's CoD decoder normalizes over the complete latent feature field before its
                // pixel MLP. Spatial tiles would change that normalization and therefore the image;
                // a full-edge call is ordinary decode, not a bounded-memory implementation.
                MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: "Mage CoD decode contains full-frame normalization and has no parity-safe independent spatial tiles".to_owned(),
                    }
                }
                MemoryStrategy::BoundedTransformerResidency if !streamable => {
                    MemoryStrategySupport::Missing
                }
                _ => MemoryStrategySupport::Implemented,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                        transformer_window_components: vec![TransformerComponent::Dit],
                        ..Default::default()
                    }
                }
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();

    Ok(MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components: Vec::new(),
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        // One network, one field (epic SC-22657, E1; feature-end ruling SC-22667). The Edit
        // routes encode their references through the same CoD autoencoder weights that decode, so
        // the VAE is resident during `Conditioning` as well as `Decode` there — but its bytes are
        // charged exactly once, in `decoder_bytes`. Folding them into `conditioning_bytes` too
        // (the previous shape) made `base_bytes` no longer its own decomposition on every Edit
        // route. The contract has no per-phase residency declaration for a base component, so
        // that co-residency is stated here rather than in the byte fields.
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: 0,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

macro_rules! contract_fn {
    ($name:ident, $id:expr) => {
        pub fn $name(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
            provider_contract_for($id, spec)
        }
    };
}

contract_fn!(contract_rl, config::MODEL_ID);
contract_fn!(contract_base, config::BASE_MODEL_ID);
contract_fn!(contract_turbo, config::TURBO_MODEL_ID);
contract_fn!(contract_edit, config::EDIT_MODEL_ID);
contract_fn!(contract_edit_base, config::EDIT_BASE_MODEL_ID);
contract_fn!(contract_edit_turbo, config::EDIT_TURBO_MODEL_ID);

pub fn resolved_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    match spec.quantize {
        Some(Quant::Q4) => Ok(Some(Quant::Q4)),
        Some(Quant::Q8) => Ok(Some(Quant::Q8)),
        None => Ok(None),
        Some(other) => Err(gen_core::Error::Unsupported(format!(
            "Mage-Flow does not support the {other:?} numeric tier"
        ))),
    }
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let quant = resolved_quant(spec)?;
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: crate::quant::active_component_precision_floors(quant),
    })
}

fn route_is_supported(provider_id: &str, context: &MemoryRunContext) -> bool {
    if context.overlay.is_some() || context.use_pid || context.has_phases {
        return false;
    }
    if is_edit(provider_id) {
        context.mode == MemoryMode::Edit
            && (1..=8).contains(&context.geometry.reference_count)
            && context.has_reference
    } else {
        context.mode == MemoryMode::TextToImage
            && context.geometry.reference_count == 0
            && !context.has_reference
    }
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: loaded_quant,
            component_precision_floors: crate::quant::active_component_precision_floors(
                loaded_quant,
            ),
        }),
        None,
    ) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if !route_is_supported(&contract.provider_id, context) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: unsupported memory route mode={} references={} overlay={:?}",
            contract.provider_id,
            context.mode.as_key(),
            context.geometry.reference_count,
            context.overlay
        )));
    }
    if context.geometry.batch != 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: memory calibration is single-image only",
            contract.provider_id
        )));
    }
    Ok(())
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_quant(spec).and_then(|quant| validate_context(contract, context, quant)) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestBinding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        Self {
            address: std::ptr::from_ref(request).addr(),
            geometry: MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(1),
                reference_count: request.image_reference_count(),
            },
            memory: request.memory,
            use_pid: request.use_pid,
            has_phases: request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty()),
        }
    }
}

struct ActiveAdmission {
    token: u64,
    context: MemoryRunContext,
    expected_memory: Option<GenerationMemory>,
    binding: Option<RequestBinding>,
    consumed: bool,
}

#[derive(Default)]
struct AdmissionState {
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<ActiveAdmission>,
}

#[derive(Clone)]
pub(crate) struct AdmissionRegistry {
    provider_id: &'static str,
    inner: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub(crate) fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request is active",
                self.provider_id
            )));
        }
        state.approved_context = Some(context.clone());
        Ok(())
    }

    pub(crate) fn clear_approval(&self) {
        candle_gen::lock_recover(&self.inner).approved_context = None;
    }

    fn begin(
        &self,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        expected_memory: Option<GenerationMemory>,
    ) -> gen_core::Result<u64> {
        if contract.provider_id != self.provider_id {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory contract belongs to {}",
                self.provider_id, contract.provider_id
            )));
        }
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request scope is active",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request skipped the safety handshake",
                self.provider_id
            ))
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory context changed after safety approval",
                self.provider_id
            )));
        }
        state.next_token = state.next_token.wrapping_add(1).max(1);
        let token = state.next_token;
        state.active = Some(ActiveAdmission {
            token,
            context: context.clone(),
            expected_memory,
            binding: None,
            consumed: false,
        });
        Ok(token)
    }

    fn configure(&self, token: u64, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let active = state.active.as_mut().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request scope is no longer active",
                self.provider_id
            ))
        })?;
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed memory request",
                self.provider_id
            )));
        }
        active.binding = Some(binding);
        Ok(())
    }

    pub(crate) fn consume_for_generate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let constrained = request
            .memory
            .is_some_and(|memory| memory != GenerationMemory::default());
        let Some(active) = state.active.as_mut() else {
            return if constrained {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: constrained request has no active admission",
                    self.provider_id
                )))
            } else {
                Ok(())
            };
        };
        if active.binding.as_ref() != Some(&RequestBinding::from_request(request))
            || active.consumed
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed or admission was already consumed",
                self.provider_id
            )));
        }
        active.consumed = true;
        Ok(())
    }

    fn finish(&self, token: u64) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: stale memory token cannot finish",
                self.provider_id
            )))
        }
    }

    fn abandon(&self, token: u64) {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
        }
    }
}

pub struct MageMemoryScope {
    device: Device,
    provider_id: String,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
    attention_chunk_sizes: Vec<u32>,
    transformer_window: Option<u32>,
    admission: Option<AdmissionRegistry>,
    token: Option<u64>,
    finished: bool,
}

impl MageMemoryScope {
    pub fn new(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> Self {
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .expect("Mage contract publishes bounded attention");
        Self {
            device,
            provider_id: contract.provider_id.clone(),
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            use_pid: context.use_pid,
            has_phases: context.has_phases,
            attention_chunk_sizes: attention.parameters.attention_chunk_sizes.clone(),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            admission: None,
            token: None,
            finished: false,
        }
    }

    pub(crate) fn new_bound(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(
            contract,
            context,
            contract.generation_memory(&context.selection),
        )?;
        let mut scope = Self::new(device, contract, context);
        scope.admission = Some(admission);
        scope.token = Some(token);
        Ok(scope)
    }

    fn active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(format!(
                "{}: memory request scope is finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }
}

impl MemoryRequestScope for MageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.active()?;
        let geometry = MemoryGeometry {
            width: request.width,
            height: request.height,
            batch: request.count,
            frames: request.frames.unwrap_or(1),
            reference_count: request.image_reference_count(),
        };
        if geometry != self.geometry {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request geometry changed after admission",
                self.provider_id
            )));
        }
        let has_phases = request
            .phases
            .as_ref()
            .is_some_and(|phases| !phases.is_empty());
        if request.use_pid != self.use_pid || has_phases != self.has_phases {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: forbidden request axes changed after admission",
                self.provider_id
            )));
        }
        request.memory = self.memory;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.configure(token, request)?;
        }
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.active()?;
        let _ = (tile_edge, overlap, geometry);
        Err(gen_core::Error::Unsupported(format!(
            "{}: bounded decode is structurally unavailable for the full-frame CoD decoder",
            self.provider_id
        )))
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.active()?;
        if self.attention_chunk_sizes.contains(&chunk_size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk {chunk_size} is not admitted",
                self.provider_id
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.active()?;
        let Some(window) = self.transformer_window else {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer streaming was not selected",
                self.provider_id
            )));
        };
        if first_block >= TRANSFORMER_BLOCKS
            || block_count == 0
            || !first_block.is_multiple_of(window)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        let expected = window.min(TRANSFORMER_BLOCKS - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: expected {expected} blocks at {first_block}, got {block_count}",
                self.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.active()?;
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.finish(token)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl Drop for MageMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            if let (Some(admission), Some(token)) = (&self.admission, self.token) {
                admission.abandon(token);
            }
            self.finished = true;
        }
    }
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let route = if is_edit(&contract.provider_id) {
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    } else {
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    };
    Ok(vec![gen_core::MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            resolved_numeric_tier(spec)?,
            route,
        )?,
    )])
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(contract, context, resolved_quant(spec)?)?;
    Ok(Some(Box::new(MageMemoryScope::new(
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal header-only bf16 safetensors file. The payload is zero-filled: every assertion here
    /// is over tensor *geometry*, which lives in the header, and nothing reads a value.
    fn write_bf16(path: &std::path::Path, tensors: &[(&str, &[usize])]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for &(name, shape) in tensors {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(vec![0_u8; offset as usize]);
        std::fs::write(path, bytes).unwrap();
    }

    /// A snapshot carrying the Edit-only tensors alongside the ones every route loads: the Qwen3-VL
    /// vision tower inside `text_encoder/`, and the CoD encoder inside `vae/`.
    fn multimodal_snapshot(tmp: &tempfile::TempDir) -> LoadSpec {
        let spec = architecture_spec(tmp, 12);
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        let root = root.clone();
        write_bf16(
            &root.join("text_encoder/model.safetensors"),
            &[
                ("model.language_model.embed_tokens.weight", &[64, 8]),
                ("model.language_model.norm.weight", &[8]),
                ("model.visual.blocks.0.attn.qkv.weight", &[32, 8]),
            ],
        );
        write_bf16(
            &root.join("vae/model.safetensors"),
            &[
                ("student.decoder.proj_out.weight", &[16, 8]),
                ("student.dconv_encoder.proj_out.weight", &[24, 8]),
            ],
        );
        write_bf16(
            &root.join("transformer/model.safetensors"),
            &[("img_in.weight", &[64, 64])],
        );
        spec
    }

    /// Feature-end review (SC-22667, E1). `MagePipeline::load_components` builds the text encoder
    /// with `multimodal = false` and the autoencoder through `MageVae::load` (`with_encoder =
    /// false`), so a generation route never materializes the Qwen3-VL vision tower or the CoD
    /// encoder — and must not charge either. The Edit provider passes `true` / `load_full` and does.
    ///
    /// Mutation that fails this: dropping the `multimodal ||` guards from `loaded_component_bytes`
    /// so both components keep the whole directory — the generation route then reads 1_568 B of
    /// conditioning and 640 B of decoder, the Edit figures, on every text-to-image render.
    #[test]
    fn generation_routes_do_not_charge_the_edit_only_vision_tower_or_vae_encoder() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = multimodal_snapshot(&tmp);
        // language stack: the 64*8 embedding bf16 -> 1_024 B plus the final norm, 8 elements the
        // loader holds f32 -> 32 B; vision tower 32*8 -> 512 B.
        // decoder half 16*8 -> 256 B; CoD encoder half 24*8 -> 384 B.
        let generation = contract_rl(&spec).unwrap();
        assert_eq!(generation.asset_facts.conditioning_bytes, 1_024 + 32);
        assert_eq!(generation.asset_facts.decoder_bytes, 256);

        let edit = contract_edit(&spec).unwrap();
        assert_eq!(edit.asset_facts.conditioning_bytes, 1_024 + 32 + 512);
        assert_eq!(edit.asset_facts.decoder_bytes, 256 + 384);

        for contract in [&generation, &edit] {
            gen_core_testkit::check_memory_contract_asset_facts(contract)
                .unwrap_or_else(|errors| panic!("{errors:?}"));
        }
        assert!(
            generation.asset_facts.base_bytes < edit.asset_facts.base_bytes,
            "the Edit route holds strictly more than the generation route"
        );
    }

    /// Feature-end review (SC-22667, E1). A quantized Mage load never holds the dense projections:
    /// `MageTransformer::load_with_quant` stages on CPU and folds all 174 live projections onto the
    /// device as GGUF blocks, with `norm_out.linear` floored to Q8 by
    /// [`PrecisionFloorComponent::TransformerHead`] even when Q4 is selected.
    ///
    /// Mutation that fails this: returning `None` from `loaded_component_bytes`'s DiT `folds`
    /// closure, which restores the dense 16_512 B figure for every tier.
    #[test]
    fn a_quantized_load_charges_the_folded_projections_not_the_dense_source() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = architecture_spec(&tmp, 12);
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        write_bf16(
            &root.join("transformer/model.safetensors"),
            &[
                ("img_in.weight", &[64, 64]),
                ("norm_out.linear.weight", &[64, 64]),
                ("txt_norm.weight", &[64]),
            ],
        );
        // 4_096-element projections: bf16 8_192 B, Q4_0 128 blocks * 18 B = 2_304 B, Q8_0
        // 128 * 34 = 4_352 B. The rank-1 norm is never folded: 64 * 2 = 128 B.
        for (quant, expected) in [
            (None, 8_192 + 8_192 + 128),
            // Q4 body + the Q8-floored head.
            (Some(Quant::Q4), 2_304 + 4_352 + 128),
            (Some(Quant::Q8), 4_352 + 4_352 + 128),
        ] {
            let mut spec = spec.clone();
            spec.quantize = quant;
            let contract = contract_rl(&spec).unwrap();
            assert_eq!(
                contract.asset_facts.transformer_bytes, expected,
                "{quant:?} transformer bytes"
            );
        }
    }

    /// Header-only safetensors with explicit dtypes, for the packed-tier fixture below.
    fn write_typed(path: &std::path::Path, tensors: &[(&str, &str, &[usize])]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for &(name, dtype, shape) in tensors {
            let width = match dtype {
                "BF16" => 2_u64,
                "F32" | "U32" => 4,
                other => panic!("unhandled fixture dtype {other}"),
            };
            let bytes = shape.iter().product::<usize>() as u64 * width;
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(vec![0_u8; offset as usize]);
        std::fs::write(path, bytes).unwrap();
    }

    /// Feature-end review (SC-22667). A **physically packed** q4/q8 tier is not priced from its
    /// on-disk MLX affine triple: `crate::quant::linear` hands the triple to
    /// `QLinear::from_packed_gs` → `repack_packed_weight`, which lands a GGUF `Q4_1` (20 B / 32
    /// values) or `Q8_0` (34 B / 32) tensor and drops the codes and both sidecars. The `.bias` of
    /// such a projection stays bf16 (`QLinear::fold` returns early on a quantized base).
    ///
    /// Mutation that fails this: pricing the triple through the dense arm (the previous shape) —
    /// `U32` codes at their stored 2_048 B plus `scales`/`biases` at 256 B, i.e. 2_304 B for a
    /// projection that lands as 2_560 B; and, separately, charging the `scales`/`biases` planes on
    /// top of the repacked tensor.
    #[test]
    fn a_physically_packed_tier_is_priced_as_its_repacked_gguf_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = architecture_spec(&tmp, 12);
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        // A q4 tier at MLX group 64: a `[64, 64]` projection stores as `U32 [64, 8]` codes plus
        // bf16 `[64, 1]` scales and biases.
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut config = config;
        config["quantization"] = serde_json::json!({"bits": 4, "group_size": 64});
        std::fs::write(
            root.join("transformer/config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        write_typed(
            &root.join("transformer/model.safetensors"),
            &[
                ("img_in.weight", "U32", &[64, 8]),
                ("img_in.scales", "BF16", &[64, 1]),
                ("img_in.biases", "BF16", &[64, 1]),
                ("img_in.bias", "BF16", &[64]),
                ("txt_norm.weight", "BF16", &[64]),
            ],
        );
        let headers = gen_core::safetensors_path_tensor_headers(root.join("transformer")).unwrap();
        let header = |name: &str| headers.iter().find(|h| h.name == name).unwrap();
        let repacked = candle_gen::quant::mlx_packed_qtensor_resident_bytes(
            header("img_in.weight"),
            header("img_in.scales"),
            header("img_in.biases"),
            64,
        )
        .unwrap();
        // 4_096 values as `Q4_1`: 128 blocks * 20 B.
        assert_eq!(repacked, 2_560);

        let mut spec = spec.clone();
        spec.quantize = Some(Quant::Q4);
        let contract = contract_rl(&spec).unwrap();
        assert_eq!(
            contract.asset_facts.transformer_bytes,
            // The repacked tensor, the bias at the store width (no fold ran), the norm.
            repacked + 64 * 2 + 64 * 2,
            "a packed tier is priced as the GGUF tensor the repack lands"
        );
    }

    /// Feature-end review (SC-22667). Two leaf families are held f32 after a bf16 open and were
    /// priced at 2 B: the `.bias` of a projection the request-time fold quantizes (`QLinear::fold`
    /// promotes it for the post-matmul add), and the language stack's norms, which
    /// `candle_gen_boogu::text_encoder` reads through `get_f32` (`input_layernorm`,
    /// `post_attention_layernorm`, `q_norm`, `k_norm`, the final `norm`).
    ///
    /// Mutation that fails this: pricing `.bias` and the norm leaves through the dense arm at
    /// `COMPONENT_WIDTH` — the DiT reads 2_304 + 128 (bias at 2 B) on Q4 and the text encoder
    /// reads 1_024 + 64 (norms at 2 B) instead of 1_024 + 128.
    #[test]
    fn folded_biases_and_text_encoder_norms_are_priced_f32() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = architecture_spec(&tmp, 12);
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        write_bf16(
            &root.join("transformer/model.safetensors"),
            &[("img_in.weight", &[64, 64]), ("img_in.bias", &[64])],
        );
        write_bf16(
            &root.join("text_encoder/model.safetensors"),
            &[
                ("model.language_model.embed_tokens.weight", &[64, 8]),
                ("model.language_model.layers.0.input_layernorm.weight", &[8]),
                (
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[8],
                ),
                (
                    "model.language_model.layers.0.self_attn.q_norm.weight",
                    &[4],
                ),
                (
                    "model.language_model.layers.0.self_attn.k_norm.weight",
                    &[4],
                ),
                ("model.language_model.norm.weight", &[8]),
                // The vision tower's norms are outside the language stack's f32 rule and outside
                // a generation route altogether.
                ("model.visual.blocks.0.norm1.weight", &[8]),
            ],
        );
        // Dense: the bias stays bf16 (128 B); the norms are f32 on every tier.
        let norms_f32 = (8 + 8 + 4 + 4 + 8) * 4;
        let dense = contract_rl(&spec).unwrap();
        assert_eq!(dense.asset_facts.transformer_bytes, 8_192 + 128);
        assert_eq!(
            dense.asset_facts.conditioning_bytes,
            64 * 8 * 2 + norms_f32,
            "the language norms load f32 under the bf16 store"
        );
        // Q4: the folded projection's bias is promoted to f32 (256 B).
        let mut q4 = spec.clone();
        q4.quantize = Some(Quant::Q4);
        let folded = contract_rl(&q4).unwrap();
        assert_eq!(folded.asset_facts.transformer_bytes, 2_304 + 256);
        assert_eq!(
            folded.asset_facts.conditioning_bytes,
            64 * 8 * 2 + norms_f32
        );
    }

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("sc15813-mage-packed-contract");
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    /// A snapshot whose `transformer/config.json` carries the published Mage-Flow axes the loader
    /// reads through [`config::MageConfig::from_json`]. `depth` is a parameter so a drifting
    /// snapshot can be exercised.
    fn architecture_spec(tmp: &tempfile::TempDir, depth: u64) -> LoadSpec {
        let root = tmp.path().join(format!("sc22661-mage-depth-{depth}"));
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "in_channels": 128,
                "out_channels": 128,
                "context_in_dim": 2560,
                "hidden_size": 3072,
                "num_heads": 24,
                "depth": depth,
                "axes_dim": [16, 56, 56],
                "checkpoint": false,
                "patch_size": 1,
            }))
            .unwrap(),
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root))
    }

    /// Feature-end review (SC-22667, E1): on the Edit routes the CoD autoencoder encodes the
    /// references AND decodes the result, but it is one network and is charged once, in
    /// `decoder_bytes`; `conditioning_bytes` is the text encoder alone and `base_bytes` is exactly
    /// the sum of the three fields. Nonzero component bytes are used on purpose: the zero-byte
    /// weights-free fixture satisfies `base == sum` for any decomposition.
    ///
    /// Mutation that fails this: `conditioning_bytes: text_encoder + vae` on the Edit routes (the
    /// shape under review) — conditioning reads 16 + 8 and `check_memory_contract_asset_facts`
    /// reports `base_bytes 36 != 12 + 24 + 8 = 44`.
    #[test]
    fn edit_routes_charge_the_shared_autoencoder_once_in_decoder_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let components = gen_core::PerComponentBytes {
            text_encoder: 16,
            dit: 12,
            vae: 8,
        };
        for id in PROVIDER_IDS {
            let contract =
                provider_contract_with_components(id, &architecture_spec(&tmp, 12), components)
                    .unwrap();
            assert_eq!(
                contract.asset_facts.conditioning_bytes, 16,
                "{id}: conditioning is the text encoder alone"
            );
            assert_eq!(contract.asset_facts.transformer_bytes, 12);
            assert_eq!(contract.asset_facts.decoder_bytes, 8);
            assert_eq!(contract.asset_facts.base_bytes, 16 + 12 + 8);
            gen_core_testkit::check_memory_contract_asset_facts(&contract)
                .unwrap_or_else(|errors| panic!("{id}: {errors:?}"));
        }
    }

    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let tmp = tempfile::tempdir().unwrap();
        let expected = gen_core::MemoryArchitectureFacts {
            attention_heads: Some(24),
            head_dim: Some(128),
            transformer_blocks: Some(12),
            patch_size: Some(1),
            latent_channels: Some(128),
            vae_spatial_scale: Some(16),
            // Structurally absent: Mage's CoD decoder is a still-image decoder, so there is no
            // frames-per-latent temporal axis to declare.
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };
        for id in PROVIDER_IDS {
            let contract = provider_contract_for(id, &architecture_spec(&tmp, 12)).unwrap();
            assert_eq!(contract.architecture_facts, expected, "{id}");
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }

        // SC-22667, the `MemoryArchitectureFacts` preset-fallback rule ("fall back to a preset only
        // where the LOADER itself falls back to that preset; otherwise declare the axis None"), as
        // applied in its review round: `MageConfig::from_json` is a reader that REQUIRES every key
        // and then `validate`s the geometry against the frozen RL trunk, so a config it refuses has
        // no loadable geometry at all — not a missing `head_dim` beside three honest axes, but no
        // trunk axis whatsoever. The WHOLE DiT block is gated on that parse; the decoder axes are
        // code-constructed presets and survive.
        //
        // Mutation that fails this: publishing `num_heads` / `depth` / `patch_size` off the JSON
        // keys while gating only `head_dim` (the previous shape), which reports `Some(24)`,
        // `Some(12)`, `Some(1)` beside a `None` width for a snapshot that never loads.
        let decoder_only = gen_core::MemoryArchitectureFacts {
            latent_channels: Some(128),
            vae_spatial_scale: Some(16),
            activation_dtype_width: Some(2),
            ..Default::default()
        };
        let refused = |name: &str, config: &[u8]| {
            let root = tmp.path().join(name);
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(root.join("transformer/config.json"), config).unwrap();
            assert!(
                config::MageConfig::from_json(std::str::from_utf8(config).unwrap()).is_err(),
                "{name}: the premise of the assertion: this config does not load"
            );
            provider_contract_for(config::MODEL_ID, &LoadSpec::new(WeightsSource::Dir(root)))
                .unwrap()
                .architecture_facts
        };
        // A partial file: every required key but `hidden_size` (and the rest) is missing.
        assert_eq!(
            refused(
                "sc22661-mage-no-hidden-size",
                br#"{"num_heads": 24, "depth": 12, "patch_size": 1}"#
            ),
            decoder_only
        );
        // A drifted geometry the loader's `validate` refuses: sixteen blocks, sixteen heads.
        let drifted = architecture_spec(&tmp, 16);
        assert!(
            provider_contract_for(config::MODEL_ID, &drifted)
                .unwrap()
                .architecture_facts
                == decoder_only,
            "a depth the frozen geometry refuses publishes no trunk axis"
        );
        assert_eq!(
            refused(
                "sc22661-mage-16-heads",
                br#"{"hidden_size": 3072, "num_heads": 16, "depth": 12, "patch_size": 1}"#
            ),
            decoder_only
        );
        // A non-uniform pair: no honest width exists, and the load does not happen either.
        assert_eq!(
            refused(
                "sc22661-mage-ragged",
                br#"{"hidden_size": 3073, "num_heads": 24, "depth": 12, "patch_size": 1}"#
            ),
            decoder_only
        );

        // The registry's weights-free surface names a sentinel that is not on disk, so nothing
        // about the pipeline is resolved there and every axis stays undeclared.
        let weights_free = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert!(provider_contract_for(config::MODEL_ID, &weights_free)
            .unwrap()
            .architecture_facts
            .is_empty());
    }

    #[test]
    fn every_route_has_a_distinct_conformant_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fingerprints = std::collections::BTreeSet::new();
        for id in PROVIDER_IDS {
            let contract = provider_contract_for(id, &spec(&tmp)).unwrap();
            assert!(
                contract.conformance_errors().is_empty(),
                "{id}: {:?}",
                contract.conformance_errors()
            );
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert!(fingerprints.insert(contract.calibration.as_ref().unwrap().fingerprint.clone()));
            for strategy in MemoryStrategy::ALL {
                let support = &contract.capability(strategy).unwrap().support;
                if strategy == MemoryStrategy::BoundedDecode {
                    assert!(matches!(
                        support,
                        MemoryStrategySupport::StructurallyNotApplicable { .. }
                    ));
                } else {
                    assert_eq!(support, &MemoryStrategySupport::Implemented);
                }
            }
        }
    }

    #[test]
    fn candidate_domains_and_block_geometry_are_exact() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        assert_eq!(TRANSFORMER_BLOCKS, 12);
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert!(matches!(
            decode.support,
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert!(decode.parameters.decode_tile_edges.is_empty());
        assert!(decode.parameters.decode_overlaps.is_empty());
        assert!(!contract.lifecycle.decode_tiling);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn t2i_and_edit_contexts_do_not_cross_authorize() {
        let tmp = tempfile::tempdir().unwrap();
        let t2i = contract_rl(&spec(&tmp)).unwrap();
        let edit = contract_edit(&spec(&tmp)).unwrap();
        let t2i_context =
            registered_valid_fixture(&spec(&tmp), &t2i, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        let edit_context =
            registered_valid_fixture(&spec(&tmp), &edit, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        assert!(validate_context(&t2i, &t2i_context, Some(Quant::Q4)).is_ok());
        assert!(validate_context(&edit, &edit_context, Some(Quant::Q4)).is_ok());
        assert!(validate_context(&t2i, &edit_context, Some(Quant::Q4)).is_err());
        assert!(validate_context(&edit, &t2i_context, Some(Quant::Q4)).is_err());
    }

    #[test]
    fn registered_receipts_bind_only_floors_active_for_the_loaded_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let base_spec = spec(&tmp);
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut selected_spec = base_spec.clone();
            selected_spec.quantize = quant;
            let contract = contract_rl(&selected_spec).unwrap();
            let tier = resolved_numeric_tier(&selected_spec).unwrap();
            assert_eq!(
                tier.component_precision_floors,
                crate::quant::active_component_precision_floors(quant),
                "resolved receipt must be tier-exact for {quant:?}"
            );
            let context = registered_valid_fixture(
                &selected_spec,
                &contract,
                MemoryStrategy::StagedResidency,
            )
            .unwrap()
            .remove(0)
            .context;
            assert!(validate_context(&contract, &context, quant).is_ok());

            if quant != Some(Quant::Q4) {
                let mut over_bound = context;
                over_bound.selection.tier.component_precision_floors =
                    crate::quant::COMPONENT_PRECISION_FLOORS;
                let error = validate_context(&contract, &over_bound, quant).unwrap_err();
                assert!(error.to_string().contains("does not match loaded tier"));
            } else {
                let mut under_bound = context;
                under_bound.selection.tier.component_precision_floors = &[];
                let error = validate_context(&contract, &under_bound, quant).unwrap_err();
                assert!(error.to_string().contains("does not match loaded tier"));
            }
        }
    }

    #[test]
    fn stale_fingerprint_and_resident_streaming_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut eager = spec(&tmp);
        eager.load_shape = LoadShape::EagerMaterialization;
        let contract = contract_rl(&eager).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        let contract = contract_rl(&spec(&tmp)).unwrap();
        let mut context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .remove(0)
                .context;
        context.calibration_fingerprint.push_str(":stale");
        assert!(matches!(
            registered_safety_check(&spec(&tmp), &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn dense_quantized_directory_does_not_advertise_device_format_streaming() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        let mut dense = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        dense.load_shape = LoadShape::DeferredMaterialization;

        let contract = contract_rl(&dense).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn later_rungs_do_not_engage_structurally_unavailable_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        assert!(!contract.engages(
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedDecode
        ));
        let context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .remove(0)
                .context;
        assert!(
            !contract
                .generation_memory(&context.selection)
                .unwrap()
                .tile_vae_decode
        );
        let mut scope = MageMemoryScope::new(Device::Cpu, &contract, &context);
        assert!(scope.configure_decode(1024, 1, context.geometry).is_err());
    }

    #[test]
    fn request_binding_rejects_pid_and_phase_mutation_before_or_after_configuration() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        let context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;

        let assert_mutation_rejected = |mutate: fn(&mut GenerationRequest)| {
            let admission = AdmissionRegistry::new(config::MODEL_ID);
            admission.approve(&context).unwrap();
            let mut scope =
                MageMemoryScope::new_bound(Device::Cpu, &contract, &context, admission.clone())
                    .unwrap();
            let mut request = GenerationRequest {
                prompt: "fixture".to_owned(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            mutate(&mut request);
            assert!(scope.configure_request(&mut request).is_err());

            let admission = AdmissionRegistry::new(config::MODEL_ID);
            admission.approve(&context).unwrap();
            let mut scope =
                MageMemoryScope::new_bound(Device::Cpu, &contract, &context, admission.clone())
                    .unwrap();
            let mut request = GenerationRequest {
                prompt: "fixture".to_owned(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            scope.configure_request(&mut request).unwrap();
            mutate(&mut request);
            assert!(admission.consume_for_generate(&request).is_err());
        };

        assert_mutation_rejected(|request| request.use_pid = true);
        assert_mutation_rejected(|request| {
            request.phases = Some(vec![gen_core::GenerationPhase {
                steps: 1,
                ..Default::default()
            }]);
        });
    }
}
