//! Load-exact Candle Wan TI2V-5B video memory-strategy contract (SC-19223).
//!
//! This route is intentionally narrow. It covers the real text-to-video path at 24 fps with no
//! conditioning, auxiliary components, or adapters; unsupported generator surfaces continue to
//! load without a contract so caller-side admission fails open. The loaded generator owns the
//! production contract and the registry resolves the same header-derived declaration. A separate
//! zero-byte identity exists only for weights-free catalog behavior checks.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::weightsmeta::Dtype;
use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryWindowMaterialization, OffloadPolicy, Precision, Quant, ResidentRequestMemory,
    WeightsSource,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryMode};
use candle_gen::{CandleError, Result as CandleResult};

use crate::config::{
    TransformerConfig, DEFAULT_FPS, DEFAULT_FRAMES, MAX_AREA_5B, MIN_SIZE, MODEL_ID, SIZE_MULTIPLE,
};
use crate::MAX_WAN_FRAMES;

pub const CALIBRATION_FINGERPRINT_RESIDENT: &str =
    "sc-19223-wan2-2-ti2v-5b-candle-resident-load-v1";
pub const CALIBRATION_FINGERPRINT_SEQUENTIAL: &str =
    "sc-19223-wan2-2-ti2v-5b-candle-sequential-load-v1";
#[cfg(any(feature = "cuda", test))]
const STATIC_CALIBRATION_FINGERPRINT_RESIDENT: &str =
    "sc-19223-wan2-2-ti2v-5b-candle-registry-resident-v1";
#[cfg(any(feature = "cuda", test))]
const STATIC_CALIBRATION_FINGERPRINT_SEQUENTIAL: &str =
    "sc-19223-wan2-2-ti2v-5b-candle-registry-sequential-v1";

pub const DECODE_OVERLAP: u32 = 64;
/// Every production candidate is below Wan's 480-pixel minimum side, so selecting rung 2 always
/// exercises the spatial tiler instead of silently falling through to the automatic single pass.
pub const DECODE_TILE_EDGES: &[u32] = &[448, 384, 320, 256, 192];
const PACK_GROUP_SIZE: usize = crate::candle_tier_build::TIER_GROUP_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NumericAssets {
    facts: MemoryAssetFacts,
    tier: MemoryNumericTier,
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> gen_core::Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {label} resident byte total overflows u64"
            ))
        })
    })
}

fn tensor_elements(
    component: &str,
    tensor: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> gen_core::Result<u64> {
    let elements = tensor.shape.iter().try_fold(1_u64, |total, &dimension| {
        let dimension = u64::try_from(dimension).map_err(|_| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {component} tensor {} has an unrepresentable dimension",
                tensor.name
            ))
        })?;
        total.checked_mul(dimension).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {component} tensor {} element count overflows u64",
                tensor.name
            ))
        })
    })?;
    let stored = elements
        .checked_mul(tensor.dtype.size() as u64)
        .ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {component} tensor {} stored byte count overflows u64",
                tensor.name
            ))
        })?;
    if stored != tensor.data_bytes {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: {component} tensor {} declares {} bytes but its dtype/shape require {stored}",
            tensor.name, tensor.data_bytes
        )));
    }
    Ok(elements)
}

fn component_headers(
    path: &Path,
    component: &str,
) -> gen_core::Result<Vec<gen_core::weightsmeta::SafetensorsTensorHeader>> {
    // Mirror `component_vb` exactly: only direct, non-hidden children selected by the shared
    // lexical shard resolver participate, and later shards win duplicate tensor keys. The generic
    // weightsmeta directory helper is intentionally recursive and therefore is not load-exact for
    // this provider.
    let files = candle_gen::sorted_safetensors(path, component).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: {component} is not a complete safetensors component at {}: {error}",
            path.display()
        ))
    })?;
    let mut by_name = BTreeMap::new();
    for file in files {
        if !std::fs::metadata(&file)
            .map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: {component} shard {} is unreadable: {error}",
                    file.display()
                ))
            })?
            .is_file()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {component} shard {} is not a file",
                file.display()
            )));
        }
        let headers =
            gen_core::weightsmeta::safetensors_path_tensor_headers(&file).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: {component} shard {} is not header-readable: {error}",
                    file.display()
                ))
            })?;
        for tensor in headers {
            by_name.insert(tensor.name.clone(), tensor);
        }
    }
    let headers = by_name.into_values().collect::<Vec<_>>();
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: {component} has no tensors at {}",
            path.display()
        )));
    }
    for tensor in &headers {
        tensor_elements(component, tensor)?;
    }
    Ok(headers)
}

fn projected_dense_component_bytes(
    path: &Path,
    component: &str,
    float_bytes: u64,
) -> gen_core::Result<u64> {
    let headers = component_headers(path, component)?;
    if let Some(tensor) = headers.iter().find(|tensor| {
        is_packed_leaf(&tensor.name)
            || (tensor.dtype == Dtype::U32 && tensor.name.ends_with(".weight"))
    }) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated {component} is dense, but {} is an affine-packed tensor leaf",
            tensor.name
        )));
    }
    headers.iter().try_fold(0_u64, |total, tensor| {
        // FP8 posture: `is_float` excludes `F8_E4M3`, so a scaled-fp8 component is refused here
        // rather than priced at `float_bytes`. That is deliberate — the calibrated route is the
        // snapshot directory (`validate_contract_route`), whose loader casts float storage to the
        // compute dtype; the ComfyUI scaled-fp8 seam in `crate::comfyui` dequantizes through its own
        // `.scale_weight` companions and is a different, uncalibrated route.
        if !tensor.is_float() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated dense {component} tensor {} must use a floating source dtype; the loader would cast {:?} to its resident compute dtype",
                tensor.name, tensor.dtype
            )));
        }
        let bytes = tensor_elements(component, tensor)?
            .checked_mul(float_bytes)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: {component} tensor {} projected byte count overflows u64",
                    tensor.name
                ))
            })?;
        total.checked_add(bytes).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {component} projected resident byte total overflows u64"
            ))
        })
    })
}

fn packed_marker(transformer: &Path) -> gen_core::Result<Option<Quant>> {
    let path = transformer.join("quantize_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: invalid transformer quantize_config.json at {}: {error}",
                path.display()
            ))
        })?;
    let bits = json.get("bits").and_then(serde_json::Value::as_u64);
    let group = json
        .get("quantization")
        .and_then(|value| value.get("group_size"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(PACK_GROUP_SIZE as u64);
    if group != PACK_GROUP_SIZE as u64 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: transformer packed marker declares group size {group}; the Wan loader consumes group {PACK_GROUP_SIZE}"
        )));
    }
    match bits {
        Some(4) => Ok(Some(Quant::Q4)),
        Some(8) => Ok(Some(Quant::Q8)),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: transformer packed marker must declare top-level bits 4 or 8"
        ))),
    }
}

fn is_packed_leaf(name: &str) -> bool {
    name.ends_with(".scales") || name.ends_with(".biases")
}

fn validate_packed_triple(
    base: &str,
    weight: &gen_core::weightsmeta::SafetensorsTensorHeader,
    scales: &gen_core::weightsmeta::SafetensorsTensorHeader,
    biases: &gen_core::weightsmeta::SafetensorsTensorHeader,
    quant: Quant,
) -> gen_core::Result<u64> {
    if weight.dtype != Dtype::U32 || weight.shape.len() != 2 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base}.weight must be rank-2 U32 codes"
        )));
    }
    // FP8 posture: an affine-packed grid's scales/biases are read as real floats, never as fp8
    // codes, so `is_float`'s exclusion of `F8_E4M3` is the intended refusal here.
    if !scales.is_float()
        || !biases.is_float()
        || scales.shape.len() != 2
        || biases.shape.len() != 2
        || scales.shape != biases.shape
        || scales.shape.first() != weight.shape.first()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} scales/biases must be matching rank-2 floating grids with the code-row count"
        )));
    }
    let input = scales.shape[1]
        .checked_mul(PACK_GROUP_SIZE)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: packed transformer {base} logical input overflows usize"
            ))
        })?;
    let packed_width = weight.shape[1].checked_mul(32).ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} code width overflows usize"
        ))
    })?;
    if input == 0 || !packed_width.is_multiple_of(input) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} has inconsistent code/scales geometry"
        )));
    }
    let packed_bits = packed_width.checked_div(input).ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} has inconsistent code/scales geometry"
        ))
    })?;
    if packed_bits != quant.bits() as usize {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} encodes Q{packed_bits}, but the sidecar declares Q{}",
            quant.bits()
        )));
    }
    let elements = u64::try_from(weight.shape[0])
        .ok()
        .and_then(|rows| {
            u64::try_from(input)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: packed transformer {base} logical element count overflows u64"
            ))
        })?;
    if !elements.is_multiple_of(32) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: packed transformer {base} logical element count {elements} is not GGML block aligned"
        )));
    }
    // The real loader converts the affine source triple into resident GGML Q4_1 (20 bytes/32
    // elements) or Q8_0 (34 bytes/32 elements). The source codes/scales/biases are temporary and are
    // therefore not counted a second time.
    let block_bytes = match quant {
        Quant::Q4 => 20_u64,
        Quant::Q8 => 34_u64,
        Quant::Nvfp4 => {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: NVFP4 is not a Wan affine-packed tier"
            )))
        }
    };
    (elements / 32).checked_mul(block_bytes).ok_or_else(|| {
        gen_core::Error::Msg(format!(
            "{MODEL_ID}: packed transformer {base} resident byte count overflows u64"
        ))
    })
}

fn transformer_assets(root: &Path) -> gen_core::Result<(u64, Option<Quant>)> {
    let path = root.join("transformer");
    let headers = component_headers(&path, "transformer")?;
    let marker = packed_marker(&path)?;
    let tensors = headers
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<BTreeMap<_, _>>();

    let Some(quant) = marker else {
        if tensors.values().any(|tensor| {
            is_packed_leaf(&tensor.name)
                || (tensor.name.ends_with(".weight") && tensor.dtype == Dtype::U32)
        }) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: dense transformer contains affine-packed leaves without a valid quantize_config.json"
            )));
        }
        let bytes = tensors.values().try_fold(0_u64, |total, tensor| {
            // FP8 posture: `is_float` excludes `F8_E4M3`, so a scaled-fp8 DiT is refused instead of
            // being priced at the bf16 width its `.scale_weight` companions would only reach through
            // the separate `crate::comfyui` dequant seam.
            if !tensor.is_float() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: calibrated dense transformer tensor {} must use a floating source dtype; the loader would cast {:?} to bf16",
                    tensor.name, tensor.dtype
                )));
            }
            let bytes = tensor_elements("transformer", tensor)?
                .checked_mul(2)
                .ok_or_else(|| {
                    gen_core::Error::Msg(format!(
                        "{MODEL_ID}: transformer projected byte count overflows u64"
                    ))
                })?;
            total.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: transformer resident byte total overflows u64"
                ))
            })
        })?;
        return Ok((bytes, None));
    };

    for tensor in tensors
        .values()
        .filter(|tensor| is_packed_leaf(&tensor.name))
    {
        let suffix = if tensor.name.ends_with(".scales") {
            ".scales"
        } else {
            ".biases"
        };
        let base = tensor.name.strip_suffix(suffix).expect("suffix checked");
        if !tensors.contains_key(&format!("{base}.weight")) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: transformer has orphan packed leaf {}",
                tensor.name
            )));
        }
    }

    let mut packed_bases = BTreeSet::new();
    let mut packed_bytes = 0_u64;
    for weight in tensors
        .values()
        .filter(|tensor| tensor.name.ends_with(".weight") && tensor.shape.len() == 2)
    {
        let base = weight.name.strip_suffix(".weight").expect("suffix checked");
        let scales = tensors.get(&format!("{base}.scales"));
        let biases = tensors.get(&format!("{base}.biases"));
        let (Some(scales), Some(biases)) = (scales, biases) else {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: packed transformer has an unpacked or partial rank-2 weight {base}"
            )));
        };
        packed_bytes = packed_bytes
            .checked_add(validate_packed_triple(base, weight, scales, biases, quant)?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: packed transformer resident byte total overflows u64"
                ))
            })?;
        packed_bases.insert(base.to_owned());
    }
    if packed_bases.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: transformer quantization marker has no packed rank-2 weights"
        )));
    }

    let dense_bytes = tensors.values().try_fold(0_u64, |total, tensor| {
        if is_packed_leaf(&tensor.name) {
            return Ok(total);
        }
        if let Some(base) = tensor.name.strip_suffix(".weight") {
            if packed_bases.contains(base) {
                return Ok(total);
            }
        }
        if tensor.dtype == Dtype::U32 && tensor.name.ends_with(".weight") {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: transformer contains unaccounted packed weight {}",
                tensor.name
            )));
        }
        // FP8 posture: `is_float` excludes `F8_E4M3`, so the only non-float storage a packed
        // transformer may retain is the U32 code grid accounted above — a stray fp8 leaf is refused.
        if !tensor.is_float() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: packed transformer ordinary tensor {} must use a floating source dtype; only an exact packed triple may retain U32 codes",
                tensor.name
            )));
        }
        let bytes = tensor_elements("transformer", tensor)?
            .checked_mul(2)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: transformer projected byte count overflows u64"
                ))
            })?;
        total.checked_add(bytes).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: transformer resident byte total overflows u64"
            ))
        })
    })?;
    Ok((
        checked_sum("transformer", [packed_bytes, dense_bytes])?,
        Some(quant),
    ))
}

fn validate_contract_route(spec: &LoadSpec) -> gen_core::Result<&Path> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated memory admission requires the snapshot directory used by the loader"
        )));
    };
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: the provider has no deferred block loader; calibrated admission requires EagerMaterialization"
        )));
    }
    if spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: the calibrated Candle route uses the provider-native bf16 tier"
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated memory admission covers the plain TI2V-5B route without controls, PiD, external components, or adapters"
        )));
    }
    Ok(root)
}

fn production_assets(
    spec: &LoadSpec,
    dit_source: &crate::DitSource,
) -> gen_core::Result<NumericAssets> {
    let root = validate_contract_route(spec)?;
    if let crate::DitSource::NativeGguf(path) = dit_source {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: the native-GGUF DiT source at {} has no calibrated memory contract",
            path.display()
        )));
    }
    let conditioning_bytes =
        projected_dense_component_bytes(&root.join("text_encoder"), "text_encoder", 2)?;
    let (transformer_bytes, quant) = transformer_assets(root)?;
    if spec.quantize != quant {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: requested tier {:?} does not match the header/sidecar-derived transformer tier {:?}",
            spec.quantize, quant
        )));
    }
    let decoder_bytes = projected_dense_component_bytes(&root.join("vae"), "vae", 4)?;
    let base_bytes = checked_sum(
        "base model",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    Ok(NumericAssets {
        facts: MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes: 0,
        },
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        },
    })
}

fn calibration_fingerprint(spec: &LoadSpec, fixture: bool) -> &'static str {
    match (fixture, spec.offload_policy) {
        (false, OffloadPolicy::Resident) => CALIBRATION_FINGERPRINT_RESIDENT,
        (false, OffloadPolicy::Sequential) => CALIBRATION_FINGERPRINT_SEQUENTIAL,
        #[cfg(any(feature = "cuda", test))]
        (true, OffloadPolicy::Resident) => STATIC_CALIBRATION_FINGERPRINT_RESIDENT,
        #[cfg(any(feature = "cuda", test))]
        (true, OffloadPolicy::Sequential) => STATIC_CALIBRATION_FINGERPRINT_SEQUENTIAL,
        #[cfg(not(any(feature = "cuda", test)))]
        (true, _) => unreachable!("fixture fingerprints are not built outside CUDA/tests"),
    }
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
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// Snapshot-gated architecture facts for the TI2V-5B route (epic SC-22657, E2).
///
/// This module's contract covers exactly one provider — `config::MODEL_ID`
/// (`wan2_2_ti2v_5b`) — so the geometry is read from the same Rust presets the loader instantiates
/// for it, not from a `config.json` the loader never opens. Wan's Candle loader does not parse the
/// diffusers `transformer/config.json` at all: it builds [`TransformerConfig::ti2v_5b`] and
/// [`crate::config::VaeConfig::ti2v_5b`] directly, and the A14B presets ([`TransformerConfig::t2v_14b`] /
/// [`TransformerConfig::i2v_14b`], `Vae16Config`) belong to the `wan14b` / `vace` generators, which
/// publish no [`MemoryProviderContract`] at all. Declaring the A14B geometry here would be a fact
/// about a route this contract never describes.
///
/// The VAE scales come from the route's own load-bearing geometry constant,
/// [`crate::Ti2vProviderVae::VAE_TILING`] (`VaeTiling::WAN22`, x16 spatial / x4 causal temporal) —
/// the same value the decoder plans its tiling from — rather than from a restated literal.
///
/// A weights-free contract (the registry's sentinel surface, which is not on disk) publishes
/// `MemoryArchitectureFacts::default()`: nothing about the pipeline that *would* load has been
/// resolved, so no axis is knowable.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    let dit = TransformerConfig::ti2v_5b();
    let vae = crate::config::VaeConfig::ti2v_5b();
    let tiling = crate::Ti2vProviderVae::VAE_TILING;
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(dit.num_heads),
        // `head_dim` is declared by the preset itself (`dim == num_heads * head_dim` = 3072), so it
        // is read rather than re-derived from the product.
        head_dim: af::declared(dit.head_dim),
        transformer_blocks: af::declared(dit.num_layers),
        // `patch` is `(p_t, p_h, p_w) = (1, 2, 2)`; the spatial entry is the axis this fact names.
        patch_size: af::declared(dit.patch.1),
        // The z48 VAE's own `z_dim` is the encoder's declaration of what it produces; the DiT's
        // `in_channels` is the consumer's view of the same 48 channels.
        latent_channels: af::declared(vae.z_dim),
        vae_spatial_scale: u32::try_from(tiling.spatial_scale)
            .ok()
            .filter(|scale| *scale != 0),
        vae_temporal_scale: u32::try_from(tiling.temporal_scale)
            .ok()
            .filter(|scale| *scale != 0),
        // The 5B DiT runs bf16 unconditionally (`lib.rs: DIT_DTYPE`), so this is the activation
        // width actually materialized rather than a memory-model literal.
        activation_dtype_width: af::dtype_width(crate::DIT_DTYPE),
    }
}

fn build_contract(
    spec: &LoadSpec,
    facts: MemoryAssetFacts,
    fingerprint: &str,
) -> gen_core::Result<MemoryProviderContract> {
    validate_contract_route(spec)?;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    Ok(MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        // Wan declares no decode-quality geometry policy table, so this route carries no semantic
        // decode authority — the fail-closed default every other candle provider contract uses.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        // Resident means "install no new request controls". This preserves a caller's stored
        // Resident/Sequential load policy; an explicit staged selection still requests Sequential.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::DecodeTileArea,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)),
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

fn production_contract_and_tier(
    spec: &LoadSpec,
    dit_source: &crate::DitSource,
) -> gen_core::Result<(MemoryProviderContract, MemoryNumericTier)> {
    let assets = production_assets(spec, dit_source)?;
    let contract = build_contract(spec, assets.facts, calibration_fingerprint(spec, false))?;
    Ok((contract, assets.tier))
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    production_contract_and_tier(spec, &crate::DitSource::from_environment())
        .map(|(contract, _)| contract)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
    dit_source: &crate::DitSource,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    // Unsupported or unprovable surfaces remain usable through the historical generator path but
    // intentionally expose no admission contract. This is the provider-side fail-open boundary.
    Ok(production_contract_and_tier(spec, dit_source).ok())
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        MemoryAssetFacts::default(),
        calibration_fingerprint(spec, true),
    )
}

#[cfg(any(feature = "cuda", test))]
fn fixture_contract(contract: &MemoryProviderContract) -> bool {
    contract.calibration.as_ref().is_some_and(|identity| {
        matches!(
            identity.fingerprint.as_str(),
            STATIC_CALIBRATION_FINGERPRINT_RESIDENT | STATIC_CALIBRATION_FINGERPRINT_SEQUENTIAL
        )
    })
}

#[cfg(any(feature = "cuda", test))]
fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    production_assets(spec, &crate::DitSource::from_environment()).map(|assets| assets.tier)
}

#[cfg(any(feature = "cuda", test))]
fn registered_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<MemoryNumericTier> {
    if fixture_contract(contract) {
        if spec.quantize == Some(Quant::Nvfp4) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: NVFP4 is not a Wan affine-packed tier"
            )));
        }
        return Ok(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: spec.quantize,
            component_precision_floors: &[],
        });
    }
    resolved_numeric_tier(spec)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires a tile-edge cap"
        ))
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode requires a tile overlap"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode tile-edge cap {edge} is outside the production domain {DECODE_TILE_EDGES:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode overlap {overlap} is not the production overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.mode.as_key() != "text_to_video"
            || context.geometry.reference_count != 0
            || context.has_reference
            || context.use_pid
            || context.has_phases
            || context.overlay.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route is plain, single-phase text_to_video without references, PiD, or overlays"
            )));
        }
        let geometry = context.geometry;
        let area = u64::from(geometry.width) * u64::from(geometry.height);
        if geometry.batch != 1
            || !(MIN_SIZE..=1280).contains(&geometry.width)
            || !(MIN_SIZE..=1280).contains(&geometry.height)
            || !geometry.width.is_multiple_of(SIZE_MULTIPLE)
            || !geometry.height.is_multiple_of(SIZE_MULTIPLE)
            || area > MAX_AREA_5B as u64
            || geometry.frames == 0
            || geometry.frames as usize > MAX_WAN_FRAMES
            || !((geometry.frames - 1).is_multiple_of(4))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: unsupported calibrated geometry {}x{}x{} frames={}",
                geometry.width, geometry.height, geometry.batch, geometry.frames
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
        Some(loaded_tier),
        Some(&route_gate),
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier(spec, contract) {
        Ok(tier) => safety_check(contract, tier, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[cfg(any(feature = "cuda", test))]
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
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        registered_tier(spec, contract)?,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".to_owned()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 832;
    context.geometry.height = 480;
    context.geometry.frames = DEFAULT_FRAMES;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free Wan memory behavior".to_owned();
    Ok(vec![fixture])
}

struct WanMemoryRequestScope {
    inner: candle_gen::request_scope::CandleRequestScopeCore,
}

impl WanMemoryRequestScope {
    fn validate_request(request: &GenerationRequest) -> gen_core::Result<()> {
        if !request.conditioning.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route requires empty T2V conditioning"
            )));
        }
        if request.enhance_prompt || request.use_uncensored_enhancer {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include prompt enhancement"
            )));
        }
        if request.video_mode.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include video_mode variants"
            )));
        }
        if request.phases.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include multi-phase generation"
            )));
        }
        let fps = request.fps.unwrap_or(DEFAULT_FPS);
        if fps != DEFAULT_FPS {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route requires exactly {DEFAULT_FPS} fps, got {fps}"
            )));
        }
        Ok(())
    }
}

impl MemoryRequestScope for WanMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        Self::validate_request(request)?;
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

fn begin_with_device(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, loaded_tier, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        MODEL_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        TransformerConfig::ti2v_5b().num_layers,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = DEFAULT_FRAMES;
    Ok(Some(Box::new(WanMemoryRequestScope {
        inner: candle_gen::request_scope::CandleRequestScopeCore::new(config),
    })))
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_device(contract, loaded_tier, device, context)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_device(
        contract,
        registered_tier(spec, contract)?,
        Device::Cpu,
        context,
    )
}

/// Validate and extract the selected rung-2 maximum tile edge from the actual request carrier.
/// Automatic legacy budgeted decode returns `None`; a selected rung returns `Some(cap)` and is
/// intersected with the live free-VRAM planner by [`crate::vae::WanVae`].
pub(crate) fn selected_decode_cap(request: &GenerationRequest) -> CandleResult<Option<u32>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: decode parameters were supplied without selecting bounded decode"
            )));
        }
        return Ok(None);
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    Ok(memory.decode_tile_edge)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: memory_strategy_contract,
    safety_check: registered_safety_check,
};

/// TI2V-5B advertises `supported_quants: [Q4, Q8]` plus the dense bf16 tier under both offload
/// policies, so it witnesses the common Candle registry tiers — but only the eager half of the
/// materialization axis. The provider has no deferred block loader: `validate_contract_route`
/// rejects `DeferredMaterialization` on the production route and on the weights-free route alike,
/// so publishing the deferred selectors would advertise a load surface no contract can be built
/// for. The witness set is deliberately the provider's own finite inventory, not the shared
/// default.
#[cfg(any(feature = "cuda", test))]
fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_memory_strategy_contract,
        surface_specs: memory_contract_surface_specs,
    };

#[cfg(any(feature = "cuda", test))]
pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_gen::candle_core::{
        safetensors as candle_safetensors, DType as CandleDType, Tensor,
    };
    use candle_gen::gen_core::{
        AdapterKind, AdapterSpec, GenerationMemory, Generator, MemoryBudget, MemoryGeometry,
        MemoryMode, MemoryRunOutcome, MemoryStrategyParameters,
    };

    use super::*;

    fn write_component(
        root: &Path,
        name: &str,
        tensors: impl IntoIterator<Item = (&'static str, Tensor)>,
    ) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(&dir.join("model.safetensors"), tensors);
    }

    fn write_shard(path: &Path, tensors: impl IntoIterator<Item = (&'static str, Tensor)>) {
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        candle_safetensors::save(&tensors, path).unwrap();
    }

    fn fixture_root(quant: Option<Quant>) -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        write_component(
            root,
            "text_encoder",
            [(
                "encoder.weight",
                Tensor::zeros((2, 3), CandleDType::F16, &Device::Cpu).unwrap(),
            )],
        );
        write_component(
            root,
            "vae",
            [(
                "decoder.weight",
                Tensor::zeros((2, 3), CandleDType::BF16, &Device::Cpu).unwrap(),
            )],
        );
        match quant {
            None => write_component(
                root,
                "transformer",
                [(
                    "proj.weight",
                    Tensor::zeros((2, 64), CandleDType::F32, &Device::Cpu).unwrap(),
                )],
            ),
            Some(quant @ (Quant::Q4 | Quant::Q8)) => {
                let code_cols = if quant == Quant::Q4 { 8 } else { 16 };
                write_component(
                    root,
                    "transformer",
                    [
                        (
                            "proj.weight",
                            Tensor::zeros((2, code_cols), CandleDType::U32, &Device::Cpu).unwrap(),
                        ),
                        (
                            "proj.scales",
                            Tensor::zeros((2, 1), CandleDType::F32, &Device::Cpu).unwrap(),
                        ),
                        (
                            "proj.biases",
                            Tensor::zeros((2, 1), CandleDType::F32, &Device::Cpu).unwrap(),
                        ),
                        (
                            "proj.bias",
                            Tensor::zeros(2, CandleDType::F32, &Device::Cpu).unwrap(),
                        ),
                        (
                            "patch_embedding.weight",
                            Tensor::zeros((2, 1, 1, 1, 1), CandleDType::F32, &Device::Cpu).unwrap(),
                        ),
                    ],
                );
                std::fs::write(
                    root.join("transformer/quantize_config.json"),
                    format!(
                        "{{\"bits\":{},\"quantization\":{{\"group_size\":64}}}}",
                        quant.bits()
                    ),
                )
                .unwrap();
            }
            Some(Quant::Nvfp4) => unreachable!(),
        }
        temp
    }

    fn spec(root: &Path, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()));
        spec.quantize = quant;
        spec
    }

    fn fixture_context(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryRunContext {
        registered_valid_fixtures(spec, contract, strategy)
            .unwrap()
            .pop()
            .unwrap()
            .context
    }

    /// AC (epic SC-22657, E2): the TI2V-5B contract publishes the architecture axes of the config
    /// the loader actually instantiates, and the weights-free surface publishes none of them.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let temp = fixture_root(None);
        let spec = spec(temp.path(), None);
        let contract = memory_strategy_contract(&spec).unwrap();
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // `TransformerConfig::ti2v_5b()`: 24 heads x 128 = dim 3072, 30 layers.
                attention_heads: Some(24),
                head_dim: Some(128),
                transformer_blocks: Some(30),
                // `patch = (1, 2, 2)`; the spatial (index 1) entry.
                patch_size: Some(2),
                // `VaeConfig::ti2v_5b().z_dim` — the z48 Wan 2.2 autoencoder.
                latent_channels: Some(48),
                // `Ti2vProviderVae::VAE_TILING == VaeTiling::WAN22`: x16 spatial, x4 causal temporal.
                vae_spatial_scale: Some(16),
                vae_temporal_scale: Some(4),
                // `lib.rs: DIT_DTYPE = DType::BF16`.
                activation_dtype_width: Some(2),
            }
        );
        assert!(contract.architecture_facts.has_declared_architecture_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The registry's weights-free surface names a sentinel that is not on disk: nothing about
        // the pipeline that would load has been resolved, so every axis stays absent.
        let weights_free_spec = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        let weights_free = weights_free_memory_strategy_contract(&weights_free_spec).unwrap();
        assert!(weights_free.architecture_facts.is_empty());

        // A14B (`t2v_14b` / `i2v_14b`, z16 VAE at x8 spatial) is a different generator with no
        // memory contract of its own, so this contract must not claim its geometry.
        assert_eq!(contract.provider_id, MODEL_ID);
    }

    #[test]
    fn contract_declares_only_the_three_real_wan_mechanisms() {
        let temp = fixture_root(None);
        let spec = spec(temp.path(), None);
        let contract = memory_strategy_contract(&spec).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        for implemented in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                contract.capability(implemented).unwrap().support,
                MemoryStrategySupport::Implemented
            );
        }
        for missing in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(missing).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
        assert!(contract.formula.uses(MemoryFormulaVariable::FrameCount));
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );

        let tier = resolved_numeric_tier(&spec).unwrap();
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, tier, false)
            .unwrap();
        let staged = contract
            .representative_selection(MemoryStrategy::StagedResidency, tier, false)
            .unwrap();
        let bounded = contract
            .representative_selection(MemoryStrategy::BoundedDecode, tier, false)
            .unwrap();
        assert_eq!(contract.generation_memory(&resident), None);
        assert_eq!(
            contract.generation_memory(&staged),
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
        let bounded = contract.generation_memory(&bounded).unwrap();
        assert!(!bounded.stage_residency);
        assert!(bounded.tile_vae_decode);
        assert_eq!(bounded.decode_tile_edge, Some(DECODE_TILE_EDGES[0]));
        assert_eq!(bounded.decode_overlap, Some(DECODE_OVERLAP));
    }

    #[test]
    fn dense_q4_and_q8_asset_facts_match_the_real_loaded_representations() {
        for (quant, transformer_bytes, base_bytes) in [
            (None, 256, 292),
            (Some(Quant::Q4), 88, 124),
            (Some(Quant::Q8), 144, 180),
        ] {
            let temp = fixture_root(quant);
            let spec = spec(temp.path(), quant);
            let (contract, tier) =
                production_contract_and_tier(&spec, &crate::DitSource::Snapshot).unwrap();
            assert_eq!(tier.precision, Precision::Bf16);
            assert_eq!(tier.quant, quant);
            assert_eq!(contract.asset_facts.conditioning_bytes, 12);
            assert_eq!(contract.asset_facts.transformer_bytes, transformer_bytes);
            assert_eq!(contract.asset_facts.decoder_bytes, 24);
            assert_eq!(contract.asset_facts.base_bytes, base_bytes);
            assert_eq!(contract.asset_facts.overlay_bytes, 0);
        }
    }

    #[test]
    fn asset_facts_use_the_loaders_direct_sorted_last_shard_wins_set() {
        let temp = fixture_root(None);
        let spec = spec(temp.path(), None);
        let baseline = memory_strategy_contract(&spec)
            .unwrap()
            .asset_facts
            .conditioning_bytes;

        let nested = temp.path().join("text_encoder/stale");
        std::fs::create_dir_all(&nested).unwrap();
        write_shard(
            &nested.join("zz.safetensors"),
            [(
                "encoder.weight",
                Tensor::zeros((20, 30), CandleDType::F32, &Device::Cpu).unwrap(),
            )],
        );
        assert_eq!(
            memory_strategy_contract(&spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes,
            baseline,
            "nested stale shards are not in component_vb's load set"
        );

        write_shard(
            &temp.path().join("text_encoder/zz-last.safetensors"),
            [(
                "encoder.weight",
                Tensor::zeros((4, 5), CandleDType::F32, &Device::Cpu).unwrap(),
            )],
        );
        assert_eq!(
            memory_strategy_contract(&spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes,
            40,
            "the lexically last direct shard must win duplicate keys like Candle mmap"
        );

        std::fs::create_dir(temp.path().join("text_encoder/bad.safetensors")).unwrap();
        assert!(memory_strategy_contract(&spec).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn asset_facts_follow_direct_file_symlinks_and_reject_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let temp = fixture_root(None);
        let outside = temp.path().join("outside.safetensors");
        write_shard(
            &outside,
            [(
                "encoder.weight",
                Tensor::zeros((3, 7), CandleDType::F32, &Device::Cpu).unwrap(),
            )],
        );
        symlink(
            &outside,
            temp.path().join("text_encoder/zz-linked.safetensors"),
        )
        .unwrap();
        let spec = spec(temp.path(), None);
        assert_eq!(
            memory_strategy_contract(&spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes,
            42
        );

        let directory = temp.path().join("outside-dir");
        std::fs::create_dir(&directory).unwrap();
        symlink(
            &directory,
            temp.path().join("text_encoder/zzz-dir.safetensors"),
        )
        .unwrap();
        assert!(memory_strategy_contract(&spec).is_err());
    }

    #[test]
    fn ordinary_non_float_leaves_never_publish_a_cast_underpriced_contract() {
        for component in ["text_encoder", "vae"] {
            let temp = fixture_root(None);
            write_component(
                temp.path(),
                component,
                [(
                    "ordinary.weight",
                    Tensor::zeros((2, 3), CandleDType::U8, &Device::Cpu).unwrap(),
                )],
            );
            assert!(memory_strategy_contract(&spec(temp.path(), None)).is_err());
        }

        let dense = fixture_root(None);
        write_component(
            dense.path(),
            "transformer",
            [(
                "proj.weight",
                Tensor::zeros((2, 64), CandleDType::U8, &Device::Cpu).unwrap(),
            )],
        );
        assert!(memory_strategy_contract(&spec(dense.path(), None)).is_err());

        let packed = fixture_root(Some(Quant::Q4));
        let mut tensors = HashMap::new();
        tensors.insert(
            "proj.weight".to_owned(),
            Tensor::zeros((2, 8), CandleDType::U32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "proj.scales".to_owned(),
            Tensor::zeros((2, 1), CandleDType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "proj.biases".to_owned(),
            Tensor::zeros((2, 1), CandleDType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "proj.bias".to_owned(),
            Tensor::zeros(2, CandleDType::U8, &Device::Cpu).unwrap(),
        );
        candle_safetensors::save(
            &tensors,
            packed.path().join("transformer/model.safetensors"),
        )
        .unwrap();
        assert!(memory_strategy_contract(&spec(packed.path(), Some(Quant::Q4))).is_err());

        // The official U32 code + floating affine grid remains admitted and is covered by the
        // exact q4/q8 byte goldens above.
        assert!(memory_strategy_contract(&spec(
            fixture_root(Some(Quant::Q4)).path(),
            Some(Quant::Q4)
        ))
        .is_ok());
    }

    #[test]
    fn packed_sidecar_and_header_mutations_remove_admission() {
        let temp = fixture_root(Some(Quant::Q4));
        let mut spec = spec(temp.path(), Some(Quant::Q4));
        assert!(memory_strategy_contract(&spec).is_ok());

        std::fs::write(
            temp.path().join("transformer/quantize_config.json"),
            r#"{"bits":8,"quantization":{"group_size":64}}"#,
        )
        .unwrap();
        assert!(memory_strategy_contract(&spec).is_err());
        assert!(contract_for_loaded(&spec, &crate::DitSource::Snapshot)
            .unwrap()
            .is_none());

        std::fs::write(
            temp.path().join("transformer/quantize_config.json"),
            r#"{"bits":4,"quantization":{"group_size":32}}"#,
        )
        .unwrap();
        assert!(memory_strategy_contract(&spec).is_err());

        spec.quantize = Some(Quant::Q8);
        std::fs::write(
            temp.path().join("transformer/quantize_config.json"),
            r#"{"bits":4,"quantization":{"group_size":64}}"#,
        )
        .unwrap();
        assert!(memory_strategy_contract(&spec).is_err());
        assert!(contract_for_loaded(&spec, &crate::DitSource::Snapshot)
            .unwrap()
            .is_none());
    }

    #[test]
    fn packed_or_orphan_dense_component_leaves_remove_admission() {
        let dev = Device::Cpu;
        for component in ["text_encoder", "vae"] {
            let temp = fixture_root(None);
            write_component(
                temp.path(),
                component,
                [
                    (
                        "proj.weight",
                        Tensor::zeros((2, 8), CandleDType::U32, &dev).unwrap(),
                    ),
                    (
                        "proj.scales",
                        Tensor::zeros((2, 1), CandleDType::F32, &dev).unwrap(),
                    ),
                    (
                        "proj.biases",
                        Tensor::zeros((2, 1), CandleDType::F32, &dev).unwrap(),
                    ),
                ],
            );
            let spec = spec(temp.path(), None);
            assert!(
                production_contract_and_tier(&spec, &crate::DitSource::Snapshot).is_err(),
                "a valid affine-packed {component} is outside the dense calibration"
            );
            assert!(contract_for_loaded(&spec, &crate::DitSource::Snapshot)
                .unwrap()
                .is_none());
        }

        for (name, tensor) in [
            (
                "orphan.scales",
                Tensor::zeros((2, 1), CandleDType::F32, &dev).unwrap(),
            ),
            (
                "orphan.weight",
                Tensor::zeros((2, 8), CandleDType::U32, &dev).unwrap(),
            ),
        ] {
            let temp = fixture_root(None);
            write_component(temp.path(), "text_encoder", [(name, tensor)]);
            let spec = spec(temp.path(), None);
            assert!(
                production_contract_and_tier(&spec, &crate::DitSource::Snapshot).is_err(),
                "malformed packed leaf {name} must fail open before publication"
            );
        }
    }

    #[test]
    fn native_gguf_source_never_receives_the_snapshot_contract() {
        let temp = fixture_root(None);
        let spec = spec(temp.path(), None);
        let source = crate::DitSource::NativeGguf("/weights/wan-q4-k-m.gguf".into());

        assert!(production_contract_and_tier(&spec, &source).is_err());
        assert!(contract_for_loaded(&spec, &source).unwrap().is_none());
        let loaded = crate::build_generator_with_source(&spec, source.clone()).unwrap();
        assert_eq!(loaded.memory_strategy_contract(), None);
        assert_eq!(loaded.dit_source, source);
    }

    /// The registry conformance walk constructs a weights-free contract for **every** selector the
    /// fixture publishes and fails the whole catalog when one errors. That walk only runs in the
    /// CUDA catalog lane, so this asserts the same property here, where it is reachable on CPU: the
    /// published witness set is exactly the set this provider can build, and the deferred half the
    /// shared Candle default would add is genuinely absent rather than silently dropped.
    #[test]
    fn every_published_contract_surface_builds_and_no_deferred_surface_is_published() {
        // Walk the registration itself, not the local helper: the registry reads this field, and a
        // registration pointed back at the shared Candle default is exactly the regression.
        let surfaces = (MEMORY_FIXTURE.surface_specs)();
        assert_eq!(
            surfaces.len(),
            gen_core::candle_memory_contract_surface_specs().len() / 2,
            "the witness set is the eager half of the shared Candle surface"
        );
        for surface in &surfaces {
            assert_eq!(
                surface.selector.load_shape,
                LoadShape::EagerMaterialization,
                "{} has no deferred block loader",
                surface.selector.id()
            );
            (MEMORY_FIXTURE.contract)(&surface.spec).unwrap_or_else(|error| {
                panic!("surface {} must build: {error}", surface.selector.id())
            });
        }
        assert!(
            gen_core::candle_memory_contract_surface_specs()
                .into_iter()
                .filter(|surface| surface.selector.load_shape == LoadShape::DeferredMaterialization)
                .all(|surface| weights_free_memory_strategy_contract(&surface.spec).is_err()),
            "a deferred surface that now builds must be published, not filtered out"
        );
    }

    #[test]
    fn unsupported_load_surfaces_fail_open_without_narrowing_the_generator() {
        let temp = fixture_root(None);
        let base = spec(temp.path(), None);
        for unsupported in [
            {
                let mut value = base.clone();
                value.load_shape = LoadShape::DeferredMaterialization;
                value
            },
            {
                let mut value = base.clone();
                value.precision = Precision::Fp32;
                value
            },
            {
                let mut value = base.clone();
                value.adapters.push(AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                ));
                value
            },
        ] {
            assert!(memory_strategy_contract(&unsupported).is_err());
            assert!(
                contract_for_loaded(&unsupported, &crate::DitSource::Snapshot)
                    .unwrap()
                    .is_none()
            );
            let generator = crate::build_generator(&unsupported).unwrap();
            assert_eq!(generator.memory_strategy_contract(), None);
        }
    }

    #[test]
    fn loaded_policy_is_preserved_and_request_staging_only_moves_toward_sequential() {
        let temp = fixture_root(None);
        let mut resident_spec = spec(temp.path(), None);
        let resident = crate::build_generator(&resident_spec).unwrap();
        assert_eq!(
            resident.request_offload(&GenerationRequest::default()),
            OffloadPolicy::Resident
        );
        let staged_request = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            resident.request_offload(&staged_request),
            OffloadPolicy::Sequential
        );

        resident_spec.offload_policy = OffloadPolicy::Sequential;
        let sequential = crate::build_generator(&resident_spec).unwrap();
        assert_eq!(
            sequential.request_offload(&GenerationRequest::default()),
            OffloadPolicy::Sequential
        );
        assert_eq!(
            sequential.request_offload(&GenerationRequest {
                memory: Some(GenerationMemory::default()),
                ..Default::default()
            }),
            OffloadPolicy::Sequential
        );
        assert_ne!(
            resident
                .memory_strategy_contract()
                .unwrap()
                .calibration
                .as_ref()
                .unwrap()
                .fingerprint,
            sequential
                .memory_strategy_contract()
                .unwrap()
                .calibration
                .as_ref()
                .unwrap()
                .fingerprint
        );
    }

    #[test]
    fn registry_and_loaded_generator_return_the_same_load_exact_contract() {
        let temp = fixture_root(Some(Quant::Q8));
        let spec = spec(temp.path(), Some(Quant::Q8));
        let registry = crate::provider_registry().unwrap();
        let registered = registry
            .memory_strategy_contract(MODEL_ID, &spec)
            .unwrap()
            .unwrap();
        let loaded = registry.load(MODEL_ID, &spec).unwrap();
        assert_eq!(loaded.memory_strategy_contract(), Some(&registered));
    }

    #[test]
    fn route_scope_binds_t2v_fps_and_default_frames_before_installing_controls() {
        let spec = LoadSpec::new(WeightsSource::Dir("/weights-free-wan".into()));
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let context = fixture_context(&spec, &contract, MemoryStrategy::BoundedDecode);

        for mutation in ["conditioning", "fps", "frames", "video_mode", "phases"] {
            let mut scope = registered_begin_request(&spec, &contract, &context)
                .unwrap()
                .unwrap();
            let mut request = MemoryBehaviorFixture::new(context.clone()).request;
            match mutation {
                "conditioning" => request
                    .conditioning
                    .push(gen_core::Conditioning::Reference {
                        image: gen_core::Image {
                            width: 1,
                            height: 1,
                            pixels: vec![0, 0, 0],
                        },
                        strength: None,
                    }),
                "fps" => request.fps = Some(DEFAULT_FPS + 1),
                "frames" => request.frames = Some(DEFAULT_FRAMES - 4),
                "video_mode" => request.video_mode = Some("variant".into()),
                "phases" => request.phases = Some(Vec::new()),
                _ => unreachable!(),
            }
            assert!(scope.configure_request(&mut request).is_err(), "{mutation}");
            assert_eq!(
                request.memory, None,
                "{mutation} mutated controls before reject"
            );
            scope
                .finish(MemoryRunOutcome::Error {
                    message: format!("rejected {mutation}"),
                })
                .unwrap();
        }

        let mut scope = registered_begin_request(&spec, &contract, &context)
            .unwrap()
            .unwrap();
        let mut request = MemoryBehaviorFixture::new(context.clone()).request;
        // `frames` is carried across from the fixture geometry by `MemoryBehaviorFixture::new`
        // (sc-19591), so the scope receives the calibrated clip length stated rather than left to
        // `CandleRequestScopeCore`'s `default_frames`. `fps` has no geometry axis and still arrives
        // unstated, which `validate_request` resolves to `DEFAULT_FPS`.
        assert_eq!(request.frames, Some(context.geometry.frames));
        assert_eq!(request.frames, Some(DEFAULT_FRAMES));
        assert_eq!(request.fps, None);
        scope.configure_request(&mut request).unwrap();
        assert!(request.memory.unwrap().tile_vae_decode);
        scope
            .configure_decode(DECODE_TILE_EDGES[0], DECODE_OVERLAP, context.geometry)
            .unwrap();
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
    }

    #[test]
    fn supported_route_rejections_are_fail_closed_before_begin() {
        let spec = LoadSpec::new(WeightsSource::Dir("/weights-free-wan".into()));
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let context = fixture_context(&spec, &contract, MemoryStrategy::StagedResidency);
        assert_eq!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );

        let mut mutations = Vec::new();
        let mut wrong_mode = context.clone();
        wrong_mode.mode = MemoryMode::TextToImage;
        mutations.push(wrong_mode);
        let mut wrong_geometry = context.clone();
        wrong_geometry.geometry = MemoryGeometry {
            frames: DEFAULT_FRAMES + 1,
            ..wrong_geometry.geometry
        };
        mutations.push(wrong_geometry);
        let mut wrong_tier = context.clone();
        wrong_tier.selection.tier.quant = Some(Quant::Q4);
        mutations.push(wrong_tier);
        let mut stale = context.clone();
        stale.calibration_fingerprint.push_str("-stale");
        mutations.push(stale);
        let mut over_budget = context.clone();
        over_budget.budget = MemoryBudget {
            total_bytes: over_budget.predicted_peak_bytes - 1,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        };
        mutations.push(over_budget);

        for mutation in mutations {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &mutation),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(registered_begin_request(&spec, &contract, &mutation).is_err());
        }
    }

    #[test]
    fn decode_carrier_rejects_irrelevant_or_mutated_parameters() {
        let request = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(selected_decode_cap(&request).unwrap(), Some(256));

        for memory in [
            GenerationMemory {
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            },
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            },
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP + 1),
                ..Default::default()
            },
        ] {
            assert!(selected_decode_cap(&GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            })
            .is_err());
        }
    }

    #[test]
    fn selection_parameter_mutations_are_rejected_by_the_contract() {
        let spec = LoadSpec::new(WeightsSource::Dir("/weights-free-wan".into()));
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let mut context = fixture_context(&spec, &contract, MemoryStrategy::BoundedDecode);
        context.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(matches!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
