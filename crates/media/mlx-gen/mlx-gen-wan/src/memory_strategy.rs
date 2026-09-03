//! Load-exact MLX Wan TI2V-5B video memory-strategy contract (SC-19236).
//!
//! The calibrated surface is deliberately narrower than [`crate::model::Wan`]'s complete creative
//! surface: plain text-to-video through `wan2_2_ti2v_5b`, with the provider-native resident load
//! shape and the existing z48 VAE decoder. Unsupported loads keep working through the historical
//! generator path but publish no contract. Malformed assets on an otherwise eligible load fail
//! closed instead of manufacturing resident-byte facts.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use mlx_gen::gen_core::weightsmeta::{Dtype, SafetensorsTensorHeader};
use mlx_gen::gen_core::{
    self, ComponentPrecisionFloor, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    OffloadPolicy, Precision, PrecisionFloorComponent, Quant, ResidentRequestMemory, WeightsSource,
};

use crate::config::{WanModelConfig, WanQuant, MAX_AREA_5B, MIN_SIZE, SIZE_MULTIPLE};
use crate::model::{MODEL_ID, TE_QUANT_BITS};

const CALIBRATION_PREFIX: &str = "sc-19236-wan2-2-ti2v-5b-mlx";
const STATIC_CALIBRATION_PREFIX: &str = "sc-19236-wan2-2-ti2v-5b-mlx-registry";
const PACK_GROUP_SIZE: usize = mlx_gen::quant::DEFAULT_GROUP_SIZE as usize;

/// The Q4 DiT route deliberately keeps the UMT5 projection stack at Q8. This is part of numeric
/// identity, not an implementation detail a caller should have to rediscover from provider source.
pub const COMPONENT_PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
    component: PrecisionFloorComponent::TextEncoder,
    selected_tier: Quant::Q4,
    resident_tier: Quant::Q8,
}];

fn active_floors(quant: Option<Quant>) -> &'static [ComponentPrecisionFloor] {
    if quant == Some(Quant::Q4) {
        COMPONENT_PRECISION_FLOORS
    } else {
        &[]
    }
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

fn tensor_elements(component: &str, tensor: &SafetensorsTensorHeader) -> gen_core::Result<u64> {
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
            "{MODEL_ID}: {component} tensor {} declares {} bytes but dtype/shape require {stored}",
            tensor.name, tensor.data_bytes
        )));
    }
    Ok(elements)
}

struct Headers {
    component: &'static str,
    tensors: BTreeMap<String, SafetensorsTensorHeader>,
    accounted: HashSet<String>,
}

impl Headers {
    fn read(path: &Path, component: &'static str) -> gen_core::Result<Self> {
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path).map_err(
            |error| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: {component} is not a complete header-readable safetensors component at {}: {error}",
                    path.display()
                ))
            },
        )?;
        if headers.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {component} has no tensors at {}",
                path.display()
            )));
        }
        for tensor in &headers {
            tensor_elements(component, tensor)?;
        }
        Ok(Self {
            component,
            tensors: headers
                .into_iter()
                .map(|tensor| (tensor.name.clone(), tensor))
                .collect(),
            accounted: HashSet::new(),
        })
    }

    fn take(
        &mut self,
        name: &str,
        shape: &[usize],
        require_float: bool,
    ) -> gen_core::Result<SafetensorsTensorHeader> {
        let tensor = self.tensors.get(name).cloned().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} is missing required tensor {name}",
                self.component
            ))
        })?;
        if tensor.shape != shape {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} tensor {name} has shape {:?}, expected {shape:?}",
                self.component, tensor.shape
            )));
        }
        // FP8 posture: `is_float` excludes `F8_E4M3`. MLX Wan reads canonical BF16 storage (see
        // `stored`/`materialized` below) through the plain `Weights::from_file` path, which rejects
        // fp8 files outright — the fp8-decoding loader (`from_file_with_fp8`) is opt-in and this
        // provider does not use it, so refusing fp8 here matches the loader it prices for.
        if require_float && !tensor.is_float() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} tensor {name} must be floating, got {:?}",
                self.component, tensor.dtype
            )));
        }
        self.accounted.insert(name.to_owned());
        Ok(tensor)
    }

    fn stored(&mut self, name: &str, shape: &[usize]) -> gen_core::Result<u64> {
        let tensor = self.take(name, shape, true)?;
        if tensor.dtype != Dtype::BF16 {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} tensor {name} must use canonical BF16 storage, got {:?}",
                self.component, tensor.dtype
            )));
        }
        Ok(tensor.data_bytes)
    }

    fn materialized(&mut self, name: &str, shape: &[usize], width: u64) -> gen_core::Result<u64> {
        let tensor = self.take(name, shape, true)?;
        if tensor.dtype != Dtype::BF16 {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} tensor {name} must use canonical BF16 source storage, got {:?}",
                self.component, tensor.dtype
            )));
        }
        tensor_elements(self.component, &tensor)?
            .checked_mul(width)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: {} tensor {name} projected byte count overflows u64",
                    self.component
                ))
            })
    }

    fn f32_projected(&mut self, name: &str, shape: &[usize], width: u64) -> gen_core::Result<u64> {
        let tensor = self.take(name, shape, true)?;
        if tensor.dtype != Dtype::F32 {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} tensor {name} must use canonical F32 source storage, got {:?}",
                self.component, tensor.dtype
            )));
        }
        tensor_elements(self.component, &tensor)?
            .checked_mul(width)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: {} tensor {name} projected byte count overflows u64",
                    self.component
                ))
            })
    }

    fn packed_present(&self, base: &str) -> bool {
        self.tensors.contains_key(&format!("{base}.scales"))
            || self.tensors.contains_key(&format!("{base}.biases"))
    }

    fn dense_linear(&mut self, base: &str, out: usize, input: usize) -> gen_core::Result<u64> {
        if self.packed_present(base) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} dense tier contains packed leaves for {base}",
                self.component
            )));
        }
        self.stored(&format!("{base}.weight"), &[out, input])
    }

    fn packed_linear(
        &mut self,
        base: &str,
        out: usize,
        input: usize,
        bits: i32,
        group_size: usize,
    ) -> gen_core::Result<u64> {
        if !matches!(bits, 4 | 8)
            || group_size == 0
            || input < group_size
            || !input.is_multiple_of(group_size)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {base} cannot be represented as Q{bits} group-{group_size}"
            )));
        }
        let packed_columns = input
            .checked_mul(bits as usize)
            .and_then(|bits| bits.checked_div(32))
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{MODEL_ID}: {base} packed width overflows usize"))
            })?;
        let groups = input / group_size;
        let weight = self.take(&format!("{base}.weight"), &[out, packed_columns], false)?;
        let scales = self.take(&format!("{base}.scales"), &[out, groups], true)?;
        let biases = self.take(&format!("{base}.biases"), &[out, groups], true)?;
        if weight.dtype != Dtype::U32 || scales.dtype != Dtype::BF16 || biases.dtype != Dtype::BF16
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} packed {base} must use U32 codes and BF16 scales/biases",
                self.component
            )));
        }
        checked_sum(
            &format!("{} packed {base}", self.component),
            [weight.data_bytes, scales.data_bytes, biases.data_bytes],
        )
    }

    /// UMT5 accepts a canonical dense source and packs it at load, or a byte-equivalent prepacked
    /// triple. Both resolve to the same resident Q8 projection.
    fn runtime_quantized_linear(
        &mut self,
        base: &str,
        out: usize,
        input: usize,
        bits: i32,
        group_size: usize,
    ) -> gen_core::Result<u64> {
        if self.packed_present(base) {
            return self.packed_linear(base, out, input, bits, group_size);
        }
        let dense = self.take(&format!("{base}.weight"), &[out, input], true)?;
        if dense.dtype != Dtype::BF16 {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} runtime-packed {base} must start from canonical BF16 storage",
                self.component
            )));
        }
        let elements = u64::try_from(out)
            .ok()
            .and_then(|out| {
                u64::try_from(input)
                    .ok()
                    .and_then(|input| out.checked_mul(input))
            })
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: {base} logical element count overflows u64"
                ))
            })?;
        let codes = elements
            .checked_mul(bits as u64)
            .and_then(|bits| bits.checked_div(8))
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{MODEL_ID}: {base} packed code bytes overflow u64"))
            })?;
        let tables = u64::try_from(out)
            .ok()
            .and_then(|out| {
                u64::try_from(input / group_size)
                    .ok()
                    .and_then(|groups| out.checked_mul(groups))
            })
            .and_then(|entries| entries.checked_mul(4))
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: {base} quantization-table bytes overflow u64"
                ))
            })?;
        codes.checked_add(tables).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: {base} runtime-packed bytes overflow u64"
            ))
        })
    }

    fn finish(self) -> gen_core::Result<u64> {
        for tensor in self.tensors.values() {
            if self.accounted.contains(&tensor.name) {
                continue;
            }
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} contains unconsumed tensor {} (orphan packed leaf or unsupported architecture)",
                self.component, tensor.name
            )));
        }
        Ok(0)
    }
}

fn supported_load_surface(spec: &LoadSpec) -> bool {
    matches!(&spec.weights, WeightsSource::Dir(_))
        && spec.precision == Precision::Bf16
        && spec.offload_policy == OffloadPolicy::Resident
        && spec.load_shape == LoadShape::EagerMaterialization
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
        && spec.adapters.is_empty()
        && spec.components.is_empty()
}

fn validate_load_surface(spec: &LoadSpec) -> gen_core::Result<&Path> {
    if !supported_load_surface(spec) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated memory admission requires a plain directory-backed BF16 Resident/Eager load without controls, external components, PiD, identity, or adapters"
        )));
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        unreachable!("supported_load_surface checked the source")
    };
    Ok(root)
}

fn canonical_config(root: &Path) -> gen_core::Result<WanModelConfig> {
    let path = root.join("config.json");
    if !path.is_file() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated memory admission requires the checkpoint's authoritative config.json"
        )));
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: invalid config.json at {}: {error}",
                path.display()
            ))
        })?;
    let quantization = match json.get("quantization") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) if value.is_object() => {
            let bits = value
                .get("bits")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{MODEL_ID}: config quantization must explicitly declare bits"
                    ))
                })?;
            let group_size = value
                .get("group_size")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{MODEL_ID}: config quantization must explicitly declare group_size"
                    ))
                })?;
            Some(WanQuant {
                bits: i32::try_from(bits).map_err(|_| {
                    gen_core::Error::Unsupported(format!(
                        "{MODEL_ID}: config quantization bits are outside i32"
                    ))
                })?,
                group_size: i32::try_from(group_size).map_err(|_| {
                    gen_core::Error::Unsupported(format!(
                        "{MODEL_ID}: config quantization group_size is outside i32"
                    ))
                })?,
            })
        }
        Some(_) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: config quantization must be an object"
            )))
        }
    };
    let config = WanModelConfig::from_config_json(&json);
    let mut canonical = WanModelConfig::wan22_ti2v_5b();
    canonical.quantization = quantization;
    if config != canonical || json != canonical.to_json() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: config.json is not the complete canonical dense Wan2.2 TI2V-5B configuration"
        )));
    }
    Ok(config)
}

fn numeric_tier_from_config(
    spec: &LoadSpec,
    config: &WanModelConfig,
) -> gen_core::Result<MemoryNumericTier> {
    let quant = match config.quantization {
        None => None,
        Some(WanQuant {
            bits: 4,
            group_size,
        }) if group_size == PACK_GROUP_SIZE as i32 => Some(Quant::Q4),
        Some(WanQuant {
            bits: 8,
            group_size,
        }) if group_size == PACK_GROUP_SIZE as i32 => Some(Quant::Q8),
        Some(WanQuant { bits, group_size }) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: config declares unsupported affine Q{bits} group-{group_size}; expected Q4/Q8 group-{PACK_GROUP_SIZE}"
            )))
        }
    };
    if let Some(requested) = spec.quantize {
        if Some(requested) != quant {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: requested tier {requested:?} does not match config.json's authoritative checkpoint tier {quant:?}"
            )));
        }
    }
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: active_floors(quant),
    })
}

fn fixture_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    if spec.quantize == Some(Quant::Nvfp4) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: NVFP4 is not an MLX affine tier"
        )));
    }
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: spec.quantize,
        component_precision_floors: active_floors(spec.quantize),
    })
}

fn conditioning_bytes(root: &Path, config: &WanModelConfig) -> gen_core::Result<u64> {
    let mut headers = Headers::read(&root.join("t5_encoder.safetensors"), "UMT5 text encoder")?;
    let mut total = headers.stored(
        "token_embedding.weight",
        &[config.t5_vocab_size, config.t5_dim],
    )?;
    if headers.packed_present("token_embedding") {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated UMT5 route keeps token_embedding dense"
        )));
    }
    let quantized = config.quantization.is_some();
    for block in 0..config.t5_num_layers {
        let prefix = format!("blocks.{block}");
        total = checked_sum(
            "UMT5",
            [
                total,
                headers.materialized(&format!("{prefix}.norm1.weight"), &[config.t5_dim], 4)?,
                headers.materialized(&format!("{prefix}.norm2.weight"), &[config.t5_dim], 4)?,
                headers.materialized(
                    &format!("{prefix}.pos_embedding.embedding.weight"),
                    &[config.t5_num_buckets, config.t5_num_heads],
                    4,
                )?,
            ],
        )?;
        for (suffix, out, input) in [
            ("attn.q", config.t5_dim_attn, config.t5_dim),
            ("attn.k", config.t5_dim_attn, config.t5_dim),
            ("attn.v", config.t5_dim_attn, config.t5_dim),
            ("attn.o", config.t5_dim, config.t5_dim_attn),
            ("ffn.gate_proj", config.t5_dim_ffn, config.t5_dim),
            ("ffn.fc1", config.t5_dim_ffn, config.t5_dim),
            ("ffn.fc2", config.t5_dim, config.t5_dim_ffn),
        ] {
            let base = format!("{prefix}.{suffix}");
            let bytes = if quantized {
                headers.runtime_quantized_linear(
                    &base,
                    out,
                    input,
                    TE_QUANT_BITS,
                    PACK_GROUP_SIZE,
                )?
            } else {
                headers.dense_linear(&base, out, input)?
            };
            total = checked_sum("UMT5", [total, bytes])?;
        }
    }
    total = checked_sum(
        "UMT5",
        [
            total,
            headers.materialized("norm.weight", &[config.t5_dim], 4)?,
            headers.finish()?,
        ],
    )?;
    Ok(total)
}

fn transformer_bytes(root: &Path, config: &WanModelConfig) -> gen_core::Result<u64> {
    let mut headers = Headers::read(&root.join("model.safetensors"), "Wan transformer")?;
    let patch_volume = config
        .patch_size
        .0
        .checked_mul(config.patch_size.1)
        .and_then(|value| value.checked_mul(config.patch_size.2))
        .ok_or_else(|| gen_core::Error::Msg(format!("{MODEL_ID}: patch volume overflows usize")))?;
    let patch_input = config
        .in_dim
        .checked_mul(patch_volume)
        .ok_or_else(|| gen_core::Error::Msg(format!("{MODEL_ID}: patch input overflows usize")))?;
    let patch_output = config
        .out_dim
        .checked_mul(patch_volume)
        .ok_or_else(|| gen_core::Error::Msg(format!("{MODEL_ID}: patch output overflows usize")))?;
    let six_dim = config.dim.checked_mul(6).ok_or_else(|| {
        gen_core::Error::Msg(format!("{MODEL_ID}: six-dim width overflows usize"))
    })?;

    let mut total = 0_u64;
    for (base, out, input) in [
        ("patch_embedding_proj", config.dim, patch_input),
        ("text_embedding_0", config.dim, config.text_dim),
        ("text_embedding_1", config.dim, config.dim),
        ("time_embedding_0", config.dim, config.freq_dim),
        ("time_embedding_1", config.dim, config.dim),
        ("time_projection", six_dim, config.dim),
        ("head.head", patch_output, config.dim),
    ] {
        total = checked_sum(
            "Wan transformer",
            [
                total,
                headers.dense_linear(base, out, input)?,
                headers.stored(&format!("{base}.bias"), &[out])?,
            ],
        )?;
    }
    total = checked_sum(
        "Wan transformer",
        [
            total,
            headers.materialized("head.modulation", &[1, 2, config.dim], 4)?,
        ],
    )?;

    for block in 0..config.num_layers {
        let prefix = format!("blocks.{block}");
        total = checked_sum(
            "Wan transformer",
            [
                total,
                headers.materialized(&format!("{prefix}.modulation"), &[1, 6, config.dim], 4)?,
                headers.materialized(&format!("{prefix}.norm3.weight"), &[config.dim], 4)?,
                headers.materialized(&format!("{prefix}.norm3.bias"), &[config.dim], 4)?,
            ],
        )?;
        for norm in [
            "self_attn.norm_q.weight",
            "self_attn.norm_k.weight",
            "cross_attn.norm_q.weight",
            "cross_attn.norm_k.weight",
        ] {
            total = checked_sum(
                "Wan transformer",
                [
                    total,
                    headers.stored(&format!("{prefix}.{norm}"), &[config.dim])?,
                ],
            )?;
        }
        for (suffix, out, input) in [
            ("self_attn.q", config.dim, config.dim),
            ("self_attn.k", config.dim, config.dim),
            ("self_attn.v", config.dim, config.dim),
            ("self_attn.o", config.dim, config.dim),
            ("cross_attn.q", config.dim, config.dim),
            ("cross_attn.k", config.dim, config.dim),
            ("cross_attn.v", config.dim, config.dim),
            ("cross_attn.o", config.dim, config.dim),
            ("ffn.fc1", config.ffn_dim, config.dim),
            ("ffn.fc2", config.dim, config.ffn_dim),
        ] {
            let base = format!("{prefix}.{suffix}");
            let weight = match (
                config.quantization,
                crate::convert::is_wan_quantized_linear(&base),
            ) {
                (Some(quant), true) => headers.packed_linear(
                    &base,
                    out,
                    input,
                    quant.bits,
                    quant.group_size as usize,
                )?,
                _ => headers.dense_linear(&base, out, input)?,
            };
            total = checked_sum(
                "Wan transformer",
                [
                    total,
                    weight,
                    headers.stored(&format!("{base}.bias"), &[out])?,
                ],
            )?;
        }
    }
    checked_sum("Wan transformer", [total, headers.finish()?])
}

fn decoder_bytes(root: &Path) -> gen_core::Result<u64> {
    let mut headers = Headers::read(&root.join("vae.safetensors"), "Wan z48 VAE")?;
    let mut total = 0_u64;
    // `convert_ti2v_5b` always emits the encoder and the runtime's decode stage calls
    // `Weights::cast_all(Bfloat16)` before constructing `Wan22Vae`. Therefore the projected decode
    // residency is the complete canonical VAE file, not an arbitrary sum of whatever F32 leaves a
    // file happens to contain. The schema is generated in `vae22` from the same stage widths/counts
    // used by the concrete loader; `finish` rejects every orphan/unconsumed key.
    let schema = crate::vae22::production_weight_schema();
    for tensor in schema.tensors() {
        total = checked_sum(
            "Wan z48 VAE",
            [
                total,
                headers.f32_projected(&tensor.name, &tensor.shape, 2)?,
            ],
        )?;
    }
    checked_sum("Wan z48 VAE", [total, headers.finish()?])
}

fn calibration_fingerprint(tier: MemoryNumericTier, fixture: bool) -> String {
    let prefix = if fixture {
        STATIC_CALIBRATION_PREFIX
    } else {
        CALIBRATION_PREFIX
    };
    match tier.quant {
        None => format!("{prefix}-dense-v1"),
        Some(Quant::Q4) => format!("{prefix}-q4-g{PACK_GROUP_SIZE}-teq8-v1"),
        Some(Quant::Q8) => format!("{prefix}-q8-g{PACK_GROUP_SIZE}-teq8-v1"),
        Some(Quant::Nvfp4) => unreachable!("fixture/production tier validation rejects NVFP4"),
    }
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: crate::pipeline::VAE22_SPATIAL_PX
                        .iter()
                        .map(|&edge| edge as u32)
                        .collect(),
                    decode_overlaps: vec![crate::pipeline::VAE22_SELECTED_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// Architecture axes for the Wan 2.2 TI2V-5B route this module contracts for (epic SC-22657, E2).
///
/// [`WanModelConfig::wan22_ti2v_5b`](crate::config::WanModelConfig::wan22_ti2v_5b) is this crate's
/// preset for that variant's `config.json`, which `WanModelConfig::from_model_dir` overlays a
/// snapshot's own keys onto at load. **This is the 5B route only** — the 14B trunks are a different
/// shape (dim 5120, 40 heads, 40 layers over the z16 VAE), which is why the axes are keyed on the
/// 5B preset rather than on `base()`.
///
/// TI2V-5B pairs a wider latent (48 channels) with the z48 autoencoder's x16 spatial and x4 temporal
/// scales, both read from the same preset's `vae_stride`/`vae_z_dim`. Being a **video** autoencoder,
/// `vae_temporal_scale` is a real value here rather than a structurally absent one.
///
/// When `spec` names a materialized snapshot directory this re-runs `from_model_dir` — the loader's
/// own parse — so the published axes are the snapshot's, not the preset's. On the weights-free
/// surface there is nothing to read and the preset is what the loader would start from anyway.
fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let wan = mlx_gen::architecture_facts::materialized_root(spec)
        .and_then(|root| crate::config::WanModelConfig::from_model_dir(root).ok())
        .unwrap_or_else(crate::config::WanModelConfig::wan22_ti2v_5b);
    let (_, patch_h, patch_w) = wan.patch_size;
    let (temporal_stride, spatial_stride, _) = wan.vae_stride;
    mlx_gen::gen_core::MemoryArchitectureFacts {
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

fn build_contract(
    spec: &LoadSpec,
    facts: MemoryAssetFacts,
    tier: MemoryNumericTier,
    fixture: bool,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_surface(spec)?;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    Ok(MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: false,
        },
        strategies: strategies(),
        // Wan declares no decode-quality geometry policy table, so this route carries no semantic
        // decode authority — the fail-closed default every non-declaring provider contract uses.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
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
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint(tier, fixture),
            spec.load_shape,
        )),
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

fn production_contract_and_tier(
    spec: &LoadSpec,
) -> gen_core::Result<(MemoryProviderContract, MemoryNumericTier)> {
    let root = validate_load_surface(spec)?;
    let config = canonical_config(root)?;
    let tier = numeric_tier_from_config(spec, &config)?;
    if !root.join("tokenizer.json").is_file() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: calibrated route requires tokenizer.json in the checkpoint root"
        )));
    }
    let conditioning_bytes = conditioning_bytes(root, &config)?;
    let transformer_bytes = transformer_bytes(root, &config)?;
    let decoder_bytes = decoder_bytes(root)?;
    let base_bytes = checked_sum(
        "base model",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    let facts = MemoryAssetFacts {
        base_bytes,
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes: 0,
    };
    Ok((build_contract(spec, facts, tier, false)?, tier))
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    production_contract_and_tier(spec).map(|(contract, _)| contract)
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let root = validate_load_surface(spec)?;
    numeric_tier_from_config(spec, &canonical_config(root)?)
}

pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    if !supported_load_surface(spec) {
        return Ok(None);
    }
    // A dense checkpoint plus `spec.quantize` is the historical load-time packing route. It remains
    // supported by the generator, but this contract's q4/q8 identities are deliberately tied to an
    // authoritative prepacked config/header surface. Fail open to the compatibility path instead of
    // breaking that older load or pretending its load transient shares prepacked evidence.
    if spec.quantize.is_some() {
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!("supported_load_surface checked the source")
        };
        let config_path = root.join("config.json");
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)
            .map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: invalid config.json at {}: {error}",
                    config_path.display()
                ))
            })?;
        if json
            .get("quantization")
            .is_none_or(serde_json::Value::is_null)
        {
            return Ok(None);
        }
    }
    production_contract_and_tier(spec).map(Some)
}

pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let tier = fixture_tier(spec)?;
    build_contract(spec, MemoryAssetFacts::default(), tier, true)
}

fn fixture_contract(contract: &MemoryProviderContract) -> bool {
    contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint.starts_with(STATIC_CALIBRATION_PREFIX))
}

fn registered_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<MemoryNumericTier> {
    if fixture_contract(contract) {
        fixture_tier(spec)
    } else {
        resolved_numeric_tier(spec)
    }
}

fn calibrated_default_frames() -> gen_core::Result<u32> {
    u32::try_from(WanModelConfig::wan22_ti2v_5b().frame_num).map_err(|_| {
        gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: configured default frame count exceeds u32"
        ))
    })
}

pub(crate) fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
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
    if !crate::pipeline::VAE22_SPATIAL_PX.contains(&(edge as i32)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode tile-edge cap {edge} is outside the production domain {:?}",
            crate::pipeline::VAE22_SPATIAL_PX
        )));
    }
    if overlap != crate::pipeline::VAE22_SELECTED_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: decode overlap {overlap} is not the production overlap {}",
            crate::pipeline::VAE22_SELECTED_OVERLAP
        )));
    }
    Ok(())
}

pub(crate) fn validate_geometry(
    width: u32,
    height: u32,
    frames: u32,
    batch: u32,
) -> gen_core::Result<()> {
    let area = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| gen_core::Error::Msg(format!("{MODEL_ID}: geometry area overflows u64")))?;
    if batch != 1
        || !(MIN_SIZE..=1280).contains(&width)
        || !(MIN_SIZE..=1280).contains(&height)
        || !width.is_multiple_of(SIZE_MULTIPLE)
        || !height.is_multiple_of(SIZE_MULTIPLE)
        || area > MAX_AREA_5B as u64
        || !(1..=crate::MAX_WAN_FRAMES as u32).contains(&frames)
        || !(frames - 1).is_multiple_of(4)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: unsupported calibrated geometry {width}x{height}x{batch} frames={frames}"
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
                "{MODEL_ID}: calibrated memory route is plain single-phase text_to_video without references, PiD, or overlays"
            )));
        }
        validate_geometry(
            context.geometry.width,
            context.geometry.height,
            context.geometry.frames,
            context.geometry.batch,
        )?;
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
        Ok(tier) => safety_check(contract, tier, context),
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
    if strategy != MemoryStrategy::BoundedDecode
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let tier = registered_tier(spec, contract)?;
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".to_owned()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 480;
    context.geometry.height = 480;
    context.geometry.frames = calibrated_default_frames()?;
    // A 448 cap is observable on the minimum production geometry. Keeping this fixture explicit
    // prevents a behavior check from accidentally exercising a no-op full spatial pass.
    context.selection.parameters.decode_tile_edge = Some(448);
    context.selection.parameters.decode_overlap = Some(crate::pipeline::VAE22_SELECTED_OVERLAP);
    contract.validate_selection(&context.selection)?;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free MLX Wan memory behavior".to_owned();
    Ok(vec![fixture])
}

struct WanMemoryRequestScope {
    inner: mlx_gen::request_scope::MlxRequestScopeCore,
}

impl WanMemoryRequestScope {
    fn validate_request(request: &GenerationRequest) -> gen_core::Result<()> {
        if !request.conditioning.is_empty()
            || request.strength.is_some()
            || request.control_scale.is_some()
            || request.text_style_gain.is_some()
            || request.image_guidance.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route requires empty T2V conditioning and no conditioning-only controls"
            )));
        }
        if request.audio.is_some() || request.phases.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include audio or multi-phase generation"
            )));
        }
        if request.video_mode.is_some()
            || request.trim_first_frames.is_some()
            || request.duration.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include video mode, trim, or duration variants"
            )));
        }
        if request.motion_bucket_id.is_some()
            || request.noise_aug_strength.is_some()
            || request.decode_chunk_size.is_some()
            || request.conditioning_fps.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include SVD controls"
            )));
        }
        if request.enhance_prompt
            || request.use_uncensored_enhancer
            || request.enhance_max_tokens.is_some()
            || request.enhance_temperature.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include prompt enhancement"
            )));
        }
        if request.softness.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route does not include SeedVR2 controls"
            )));
        }
        if request.use_pid || request.pid_capture_sigma.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route uses the native Wan VAE, not PiD"
            )));
        }
        let fps = request.fps.unwrap_or(24);
        if !(24..=30).contains(&fps) {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: calibrated memory route requires fps in 24..=30, got {fps}"
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
        MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        WanModelConfig::wan22_ti2v_5b().num_layers,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = calibrated_default_frames()?;
    Ok(Some(Box::new(WanMemoryRequestScope {
        inner: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    })))
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

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        contract,
        registered_tier(spec, contract)?,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

/// Translate the worker-owned generation carrier into the exact z48 decode configuration before
/// Stage 1 starts. An absent/default carrier preserves Wan's historical automatic planner. A
/// bounded-decode carrier must contain only this contract's selected edge and overlap; unsupported
/// rung flags or stray parameters fail closed rather than being silently ignored.
pub(crate) fn decode_tiling(
    request: &GenerationRequest,
    width: u32,
    height: u32,
    frames: u32,
) -> mlx_gen::Result<Option<mlx_gen::tiling::TilingConfig>> {
    let Some(memory) = request.memory else {
        return crate::pipeline::auto_tiling_budgeted(
            height as i32,
            width as i32,
            frames as i32,
            true,
        );
    };
    if memory == gen_core::GenerationMemory::default() {
        return crate::pipeline::auto_tiling_budgeted(
            height as i32,
            width as i32,
            frames as i32,
            true,
        );
    }
    let selected = gen_core::GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: memory.decode_tile_edge,
        decode_overlap: memory.decode_overlap,
        ..Default::default()
    };
    if memory != selected {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{MODEL_ID}: bounded decode carrier contains an unsupported memory-strategy rung or parameter"
        )));
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)
        .map_err(mlx_gen::Error::from)?;
    validate_geometry(width, height, frames, 1).map_err(mlx_gen::Error::from)?;
    crate::pipeline::selected_vae22_plan(
        width,
        height,
        frames,
        memory.decode_tile_edge.expect("validated tile edge"),
        memory.decode_overlap.expect("validated overlap"),
    )
    .map(|(tiling, _)| Some(tiling))
}

pub fn selected_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
) -> gen_core::Result<Option<mlx_gen::VideoDecodeMemoryProfile>> {
    if provider_id != MODEL_ID {
        return Ok(None);
    }
    validate_geometry(width, height, frames, 1)?;
    validate_decode(Some(tile_edge), Some(overlap))?;
    crate::pipeline::selected_vae22_plan(width, height, frames, tile_edge, overlap)
        .map(|(_, profile)| Some(profile))
        .map_err(Into::into)
}

pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: memory_strategy_contract,
    safety_check: registered_safety_check,
};

/// TI2V-5B witnesses each shared MLX tier exactly once. `supported_load_surface` admits only a
/// directory-backed BF16 **Resident/Eager** load, so the sequential and deferred selectors of the
/// shared MLX default have no constructible contract; publishing them would fail the registry
/// surface walk for the entire MLX catalog. The tier axis is the only one this provider varies.
pub(crate) fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.selector.offload_policy == OffloadPolicy::Resident
                && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

pub(crate) const MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_memory_strategy_contract,
        surface_specs: memory_contract_surface_specs,
    };

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        AdapterKind, AdapterSpec, AudioParams, Conditioning, GenerationMemory, GenerationPhase,
        IdentityWeights, Image, MemoryGeometry, MemoryStrategyParameters, PidWeights,
    };
    use std::io::Write;
    use std::path::PathBuf;

    /// AC (SC-22662): the TI2V-5B contract publishes the axes of the 5B trunk and z48 video VAE this
    /// crate's preset declares, and passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_the_ti2v_5b_preset_and_its_z48_vae() {
        for surface in (MEMORY_FIXTURE.surface_specs)() {
            let contract = (MEMORY_FIXTURE.contract)(&surface.spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(24),
                    // 3072 / 24, derived by `WanModelConfig::head_dim`.
                    head_dim: Some(128),
                    transformer_blocks: Some(30),
                    // `patch_size` is `(1, 2, 2)`: the square spatial patch is 2.
                    patch_size: Some(2),
                    // The z48 autoencoder, not the 14B routes' z16 one.
                    latent_channels: Some(48),
                    vae_spatial_scale: Some(16),
                    // A video autoencoder: four frames per latent unit.
                    vae_temporal_scale: Some(4),
                    // Two of the three declared phases run f32 (the z16 autoencoder and
                    // the UMT5-XXL activations), so the one scalar is the widest of them.
                    activation_dtype_width: Some(4),
                },
                "{} architecture facts",
                surface.selector.id()
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        // The published pair IS the z48 tiling geometry this provider's VAE assignment declares.
        let preset_facts = architecture_facts(&weights_free_spec());
        assert_eq!(
            crate::WAN_Z48_VAE_TILING.spatial_scale as u32,
            preset_facts.vae_spatial_scale.unwrap()
        );
        assert_eq!(
            crate::WAN_Z48_VAE_TILING.temporal_scale as u32,
            preset_facts.vae_temporal_scale.unwrap()
        );
    }

    /// A spec whose weights directory is the registry's never-created contract-surface sentinel:
    /// the weights-free path, where the preset is the only geometry there is.
    fn weights_free_spec() -> LoadSpec {
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ))
    }

    fn spec_for_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(dir.to_path_buf()))
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the axes are the snapshot's
    /// own, because the loader overlays that snapshot's `config.json` over the preset. A fixture
    /// mirroring the reference config agrees with the weights-free preset path; a fixture with one
    /// mutated key publishes the mutated axis. The second half is what an unconditional
    /// `architecture_facts()` — the shape this function had before review — fails.
    #[test]
    fn materialized_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = crate::config::WanModelConfig::wan22_ti2v_5b();

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_config(mirror.path(), &preset.to_json())),
            architecture_facts(&weights_free_spec()),
            "a snapshot mirroring the reference config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = preset.to_json();
        mutated["num_layers"] = serde_json::json!(7);
        mutated["vae_z_dim"] = serde_json::json!(32);
        let mutated_facts = architecture_facts(&spec_for_config(mutated_dir.path(), &mutated));
        assert_eq!(
            (
                mutated_facts.transformer_blocks,
                mutated_facts.latent_channels
            ),
            (Some(7), Some(32)),
            "the materialized path must publish the snapshot's geometry, not the preset's"
        );
    }

    /// `ProviderRegistry::memory_contract_surfaces` constructs a contract for **every** selector the
    /// fixture publishes and fails the entire MLX catalog when one errors, so the published witness
    /// set must be exactly the set this provider can build. Asserting it here localizes the failure
    /// to this provider instead of surfacing it as eight red `mlx-gen-catalog` tests.
    #[test]
    fn every_published_contract_surface_builds_and_only_resident_eager_is_published() {
        let surfaces = (MEMORY_FIXTURE.surface_specs)();
        assert_eq!(
            surfaces.len(),
            gen_core::mlx_memory_contract_surface_specs().len() / 4,
            "one witness per shared MLX tier, Resident/Eager only"
        );
        for surface in &surfaces {
            assert_eq!(surface.selector.offload_policy, OffloadPolicy::Resident);
            assert_eq!(surface.selector.load_shape, LoadShape::EagerMaterialization);
            (MEMORY_FIXTURE.contract)(&surface.spec).unwrap_or_else(|error| {
                panic!("surface {} must build: {error}", surface.selector.id())
            });
        }
        assert!(
            gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .filter(
                    |surface| surface.selector.offload_policy != OffloadPolicy::Resident
                        || surface.selector.load_shape != LoadShape::EagerMaterialization
                )
                .all(|surface| weights_free_memory_strategy_contract(&surface.spec).is_err()),
            "a surface that now builds must be published, not filtered out"
        );
    }

    #[derive(Clone, Debug)]
    struct HeaderTensor {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
        data_bytes: u64,
    }

    impl HeaderTensor {
        fn new(name: impl Into<String>, dtype: &'static str, shape: &[usize]) -> Self {
            let width = match dtype {
                "BF16" => 2_u64,
                "F32" | "U32" => 4,
                other => panic!("unsupported fixture dtype {other}"),
            };
            let elements = shape
                .iter()
                .try_fold(1_u64, |total, &dimension| {
                    total.checked_mul(dimension as u64)
                })
                .expect("fixture tensor size");
            Self {
                name: name.into(),
                dtype,
                shape: shape.to_vec(),
                data_bytes: elements.checked_mul(width).expect("fixture tensor bytes"),
            }
        }
    }

    fn dense(
        tensors: &mut Vec<HeaderTensor>,
        base: impl Into<String>,
        out: usize,
        input: usize,
        bias: bool,
    ) {
        let base = base.into();
        tensors.push(HeaderTensor::new(
            format!("{base}.weight"),
            "BF16",
            &[out, input],
        ));
        if bias {
            tensors.push(HeaderTensor::new(format!("{base}.bias"), "BF16", &[out]));
        }
    }

    fn packed(
        tensors: &mut Vec<HeaderTensor>,
        base: impl Into<String>,
        out: usize,
        input: usize,
        bits: i32,
        bias: bool,
    ) {
        let base = base.into();
        tensors.push(HeaderTensor::new(
            format!("{base}.weight"),
            "U32",
            &[out, input * bits as usize / 32],
        ));
        tensors.push(HeaderTensor::new(
            format!("{base}.scales"),
            "BF16",
            &[out, input / PACK_GROUP_SIZE],
        ));
        tensors.push(HeaderTensor::new(
            format!("{base}.biases"),
            "BF16",
            &[out, input / PACK_GROUP_SIZE],
        ));
        if bias {
            tensors.push(HeaderTensor::new(format!("{base}.bias"), "BF16", &[out]));
        }
    }

    fn t5_headers(config: &WanModelConfig) -> Vec<HeaderTensor> {
        let mut tensors = vec![HeaderTensor::new(
            "token_embedding.weight",
            "BF16",
            &[config.t5_vocab_size, config.t5_dim],
        )];
        for block in 0..config.t5_num_layers {
            let prefix = format!("blocks.{block}");
            tensors.push(HeaderTensor::new(
                format!("{prefix}.norm1.weight"),
                "BF16",
                &[config.t5_dim],
            ));
            tensors.push(HeaderTensor::new(
                format!("{prefix}.norm2.weight"),
                "BF16",
                &[config.t5_dim],
            ));
            tensors.push(HeaderTensor::new(
                format!("{prefix}.pos_embedding.embedding.weight"),
                "BF16",
                &[config.t5_num_buckets, config.t5_num_heads],
            ));
            for (suffix, out, input) in [
                ("attn.q", config.t5_dim_attn, config.t5_dim),
                ("attn.k", config.t5_dim_attn, config.t5_dim),
                ("attn.v", config.t5_dim_attn, config.t5_dim),
                ("attn.o", config.t5_dim, config.t5_dim_attn),
                ("ffn.gate_proj", config.t5_dim_ffn, config.t5_dim),
                ("ffn.fc1", config.t5_dim_ffn, config.t5_dim),
                ("ffn.fc2", config.t5_dim, config.t5_dim_ffn),
            ] {
                dense(
                    &mut tensors,
                    format!("{prefix}.{suffix}"),
                    out,
                    input,
                    false,
                );
            }
        }
        tensors.push(HeaderTensor::new("norm.weight", "BF16", &[config.t5_dim]));
        tensors
    }

    fn transformer_headers(config: &WanModelConfig) -> Vec<HeaderTensor> {
        let patch_volume = config.patch_size.0 * config.patch_size.1 * config.patch_size.2;
        let mut tensors = Vec::new();
        for (base, out, input) in [
            (
                "patch_embedding_proj",
                config.dim,
                config.in_dim * patch_volume,
            ),
            ("text_embedding_0", config.dim, config.text_dim),
            ("text_embedding_1", config.dim, config.dim),
            ("time_embedding_0", config.dim, config.freq_dim),
            ("time_embedding_1", config.dim, config.dim),
            ("time_projection", config.dim * 6, config.dim),
            ("head.head", config.out_dim * patch_volume, config.dim),
        ] {
            dense(&mut tensors, base, out, input, true);
        }
        tensors.push(HeaderTensor::new(
            "head.modulation",
            "BF16",
            &[1, 2, config.dim],
        ));
        for block in 0..config.num_layers {
            let prefix = format!("blocks.{block}");
            tensors.push(HeaderTensor::new(
                format!("{prefix}.modulation"),
                "BF16",
                &[1, 6, config.dim],
            ));
            for name in ["norm3.weight", "norm3.bias"] {
                tensors.push(HeaderTensor::new(
                    format!("{prefix}.{name}"),
                    "BF16",
                    &[config.dim],
                ));
            }
            for name in [
                "self_attn.norm_q.weight",
                "self_attn.norm_k.weight",
                "cross_attn.norm_q.weight",
                "cross_attn.norm_k.weight",
            ] {
                tensors.push(HeaderTensor::new(
                    format!("{prefix}.{name}"),
                    "BF16",
                    &[config.dim],
                ));
            }
            for (suffix, out, input) in [
                ("self_attn.q", config.dim, config.dim),
                ("self_attn.k", config.dim, config.dim),
                ("self_attn.v", config.dim, config.dim),
                ("self_attn.o", config.dim, config.dim),
                ("cross_attn.q", config.dim, config.dim),
                ("cross_attn.k", config.dim, config.dim),
                ("cross_attn.v", config.dim, config.dim),
                ("cross_attn.o", config.dim, config.dim),
                ("ffn.fc1", config.ffn_dim, config.dim),
                ("ffn.fc2", config.dim, config.ffn_dim),
            ] {
                let base = format!("{prefix}.{suffix}");
                match config.quantization {
                    Some(quant) => packed(&mut tensors, base, out, input, quant.bits, true),
                    None => dense(&mut tensors, base, out, input, true),
                }
            }
        }
        tensors
    }

    fn checkpoint_parts(
        quant: Option<Quant>,
    ) -> (
        serde_json::Value,
        Vec<HeaderTensor>,
        Vec<HeaderTensor>,
        Vec<HeaderTensor>,
    ) {
        let mut config = WanModelConfig::wan22_ti2v_5b();
        config.quantization = quant.map(|quant| WanQuant {
            bits: quant.bits(),
            group_size: PACK_GROUP_SIZE as i32,
        });
        let config_json = config.to_json();
        let t5 = t5_headers(&config);
        let transformer = transformer_headers(&config);
        let vae = crate::vae22::production_weight_schema()
            .tensors()
            .map(|tensor| HeaderTensor::new(&tensor.name, "F32", &tensor.shape))
            .collect();
        (config_json, transformer, t5, vae)
    }

    fn write_headers(path: &Path, tensors: &[HeaderTensor]) {
        let mut entries = serde_json::Map::new();
        let mut offset = 0_u64;
        for tensor in tensors {
            let end = offset
                .checked_add(tensor.data_bytes)
                .expect("fixture safetensors offset");
            entries.insert(
                tensor.name.clone(),
                serde_json::json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&entries).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
    }

    fn write_parts(
        root: &Path,
        config: &serde_json::Value,
        transformer: &[HeaderTensor],
        t5: &[HeaderTensor],
        vae: &[HeaderTensor],
    ) {
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(config).unwrap(),
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        write_headers(&root.join("model.safetensors"), transformer);
        write_headers(&root.join("t5_encoder.safetensors"), t5);
        write_headers(&root.join("vae.safetensors"), vae);
    }

    fn write_checkpoint(root: &Path, quant: Option<Quant>) {
        let (config, transformer, t5, vae) = checkpoint_parts(quant);
        write_parts(root, &config, &transformer, &t5, &vae);
    }

    fn spec(root: impl Into<PathBuf>) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(root.into()))
    }

    fn fixture_spec() -> LoadSpec {
        spec("/nonexistent-wan-memory-fixture").with_quant(Quant::Q4)
    }

    #[test]
    fn contract_declares_only_resident_and_bounded_decode_with_exact_realization() {
        let contract = weights_free_memory_strategy_contract(&fixture_spec()).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        for strategy in [MemoryStrategy::Resident, MemoryStrategy::BoundedDecode] {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented
            ));
        }
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing
            ));
        }
        assert!(contract.additional_prerequisites.is_empty());
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            [MemoryStrategy::Resident, MemoryStrategy::BoundedDecode]
        );
        assert_eq!(
            contract.resident_request_memory,
            ResidentRequestMemory::PreserveLoadDefaults
        );
        assert_eq!(
            contract.lifecycle.phases,
            [
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode
            ]
        );
        assert!(contract.lifecycle.synchronized_phase_release);
        assert!(contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: false,
            }
        ));
        for variable in [
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::FrameCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ] {
            assert!(contract.formula.uses(variable));
        }

        let tier = fixture_tier(&fixture_spec()).unwrap();
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, tier, false)
            .unwrap();
        let bounded = contract
            .representative_selection(MemoryStrategy::BoundedDecode, tier, false)
            .unwrap();
        assert_eq!(contract.generation_memory(&resident), None);
        assert_eq!(
            contract.generation_memory(&bounded),
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: bounded.parameters.decode_tile_edge,
                decode_overlap: bounded.parameters.decode_overlap,
                ..Default::default()
            })
        );
    }

    #[test]
    fn dense_q4_q8_headers_resolve_golden_assets_and_authoritative_tiers() {
        let expected = [
            (
                None,
                MemoryAssetFacts {
                    conditioning_bytes: 11_362_320_384,
                    transformer_bytes: 10_001_062_272,
                    decoder_bytes: 1_409_377_336,
                    base_bytes: 22_772_759_992,
                    overlay_bytes: 0,
                },
            ),
            (
                Some(Quant::Q4),
                MemoryAssetFacts {
                    conditioning_bytes: 7_021_215_744,
                    transformer_bytes: 2_946_767_232,
                    decoder_bytes: 1_409_377_336,
                    base_bytes: 11_377_360_312,
                    overlay_bytes: 0,
                },
            ),
            (
                Some(Quant::Q8),
                MemoryAssetFacts {
                    conditioning_bytes: 7_021_215_744,
                    transformer_bytes: 5_400_435_072,
                    decoder_bytes: 1_409_377_336,
                    base_bytes: 13_831_028_152,
                    overlay_bytes: 0,
                },
            ),
        ];
        for (quant, facts) in expected {
            let temp = tempfile::tempdir().unwrap();
            write_checkpoint(temp.path(), quant);
            // No requested override: config.json alone owns the tier.
            let load = spec(temp.path().to_path_buf());
            let tier = resolved_numeric_tier(&load).unwrap();
            assert_eq!(tier.quant, quant);
            assert_eq!(
                tier.component_precision_floors,
                if quant == Some(Quant::Q4) {
                    COMPONENT_PRECISION_FLOORS
                } else {
                    &[]
                }
            );
            let contract = memory_strategy_contract(&load).unwrap();
            assert_eq!(contract.asset_facts, facts);
            let fingerprint = &contract.calibration.as_ref().unwrap().fingerprint;
            match quant {
                None => assert!(fingerprint.contains("dense")),
                Some(Quant::Q4) => assert!(fingerprint.contains("q4-g64-teq8")),
                Some(Quant::Q8) => assert!(fingerprint.contains("q8-g64-teq8")),
                Some(Quant::Nvfp4) => unreachable!(),
            }

            // A supplied override is an assertion only: matching succeeds, conflicting fails.
            if let Some(quant) = quant {
                assert!(resolved_numeric_tier(&load.clone().with_quant(quant)).is_ok());
                let conflict = if quant == Quant::Q4 {
                    Quant::Q8
                } else {
                    Quant::Q4
                };
                assert!(resolved_numeric_tier(&load.clone().with_quant(conflict)).is_err());
            } else {
                assert!(resolved_numeric_tier(&load.clone().with_quant(Quant::Q4)).is_err());
            }
        }
    }

    #[test]
    fn malformed_or_partial_headers_and_config_fail_closed() {
        // Partial packed triple.
        let temp = tempfile::tempdir().unwrap();
        let (config, mut transformer, t5, vae) = checkpoint_parts(Some(Quant::Q4));
        transformer.retain(|tensor| tensor.name != "blocks.0.self_attn.q.biases");
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        // Wrong packed dtype.
        let temp = tempfile::tempdir().unwrap();
        let (config, mut transformer, t5, vae) = checkpoint_parts(Some(Quant::Q8));
        transformer
            .iter_mut()
            .find(|tensor| tensor.name == "blocks.0.self_attn.q.scales")
            .unwrap()
            .dtype = "F32";
        let scale = transformer
            .iter_mut()
            .find(|tensor| tensor.name == "blocks.0.self_attn.q.scales")
            .unwrap();
        scale.data_bytes *= 2;
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        // Non-canonical architecture and unsupported quantization manifest.
        for (field, value) in [
            ("dim", serde_json::json!(3073)),
            (
                "quantization",
                serde_json::json!({"bits": 3, "group_size": 64}),
            ),
            (
                "quantization",
                serde_json::json!({"bits": 4, "group_size": 32}),
            ),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let (mut config, transformer, t5, vae) = checkpoint_parts(None);
            config[field] = value;
            write_parts(temp.path(), &config, &transformer, &t5, &vae);
            assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());
        }
        let temp = tempfile::tempdir().unwrap();
        let (mut config, transformer, t5, vae) = checkpoint_parts(None);
        config.as_object_mut().unwrap().remove("t5_dim");
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (mut config, transformer, t5, vae) = checkpoint_parts(Some(Quant::Q4));
        config["quantization"] = serde_json::json!({"bits": 4});
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        // Missing component, unknown transformer tensor, and checked element-count overflow.
        let temp = tempfile::tempdir().unwrap();
        write_checkpoint(temp.path(), None);
        std::fs::remove_file(temp.path().join("vae.safetensors")).unwrap();
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.iter_mut()
            .find(|tensor| tensor.name == "decoder.conv1.weight")
            .unwrap()
            .dtype = "BF16";
        let decoder_conv = vae
            .iter_mut()
            .find(|tensor| tensor.name == "decoder.conv1.weight")
            .unwrap();
        decoder_conv.data_bytes /= 2;
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        // The VAE is tied to the complete loader topology: a missing decoder leaf, a wrong shape,
        // an arbitrary/orphan key, or a partial canonical TI2V encoder must all fail closed.
        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.retain(|tensor| tensor.name != "decoder.middle.1.proj_weight");
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        let time_conv = vae
            .iter_mut()
            .find(|tensor| tensor.name == "decoder.upsamples.0.upsamples.3.time_conv.weight")
            .unwrap();
        time_conv.shape[1] = 2;
        time_conv.data_bytes = time_conv
            .shape
            .iter()
            .try_fold(4_u64, |bytes, &dimension| {
                bytes.checked_mul(dimension as u64)
            })
            .unwrap();
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.push(HeaderTensor::new("decoder.arbitrary.weight", "F32", &[1]));
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.retain(|tensor| tensor.name != "encoder.head.layer_2.bias");
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        // A decoder-only z48 file is valid for the generic VAE builder, but it is not the canonical
        // TI2V-5B converter identity this provider contract calibrates: `convert_ti2v_5b` always
        // calls `convert_vae22(..., true)` and the route can encode references.
        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.retain(|tensor| {
            !matches!(tensor.name.as_str(), "conv1.weight" | "conv1.bias")
                && !tensor.name.starts_with("encoder.")
        });
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, mut vae) = checkpoint_parts(None);
        vae.push(HeaderTensor::new("encoder.orphan.weight", "F32", &[1]));
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, mut transformer, t5, vae) = checkpoint_parts(None);
        transformer.push(HeaderTensor::new("unused.weight", "BF16", &[1]));
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());

        let temp = tempfile::tempdir().unwrap();
        let (config, transformer, t5, _) = checkpoint_parts(None);
        let vae = vec![HeaderTensor {
            name: "overflow".into(),
            dtype: "F32",
            shape: vec![usize::MAX, 2],
            data_bytes: 0,
        }];
        write_parts(temp.path(), &config, &transformer, &t5, &vae);
        assert!(memory_strategy_contract(&spec(temp.path().to_path_buf())).is_err());
    }

    #[test]
    fn loaded_contract_matches_registry_and_unsupported_loads_publish_none() {
        let temp = tempfile::tempdir().unwrap();
        write_checkpoint(temp.path(), Some(Quant::Q8));
        let load_spec = spec(temp.path().to_path_buf());
        let registry = crate::provider_registry().unwrap();
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == MODEL_ID)
            .unwrap();
        let registered = (registration.contract)(&load_spec).unwrap();
        let loaded = registry.load(MODEL_ID, &load_spec).unwrap();
        assert_eq!(loaded.memory_strategy_contract(), Some(&registered));

        let mut sequential = load_spec.clone();
        sequential.offload_policy = OffloadPolicy::Sequential;
        let loaded = crate::model::load(&sequential).unwrap();
        assert!(loaded.memory_strategy_contract().is_none());
        assert!(matches!(
            loaded.memory_strategy_safety_check(&MemoryRunContext {
                // This probe asserts the Sequential route declares no contract at all, so the
                // authority value is inert here; Calibrated matches every other provider's
                // weights-free safety-check context.
                optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
                selection: gen_core::MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    parameters: MemoryStrategyParameters::default(),
                    tier: MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: Some(Quant::Q8),
                        component_precision_floors: &[],
                    },
                },
                calibration_abi: 0,
                calibration_fingerprint: String::new(),
                load_shape: LoadShape::EagerMaterialization,
                mode: MemoryMode::Other("text_to_video".into()),
                has_reference: false,
                use_pid: false,
                has_phases: false,
                geometry: MemoryGeometry {
                    width: 480,
                    height: 480,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                overlay: None,
                budget: gen_core::MemoryBudget {
                    total_bytes: 1,
                    committed_bytes: 0,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: 0,
                },
                predicted_peak_bytes: 0,
                cache_state: gen_core::MemoryCacheState::Cold,
                evidence_revision: String::new(),
            }),
            MemorySafetyDecision::Reject { .. }
        ));

        let root = temp.path().to_path_buf();
        let mut unsupported_loads = Vec::new();
        let mut mutated = load_spec.clone();
        mutated.weights = WeightsSource::File(root.join("model.safetensors"));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.precision = Precision::Fp32;
        unsupported_loads.push(mutated);
        unsupported_loads.push(sequential);
        let mut mutated = load_spec.clone();
        mutated.load_shape = LoadShape::DeferredMaterialization;
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.control = Some(WeightsSource::Dir(root.clone()));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated
            .extra_controls
            .push(WeightsSource::Dir(root.clone()));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.ip_adapter = Some(WeightsSource::Dir(root.clone()));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(root.clone()),
        });
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.identity = Some(IdentityWeights::default());
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.text_encoder = Some(WeightsSource::Dir(root.clone()));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated.adapters.push(AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        unsupported_loads.push(mutated);
        let mut mutated = load_spec.clone();
        mutated
            .components
            .insert("extra".into(), WeightsSource::Dir(root));
        unsupported_loads.push(mutated);
        for unsupported in unsupported_loads {
            assert!(
                contract_for_loaded(&unsupported).unwrap().is_none(),
                "every load axis outside the exact calibrated surface must publish no contract"
            );
        }

        let dense = tempfile::tempdir().unwrap();
        write_checkpoint(dense.path(), None);
        let runtime_quant = spec(dense.path().to_path_buf()).with_quant(Quant::Q4);
        let loaded = crate::model::load(&runtime_quant).unwrap();
        assert!(
            loaded.memory_strategy_contract().is_none(),
            "dense load-time packing remains supported but cannot borrow prepacked evidence"
        );
    }

    fn behavior_fixture() -> (LoadSpec, MemoryProviderContract, MemoryBehaviorFixture) {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .pop()
            .unwrap();
        (spec, contract, fixture)
    }

    #[test]
    fn safety_route_geometry_and_selected_decode_mutations_reject() {
        let (_spec, contract, fixture) = behavior_fixture();
        assert_eq!(
            safety_check(&contract, fixture.context.selection.tier, &fixture.context),
            MemorySafetyDecision::Accept
        );
        let rejects = |context: &MemoryRunContext| {
            assert!(matches!(
                safety_check(&contract, fixture.context.selection.tier, context),
                MemorySafetyDecision::Reject { .. }
            ));
        };

        let mut context = fixture.context.clone();
        context.mode = MemoryMode::TextToImage;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.has_reference = true;
        context.geometry.reference_count = 1;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.use_pid = true;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.has_phases = true;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.overlay = Some("synthetic".into());
        rejects(&context);
        let mut context = fixture.context.clone();
        context.calibration_abi += 1;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.calibration_fingerprint.push_str("-mismatch");
        rejects(&context);
        let mut context = fixture.context.clone();
        context.load_shape = LoadShape::DeferredMaterialization;
        rejects(&context);
        let mut context = fixture.context.clone();
        context.selection.tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        rejects(&context);
        for geometry in [
            MemoryGeometry {
                batch: 2,
                ..fixture.context.geometry
            },
            MemoryGeometry {
                width: 479,
                ..fixture.context.geometry
            },
            MemoryGeometry {
                width: 1280,
                height: 736,
                ..fixture.context.geometry
            },
            MemoryGeometry {
                frames: 2,
                ..fixture.context.geometry
            },
            MemoryGeometry {
                frames: 1029,
                ..fixture.context.geometry
            },
        ] {
            let mut context = fixture.context.clone();
            context.geometry = geometry;
            rejects(&context);
        }
        for parameters in [
            MemoryStrategyParameters {
                decode_tile_edge: Some(447),
                decode_overlap: Some(64),
                ..Default::default()
            },
            MemoryStrategyParameters {
                decode_tile_edge: Some(448),
                decode_overlap: Some(32),
                ..Default::default()
            },
        ] {
            let mut context = fixture.context.clone();
            context.selection.parameters = parameters;
            rejects(&context);
        }
    }

    fn assert_scope_rejects(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        fixture: &MemoryBehaviorFixture,
        mut request: GenerationRequest,
    ) {
        assert_eq!(
            request.memory, None,
            "rejected request started with a memory carrier"
        );
        let mut scope = registered_begin_request(spec, contract, &fixture.context)
            .unwrap()
            .unwrap();
        assert!(scope.configure_request(&mut request).is_err());
        assert_eq!(
            request.memory, None,
            "rejected request installed a memory carrier"
        );
        scope
            .finish(MemoryRunOutcome::Error {
                message: "expected mutation rejection".into(),
            })
            .unwrap();
    }

    #[test]
    fn request_scope_accepts_creative_knobs_and_rejects_every_uncalibrated_route() {
        let (spec, contract, fixture) = behavior_fixture();
        let default_frames = calibrated_default_frames().unwrap();
        assert_eq!(default_frames, 81);
        assert_eq!(fixture.context.geometry.frames, default_frames);
        for frames in [None, Some(default_frames)] {
            for fps in [None, Some(24), Some(25), Some(30)] {
                let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
                    .unwrap()
                    .unwrap();
                let mut request = GenerationRequest {
                    prompt: "creative prompt".into(),
                    negative_prompt: Some("creative negative".into()),
                    sampler: Some("euler".into()),
                    steps: Some(12),
                    guidance: Some(4.5),
                    seed: Some(7),
                    frames,
                    fps,
                    ..fixture.request.clone()
                };
                scope.configure_request(&mut request).unwrap();
                assert_eq!(
                    request.memory,
                    Some(GenerationMemory {
                        tile_vae_decode: true,
                        decode_tile_edge: Some(448),
                        decode_overlap: Some(64),
                        ..Default::default()
                    })
                );
                scope.finish(MemoryRunOutcome::Complete).unwrap();
                assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
            }
        }
        for frames in [1, 77, 85] {
            assert_scope_rejects(
                &spec,
                &contract,
                &fixture,
                GenerationRequest {
                    frames: Some(frames),
                    ..fixture.request.clone()
                },
            );
        }
        for fps in [23, 31] {
            assert_scope_rejects(
                &spec,
                &contract,
                &fixture,
                GenerationRequest {
                    fps: Some(fps),
                    ..fixture.request.clone()
                },
            );
        }

        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        let mut mutations = Vec::new();
        let mut request = fixture.request.clone();
        request.conditioning = vec![Conditioning::Reference {
            image,
            strength: None,
        }];
        mutations.push(request);
        macro_rules! mutation {
            ($field:ident = $value:expr) => {{
                let mut request = fixture.request.clone();
                request.$field = $value;
                mutations.push(request);
            }};
        }
        mutation!(strength = Some(0.5));
        mutation!(control_scale = Some(1.0));
        mutation!(text_style_gain = Some(1.0));
        mutation!(image_guidance = Some(1.0));
        mutation!(audio = Some(AudioParams::default()));
        mutation!(
            phases = Some(vec![GenerationPhase {
                steps: 1,
                ..Default::default()
            }])
        );
        mutation!(video_mode = Some("variant".into()));
        mutation!(trim_first_frames = Some(1));
        mutation!(duration = Some(1.0));
        mutation!(motion_bucket_id = Some(1.0));
        mutation!(noise_aug_strength = Some(0.1));
        mutation!(decode_chunk_size = Some(1));
        mutation!(conditioning_fps = Some(24));
        mutation!(softness = Some(0.1));
        mutation!(enhance_prompt = true);
        mutation!(use_uncensored_enhancer = true);
        mutation!(enhance_max_tokens = Some(1));
        mutation!(enhance_temperature = Some(0.1));
        mutation!(use_pid = true);
        mutation!(pid_capture_sigma = Some(0.5));
        for request in mutations {
            assert_scope_rejects(&spec, &contract, &fixture, request);
        }
    }

    #[test]
    fn selected_decode_parameters_reach_the_real_generation_carrier_exactly() {
        let request = GenerationRequest {
            width: 480,
            height: 480,
            frames: Some(1),
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(448),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tiling = decode_tiling(&request, 480, 480, 1).unwrap().unwrap();
        assert_eq!(
            tiling
                .spatial
                .map(|spatial| (spatial.tile_px, spatial.overlap_px)),
            Some((448, 64))
        );

        for memory in [
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(447),
                decode_overlap: Some(64),
                ..Default::default()
            },
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(448),
                decode_overlap: Some(64),
                stage_residency: true,
                ..Default::default()
            },
        ] {
            let mutated = GenerationRequest {
                memory: Some(memory),
                ..request.clone()
            };
            assert!(decode_tiling(&mutated, 480, 480, 1).is_err());
        }
    }

    #[test]
    fn explicit_registry_behavior_conforms_and_other_wan_routes_have_no_selected_profile() {
        let registry = crate::provider_registry().unwrap();
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &fixture_spec());
        for provider in [
            crate::MODEL_ID_T2V_14B,
            crate::MODEL_ID_I2V_14B,
            crate::MODEL_ID_VACE,
            crate::MODEL_ID_VACE_FUN,
        ] {
            assert_eq!(
                selected_video_decode_memory_profile(provider, 480, 480, 1, 448, 64).unwrap(),
                None
            );
            assert_eq!(
                crate::resolved_video_memory_numeric_tier(provider, &fixture_spec()).unwrap(),
                None
            );
        }
    }
}
