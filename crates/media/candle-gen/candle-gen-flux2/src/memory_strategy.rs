//! Candle/CUDA FLUX.2 adoption of the shared image-memory ladder (SC-15833, SC-15831).
//!
//! Dev and Klein deliberately share lifecycle and execution primitives while retaining distinct
//! provider identities, block domains, candidate ranges, and calibration fingerprints. The three
//! SceneWorks Klein catalog entries resolve to the one `flux2_klein_9b` Candle provider; entry-level
//! tier/mode/overlay measurements remain catalog-owned and cannot be inferred from this contract.

use crate::config::{Flux2Variant, FLUX2_DEV_ID, FLUX2_KLEIN_9B_ID};
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
    MemoryWindowMaterialization, PerComponentBytes, Precision, Quant, TransformerComponent,
    WeightsSource,
};
use std::sync::{Arc, Mutex};

fn selected_encoder_discovery_roots(
    source: &WeightsSource,
) -> gen_core::Result<Vec<std::path::PathBuf>> {
    let root = match source {
        WeightsSource::Dir(path) => path.as_path(),
        WeightsSource::File(path) => path.parent().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "selected text encoder file has no parent directory: {}",
                path.display()
            ))
        })?,
    };
    let mut roots = vec![std::path::absolute(root)?, std::fs::canonicalize(root)?];
    roots.sort();
    roots.dedup();
    Ok(roots)
}

/// Full output edge used by the bounded-decode hook at the representative 1024px calibration cell.
/// This intentionally does not spatially partition that cell: the saving comes from separating the
/// full-image attention-bearing head from the upsampling tail's live envelope, while preserving a
/// near-monolithic numerical path.
pub const DECODE_TILE_EDGE: u32 = 1024;
pub const DECODE_TILE_EDGES: &[u32] = &[DECODE_TILE_EDGE];
/// The shared contract requires a positive overlap domain. At the 1024px full-edge calibration cell
/// no neighboring tiles exist, so this value is an inert, exactly keyed sentinel rather than a claim
/// that spatial blending occurred.
pub const DECODE_OVERLAP: u32 = 1;
pub const DECODE_OVERLAPS: &[u32] = &[DECODE_OVERLAP];
pub const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub const BASE_DOUBLE_BLOCKS: u32 = 8;
pub const BASE_SINGLE_BLOCKS: u32 = 48;
pub const BASE_TRANSFORMER_BLOCKS: u32 = BASE_DOUBLE_BLOCKS + BASE_SINGLE_BLOCKS;
pub const CONTROL_BLOCKS: u32 = 4;
pub const CALIBRATION_FINGERPRINT: &str =
    "flux2-dev-cuda-caption-upsample-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v3";
pub const CONTROL_OVERLAY: &str = "control";

/// Full output edge at the representative 1024px cell. The FLUX.2 upsampling tail contains spatial
/// GroupNorms, so splitting it into smaller tiles changes normalization statistics across most of
/// the image; the real-weight A/B rejected the former 768/640/512 domain at max-abs 14 against the
/// provider's <=2 RGB contract. Keep the same parity-preserving full-edge realization already used
/// by FLUX.2-dev until a genuinely normalization-aware tiled decoder is implemented and measured.
pub const KLEIN_DECODE_TILE_EDGE: u32 = 1024;
pub const KLEIN_DECODE_TILE_EDGES: &[u32] = &[KLEIN_DECODE_TILE_EDGE];
/// Positive sentinel required by the shared contract. It is inert at the full-edge 1024px cell.
pub const KLEIN_DECODE_OVERLAP: u32 = 1;
pub const KLEIN_DECODE_OVERLAPS: &[u32] = &[KLEIN_DECODE_OVERLAP];
pub const KLEIN_BASE_DOUBLE_BLOCKS: u32 = 8;
pub const KLEIN_BASE_SINGLE_BLOCKS: u32 = 24;
pub const KLEIN_BASE_TRANSFORMER_BLOCKS: u32 = KLEIN_BASE_DOUBLE_BLOCKS + KLEIN_BASE_SINGLE_BLOCKS;
pub const KLEIN_CALIBRATION_FINGERPRINT: &str = "flux2-klein-cuda-shared-ladder-provider-abi-v2";

#[derive(Clone, Copy)]
struct ProviderProfile {
    provider_id: &'static str,
    decode_tile_edges: &'static [u32],
    decode_overlaps: &'static [u32],
    base_transformer_blocks: u32,
    calibration_fingerprint: &'static str,
}

fn profile(provider_id: &str) -> gen_core::Result<ProviderProfile> {
    match provider_id {
        FLUX2_DEV_ID => Ok(ProviderProfile {
            provider_id: FLUX2_DEV_ID,
            decode_tile_edges: DECODE_TILE_EDGES,
            decode_overlaps: DECODE_OVERLAPS,
            base_transformer_blocks: BASE_TRANSFORMER_BLOCKS,
            calibration_fingerprint: CALIBRATION_FINGERPRINT,
        }),
        FLUX2_KLEIN_9B_ID => Ok(ProviderProfile {
            provider_id: FLUX2_KLEIN_9B_ID,
            decode_tile_edges: KLEIN_DECODE_TILE_EDGES,
            decode_overlaps: KLEIN_DECODE_OVERLAPS,
            base_transformer_blocks: KLEIN_BASE_TRANSFORMER_BLOCKS,
            calibration_fingerprint: KLEIN_CALIBRATION_FINGERPRINT,
        }),
        _ => Err(gen_core::Error::Unsupported(format!(
            "unknown FLUX.2 memory provider {provider_id}"
        ))),
    }
}

fn streamable(spec: &LoadSpec) -> bool {
    // File and Dir intentionally share this provider/calibration identity: their executable phase
    // graph and output semantics are the same. The evidence matrix has no load-source axis, however,
    // so a Dir-measured rung-4 cell cannot be claimed for File. Keep imported File rung 4 Missing
    // until its pinned/re-openable implementation is independently measured.
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
}

fn ggml_projection_bytes(
    tensor: &gen_core::weightsmeta::SafetensorsTensorHeader,
    quant: Quant,
    component: &str,
) -> gen_core::Result<u64> {
    let [out, input] = tensor.shape.as_slice() else {
        return tensor.materialized_bytes(4);
    };
    if !input.is_multiple_of(32) {
        return Err(gen_core::Error::Unsupported(format!(
            "FLUX.2 {component} projection {:?} has input width {input}, which cannot be folded to {quant:?}",
            tensor.name
        )));
    }
    let elements = u64::try_from(*out)
        .ok()
        .and_then(|out| {
            u64::try_from(*input)
                .ok()
                .and_then(|input| out.checked_mul(input))
        })
        .ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "FLUX.2 {component} projection {:?} element count overflow",
                tensor.name
            ))
        })?;
    let bytes_per_block = match quant {
        // Candle's load-time fold uses native GGML Q4_0/Q8_0: 32 values plus one f16 scale.
        Quant::Q4 => 18_u64,
        Quant::Q8 => 34_u64,
        Quant::Nvfp4 => {
            return Err(gen_core::Error::Unsupported(
                "FLUX.2 File imports do not support NVFP4 folding".into(),
            ))
        }
    };
    (elements / 32).checked_mul(bytes_per_block).ok_or_else(|| {
        gen_core::Error::Msg(format!(
            "FLUX.2 {component} projection {:?} packed byte size overflow",
            tensor.name
        ))
    })
}

/// The compute dtype every FLUX.2 candle component materializes at (`Pipeline::dtype` is
/// `DType::F32`; the MMDiT math is parity-sensitive). Dense residency is therefore priced at 4
/// bytes per **logical** element, NOT at the codec's bf16 resident encoding — `PlannedDitWeights`
/// casts each decoded dense tensor to this dtype before it lands on the device.
const COMPUTE_DTYPE_BYTES: u64 = 4;

/// The same compute dtype as a `DType`, so the published activation width and
/// [`COMPUTE_DTYPE_BYTES`] cannot drift apart: `lib.rs` pins `dtype: DType::F32` for every FLUX.2
/// component.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::F32;

/// The residency policy the klein fit gate prices against: the SAME probe
/// [`Flux2Pipeline::load_klein_planned_dit`](Pipeline::load_klein_planned_dit) runs, on the same
/// process-default device the concrete loader selects
/// ([`candle_gen::default_device`], `load_variant_concrete`). Pricing and loading must not disagree
/// about whether an NVFP4 row stays packed — that disagreement IS the fit gate falsely rejecting or
/// falsely admitting.
///
/// A device that will not construct is priced dense. That is the conservative direction (dense is
/// the larger of the two residencies) and it is also the truth: a loader that cannot build the
/// device cannot take the native leg either.
fn klein_pricing_residency() -> candle_gen::logical_weights::CandleCodecResidency {
    match candle_gen::default_device() {
        // The SAME policy the loader plans under — including the MAJOR 10 fp8 mask (sc-11045 fix
        // round): one definition, so pricing and loading cannot drift.
        Ok(device) => crate::single_file::klein_import_residency(&device),
        Err(_) => candle_gen::logical_weights::CandleCodecResidency::DENSE,
    }
}

/// Price an imported klein BFL/NVFP4 single-file DiT from the **compiled plan** — the same plan the
/// loader consumes (sc-21485 review, blocker).
///
/// # Why the header sum cannot do this
///
/// `f32_or_packed_tensor_headers` prices from on-disk tensor headers, and an NVFP4 layer's header
/// is a `U8 [rows, cols / 2]` nibble matrix whose `weight_scale` companion looks like a
/// source-only scale. Charging it `materialized_bytes(4)` costs 2 bytes per *logical* element and
/// drops the block scales entirely, which is wrong in both directions and by different factors:
///
/// * on `sm_120` the rows stay [`ResidencyMode::Packed`] — nibbles plus both scale levels, roughly
///   0.56 bytes per logical element — so the header sum over-prices by ~3.5x and the gate falsely
///   rejects;
/// * below the floor the rows fall to dense `F32` — 4 bytes per logical element — so the header sum
///   under-prices by exactly 2x and the gate falsely admits.
///
/// The plan already knows both, per tensor, from the codec's own geometry. Packed rows take the
/// plan's `residency.resident_bytes` (stored nibbles, already sliced for a transformed output) plus
/// the retained companions' bytes; dense rows are re-priced at [`COMPUTE_DTYPE_BYTES`] over the
/// **logical** shape, because this pipeline casts every dense decode to F32 rather than keeping the
/// codec's bf16 resident encoding.
/// `cfg` is the architecture whose true geometry the mapping declares — always
/// `Flux2Variant::Klein9b.config()` in production; a parameter only so the unit test can pin the
/// byte totals at a fixture width instead of the 9B one.
fn klein_planned_dit_bytes(
    dit_path: &std::path::Path,
    cfg: &crate::config::Flux2Config,
    residency: &dyn gen_core::checkpoint_codec::CodecResidencyPolicy,
) -> gen_core::Result<u64> {
    use gen_core::checkpoint_codec::ResidencyMode;

    let mapping = crate::single_file::Flux2BflToDiffusersMapping::new(cfg);
    let plan = candle_gen::logical_weights::plan_logical_weights(dit_path, &mapping, residency)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
    let overflow = || gen_core::Error::Msg("FLUX.2 imported DiT resident byte sum overflow".into());
    // Pricing and loading must not disagree about what a row holds resident, and the loader's
    // decision is not the plan's residency alone: the provider role table sends the outlier class
    // of `Packed`-priced rows to **W4A16**, which dequantizes once at construction and holds the
    // full dense **bf16** weight (2 B per logical element — `Nvfp4Linear::new_dequant` stores
    // BF16, unlike the plan-dense rows this pipeline casts to F32). Pricing those rows at packed
    // bytes under-priced the pinned klein artifact by ~0.94 GB and falsely admitted it
    // (sc-11045 feature-end review, BLOCKER 4). The same role table the loader consults prices
    // here, so the two cannot drift without failing the loader's representation cross-check.
    let roles = crate::nvfp4_roles::KleinRoleTable::new(cfg);
    const W4A16_RESIDENT_BYTES_PER_ELEMENT: u64 = 2;
    let stays_packed = |logical_key: &str| -> bool {
        let base = logical_key.strip_suffix(".weight").unwrap_or(logical_key);
        roles.execution_role(base).is_packed_w4a4()
    };
    let mut total = 0_u64;
    // Physical keys with at least one logical output that genuinely stays packed W4A4: only those
    // owners retain their scale companions (a role-dense W4A16 construction consumes them in the
    // one-time dequant).
    let mut packed_owner: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for tensor in &plan.tensors {
        let bytes = match tensor.residency.mode {
            ResidencyMode::Packed if stays_packed(&tensor.logical_key) => {
                packed_owner.insert(tensor.physical_key.as_str());
                tensor.residency.resident_bytes
            }
            // A Packed-priced row the role table serves W4A16: resident dense bf16.
            ResidencyMode::Packed => tensor
                .shape
                .iter()
                .try_fold(W4A16_RESIDENT_BYTES_PER_ELEMENT, |acc: u64, dim| {
                    acc.checked_mul(*dim as u64)
                })
                .ok_or_else(overflow)?,
            ResidencyMode::Dense => tensor
                .shape
                .iter()
                .try_fold(COMPUTE_DTYPE_BYTES, |acc: u64, dim| {
                    acc.checked_mul(*dim as u64)
                })
                .ok_or_else(overflow)?,
        };
        total = total.checked_add(bytes).ok_or_else(overflow)?;
    }
    for companion in &plan.companions {
        if companion.resident_bytes > 0
            && !packed_owner.contains(companion.owner_physical_key.as_str())
        {
            // Retained-scale pricing for an owner whose every output is role-dense: the W4A16
            // dequant consumes the scales, so nothing stays resident.
            continue;
        }
        total = total
            .checked_add(companion.resident_bytes)
            .ok_or_else(overflow)?;
    }
    Ok(total)
}

fn f32_or_packed_component_bytes(
    path: &std::path::Path,
    quant: Option<Quant>,
    component: &str,
    keep_embedding_dense: bool,
    inline_fp8_scales: bool,
) -> gen_core::Result<u64> {
    let tensors = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    f32_or_packed_tensor_headers(
        &tensors,
        quant,
        component,
        keep_embedding_dense,
        inline_fp8_scales,
        &path.display().to_string(),
    )
}

fn f32_or_packed_tensor_headers(
    tensors: &[gen_core::weightsmeta::SafetensorsTensorHeader],
    quant: Option<Quant>,
    component: &str,
    keep_embedding_dense: bool,
    inline_fp8_scales: bool,
    source: &str,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;
    use std::collections::HashMap;

    if tensors.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "FLUX.2 {component} '{source}' contains no tensors"
        )));
    }
    let by_name: HashMap<&str, &gen_core::weightsmeta::SafetensorsTensorHeader> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let packed_bases = tensors
        .iter()
        .filter_map(|tensor| tensor.name.strip_suffix(".scales"))
        .collect::<std::collections::HashSet<_>>();
    tensors.iter().try_fold(0_u64, |total, tensor| {
        let source_only =
            tensor.name.ends_with(".weight_scale") || tensor.name.ends_with(".input_scale");
        if source_only {
            return Ok(total);
        }

        if tensor
            .name
            .strip_suffix(".scales")
            .or_else(|| tensor.name.strip_suffix(".biases"))
            .is_some_and(|base| packed_bases.contains(base))
        {
            return Ok(total);
        }

        if let Some(base) = tensor
            .name
            .strip_suffix(".weight")
            .filter(|base| packed_bases.contains(base))
        {
            let scales_name = format!("{base}.scales");
            let biases_name = format!("{base}.biases");
            let scales = by_name.get(scales_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} packed weight {:?} is missing {scales_name:?}",
                    tensor.name
                ))
            })?;
            let biases = by_name.get(biases_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} packed weight {:?} is missing {biases_name:?}",
                    tensor.name
                ))
            })?;
            let loaded = candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                tensor,
                scales,
                biases,
                candle_gen::quant::MLX_GROUP_SIZE,
            )?;
            return total.checked_add(loaded).ok_or_else(|| {
                gen_core::Error::Msg(format!("FLUX.2 {component} resident byte sum overflow"))
            });
        }

        if inline_fp8_scales && tensor.dtype == Dtype::F8_E4M3 {
            let Some(base) = tensor.name.strip_suffix(".weight") else {
                return Err(gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} fp8 tensor {:?} is not a projection weight",
                    tensor.name
                )));
            };
            let scale_name = format!("{base}.weight_scale");
            let scale = by_name.get(scale_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} fp8 weight {:?} is missing {scale_name:?}",
                    tensor.name
                ))
            })?;
            if scale.element_count()? == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} scale {scale_name:?} is empty"
                )));
            }
        }

        let loaded = match tensor.dtype {
            // Both the imported map (`build_comfyui_dit_map`) and snapshot VarBuilders request the
            // pipeline's F32 dtype. Accepted integer tensors are therefore cast to F32 too; a rank-2
            // projection then takes the same GGML fold as a floating source when Q4/Q8 is selected.
            Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::F8_E4M3
            | Dtype::F16
            | Dtype::BF16
            | Dtype::F32
            | Dtype::F64 => {
                if let Some(quant) = quant.filter(|_| {
                    tensor.name.ends_with(".weight")
                        && tensor.shape.len() == 2
                        && !(keep_embedding_dense && tensor.name.ends_with("embed_tokens.weight"))
                }) {
                    ggml_projection_bytes(tensor, quant, component)?
                } else {
                    tensor.materialized_bytes(4)?
                }
            }
            dtype => {
                return Err(gen_core::Error::Unsupported(format!(
                    "FLUX.2 {component} tensor {:?} uses unsupported Candle dtype {dtype:?}",
                    tensor.name
                )))
            }
        };
        total.checked_add(loaded).ok_or_else(|| {
            gen_core::Error::Msg(format!("FLUX.2 {component} resident byte sum overflow"))
        })
    })
}

fn resident_components(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<Vec<MemoryResidentComponent>> {
    let mut out = Vec::new();
    if provider_id == FLUX2_DEV_ID {
        if let Some(control) = spec.control.as_ref() {
            let resident_bytes = match control {
                WeightsSource::Dir(path) => gen_core::weightsmeta::safetensors_path_bytes(path),
                WeightsSource::File(path) => {
                    spec.read_file_unchanged_if_prepared(path, |p| -> gen_core::Result<u64> {
                        Ok(gen_core::weightsmeta::safetensors_path_bytes(p))
                    })?
                }
            };
            if resident_bytes > 0 {
                out.push(MemoryResidentComponent {
                    id: "flux2_dev_fun_controlnet_union".to_owned(),
                    kind: MemoryComponentKind::ControlBranch,
                    resident_bytes,
                    // SC-15833 windows the 56-block base. The four overlay blocks remain resident and
                    // are therefore charged explicitly rather than hidden inside the base estimate.
                    bounded_by: None,
                    residency: MemoryComponentResidency::WholeRender,
                });
            }
        }
    }
    Ok(out)
}

/// Architecture axes for one FLUX.2 variant (epic SC-22657, E2).
///
/// The loader never parses `transformer/config.json`: the transformer is constructed from the
/// hardcoded [`crate::config::Flux2Config`] the variant selects, so those same fields are what the
/// built model actually has. Selecting the variant here mirrors the loader's own selection, which
/// is why dev (48 heads, 8 + 48 blocks) and klein-9b (32 heads, 8 + 24 blocks) publish different
/// geometry from one function.
///
/// `transformer_blocks` is the **total** trunk depth: FLUX.2 stacks the double-stream blocks and
/// then the single-stream blocks in one sequence, and every one of them is a block-window
/// materialization unit. `patch_size` has no config field of its own, so it is *derived* from the
/// pair that implies it — `in_channels 128 / num_latent_channels 32 = 4`, a 2x2 neighbourhood —
/// through [`candle_gen::architecture_facts::patch_size_from_channels`], rather than written as a
/// literal that would keep saying 2 if a variant ever repacked. `vae_temporal_scale` stays `None`: the FLUX.2
/// `AutoencoderKLFlux2` is an **image** autoencoder with no temporal axis, and a structurally
/// absent axis is declared absent rather than zero (E2).
fn architecture_facts(variant: Flux2Variant, spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    // Weights-free contract surfaces name a sentinel path that is deliberately not on disk: no
    // pipeline has been resolved there, so no axis is knowable and every one stays `None`.
    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    let config = variant.config();
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(config.num_heads),
        head_dim: af::declared(config.head_dim),
        transformer_blocks: af::declared(config.num_double_layers + config.num_single_layers),
        // The packing edge the trunk's own input width implies: `in_channels / num_latent_channels`
        // is the neighbourhood area, and its square root is the axis.
        patch_size: af::patch_size_from_channels(
            af::declared(config.in_channels),
            af::declared(config.num_latent_channels),
        ),
        latent_channels: af::declared(config.num_latent_channels),
        vae_spatial_scale: af::declared(config.vae_scale_factor),
        // Structurally absent: the FLUX.2 image autoencoder has no frames-per-latent axis.
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

pub(crate) fn composed_provider_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let profile = profile(provider_id)?;
    // Klein accepts imported single-file weights since sc-21485 (the universal BFL/NVFP4 single
    // file); the File pricing arm below covers both variants. The remaining klein File refusal —
    // a load-time Q4/Q8 fold over the pre-quantized source — lives in `validate_load_spec`, which
    // `provider_contract_for` already routes through, so the loader and this contract refuse the
    // same specs with the same message.
    let streamable = streamable(spec);
    let mut components = match &spec.weights {
        WeightsSource::Dir(_) => PerComponentBytes::from_spec_subdirs(
            spec,
            &["text_encoder"],
            &["transformer"],
            &["vae"],
        )
        .unwrap_or_default(),
        WeightsSource::File(dit) => {
            let base = gen_core::require_base_snapshot(spec, provider_id)?;
            let quant = resolved_quant(spec)?;
            PerComponentBytes {
                text_encoder: f32_or_packed_component_bytes(
                    &base.join("text_encoder"),
                    quant,
                    "base text encoder",
                    true,
                    false,
                )?,
                dit: spec.read_file_unchanged_if_prepared(dit, |p| {
                    if provider_id == FLUX2_KLEIN_9B_ID {
                        // The klein universal single file is consumed through the shared
                        // logical-weight plan, so it is PRICED from that plan — see
                        // `klein_planned_dit_bytes` for why the header sum cannot express NVFP4.
                        // `quant` is always `None` here: `validate_load_spec` refuses a Q4/Q8 fold
                        // over a pre-quantized klein source before this contract is composed.
                        klein_planned_dit_bytes(
                            p,
                            &Flux2Variant::Klein9b.config(),
                            &klein_pricing_residency(),
                        )
                    } else {
                        f32_or_packed_component_bytes(p, quant, "imported DiT", false, true)
                    }
                })?,
                vae: f32_or_packed_component_bytes(
                    &base.join("vae"),
                    None,
                    "base VAE",
                    false,
                    false,
                )?,
            }
        }
    };
    // An explicit encoder is a load-bearing authored selection, so price the same route-specific,
    // contract-validated tensor surface the concrete loader materializes. Raw direct-shard sums can
    // include a complete alternate snapshot's unused visual tower, unloaded decoder tail, or other
    // unrelated tensors and would make the fit gate disagree with the admitted runtime.
    let base = gen_core::require_base_snapshot(spec, provider_id)?;
    // The same variant selection the loader makes: it picks the `Flux2Config` the transformer is
    // built from, so it also picks the geometry this contract is allowed to publish.
    let variant = match provider_id {
        FLUX2_DEV_ID => Flux2Variant::Dev,
        FLUX2_KLEIN_9B_ID => Flux2Variant::Klein9b,
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "unknown FLUX.2 memory provider {provider_id}"
            )))
        }
    };
    let has_authored_encoder =
        spec.text_encoder.is_some() || base.join("text_encoder/config.json").is_file();
    if has_authored_encoder {
        let builtin = WeightsSource::Dir(base.join("text_encoder"));
        let source = spec.text_encoder.as_ref().unwrap_or(&builtin);
        let roots = selected_encoder_discovery_roots(source)?;
        let facts = variant
            .encoder_contract()
            .validate_source_for_discovery(source, &roots)?;
        let headers = facts.materialized_language_tensor_headers();
        let text_encoder_quant = (provider_id == FLUX2_DEV_ID)
            .then(|| resolved_quant(spec))
            .transpose()?
            .flatten();
        components.text_encoder = f32_or_packed_tensor_headers(
            headers,
            text_encoder_quant,
            "selected text encoder",
            true,
            false,
            "selected direct-shard inventory",
        )?;
    }
    let resident_components = resident_components(provider_id, spec)?;
    let overlay_bytes = resident_components
        .iter()
        .map(|component| component.resident_bytes)
        .sum();
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::BoundedTransformerResidency && !streamable {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: profile.decode_tile_edges.to_vec(),
                    decode_overlaps: profile.decode_overlaps.to_vec(),
                    ..Default::default()
                },
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
        architecture_facts: architecture_facts(variant, spec),
        provider_id: profile.provider_id.to_owned(),
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
        // This provider's constrained implementations load request-scoped phases. The explicit
        // edge is realization-owned, not an assumption made by the shared ladder.
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
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
            // The hook is implemented, but each provider's sole 1024px production candidate is
            // full-edge and therefore does not spatially partition the representative cell.
            decode_tiling: true,
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
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components,
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            profile.calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub fn provider_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let variant = match provider_id {
        FLUX2_DEV_ID => Flux2Variant::Dev,
        FLUX2_KLEIN_9B_ID => Flux2Variant::Klein9b,
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "unknown FLUX.2 memory provider {provider_id}"
            )))
        }
    };
    crate::validate_load_spec(variant, spec)?;
    composed_provider_contract_for(provider_id, spec)
}

pub fn provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(FLUX2_DEV_ID, spec)
}

pub fn klein_provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(FLUX2_KLEIN_9B_ID, spec)
}

pub fn contract_for_variant(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(variant.id(), spec)
}

fn packed_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    // Imported ComfyUI File weights carry their own fp8 representation and may request the same
    // explicit on-the-fly Q4/Q8 fold as the historical shim. They do not have a snapshot transformer
    // config from which a packed tier could be inferred.
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(None);
    };
    let config = root.join("transformer/config.json");
    let packed = match std::fs::read_to_string(&config) {
        Ok(text) => {
            let value = serde_json::from_str::<serde_json::Value>(&text).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "flux2_dev: parse {}: {error}",
                    config.display()
                ))
            })?;
            candle_gen::quant::PackedConfig::from_config(&value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: read {}: {error}",
                config.display()
            )))
        }
    };
    packed
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

pub fn resolved_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    let packed = packed_quant(spec)?;
    match (spec.quantize, packed) {
        (Some(requested), Some(stored)) if requested != stored => {
            Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: requested {requested:?} but snapshot stores {stored:?}"
            )))
        }
        (Some(requested), _) => Ok(Some(requested)),
        (None, stored) => Ok(stored),
    }
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: resolved_quant(spec)?,
        component_precision_floors: &[],
    })
}

fn route_is_supported(provider_id: &str, context: &MemoryRunContext) -> bool {
    match (
        &context.mode,
        context.geometry.reference_count,
        context.overlay.as_deref(),
    ) {
        (MemoryMode::TextToImage, 0, None) => true,
        (MemoryMode::Edit, 1..=8, None) => true,
        (MemoryMode::Other(mode), 1..=8, None)
            if mode == "character_image" || mode == "style_variations" =>
        {
            true
        }
        (MemoryMode::TextToImage, 0, Some(CONTROL_OVERLAY)) if provider_id == FLUX2_DEV_ID => true,
        _ => false,
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
            component_precision_floors: &[],
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
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: PiD cannot consume the native FLUX.2 VAE memory selection",
            contract.provider_id
        )));
    }
    if context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: optimized memory strategies do not cover multi-phase denoise",
            contract.provider_id
        )));
    }
    Ok(())
}

pub fn admission_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(contract, context, loaded_quant) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn validate_registered_generator_context(
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if context.mode != MemoryMode::TextToImage
        || context.geometry.reference_count != 0
        || context.has_reference
        || context.overlay.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "flux2_dev: registered generator admits text-to-image without references or overlays only"
                .to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Flux2RequestBinding {
    request_address: usize,
    geometry: MemoryGeometry,
    use_pid: bool,
    has_phases: bool,
    memory: Option<GenerationMemory>,
}

impl Flux2RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        Self {
            request_address: std::ptr::from_ref(request).addr(),
            geometry: MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(1),
                reference_count: request.image_reference_count(),
            },
            use_pid: request.use_pid,
            has_phases: request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty()),
            memory: request.memory,
        }
    }
}

struct Flux2ActiveAdmission {
    token: u64,
    context: MemoryRunContext,
    expected_memory: Option<GenerationMemory>,
    binding: Option<Flux2RequestBinding>,
    consumed: bool,
}

#[derive(Default)]
struct Flux2AdmissionState {
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<Flux2ActiveAdmission>,
}

/// Provider-local, one-shot authorization joining `begin`/`configure` to the exact request object
/// later passed to `generate`. The opaque token never enters `GenerationRequest`, so cloning or
/// copying its memory knobs cannot transfer authorization to another request.
#[derive(Clone)]
pub(crate) struct Flux2AdmissionRegistry {
    provider_id: &'static str,
    inner: Arc<Mutex<Flux2AdmissionState>>,
}

impl Flux2AdmissionRegistry {
    pub(crate) fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: Arc::new(Mutex::new(Flux2AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: cannot replace safety approval while a request scope is active",
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
                "{}: another memory request scope is already active",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request begin skipped the safety handshake",
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
        state.active = Some(Flux2ActiveAdmission {
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
        if active.token != token || active.binding.is_some() || active.consumed {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale, reused, or already-configured memory token",
                self.provider_id
            )));
        }
        let binding = Flux2RequestBinding::from_request(request);
        if binding.geometry != active.context.geometry
            || binding.use_pid != active.context.use_pid
            || binding.has_phases != active.context.has_phases
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed while configuring memory admission",
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
                    "{}: constrained memory request has no active admission token",
                    self.provider_id
                )))
            } else {
                Ok(())
            };
        };
        let binding = active.binding.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: active memory request was not configured",
                self.provider_id
            ))
        })?;
        if binding != &Flux2RequestBinding::from_request(request) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request or memory strategy changed after admission",
                self.provider_id
            )));
        }
        if active.consumed {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory admission token was already consumed",
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

pub struct Flux2MemoryScope {
    device: Device,
    provider_id: String,
    decode_tile_edges: Vec<u32>,
    decode_overlaps: Vec<u32>,
    attention_chunk_sizes: Vec<u32>,
    base_transformer_blocks: u32,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    has_phases: bool,
    admission: Option<Flux2AdmissionRegistry>,
    token: Option<u64>,
    finished: bool,
}

impl Flux2MemoryScope {
    pub fn new(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> Self {
        let decode = contract
            .capability(MemoryStrategy::BoundedDecode)
            .expect("FLUX.2 contract publishes bounded decode");
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .expect("FLUX.2 contract publishes bounded attention");
        Self {
            device,
            provider_id: contract.provider_id.clone(),
            decode_tile_edges: decode.parameters.decode_tile_edges.clone(),
            decode_overlaps: decode.parameters.decode_overlaps.clone(),
            attention_chunk_sizes: attention.parameters.attention_chunk_sizes.clone(),
            base_transformer_blocks: profile(&contract.provider_id)
                .expect("validated FLUX.2 provider contract")
                .base_transformer_blocks,
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            use_pid: context.use_pid,
            has_phases: context.has_phases,
            admission: None,
            token: None,
            finished: false,
        }
    }

    pub(crate) fn new_bound(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: Flux2AdmissionRegistry,
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
                "{}: memory request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }
}

impl MemoryRequestScope for Flux2MemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.active()?;
        if request.use_pid != self.use_pid
            || request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count != self.geometry.batch
            || request.image_reference_count() != self.geometry.reference_count
            || request.frames.unwrap_or(1) != self.geometry.frames
            || request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty())
                != self.has_phases
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request route or geometry changed after admission",
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
        if geometry != self.geometry {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: decode geometry changed after admission",
                self.provider_id
            )));
        }
        if self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: PiD has no admitted FLUX.2 VAE decode plan",
                self.provider_id
            )));
        }
        if self.decode_tile_edges.contains(&tile_edge) && self.decode_overlaps.contains(&overlap) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: decode does not publish {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.active()?;
        if self.attention_chunk_sizes.contains(&chunk_size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk size is not in {:?}, got {chunk_size}",
                self.provider_id, self.attention_chunk_sizes
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
                "{}: bounded transformer residency was not selected",
                self.provider_id
            )));
        };
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        if first_block >= self.base_transformer_blocks {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer window starts past the {}-block base",
                self.provider_id, self.base_transformer_blocks
            )));
        }
        let expected = window.min(self.base_transformer_blocks - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
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

impl Drop for Flux2MemoryScope {
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

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_quant(spec) {
        Ok(quant) => admission_safety_check(contract, context, quant),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
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
    let tier = resolved_numeric_tier(spec)?;
    let routes = [
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Other("character_image".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Other("style_variations".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    ];
    routes
        .into_iter()
        .map(|route| {
            gen_core::standard_memory_behavior_context(contract, strategy, tier, route)
                .map(gen_core::MemoryBehaviorFixture::new)
        })
        .collect()
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(contract, context, resolved_quant(spec)?)?;
    Ok(Some(Box::new(Flux2MemoryScope::new(
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spec() -> LoadSpec {
        let mut spec =
            LoadSpec::new(WeightsSource::Dir(PathBuf::from("/flux2-dev"))).with_quant(Quant::Q4);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    /// AC (epic SC-22657, E2): each FLUX.2 provider publishes the geometry of the `Flux2Config`
    /// its own loader builds — dev and klein-9b differ, and neither is inferred from a config the
    /// loader ignores — the contract passes the shared facts conformance check, and the weights-free
    /// surface (whose sentinel root is not on disk) publishes nothing at all.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let tmp = tempfile::tempdir().unwrap();
        let mut snapshot = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()));
        snapshot.load_shape = LoadShape::DeferredMaterialization;
        for (id, attention_heads, transformer_blocks) in [
            // `Flux2Config::dev()`: 48 heads, 8 double + 48 single blocks.
            (FLUX2_DEV_ID, 48, 8 + 48),
            // `Flux2Config::klein_9b()`: 32 heads, 8 double + 24 single blocks.
            (FLUX2_KLEIN_9B_ID, 32, 8 + 24),
        ] {
            let contract = composed_provider_contract_for(id, &snapshot).unwrap();
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(attention_heads),
                    // `Flux2Config::head_dim` — constant across variants.
                    head_dim: Some(128),
                    transformer_blocks: Some(transformer_blocks),
                    // The 2x2 packing outside the trunk (`in_channels 128 = 32 * 2 * 2`).
                    patch_size: Some(2),
                    // `Flux2Config::num_latent_channels`.
                    latent_channels: Some(32),
                    // `Flux2Config::vae_scale_factor`.
                    vae_spatial_scale: Some(8),
                    // Structurally absent: `AutoencoderKLFlux2` is an image autoencoder.
                    vae_temporal_scale: None,
                    // `lib.rs: dtype: DType::F32`, the same width as `COMPUTE_DTYPE_BYTES`.
                    activation_dtype_width: Some(4),
                },
                "{id} architecture facts"
            );
            assert_eq!(
                u64::from(contract.architecture_facts.activation_dtype_width.unwrap()),
                COMPUTE_DTYPE_BYTES,
                "{id}: the published activation width and the pricing width must not drift"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);

            // The registry's weights-free surface resolves nothing on disk, so no axis is knowable.
            let weights_free = LoadSpec::new(WeightsSource::Dir(
                "/__sceneworks_memory_contract_surface__".into(),
            ));
            let contract = composed_provider_contract_for(id, &weights_free).unwrap();
            assert!(
                contract.architecture_facts.is_empty(),
                "{id} weights-free facts must be empty"
            );
            // A weights-free contract legitimately declares nothing, so the E2 config-derived gate
            // does not apply to it; the byte-decomposition half of the conformance walk still does.
            gen_core_testkit::assert_memory_contract_asset_facts_conform(&contract);
        }
    }

    fn write_typed_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize], usize)]) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn append_sparse_f16_tensor(path: &std::path::Path, name: &str, shape: &[usize]) {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        // `OPEN_EXISTING` keeps whatever flag the fixture writer set, so this only re-asserts it —
        // but the appended span is what would allocate if the base file arrived dense.
        gen_core_testkit::mark_sparse(path);
        let mut encoded_len = [0_u8; 8];
        file.read_exact(&mut encoded_len).unwrap();
        let mut encoded = vec![0_u8; u64::from_le_bytes(encoded_len) as usize];
        file.read_exact(&mut encoded).unwrap();
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&encoded).unwrap();
        let start = header
            .values()
            .filter_map(|entry| entry["data_offsets"][1].as_u64())
            .max()
            .unwrap_or(0);
        let bytes = shape
            .iter()
            .try_fold(2_u64, |total, dimension| {
                total.checked_mul(*dimension as u64)
            })
            .unwrap();
        let end = start.checked_add(bytes).unwrap();
        assert!(header
            .insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            )
            .is_none());
        let encoded = serde_json::to_vec(&header).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + end).unwrap();
    }

    #[derive(Clone, Copy, Debug)]
    enum EncoderSelection {
        Builtin,
        OverrideDir,
        OverrideFile,
        CompleteSnapshot,
    }

    fn directory_spec_with_encoder(
        tmp: &tempfile::TempDir,
        variant: Flux2Variant,
        selection: EncoderSelection,
    ) -> (LoadSpec, std::path::PathBuf) {
        let root = tmp.path().join("base");
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_typed_safetensors(
                &root.join(component).join("model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            variant.encoder_contract(),
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        let selected = match selection {
            EncoderSelection::Builtin => tmp.path().join("base/text_encoder"),
            EncoderSelection::OverrideDir | EncoderSelection::OverrideFile => {
                let selected = tmp.path().join("selected-text-encoder");
                gen_core_testkit::write_encoder_contract_fixture(
                    &selected,
                    variant.encoder_contract(),
                )
                .unwrap();
                spec.text_encoder = Some(match selection {
                    EncoderSelection::OverrideDir => WeightsSource::Dir(selected.clone()),
                    EncoderSelection::OverrideFile => {
                        WeightsSource::File(selected.join("model.safetensors"))
                    }
                    _ => unreachable!(),
                });
                selected
            }
            EncoderSelection::CompleteSnapshot => {
                let selected = tmp.path().join("selected-snapshot");
                gen_core_testkit::write_encoder_contract_fixture(
                    &selected.join("text_encoder"),
                    variant.encoder_contract(),
                )
                .unwrap();
                spec.text_encoder = Some(WeightsSource::Dir(selected.clone()));
                selected.join("text_encoder")
            }
        };
        (spec, selected)
    }

    fn packed_dev_directory_spec(tmp: &tempfile::TempDir, bits: i32) -> LoadSpec {
        let root = tmp.path().join("packed-dev");
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_typed_safetensors(
                &root.join(component).join("model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &root.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
            Some(bits),
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.quantize = Some(match bits {
            4 => Quant::Q4,
            8 => Quant::Q8,
            _ => panic!("test supports Q4/Q8 only"),
        });
        spec
    }

    fn file_spec(tmp: &tempfile::TempDir, quant: Option<Quant>) -> LoadSpec {
        let root = tmp.path().join("base");
        for component in ["text_encoder", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
        )
        .unwrap();
        write_typed_safetensors(
            &root.join("vae/model.safetensors"),
            &[
                ("decoder.weight", "BF16", &[4], 8),
                ("decoder.bias", "U8", &[4], 4),
            ],
        );
        let dit = tmp.path().join("dit.safetensors");
        write_typed_safetensors(
            &dit,
            &[
                ("double_blocks.0.img_mlp.0.weight", "F8_E4M3", &[2, 32], 64),
                ("double_blocks.0.img_mlp.0.weight_scale", "F32", &[], 4),
                ("double_blocks.0.img_mlp.0.input_scale", "F32", &[], 4),
                ("double_blocks.0.img_mlp.2.weight", "U8", &[2, 32], 64),
                ("double_blocks.0.img_mlp.0.bias", "F16", &[2], 4),
            ],
        );
        let mut spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(root));
        spec.quantize = quant;
        spec
    }

    #[test]
    fn imported_file_asset_facts_follow_fp8_dequant_and_ggml_packing() {
        let tmp = tempfile::tempdir().unwrap();
        let dense_spec = file_spec(&tmp, None);
        let dense = provider_contract(&dense_spec).unwrap();
        let dense_conditioning = crate::config::DEV_ENCODER_CONTRACT
            .source_for_load(
                &dense_spec,
                gen_core::require_base_snapshot(&dense_spec, FLUX2_DEV_ID).unwrap(),
            )
            .unwrap();
        let dense_conditioning = f32_or_packed_tensor_headers(
            &dense_conditioning.tensor_headers().unwrap(),
            None,
            "selected text encoder",
            true,
            false,
            "test inventory",
        )
        .unwrap();
        assert_eq!(dense.asset_facts.conditioning_bytes, dense_conditioning);
        assert_eq!(dense.asset_facts.transformer_bytes, 520);
        assert_eq!(dense.asset_facts.decoder_bytes, 32);

        let packed_spec = file_spec(&tmp, Some(Quant::Q4));
        let packed = provider_contract(&packed_spec).unwrap();
        let packed_conditioning = crate::config::DEV_ENCODER_CONTRACT
            .source_for_load(
                &packed_spec,
                gen_core::require_base_snapshot(&packed_spec, FLUX2_DEV_ID).unwrap(),
            )
            .unwrap();
        let packed_conditioning = f32_or_packed_tensor_headers(
            &packed_conditioning.tensor_headers().unwrap(),
            Some(Quant::Q4),
            "selected text encoder",
            true,
            false,
            "test inventory",
        )
        .unwrap();
        assert_eq!(packed.asset_facts.conditioning_bytes, packed_conditioning);
        assert_eq!(packed.asset_facts.transformer_bytes, 36 + 36 + 8);
        assert_eq!(packed.asset_facts.decoder_bytes, 32);
        assert_eq!(
            packed.asset_facts.base_bytes,
            packed_conditioning + packed.asset_facts.transformer_bytes + 32
        );
    }

    #[test]
    fn imported_file_contract_and_loader_share_the_full_typed_field_matrix() {
        let tmp = tempfile::tempdir().unwrap();
        let valid = file_spec(&tmp, Some(Quant::Q4));
        let mut cases = vec![("valid", valid.clone())];
        cases.push(("dense-is-accepted", file_spec(&tmp, None)));

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        cases.push(("precision-is-accepted", precision));
        let mut pid = valid.clone();
        pid.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File(tmp.path().join("pid.safetensors")),
            gemma: WeightsSource::Dir(tmp.path().join("gemma")),
        });
        cases.push(("pid-is-accepted", pid));

        let mut adapter = valid.clone();
        adapter.adapters.push(gen_core::AdapterSpec::new(
            tmp.path().join("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        cases.push(("adapter", adapter));
        let mut nvfp4 = valid.clone();
        nvfp4.quantize = Some(Quant::Nvfp4);
        cases.push(("nvfp4", nvfp4));
        let mut control = valid.clone();
        control.control = Some(WeightsSource::File(tmp.path().join("control.safetensors")));
        cases.push(("control", control));
        let mut extra = valid.clone();
        extra
            .extra_controls
            .push(WeightsSource::File(tmp.path().join("extra.safetensors")));
        cases.push(("extra-control", extra));
        let mut ip = valid.clone();
        ip.ip_adapter = Some(WeightsSource::File(tmp.path().join("ip.safetensors")));
        cases.push(("ip-adapter", ip));
        let mut identity = valid.clone();
        identity.identity = Some(gen_core::IdentityWeights::default());
        cases.push(("identity", identity));
        let mut external_te = valid.clone();
        let external_te_root = tmp.path().join("external-te");
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &external_te_root,
            crate::config::DEV_ENCODER_CONTRACT,
            Some(4),
        )
        .unwrap();
        external_te.text_encoder = Some(WeightsSource::Dir(external_te_root));
        cases.push(("external-text-encoder", external_te));
        let mut unknown = valid.clone();
        unknown.components.insert(
            "unknown".into(),
            WeightsSource::File(tmp.path().join("unknown.safetensors")),
        );
        cases.push(("unknown-component", unknown));

        for (name, spec) in cases {
            assert_eq!(
                crate::validate_load_spec(Flux2Variant::Dev, &spec).is_ok(),
                provider_contract(&spec).is_ok(),
                "File loader/contract validation drift for {name}"
            );
        }
    }

    #[test]
    fn selected_encoder_asset_facts_ignore_unmaterialized_route_tensors() {
        for variant in [Flux2Variant::Klein9b, Flux2Variant::Dev] {
            for selection in [
                EncoderSelection::Builtin,
                EncoderSelection::OverrideDir,
                EncoderSelection::OverrideFile,
                EncoderSelection::CompleteSnapshot,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let (spec, selected) = directory_spec_with_encoder(&tmp, variant, selection);
                let conditioning = || {
                    provider_contract_for(variant.id(), &spec)
                        .unwrap()
                        .asset_facts
                        .conditioning_bytes
                };
                let baseline = conditioning();
                let prefix = match variant {
                    Flux2Variant::Klein9b => "model",
                    Flux2Variant::Dev => "language_model.model",
                };
                for (name, shape) in [
                    ("visual.unused.weight".to_owned(), vec![17]),
                    (
                        format!(
                            "{prefix}.layers.{}.unused_projection.weight",
                            variant.encoder_contract().loaded_hidden_layers
                        ),
                        vec![19],
                    ),
                    (format!("{prefix}.unused_projection.weight"), vec![23]),
                ] {
                    append_sparse_f16_tensor(&selected.join("model.safetensors"), &name, &shape);
                    assert_eq!(
                        conditioning(),
                        baseline,
                        "{} {selection:?} charged ignored tensor {name}",
                        variant.id()
                    );
                }
            }
        }
    }

    #[test]
    fn packed_dev_conditioning_prices_the_runtime_qtensor_format_once() {
        let contract = crate::config::DEV_ENCODER_CONTRACT;
        let attention_width = contract.num_attention_heads * contract.head_dim;
        let kv_width = contract.num_key_value_heads * contract.head_dim;
        let matrix_elements = contract.vocab_size * contract.hidden_size
            + contract.loaded_hidden_layers
                * (2 * attention_width * contract.hidden_size
                    + 2 * kv_width * contract.hidden_size
                    + 3 * contract.intermediate_size * contract.hidden_size);
        let dense_vector_bytes = contract.loaded_hidden_layers * 2 * contract.hidden_size * 4;

        for (bits, bytes_per_block) in [(4, 20_u64), (8, 34_u64)] {
            let tmp = tempfile::tempdir().unwrap();
            let spec = packed_dev_directory_spec(&tmp, bits);
            let expected = u64::try_from(matrix_elements / candle_gen::quant::QUANT_BLOCK).unwrap()
                * bytes_per_block
                + u64::try_from(dense_vector_bytes).unwrap();
            assert_eq!(
                provider_contract(&spec)
                    .unwrap()
                    .asset_facts
                    .conditioning_bytes,
                expected,
                "Q{bits} must count each Q4_1/Q8_0 tensor and no transient affine sidecars"
            );
        }
    }

    /// Klein accepts File sources since sc-21485, so the loader/contract agreement is now pinned
    /// on the one File-source refusal that remains: a load-time Q4/Q8 fold over the pre-quantized
    /// single file. Both seams must reject the identical spec with the identical message.
    #[test]
    fn klein_load_and_memory_contract_reject_the_same_quantized_file_source() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("klein.safetensors");
        std::fs::write(&file, b"klein single file").unwrap();
        let mut spec = LoadSpec::new(WeightsSource::File(file))
            .with_component(
                gen_core::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(tmp.path().join("klein-base")),
            )
            .with_quant(Quant::Q8);
        spec.prepare_file_sources().unwrap();
        let load_error = crate::load_klein(&spec)
            .err()
            .expect("Klein loader must reject a quantized File source")
            .to_string();
        let contract_error = klein_provider_contract(&spec).unwrap_err().to_string();
        assert_eq!(contract_error, load_error);
        assert!(contract_error.contains("pre-quantized"), "{contract_error}");
    }

    fn capability(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> &MemoryStrategyCapability {
        contract
            .strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
            .expect("strategy capability")
    }

    #[test]
    fn dev_contract_is_distinct_and_publishes_all_candidate_ranges() {
        let contract = provider_contract(&spec()).unwrap();
        assert!(contract.conformance_errors().is_empty());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert_eq!(contract.provider_id, FLUX2_DEV_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            CALIBRATION_FINGERPRINT
        );
        assert_ne!(
            CALIBRATION_FINGERPRINT,
            crate::RESIDENCY_CALIBRATION_FINGERPRINT
        );
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                capability(&contract, strategy).support,
                MemoryStrategySupport::Implemented
            );
        }
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_overlaps,
            DECODE_OVERLAPS
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedAttention)
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency)
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn klein_contract_is_distinct_and_publishes_parity_preserving_candidate_ranges() {
        let contract = klein_provider_contract(&spec()).unwrap();
        assert!(contract.conformance_errors().is_empty());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert_eq!(contract.provider_id, FLUX2_KLEIN_9B_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            KLEIN_CALIBRATION_FINGERPRINT
        );
        assert_ne!(KLEIN_CALIBRATION_FINGERPRINT, CALIBRATION_FINGERPRINT);
        assert_eq!(KLEIN_BASE_TRANSFORMER_BLOCKS, 32);
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_tile_edges,
            KLEIN_DECODE_TILE_EDGES
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_overlaps,
            KLEIN_DECODE_OVERLAPS
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedAttention)
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency)
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn klein_scope_rejects_unsupported_tiled_candidate_and_block_domain() {
        let contract = klein_provider_contract(&spec()).unwrap();
        let context = registered_valid_fixture(
            &spec(),
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        let mut scope = Flux2MemoryScope::new(Device::Cpu, &contract, &context);
        assert!(scope.configure_decode(768, 128, context.geometry).is_err());
        assert!(scope
            .configure_decode(
                KLEIN_DECODE_TILE_EDGE,
                KLEIN_DECODE_OVERLAP,
                context.geometry,
            )
            .is_ok());
        assert!(scope
            .materialize_transformer_window(KLEIN_BASE_TRANSFORMER_BLOCKS, 1)
            .is_err());
        assert!(scope
            .materialize_transformer_window(KLEIN_BASE_TRANSFORMER_BLOCKS - 1, 1)
            .is_ok());
    }

    #[test]
    fn klein_route_rejects_dev_only_control_overlay() {
        let dev_contract = provider_contract(&spec()).unwrap();
        let mut context = registered_valid_fixture(
            &spec(),
            &dev_contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        context.overlay = Some(CONTROL_OVERLAY.to_owned());
        assert!(route_is_supported(FLUX2_DEV_ID, &context));
        assert!(!route_is_supported(FLUX2_KLEIN_9B_ID, &context));
    }

    #[test]
    fn klein_contract_never_inherits_dev_control_residency_identity() {
        let mut spec = spec();
        spec.control = Some(WeightsSource::File("control.safetensors".into()));
        let contract = composed_provider_contract_for(FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = contract.formula
        else {
            panic!("FLUX.2 contract must use the component-phase formula")
        };
        assert!(resident_components.is_empty());
    }

    #[test]
    fn resident_load_shape_fails_closed_for_rung_four() {
        let mut spec = spec();
        spec.load_shape = LoadShape::EagerMaterialization;
        let contract = provider_contract(&spec).unwrap();
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency).support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn block_domain_excludes_resident_control_overlay() {
        assert_eq!(BASE_TRANSFORMER_BLOCKS, 56);
        assert_eq!(CONTROL_BLOCKS, 4);
        assert_ne!(
            BASE_TRANSFORMER_BLOCKS,
            BASE_TRANSFORMER_BLOCKS + CONTROL_BLOCKS
        );
    }

    #[test]
    fn stale_identity_pid_and_route_mutation_fail_closed() {
        let spec = spec();
        let contract = provider_contract(&spec).unwrap();
        let mut fixture = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        fixture.context.calibration_fingerprint = "stale-flux2".to_owned();
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
        fixture.context.calibration_fingerprint = CALIBRATION_FINGERPRINT.to_owned();
        fixture.context.use_pid = true;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
        fixture.context.use_pid = false;
        fixture.context.geometry.reference_count = 1;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn active_admission_is_one_shot_request_local_and_non_transferable() {
        let spec = spec();
        let contract = provider_contract(&spec).unwrap();
        let context = registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(0)
            .context;
        let registry = Flux2AdmissionRegistry::new(FLUX2_DEV_ID);

        let wrong_provider = Flux2AdmissionRegistry::new("not_flux2_dev");
        wrong_provider.approve(&context).unwrap();
        assert!(
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, wrong_provider,).is_err()
        );

        let manual_memory = contract.generation_memory(&context.selection).unwrap();
        let manual = GenerationRequest {
            prompt: "manual".to_owned(),
            memory: Some(manual_memory),
            ..Default::default()
        };
        assert!(registry.consume_for_generate(&manual).is_err());

        assert!(
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone(),)
                .is_err(),
            "begin without safety approval must fail"
        );

        registry.approve(&context).unwrap();
        let mut unconfigured =
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone())
                .unwrap();
        assert!(registry
            .consume_for_generate(&GenerationRequest {
                prompt: "unconfigured".to_owned(),
                ..Default::default()
            })
            .is_err());
        unconfigured.finish(MemoryRunOutcome::Canceled).unwrap();

        registry.approve(&context).unwrap();
        let mut scope =
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone())
                .unwrap();
        let mut request = GenerationRequest {
            prompt: "bound".to_owned(),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let copied = request.clone();
        assert!(registry.consume_for_generate(&copied).is_err());

        request.width /= 2;
        assert!(registry.consume_for_generate(&request).is_err());
        request.width *= 2;
        let expected_memory = request.memory;
        request.memory = Some(GenerationMemory::default());
        assert!(registry.consume_for_generate(&request).is_err());
        request.memory = expected_memory;

        registry.consume_for_generate(&request).unwrap();
        assert!(registry.consume_for_generate(&request).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }
}

#[cfg(test)]
mod klein_nvfp4_pricing_tests {
    use super::*;
    use candle_gen::gen_core::checkpoint_codec::{
        CheckpointCodecRegistration, CodecResidencyPolicy, ResidencyMode, TensorCodecSpec,
        NVFP4_CODEC,
    };
    use candle_gen::logical_weights::CandleCodecResidency;
    use std::collections::BTreeMap;
    use std::path::Path;

    /// Residency policy forcing NVFP4 rows packed on any host — the CPU-lane stand-in for an
    /// `sm_120` device, the same pattern `single_file`'s conformance tests use. Without it this
    /// test could only ever observe the dense half of a two-sided pricing bug.
    struct ForcePackedNvfp4;

    impl CodecResidencyPolicy for ForcePackedNvfp4 {
        fn residency(
            &self,
            codec: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            stored_shape: &[usize],
        ) -> ResidencyMode {
            if codec.codec_id == NVFP4_CODEC.codec_id {
                return ResidencyMode::Packed;
            }
            CandleCodecResidency::DENSE.residency(codec, spec, stored_shape)
        }
    }

    /// A klein-shaped NVFP4-**mixed** single file at fixture width, the real artifact's shape in
    /// miniature: one dense BF16 embedder plus one `.comfy_quant`-described NVFP4 fused qkv with
    /// both scale levels, declared file-wide the way the Kitchen exporter writes it.
    ///
    /// Geometry: `inner = 128` (NVFP4-legal: each 128-row qkv slice is exactly one scale-factor
    /// atom tile), `in_channels = 8`.
    fn fixture_cfg() -> crate::config::Flux2Config {
        let mut cfg = Flux2Variant::Klein9b.config();
        cfg.num_double_layers = 2;
        cfg.num_single_layers = 1;
        cfg.num_heads = 16;
        cfg.head_dim = 8;
        cfg.in_channels = 8;
        cfg
    }

    /// `block` selects which double block carries the packed qkv: block **1** is interior (its
    /// `to_q/to_k/to_v` stay genuinely packed under the role table), block **0** is the leading
    /// edge (its whole surface is the W4A16 outlier class) — the two sides of the mixed policy the
    /// pricing must distinguish (sc-11045 fix round, BLOCKER 4).
    fn write_nvfp4_mixed_klein_file(path: &Path, block: usize) -> (usize, usize) {
        let (inner, in_channels) = (128usize, 8usize);
        let (rows, cols) = (3 * inner, inner);
        let embedder = vec![0u8; inner * in_channels * 2];
        let packed = vec![0u8; rows * cols / 2];
        let scale_shape = candle_gen::gen_core::nvfp4_scale_shape([rows, cols]).to_vec();
        let scales = vec![0x38u8; scale_shape.iter().product::<usize>()];
        let global = 1.0f32.to_le_bytes();
        let qkv = format!("double_blocks.{block}.img_attn.qkv");
        let (weight_key, scale_key, global_key) = (
            format!("{qkv}.weight"),
            format!("{qkv}.weight_scale"),
            format!("{qkv}.weight_scale_2"),
        );
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "img_in.weight",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::BF16,
                vec![inner, in_channels],
                &embedder,
            )
            .unwrap(),
        );
        tensors.insert(
            weight_key.as_str(),
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::U8,
                vec![rows, cols / 2],
                &packed,
            )
            .unwrap(),
        );
        tensors.insert(
            scale_key.as_str(),
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F8_E4M3,
                scale_shape,
                &scales,
            )
            .unwrap(),
        );
        tensors.insert(
            global_key.as_str(),
            ::safetensors::tensor::TensorView::new(::safetensors::Dtype::F32, vec![], &global)
                .unwrap(),
        );
        let metadata = std::collections::HashMap::from([(
            "_quantization_metadata".to_string(),
            format!(
                r#"{{"format_version": "1.0", "layers": {{"{qkv}": {{"format": "nvfp4", "group_size": 16, "orig_dtype": "torch.bfloat16", "orig_shape": [384, 128]}}}}}}"#
            ),
        )]);
        ::safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
        (inner, in_channels)
    }

    /// sc-21485 review (blocker). The klein `WeightsSource::File` DiT is priced from the compiled
    /// plan, so NVFP4 costs what it actually costs in BOTH residency modes.
    ///
    /// The two totals are derived here from the geometry, not read off the implementation:
    ///
    /// * dense — every logical element at the pipeline's F32 compute dtype (NOT the codec's bf16
    ///   resident encoding, which `PlannedDitWeights` casts away): `img_in` 128x8 plus the three
    ///   128x128 qkv slices, x4 bytes; retained companions cost nothing on a dense decode;
    /// * packed — `img_in` still dense F32, but each qkv slice keeps its stored nibbles
    ///   (128x128 four-bit codes = 8192 B), and the retained scale surface is charged: the E4M3
    ///   block scales partition the stored `[384, 8]` `to_blocked` matrix exactly, and each of the
    ///   three outputs retains its own copy of the scalar F32 global scale.
    ///
    /// Mutation witness (RUN, both modes): restore the old arm by making
    /// `klein_planned_dit_bytes` delegate to
    /// `f32_or_packed_component_bytes(path, None, "imported DiT", false, true)`. It charges the
    /// packed `U8 [384, 64]` header `materialized_bytes(4)` and drops `weight_scale`, giving one
    /// mode-independent total that over-prices the packed mode and under-prices the dense one —
    /// precisely the false-reject / false-admit pair this test forbids.
    #[test]
    fn the_klein_file_dit_is_priced_from_the_plan_in_both_residency_modes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-nvfp4-mixed.safetensors");
        // Block 1 — interior, so the role table keeps its qkv slices genuinely packed and this
        // test measures the packed arm of the pricing.
        let (inner, in_channels) = write_nvfp4_mixed_klein_file(&path, 1);
        let cfg = fixture_cfg();
        assert_eq!((cfg.inner_dim(), cfg.in_channels), (inner, in_channels));

        let embedder_bytes = (inner * in_channels * 4) as u64;
        let qkv_dense_bytes = (3 * inner * inner * 4) as u64;
        let dense = klein_planned_dit_bytes(&path, &cfg, &CandleCodecResidency::DENSE)
            .expect("dense pricing");
        assert_eq!(dense, embedder_bytes + qkv_dense_bytes);
        assert_eq!(dense, 200_704);

        let qkv_nibble_bytes = (3 * inner * inner / 2) as u64;
        let block_scale_bytes = candle_gen::gen_core::nvfp4_scale_shape([3 * inner, inner])
            .iter()
            .product::<usize>() as u64;
        let global_scale_bytes = 3 * 4;
        let packed =
            klein_planned_dit_bytes(&path, &cfg, &ForcePackedNvfp4).expect("packed pricing");
        assert_eq!(
            packed,
            embedder_bytes + qkv_nibble_bytes + block_scale_bytes + global_scale_bytes
        );
        assert_eq!(packed, 31_756);

        // The two modes must not collapse onto one number: a pricing arm blind to residency is the
        // defect, and `packed < dense` is the whole reason the native leg exists.
        assert!(
            packed < dense,
            "packed NVFP4 residency must cost less than the dense fallback ({packed} vs {dense})"
        );

        // The header-sum arm this replaced is mode-independent and matches NEITHER total — the
        // mutation above simply restores it.
        let header_sum =
            f32_or_packed_component_bytes(&path, None, "imported DiT", false, true).unwrap();
        assert_ne!(header_sum, dense);
        assert_ne!(header_sum, packed);
    }

    /// **BLOCKER 4 (sc-11045 feature-end review): role-dense (W4A16) rows are priced at the dense
    /// bf16 they actually hold, never at packed bytes.**
    ///
    /// The same fixture geometry with the packed qkv in double block **0** — the leading edge,
    /// whose whole surface the role table sends to W4A16. The loader will dequantize each slice to
    /// a resident dense **bf16** weight and consume the scales, so the pricing must charge
    /// 2 B per logical element and nothing for the companions. Pricing these rows at packed bytes
    /// (the pre-fix arm) under-prices by `dense_bf16 - packed` per row — ~0.94 GB across the
    /// pinned klein artifact — and falsely admits.
    ///
    /// # Mutation witness (RUN)
    ///
    /// Restore the packed-only pricing — make the `ResidencyMode::Packed` match arm
    /// unconditionally return `tensor.residency.resident_bytes` (and drop the companion
    /// `packed_owner` filter): the total collapses onto `packed_only` below and both
    /// `assert_eq!(mixed, ...)` and `assert!(mixed > packed_only)` go red.
    #[test]
    fn role_dense_packed_rows_are_priced_at_their_dense_bf16_residency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("klein-nvfp4-edge.safetensors");
        let (inner, in_channels) = write_nvfp4_mixed_klein_file(&path, 0);
        let cfg = fixture_cfg();

        let embedder_bytes = (inner * in_channels * 4) as u64;
        // Three role-dense W4A16 slices: dense bf16 (2 B/elt), scales consumed.
        let qkv_w4a16_bytes = (3 * inner * inner * 2) as u64;
        let mixed =
            klein_planned_dit_bytes(&path, &cfg, &ForcePackedNvfp4).expect("mixed-policy pricing");
        assert_eq!(mixed, embedder_bytes + qkv_w4a16_bytes);
        assert_eq!(mixed, 102_400);

        // The falsely-admitting number the review measured: nibbles + retained scales.
        let packed_only = {
            let qkv_nibble_bytes = (3 * inner * inner / 2) as u64;
            let block_scale_bytes = candle_gen::gen_core::nvfp4_scale_shape([3 * inner, inner])
                .iter()
                .product::<usize>() as u64;
            embedder_bytes + qkv_nibble_bytes + block_scale_bytes + (3 * 4)
        };
        assert!(
            mixed > packed_only,
            "a W4A16 row holds dense bf16, which must price above the packed bytes the old arm \
             charged ({mixed} vs {packed_only})"
        );
    }
}
