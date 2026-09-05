//! Candle/CUDA Z-Image adoption of the shared five-rung image memory contract (sc-15815).
//!
//! `z_image_edit` is a SceneWorks catalog alias for the `z_image_turbo` provider and therefore
//! inherits the Turbo contract. The two bespoke control providers are registered separately so the
//! matrix records their current gaps rather than inferring capability from the plain provider.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, LoadSpec, MemoryNumericTier, MemoryProviderContract, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, Precision, Quant, WeightsSource,
};
use candle_gen::gen_core::{
    LoadShape, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges,
    MemoryPrerequisiteScope, MemoryResidentComponent, MemoryStrategyCapability,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWindowMaterialization,
    PerComponentBytes,
};
use candle_transformers::models::z_image::transformer::Config as DitConfig;
use candle_transformers::models::z_image::vae::VaeConfig;
use gen_core::MemoryPhase;
#[cfg(any(feature = "cuda", test))]
use gen_core::MemoryRequestScope;
#[cfg(test)]
use gen_core::{GenerationMemory, GenerationRequest, MemoryGeometry, MemoryRunOutcome};

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
    // A Hugging Face cache snapshot symlinks every file into the repository's sibling `blobs/`
    // tree, so the shared helper authorizes that repository directory too (sc-22727).
    Ok(gen_core::hf_cache_discovery_roots(root)?)
}

pub(crate) const DECODE_TILE_EDGE: u32 = 512;
pub(crate) const DECODE_OVERLAP: u32 = 128;
pub(crate) const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub(crate) const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 15, 30];
pub(crate) const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub(crate) const CALIBRATION_FINGERPRINT: &str =
    "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2";
/// Provider-local identity of the typed auxiliary control component. Stable across loads: the
/// bespoke control routes hold exactly one control network.
pub(crate) const CONTROL_COMPONENT_ID: &str = "z-image-control-branch";
/// The lazily built f32 `VaeEncoder` a `Reference` request opts into (SC-22667). It is a **second**
/// copy of the `encoder.*` weights: the resident `AutoEncoderKL` already holds both halves at the
/// pipeline dtype (those bytes are in `decoder_bytes`), and img2img builds this one on its first
/// request at [`crate::common::ENC_DTYPE`].
pub(crate) const REFERENCE_ENCODER_COMPONENT_ID: &str = "z-image-vae-reference-encoder";
/// Key prefix of the autoencoder's encoder half in both the diffusers and the LDM/ComfyUI layouts.
const VAE_ENCODER_PREFIX: &str = "encoder.";
/// Element width of the resident bf16 pipeline dtype every Dir/File component is opened at.
const PIPELINE_FLOAT_WIDTH: u64 = 2;
/// Element width of [`crate::common::ENC_DTYPE`], the lazily built reference encoder's dtype.
const REFERENCE_ENCODER_FLOAT_WIDTH: u64 = 4;
pub(crate) const CONTROL_CALIBRATION_FINGERPRINT: &str =
    "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2";

/// Activation dtype the loaded Z-Image pipeline computes in. `lib.rs` pins `DType::BF16`
/// unconditionally ("Z-Image is a bf16 model; load at bf16 regardless of the CPU-default dtype"),
/// so this is the provider's real activation width rather than a memory-model literal.
const ACTIVATION_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;

/// Architecture axes for the Z-Image routes (epic SC-22657, E2).
///
/// These are the **loader's** geometry, not the snapshot's `config.json`. `pipeline.rs` builds
/// every DiT from `DitConfig::z_image_turbo()` and every autoencoder from `VaeConfig::z_image()`
/// — the component `config.json` files are read at load only for their `quantization` block — so
/// a snapshot whose `transformer/config.json` says `n_layers: 24` still loads a 30-block trunk.
/// Publishing what that file says (the shape the feature-end review caught) would describe a
/// pipeline this crate never constructs; the axes therefore come off the same two presets handed
/// to the builders, the way `candle-gen-qwen-image` and `candle-gen-flux2` publish theirs.
///
/// `head_dim` is `dim / n_heads` as `DitConfig::head_dim` computes it, published through the
/// shared divisibility rule. `vae_spatial_scale` is the halving count of `block_out_channels`
/// (four stages, three halvings, x8). `vae_temporal_scale` stays `None`: Z-Image ships the FLUX.1
/// image AutoencoderKL, which has no temporal axis at all, and a structurally absent axis is
/// declared absent, never zero.
///
/// A weights-free contract — the registry's sentinel surface, or a single-file ComfyUI import —
/// publishes `MemoryArchitectureFacts::default()`: no pipeline has been resolved there. The gate is
/// [`candle_gen::architecture_facts::snapshot_root`], an existing directory, rather than a bare
/// `WeightsSource::Dir` match.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    // Exactly the presets `pipeline.rs` hands to the DiT and `AutoEncoderKL` builders.
    let dit = DitConfig::z_image_turbo();
    let vae = VaeConfig::z_image();
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(dit.n_heads),
        head_dim: af::head_dim(af::declared(dit.dim), af::declared(dit.n_heads)),
        transformer_blocks: af::declared(dit.n_layers),
        patch_size: dit.all_patch_size.first().copied().and_then(af::declared),
        latent_channels: af::declared(vae.latent_channels),
        // Each stage after the first halves both spatial axes: four stages give the x8 scale.
        vae_spatial_scale: af::declared(vae.block_out_channels.len())
            .and_then(|stages| stages.checked_sub(1))
            .and_then(|downsamples| (downsamples <= 5).then(|| 1_u32 << downsamples)),
        vae_temporal_scale: None,
        activation_dtype_width: af::dtype_width(ACTIVATION_DTYPE),
    }
}

fn ordinary_streamable(spec: &LoadSpec) -> bool {
    spec.precision == Precision::Bf16
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.quantize.is_none()
        && spec.pid.is_none()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.components.is_empty()
}

fn control_streamable(spec: &LoadSpec) -> bool {
    spec.precision == Precision::Bf16
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none()
        && spec.control.is_some()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.components.is_empty()
}

fn set_transformer_streamability(contract: &mut MemoryProviderContract, streamable: bool) {
    let capability = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
        .expect("Z-Image contracts always publish the complete strategy ladder");
    capability.support = if streamable {
        MemoryStrategySupport::Implemented
    } else {
        MemoryStrategySupport::Missing
    };
    capability.parameters.transformer_window_sizes = if streamable {
        TRANSFORMER_WINDOW_SIZES.to_vec()
    } else {
        Vec::new()
    };
    contract.lifecycle.transformer_window_materialization = streamable;
}

fn surface_selector_matches_spec(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<()> {
    let tier_matches = match surface.resolved_artifact_tier() {
        gen_core::MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == Precision::Bf16 && surface.spec.quantize.is_none()
        }
        gen_core::MemoryContractSurfaceTier::Q4 => surface.spec.quantize == Some(Quant::Q4),
        gen_core::MemoryContractSurfaceTier::Q8 => surface.spec.quantize == Some(Quant::Q8),
        gen_core::MemoryContractSurfaceTier::Nvfp4 => false,
    };
    if tier_matches
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(gen_core::Error::Msg(format!(
            "Z-Image memory surface selector '{}' does not match its weights-free LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Convert the generic finite-surface witness into the executable Z-Image load shape.
///
/// Q4/Q8 are already-packed artifact tiers selected before provider load. They therefore reach the
/// real loader with `quantize == None`; retaining the generic witness' `Some(Q4|Q8)` here would mean
/// unsupported on-the-fly packing of a dense source.
fn normalized_surface_spec(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<LoadSpec> {
    surface_selector_matches_spec(surface)?;
    let mut spec = surface.spec.clone();
    spec.quantize = None;
    Ok(spec)
}

pub(crate) fn weights_free_surface_contract(
    provider_id: &str,
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let spec = normalized_surface_spec(surface)?;
    provider_contract(provider_id, &spec)
}

pub(crate) fn weights_free_control_surface_contract(
    provider_id: &str,
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let spec = normalized_surface_spec(surface)?;
    if spec.control.is_none() {
        return Err(gen_core::Error::Msg(format!(
            "{provider_id}: control contract surface is missing its mandatory control source"
        )));
    }
    control_contract(provider_id, &spec)
}

fn imported_tensor_bytes(
    tensor: &gen_core::weightsmeta::SafetensorsTensorHeader,
    loaded_name: &str,
    component: &str,
) -> gen_core::Result<u64> {
    imported_tensor_bytes_at(tensor, loaded_name, component, PIPELINE_FLOAT_WIDTH)
}

/// [`imported_tensor_bytes`] at an explicit float width: the resident pipeline opens every
/// component bf16 (`2`), while the lazily built reference encoder opens the same `encoder.*` rows
/// at [`crate::common::ENC_DTYPE`] (`4`).
fn imported_tensor_bytes_at(
    tensor: &gen_core::weightsmeta::SafetensorsTensorHeader,
    loaded_name: &str,
    component: &str,
    float_width: u64,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;

    // `candle_core::safetensors::load` first materializes every source tensor. U16 is promoted to
    // Candle's U32 storage; the remaining accepted integer widths stay native. Every float then
    // passes through `normalize_fp8_map(..., dtype)`, including dense f32/f64 and plain/scaled fp8.
    let loaded = match tensor.dtype {
        Dtype::U8 | Dtype::U32 | Dtype::I16 | Dtype::I32 | Dtype::I64 => tensor.data_bytes,
        Dtype::U16 => tensor.materialized_bytes(4)?,
        Dtype::F8_E4M3 | Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => {
            tensor.materialized_bytes(float_width)?
        }
        dtype => {
            return Err(gen_core::Error::Unsupported(format!(
                "z-image imported {component} tensor {:?} uses unsupported Candle dtype {dtype:?}",
                tensor.name
            )))
        }
    };
    // Test the key *after* any combined-checkpoint component prefix is stripped. That is the key
    // `normalize_fp8_map` sees, so a combined `model.diffusion_model.scaled_fp8` marker is omitted
    // exactly like a standalone transformer's unprefixed `scaled_fp8` marker.
    if loaded_name == "scaled_fp8"
        || loaded_name.ends_with(".scale_weight")
        || loaded_name.ends_with(".weight_scale")
        || loaded_name.ends_with(".scale_input")
        || loaded_name.ends_with(".input_scale")
    {
        Ok(0)
    } else {
        Ok(loaded)
    }
}

fn single_file_tensor_bytes(path: &std::path::Path, component: &str) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    imported_tensor_headers_bytes(&headers, component, &path.display().to_string())
}

/// How one packed MLX affine triple (`{base}.weight` u32 codes + `.scales` + `.biases`) lands on
/// the device for a given component — the two things this loader does with one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackedResidency {
    /// Repacked **once** into a GGML `Q4_1`/`Q8_0` [`candle_gen::candle_core::quantized::QTensor`]
    /// — the transformer (`PackedDit`, whether resident or block-streamed from the device-format
    /// cache) and the text encoder (`PackedTe`). The source triple is a transient host input and
    /// the cache sidecar is a file-backed copy of the same bytes, so neither is priced beside it.
    Qtensor,
    /// Dequantized to a **dense** float matrix at `float_width` — the VAE's eight mid-block
    /// attention projections (`Pipeline::vae_vb_dequantized_on`), which the stock `AutoEncoderKL`
    /// reads as ordinary Linear weights.
    Dense { float_width: u64 },
}

/// Resident bytes of one packed triple after [`PackedResidency::Dense`] dequantization:
/// `out × in × float_width`, with `in = scale_columns × group_size` exactly as
/// [`candle_gen::quant::mlx_packed_qtensor_resident_bytes`] recovers it.
fn packed_triple_dense_bytes(
    weight: &gen_core::weightsmeta::SafetensorsTensorHeader,
    scales: &gen_core::weightsmeta::SafetensorsTensorHeader,
    group_size: usize,
    float_width: u64,
) -> gen_core::Result<u64> {
    let [out, _] = weight.shape.as_slice() else {
        return Err(gen_core::Error::Unsupported(format!(
            "packed weight {:?} must be rank 2, got {:?}",
            weight.name, weight.shape
        )));
    };
    let [_, scale_columns] = scales.shape.as_slice() else {
        return Err(gen_core::Error::Unsupported(format!(
            "packed scales {:?} must be rank 2, got {:?}",
            scales.name, scales.shape
        )));
    };
    u64::try_from(*out)
        .ok()
        .zip(u64::try_from(*scale_columns).ok())
        .zip(u64::try_from(group_size).ok())
        .and_then(|((out, scale_columns), group_size)| {
            out.checked_mul(scale_columns)?
                .checked_mul(group_size)?
                .checked_mul(float_width)
        })
        .ok_or_else(|| {
            gen_core::Error::Msg(format!("packed dense byte overflow for {:?}", weight.name))
        })
}

/// Price a header inventory the way the Candle loader materializes it: every packed affine triple
/// is charged **once** per `residency` (its `.scales`/`.biases` siblings are transient inputs and
/// contribute nothing), and every other tensor is priced by `dense`.
fn packed_aware_headers_bytes(
    headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
    group_size: usize,
    residency: PackedResidency,
    component: &str,
    dense: impl Fn(&gen_core::weightsmeta::SafetensorsTensorHeader) -> gen_core::Result<u64>,
) -> gen_core::Result<u64> {
    use std::collections::{HashMap, HashSet};

    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<HashMap<_, _>>();
    let packed_bases = headers
        .iter()
        .filter_map(|header| header.name.strip_suffix(".scales"))
        .collect::<HashSet<_>>();
    headers.iter().try_fold(0_u64, |total, header| {
        if header
            .name
            .strip_suffix(".scales")
            .or_else(|| header.name.strip_suffix(".biases"))
            .is_some_and(|base| packed_bases.contains(base))
        {
            return Ok(total);
        }
        let resident = if let Some(base) = header
            .name
            .strip_suffix(".weight")
            .filter(|base| packed_bases.contains(base))
        {
            let scales_name = format!("{base}.scales");
            let biases_name = format!("{base}.biases");
            let scales = by_name.get(scales_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "z-image packed {component} weight {:?} is missing {scales_name:?}",
                    header.name
                ))
            })?;
            let biases = by_name.get(biases_name.as_str()).ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "z-image packed {component} weight {:?} is missing {biases_name:?}",
                    header.name
                ))
            })?;
            match residency {
                PackedResidency::Qtensor => candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                    header, scales, biases, group_size,
                )?,
                PackedResidency::Dense { float_width } => {
                    packed_triple_dense_bytes(header, scales, group_size, float_width)?
                }
            }
        } else {
            dense(header)?
        };
        total.checked_add(resident).ok_or_else(|| {
            gen_core::Error::Msg(format!("z-image {component} resident byte sum overflow"))
        })
    })
}

/// The autoencoder priced the way this loader holds it (SC-22667).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VaeBytes {
    /// Every VAE tensor at the pipeline dtype. The stock `AutoEncoderKL` constructor builds **both**
    /// `encoder.*` and `decoder.*` from the same `VarBuilder` (`Pipeline::load_vae`), so a plain
    /// text-to-image render keeps both halves resident; that whole object is `decoder_bytes`.
    resident: u64,
    /// The `encoder.*` rows again, at [`crate::common::ENC_DTYPE`]: the separate `VaeEncoder` that
    /// `ZImageBaseGenerator::vae_encoder` builds on the first `Reference` request and never for a
    /// text-to-image workload. Declared as a [`gen_core::MemoryComponentKind::ReferenceEncoder`]
    /// auxiliary component, not folded into any base field.
    reference_encoder: u64,
}

impl VaeBytes {
    /// `name_of` yields the key the loader sees (a combined checkpoint strips its component prefix
    /// first, so `first_stage_model.encoder.*` and `vae.encoder.*` both test as `encoder.*`).
    fn from_headers(
        headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
        group_size: usize,
    ) -> gen_core::Result<Self> {
        let resident = packed_aware_headers_bytes(
            headers,
            group_size,
            PackedResidency::Dense {
                float_width: PIPELINE_FLOAT_WIDTH,
            },
            "VAE",
            |header| imported_tensor_bytes(header, &header.name, "VAE"),
        )?;
        let encoder_headers = headers
            .iter()
            .filter(|header| header.name.starts_with(VAE_ENCODER_PREFIX))
            .cloned()
            .collect::<Vec<_>>();
        let reference_encoder = packed_aware_headers_bytes(
            &encoder_headers,
            group_size,
            PackedResidency::Dense {
                float_width: REFERENCE_ENCODER_FLOAT_WIDTH,
            },
            "VAE reference encoder",
            |header| {
                imported_tensor_bytes_at(
                    header,
                    &header.name,
                    "VAE reference encoder",
                    REFERENCE_ENCODER_FLOAT_WIDTH,
                )
            },
        )?;
        Ok(Self {
            resident,
            reference_encoder,
        })
    }

    fn from_file(path: &std::path::Path) -> gen_core::Result<Self> {
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
        let vae = Self::from_headers(&headers, candle_gen::quant::MLX_GROUP_SIZE)?;
        if vae.resident == 0 {
            return Err(gen_core::Error::Msg(format!(
                "z-image imported VAE '{}' contains no tensor bytes",
                path.display()
            )));
        }
        Ok(vae)
    }
}

/// The base-model split plus the request-opted reference encoder, for every source shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MaterializedComponents {
    base: PerComponentBytes,
    reference_encoder: u64,
}

/// Tensor headers of the snapshot component `sub/`, or `None` for a weights-free declaration seam
/// (no `.safetensors` under it). The header walk skips the hidden device-format cache a streamed
/// packed component leaves beside its source file, so a snapshot that has been loaded on a packed
/// tier prices exactly the tensors the loader opens (SC-22667, E1).
fn snapshot_component_headers(
    root: &std::path::Path,
    sub: &str,
) -> gen_core::Result<Option<Vec<gen_core::weightsmeta::SafetensorsTensorHeader>>> {
    let dir = root.join(sub);
    if gen_core::safetensors_path_bytes(&dir) == 0 {
        return Ok(None);
    }
    gen_core::weightsmeta::safetensors_path_tensor_headers(&dir).map(Some)
}

/// The packed group size `sub/config.json` declares, or MLX's default when the component carries no
/// `quantization` block — which is also how the loader reads it (`PackedConfig::from_config`).
fn snapshot_component_group_size(root: &std::path::Path, sub: &str) -> gen_core::Result<usize> {
    Ok(crate::pipeline::packed_config_at(root, sub)
        .map_err(gen_core::Error::backend)?
        .and_then(|packed| usize::try_from(packed.group_size).ok())
        .filter(|group_size| *group_size > 0)
        .unwrap_or(candle_gen::quant::MLX_GROUP_SIZE))
}

/// Bytes the transformer occupies once loaded from `root/transformer/`: a dense tier at the
/// pipeline dtype; a packed tier as one GGML `Q4_1`/`Q8_0` QTensor per projection plus its dense
/// residual (norms, embeddings, modulation) at the pipeline dtype. This is what both the resident
/// `PackedDit::new` build and the block-streamed build hold on the device — the streamed build
/// copies the same GGML bytes out of the device-format cache instead of repacking, so the cache is
/// **not** a second copy to price. Before this, the on-disk sum recursed into that cache and priced
/// the q4 transformer at 7.31 GB against a 3.47 GB checkpoint (SC-22667).
fn snapshot_transformer_bytes(root: &std::path::Path) -> gen_core::Result<u64> {
    let Some(headers) = snapshot_component_headers(root, "transformer")? else {
        return Ok(0);
    };
    packed_aware_headers_bytes(
        &headers,
        snapshot_component_group_size(root, "transformer")?,
        PackedResidency::Qtensor,
        "transformer",
        |header| imported_tensor_bytes(header, &header.name, "transformer"),
    )
}

fn snapshot_vae_bytes(root: &std::path::Path) -> gen_core::Result<VaeBytes> {
    let Some(headers) = snapshot_component_headers(root, "vae")? else {
        return Ok(VaeBytes::default());
    };
    VaeBytes::from_headers(&headers, snapshot_component_group_size(root, "vae")?)
}

/// The typed declaration of a non-zero reference encoder (`None` when the source has no
/// `encoder.*` rows, e.g. a decoder-only fixture).
fn reference_encoder_component(bytes: u64) -> Option<MemoryResidentComponent> {
    (bytes > 0).then(|| MemoryResidentComponent {
        id: REFERENCE_ENCODER_COMPONENT_ID.to_owned(),
        kind: gen_core::MemoryComponentKind::ReferenceEncoder,
        resident_bytes: bytes,
        bounded_by: None,
        residency: gen_core::MemoryComponentResidency::WholeRender,
    })
}

/// Append typed auxiliary components to a phase formula, promoting a `PhaseEnvelope` to a
/// `ComponentPhaseEnvelope` the first time one is declared. A formula with nothing to append is
/// returned unchanged, so a source with no auxiliary network keeps its historical shape.
fn with_resident_components(
    formula: MemoryFormulaKind,
    extra: Vec<MemoryResidentComponent>,
    provider_id: &str,
) -> gen_core::Result<MemoryFormulaKind> {
    if extra.is_empty() {
        return Ok(formula);
    }
    match formula {
        MemoryFormulaKind::PhaseEnvelope { phases, variables } => {
            Ok(MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables,
                resident_components: extra,
            })
        }
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            mut resident_components,
        } => {
            resident_components.extend(extra);
            Ok(MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables,
                resident_components,
            })
        }
        other => Err(gen_core::Error::Msg(format!(
            "{provider_id}: expected a phase-envelope base formula, got {other:?}"
        ))),
    }
}

fn imported_tensor_headers_bytes(
    headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
    component: &str,
    source: &str,
) -> gen_core::Result<u64> {
    let bytes = headers.iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(imported_tensor_bytes(tensor, &tensor.name, component)?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "z-image imported {component} resident byte sum overflow"
                ))
            })
    })?;
    if bytes == 0 {
        return Err(gen_core::Error::Msg(format!(
            "z-image imported {component} '{source}' contains no tensor bytes"
        )));
    }
    Ok(bytes)
}

fn materialized_text_encoder_headers_bytes(
    headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
) -> gen_core::Result<u64> {
    let group_size = crate::ENCODER_CONTRACT
        .packing
        .expect("Z-Image's executable encoder contract is packable")
        .group_size;
    let bytes = packed_aware_headers_bytes(
        headers,
        group_size,
        PackedResidency::Qtensor,
        "text encoder",
        |header| imported_tensor_bytes(header, &header.name, "text encoder"),
    )?;
    if bytes == 0 {
        return Err(gen_core::Error::Msg(
            "z-image validated text encoder contains no materialized tensor bytes".into(),
        ));
    }
    Ok(bytes)
}

fn selected_encoder_has_authoritative_config(source: &WeightsSource) -> bool {
    match source {
        WeightsSource::File(path) => path
            .parent()
            .is_some_and(|parent| parent.join("config.json").is_file()),
        WeightsSource::Dir(path) => {
            path.join("config.json").is_file() || path.join("text_encoder/config.json").is_file()
        }
    }
}

/// Exact Candle materialization for a source that carries the behavior evidence required by the
/// executable encoder contract. `None` preserves the catalog's weights-free declaration seam for a
/// path that has no authored config yet; any real, authored source is validated and projected from
/// the precise 36-layer language surface rather than its raw shard inventory.
fn validated_materialized_text_encoder_bytes(
    source: &WeightsSource,
    comfyui_file: bool,
) -> gen_core::Result<Option<u64>> {
    let selected = if comfyui_file && matches!(source, WeightsSource::File(_)) {
        let roots = selected_encoder_discovery_roots(source)?;
        let inventory = gen_core::encoder_contract::text_encoder_source_inventory_for_discovery(
            source, &roots,
        )?;
        if inventory
            .tensor_headers()
            .iter()
            .any(|header| header.name == "model.embed_tokens.weight")
        {
            crate::ENCODER_CONTRACT.validate_comfyui_source_for_discovery(source, &roots)?
        } else if selected_encoder_has_authoritative_config(source) {
            crate::ENCODER_CONTRACT.validate_source_for_discovery(source, &roots)?
        } else {
            return Ok(None);
        }
    } else if selected_encoder_has_authoritative_config(source) {
        let roots = selected_encoder_discovery_roots(source)?;
        crate::ENCODER_CONTRACT.validate_source_for_discovery(source, &roots)?
    } else {
        return Ok(None);
    };
    materialized_text_encoder_headers_bytes(selected.materialized_language_tensor_headers())
        .map(Some)
}

fn combined_file_components(path: &std::path::Path) -> gen_core::Result<MaterializedComponents> {
    let mut components = PerComponentBytes::default();
    let mut mapped_text_encoder_headers = Vec::new();
    let mut mapped_vae_headers = Vec::new();
    let mut mapped_keys = [
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
    ];
    for tensor in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
        let Some((component, mapped)) = crate::comfyui::combined_component_key(&tensor.name) else {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} has no component mapping",
                tensor.name
            )));
        };
        if mapped.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} maps to an empty component key",
                tensor.name
            )));
        }
        let (component_index, component_name) = match component {
            crate::comfyui::CombinedComponent::Transformer => (0, "transformer"),
            crate::comfyui::CombinedComponent::TextEncoder => (1, "text encoder"),
            crate::comfyui::CombinedComponent::Vae => (2, "VAE"),
        };
        if !mapped_keys[component_index].insert(mapped.to_owned()) {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} collides at {component_name} key {mapped:?}",
                tensor.name
            )));
        }
        match component {
            crate::comfyui::CombinedComponent::Transformer => {
                let resident_bytes = imported_tensor_bytes(&tensor, mapped, component_name)?;
                components.dit = components.dit.checked_add(resident_bytes).ok_or_else(|| {
                    gen_core::Error::Msg(
                        "z-image combined transformer resident byte sum overflow".into(),
                    )
                })?
            }
            crate::comfyui::CombinedComponent::TextEncoder => {
                mapped_text_encoder_headers.push(gen_core::weightsmeta::SafetensorsTensorHeader {
                    name: mapped.to_owned(),
                    ..tensor
                });
            }
            crate::comfyui::CombinedComponent::Vae => {
                mapped_vae_headers.push(gen_core::weightsmeta::SafetensorsTensorHeader {
                    name: mapped.to_owned(),
                    ..tensor
                });
            }
        }
    }
    let vae = VaeBytes::from_headers(&mapped_vae_headers, candle_gen::quant::MLX_GROUP_SIZE)?;
    components.vae = vae.resident;
    components.text_encoder = if mapped_text_encoder_headers
        .iter()
        .any(|header| header.name == "model.embed_tokens.weight")
    {
        crate::ENCODER_CONTRACT
            .validate_embedded_comfyui_file(path, crate::comfyui::COMBINED_TEXT_ENCODER_PREFIXES)?;
        let headers = crate::ENCODER_CONTRACT
            .materialized_dense_language_tensor_headers(&mapped_text_encoder_headers)?;
        materialized_text_encoder_headers_bytes(&headers)?
    } else {
        imported_tensor_headers_bytes(
            &mapped_text_encoder_headers,
            "text encoder",
            "combined checkpoint mapped inventory",
        )?
    };
    for (component, bytes) in [
        ("transformer", components.dit),
        ("text encoder", components.text_encoder),
        ("VAE", components.vae),
    ] {
        if bytes == 0 {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint '{}' is missing the {component} component",
                path.display()
            )));
        }
    }
    Ok(MaterializedComponents {
        base: components,
        reference_encoder: vae.reference_encoder,
    })
}

fn imported_file_components(
    spec: &LoadSpec,
    primary: &std::path::Path,
    provider_id: &str,
) -> gen_core::Result<MaterializedComponents> {
    let _ = gen_core::require_base_snapshot(spec, provider_id)?;
    let legacy_text_encoder = spec
        .components
        .get(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
    if spec.text_encoder.is_some() && legacy_text_encoder.is_some() {
        return Err(gen_core::Error::Msg(format!(
            "{provider_id}: text encoder was supplied through both LoadSpec::text_encoder and legacy component '{}'",
            gen_core::COMFYUI_TEXT_ENCODER_COMPONENT
        )));
    }
    let text_encoder = spec.text_encoder.as_ref().or(legacy_text_encoder);
    let vae = spec.components.get(gen_core::COMFYUI_VAE_COMPONENT);
    let text_encoder_bytes = |source: &WeightsSource| -> gen_core::Result<u64> {
        if let Some(bytes) = validated_materialized_text_encoder_bytes(source, true)? {
            return Ok(bytes);
        }
        let roots = selected_encoder_discovery_roots(source)?;
        let inventory = gen_core::encoder_contract::text_encoder_source_inventory_for_discovery(
            source, &roots,
        )?;
        imported_tensor_headers_bytes(
            inventory.tensor_headers(),
            "text encoder",
            "direct-shard inventory",
        )
    };
    match (text_encoder, vae) {
        (None, None) => {
            spec.read_file_unchanged_if_prepared(primary, combined_file_components)
        }
        (Some(text_encoder), Some(WeightsSource::File(vae))) => {
            let vae = spec.read_file_unchanged_if_prepared(vae, VaeBytes::from_file)?;
            Ok(MaterializedComponents {
                base: PerComponentBytes {
                    text_encoder: text_encoder_bytes(text_encoder)?,
                    dit: spec.read_file_unchanged_if_prepared(primary, |p| {
                        single_file_tensor_bytes(p, "transformer")
                    })?,
                    vae: vae.resident,
                },
                reference_encoder: vae.reference_encoder,
            })
        }
        (Some(text_encoder), None) => {
            let mut components =
                spec.read_file_unchanged_if_prepared(primary, combined_file_components)?;
            components.base.text_encoder = text_encoder_bytes(text_encoder)?;
            Ok(components)
        }
        (_, Some(WeightsSource::Dir(path))) => Err(gen_core::Error::Msg(format!(
            "{provider_id}: component '{}' must be a file, not {}",
            gen_core::COMFYUI_VAE_COMPONENT,
            path.display()
        ))),
        _ => Err(gen_core::Error::Msg(format!(
            "{provider_id}: separate ComfyUI import requires a text encoder and '{}', or neither for a combined checkpoint",
            gen_core::COMFYUI_VAE_COMPONENT
        ))),
    }
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if provider_id == crate::base::MODEL_ID
        && matches!(spec.weights, gen_core::WeightsSource::File(_))
    {
        return Err(gen_core::Error::Msg(
            "z_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
             not a single .safetensors file"
                .into(),
        ));
    }
    // This declaration seam must remain usable before model assets exist locally. Once an authored
    // config or a contract-complete ComfyUI language signature is present, however, validate and
    // price the exact materialized surface. Synthetic/catalog-only paths without that evidence keep
    // the historical raw-inventory fallback and make no validation claim.
    // File and Dir intentionally retain one provider/calibration identity: the executable provider,
    // phase graph, and output semantics are the same. The promoted matrix has no load-source axis,
    // however, so only Dir advertises rung 4 until the pinned/re-openable File path has its own real
    // measurement. A Dir rung-4 cell must never be relabeled as File evidence.
    let streamable = ordinary_streamable(spec);
    let explicit_text_encoder = spec.text_encoder.as_ref();
    let legacy_text_encoder = spec
        .components
        .get(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
    if explicit_text_encoder.is_some() && legacy_text_encoder.is_some() {
        return Err(gen_core::Error::Msg(format!(
            "{provider_id}: text encoder was supplied through both LoadSpec::text_encoder and legacy component '{}'",
            gen_core::COMFYUI_TEXT_ENCODER_COMPONENT
        )));
    }
    let selected_text_encoder = explicit_text_encoder.or(legacy_text_encoder);
    let components = match &spec.weights {
        gen_core::WeightsSource::Dir(root) => {
            // E1: the transformer and VAE are priced from the headers of the files the loader
            // opens, at the dtype/format it materializes them in — not from the on-disk sum of
            // `transformer/` and `vae/`, which recursed into the device-format cache and counted
            // the whole autoencoder as the render-resident decoder.
            let vae = snapshot_vae_bytes(root)?;
            let mut components = MaterializedComponents {
                base: PerComponentBytes {
                    // The weights-free fallback for an unauthored encoder; the validated
                    // materialization below replaces it whenever a config exists.
                    text_encoder: gen_core::safetensors_path_bytes(root.join("text_encoder")),
                    dit: snapshot_transformer_bytes(root)?,
                    vae: vae.resident,
                },
                reference_encoder: vae.reference_encoder,
            };
            let builtin = WeightsSource::Dir(root.join("text_encoder"));
            let effective = selected_text_encoder.unwrap_or(&builtin);
            if let Some(bytes) = validated_materialized_text_encoder_bytes(effective, false)? {
                components.base.text_encoder = bytes;
            } else if selected_text_encoder.is_some() {
                let roots = selected_encoder_discovery_roots(effective)?;
                components.base.text_encoder =
                    gen_core::encoder_contract::text_encoder_source_inventory_for_discovery(
                        effective, &roots,
                    )?
                    .source_bytes();
            }
            components
        }
        gen_core::WeightsSource::File(path) => imported_file_components(spec, path, provider_id)?,
    };
    let MaterializedComponents {
        base: components,
        reference_encoder,
    } = components;
    let formula = with_resident_components(
        MemoryFormulaKind::PhaseEnvelope {
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
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
        },
        reference_encoder_component(reference_encoder)
            .into_iter()
            .collect(),
        provider_id,
    )?;
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
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                        ..Default::default()
                    }
                }
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();

    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            // Packed q4/q8 `layers.N` projections are prepared once as content-addressed GGML
            // sidecars; each window maps and transfers only those already-device-format bytes.
            // Dense snapshots already transfer their stored tensor format directly.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        // PiD replaces the native VAE with a separately planned decoder. Until that decoder accepts
        // this provider's bounded host-decode route, the request safety gate rejects optimized PiD
        // runs.
        pid_decode_routes: None,
        load_shape: spec.load_shape,
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
            phases,
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula,
        calibration: Some(MemoryCalibrationIdentity::new(
            CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        // SC-22667 / E1: `base_bytes` is the text encoder + transformer + the render-resident
        // autoencoder; the lazily built f32 reference encoder is the one auxiliary network a plain
        // render never materializes, declared once in `overlay_bytes` with its typed component.
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: reference_encoder,
        },
        architecture_facts: architecture_facts(spec),
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// Explicit contract for the bespoke dual-network control routes. The control encoder, text encoder,
/// denoiser, and decoder are phase-loaded; both the base and control main stacks honor the selected
/// transformer window.
pub(crate) fn control_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = provider_contract(provider_id, spec)?;
    set_transformer_streamability(&mut contract, control_streamable(spec));
    let overlay_bytes = match spec.control.as_ref() {
        Some(gen_core::WeightsSource::Dir(path)) => gen_core::safetensors_path_bytes(path),
        Some(gen_core::WeightsSource::File(path)) => spec
            .read_file_unchanged_if_prepared(path, |p| -> gen_core::Result<u64> {
                Ok(gen_core::safetensors_path_bytes(p))
            })?,
        None => 0,
    };
    // SC-22660 / E1: the control network is declared exactly **once**, in `overlay_bytes`, with a
    // typed auxiliary `ControlBranch` component carrying the same total (mirroring the MLX Z-Image
    // control contract). `base_bytes` stays the base-model sum identity
    // `conditioning + transformer + decoder`, so it neither double-counts the overlay nor hides it
    // inside `transformer_bytes`. Previously the same bytes were declared on three legs at once and
    // `conditioning_bytes` was raised to `max(conditioning, decoder)` — a total borrowed from
    // another component, which is the exact defect the facts check now rejects. The overlay is
    // **added** to the base contract's, which already carries the reference encoder (SC-22667).
    contract.asset_facts.overlay_bytes = contract
        .asset_facts
        .overlay_bytes
        .saturating_add(overlay_bytes);
    if overlay_bytes > 0 {
        let control = gen_core::MemoryResidentComponent {
            id: CONTROL_COMPONENT_ID.to_owned(),
            kind: gen_core::MemoryComponentKind::ControlBranch,
            resident_bytes: overlay_bytes,
            // The control stack's own blocks ride the selected transformer window whenever the
            // base stack does; a non-streamable load bounds it by no rung at all.
            bounded_by: control_streamable(spec)
                .then_some(MemoryStrategy::BoundedTransformerResidency),
            residency: gen_core::MemoryComponentResidency::WholeRender,
        };
        contract.formula = with_resident_components(
            std::mem::replace(
                &mut contract.formula,
                MemoryFormulaKind::PhaseEnvelope {
                    phases: Vec::new(),
                    variables: Vec::new(),
                },
            ),
            vec![control],
            provider_id,
        )?;
    }
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CONTROL_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    Ok(contract)
}

pub(crate) fn request_scope(
    provider_id: &'static str,
    device: Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        candle_transformers::models::z_image::transformer::Config::z_image_turbo().n_layers,
        move |use_pid, tile_edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: PiD uses an alternate decoder whose explicit tile plan is not yet wired"
                )));
            }
            if tile_edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: native decode tiling is fixed at {DECODE_TILE_EDGE}/{DECODE_OVERLAP}, got {tile_edge}/{overlap}"
                )))
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
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
}

pub(crate) fn validate_context(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(provider_id, contract, context, loaded_quant)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: optimized memory strategies do not cover multi-phase denoise"
        )));
    }
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: PiD uses an alternate decode planner and cannot consume this native-VAE \
             memory selection"
        )));
    }
    Ok(())
}

/// The complete pre-request admission predicate shared by loaded generators and the weights-free
/// registry. Keep this as the decision-form adapter over [`validate_context`] so neither route can
/// admit a context that [`registered_begin_request`] (or a loaded generator's begin hook) rejects.
pub(crate) fn admission_safety_check(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(provider_id, contract, context, loaded_quant) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn safety_check(
    _provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: loaded_quant,
            component_precision_floors: &[],
        }),
        None,
    )
}

pub(crate) fn snapshot_quant_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<Option<Quant>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => return Ok(None),
    };
    crate::pipeline::packed_config_at(root, "transformer")
        .map_err(gen_core::Error::backend)?
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match snapshot_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => admission_safety_check(&contract.provider_id, contract, context, quant),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let is_control = contract.provider_id.ends_with("_control");
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: snapshot_quant_tier(spec, &contract.provider_id)?,
            component_precision_floors: &[],
        },
        gen_core::MemoryBehaviorRoute {
            mode: if is_control {
                gen_core::MemoryMode::ImageToImage
            } else {
                gen_core::MemoryMode::TextToImage
            },
            reference_count: u32::from(is_control),
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let quant = snapshot_quant_tier(spec, provider_id)?;
    validate_context(provider_id, contract, context, quant)?;
    Ok(Some(Box::new(request_scope(
        provider_id,
        Device::Cpu,
        contract,
        context,
    )?)))
}

#[cfg(test)]
fn generation_memory(
    contract: &MemoryProviderContract,
    selection: gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    contract.generation_memory(&selection)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-22727: `selected_encoder_discovery_roots` widened confinement to the Hugging Face cache
    /// **repository** so a snapshot's `blobs/` symlink targets are admitted. This crate's own
    /// widening is asserted here: the blobs target of an HF-cache-shaped snapshot is admitted, and
    /// a target symlinked outside `models--<org>--<repo>` is still refused.
    ///
    /// Unix-only: the fixture is built from relative symlinks, exactly as `huggingface_hub` lays a
    /// snapshot out. This test needs no GPU and runs on macOS.
    #[cfg(unix)]
    #[test]
    fn selected_encoder_discovery_roots_admit_cache_blobs_and_refuse_an_escape() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        // Author a contract-valid encoder, then relayout it as a cache snapshot whose files are
        // relative symlinks into the repository's sibling `blobs/` tree.
        let staged = temp.path().join("staged");
        gen_core_testkit::write_encoder_contract_fixture(&staged, crate::ENCODER_CONTRACT).unwrap();
        let repository = temp.path().join("hub/models--org--repo");
        let blobs = repository.join("blobs");
        let component = repository
            .join("snapshots/0123456789abcdef")
            .join("text_encoder");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&component).unwrap();
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&staged).unwrap() {
            let path = entry.unwrap().path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().unwrap().to_owned();
            let blob = format!("blob-{}", name.to_string_lossy());
            std::fs::copy(&path, blobs.join(&blob)).unwrap();
            symlink(
                std::path::Path::new("../../../blobs").join(&blob),
                component.join(&name),
            )
            .unwrap();
            names.push(name);
        }

        let source = WeightsSource::Dir(component.clone());
        let roots = selected_encoder_discovery_roots(&source).unwrap();
        assert!(
            roots.contains(&std::fs::canonicalize(&repository).unwrap()),
            "{roots:?}"
        );
        assert!(
            !roots.contains(&std::fs::canonicalize(temp.path().join("hub")).unwrap()),
            "{roots:?}"
        );
        crate::ENCODER_CONTRACT
            .validate_source_for_discovery(&source, &roots)
            .expect("a cache snapshot's blobs targets are inside the authorized repository");

        // Repoint one shard at a file outside `models--org--repo`: still refused.
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let shard = names
            .iter()
            .find(|name| name.to_string_lossy().ends_with(".safetensors"))
            .expect("fixture writes a safetensors shard")
            .clone();
        std::fs::copy(staged.join(&shard), outside.join(&shard)).unwrap();
        std::fs::remove_file(component.join(&shard)).unwrap();
        symlink(outside.join(&shard), component.join(&shard)).unwrap();
        let error = crate::ENCODER_CONTRACT
            .validate_source_for_discovery(&source, &roots)
            .unwrap_err()
            .to_string();
        assert!(error.contains("escapes authorized model roots"), "{error}");
    }
    use candle_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, Precision, WeightsSource,
    };

    fn spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    fn rung_four_support(contract: &MemoryProviderContract) -> MemoryStrategySupport {
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("rung four")
            .support
            .clone()
    }

    #[test]
    fn production_streamability_fails_closed_on_unsupported_load_axes() {
        let eligible = spec();
        assert!(ordinary_streamable(&eligible));

        let mut adapted = eligible.clone();
        adapted.adapters.push(gen_core::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        assert!(ordinary_streamable(&adapted), "plain Z replays adapters");

        let mut external_text_encoder = eligible.clone();
        external_text_encoder.text_encoder = Some(WeightsSource::Dir("/text-encoder".into()));
        assert!(
            ordinary_streamable(&external_text_encoder),
            "validated external text encoders remain a supported Z surface"
        );

        let mut identity = eligible.clone();
        identity.identity = Some(gen_core::IdentityWeights::default());
        let ineligible = [
            eligible.clone().with_quant(Quant::Q4),
            eligible.clone().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
            eligible
                .clone()
                .with_control(WeightsSource::File("/control.safetensors".into())),
            eligible
                .clone()
                .with_extra_control(WeightsSource::File("/control-2.safetensors".into())),
            eligible
                .clone()
                .with_ip_adapter(WeightsSource::File("/ip-adapter.safetensors".into())),
            identity,
            eligible.clone().with_component(
                "unknown",
                WeightsSource::File("/component.safetensors".into()),
            ),
            LoadSpec::new(WeightsSource::File("/model.safetensors".into()))
                .with_load_shape(LoadShape::DeferredMaterialization),
        ];
        for spec in ineligible {
            assert!(!ordinary_streamable(&spec), "must fail closed: {spec:?}");
        }

        let control = eligible
            .clone()
            .with_control(WeightsSource::File("/control.safetensors".into()));
        assert!(control_streamable(&control));
        assert!(
            !control_streamable(&eligible),
            "control source is mandatory"
        );
        assert!(!control_streamable(&adapted.with_control(
            WeightsSource::File("/control.safetensors".into())
        )));
    }

    #[test]
    fn selector_surface_normalizes_prepacked_tiers_and_requires_control_witness() {
        for surface in gen_core::candle_memory_contract_surface_specs() {
            let normalized = normalized_surface_spec(&surface).expect("valid common selector");
            assert_eq!(
                normalized.quantize,
                None,
                "{} must reach Z as an already-packed artifact tier",
                surface.selector.id()
            );
        }

        let mut q4 = gen_core::candle_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == gen_core::MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == gen_core::OffloadPolicy::Resident
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .expect("q4 resident deferred surface");
        let plain = weights_free_surface_contract(crate::MODEL_ID, &q4).unwrap();
        assert_eq!(
            rung_four_support(&plain),
            MemoryStrategySupport::Implemented
        );

        assert!(weights_free_control_surface_contract("z_image_turbo_control", &q4).is_err());
        q4.spec.control = Some(WeightsSource::Dir("/synthetic-control".into()));
        let control = weights_free_control_surface_contract("z_image_turbo_control", &q4).unwrap();
        assert_eq!(
            rung_four_support(&control),
            MemoryStrategySupport::Implemented
        );

        q4.spec.quantize = Some(Quant::Q8);
        assert!(
            weights_free_surface_contract(crate::MODEL_ID, &q4).is_err(),
            "selector/quant mutation must fail closed"
        );
    }

    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, usize)]) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, bytes) in tensors {
            let bytes = *bytes;
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [bytes],
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(vec![0_u8; offset]);
        std::fs::write(path, file).unwrap();
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
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.resize(file.len() + offset, 0);
        std::fs::write(path, file).unwrap();
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

    fn directory_spec(tmp: &tempfile::TempDir) -> (LoadSpec, std::path::PathBuf) {
        let root = tmp.path().join("snapshot");
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_typed_safetensors(
                &root.join(component).join("model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();
        (
            LoadSpec::new(WeightsSource::Dir(root.clone())),
            root.join("text_encoder/model.safetensors"),
        )
    }

    fn packed_directory_spec(tmp: &tempfile::TempDir, bits: i32) -> LoadSpec {
        let root = tmp.path().join(format!("packed-q{bits}"));
        for component in ["transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            write_typed_safetensors(
                &root.join(component).join("model.safetensors"),
                &[("probe", "BF16", &[1], 2)],
            );
        }
        std::fs::write(
            root.join("transformer/config.json"),
            format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        )
        .unwrap();
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
            Some(bits),
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root))
    }

    fn separate_file_spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        gen_core_testkit::write_encoder_contract_tokenizer_fixture(&base, crate::ENCODER_CONTRACT)
            .unwrap();
        let dit = tmp.path().join("dit.safetensors");
        let text_root = tmp.path().join("text-encoder");
        let vae = tmp.path().join("vae.safetensors");
        write_safetensors(&dit, &[("block.weight", 32)]);
        gen_core_testkit::write_encoder_contract_fixture(&text_root, crate::ENCODER_CONTRACT)
            .expect("validation-complete text encoder fixture");
        write_safetensors(&vae, &[("decoder.weight", 8)]);
        LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base))
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text_root.join("model.safetensors")),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae))
    }

    fn combined_file_spec(tmp: &tempfile::TempDir) -> (LoadSpec, std::path::PathBuf) {
        use std::io::Write as _;

        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        gen_core_testkit::write_encoder_contract_tokenizer_fixture(&base, crate::ENCODER_CONTRACT)
            .unwrap();

        let encoder_root = tmp.path().join("encoder-fixture");
        gen_core_testkit::write_encoder_contract_fixture(&encoder_root, crate::ENCODER_CONTRACT)
            .unwrap();
        let encoder_headers = gen_core::weightsmeta::safetensors_path_tensor_headers(
            encoder_root.join("model.safetensors"),
        )
        .unwrap();

        let combined = tmp.path().join("combined.safetensors");
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        for tensor in encoder_headers {
            let end = offset.checked_add(tensor.data_bytes).unwrap();
            header.insert(
                format!("conditioner.embedders.0.transformer.{}", tensor.name),
                serde_json::json!({
                    "dtype": format!("{:?}", tensor.dtype),
                    "shape": tensor.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        for (name, bytes) in [
            ("model.diffusion_model.block.weight", 2_u64),
            ("first_stage_model.decoder.weight", 2_u64),
        ] {
            let end = offset.checked_add(bytes).unwrap();
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [bytes],
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(&combined).unwrap();
        // The combined checkpoint carries the encoder's whole multi-GB payload span. Flag it
        // between the create and the extend, or NTFS allocates all of it (see `mark_sparse`).
        gen_core_testkit::mark_sparse(&combined);
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + offset).unwrap();

        (
            LoadSpec::new(WeightsSource::File(combined.clone()))
                .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base)),
            combined,
        )
    }

    #[test]
    fn file_asset_facts_follow_bf16_materialization_and_omit_fp8_companions() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        let dit = tmp.path().join("dit.safetensors");
        let text = tmp.path().join("text.safetensors");
        let vae = tmp.path().join("vae.safetensors");
        write_typed_safetensors(
            &dit,
            &[
                ("block.weight", "F8_E4M3", &[2, 4], 8),
                ("block.weight_scale", "F32", &[], 4),
                ("dense.weight", "F32", &[3], 12),
            ],
        );
        write_typed_safetensors(&text, &[("layer.weight", "F32", &[3], 12)]);
        write_typed_safetensors(&vae, &[("decoder.weight", "BF16", &[4], 8)]);
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base))
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae));
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.asset_facts.transformer_bytes, 16 + 6);
        assert_eq!(contract.asset_facts.conditioning_bytes, 6);
        assert_eq!(contract.asset_facts.decoder_bytes, 8);
        assert_eq!(contract.asset_facts.base_bytes, 36);
    }

    #[test]
    fn file_contract_prices_every_loadable_typed_source_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let valid = separate_file_spec(&tmp);
        let mut accepted = vec![("valid", valid.clone())];

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        accepted.push(("precision-is-accepted", precision));
        let mut adapter = valid.clone();
        adapter.adapters.push(gen_core::AdapterSpec::new(
            tmp.path().join("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        accepted.push(("adapter-is-accepted", adapter));
        let mut external_te = valid.clone();
        let external_te_root = tmp.path().join("external-te");
        gen_core_testkit::write_encoder_contract_fixture(
            &external_te_root,
            crate::ENCODER_CONTRACT,
        )
        .expect("validation-complete typed text encoder fixture");
        external_te
            .components
            .remove(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
        external_te.text_encoder = Some(WeightsSource::Dir(external_te_root));
        accepted.push(("external-text-encoder", external_te));

        for (name, spec) in accepted {
            crate::validate_load_spec(&spec)
                .unwrap_or_else(|error| panic!("load gate rejected {name}: {error}"));
            provider_contract(crate::MODEL_ID, &spec)
                .unwrap_or_else(|error| panic!("memory contract rejected {name}: {error}"));
        }
    }

    #[test]
    fn imported_directory_encoder_pricing_ignores_nested_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = separate_file_spec(&tmp);
        spec.components
            .remove(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
        let external = tmp.path().join("external-text-encoder");
        gen_core_testkit::write_encoder_contract_fixture(&external, crate::ENCODER_CONTRACT)
            .unwrap();
        spec.text_encoder = Some(WeightsSource::Dir(external.clone()));
        let baseline = provider_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .asset_facts
            .conditioning_bytes;

        let nested = external.join("archive");
        std::fs::create_dir_all(&nested).unwrap();
        write_safetensors(
            &nested.join("not-a-direct-shard.safetensors"),
            &[("extra", 4096)],
        );

        assert_eq!(
            provider_contract(crate::MODEL_ID, &spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes,
            baseline
        );
    }

    fn assert_conditioning_ignores_unmaterialized_language_tensors(
        spec: &LoadSpec,
        text_encoder_path: &std::path::Path,
        route: &str,
        key_prefix: &str,
    ) {
        let conditioning = || {
            provider_contract(crate::MODEL_ID, spec)
                .unwrap()
                .asset_facts
                .conditioning_bytes
        };
        let baseline = conditioning();
        for (name, shape) in [
            (
                "model.norm.weight",
                vec![crate::ENCODER_CONTRACT.hidden_size],
            ),
            ("model.unrelated_projection.weight", vec![17]),
        ] {
            let name = format!("{key_prefix}{name}");
            append_sparse_f16_tensor(text_encoder_path, &name, &shape);
            assert_eq!(
                conditioning(),
                baseline,
                "{route} charged an unmaterialized tensor {name}"
            );
        }
    }

    #[test]
    fn conditioning_prices_only_the_materialized_36_layer_language_surface() {
        let directory = tempfile::tempdir().unwrap();
        let (directory_spec, directory_encoder) = directory_spec(&directory);
        assert_conditioning_ignores_unmaterialized_language_tensors(
            &directory_spec,
            &directory_encoder,
            "snapshot directory",
            "",
        );

        let imported = tempfile::tempdir().unwrap();
        let imported_spec = separate_file_spec(&imported);
        assert_conditioning_ignores_unmaterialized_language_tensors(
            &imported_spec,
            &imported.path().join("text-encoder/model.safetensors"),
            "imported component file",
            "",
        );

        let configless = tempfile::tempdir().unwrap();
        let configless_spec = separate_file_spec(&configless);
        std::fs::remove_file(configless.path().join("text-encoder/config.json")).unwrap();
        assert_conditioning_ignores_unmaterialized_language_tensors(
            &configless_spec,
            &configless.path().join("text-encoder/model.safetensors"),
            "configless ComfyUI component file",
            "",
        );

        let combined = tempfile::tempdir().unwrap();
        let (combined_spec, combined_file) = combined_file_spec(&combined);
        crate::validate_load_spec(&combined_spec).expect("combined route is runtime-admissible");
        assert_conditioning_ignores_unmaterialized_language_tensors(
            &combined_spec,
            &combined_file,
            "combined ComfyUI checkpoint",
            "conditioner.embedders.0.transformer.",
        );
    }

    #[test]
    fn packed_conditioning_prices_the_runtime_qtensor_format_once() {
        let contract = crate::ENCODER_CONTRACT;
        let attention_width = contract.num_attention_heads * contract.head_dim;
        let kv_width = contract.num_key_value_heads * contract.head_dim;
        let matrix_elements = contract.vocab_size * contract.hidden_size
            + contract.loaded_hidden_layers
                * (2 * attention_width * contract.hidden_size
                    + 2 * kv_width * contract.hidden_size
                    + 3 * contract.intermediate_size * contract.hidden_size);
        let dense_vector_bytes =
            contract.loaded_hidden_layers * (2 * contract.hidden_size + 2 * contract.head_dim) * 2;

        for (bits, bytes_per_block) in [(4, 20_u64), (8, 34_u64)] {
            let tmp = tempfile::tempdir().unwrap();
            let spec = packed_directory_spec(&tmp, bits);
            let expected = u64::try_from(matrix_elements / candle_gen::quant::QUANT_BLOCK).unwrap()
                * bytes_per_block
                + u64::try_from(dense_vector_bytes).unwrap();
            assert_eq!(
                provider_contract(crate::MODEL_ID, &spec)
                    .unwrap()
                    .asset_facts
                    .conditioning_bytes,
                expected,
                "Q{bits} must count each Q4_1/Q8_0 tensor and no transient affine sidecars"
            );
        }
    }

    #[test]
    fn weights_free_contract_discovery_does_not_claim_source_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = separate_file_spec(&tmp);
        spec.components
            .remove(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
        let incompatible = tmp.path().join("wrong-text-encoder.safetensors");
        write_safetensors(&incompatible, &[("layer.weight", 16)]);
        spec.text_encoder = Some(WeightsSource::File(incompatible));

        assert!(
            provider_contract(crate::MODEL_ID, &spec).is_ok(),
            "catalog discovery must remain weights-free"
        );
        let error = crate::validate_load_spec(&spec)
            .expect_err("the executable load seam must reject a missing selected encoder")
            .to_string();
        assert!(error.contains("text encoder"), "got: {error}");
    }

    #[test]
    fn weights_free_contract_does_not_require_a_builtin_encoder_to_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nonexistent");
        let spec = LoadSpec::new(WeightsSource::Dir(missing.clone()));

        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap_or_else(|error| {
            panic!(
                "weights-free catalog construction must not require or canonicalize {}: {error}",
                missing.join("text_encoder").display()
            )
        });
        assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
    }

    #[test]
    fn combined_and_separate_imports_publish_the_same_exact_phase_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        let tokenizer = tmp.path().join("tokenizer-base");
        std::fs::create_dir_all(&tokenizer).unwrap();
        let combined = tmp.path().join("combined.safetensors");
        write_safetensors(
            &combined,
            &[
                ("model.diffusion_model.block.weight", 11),
                // The combined loader strips the component prefix, then `normalize_fp8_map`
                // discards this exact marker. Asset facts must not count its source payload.
                ("model.diffusion_model.scaled_fp8", 13),
                ("conditioner.embedders.0.transformer.layer.weight", 7),
                ("first_stage_model.decoder.weight", 5),
            ],
        );
        let combined_spec = LoadSpec::new(WeightsSource::File(combined)).with_component(
            gen_core::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(tokenizer.clone()),
        );
        let combined_contract = provider_contract(crate::MODEL_ID, &combined_spec).unwrap();

        let dit = tmp.path().join("dit.safetensors");
        let text_encoder = tmp.path().join("text-encoder.safetensors");
        let vae = tmp.path().join("vae.safetensors");
        write_safetensors(&dit, &[("block.weight", 11)]);
        write_safetensors(&text_encoder, &[("layer.weight", 7)]);
        write_safetensors(&vae, &[("decoder.weight", 5)]);
        let separate_spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                gen_core::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(tokenizer),
            )
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text_encoder),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae));
        let separate_contract = provider_contract(crate::MODEL_ID, &separate_spec).unwrap();

        for contract in [&combined_contract, &separate_contract] {
            assert_eq!(contract.asset_facts.transformer_bytes, 11);
            assert_eq!(contract.asset_facts.conditioning_bytes, 7);
            assert_eq!(contract.asset_facts.decoder_bytes, 5);
            assert_eq!(contract.asset_facts.base_bytes, 23);
        }
    }

    #[test]
    fn combined_inventory_fails_closed_on_missing_or_unmapped_components() {
        let tmp = tempfile::tempdir().unwrap();
        let tokenizer = tmp.path().join("tokenizer-base");
        std::fs::create_dir_all(&tokenizer).unwrap();
        for (name, tensors, expected) in [
            (
                "missing-vae.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                ],
                "missing the VAE",
            ),
            (
                "unknown.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                    ("first_stage_model.decoder.weight", 5),
                    ("mystery.weight", 3),
                ],
                "no component mapping",
            ),
            (
                "collision.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("transformer.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                    ("first_stage_model.decoder.weight", 5),
                ],
                "collides",
            ),
        ] {
            let checkpoint = tmp.path().join(name);
            write_safetensors(&checkpoint, &tensors);
            let spec = LoadSpec::new(WeightsSource::File(checkpoint)).with_component(
                gen_core::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(tokenizer.clone()),
            );
            let error = provider_contract(crate::MODEL_ID, &spec)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn z_image_base_load_and_memory_contract_reject_the_same_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z-image.safetensors".into()));
        let load_error = crate::base::load(&spec)
            .err()
            .expect("base generator must reject File")
            .to_string();
        let contract_error = provider_contract(crate::base::MODEL_ID, &spec)
            .unwrap_err()
            .to_string();
        assert_eq!(contract_error, load_error);
        assert!(contract_error.contains("snapshot directory"));
    }

    #[test]
    fn plain_contract_shape_controls_streaming_and_rung_four() {
        let deferred = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        assert_eq!(deferred.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            deferred.calibration.as_ref().unwrap().load_shape,
            LoadShape::DeferredMaterialization
        );
        assert!(deferred.lifecycle.transformer_window_materialization);
        assert!(matches!(
            deferred
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));

        let mut eager_spec = spec();
        eager_spec.load_shape = LoadShape::EagerMaterialization;
        let eager = provider_contract(crate::MODEL_ID, &eager_spec).unwrap();
        assert_eq!(eager.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            eager.calibration.as_ref().unwrap().load_shape,
            LoadShape::EagerMaterialization
        );
        assert!(!eager.lifecycle.transformer_window_materialization);
        assert!(matches!(
            eager
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        ));
    }

    #[test]
    fn weights_free_behavior_uses_the_cpu_scope_path() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        let mut fixture =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
        let mut scope =
            registered_begin_request(crate::MODEL_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        assert_eq!(
            fixture.request.memory,
            contract.generation_memory(&fixture.context.selection)
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }

    /// The DiT/VAE config fields exactly as the published `SceneWorks/z-image-turbo` snapshot ships
    /// them (verified against the on-disk q4 snapshot). Only the keys the architecture facts read
    /// are reproduced; the values are the snapshot's, and the assertions below derive nothing from
    /// the model id.
    fn write_snapshot_component_configs(root: &std::path::Path) {
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            br#"{
                "_class_name": "ZImageTransformer2DModel",
                "all_f_patch_size": [1],
                "all_patch_size": [2],
                "axes_dims": [32, 48, 48],
                "cap_feat_dim": 2560,
                "dim": 3840,
                "in_channels": 16,
                "n_heads": 30,
                "n_kv_heads": 30,
                "n_layers": 30,
                "n_refiner_layers": 2
            }"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("vae")).unwrap();
        std::fs::write(
            root.join("vae/config.json"),
            br#"{
                "_class_name": "AutoencoderKL",
                "block_out_channels": [128, 256, 512, 512],
                "in_channels": 3,
                "latent_channels": 16,
                "layers_per_block": 2,
                "out_channels": 3,
                "sample_size": 1024
            }"#,
        )
        .unwrap();
    }

    fn snapshot_spec(root: std::path::PathBuf) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    /// AC: both Z-Image providers publish the architecture facts of the pipeline the loader
    /// actually builds — `DitConfig::z_image_turbo()` and `VaeConfig::z_image()` — and the
    /// resulting contract passes the shared facts conformance check.
    #[test]
    fn architecture_facts_are_the_loader_presets() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("z-image-turbo");
        write_snapshot_component_configs(&root);
        let spec = snapshot_spec(root.clone());
        for id in [crate::MODEL_ID, crate::base::MODEL_ID] {
            let contract = provider_contract(id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(30),
                    // `DitConfig::z_image_turbo()`: `dim 3840 / n_heads 30`.
                    head_dim: Some(128),
                    transformer_blocks: Some(30),
                    patch_size: Some(2),
                    // `VaeConfig::z_image().latent_channels`.
                    latent_channels: Some(16),
                    // Four `block_out_channels` stages => three halvings => x8.
                    vae_spatial_scale: Some(8),
                    // Z-Image ships the FLUX.1 image VAE: no temporal axis exists to declare.
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(2),
                },
                "{id} architecture facts"
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        // The published values ARE the preset's fields, not literals that happen to agree with it.
        let dit = DitConfig::z_image_turbo();
        let vae = VaeConfig::z_image();
        let facts = provider_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .architecture_facts;
        assert_eq!(facts.attention_heads, Some(dit.n_heads as u32));
        assert_eq!(facts.head_dim, Some(dit.head_dim() as u32));
        assert_eq!(facts.transformer_blocks, Some(dit.n_layers as u32));
        assert_eq!(facts.patch_size, Some(dit.all_patch_size[0] as u32));
        assert_eq!(facts.latent_channels, Some(vae.latent_channels as u32));
        assert_eq!(
            facts.vae_spatial_scale,
            Some(1 << (vae.block_out_channels.len() - 1))
        );

        // AC 1 is phrased in terms of the published `Generator::memory_strategy_contract()` surface,
        // not the crate-internal builder. A real load over the same snapshot must expose the exact
        // same facts, so the axis cannot be published on a contract nobody outside the crate sees.
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .expect("valid encoder fixture");
        let spec = snapshot_spec(root);
        let expected = provider_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .architecture_facts;
        for (label, generator) in [
            ("turbo", crate::load(&spec).unwrap()),
            ("base", crate::base::load(&spec).unwrap()),
        ] {
            let published = generator
                .memory_strategy_contract()
                .expect("unit-test loads retain their memory contract");
            assert_eq!(
                published.architecture_facts, expected,
                "{label}: memory_strategy_contract() must publish the config-derived facts"
            );
            assert!(published
                .architecture_facts
                .has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(published);
        }
    }

    /// Feature-end review (SC-22667, E2): the published facts are the **loader's**, and a
    /// divergent snapshot `config.json` does not change them. This is stated honestly rather than
    /// as a virtue: `pipeline.rs` hardcodes `DitConfig::z_image_turbo()` and `VaeConfig::z_image()`
    /// and reads the component configs only for their `quantization` block, so a snapshot whose
    /// config says `n_layers: 24` still loads — and must be priced as — a 30-block trunk. The
    /// previous test asserted the opposite ("a mutated config follows"), which described a
    /// pipeline this crate never constructs.
    ///
    /// Mutation that fails this: reading `n_layers` / `n_heads` / `latent_channels` back out of
    /// the component configs (the shape under review) — the divergent values below then surface.
    #[test]
    fn a_divergent_snapshot_config_does_not_change_the_loader_facts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("z-image-divergent");
        write_snapshot_component_configs(&root);
        let reference = provider_contract(crate::MODEL_ID, &snapshot_spec(root.clone()))
            .unwrap()
            .architecture_facts;

        let mut dit: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("transformer/config.json")).unwrap())
                .unwrap();
        dit["n_layers"] = serde_json::json!(24);
        dit["n_heads"] = serde_json::json!(7);
        dit["all_patch_size"] = serde_json::json!([4]);
        dit["in_channels"] = serde_json::json!(4);
        std::fs::write(
            root.join("transformer/config.json"),
            serde_json::to_vec(&dit).unwrap(),
        )
        .unwrap();
        let mut vae: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("vae/config.json")).unwrap()).unwrap();
        vae["latent_channels"] = serde_json::json!(4);
        vae["block_out_channels"] = serde_json::json!([1, 2, 3, 4, 5, 6, 7]);
        std::fs::write(
            root.join("vae/config.json"),
            serde_json::to_vec(&vae).unwrap(),
        )
        .unwrap();

        let facts = provider_contract(crate::MODEL_ID, &snapshot_spec(root.clone()))
            .unwrap()
            .architecture_facts;
        assert_eq!(
            facts, reference,
            "the loader ignores these keys, so the published facts must too"
        );
        assert_eq!(facts.transformer_blocks, Some(30));
        assert_eq!(facts.attention_heads, Some(30));
        assert_eq!(facts.latent_channels, Some(16));
        assert_eq!(facts.vae_spatial_scale, Some(8));

        // Nor does the absence of a component config: only the presence of a resolved snapshot
        // directory gates the facts, because that is all the loader needs to build its presets.
        std::fs::remove_file(root.join("vae/config.json")).unwrap();
        std::fs::remove_file(root.join("transformer/config.json")).unwrap();
        let facts = provider_contract(crate::MODEL_ID, &snapshot_spec(root))
            .unwrap()
            .architecture_facts;
        assert_eq!(facts, reference);
    }

    /// A contract built before any asset exists on disk declares every axis absent rather than
    /// fabricating the reference architecture.
    #[test]
    fn weights_free_contracts_publish_absent_architecture_facts() {
        for id in [crate::MODEL_ID, crate::base::MODEL_ID] {
            let contract = provider_contract(id, &spec()).unwrap();
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts::default(),
                "{id} weights-free facts"
            );
            assert!(contract.architecture_facts.is_empty());
        }
    }

    #[test]
    fn plain_contract_is_conformant_and_exposes_every_candidate_range() {
        for id in [crate::MODEL_ID, crate::base::MODEL_ID] {
            let contract = provider_contract(id, &spec()).unwrap();
            // The weights-free contract cannot satisfy the E2 architecture gate — there is no
            // config.json to read — but its byte decomposition is still a claim (E1). Checked
            // first so this entry point, not the shared conformance walk, is what reports a
            // dishonest decomposition here.
            gen_core_testkit::assert_memory_contract_asset_facts_conform(&contract);
            assert!(contract.architecture_facts.is_empty());
            assert!(contract.conformance_errors().is_empty());
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges,
                vec![DECODE_TILE_EDGE]
            );
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
    }

    fn rung_four_selection() -> MemorySelection {
        MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(4),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    fn context(contract: &MemoryProviderContract) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        MemoryRunContext {
            optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: rung_four_selection(),
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "sc-15815-test".to_owned(),
        }
    }

    #[test]
    fn rung_four_selection_maps_every_engaged_control_and_preserves_the_window() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let selection = rung_four_selection();
        contract.validate_selection(&selection).unwrap();
        let memory = generation_memory(&contract, selection).unwrap();
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(4));
    }

    #[test]
    fn request_scope_applies_exact_parameters_and_finishes_once() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let context = context(&contract);
        validate_context(crate::MODEL_ID, &contract, &context, None).unwrap();
        let mut scope = request_scope(crate::MODEL_ID, Device::Cpu, &contract, &context).unwrap();
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
            .unwrap();
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        scope.materialize_transformer_window(0, 4).unwrap();
        scope.materialize_transformer_window(28, 2).unwrap();
        let mut request = GenerationRequest {
            width: 768,
            height: 768,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            request.memory,
            generation_memory(&contract, context.selection)
        );
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
    }

    #[test]
    fn registered_behavior_treats_admitted_batch_as_a_maximum() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        let mut context = context(&contract);
        context.geometry.batch = 3;
        let mut scope = (crate::TURBO_MEMORY_BEHAVIOR.begin_request)(&spec, &contract, &context)
            .unwrap()
            .expect("registered behavior must construct the Candle scope core");
        let mut prefix = GenerationRequest {
            width: 768,
            height: 768,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut prefix).unwrap();
        scope
            .configure_decode(
                DECODE_TILE_EDGE,
                DECODE_OVERLAP,
                MemoryGeometry {
                    batch: 1,
                    ..context.geometry
                },
            )
            .unwrap();
        for geometry in [
            MemoryGeometry {
                width: context.geometry.width / 2,
                ..context.geometry
            },
            MemoryGeometry {
                height: context.geometry.height / 2,
                ..context.geometry
            },
            MemoryGeometry {
                frames: context.geometry.frames + 1,
                ..context.geometry
            },
            MemoryGeometry {
                reference_count: 1,
                ..context.geometry
            },
            MemoryGeometry {
                batch: 0,
                ..context.geometry
            },
            MemoryGeometry {
                batch: 4,
                ..context.geometry
            },
        ] {
            assert!(scope
                .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, geometry)
                .is_err());
        }
        scope.materialize_transformer_window(0, 4).unwrap();
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        assert!(scope.materialize_transformer_window(1, 4).is_err());
        let mut overflow = GenerationRequest {
            width: 768,
            height: 768,
            count: 4,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut overflow).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.configure_request(&mut prefix).is_err());
    }

    #[test]
    fn stale_fingerprint_and_pid_optimized_routes_are_rejected_before_execution() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let mut stale = context(&contract);
        stale.calibration_fingerprint.push_str("-stale");
        assert!(validate_context(crate::MODEL_ID, &contract, &stale, None).is_err());

        let mut pid = context(&contract);
        pid.use_pid = true;
        let error = validate_context(crate::MODEL_ID, &contract, &pid, None).unwrap_err();
        assert!(error.to_string().contains("PiD"));
    }

    #[test]
    fn registered_admission_matches_loaded_begin_request_policy() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();

        let mut phases = context(&contract);
        phases.has_phases = true;
        let error = validate_context(crate::MODEL_ID, &contract, &phases, None).unwrap_err();
        assert!(error.to_string().contains("multi-phase"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &phases),
            admission_safety_check(crate::MODEL_ID, &contract, &phases, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &phases),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));

        let mut pid = context(&contract);
        pid.use_pid = true;
        let error = validate_context(crate::MODEL_ID, &contract, &pid, None).unwrap_err();
        assert!(error.to_string().contains("PiD"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &pid),
            admission_safety_check(crate::MODEL_ID, &contract, &pid, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &pid),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));

        let mut stale = context(&contract);
        stale.calibration_fingerprint.push_str("-stale");
        let error = validate_context(crate::MODEL_ID, &contract, &stale, None).unwrap_err();
        assert!(error.to_string().contains("calibration handshake mismatch"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &stale),
            admission_safety_check(crate::MODEL_ID, &contract, &stale, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &stale),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));
    }

    #[test]
    fn packed_q4_and_q8_snapshots_bind_weights_free_admission_to_the_detected_tier() {
        for (bits, actual, wrong) in [
            (4, Quant::Q4, Some(Quant::Q8)),
            (8, Quant::Q8, Some(Quant::Q4)),
        ] {
            let root_tmp = tempfile::tempdir().unwrap();
            let root = root_tmp.path().to_path_buf();
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            for (provider_id, control) in [
                (crate::MODEL_ID, false),
                (crate::base::MODEL_ID, false),
                ("z_image_turbo_control", true),
                ("z_image_control", true),
            ] {
                let contract = if control {
                    control_contract(provider_id, &spec).unwrap()
                } else {
                    provider_contract(provider_id, &spec).unwrap()
                };
                let mut actual_context = context(&contract);
                actual_context.selection.strategy = MemoryStrategy::Resident;
                actual_context.selection.parameters = Default::default();
                actual_context.selection.tier.quant = Some(actual);
                assert_eq!(
                    registered_safety_check(&spec, &contract, &actual_context),
                    MemorySafetyDecision::Accept
                );
                for selected in [None, wrong] {
                    let mut wrong_context = context(&contract);
                    wrong_context.selection.tier.quant = selected;
                    assert!(matches!(
                        registered_safety_check(&spec, &contract, &wrong_context),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("does not match loaded tier")
                    ));
                    assert!(matches!(
                        safety_check(provider_id, &contract, &wrong_context, Some(actual)),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("does not match loaded tier")
                    ));
                }
            }
        }
    }

    #[test]
    fn adapters_preserve_the_full_streamed_transformer_ladder() {
        let mut spec = spec();
        spec.adapters.push(gen_core::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
    }

    #[test]
    fn block_materialization_reports_device_format_transfer_for_dense_and_packed_snapshots() {
        let dense = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        assert!(matches!(
            dense.backend,
            MemoryBackendRealization::CandleCuda {
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
                ..
            }
        ));

        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            br#"{"quantization":{"group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let packed_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let packed = provider_contract(crate::MODEL_ID, &packed_spec).unwrap();
        assert!(matches!(
            packed.backend,
            MemoryBackendRealization::CandleCuda {
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
                ..
            }
        ));
    }

    /// A *materialized* control fixture: the base snapshot's component configs (so the architecture
    /// facts are read rather than defaulted) plus a control directory holding a real
    /// `.safetensors` file, so `safetensors_path_bytes` returns a non-zero overlay and every
    /// overlay assertion below is load-bearing.
    fn materialized_control_fixture(tmp: &tempfile::TempDir) -> (LoadSpec, u64) {
        let root = tmp.path().join("z-image-control-snapshot");
        write_snapshot_component_configs(&root);
        // Distinct, non-zero base components: a decomposition check over three zeroes proves
        // nothing, and a decoder total larger than the conditioning total is what makes a
        // `conditioning.max(decoder)` fold observable.
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .expect("valid encoder fixture");
        write_safetensors(
            &root.join("transformer/diffusion_pytorch_model.safetensors"),
            &[("blocks.0.weight", 8192)],
        );
        write_safetensors(
            &root.join("vae/diffusion_pytorch_model.safetensors"),
            &[("decoder.conv_in.weight", 1_000_000)],
        );
        let control = root.join("control");
        std::fs::create_dir_all(&control).unwrap();
        write_safetensors(
            &control.join("diffusion_pytorch_model.safetensors"),
            &[
                ("control.block.0.weight", 4096),
                ("control.embed.weight", 512),
            ],
        );
        let overlay_bytes = gen_core::safetensors_path_bytes(&control);
        assert!(
            overlay_bytes > 0,
            "the control fixture must be materialized; a nonexistent path makes every overlay \
             assertion vacuous"
        );
        (
            snapshot_spec(root).with_control(WeightsSource::Dir(control)),
            overlay_bytes,
        )
    }

    /// The control network is declared **once**, in `overlay_bytes` plus one typed auxiliary
    /// `ControlBranch` component. `base_bytes` keeps the base-model sum identity, so the same bytes
    /// are never counted on three legs at once (SC-22657 E1).
    #[test]
    fn control_contract_declares_the_control_network_exactly_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (spec, overlay_bytes) = materialized_control_fixture(&tmp);
        for id in ["z_image_turbo_control", "z_image_control"] {
            let plain = provider_contract(id, &spec).unwrap();
            let contract = control_contract(id, &spec).unwrap();
            let facts = &contract.asset_facts;

            // Non-vacuity: every base component is separately priced, so the decomposition
            // assertions below are over three distinct non-zero totals rather than three zeroes.
            assert!(facts.conditioning_bytes > 0, "{id} conditioning bytes");
            assert!(facts.transformer_bytes > 0, "{id} transformer bytes");
            assert!(facts.decoder_bytes > 0, "{id} decoder bytes");

            assert_eq!(facts.overlay_bytes, overlay_bytes, "{id} overlay bytes");
            assert_eq!(
                facts.base_bytes,
                facts.conditioning_bytes + facts.transformer_bytes + facts.decoder_bytes,
                "{id}: base_bytes must stay the base-model sum identity"
            );
            assert_eq!(
                facts.base_bytes, plain.asset_facts.base_bytes,
                "{id}: the overlay must not be folded into base_bytes"
            );
            assert_eq!(
                facts.transformer_bytes, plain.asset_facts.transformer_bytes,
                "{id}: the overlay must not be folded into transformer_bytes"
            );
            assert_eq!(
                facts.conditioning_bytes, plain.asset_facts.conditioning_bytes,
                "{id}: conditioning_bytes must stay the text-encoder total"
            );

            let components = contract.resident_components();
            assert_eq!(components.len(), 1, "{id} resident components");
            assert_eq!(components[0].id, CONTROL_COMPONENT_ID);
            assert_eq!(
                components[0].kind,
                gen_core::MemoryComponentKind::ControlBranch
            );
            assert_eq!(components[0].resident_bytes, overlay_bytes);
            assert_eq!(
                components[0].bounded_by,
                Some(MemoryStrategy::BoundedTransformerResidency)
            );
            assert_eq!(contract.auxiliary_resident_bytes(), overlay_bytes);
            assert_eq!(
                contract.total_resident_bytes(),
                facts.base_bytes + overlay_bytes,
                "{id}: a warm control provider holds the base model and the control network"
            );

            assert!(
                contract.conformance_errors().is_empty(),
                "{id}: {:?}",
                contract.conformance_errors()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// `conditioning_bytes` is the **text-encoder** total, full stop. The retired shape raised it to
    /// `max(conditioning, decoder)`, which on a snapshot whose decoder outweighs its conditioning
    /// stack borrows the decoder's bytes for a component that does not hold them — the exact E1
    /// defect the facts check exists to reject. This fixture ships a priced decoder and no text
    /// encoder, so that fold is observable rather than the no-op it is on a full snapshot.
    #[test]
    fn control_conditioning_bytes_never_borrow_the_decoder_total() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("z-image-no-encoder");
        write_snapshot_component_configs(&root);
        write_safetensors(
            &root.join("vae/diffusion_pytorch_model.safetensors"),
            &[("decoder.conv_in.weight", 1_000_000)],
        );
        let control = root.join("control");
        std::fs::create_dir_all(&control).unwrap();
        write_safetensors(&control.join("control.safetensors"), &[("w", 4096)]);
        let spec = snapshot_spec(root).with_control(WeightsSource::Dir(control));

        for id in ["z_image_turbo_control", "z_image_control"] {
            let contract = control_contract(id, &spec).unwrap();
            let facts = &contract.asset_facts;
            assert!(facts.decoder_bytes > 0, "{id}: the decoder must be priced");
            assert_eq!(
                facts.conditioning_bytes, 0,
                "{id}: no text encoder is present, so no conditioning bytes exist to declare"
            );
            assert!(
                contract.conformance_errors().is_empty(),
                "{id}: {:?}",
                contract.conformance_errors()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// SC-22667 / E1 (DEFECT 1): a packed transformer that has been block-streamed carries the
    /// device-format cache beside its source file. `transformer_bytes` is the set the loader
    /// holds — one GGML `Q4_1`/`Q8_0` QTensor per packed projection plus the dense residual at
    /// bf16 — and that set is the same whether the bytes are repacked from the source or copied
    /// out of the cache. Summing the two files priced the real q4 snapshot at 7.31 GB against a
    /// 3.47 GB checkpoint, and the resident rung's measured level then sat *below* the contract.
    #[test]
    fn transformer_bytes_price_the_loaded_checkpoint_not_the_device_format_cache() {
        for (bits, packed_columns, bytes_per_block) in [(4_i32, 8_usize, 20_u64), (8, 16, 34)] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().join(format!("packed-q{bits}-streamed"));
            let transformer = root.join("transformer");
            std::fs::create_dir_all(&transformer).unwrap();
            std::fs::write(
                transformer.join("config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let code_bytes = 2 * packed_columns * 4;
            write_typed_safetensors(
                &transformer.join("model.safetensors"),
                &[
                    (
                        "layers.0.attention.qkv.weight",
                        "U32",
                        &[2, packed_columns],
                        code_bytes,
                    ),
                    ("layers.0.attention.qkv.scales", "BF16", &[2, 1], 4),
                    ("layers.0.attention.qkv.biases", "BF16", &[2, 1], 4),
                    ("x_embedder.weight", "BF16", &[3], 6),
                ],
            );
            std::fs::create_dir_all(root.join("vae")).unwrap();
            write_typed_safetensors(
                &root.join("vae/model.safetensors"),
                &[("decoder.conv_in.weight", "BF16", &[1], 2)],
            );
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &root.join("text_encoder"),
                crate::ENCODER_CONTRACT,
                Some(bits),
            )
            .unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            // 2 x 64 elements = 4 GGML blocks, plus the 3-element bf16 residual.
            let expected = 4 * bytes_per_block + 6;
            let cold = provider_contract(crate::MODEL_ID, &spec).unwrap();
            assert_eq!(
                cold.asset_facts.transformer_bytes, expected,
                "Q{bits}: a never-streamed snapshot prices the GGML set from the source header"
            );

            // A prior block-streamed load left the content-addressed GGML sidecars behind.
            let cache = transformer.join(gen_core::CANDLE_DEVICE_FORMAT_CACHE_DIR);
            std::fs::create_dir_all(&cache).unwrap();
            write_typed_safetensors(
                &cache.join(format!(
                    "0104af902ed702222b65c9b031dfe9f905e5335d7c1a624200a66e2ea7718102.q{bits}_{}.safetensors",
                    if bits == 4 { 1 } else { 0 }
                )),
                &[("weight", "U8", &[4096], 4096)],
            );
            assert!(
                gen_core::safetensors_path_bytes(&cache) > 4096,
                "the cache fixture must be a real weight file or the assertion below is vacuous"
            );
            let warm = provider_contract(crate::MODEL_ID, &spec).unwrap();
            assert_eq!(
                warm.asset_facts.transformer_bytes, expected,
                "Q{bits}: the device-format cache is a copy of the same bytes, never a second component"
            );
            assert_eq!(
                warm.asset_facts, cold.asset_facts,
                "Q{bits}: the cache changes no fact"
            );
            assert_eq!(
                gen_core::safetensors_dir_bytes(&transformer),
                std::fs::metadata(transformer.join("model.safetensors"))
                    .unwrap()
                    .len(),
                "Q{bits}: the shared on-disk walker must skip the cache too"
            );
        }
    }

    /// SC-22667 / E1 (DEFECT 2): the stock `AutoEncoderKL` builds both halves at the pipeline
    /// dtype, so the render-resident `decoder_bytes` is the whole `vae/` file at bf16 (packed
    /// mid-block triples dequantized dense). The f32 `VaeEncoder` that img2img builds on its first
    /// `Reference` request is a second, lazily materialized network: declared once as a
    /// `ReferenceEncoder` auxiliary component in `overlay_bytes`, never in a base field.
    #[test]
    fn decoder_bytes_hold_the_resident_autoencoder_and_declare_the_lazy_reference_encoder() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("z-image-vae-split");
        write_snapshot_component_configs(&root);
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();
        write_typed_safetensors(
            &root.join("transformer/model.safetensors"),
            &[("layers.0.w", "BF16", &[16], 32)],
        );
        write_typed_safetensors(
            &root.join("vae/model.safetensors"),
            &[
                ("decoder.conv_in.weight", "BF16", &[100], 200),
                ("encoder.conv_in.weight", "BF16", &[40], 80),
                (
                    "encoder.mid_block.attentions.0.to_q.weight",
                    "U32",
                    &[2, 8],
                    64,
                ),
                (
                    "encoder.mid_block.attentions.0.to_q.scales",
                    "BF16",
                    &[2, 1],
                    4,
                ),
                (
                    "encoder.mid_block.attentions.0.to_q.biases",
                    "BF16",
                    &[2, 1],
                    4,
                ),
            ],
        );
        // Resident (bf16): decoder 100 el + encoder 40 el + the dequantized 2x64 triple.
        const RESIDENT_VAE_BYTES: u64 = 200 + 80 + 2 * 64 * 2;
        // Lazy f32 encoder: the 40 el + the same 2x64 triple dequantized at 4 bytes.
        const REFERENCE_ENCODER_BYTES: u64 = 40 * 4 + 2 * 64 * 4;
        let spec = snapshot_spec(root.clone());

        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        let facts = contract.asset_facts;
        assert_eq!(facts.decoder_bytes, RESIDENT_VAE_BYTES);
        assert_eq!(facts.transformer_bytes, 32);
        assert_eq!(facts.overlay_bytes, REFERENCE_ENCODER_BYTES);
        assert_eq!(
            facts.base_bytes,
            facts.conditioning_bytes + 32 + RESIDENT_VAE_BYTES,
            "base_bytes excludes the request-opted encoder copy"
        );
        let components = contract.resident_components();
        assert_eq!(components.len(), 1, "{components:?}");
        assert_eq!(components[0].id, REFERENCE_ENCODER_COMPONENT_ID);
        assert_eq!(
            components[0].kind,
            gen_core::MemoryComponentKind::ReferenceEncoder
        );
        assert_eq!(components[0].resident_bytes, REFERENCE_ENCODER_BYTES);
        assert_eq!(components[0].bounded_by, None);
        assert_eq!(contract.auxiliary_resident_bytes(), REFERENCE_ENCODER_BYTES);
        assert_eq!(
            contract.total_resident_bytes(),
            facts.base_bytes + REFERENCE_ENCODER_BYTES
        );
        assert!(
            matches!(
                contract.formula,
                MemoryFormulaKind::ComponentPhaseEnvelope { .. }
            ),
            "a declared auxiliary component needs the typed listing"
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The control route stacks its branch beside the reference encoder: two typed auxiliary
        // components, one `overlay_bytes` equal to their sum, base facts untouched.
        let control = root.join("control");
        std::fs::create_dir_all(&control).unwrap();
        write_safetensors(&control.join("control.safetensors"), &[("w", 4096)]);
        let control_bytes = gen_core::safetensors_path_bytes(&control);
        assert!(control_bytes > 0);
        let control_contract = control_contract(
            "z_image_turbo_control",
            &spec.with_control(WeightsSource::Dir(control)),
        )
        .unwrap();
        assert_eq!(
            control_contract.asset_facts.overlay_bytes,
            REFERENCE_ENCODER_BYTES + control_bytes
        );
        assert_eq!(control_contract.asset_facts.base_bytes, facts.base_bytes);
        let ids = control_contract
            .resident_components()
            .iter()
            .map(|component| component.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![REFERENCE_ENCODER_COMPONENT_ID, CONTROL_COMPONENT_ID]
        );
        assert_eq!(
            control_contract.auxiliary_resident_bytes(),
            REFERENCE_ENCODER_BYTES + control_bytes
        );
        assert!(
            control_contract.conformance_errors().is_empty(),
            "{:?}",
            control_contract.conformance_errors()
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&control_contract);

        // A decoder-only source has no reference encoder to declare and keeps the plain shape.
        let decoder_only = tmp.path().join("z-image-decoder-only");
        write_snapshot_component_configs(&decoder_only);
        write_typed_safetensors(
            &decoder_only.join("vae/model.safetensors"),
            &[("decoder.conv_in.weight", "BF16", &[100], 200)],
        );
        let plain = provider_contract(crate::MODEL_ID, &snapshot_spec(decoder_only)).unwrap();
        assert_eq!(plain.asset_facts.decoder_bytes, 200);
        assert_eq!(plain.asset_facts.overlay_bytes, 0);
        assert!(plain.resident_components().is_empty());
        assert!(matches!(
            plain.formula,
            MemoryFormulaKind::PhaseEnvelope { .. }
        ));
    }

    #[test]
    fn control_routes_publish_the_full_executable_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (spec, _) = materialized_control_fixture(&tmp);
        for id in ["z_image_turbo_control", "z_image_control"] {
            let contract = control_contract(id, &spec).unwrap();
            assert!(
                contract.conformance_errors().is_empty(),
                "{id}: {:?}",
                contract.conformance_errors()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                CONTROL_CALIBRATION_FINGERPRINT
            );
            assert_ne!(
                contract.calibration.as_ref().unwrap().fingerprint,
                CALIBRATION_FINGERPRINT
            );
            assert_eq!(contract.load_shape, LoadShape::DeferredMaterialization);
            assert!(contract
                .strategies
                .iter()
                .all(|capability| { capability.support == MemoryStrategySupport::Implemented }));
            assert!(contract.lifecycle.synchronized_phase_release);
            assert!(contract.lifecycle.decode_tiling);
            assert!(contract.lifecycle.attention_chunking);
            assert!(contract.lifecycle.transformer_window_materialization);
        }
    }
}
