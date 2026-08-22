//! Assemble the Anima components from the on-disk `split_files/` layout — the candle transcription of
//! `mlx-gen-anima`'s `loader.rs`.
//!
//! The DiT safetensors bundles BOTH the Cosmos DiT (`{prefix}.*`) and the `AnimaTextConditioner`
//! (`{prefix}.llm_adapter.*`). We detect the root `{prefix}` from the checkpoint keys and build both
//! from the same mmap'd VarBuilder with their respective sub-prefixes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::{self as cst, MmapedSafetensors};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{AdapterSpec, Quant, WeightsSource};
use candle_gen::{CandleError, Result};

use crate::adapters::{apply_anima_adapters, install_anima_residuals};
use crate::conditioner::AnimaTextConditioner;
use crate::config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::{load_vae, QwenVae};

/// A key that unambiguously fixes the DiT root prefix (present in every Anima DiT file). The root is
/// `net` for the base cut, `model.diffusion_model` for turbo/aesthetic — so we DETECT it. A hardcoded
/// `net.` would silently drop the 134.7M-param conditioner (`{prefix}.llm_adapter.*`) for two of the
/// three variants (the exact bug in HuggingFace's own `convert_anima_to_diffusers.py`).
const PREFIX_ANCHOR: &str = ".x_embedder.proj.1.weight";

pub(crate) const TEXT_ENCODER_FILE: &str = "text_encoders/qwen_3_06b_base.safetensors";
pub(crate) const VAE_FILE: &str = "vae/qwen_image_vae.safetensors";

/// The compute dtype for the DiT / conditioner / text encoder: bf16 on the GPU backends (the native
/// checkpoint dtype), f32 on CPU (bf16 CPU kernels are slow/unsupported, and f32 is the parity lane).
pub fn compute_dtype() -> DType {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        DType::BF16
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        DType::F32
    }
}

/// Detect the DiT root prefix (`net` or `model.diffusion_model`) from a safetensors file's keys — port
/// of `detect_dit_prefix`. Errors (never assumes) if no anchor key is present.
pub fn detect_dit_prefix(dit_path: &Path) -> Result<String> {
    // Header-only mmap: reads the tensor names without materializing any weight data.
    // SAFETY: read-only, process-owned weight file, mapped only to read the header here.
    let st = unsafe { MmapedSafetensors::new(dit_path)? };
    st.tensors()
        .into_iter()
        .map(|(k, _)| k)
        .find(|k| k.ends_with(PREFIX_ANCHOR))
        .map(|k| k[..k.len() - PREFIX_ANCHOR.len()].to_string())
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "anima: no DiT root prefix found in {} (no key ending in {PREFIX_ANCHOR})",
                dit_path.display()
            ))
        })
}

/// Resolve the physical Q4/Q8 tier stored in `variant`'s DiT file. The header supplies both the
/// packed-code and scale shapes, so a caller cannot relabel Q4 bytes as Q8 (or vice versa) through a
/// `LoadSpec`. No tensor data is materialized.
pub fn dit_quant_tier(source: &WeightsSource, variant: Variant) -> Result<Option<Quant>> {
    let root = resolve_split_files(source)?;
    let dit_path = root.join("diffusion_models").join(variant.dit_filename());
    if !dit_path.is_file() {
        return Err(CandleError::Msg(format!(
            "anima: DiT file not found: {}",
            dit_path.display()
        )));
    }
    dit_path_quant_tier(&dit_path)
}

/// Header-only physical packed-tier detection. Every packed affine pair must agree on the bit width;
/// a companion `config.json` may state its group size, otherwise the immutable Anima packing convention
/// is the shared MLX group-64 format.
fn dit_path_quant_tier(dit_path: &Path) -> Result<Option<Quant>> {
    // Header-only mmap: reads tensor metadata without materializing any weight data.
    // SAFETY: read-only, process-owned weight file, mapped only to read the header here.
    let st = unsafe { MmapedSafetensors::new(dit_path)? };
    let config_path = dit_path.with_file_name("config.json");
    let packed_config = if config_path.is_file() {
        let text = std::fs::read_to_string(&config_path).map_err(|error| {
            CandleError::Msg(format!(
                "anima: cannot read packed config {}: {error}",
                config_path.display()
            ))
        })?;
        let config = serde_json::from_str::<serde_json::Value>(&text).map_err(|error| {
            CandleError::Msg(format!(
                "anima: cannot parse packed config {}: {error}",
                config_path.display()
            ))
        })?;
        candle_gen::quant::PackedConfig::from_config(&config)
    } else {
        None
    };
    let group_size = packed_config
        .map(|config| usize::try_from(config.group_size))
        .transpose()
        .map_err(|_| {
            CandleError::Msg(format!(
                "anima: packed config has invalid group size in {}",
                config_path.display()
            ))
        })?
        .unwrap_or(candle_gen::quant::MLX_GROUP_SIZE);
    if group_size == 0 {
        return Err(CandleError::Msg(format!(
            "anima: packed config has zero group size in {}",
            config_path.display()
        )));
    }

    let mut tier = None;
    for (key, scales) in st.tensors() {
        let Some(base) = key.strip_suffix(".scales") else {
            continue;
        };
        let weight = st.get(&format!("{base}.weight")).map_err(|_| {
            CandleError::Msg(format!(
                "anima: packed scale `{key}` has no companion `{base}.weight` in {}",
                dit_path.display()
            ))
        })?;
        let weight_shape = weight.shape();
        let scales_shape = scales.shape();
        if weight_shape.len() != 2
            || scales_shape.len() != 2
            || weight_shape[0] != scales_shape[0]
            || scales_shape[1] == 0
        {
            return Err(CandleError::Msg(format!(
                "anima: malformed packed pair `{base}` in {}",
                dit_path.display()
            )));
        }
        let input = scales_shape[1].checked_mul(group_size).ok_or_else(|| {
            CandleError::Msg(format!(
                "anima: packed pair `{base}` input width overflows in {}",
                dit_path.display()
            ))
        })?;
        let encoded = weight_shape[1].checked_mul(32).ok_or_else(|| {
            CandleError::Msg(format!(
                "anima: packed pair `{base}` code width overflows in {}",
                dit_path.display()
            ))
        })?;
        if input == 0 || !encoded.is_multiple_of(input) {
            return Err(CandleError::Msg(format!(
                "anima: packed pair `{base}` has inconsistent code/scale shapes in {}",
                dit_path.display()
            )));
        }
        let observed = match encoded / input {
            4 => Quant::Q4,
            8 => Quant::Q8,
            bits => {
                return Err(CandleError::Msg(format!(
                    "anima: packed pair `{base}` declares unsupported Q{bits} width in {}",
                    dit_path.display()
                )));
            }
        };
        if let Some(config) = packed_config {
            let configured = match config.bits {
                4 => Quant::Q4,
                8 => Quant::Q8,
                bits => {
                    return Err(CandleError::Msg(format!(
                        "anima: packed config declares unsupported Q{bits} width in {}",
                        config_path.display()
                    )));
                }
            };
            if configured != observed {
                return Err(CandleError::Msg(format!(
                    "anima: packed config {:?} disagrees with physical {:?} codes in {}",
                    configured,
                    observed,
                    dit_path.display()
                )));
            }
        }
        if let Some(previous) = tier.replace(observed) {
            if previous != observed {
                return Err(CandleError::Msg(format!(
                    "anima: packed DiT mixes {:?} and {:?} tensors in {}",
                    previous,
                    observed,
                    dit_path.display()
                )));
            }
        }
    }
    Ok(tier)
}

/// Compatibility discriminator for loader branches that only need dense versus packed. Tier identity
/// still comes from [`dit_quant_tier`] at admission.
pub(crate) fn dit_is_packed(source: &WeightsSource, variant: Variant) -> Result<bool> {
    Ok(dit_quant_tier(source, variant)?.is_some())
}

/// Resolve the `split_files/` directory holding `diffusion_models/`, `text_encoders/`, `vae/`.
pub(crate) fn resolve_split_files(source: &WeightsSource) -> Result<PathBuf> {
    match source {
        WeightsSource::Dir(p) => {
            if p.join("diffusion_models").is_dir() {
                Ok(p.clone())
            } else if p.join("split_files").join("diffusion_models").is_dir() {
                Ok(p.join("split_files"))
            } else {
                Err(CandleError::Msg(format!(
                    "anima: {} is not an Anima split_files dir (no diffusion_models/ or \
                     split_files/diffusion_models/)",
                    p.display()
                )))
            }
        }
        WeightsSource::File(dit) => dit
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "anima: cannot resolve split_files/ from DiT file {}",
                    dit.display()
                ))
            }),
    }
}

/// Load every tensor of the Anima DiT single-file checkpoint into a CPU key→`Tensor` map (native
/// dtype) for the adapter-merge path — the fold runs on CPU in f32, then the merged map is cast to the
/// compute dtype + moved to the device. Only taken when adapters are present (it gives up the mmap the
/// adapter-free path keeps, so the plain model pays nothing).
fn load_dit_map(path: &Path) -> Result<HashMap<String, Tensor>> {
    Ok(cst::load(path, &Device::Cpu)?)
}

/// The assembled Anima components for one variant.
pub struct AnimaComponents {
    pub dit: CosmosDiT,
    pub conditioner: AnimaTextConditioner,
    pub text_encoder: AnimaQwen3,
    pub vae: QwenVae,
    pub tokenizers: AnimaTokenizers,
    /// The compute dtype the DiT / conditioner / text encoder run at (bf16 on GPU, f32 on CPU).
    pub dtype: DType,
}

/// The conditioning half of an Anima request.  It deliberately owns the Qwen3 encoder and the
/// bundled conditioner, but not the DiT or VAE, so a staged request can release it before the
/// denoise/decode phase without changing the prompt, seed, schedule, or compute dtype.
pub struct AnimaConditioningComponents {
    pub conditioner: AnimaTextConditioner,
    pub text_encoder: AnimaQwen3,
    pub tokenizers: AnimaTokenizers,
    pub dtype: DType,
}

/// The denoise/decode half of an Anima request.  Kept separate from
/// [`AnimaConditioningComponents`] for the request-scoped Candle residency path.
pub struct AnimaRenderComponents {
    pub dit: CosmosDiT,
    pub vae: QwenVae,
    pub dtype: DType,
}

impl AnimaComponents {
    /// Load all components for `variant`. `adapters` are LoRA/LoKr `.safetensors` baked onto the DiT +
    /// bundled conditioner at load (stacked, mixed) — empty for the plain model.
    pub fn load(
        source: &WeightsSource,
        variant: Variant,
        device: &Device,
        adapters: &[AdapterSpec],
    ) -> Result<Self> {
        let root = resolve_split_files(source)?;
        let dit_path = root.join("diffusion_models").join(variant.dit_filename());
        if !dit_path.is_file() {
            return Err(CandleError::Msg(format!(
                "anima: DiT file not found: {}",
                dit_path.display()
            )));
        }
        let dtype = compute_dtype();

        // The DiT file carries both the Cosmos DiT and the bundled conditioner. Detect the root prefix
        // (`net` or `model.diffusion_model`), then build both from ONE VarBuilder.
        let prefix = detect_dit_prefix(&dit_path)?;
        // A packed (Q4/Q8) DiT stores u32 codes + `.scales`, so a LoRA/LoKr has no dense `.weight` to
        // fold into. When adapters are present on a packed tier we build the model from the mmap (the
        // packed codes survive load) and install the adapters as **forward-time residuals** afterwards
        // (`y = base(x) + scale·(xA)B`, sc-10640 / epic 10043). A dense tier keeps the weight-level fold,
        // byte-for-byte unchanged, and the plain model has no adapters at all.
        let packed_with_adapters =
            !adapters.is_empty() && dit_path_quant_tier(&dit_path)?.is_some();
        let dit_vb = if adapters.is_empty() || packed_with_adapters {
            // Plain model, OR packed + adapters (residuals installed post-build): mmap the checkpoint
            // directly — no fold, so the packed codes are never cast/materialized.
            candle_gen::mmap_var_builder(std::slice::from_ref(&dit_path), dtype, device)?
        } else {
            // Dense tier + adapters: fold every LoRA/LoKr delta into the base weights at the
            // safetensors-key level (merge, don't residual — the DiT is chaos-sensitive), then build
            // from the merged map. The 448 DiT + 60 conditioner targets both fold into
            // `{prefix}.{path}.weight`; a target that fails to route hard-errors (no silent partial —
            // sc-10274).
            let mut base = load_dit_map(&dit_path)?;
            let _report = apply_anima_adapters(&mut base, &prefix, adapters)?;
            // Unify to the compute dtype + device (the fold ran in f32 on CPU) and build.
            let merged: HashMap<String, Tensor> = base
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(dtype)?.to_device(device)?)))
                .collect::<Result<_>>()?;
            VarBuilder::from_tensors(merged, dtype, device)
        };
        let mut dit = CosmosDiT::new(&dit_vb.pp(&prefix), DitConfig::anima())?;
        let mut conditioner = AnimaTextConditioner::new(
            &dit_vb.pp(&prefix).pp("llm_adapter"),
            ConditionerConfig::anima(),
        )?;
        // Packed tier + adapters: install the LoRA(s) as forward-time residuals over the packed DiT +
        // dense conditioner. (The dense-tier fold above already baked adapters into the weights, and the
        // plain model has none.) LoKr/LoHa on a packed tier hard-errors here — sc-10713.
        if packed_with_adapters {
            let _report = install_anima_residuals(&mut dit, &mut conditioner, adapters)?;
        }

        let te_path = root.join(TEXT_ENCODER_FILE);
        let te_vb = candle_gen::mmap_var_builder(std::slice::from_ref(&te_path), dtype, device)?;
        let text_encoder = AnimaQwen3::new(&te_vb.pp("model"), &Qwen3Config::anima())?;

        let vae = load_vae(root.join(VAE_FILE), device)?;
        let tokenizers = AnimaTokenizers::load()?;

        Ok(Self {
            dit,
            conditioner,
            text_encoder,
            vae,
            tokenizers,
            dtype,
        })
    }
}

impl AnimaConditioningComponents {
    /// Load only the components needed to turn prompts into immutable DiT conditioning.  Staged
    /// residency intentionally refuses adapters at the caller: a packed adapter residual spans
    /// both this conditioner and the DiT, and splitting it would otherwise risk a partial overlay.
    pub fn load(source: &WeightsSource, variant: Variant, device: &Device) -> Result<Self> {
        let root = resolve_split_files(source)?;
        let dit_path = root.join("diffusion_models").join(variant.dit_filename());
        if !dit_path.is_file() {
            return Err(CandleError::Msg(format!(
                "anima: DiT file not found: {}",
                dit_path.display()
            )));
        }
        let dtype = compute_dtype();
        let prefix = detect_dit_prefix(&dit_path)?;
        let dit_vb = candle_gen::mmap_var_builder(std::slice::from_ref(&dit_path), dtype, device)?;
        let conditioner = AnimaTextConditioner::new(
            &dit_vb.pp(&prefix).pp("llm_adapter"),
            ConditionerConfig::anima(),
        )?;
        let te_path = root.join(TEXT_ENCODER_FILE);
        let te_vb = candle_gen::mmap_var_builder(std::slice::from_ref(&te_path), dtype, device)?;
        Ok(Self {
            conditioner,
            text_encoder: AnimaQwen3::new(&te_vb.pp("model"), &Qwen3Config::anima())?,
            tokenizers: AnimaTokenizers::load()?,
            dtype,
        })
    }
}

impl AnimaRenderComponents {
    /// Load only the DiT and VAE for an adapter-free staged request.  Adapter-bearing requests
    /// remain resident because their exact load-time overlay spans the split boundary.
    pub fn load(source: &WeightsSource, variant: Variant, device: &Device) -> Result<Self> {
        let root = resolve_split_files(source)?;
        let dit_path = root.join("diffusion_models").join(variant.dit_filename());
        if !dit_path.is_file() {
            return Err(CandleError::Msg(format!(
                "anima: DiT file not found: {}",
                dit_path.display()
            )));
        }
        let dtype = compute_dtype();
        let prefix = detect_dit_prefix(&dit_path)?;
        let dit_vb = candle_gen::mmap_var_builder(std::slice::from_ref(&dit_path), dtype, device)?;
        Ok(Self {
            dit: CosmosDiT::new(&dit_vb.pp(&prefix), DitConfig::anima())?,
            vae: load_vae(root.join(VAE_FILE), device)?,
            dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Tensor;
    use std::collections::HashMap;

    /// Write a one-tensor safetensors whose only key is `{root}.x_embedder.proj.1.weight`, then assert
    /// `detect_dit_prefix` recovers `{root}` — covering **both** shipped DiT roots (`net` for the base
    /// cut, `model.diffusion_model` for turbo/aesthetic). A hardcoded `net.` would mis-detect the second.
    fn write_anchor(dir: &std::path::Path, root: &str) -> PathBuf {
        let path = dir.join(format!("{}.safetensors", root.replace('.', "_")));
        let mut m = HashMap::new();
        m.insert(
            format!("{root}.x_embedder.proj.1.weight"),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&m, &path).unwrap();
        path
    }

    #[test]
    fn detect_dit_prefix_covers_both_roots() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();

        for root in ["net", "model.diffusion_model"] {
            let path = write_anchor(&dir, root);
            assert_eq!(
                detect_dit_prefix(&path).unwrap(),
                root,
                "prefix must be detected, not hardcoded, for root {root:?}"
            );
        }

        // A file with no anchor key errors (never assumes a prefix).
        let mut m = HashMap::new();
        m.insert(
            "something.else.weight".to_string(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap(),
        );
        let bad = dir.join("noanchor.safetensors");
        candle_gen::candle_core::safetensors::save(&m, &bad).unwrap();
        assert!(detect_dit_prefix(&bad).is_err(), "no anchor key ⇒ error");
    }

    #[test]
    fn packed_header_derives_exact_bit_width_and_rejects_config_lies() {
        let temp = tempfile::tempdir().unwrap();
        let write_packed = |name: &str, columns: usize, config_bits: Option<u8>| {
            let dir = temp.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("anima.safetensors");
            let mut tensors = HashMap::new();
            tensors.insert(
                "net.x_embedder.proj.1.weight".to_owned(),
                Tensor::zeros((2, columns), DType::U32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                "net.x_embedder.proj.1.scales".to_owned(),
                Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
            );
            candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
            if let Some(bits) = config_bits {
                std::fs::write(
                    dir.join("config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
            }
            path
        };
        assert_eq!(
            dit_path_quant_tier(&write_packed("q4", 8, Some(4))).unwrap(),
            Some(Quant::Q4)
        );
        assert_eq!(
            dit_path_quant_tier(&write_packed("q8", 16, Some(8))).unwrap(),
            Some(Quant::Q8)
        );
        assert!(
            dit_path_quant_tier(&write_packed("lying-config", 8, Some(8))).is_err(),
            "config Q8 must not relabel physical Q4 codes"
        );
    }
}
