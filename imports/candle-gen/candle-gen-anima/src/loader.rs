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
use candle_gen::gen_core::{AdapterSpec, WeightsSource};
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

const TEXT_ENCODER_FILE: &str = "text_encoders/qwen_3_06b_base.safetensors";
const VAE_FILE: &str = "vae/qwen_image_vae.safetensors";

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

/// Resolve `variant`'s DiT file and report whether it is a **packed** (Q4/Q8) tier — i.e. its
/// safetensors header carries any `.scales` code tensor. Header-only (no weight data materialized), so
/// it is cheap enough to gate a `LoadSpec` on. Lets the generator fail fast when a quantized tier is
/// requested against a dense checkpoint instead of silently loading bf16 — a tier downgrade the caller
/// never sees (`lin()` packed-detects PER-TENSOR, so a dense checkpoint just builds dense).
pub fn dit_is_packed(source: &WeightsSource, variant: Variant) -> Result<bool> {
    let root = resolve_split_files(source)?;
    let dit_path = root.join("diffusion_models").join(variant.dit_filename());
    if !dit_path.is_file() {
        return Err(CandleError::Msg(format!(
            "anima: DiT file not found: {}",
            dit_path.display()
        )));
    }
    dit_path_is_packed(&dit_path)
}

/// Header-only check: does this DiT safetensors file carry any `.scales` code tensor (a packed Q4/Q8
/// tier)? Reads only the tensor names, no weight data. The single source of truth for "is this DiT
/// packed" — the `LoadSpec` gate, the tier-mismatch guard, and the adapter fold-vs-residual branch all
/// route through it.
fn dit_path_is_packed(dit_path: &Path) -> Result<bool> {
    // Header-only mmap: reads the tensor names without materializing any weight data.
    // SAFETY: read-only, process-owned weight file, mapped only to read the header here.
    let st = unsafe { MmapedSafetensors::new(dit_path)? };
    Ok(st
        .tensors()
        .into_iter()
        .any(|(k, _)| k.ends_with(".scales")))
}

/// Resolve the `split_files/` directory holding `diffusion_models/`, `text_encoders/`, `vae/`.
fn resolve_split_files(source: &WeightsSource) -> Result<PathBuf> {
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
        let packed_with_adapters = !adapters.is_empty() && dit_path_is_packed(&dit_path)?;
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
        let dir = std::env::temp_dir().join(format!("anima_prefix_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

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

        let _ = std::fs::remove_dir_all(&dir);
    }
}
