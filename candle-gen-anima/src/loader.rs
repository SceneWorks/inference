//! Assemble the Anima components from the on-disk `split_files/` layout — the candle transcription of
//! `mlx-gen-anima`'s `loader.rs`.
//!
//! The DiT safetensors bundles BOTH the Cosmos DiT (`{prefix}.*`) and the `AnimaTextConditioner`
//! (`{prefix}.llm_adapter.*`). We detect the root `{prefix}` from the checkpoint keys and build both
//! from the same mmap'd VarBuilder with their respective sub-prefixes.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::WeightsSource;
use candle_gen::{CandleError, Result};

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
    /// Load all components for `variant` from a weights source (a `split_files/` dir or a DiT file).
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

        // The DiT file carries both the Cosmos DiT and the bundled conditioner. Detect the root prefix
        // (`net` or `model.diffusion_model`), then build both from ONE mmap'd VarBuilder.
        let prefix = detect_dit_prefix(&dit_path)?;
        let dit_vb = candle_gen::mmap_var_builder(std::slice::from_ref(&dit_path), dtype, device)?;
        let dit = CosmosDiT::new(&dit_vb.pp(&prefix), DitConfig::anima())?;
        let conditioner = AnimaTextConditioner::new(
            &dit_vb.pp(&prefix).pp("llm_adapter"),
            ConditionerConfig::anima(),
        )?;

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
