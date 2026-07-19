//! Real-checkpoint loading from a Krea 2 snapshot (standard diffusers multi-component tree):
//! `text_encoder/` (Qwen3-VL-4B condition encoder), `transformer/` (single-stream DiT), `vae/`
//! (Qwen-Image `AutoencoderKLQwenImage`, loaded via [`crate::vae::load_vae`]). The transformer +
//! text-encoder checkpoints are identity-keyed (diffusers names = the module tree), so
//! [`Weights::from_dir`] drops straight in; the VAE remap lives in `mlx-gen-qwen-image`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_boogu::VisionTower;

use crate::config::Krea2Config;
use crate::text_encoder::{krea_vision_config, KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;

fn prepare_text_weights(mut w: Weights) -> Result<Weights> {
    let packed: std::collections::HashSet<String> = w
        .keys()
        .filter_map(|key| key.strip_suffix(".scales").map(str::to_owned))
        .collect();
    w.cast_matching(mlx_rs::Dtype::Bfloat16, |key| {
        key.starts_with("language_model.")
            && key.ends_with(".weight")
            && !key.contains("norm")
            && !packed.contains(key.strip_suffix(".weight").unwrap_or(key))
    })?;
    w.cast_matching(mlx_rs::Dtype::Float32, |key| {
        key.starts_with("language_model.") && key.ends_with("norm.weight")
    })?;
    Ok(w)
}

/// Load the Qwen3-VL-4B condition encoder from a snapshot's `text_encoder/` dir. The text tower lives
/// under `language_model.*`; the visual tower (`visual.*`) is assembled separately by
/// [`load_vision_tower`] only when image-grounded (edit) encoding is needed.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<KreaTextEncoder> {
    let root = root.as_ref();
    let cfg = KreaTeConfig::from_snapshot(root)?;
    let w = prepare_text_weights(Weights::from_dir(root.join("text_encoder"))?)?;
    KreaTextEncoder::from_weights(&w, "language_model", &cfg)
}

/// Load the Qwen3-VL-4B **vision tower** from the same `text_encoder/` dir (epic 10871 P2.1, sc-10879):
/// the `visual.*` subtree that text-to-image never assembles. Casts the (small, parity-grade) vision
/// subtree to f32 before building — mirroring boogu's `load_vision_tower` — and feeds the shared
/// [`mlx_gen_boogu::VisionTower`] the Krea-4B [`krea_vision_config`]. Krea keys are `visual.*` (diffusers
/// naming), unlike boogu's `model.visual.*`.
pub fn load_vision_tower(root: impl AsRef<Path>) -> Result<VisionTower> {
    let root = root.as_ref();
    let mut w = Weights::from_dir(root.join("text_encoder"))?;
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("visual."))
        .map(String::from)
        .collect();
    for k in keys {
        let t = w.require(&k)?.as_dtype(mlx_rs::Dtype::Float32)?;
        w.insert(k, t);
    }
    VisionTower::from_weights(&w, krea_vision_config(), "visual")
}

/// Load the single-stream DiT from a snapshot's `transformer/` dir: parse + validate the config, load
/// the (identity-keyed diffusers) weights, validate the architecture against the config, then assemble
/// the model. A pre-quantized snapshot loads through the same path (`quant::lin` auto-detects packed
/// keys); a dense bf16 build is quantized later via [`crate::pipeline::KreaPipeline::quantize`].
pub fn load_transformer(root: impl AsRef<Path>) -> Result<Krea2Transformer> {
    let root = root.as_ref();
    let cfg = Krea2Config::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    Krea2Transformer::from_weights(&w, &cfg)
}
