//! Assemble the Anima components from the on-disk `split_files/` layout:
//! `diffusion_models/anima-{variant}-v1.0.safetensors` (DiT + bundled `net.llm_adapter.*`
//! conditioner), `text_encoders/qwen_3_06b_base.safetensors`, `vae/qwen_image_vae.safetensors`.
//!
//! The DiT safetensors bundles BOTH the Cosmos DiT (Cosmos naming, `net.*`) and the
//! `AnimaTextConditioner` (`net.llm_adapter.*`). We load it once and build both from the same
//! `Weights` with their respective key prefixes — the `net.llm_adapter.` split is exactly the Anima
//! convert script's `split_anima_transformer_checkpoint`.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::conditioner::AnimaTextConditioner;
use crate::config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::{load_vae, QwenVae};

/// The conditioner-splitting marker (port of the Anima convert script's
/// `split_anima_transformer_checkpoint`, which splits on `llm_adapter.`). The DiT root prefix varies
/// by file — `net` (base) or `model.diffusion_model` (turbo/aesthetic) — so we detect it rather than
/// hardcode it.
const ADAPTER_MARKER: &str = "llm_adapter.";
/// A key that unambiguously fixes the DiT root prefix (present in every Anima DiT file).
const PREFIX_ANCHOR: &str = ".x_embedder.proj.1.weight";

const TEXT_ENCODER_FILE: &str = "text_encoders/qwen_3_06b_base.safetensors";
const VAE_FILE: &str = "vae/qwen_image_vae.safetensors";

/// Detect the DiT root prefix (`net` or `model.diffusion_model`) from the checkpoint keys.
fn detect_dit_prefix(w: &Weights) -> Result<String> {
    w.keys()
        .find(|k| k.ends_with(PREFIX_ANCHOR))
        .map(|k| k[..k.len() - PREFIX_ANCHOR.len()].to_string())
        .ok_or_else(|| {
            Error::Msg(format!(
                "anima: no DiT root prefix found (no key ending in {PREFIX_ANCHOR})"
            ))
        })
}

/// Split a loaded Anima DiT checkpoint's keys into `(dit_keys, adapter_keys)` — any key containing
/// `llm_adapter.` is the conditioner, everything else is the Cosmos DiT (prefix-agnostic).
pub fn split_anima_keys(w: &Weights) -> (Vec<String>, Vec<String>) {
    let mut dit = Vec::new();
    let mut adapter = Vec::new();
    for k in w.keys() {
        if k.contains(ADAPTER_MARKER) {
            adapter.push(k.to_string());
        } else {
            dit.push(k.to_string());
        }
    }
    (dit, adapter)
}

/// Resolve the `split_files/` directory holding `diffusion_models/`, `text_encoders/`, `vae/`.
///
/// - `Dir(p)`: `p` itself if it already contains `diffusion_models/`, else `p/split_files`.
/// - `File(dit)`: the DiT's grandparent (`.../split_files/diffusion_models/x.safetensors` → `split_files`).
fn resolve_split_files(source: &WeightsSource) -> Result<PathBuf> {
    match source {
        WeightsSource::Dir(p) => {
            if p.join("diffusion_models").is_dir() {
                Ok(p.clone())
            } else if p.join("split_files").join("diffusion_models").is_dir() {
                Ok(p.join("split_files"))
            } else {
                Err(Error::Msg(format!(
                    "anima: {} is not an Anima split_files dir (no diffusion_models/ or split_files/diffusion_models/)",
                    p.display()
                )))
            }
        }
        WeightsSource::File(dit) => dit
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                Error::Msg(format!(
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
}

impl AnimaComponents {
    /// Load all components for `variant` from a weights source (a `split_files/` dir or a DiT file).
    pub fn load(source: &WeightsSource, variant: Variant) -> Result<Self> {
        let root = resolve_split_files(source)?;
        let dit_path = root.join("diffusion_models").join(variant.dit_filename());
        if !dit_path.is_file() {
            return Err(Error::Msg(format!(
                "anima: DiT file not found: {}",
                dit_path.display()
            )));
        }

        // The DiT file carries both the Cosmos DiT and the bundled conditioner. The root prefix is
        // `net` (base) or `model.diffusion_model` (turbo/aesthetic) — detect it.
        let dit_weights = Weights::from_file(&dit_path)?;
        let prefix = detect_dit_prefix(&dit_weights)?;
        let dit = CosmosDiT::from_weights(&dit_weights, &prefix, DitConfig::anima())?;
        let conditioner = AnimaTextConditioner::from_weights(
            &dit_weights,
            &format!("{prefix}.llm_adapter"),
            ConditionerConfig::anima(),
        )?;

        let te_weights = Weights::from_file(root.join(TEXT_ENCODER_FILE))?;
        let text_encoder = AnimaQwen3::from_weights(&te_weights, "model", &Qwen3Config::anima())?;

        let vae = load_vae(root.join(VAE_FILE))?;
        let tokenizers = AnimaTokenizers::load()?;

        Ok(Self {
            dit,
            conditioner,
            text_encoder,
            vae,
            tokenizers,
        })
    }
}
