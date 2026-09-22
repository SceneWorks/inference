//! Snapshot loaders — the `Qwen/Qwen-Image-2.1` diffusers layout, received as a caller-provisioned
//! local directory (`WeightsSource::Dir`; inference never self-fetches):
//!
//! ```text
//! <root>/
//!   model_index.json
//!   processor/tokenizer.json            (+ tokenizer_config.json, chat_template.jinja, …)
//!   scheduler/scheduler_config.json
//!   text_encoder/config.json + model-*.safetensors   (Qwen3VLForConditionalGeneration)
//!   transformer/config.json + diffusion_pytorch_model-*.safetensors
//!   vae/config.json + diffusion_pytorch_model.safetensors
//! ```
//!
//! Weights load dense at their on-disk dtype (bf16 released; f32 for the tiny parity snapshot) and
//! every geometry comes from the component's own `config.json`, so one code path serves both.

use std::path::{Path, PathBuf};

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::{SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig};
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// Prefix of the language tower inside `Qwen3VLForConditionalGeneration` checkpoints.
pub const TEXT_ENCODER_PREFIX: &str = "model.language_model";
/// Longest prompt (template included) the tokenizer keeps; upstream never truncates, and the
/// released tokenizer's `model_max_length` is 262144 — this only bounds a pathological prompt.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// `<|endoftext|>`, the released tokenizer's pad token (unused: prompts are never padded).
const PAD_TOKEN_ID: i32 = 151_643;

/// The snapshot directory behind a spec, or an actionable error for a single-file source.
pub fn snapshot_root(source: &WeightsSource) -> Result<&Path> {
    match source {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "qwen_image_2_1 expects a snapshot directory (processor/ scheduler/ text_encoder/ \
             transformer/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// `processor/tokenizer.json` (falling back to `tokenizer/tokenizer.json`).
pub fn tokenizer_path(root: &Path) -> Result<PathBuf> {
    for candidate in ["processor/tokenizer.json", "tokenizer/tokenizer.json"] {
        let path = root.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error::Msg(format!(
        "qwen_image_2_1: no processor/tokenizer.json under {}",
        root.display()
    )))
}

/// The prompt tokenizer. The template is rendered by [`crate::prompt_template`] and tokenized
/// preformatted, so the chat-template axis is `None` here.
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    Ok(TextTokenizer::from_file(
        tokenizer_path(root)?,
        TokenizerConfig {
            max_length: MAX_PROMPT_TOKENS,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )?)
}

/// The Qwen3-VL language tower from `<text_encoder>/` (config + shards).
pub fn load_text_encoder_from(dir: &Path) -> Result<QwenImage21TextEncoder> {
    let cfg = TextEncoderConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(dir)?;
    QwenImage21TextEncoder::from_weights(&w, TEXT_ENCODER_PREFIX, &cfg)
}

/// The Qwen3-VL language tower from `<root>/text_encoder/`.
pub fn load_text_encoder(root: &Path) -> Result<QwenImage21TextEncoder> {
    load_text_encoder_from(&root.join("text_encoder"))
}

/// The DiT from `<root>/transformer/`.
pub fn load_transformer(root: &Path) -> Result<QwenImage21Transformer> {
    let dir = root.join("transformer");
    let cfg = TransformerConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(&dir)?;
    QwenImage21Transformer::from_weights(&w, &cfg)
}

/// The RGBA VAE from `<root>/vae/`.
pub fn load_vae(root: &Path) -> Result<QwenImage21Vae> {
    let dir = root.join("vae");
    let cfg = VaeConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(&dir)?;
    QwenImage21Vae::from_weights(&w, &cfg)
}

/// `<root>/scheduler/scheduler_config.json`, or the frozen production values when the snapshot
/// ships none.
pub fn load_scheduler_config(root: &Path) -> Result<SchedulerConfig> {
    let path = root.join("scheduler").join("scheduler_config.json");
    if path.is_file() {
        SchedulerConfig::from_json_file(&path)
    } else {
        Ok(SchedulerConfig::production())
    }
}
