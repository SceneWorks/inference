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
//! Every geometry comes from the component's own `config.json`, so the miniature parity snapshot
//! and the production snapshot go through one code path. Unlike the MLX twin — which loads dense at
//! the on-disk dtype — candle's `VarBuilder` casts on read, so each component names the dtype it
//! computes in ([`compute_dtype`]): f32 on CPU (the parity lane) and bf16 on CUDA for the DiT and
//! the VAE, matching the released checkpoint's own dtype.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::WeightsSource;
use candle_gen::{CandleError as Error, Result};

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
/// Error label every loader message carries.
pub(crate) const LABEL: &str = "qwen_image_2_1";

/// The dtype this backend computes the weight-heavy components in: the checkpoint's own bf16 on a
/// GPU backend, f32 on CPU (candle's CPU half-precision kernels are slow, and the parity lane wants
/// f32 anyway). The text tower always runs f32 activations, matching the MLX twin.
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

/// The Qwen3-VL language tower from `<dir>/` (config + shards) on `device`.
pub fn load_text_encoder_from(dir: &Path, device: &Device) -> Result<QwenImage21TextEncoder> {
    let cfg = TextEncoderConfig::from_json_file(&dir.join("config.json"))?;
    // f32 activations over the checkpoint's weights, exactly as the MLX twin does.
    let vb = candle_gen::loader::load_sorted_mmap(dir, DType::F32, device, LABEL)?;
    QwenImage21TextEncoder::new(&cfg, vb, TEXT_ENCODER_PREFIX)
}

/// The Qwen3-VL language tower from `<root>/text_encoder/`.
pub fn load_text_encoder(root: &Path, device: &Device) -> Result<QwenImage21TextEncoder> {
    load_text_encoder_from(&root.join("text_encoder"), device)
}

/// The DiT from `<root>/transformer/`.
pub fn load_transformer(root: &Path, device: &Device) -> Result<QwenImage21Transformer> {
    let cfg = TransformerConfig::from_json_file(&root.join("transformer").join("config.json"))?;
    let vb = candle_gen::loader::component_vb(root, "transformer", compute_dtype(), device, LABEL)?;
    QwenImage21Transformer::new(&cfg, vb)
}

/// The RGBA VAE from `<root>/vae/`.
pub fn load_vae(root: &Path, device: &Device) -> Result<QwenImage21Vae> {
    let cfg = VaeConfig::from_json_file(&root.join("vae").join("config.json"))?;
    let vb = candle_gen::loader::component_vb(root, "vae", compute_dtype(), device, LABEL)?;
    QwenImage21Vae::new(&cfg, vb)
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
