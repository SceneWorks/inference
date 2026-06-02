//! Real-checkpoint loading for FLUX.2-klein from a `black-forest-labs/FLUX.2-klein-9B` snapshot
//! directory (standard diffusers multi-component tree):
//! ```text
//!   <root>/tokenizer/tokenizer.json
//!   <root>/text_encoder/*.safetensors   (Qwen3, `model.*` keys)
//!   <root>/transformer/*.safetensors    (S3)
//!   <root>/vae/*.safetensors            (S2)
//! ```
//! The Qwen3 `text_encoder` layout maps directly onto the encoder under the `"model"` prefix
//! (the fork's mapping only strips `model.`), so it needs no remap.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};

/// Qwen2 pad token id (`<|endoftext|>`).
pub const PAD_TOKEN_ID: i32 = 151643;
/// The fork's `LanguageTokenizer` max_length for the FLUX.2 `qwen3` tokenizer.
pub const MAX_LENGTH: usize = 512;

/// Load the Qwen2 tokenizer with FLUX.2's chat template (`enable_thinking=False`) and the fork's
/// padding policy (`padding="max_length"` → every prompt padded to 512).
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let path = root.join("tokenizer/tokenizer.json");
    TextTokenizer::from_file(
        path,
        TokenizerConfig {
            max_length: MAX_LENGTH,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::QwenInstructNoThink,
            pad_to_max_length: true,
        },
    )
}

/// Load the Qwen3 text encoder. The on-disk `model.*` keys map directly onto the encoder tree
/// under the `"model"` prefix — no remap needed.
pub fn load_text_encoder(root: &Path) -> Result<Qwen3TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    Qwen3TextEncoder::from_weights(&w, "model", &Qwen3TextEncoderConfig::klein_9b())
}
