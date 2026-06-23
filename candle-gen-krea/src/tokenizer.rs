//! Krea 2 condition tokenization (sc-7569) — the Qwen3-VL prompt template + fast `Qwen2Tokenizer`
//! that turns a text prompt into the `input_ids` the condition encoder consumes. Port of
//! `mlx-gen-krea`'s `text_encoder/tokenizer.rs`.
//!
//! The reference `Qwen3VLConditioner` wraps the user text in a fixed system-instruction template + an
//! `assistant` generation cue, tokenizes (`add_special_tokens=false`; the `<|im_start|>`/`<|im_end|>`
//! markers are added-tokens in `tokenizer.json`), runs Qwen3-VL, then drops the leading
//! [`PREFIX_TOKENS`] system-prefix tokens from the conditioning (the encoder does the drop). We render
//! the exact template string ourselves and encode it. The reference pads to `max_length`, which only
//! adds masked tokens; for the per-sample `B = 1` path the natural length is numerically equivalent
//! (the encoder runs masked and the DiT trims padding), so we emit the natural-length ids.

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

/// System-instruction prefix (reference `prompt_template_encode_prefix`). Tokenizes to exactly
/// [`PREFIX_TOKENS`] tokens — the slice the encoder drops.
pub const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

/// `assistant` generation cue appended after the user text (reference `prompt_template_encode_suffix`).
pub const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Number of leading template-prefix tokens dropped from the conditioning (reference
/// `prompt_template_encode_start_idx`); [`PREFIX`] tokenizes to this many.
pub const PREFIX_TOKENS: usize = 34;

/// Qwen <|endoftext|> id — the pad token (unused on the natural-length path).
const PAD_TOKEN_ID: i32 = 151643;

/// Render the full template string for a user prompt: `{PREFIX}{user}{SUFFIX}`.
fn render(user: &str) -> String {
    format!("{PREFIX}{user}{SUFFIX}")
}

/// The Krea condition tokenizer: the snapshot's `tokenizer/tokenizer.json` wrapped to render the Krea
/// template and encode it. Builds `input_ids` directly on the model device.
pub struct KreaTokenizer {
    inner: TextTokenizer,
    device: Device,
}

impl KreaTokenizer {
    /// Load from a snapshot's `tokenizer/tokenizer.json`.
    pub fn from_snapshot(root: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            root.as_ref().join("tokenizer").join("tokenizer.json"),
            TokenizerConfig {
                // We render the template string ourselves and call `encode_ids` directly, so the
                // config template/padding are inert.
                max_length: 512,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("krea: load tokenizer: {e}")))?;
        Ok(Self {
            inner,
            device: device.clone(),
        })
    }

    /// Encode a rendered string to ids (`add_special_tokens=false`, matching the reference).
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.inner
            .encode_ids(text, false)
            .map_err(|e| CandleError::Msg(format!("krea: tokenize: {e}")))
    }

    /// Raw id vector for the templated prompt (parity testing against the reference `input_ids`).
    pub fn ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(&render(prompt))
    }

    /// Token count of the bare [`PREFIX`] (should equal [`PREFIX_TOKENS`]).
    pub fn prefix_len(&self) -> Result<usize> {
        Ok(self.encode(PREFIX)?.len())
    }

    /// Encode the templated prompt → `input_ids` `[1, L]` u32. The encoder drops the leading
    /// [`PREFIX_TOKENS`] from the resulting conditioning.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let ids = self.ids(prompt)?;
        if ids.is_empty() {
            return Err(CandleError::Msg("krea: empty token sequence".into()));
        }
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        Ok(Tensor::from_vec(ids, (1, len), &self.device)?)
    }
}
