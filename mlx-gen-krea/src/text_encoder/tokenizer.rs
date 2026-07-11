//! Krea 2 condition tokenization (sc-7569) — the Qwen3-VL prompt template + fast `Qwen2Tokenizer`
//! that turns a text prompt into the `input_ids` the condition encoder consumes.
//!
//! The reference `Qwen3VLConditioner` wraps the user text in a fixed system-instruction template and
//! an `assistant` generation cue, tokenizes (`add_special_tokens` markers are literal in the string),
//! runs Qwen3-VL, then drops the leading [`PREFIX_TOKENS`] system-prefix tokens from the conditioning.
//! We render the exact template string ourselves and encode with `add_special_tokens=false`, mirroring
//! the reference `tokenizer(text)` path (the `<|im_start|>` / `<|im_end|>` markers are added-tokens in
//! `tokenizer.json`). Padding to `max_length` is a reference detail that only adds masked tokens; for
//! the per-sample `B = 1` path the natural length is numerically equivalent (the encoder runs masked
//! and the DiT trims padding), so we emit the natural-length ids.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::Result;
use mlx_rs::Array;

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

/// Qwen3-VL vision markers — added-tokens in `tokenizer.json` (like `<|im_start|>`), so rendering them
/// as literal strings + `encode(add_special_tokens=false)` maps each to its single id
/// (151652 / 151655 / 151653). The image-grounded (edit) template wraps each reference as
/// `<|vision_start|>` + `<|image_pad|>`×n + `<|vision_end|>`, where n is that image's merged vision-token
/// count (from the vision tower); the encoder then replaces the `<|image_pad|>` positions with the
/// vision features (epic 10871 P2).
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";

/// Render the full template string for a user prompt:
/// `{PREFIX}{user}{SUFFIX}`.
fn render(user: &str) -> String {
    format!("{PREFIX}{user}{SUFFIX}")
}

/// Render the image-grounded (edit) template: the same system [`PREFIX`] + user role, with each
/// reference's vision block (`<|vision_start|><|image_pad|>×n<|vision_end|>`) preceding the instruction,
/// then [`SUFFIX`]. `num_image_tokens[k]` is the merged vision-token count for reference `k`.
///
/// NB the exact edit template (system prompt + marker/instruction layout) must match the reference
/// ComfyUI-Krea2Edit node the LoRA was trained against; this is validated on real weights in P2.3 — a
/// mismatch shifts the tokenization the LoRA expects (and the [`PREFIX_TOKENS`] drop count).
fn render_with_images(instruction: &str, num_image_tokens: &[usize]) -> String {
    let mut vision = String::new();
    for &n in num_image_tokens {
        vision.push_str(VISION_START);
        for _ in 0..n {
            vision.push_str(IMAGE_PAD);
        }
        vision.push_str(VISION_END);
    }
    format!("{PREFIX}{vision}{instruction}{SUFFIX}")
}

/// The Krea condition tokenizer: the snapshot's `tokenizer/tokenizer.json` wrapped to render the Krea
/// template and encode it.
pub struct KreaTokenizer {
    inner: TextTokenizer,
}

impl KreaTokenizer {
    /// Load from a snapshot's `tokenizer/tokenizer.json`.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
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
        )?;
        Ok(Self { inner })
    }

    /// Encode a rendered string to ids (`add_special_tokens=false`, matching the reference).
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self.inner.encode_ids(text, false)?)
    }

    /// Raw id vector for the templated prompt (parity testing against the reference `input_ids`).
    pub fn ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(&render(prompt))
    }

    /// Token count of the bare [`PREFIX`] (should equal [`PREFIX_TOKENS`]).
    pub fn prefix_len(&self) -> Result<usize> {
        Ok(self.encode(PREFIX)?.len())
    }

    /// Encode the templated prompt → `(input_ids, attention_mask)` `[1, L]` int32 (mask all-ones: no
    /// padding on the natural-length path). The encoder drops the leading [`PREFIX_TOKENS`].
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        let ids = self.ids(prompt)?;
        let len = ids.len() as i32;
        let mask = vec![1i32; ids.len()];
        Ok((
            Array::from_slice(&ids, &[1, len]),
            Array::from_slice(&mask, &[1, len]),
        ))
    }

    /// Encode the image-grounded (edit) template → `(input_ids, attention_mask)` `[1, L]` int32
    /// (mask all-ones). `num_image_tokens[k]` is reference `k`'s merged vision-token count (from
    /// [`mlx_gen_boogu::VisionTower::forward`]) — the number of `<|image_pad|>` placeholders the encoder
    /// then fills with vision features (epic 10871 P2). See [`render_with_images`] for the template caveat.
    pub fn encode_with_images(
        &self,
        instruction: &str,
        num_image_tokens: &[usize],
    ) -> Result<(Array, Array)> {
        let ids = self.encode(&render_with_images(instruction, num_image_tokens))?;
        let len = ids.len() as i32;
        let mask = vec![1i32; ids.len()];
        Ok((
            Array::from_slice(&ids, &[1, len]),
            Array::from_slice(&mask, &[1, len]),
        ))
    }
}
