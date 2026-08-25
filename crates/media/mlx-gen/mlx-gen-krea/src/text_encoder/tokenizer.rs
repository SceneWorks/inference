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

/// Max prompt tokens the text (t2i) execution admits, template prefix included. The twin of
/// candle-gen-krea's `pipeline::MAX_TEXT_TOKENS`, at the same 1024: the repo's deliberate posture
/// (sc-9047) is to **fail loud** at the cap rather than silently truncate the way the upstream
/// reference does at 512, so an over-length prompt returns an actionable admission error instead of
/// running the Qwen3-VL encoder out-of-distribution on a sequence it was never conditioned for.
///
/// Enforced by `check_len` from [`KreaTokenizer::encode_prompt`], and declared on
/// `crate::model::PROMPT_EXECUTIONS[krea_t2i].length` so the cross-backend contract gate sees the
/// same number the runtime enforces.
pub const MAX_TEXT_TOKENS: usize = 1024;

/// Max tokens for the **image-grounded edit** execution — the twin of candle-gen-krea's
/// `pipeline::MAX_EDIT_TOKENS`. Far larger than [`MAX_TEXT_TOKENS`] because the edit template emits
/// one `<|image_pad|>` per merged vision token (a single ~1 MP reference is ~1000 tokens, two push
/// past 2000); the grounded encoder builds an MRoPE table sized to the actual sequence, so this is a
/// guard against a pathologically large reference set rather than a RoPE-table bound. Enforced by
/// `check_len` from [`KreaTokenizer::encode_with_images`].
pub const MAX_EDIT_TOKENS: usize = 8192;

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
        let root = root.as_ref();
        let source = crate::model::ENCODER_CONTRACT.validate_source_against_base(
            &mlx_gen::WeightsSource::Dir(root.join("text_encoder")),
            root,
        )?;
        Self::from_validated_source(&source)
    }

    pub(crate) fn from_validated_source(
        source: &mlx_gen::gen_core::ValidatedEncoderSource,
    ) -> Result<Self> {
        source.read_tokenizer_unchanged(Self::from_path)
    }

    fn from_path(path: &Path) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            path,
            TokenizerConfig {
                // We render the template string ourselves and call `encode_ids` directly, so the
                // config template/padding are inert. `max_length` carries [`MAX_TEXT_TOKENS`] rather
                // than the reference's 512 anyway: inert or not, a second, smaller number in this
                // file would misstate the cap this tokenizer actually enforces (via `check_len`).
                max_length: MAX_TEXT_TOKENS,
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
    ///
    /// An empty or over-[`MAX_TEXT_TOKENS`] sequence is rejected up front with a clear admission
    /// error (sc-9047), matching candle-gen-krea's `KreaTokenizer::encode_prompt`, rather than being
    /// handed to the encoder out-of-distribution.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        let ids = self.ids(prompt)?;
        check_len(ids.len(), MAX_TEXT_TOKENS)?;
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
    /// then fills with vision features (epic 10871 P2). See `render_with_images` for the template caveat.
    pub fn encode_with_images(
        &self,
        instruction: &str,
        num_image_tokens: &[usize],
    ) -> Result<(Array, Array)> {
        let ids = self.encode(&render_with_images(instruction, num_image_tokens))?;
        check_len(ids.len(), MAX_EDIT_TOKENS)?;
        let len = ids.len() as i32;
        let mask = vec![1i32; ids.len()];
        Ok((
            Array::from_slice(&ids, &[1, len]),
            Array::from_slice(&mask, &[1, len]),
        ))
    }
}

/// Validate a templated-prompt token count against the execution's cap: an empty sequence, or one
/// longer than `max_tokens`, returns a clear admission error naming the cap and the actual length —
/// instead of letting the Qwen3-VL encoder run on a sequence outside the trained contract. Pure, so
/// it is unit-testable without a real snapshot tokenizer.
///
/// Deliberately the same shape and wording as candle-gen-krea's `tokenizer::check_len` (sc-9047):
/// both lanes name `max_text_tokens=<cap>` and the actual length, so a caller sees one message
/// regardless of which backend admitted the request.
fn check_len(len: usize, max_tokens: usize) -> Result<()> {
    if len == 0 {
        return Err(mlx_gen::Error::Msg("krea: empty token sequence".into()));
    }
    if len > max_tokens {
        return Err(mlx_gen::Error::Msg(format!(
            "krea: prompt has {len} tokens (incl. the {PREFIX_TOKENS}-token template prefix), \
             exceeds max_text_tokens={max_tokens}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_len_rejects_over_cap_with_clear_message() {
        // An over-length prompt returns an actionable admission error naming the cap and the actual
        // length — NOT a silent out-of-distribution encode (the upstream reference truncates at 512;
        // this lane fails loud at 1024, matching candle-gen-krea).
        let err = check_len(MAX_TEXT_TOKENS + 1, MAX_TEXT_TOKENS)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1025"), "names the actual length: {err}");
        assert!(err.contains("max_text_tokens=1024"), "names the cap: {err}");
    }

    #[test]
    fn check_len_accepts_at_and_below_cap() {
        // Exactly-at-cap is admitted (the bound is inclusive, as on candle), as is anything under it.
        assert!(check_len(MAX_TEXT_TOKENS, MAX_TEXT_TOKENS).is_ok());
        assert!(check_len(1, MAX_TEXT_TOKENS).is_ok());
    }

    #[test]
    fn check_len_rejects_empty() {
        assert!(check_len(0, MAX_TEXT_TOKENS)
            .unwrap_err()
            .to_string()
            .contains("empty token sequence"));
    }

    #[test]
    fn declared_length_policy_matches_the_enforced_caps() {
        // The contract the cross-backend gate reads and the cap `check_len` enforces are the same
        // number on both executions. Weights-free, so it catches a contract edited without the
        // runtime (or vice versa) on every CI run, not only on a real-weight lane.
        use mlx_gen::gen_core::EncoderPromptLengthPolicy::RejectAbove;
        for (purpose, expected) in [
            ("krea_t2i", MAX_TEXT_TOKENS),
            ("krea_edit", MAX_EDIT_TOKENS),
        ] {
            let execution = crate::model::PROMPT_EXECUTIONS
                .iter()
                .find(|e| e.purpose == purpose)
                .unwrap_or_else(|| panic!("no `{purpose}` execution declared"));
            assert_eq!(
                execution.length,
                RejectAbove {
                    max_tokens: expected
                },
                "`{purpose}` declares a length policy the tokenizer does not enforce",
            );
        }
    }

    #[test]
    fn edit_cap_is_larger_than_the_text_cap() {
        // The grounded edit execution admits far more (vision-token pads), so a sequence over the
        // t2i cap but under the edit cap is admitted on the edit path and rejected on the text path.
        let between = MAX_TEXT_TOKENS + 1;
        assert!(between < MAX_EDIT_TOKENS);
        assert!(check_len(between, MAX_EDIT_TOKENS).is_ok());
        assert!(check_len(between, MAX_TEXT_TOKENS).is_err());
        assert!(check_len(MAX_EDIT_TOKENS + 1, MAX_EDIT_TOKENS).is_err());
    }
}
