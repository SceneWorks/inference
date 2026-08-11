//! H3 condition tokenization — the chat template plus the seven special tokens MiniMax declares
//! but never ships in a vocabulary.
//!
//! # `<d>` is declared in exactly one file, and its id is a function of list ORDER
//!
//! MiniMax's only edit to the whole `text_encoder/` component is `tokenizer_config.json`, where
//! `additional_special_tokens` grows from upstream Qwen's 13 entries to 20. The seven additions —
//! [`MINIMAX_ADDED_SPECIALS`] — appear in **no** vocabulary artifact: not `vocab.json`, not
//! `tokenizer.json`'s `added_tokens`, not `added_tokens_decoder`. `transformers` assigns their ids
//! at load time by appending, in list order, past the last real added token (151668):
//!
//! | id | token | | id | token |
//! |---|---|---|---|---|
//! | 151669 | `<d>` | | 151673 | `<|lyrics_end|>` |
//! | 151670 | `</d>` | | 151674 | `<|caption_start|>` |
//! | 151671 | `<|cutoff|>` | | 151675 | `<|caption_end|>` |
//! | 151672 | `<|lyrics_start|>` | | | |
//!
//! Two hazards follow, and both are tested:
//!
//! 1. **Loading `tokenizer.json` alone silently mis-tokenizes.** Its vocabulary stops at 151669
//!    entries, so a bare fast tokenizer BPE-splits `<d>` into `[90707, 29]` (`'<d'`, `'>'`) instead
//!    of the single id 151669. This type registers the specials so the ids match `transformers`.
//! 2. **The ids are positional.** Reordering that JSON array repoints `<d>`, so
//!    [`MiniMaxH3Tokenizer::from_snapshot`] *derives* the map from the shipped file rather than
//!    trusting the constants, and [`SpecialTokens::derive`] is pinned by its own test.
//!
//! # `<d>` is inert against the open weights
//!
//! Because the checkpoint is upstream Qwen3-VL-32B unmodified (see the module docs), embedding rows
//! 151669-151675 were never trained: their L2 norms (mean 0.505) are statistically identical to the
//! unused padding tail at 151676-151935 (mean 0.503), and clearly separated from Qwen's genuinely
//! trained added tokens (mean 0.662). The card's `<d>[English] …</d>` prompt syntax is consumed by
//! the withheld hosted **H3-Context-IR** component, not by these weights. This module tokenizes
//! `<d>` *correctly* — it does not claim the model understands it.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::Array;
use tokenizers::{AddedToken, Tokenizer};

use mlx_gen::{Error, Result};

/// The seven specials MiniMax adds over upstream Qwen, in the exact order they appear in
/// `tokenizer_config.json`'s `additional_special_tokens`. **Order is the id assignment**, so this
/// list is not merely a set.
pub const MINIMAX_ADDED_SPECIALS: [&str; 7] = [
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
];

/// The first id `transformers` assigns to a MiniMax-added special — one past the last entry in
/// `added_tokens_decoder` (151668).
pub const FIRST_ADDED_ID: i32 = 151669;

/// User-turn template prefix rendered by the shipped `chat_template.json` for a bare text prompt.
///
/// Derived by running `apply_chat_template([{"role": "user", "content": prompt}],
/// add_generation_prompt=True)` against the shipped template and splitting on the prompt — **not**
/// asserted from the template string here. `tests/te_parity.rs` pins this against a fixture dumped
/// from the real `chat_template.json`.
pub const PREFIX: &str = "<|im_start|>user\n";

/// Generation cue the same render appends after the prompt.
pub const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Leading template tokens dropped from the conditioning. [`PREFIX`] tokenizes to exactly this many
/// (`[151644, 872, 198]`).
///
/// H3 ships no conditioner, so there is no published `prompt_template_encode_start_idx` to copy:
/// this is *derived* from the shipped template. A system turn would lengthen it (14 tokens for a
/// short system message), which is why [`MiniMaxH3Tokenizer::prefix_len`] exists — callers that
/// introduce a system prompt must re-derive rather than reuse this constant.
pub const PREFIX_TOKENS: usize = 3;

/// Qwen `<|endoftext|>` — the pad id (unused on the natural-length path).
const PAD_TOKEN_ID: i32 = 151643;

const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";

/// The resolved id for every special token this crate names, derived from the shipped
/// `tokenizer_config.json` rather than hardcoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// `additional_special_tokens` → id, in declaration order.
    pub ids: BTreeMap<String, i32>,
}

impl SpecialTokens {
    /// Derive the id map the way `transformers` does: walk `additional_special_tokens` in order,
    /// keep the id of anything already in the vocabulary, and assign `next_free_id`, incrementing,
    /// to anything that is not.
    ///
    /// `known` resolves a token already present in `tokenizer.json`; `next_free_id` is one past the
    /// highest existing id (151669 for the shipped file).
    pub fn derive(
        additional_special_tokens: &[String],
        known: impl Fn(&str) -> Option<i32>,
        next_free_id: i32,
    ) -> Self {
        let mut ids = BTreeMap::new();
        let mut next = next_free_id;
        for tok in additional_special_tokens {
            let id = known(tok).unwrap_or_else(|| {
                let id = next;
                next += 1;
                id
            });
            ids.insert(tok.clone(), id);
        }
        Self { ids }
    }

    /// Look up one token's id.
    pub fn get(&self, token: &str) -> Option<i32> {
        self.ids.get(token).copied()
    }

    /// The `<d>` id — 151669 for the shipped tokenizer config.
    pub fn dialogue_open(&self) -> Option<i32> {
        self.get("<d>")
    }

    /// The `</d>` id — 151670 for the shipped tokenizer config.
    pub fn dialogue_close(&self) -> Option<i32> {
        self.get("</d>")
    }
}

/// The H3 condition tokenizer: the snapshot's `tokenizer/tokenizer.json`, with MiniMax's
/// `additional_special_tokens` registered so ids match `transformers`.
pub struct MiniMaxH3Tokenizer {
    inner: Tokenizer,
    specials: SpecialTokens,
}

impl MiniMaxH3Tokenizer {
    /// Load from a snapshot root, reading `tokenizer/tokenizer.json` and deriving the special-token
    /// map from `tokenizer/tokenizer_config.json`.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let dir = root.as_ref().join("tokenizer");
        let cfg = std::fs::read_to_string(dir.join("tokenizer_config.json")).map_err(|e| {
            Error::Msg(format!(
                "minimax-h3 tokenizer: read {}: {e}",
                dir.join("tokenizer_config.json").display()
            ))
        })?;
        Self::from_files(dir.join("tokenizer.json"), &cfg)
    }

    /// Load from an explicit `tokenizer.json` path plus the text of a `tokenizer_config.json`.
    /// Split out so the derivation is testable without a snapshot.
    pub fn from_files(tokenizer_json: impl AsRef<Path>, tokenizer_config: &str) -> Result<Self> {
        let mut inner = Tokenizer::from_file(tokenizer_json.as_ref())
            .map_err(|e| Error::Msg(format!("minimax-h3 tokenizer: load tokenizer.json: {e}")))?;

        let cfg: serde_json::Value = serde_json::from_str(tokenizer_config).map_err(|e| {
            Error::Msg(format!("minimax-h3 tokenizer: parse tokenizer_config: {e}"))
        })?;
        let additional: Vec<String> = cfg
            .get("additional_special_tokens")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        if additional.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 tokenizer: tokenizer_config.json has no additional_special_tokens; \
                 `<d>` and the six other MiniMax specials are declared nowhere else"
                    .into(),
            ));
        }

        // Register anything the fast tokenizer does not already know, IN ORDER — this is what makes
        // `<d>` a single id instead of a two-token BPE split.
        let next_free_id = inner.get_vocab_size(true) as i32;
        let unknown: Vec<AddedToken> = additional
            .iter()
            .filter(|t| inner.token_to_id(t).is_none())
            .map(|t| AddedToken::from(t.clone(), true))
            .collect();
        if !unknown.is_empty() {
            inner.add_special_tokens(&unknown);
        }

        let specials = SpecialTokens::derive(
            &additional,
            |t| inner.token_to_id(t).map(|id| id as i32),
            next_free_id,
        );
        Ok(Self { inner, specials })
    }

    /// The derived special-token id map.
    pub fn specials(&self) -> &SpecialTokens {
        &self.specials
    }

    /// Encode a rendered string to ids (`add_special_tokens=false`, matching the reference path).
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Msg(format!("minimax-h3 tokenizer: encode: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Render the user-turn template around `prompt`.
    fn render(prompt: &str) -> String {
        format!("{PREFIX}{prompt}{SUFFIX}")
    }

    /// Render the grounded template: one `<|vision_start|><|image_pad|>×n<|vision_end|>` block per
    /// reference, before the prompt.
    fn render_with_images(prompt: &str, num_image_tokens: &[usize]) -> String {
        let mut vision = String::new();
        for &n in num_image_tokens {
            vision.push_str(VISION_START);
            for _ in 0..n {
                vision.push_str(IMAGE_PAD);
            }
            vision.push_str(VISION_END);
        }
        format!("{PREFIX}{vision}{prompt}{SUFFIX}")
    }

    /// Raw id vector for the templated prompt (parity testing against the reference `input_ids`).
    pub fn ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(&Self::render(prompt))
    }

    /// Token count of the bare [`PREFIX`] — should equal [`PREFIX_TOKENS`]. Callers that introduce
    /// a system turn must use this rather than the constant.
    pub fn prefix_len(&self) -> Result<usize> {
        Ok(self.encode(PREFIX)?.len())
    }

    /// Encode ids for an arbitrary pre-rendered string (used by the template fixtures).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<i32>> {
        self.encode(text)
    }

    /// Encode the templated prompt → `(input_ids, attention_mask)` `[1, L]` int32, mask all-ones
    /// (no padding on the natural-length path).
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        Self::to_arrays(self.ids(prompt)?)
    }

    /// Encode the grounded template → `(input_ids, attention_mask)` `[1, L]` int32.
    /// `num_image_tokens[k]` is reference `k`'s merged vision-token count.
    pub fn encode_with_images(
        &self,
        prompt: &str,
        num_image_tokens: &[usize],
    ) -> Result<(Array, Array)> {
        Self::to_arrays(self.encode(&Self::render_with_images(prompt, num_image_tokens))?)
    }

    fn to_arrays(ids: Vec<i32>) -> Result<(Array, Array)> {
        let len = ids.len() as i32;
        let mask = vec![1i32; ids.len()];
        Ok((
            Array::from_slice(&ids, &[1, len]),
            Array::from_slice(&mask, &[1, len]),
        ))
    }

    /// The pad id, exposed for callers that pad to a fixed length.
    pub const fn pad_token_id() -> i32 {
        PAD_TOKEN_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream Qwen's 13 `additional_special_tokens`, in order.
    const QWEN_SPECIALS: [&str; 13] = [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ];

    /// The shipped ids for the 13 Qwen specials.
    fn qwen_known(tok: &str) -> Option<i32> {
        let table = [
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
            ("<|object_ref_start|>", 151646),
            ("<|object_ref_end|>", 151647),
            ("<|box_start|>", 151648),
            ("<|box_end|>", 151649),
            ("<|quad_start|>", 151650),
            ("<|quad_end|>", 151651),
            ("<|vision_start|>", 151652),
            ("<|vision_end|>", 151653),
            ("<|vision_pad|>", 151654),
            ("<|image_pad|>", 151655),
            ("<|video_pad|>", 151656),
        ];
        table.iter().find(|(t, _)| *t == tok).map(|(_, id)| *id)
    }

    fn shipped_list() -> Vec<String> {
        QWEN_SPECIALS
            .iter()
            .chain(MINIMAX_ADDED_SPECIALS.iter())
            .map(|s| (*s).to_owned())
            .collect()
    }

    /// The derivation must reproduce the ids `transformers` actually assigns (measured against the
    /// real snapshot): known tokens keep their vocabulary id, the seven new ones get 151669-151675
    /// in declaration order.
    #[test]
    fn derives_the_transformers_id_assignment() {
        let st = SpecialTokens::derive(&shipped_list(), qwen_known, FIRST_ADDED_ID);
        assert_eq!(st.get("<|im_start|>"), Some(151644));
        assert_eq!(st.get("<|video_pad|>"), Some(151656));
        assert_eq!(st.get("<d>"), Some(151669));
        assert_eq!(st.get("</d>"), Some(151670));
        assert_eq!(st.get("<|cutoff|>"), Some(151671));
        assert_eq!(st.get("<|lyrics_start|>"), Some(151672));
        assert_eq!(st.get("<|lyrics_end|>"), Some(151673));
        assert_eq!(st.get("<|caption_start|>"), Some(151674));
        assert_eq!(st.get("<|caption_end|>"), Some(151675));
        assert_eq!(st.dialogue_open(), Some(151669));
        assert_eq!(st.dialogue_close(), Some(151670));
    }

    /// **The id is a function of list ORDER, not of the token string.** Swapping two entries must
    /// move the ids — if this ever passes unchanged, the derivation has stopped being positional
    /// and a future upstream reorder would go undetected.
    #[test]
    fn added_token_ids_follow_declaration_order() {
        let mut reordered = shipped_list();
        let n = reordered.len();
        reordered.swap(n - 7, n - 6); // swap `<d>` and `</d>`
        let st = SpecialTokens::derive(&reordered, qwen_known, FIRST_ADDED_ID);
        assert_eq!(st.get("</d>"), Some(151669));
        assert_eq!(st.get("<d>"), Some(151670));
        // The pre-existing Qwen tokens are unaffected — they resolve from the vocabulary.
        assert_eq!(st.get("<|im_start|>"), Some(151644));
    }

    /// A token already in the vocabulary must never be re-assigned a fresh id.
    #[test]
    fn known_tokens_keep_their_vocabulary_id() {
        let st = SpecialTokens::derive(&shipped_list(), qwen_known, FIRST_ADDED_ID);
        for t in QWEN_SPECIALS {
            assert_eq!(st.get(t), qwen_known(t), "{t} was re-assigned");
        }
    }

    /// The seven MiniMax ids must be contiguous from [`FIRST_ADDED_ID`] and land inside the
    /// embedding matrix (`vocab_size` 151936) — otherwise they would index out of the table.
    #[test]
    fn added_ids_are_contiguous_and_within_the_embedding_table() {
        let st = SpecialTokens::derive(&shipped_list(), qwen_known, FIRST_ADDED_ID);
        for (i, t) in MINIMAX_ADDED_SPECIALS.iter().enumerate() {
            assert_eq!(st.get(t), Some(FIRST_ADDED_ID + i as i32), "{t}");
        }
        let max = FIRST_ADDED_ID + MINIMAX_ADDED_SPECIALS.len() as i32 - 1;
        assert_eq!(max, 151675);
        assert!(max < super::super::MiniMaxH3TeConfig::qwen3_vl_32b().vocab_size);
    }

    /// The rendered template must be exactly prefix + prompt + suffix, with no stray whitespace.
    #[test]
    fn render_is_prefix_prompt_suffix() {
        assert_eq!(
            MiniMaxH3Tokenizer::render("a cat"),
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// The grounded render places one vision block per reference, before the prompt, with exactly
    /// `n` image pads each.
    #[test]
    fn grounded_render_places_one_vision_block_per_reference() {
        let r = MiniMaxH3Tokenizer::render_with_images("x", &[2, 1]);
        assert_eq!(
            r,
            "<|im_start|>user\n\
             <|vision_start|><|image_pad|><|image_pad|><|vision_end|>\
             <|vision_start|><|image_pad|><|vision_end|>\
             x<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(r.matches(IMAGE_PAD).count(), 3);
        assert_eq!(r.matches(VISION_START).count(), 2);
    }
}
