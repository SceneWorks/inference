//! H3 condition tokenization — the presentation the DiT is conditioned on, plus the seven special
//! tokens MiniMax declares but never ships in a vocabulary.
//!
//! # There is no chat template (sc-18741)
//!
//! The official conditioner builds every presentation as `tokenizer(text,
//! add_special_tokens=False)` over text it assembles itself, with **no chat template**. See
//! [`APPLIES_CHAT_TEMPLATE`] for the full contract and for what sc-17143 got wrong and why.
//!
//! `text_encoder/chat_template.json` exists in the component only because the text encoder is a
//! byte-identical copy of `Qwen/Qwen3-VL-32B-Instruct`, where that file drives *chat*. Its presence
//! is not evidence that MiniMax-H3 conditions through it.
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

/// **The presentation contract: MiniMax-H3 applies no chat template and adds no special tokens.**
///
/// Every presentation the official conditioner builds — `t2va`, `fl2va` and `ref2va` alike — is
/// `tokenizer(text, add_special_tokens=False)` over text this module assembles itself. There is no
/// `<|im_start|>user\n` turn, no `<|im_end|>\n<|im_start|>assistant\n` generation cue, and
/// therefore **no template prefix to slice off the context**.
///
/// This constant exists so the absence is a *pinned* property with a test attached rather than the
/// silent consequence of some code having been deleted. See
/// `tests/te_parity.rs::presentation_applies_no_chat_template`, which additionally checks the port
/// against the exact ids the (wrong) chat-template render would have produced.
///
/// # sc-18741
///
/// sc-17143 could not check this: it was written before `MiniMaxH3` landed on diffusers `main`
/// (PR #14355, merged 2026-08-05, in no tagged release), so no reference conditioner existed. It
/// reasonably *derived* a 3-token prefix by rendering the shipped `chat_template.json` — a file
/// which is present in the component only because the text encoder is a byte-identical copy of
/// `Qwen/Qwen3-VL-32B-Instruct`, where it is used for chat, not for conditioning. The port then
/// rendered `PREFIX + prompt + SUFFIX` and dropped 3 leading tokens. Measured against the real
/// tokenizer, that slice lands exactly on the `<|im_start|>user\n` boundary for ordinary prompts, so
/// the damage is **not** lost prompt tokens — it is the 5-token generation cue
/// `<|im_end|>\n<|im_start|>assistant\n` (`[151645, 198, 151644, 77091, 198]`) that nothing ever
/// removed. The DiT was conditioned on `prompt + 5 rows of chat-turn control tokens`: 16 rows
/// instead of 11 for a 9-word prompt. A prompt that *begins* with whitespace loses a real token too,
/// because the tokenizer merges the template's trailing newline into it. Nothing failed loudly; the
/// conditioning was simply off-distribution.
pub const APPLIES_CHAT_TEMPLATE: bool = false;

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

    /// Encode a string to ids with `add_special_tokens=false` — the reference conditioner's one and
    /// only tokenizer call.
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Msg(format!("minimax-h3 tokenizer: encode: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// The `ref2va` / `fl2va` visual presentation: a `"<Picture i>: "` label and a vision block per
    /// reference, then the prompt **verbatim**.
    ///
    /// Mirrors `MiniMaxH3FL2VATextEncoderStep` / `MiniMaxH3Ref2VATextEncoderStep`: labels are
    /// numbered from 1 per modality, and there is no chat template anywhere in it. sc-17148 owns
    /// the rest of that presentation (keyframe conditioning latents, the per-row modality tags that
    /// mark vision rows as *video*, and the `<Audio j>` / `<Video k>` limbs); this covers the image
    /// half the crate already reaches through [`crate::text_encoder::encode_grounded`].
    fn render_with_images(prompt: &str, num_image_tokens: &[usize]) -> String {
        let mut out = String::new();
        for (i, &n) in num_image_tokens.iter().enumerate() {
            out.push_str(&format!("<Picture {}>: ", i + 1));
            out.push_str(VISION_START);
            for _ in 0..n {
                out.push_str(IMAGE_PAD);
            }
            out.push_str(VISION_END);
        }
        out.push_str(prompt);
        out
    }

    /// Raw id vector for a `t2va` prompt: the prompt **verbatim**, no chat template, no special
    /// tokens — `tokenizer(prompt, add_special_tokens=False)` exactly as the reference conditioner
    /// spells it. See [`APPLIES_CHAT_TEMPLATE`].
    pub fn ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(prompt)
    }

    /// Encode ids for an arbitrary pre-rendered string (used by the presentation fixtures).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<i32>> {
        self.encode(text)
    }

    /// Encode a `t2va` prompt → `(input_ids, attention_mask)` `[1, L]` int32, mask all-ones (no
    /// padding on the natural-length path).
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        Self::to_arrays(self.ids(prompt)?)
    }

    /// Encode the visual presentation → `(input_ids, attention_mask)` `[1, L]` int32.
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

    /// **The sc-18741 contract.** No chat template is applied anywhere, so the presentation for a
    /// bare prompt is the prompt itself — byte for byte, no turn markers, no generation cue.
    ///
    /// Written against the literal template strings sc-17143 used, so reintroducing either one
    /// fails here rather than silently shifting the conditioning.
    #[test]
    fn no_chat_template_is_applied() {
        const { assert!(!APPLIES_CHAT_TEMPLATE) };
        let presentation = MiniMaxH3Tokenizer::render_with_images("a cat", &[]);
        assert_eq!(
            presentation, "a cat",
            "a reference-free presentation is the prompt"
        );
        for residue in [
            "<|im_start|>",
            "<|im_end|>",
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        ] {
            assert!(
                !presentation.contains(residue),
                "chat-template residue {residue:?} is back in the presentation (sc-18741)"
            );
        }
    }

    /// The visual presentation places a numbered `"<Picture i>: "` label and one vision block per
    /// reference, then the prompt verbatim — and still no template.
    #[test]
    fn grounded_render_labels_each_reference_and_appends_the_prompt_verbatim() {
        let r = MiniMaxH3Tokenizer::render_with_images("x", &[2, 1]);
        assert_eq!(
            r,
            "<Picture 1>: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>\
             <Picture 2>: <|vision_start|><|image_pad|><|vision_end|>\
             x"
        );
        assert_eq!(r.matches(IMAGE_PAD).count(), 3);
        assert_eq!(r.matches(VISION_START).count(), 2);
        // Labels are numbered from 1, per modality — not 0-indexed.
        assert!(r.starts_with("<Picture 1>: "));
        assert!(
            r.ends_with('x'),
            "the prompt is appended verbatim, with no cue after it"
        );
        assert!(
            !r.contains("<|im_start|>"),
            "no chat template on the grounded path either"
        );
    }
}
