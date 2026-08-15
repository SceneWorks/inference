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
//!
//! # Why `tokenizers` directly instead of `gen_core::tokenizer::TextTokenizer`
//!
//! Same reason as the MLX sibling: hazard 1 above is only avoidable by *registering* the seven
//! specials as added tokens, and [`candle_gen::gen_core::tokenizer::TextTokenizer`] exposes no way
//! to do that — it holds its `tokenizers::Tokenizer` privately, and its public surface carries
//! **no `add_special_tokens` and no `token_to_id`** (only construction, `tokenize*`, `encode*_ids`,
//! `decode`, `config` and `constraint_decode_table`). So neither the registration nor
//! [`MiniMaxH3Tokenizer::special_id`]'s vocabulary lookup can be expressed through it.
//!
//! # Where this port differs from the MLX sibling, and why
//!
//! Token ids are **`u32`** here, not the MLX lane's `i32`, and the two tensors are built on a
//! caller-supplied [`Device`] rather than materialising wherever the framework happens to put them:
//!
//! * `tokenizers` already hands back `u32`, and candle's embedding lookup (`index_select`) requires
//!   a `u32` index tensor — which is why [`super::encoder`] reads the ids back as `u32`, as does
//!   the sibling `candle_gen_boogu::text_encoder`. Carrying `i32` through the id space would buy
//!   one lossy cast per encode and no expressiveness: no id in this vocabulary is negative.
//! * The attention mask is `u32` too. It is a 0/1 indicator, never a value in the math — the
//!   encoder folds it into an additive f32 mask itself
//!   (`attention_mask.to_dtype(DType::F32)` in [`super::encoder`]), so emitting f32 here would
//!   only invite it being broadcast-added as-is.
//! * The per-row modality tags are `u32` for a harder reason — [`TEXT_TAG`] / [`VIDEO_TAG`] are
//!   `u32` in [`crate::denoise::packing`] on this side, and `PackedLayout::build` takes
//!   `text_token_tags: &[u32]`. [`MiniMaxH3Tokenizer::encode_fl2va`]'s third return value feeds
//!   that argument directly.
//! * candle tensors are device-bound, unlike MLX's unified-memory arrays, so every path that
//!   produces a tensor takes `device: &Device`.

use std::collections::BTreeMap;
use std::path::Path;

use tokenizers::{AddedToken, Tokenizer};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::denoise::packing::{TEXT_TAG, VIDEO_TAG};
use crate::reference::ReferencePresentation;

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
pub const FIRST_ADDED_ID: u32 = 151669;

/// **The presentation contract: MiniMax-H3 applies no chat template and adds no special tokens.**
///
/// Every presentation the official conditioner builds — `t2va`, `fl2va` and `ref2va` alike — is
/// `tokenizer(text, add_special_tokens=False)` over text this module assembles itself. There is no
/// `<|im_start|>user\n` turn, no `<|im_end|>\n<|im_start|>assistant\n` generation cue, and
/// therefore **no template prefix to slice off the context**.
///
/// This constant exists so the absence is a *pinned* property with a test attached rather than the
/// silent consequence of some code having been deleted — see `no_chat_template_is_applied` below,
/// which is written against the literal template strings sc-17143 used. The MLX sibling
/// additionally holds its port to the exact ids the (wrong) chat-template render would have
/// produced, in `tests/te_parity.rs::presentation_applies_no_chat_template`; that comparison needs
/// a real snapshot tokenizer, so it belongs in this crate's `tests/` rather than here.
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
const PAD_TOKEN_ID: u32 = 151643;

const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";
/// The pad `ref2va` video references use. `fl2va` uses [`IMAGE_PAD`] and never this one, which is
/// what lets the grounded forward tell a clip's blocks from a still's.
const VIDEO_PAD: &str = "<|video_pad|>";

/// One entry of the `ref2va` presentation, before any vocabulary is consulted.
///
/// See [`MiniMaxH3Tokenizer::ref2va_emit_plan`]. The two variants correspond to the reference
/// conditioner's two emit closures, and the distinction is load-bearing at the **tag** level: a
/// `Text` run is tagged [`TEXT_TAG`] and a `Vision` block — start marker, pads and end marker alike
/// — is tagged [`VIDEO_TAG`], which addresses a different block of the DiT's AdaLN modulation table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ref2VaEmit {
    /// A literal string, tokenized verbatim with no special tokens.
    Text(String),
    /// One `<|vision_start|>` … pads … `<|vision_end|>` block.
    Vision {
        /// `true` for `<|video_pad|>` (a clip's block), `false` for `<|image_pad|>` (a still's).
        /// The two pads are what let the grounded forward tell a clip's blocks from a still's.
        video_pad: bool,
        /// Merged vision tokens between the two markers.
        num_tokens: usize,
    },
}

/// The resolved id for every special token this crate names, derived from the shipped
/// `tokenizer_config.json` rather than hardcoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// `additional_special_tokens` → id, in declaration order.
    pub ids: BTreeMap<String, u32>,
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
        known: impl Fn(&str) -> Option<u32>,
        next_free_id: u32,
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
    pub fn get(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// The `<d>` id — 151669 for the shipped tokenizer config.
    pub fn dialogue_open(&self) -> Option<u32> {
        self.get("<d>")
    }

    /// The `</d>` id — 151670 for the shipped tokenizer config.
    pub fn dialogue_close(&self) -> Option<u32> {
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
            CandleError::Msg(format!(
                "minimax-h3 tokenizer: read {}: {e}",
                dir.join("tokenizer_config.json").display()
            ))
        })?;
        Self::from_files(dir.join("tokenizer.json"), &cfg)
    }

    /// Load from an explicit `tokenizer.json` path plus the text of a `tokenizer_config.json`.
    /// Split out so the derivation is testable without a snapshot.
    pub fn from_files(tokenizer_json: impl AsRef<Path>, tokenizer_config: &str) -> Result<Self> {
        let mut inner = Tokenizer::from_file(tokenizer_json.as_ref()).map_err(|e| {
            CandleError::Msg(format!("minimax-h3 tokenizer: load tokenizer.json: {e}"))
        })?;

        let cfg: serde_json::Value = serde_json::from_str(tokenizer_config).map_err(|e| {
            CandleError::Msg(format!("minimax-h3 tokenizer: parse tokenizer_config: {e}"))
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
            return Err(CandleError::Msg(
                "minimax-h3 tokenizer: tokenizer_config.json has no additional_special_tokens; \
                 `<d>` and the six other MiniMax specials are declared nowhere else"
                    .into(),
            ));
        }

        // Register anything the fast tokenizer does not already know, IN ORDER — this is what makes
        // `<d>` a single id instead of a two-token BPE split.
        let next_free_id = inner.get_vocab_size(true) as u32;
        let unknown: Vec<AddedToken> = additional
            .iter()
            .filter(|t| inner.token_to_id(t).is_none())
            .map(|t| AddedToken::from(t.clone(), true))
            .collect();
        if !unknown.is_empty() {
            inner.add_special_tokens(&unknown);
        }

        let specials = SpecialTokens::derive(&additional, |t| inner.token_to_id(t), next_free_id);
        Ok(Self { inner, specials })
    }

    /// The derived special-token id map.
    pub fn specials(&self) -> &SpecialTokens {
        &self.specials
    }

    /// Encode a string to ids with `add_special_tokens=false` — the reference conditioner's one and
    /// only tokenizer call.
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 tokenizer: encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The `ref2va` / `fl2va` visual presentation: a `"<Picture i>: "` label and a vision block per
    /// reference, then the prompt **verbatim**.
    ///
    /// Mirrors `MiniMaxH3FL2VATextEncoderStep` / `MiniMaxH3Ref2VATextEncoderStep`: labels are
    /// numbered from 1 per modality, and there is no chat template anywhere in it. This is the
    /// **untagged** single-string path — the image half the crate reaches through
    /// [`crate::text_encoder::encode_grounded`]. [`Self::encode_fl2va`] is the same presentation
    /// with its per-row modality tags, and sc-17157 owns the `ref2va` remainder (the `<Audio j>` /
    /// `<Video k>` limbs and the timestamped `<|video_pad|>` blocks).
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
    pub fn ids(&self, prompt: &str) -> Result<Vec<u32>> {
        self.encode(prompt)
    }

    /// Encode ids for an arbitrary pre-rendered string (used by the presentation fixtures).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text)
    }

    /// Encode a `t2va` prompt → `(input_ids, attention_mask)` `[1, L]` u32 on `device`, mask
    /// all-ones (no padding on the natural-length path).
    pub fn encode_prompt(&self, prompt: &str, device: &Device) -> Result<(Tensor, Tensor)> {
        Self::to_arrays(self.ids(prompt)?, device)
    }

    /// Encode the visual presentation → `(input_ids, attention_mask)` `[1, L]` u32 on `device`.
    /// `num_image_tokens[k]` is reference `k`'s merged vision-token count.
    pub fn encode_with_images(
        &self,
        prompt: &str,
        num_image_tokens: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        Self::to_arrays(
            self.encode(&Self::render_with_images(prompt, num_image_tokens))?,
            device,
        )
    }

    /// Resolve one special token's id from the loaded vocabulary.
    ///
    /// Read rather than hardcoded, for the same reason [`SpecialTokens::derive`] exists:
    /// `tokenizer_config.json` is the only file MiniMax changed from upstream Qwen3-VL, so an id
    /// written as a literal here is an id nothing checks.
    pub fn special_id(&self, token: &str) -> Result<u32> {
        self.inner.token_to_id(token).ok_or_else(|| {
            CandleError::Msg(format!(
                "minimax-h3 tokenizer: `{token}` is not in the vocabulary"
            ))
        })
    }

    /// The `fl2va` presentation **with its per-row modality tags** —
    /// `(input_ids, attention_mask, token_tags)`.
    ///
    /// # Why the tags cannot be recovered afterwards
    ///
    /// A vision block's rows are tagged **video** ([`VIDEO_TAG`], 0) rather than text, and that is
    /// what the DiT's AdaLN modulation keys off: a vision row addresses a different block of the
    /// modulation table from a text row at the same timestep. The `"<Picture i>: "` label wrapped
    /// around it stays **text**.
    ///
    /// Deriving that split from a finished id vector would mean re-locating each block by scanning
    /// for pad ids, so this builds the presentation the way the reference does — ids and tags in
    /// lockstep — and the ids it emits are identical to [`Self::encode_with_images`]'s
    /// single-string path. `token_tags` is the `text_token_tags` argument of
    /// [`crate::denoise::packing::PackedLayout::build`], which is why it is `u32` here where the
    /// MLX sibling returns `i32`.
    ///
    /// `<|image_pad|>` is the pad `fl2va` uses. `<|video_pad|>` belongs to `ref2va` (sc-17157) and
    /// appears nowhere on this path.
    pub fn encode_fl2va(
        &self,
        prompt: &str,
        num_image_tokens: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<u32>)> {
        let (start, pad, end) = (
            self.special_id(VISION_START)?,
            self.special_id(IMAGE_PAD)?,
            self.special_id(VISION_END)?,
        );
        let mut ids: Vec<u32> = Vec::new();
        let mut tags: Vec<u32> = Vec::new();
        for (i, &n) in num_image_tokens.iter().enumerate() {
            if n == 0 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 tokenizer: keyframe {i} contributes no vision tokens"
                )));
            }
            let label = self.encode(&format!("<Picture {}>: ", i + 1))?;
            tags.extend(std::iter::repeat_n(TEXT_TAG, label.len()));
            ids.extend(label);
            // `<|vision_start|>`, every pad, and `<|vision_end|>` are ALL video rows.
            ids.push(start);
            ids.extend(std::iter::repeat_n(pad, n));
            ids.push(end);
            tags.extend(std::iter::repeat_n(VIDEO_TAG, n + 2));
        }
        let prompt_ids = self.encode(prompt)?;
        tags.extend(std::iter::repeat_n(TEXT_TAG, prompt_ids.len()));
        ids.extend(prompt_ids);

        if ids.len() != tags.len() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tokenizer: {} ids against {} tags",
                ids.len(),
                tags.len()
            )));
        }
        let (input_ids, mask) = Self::to_arrays(ids, device)?;
        Ok((input_ids, mask, tags))
    }

    /// Render one video reference's per-block timestamp label, `"<0.2 seconds>"`.
    ///
    /// The reference spells this `f"<{timestamp:.1f} seconds>"`, and Python's `format` rounds
    /// **half to even** on the exact binary value — which is why a 2 fps pair's mean renders as
    /// `"<0.2 seconds>"` rather than `"<0.3 seconds>"`. Rust's `{:.1}` uses the same rule on the
    /// same value, verified over the half-way cases by
    /// `the_timestamp_label_rounds_half_to_even_like_python`. Wrapped in a named function so the
    /// claim has one place to be pinned rather than being an invisible property of a format string.
    pub fn timestamp_label(seconds: f64) -> String {
        format!("<{seconds:.1} seconds>")
    }

    /// The **`ref2va` presentation** with its per-row modality tags —
    /// `(input_ids, attention_mask, token_tags)`.
    ///
    /// The sibling of [`Self::encode_fl2va`], and the differences are all semantic:
    ///
    /// * **Three modalities, numbered independently from 1.** `"<Picture i>"`, `"<Audio j>"` and
    ///   `"<Video k>"` each carry their own counter, so the third image is `<Picture 3>` however
    ///   many clips preceded it.
    /// * **A reference that carries sound emits `"<Audio j>: "` BEFORE its visual label**, mirroring
    ///   the order its rows are packed in. For a standalone audio reference that label is the whole
    ///   contribution.
    /// * **A waveform never reaches the conditioner.** An audio reference contributes a text label
    ///   and *no* vision block — [`ReferencePresentation::Audio`] carries no token count because
    ///   there is nothing to count.
    /// * **A video reference emits one timestamped vision block per merged frame pair**, each
    ///   `"<t seconds>"` label followed by its own `<|video_pad|>` block. `<|video_pad|>` appears
    ///   only here; `fl2va` uses `<|image_pad|>` and nothing else.
    ///
    /// Ids and tags are built in lockstep for the same reason [`Self::encode_fl2va`] does it:
    /// a vision block's rows are tagged **video** ([`VIDEO_TAG`]) and address a different block of
    /// the AdaLN modulation table than the text rows around them, and that split cannot be
    /// recovered from a finished id vector without re-scanning for pad ids.
    pub fn encode_ref2va(
        &self,
        prompt: &str,
        references: &[ReferencePresentation],
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<u32>)> {
        let (start, end) = (self.special_id(VISION_START)?, self.special_id(VISION_END)?);
        let (image_pad, video_pad) = (self.special_id(IMAGE_PAD)?, self.special_id(VIDEO_PAD)?);

        let mut ids: Vec<u32> = Vec::new();
        let mut tags: Vec<u32> = Vec::new();
        for emit in Self::ref2va_emit_plan(prompt, references)? {
            match emit {
                Ref2VaEmit::Text(s) => {
                    let t = self.encode(&s)?;
                    tags.extend(std::iter::repeat_n(TEXT_TAG, t.len()));
                    ids.extend(t);
                }
                Ref2VaEmit::Vision {
                    video_pad: is_video,
                    num_tokens,
                } => {
                    let pad = if is_video { video_pad } else { image_pad };
                    ids.push(start);
                    ids.extend(std::iter::repeat_n(pad, num_tokens));
                    ids.push(end);
                    // `<|vision_start|>`, every pad and `<|vision_end|>` are ALL video rows.
                    tags.extend(std::iter::repeat_n(VIDEO_TAG, num_tokens + 2));
                }
            }
        }

        if ids.len() != tags.len() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tokenizer: {} ids against {} tags",
                ids.len(),
                tags.len()
            )));
        }
        let (input_ids, mask) = Self::to_arrays(ids, device)?;
        Ok((input_ids, mask, tags))
    }

    /// The **ordered emit plan** [`Self::encode_ref2va`] tokenizes — every label and vision block,
    /// in sequence order, with no vocabulary involved.
    ///
    /// Split out because it is the whole of the presentation's *semantics* — the per-modality
    /// numbering, the audio-label-leads rule, one timestamped block per merged frame pair — and none
    /// of it needs a tokenizer. Keeping it inline would leave those rules reachable only through a
    /// path that requires a real `tokenizer.json`, so the only available test would be one that
    /// re-implemented the rules and then agreed with itself.
    ///
    /// The `prompt` is the final [`Ref2VaEmit::Text`], appended verbatim.
    pub fn ref2va_emit_plan(
        prompt: &str,
        references: &[ReferencePresentation],
    ) -> Result<Vec<Ref2VaEmit>> {
        let mut plan: Vec<Ref2VaEmit> = Vec::new();
        let (mut n_image, mut n_video, mut n_audio) = (0usize, 0usize, 0usize);

        for (i, r) in references.iter().enumerate() {
            // The `<Audio j>` label leads for every reference that contributes audio rows —
            // a standalone audio reference and a video reference carrying its own soundtrack alike.
            let has_audio = matches!(
                r,
                ReferencePresentation::Audio
                    | ReferencePresentation::Video {
                        has_audio: true,
                        ..
                    }
            );
            if has_audio {
                n_audio += 1;
                plan.push(Ref2VaEmit::Text(format!("<Audio {n_audio}>: ")));
            }
            match r {
                ReferencePresentation::Audio => {}
                ReferencePresentation::Image { num_tokens } => {
                    if *num_tokens == 0 {
                        return Err(CandleError::Msg(format!(
                            "minimax-h3 tokenizer: reference {i} is an image contributing no \
                             vision tokens"
                        )));
                    }
                    n_image += 1;
                    plan.push(Ref2VaEmit::Text(format!("<Picture {n_image}>: ")));
                    plan.push(Ref2VaEmit::Vision {
                        video_pad: false,
                        num_tokens: *num_tokens,
                    });
                }
                ReferencePresentation::Video {
                    num_tokens,
                    timestamps,
                    ..
                } => {
                    if *num_tokens == 0 {
                        return Err(CandleError::Msg(format!(
                            "minimax-h3 tokenizer: reference {i} is a clip whose vision blocks \
                             contribute no tokens"
                        )));
                    }
                    if timestamps.is_empty() {
                        return Err(CandleError::Msg(format!(
                            "minimax-h3 tokenizer: reference {i} is a clip with no vision blocks"
                        )));
                    }
                    n_video += 1;
                    plan.push(Ref2VaEmit::Text(format!("<Video {n_video}>: ")));
                    for &t in timestamps {
                        plan.push(Ref2VaEmit::Text(Self::timestamp_label(t)));
                        plan.push(Ref2VaEmit::Vision {
                            video_pad: true,
                            num_tokens: *num_tokens,
                        });
                    }
                }
            }
        }
        plan.push(Ref2VaEmit::Text(prompt.to_owned()));
        Ok(plan)
    }

    /// Build the `[1, L]` id / mask pair on `device`.
    ///
    /// Both tensors are **u32**: the ids because candle's embedding lookup is an `index_select`,
    /// which requires an unsigned index tensor, and the mask because it is a 0/1 indicator that the
    /// condition encoder folds into an additive f32 mask itself (the sibling ports read it back
    /// with a `to_dtype` first, so a float mask here would buy nothing and invite it being used
    /// additively as-is). The MLX sibling emits int32 for both; candle has no int32 embedding-index
    /// path, so this is the closest faithful spelling rather than a change of meaning.
    ///
    /// The name mirrors the MLX sibling's `to_arrays` so the two lanes stay greppable path-for-path
    /// and line-for-line, even though what it builds here is a pair of candle tensors.
    fn to_arrays(ids: Vec<u32>, device: &Device) -> Result<(Tensor, Tensor)> {
        let len = ids.len();
        let mask = vec![1u32; len];
        Ok((
            Tensor::from_vec(ids, (1, len), device)?,
            Tensor::from_vec(mask, (1, len), device)?,
        ))
    }

    /// The pad id, exposed for callers that pad to a fixed length.
    pub const fn pad_token_id() -> u32 {
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
    fn qwen_known(tok: &str) -> Option<u32> {
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
            assert_eq!(st.get(t), Some(FIRST_ADDED_ID + i as u32), "{t}");
        }
        let max = FIRST_ADDED_ID + MINIMAX_ADDED_SPECIALS.len() as u32 - 1;
        assert_eq!(max, 151675);
        assert!((max as usize) < super::super::MiniMaxH3TeConfig::qwen3_vl_32b().vocab_size);
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

    /// **`{:.1}` rounds half to EVEN, matching Python's `format`** — which is what makes a 2 fps
    /// pair's mean render as `"<0.2 seconds>"` rather than `"<0.3 seconds>"`.
    ///
    /// The exact-half cases are the whole point, and they are the only ones where half-to-even and
    /// half-away-from-zero disagree. `0.25` is the value the first merged block of every video
    /// reference actually carries (`(0.0 + 0.5) / 2`), so this is not a synthetic corner: a port
    /// that formatted with half-away-from-zero mislabels the first block of every clip.
    ///
    /// Note `0.15` and `0.35` are NOT exact halves in binary — `0.15` is just below and `0.35` just
    /// above — so their correct renderings are `0.1` and `0.3`, decided by the stored value rather
    /// than by the tie rule. They are included because a port that "fixed" the tie rule by adding an
    /// epsilon would break exactly these.
    #[test]
    fn the_timestamp_label_rounds_half_to_even_like_python() {
        // Exact binary halves: ties go to the EVEN neighbour.
        assert_eq!(
            MiniMaxH3Tokenizer::timestamp_label(0.25),
            "<0.2 seconds>",
            "0.25 ties DOWN to the even 0.2 — the first block of every 2 fps clip"
        );
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(0.75), "<0.8 seconds>");
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(1.25), "<1.2 seconds>");
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(2.75), "<2.8 seconds>");
        // Half-away-from-zero would have given 0.3 / 0.8 / 1.3 / 2.8 — so the first three
        // assertions above each discriminate the two rules.

        // Not exact halves: the stored double decides, not the tie rule.
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(0.15), "<0.1 seconds>");
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(0.35), "<0.3 seconds>");

        // Whole and integral values keep their one decimal place.
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(0.0), "<0.0 seconds>");
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(1.0), "<1.0 seconds>");
        assert_eq!(MiniMaxH3Tokenizer::timestamp_label(12.5), "<12.5 seconds>");
    }

    /// The **`ref2va` presentation plan**, driven through the shipped
    /// [`MiniMaxH3Tokenizer::ref2va_emit_plan`] rather than reconstructed here.
    ///
    /// Four things are pinned at once, and each is a rule a plausible port gets wrong:
    ///
    /// * the three modality counters advance **independently** — the second image is
    ///   `<Picture 2>` however many clips preceded it;
    /// * a reference that carries audio emits its `"<Audio j>: "` label **before** its visual one,
    ///   mirroring the order its rows are packed in;
    /// * a video reference emits **one timestamped block per merged frame pair**, on the VIDEO pad;
    /// * an audio reference contributes a label and **no vision block at all**.
    #[test]
    fn the_reference_presentation_numbers_per_modality_and_leads_with_audio() {
        use crate::reference::ReferencePresentation;

        let refs = [
            ReferencePresentation::Image { num_tokens: 4 },
            ReferencePresentation::Video {
                num_tokens: 2,
                timestamps: vec![0.25, 1.0],
                has_audio: true,
            },
            ReferencePresentation::Audio,
            ReferencePresentation::Image { num_tokens: 1 },
        ];
        let plan = MiniMaxH3Tokenizer::ref2va_emit_plan("a cellist", &refs).unwrap();

        let text = |s: &str| Ref2VaEmit::Text(s.to_owned());
        let image_block = |n| Ref2VaEmit::Vision {
            video_pad: false,
            num_tokens: n,
        };
        let video_block = |n| Ref2VaEmit::Vision {
            video_pad: true,
            num_tokens: n,
        };
        assert_eq!(
            plan,
            vec![
                text("<Picture 1>: "),
                image_block(4),
                // The clip's soundtrack label LEADS its visual label.
                text("<Audio 1>: "),
                text("<Video 1>: "),
                text("<0.2 seconds>"),
                video_block(2),
                text("<1.0 seconds>"),
                video_block(2),
                // A standalone audio reference is a label and nothing else.
                text("<Audio 2>: "),
                // ...and the image counter did not advance past it.
                text("<Picture 2>: "),
                image_block(1),
                // The prompt is appended verbatim, with no cue after it.
                text("a cellist"),
            ]
        );

        // Degenerate presentations are refused rather than emitting an empty block, which would
        // tokenize into a `<|vision_start|><|vision_end|>` pair the model has never seen.
        for (bad, expect) in [
            (
                vec![ReferencePresentation::Image { num_tokens: 0 }],
                "contributing no",
            ),
            (
                vec![ReferencePresentation::Video {
                    num_tokens: 0,
                    timestamps: vec![0.0],
                    has_audio: false,
                }],
                "contribute no tokens",
            ),
            (
                vec![ReferencePresentation::Video {
                    num_tokens: 2,
                    timestamps: Vec::new(),
                    has_audio: false,
                }],
                "no vision blocks",
            ),
        ] {
            let e = MiniMaxH3Tokenizer::ref2va_emit_plan("p", &bad)
                .unwrap_err()
                .to_string();
            assert!(e.contains(expect), "expected {expect:?}, got {e}");
        }
    }
}
