//! S6 — the LTX prompt tokenizers.
//!
//! [`LtxTokenizer`] is the **2.3** path (Gemma-3). The reference `LTX2TextEncoder.encode` runs the
//! HF Gemma tokenizer on the **raw prompt** (no chat template) with `add_special_tokens=True` (a
//! leading `<bos>`, no EOS), truncates to `max_length`, and **left-pads** to `max_length` with
//! `<pad>` (id 0) — `padding_side="left"`. Left-padding matters: it places the real tokens at the
//! high RoPE positions `[max_length−L, max_length)`, which is what the S1 Gemma forward was validated
//! against. Built on the shared core [`TextTokenizer`] (HF `tokenizer.json`, `ChatTemplate::None`);
//! the left-pad is applied here (core pads right) to keep the change crate-local.
//!
//! [`Ltx25Tokenizer`] is the **2.5** path (Gemma 4), built from the tokenizer packed **inside** the
//! single-file text encoder — no separate Gemma snapshot at all (sc-18762). Both run the same
//! `gen_core::tokenizer::ensure_single_leading_bos` policy, because the two generations break
//! differently: Gemma 4's `tokenizer.json` post-processor emits no `<bos>` (measured), while
//! Gemma 3's emits one and would be double-BOSed by an unconditional prepend.

use std::path::Path;

use mlx_rs::Array;

pub use mlx_gen::gen_core::gemma_assets::GemmaAssets;
use mlx_gen::gen_core::gemma_assets::LtxGemmaTokenizer;
use mlx_gen::tokenizer::{
    ensure_single_leading_bos, to_arrays, ChatTemplate, TextTokenizer, TokenizerConfig,
};
use mlx_gen::{Error, Result};

/// The backend-neutral packed-asset layer, re-exported so the 2.5 **weight** loader reaches the
/// same unpacked config/sidecars and the same key canonicalization this tokenizer used, rather than
/// re-deriving them. [`GemmaTeKeyMap`] accepts the ComfyUI-flattened and the legacy HF tower
/// spellings and turns every missed lookup into an error — the strict counterpart to the non-strict
/// load that once left 11 tower tensors at random init.
pub use mlx_gen::gen_core::gemma_assets::{
    flatten_gemma4_unified_key, is_gemma_asset_key, GemmaTeKeyMap,
};

/// Gemma `<pad>` token id (`tokenizer_config.json`).
const GEMMA_PAD_ID: i32 = 0;
/// The literal `<bos>` piece both Gemma generations use; its id is resolved from the vocabulary
/// rather than hard-coded, so a re-tokenized checkpoint that moves it fails loudly.
const GEMMA_BOS_TOKEN: &str = "<bos>";

/// The LTX-2.3 Gemma prompt tokenizer.
pub struct LtxTokenizer {
    inner: TextTokenizer,
    bos_id: i32,
}

impl LtxTokenizer {
    /// Load `tokenizer.json` from a Gemma snapshot directory.
    pub fn from_dir(gemma_dir: &Path) -> Result<Self> {
        let path = gemma_dir.join("tokenizer.json");
        if !path.exists() {
            return Err(Error::Msg(format!(
                "ltx tokenizer: {} not found — point LoadSpec::text_encoder at a gemma-3-12b-it \
                 snapshot dir containing tokenizer.json",
                path.display()
            )));
        }
        // No core-side truncation/padding: encode() truncates + left-pads itself (core pads right).
        let cfg = TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: GEMMA_PAD_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        };
        let inner = TextTokenizer::from_file(&path, cfg)?;
        let bos_id = inner.token_to_id(GEMMA_BOS_TOKEN).ok_or_else(|| {
            Error::Msg(format!(
                "ltx tokenizer: {} has no {GEMMA_BOS_TOKEN} token — the LTX encode path requires a \
                 leading BOS",
                path.display()
            ))
        })? as i32;
        Ok(Self { inner, bos_id })
    }

    /// Encode a raw prompt → `(1, max_length)` left-padded int32 `input_ids` + `attention_mask`.
    /// Mirrors the reference: exactly one leading `<bos>`, right-truncated to `max_length`, then
    /// left-padded with `<pad>`.
    ///
    /// Gemma-3's `tokenizer.json` post-processor already supplies the `<bos>`, so
    /// [`ensure_single_leading_bos`] is normally a no-op here — it is the explicit guard against the
    /// two ways this goes wrong (a missing BOS from a post-processor-less tokenizer, a duplicate one
    /// from an unconditional prepend), and it is the same policy the 2.5 path runs (sc-18762).
    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        if prompt.is_empty() {
            return Err(Error::Msg("ltx tokenizer: empty prompt".into()));
        }
        let out = self.inner.tokenize(prompt)?; // (1, L): <bos> + tokens, mask all 1
        let mut ids: Vec<i32> = out.ids.clone();
        if ids.len() > max_length {
            ids.truncate(max_length); // HF truncation=True keeps the leading max_length tokens
        }
        ensure_single_leading_bos(&mut ids, self.bos_id, max_length);
        let valid = ids.len();
        let pad = max_length - valid;
        // Left-pad: <pad>×pad ++ ids ; mask 0×pad ++ 1×valid.
        let mut padded = vec![GEMMA_PAD_ID; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0i32; pad];
        mask.resize(max_length, 1); // pad zeros already in place; fill the valid tail with 1s
        let n = max_length as i32;
        Ok((
            Array::from_slice(&padded, &[1, n]),
            Array::from_slice(&mask, &[1, n]),
        ))
    }

    /// Tokenize an **already chat-templated** string to a flat id list with `add_special_tokens=false`
    /// (no auto BOS — the template supplies the `<start_of_turn>` markers itself). The prompt-enhancer
    /// path (sc-2845) uses this, mirroring the reference `processor(formatted, add_special_tokens=False)`.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<i32>> {
        self.inner.encode_ids(text, false).map_err(Into::into)
    }

    /// Detokenize generated ids → text, dropping special tokens — the reference
    /// `processor.decode(generated_tokens, skip_special_tokens=True)`.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&u, true).map_err(Into::into)
    }
}

/// The LTX-**2.5** prompt tokenizer (Gemma 4), built entirely from the single-file text encoder.
///
/// LTX-2.3 needed a separate ~26.4 GB `google/gemma-3-12b-it` snapshot as a manifest co-requisite
/// just to reach `tokenizer.json`. The 2.5 text encoder packs `tokenizer.json` and its
/// `tokenizer_config.json` sidecar as tensors inside the same `.safetensors`, so
/// [`Ltx25Tokenizer::from_packed_te_file`] reads them with a header parse plus two seeks and needs
/// **no** other file (sc-18762).
#[derive(Debug)]
pub struct Ltx25Tokenizer {
    inner: LtxGemmaTokenizer,
}

impl Ltx25Tokenizer {
    /// Unpack the tokenizer out of the single-file 2.5 text encoder. Only the asset tensors' byte
    /// ranges are read — never the multi-GB weight payload.
    pub fn from_packed_te_file(te_path: &Path) -> Result<Self> {
        let assets = GemmaAssets::load(te_path)?;
        Self::from_assets(&assets)
    }

    /// Build from already-loaded [`GemmaAssets`] — for a caller that unpacks the file once and
    /// shares the config/sidecars with the weight loader.
    pub fn from_assets(assets: &GemmaAssets) -> Result<Self> {
        Ok(Self {
            inner: LtxGemmaTokenizer::from_assets(assets)?,
        })
    }

    /// The resolved `<bos>` id (2 on the shipped 2.5 tokenizer).
    pub fn bos_id(&self) -> i32 {
        self.inner.bos_id()
    }

    /// The resolved `<pad>` id (0 on the shipped 2.5 tokenizer).
    pub fn pad_id(&self) -> i32 {
        self.inner.pad_id()
    }

    /// Encode a raw prompt → `(1, max_length)` left-padded int32 `input_ids` + `attention_mask`,
    /// with exactly one leading `<bos>`. Unlike the 2.3 path an empty prompt is legal: it encodes
    /// to a lone `<bos>`, which is what the reference produces.
    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        Ok(to_arrays(&self.inner.encode(prompt, max_length)?))
    }

    /// Detokenize ids → text, dropping special tokens.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&u, true).map_err(Into::into)
    }
}
