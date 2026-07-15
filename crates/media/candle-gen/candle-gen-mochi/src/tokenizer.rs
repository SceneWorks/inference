//! Mochi's T5-XXL tokenizer. The snapshot's `tokenizer/` ships only the sentencepiece `spiece.model`
//! (no `tokenizer.json` a HF fast tokenizer can load), so — exactly like Chroma — we vendor the
//! prebuilt fast-tokenizer JSON (`assets/t5_tokenizer.json`, byte-identical to the mlx-gen-mochi /
//! Chroma asset) and load it from memory. It is the **same** google t5-v1.1-xxl tokenizer FLUX/Chroma
//! use.
//!
//! Mochi's `_get_t5_prompt_embeds` tokenizes with `padding="max_length"`, `max_length=256`,
//! `truncation=True`, `add_special_tokens=True`. We disable the JSON's baked-in padding and pad
//! **explicitly** to [`MAX_SEQUENCE_LENGTH`] on the right with pad id `0`, right-truncating any overflow
//! after the appended EOS `</s>` (id 1) — `text_encoder::tokenize` owns that padding so it can build the
//! matching 0/1 attention mask.

use candle_gen::{CandleError, Result};
use tokenizers::Tokenizer;

/// The vendored T5-XXL tokenizer (google t5-v1.1-xxl) — byte-identical to Chroma's / mlx-gen-mochi's.
const T5_TOKENIZER_JSON: &[u8] = include_bytes!("../assets/t5_tokenizer.json");

/// Mochi's `max_sequence_length` (`_get_t5_prompt_embeds` default).
pub const MAX_SEQUENCE_LENGTH: usize = 256;

/// T5 `<pad>` token id.
pub const PAD_TOKEN_ID: u32 = 0;

/// Load the vendored T5 tokenizer with the JSON's baked-in padding **disabled** (so `encode` returns
/// natural-length ids, including the appended `</s>`; [`crate::text_encoder::tokenize`] pads to
/// [`MAX_SEQUENCE_LENGTH`] itself so it controls the attention mask).
pub fn load_tokenizer() -> Result<Tokenizer> {
    let mut tok = Tokenizer::from_bytes(T5_TOKENIZER_JSON)
        .map_err(|e| CandleError::Msg(format!("mochi: load vendored T5 tokenizer: {e}")))?;
    tok.with_padding(None);
    Ok(tok)
}
