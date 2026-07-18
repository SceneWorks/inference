//! Mochi's T5-XXL tokenizer. The snapshot's `tokenizer/` ships only the sentencepiece `spiece.model`
//! (no `tokenizer.json` the Rust core [`TextTokenizer`] can load), so — exactly like Chroma — we
//! vendor the prebuilt fast-tokenizer JSON (`assets/t5_tokenizer.json`) and load it from memory. It is
//! the **same** google t5-v1.1-xxl tokenizer FLUX/Chroma use, so the file is copied verbatim from the
//! Chroma crate (materialized by `tools/build_chroma_t5_tokenizer.py`).
//!
//! Mochi's `_get_t5_prompt_embeds` tokenizes with `padding="max_length"`, `max_length=256`,
//! `truncation=True`, `add_special_tokens=True` — [`MochiTokenizer`] reproduces that policy locally,
//! including reserving the final context slot for EOS on overflow. The EOS `</s>` (id 1) is appended
//! by the vendored tokenizer's post-processor.

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig, TokenizerOutput};
use mlx_gen::Result;

/// The vendored T5-XXL tokenizer (google t5-v1.1-xxl) — byte-identical to Chroma's, since it is the
/// same tokenizer Mochi conditions on.
const T5_TOKENIZER_JSON: &str = include_str!("../assets/t5_tokenizer.json");

/// Mochi's `max_sequence_length` (`_get_t5_prompt_embeds` default).
pub const MAX_SEQUENCE_LENGTH: usize = 256;

/// T5 `<pad>` token id.
pub const PAD_TOKEN_ID: i32 = 0;

/// T5 `</s>` token id.
pub const EOS_TOKEN_ID: i32 = 1;

/// Mochi-local tokenizer policy. It reserves the last context slot for EOS on overflow rather than
/// inheriting the shared tokenization helper's generic right-truncation semantics.
pub struct MochiTokenizer {
    inner: TextTokenizer,
    max_length: usize,
}

impl MochiTokenizer {
    pub fn tokenize(&self, prompt: &str) -> Result<TokenizerOutput> {
        let mut ids = self.inner.encode_ids(prompt, true)?;
        // The vendored JSON carries baked-in right padding; recover the natural special-token-framed
        // sequence before applying Mochi's 256-token truncation and padding policy.
        while ids.last() == Some(&PAD_TOKEN_ID) {
            ids.pop();
        }
        if ids.len() > self.max_length {
            ids.truncate(self.max_length - 1);
            ids.push(EOS_TOKEN_ID);
        }
        let mut mask = vec![1; ids.len()];
        ids.resize(self.max_length, PAD_TOKEN_ID);
        mask.resize(self.max_length, 0);
        Ok(TokenizerOutput { ids, mask })
    }
}

/// Load the vendored tokenizer at Mochi's production length ([`MAX_SEQUENCE_LENGTH`]).
pub fn load_tokenizer() -> Result<MochiTokenizer> {
    load_tokenizer_with_max_len(MAX_SEQUENCE_LENGTH)
}

/// Load the vendored tokenizer at a given padded length (production uses [`MAX_SEQUENCE_LENGTH`]; a
/// test may use a smaller length — the mask logic is length-agnostic). Mirrors
/// `_get_t5_prompt_embeds`: `padding="max_length"` (`pad_to_max_length`), `truncation` at
/// `max_length`, `add_special_tokens` (the post-processor appends `</s>`), pad token 0.
pub fn load_tokenizer_with_max_len(max_length: usize) -> Result<MochiTokenizer> {
    let config = TokenizerConfig {
        max_length,
        pad_token_id: PAD_TOKEN_ID,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    Ok(MochiTokenizer {
        inner: TextTokenizer::from_json_str(T5_TOKENIZER_JSON, config)?,
        max_length,
    })
}
