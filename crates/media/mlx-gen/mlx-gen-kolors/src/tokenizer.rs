//! Kolors prompt tokenization (sc-3092) — the ChatGLM3 tokenizer the diffusers `KolorsPipeline`
//! drives, reproduced so the [`chatglm3`](crate::chatglm3) encoder receives byte-identical
//! `input_ids` / `attention_mask` / `position_ids`.
//!
//! ChatGLM3 ships only a **slow** SentencePiece tokenizer (LLaMA-style byte_fallback BPE). The fast
//! `tokenizer.json` is materialized once into the snapshot's `tokenizer/` dir by
//! `tools/build_kolors_tokenizer.py` (a faithful `LlamaConverter` replica); this wrapper loads it via
//! core [`TextTokenizer`] for the SP **content** ids and applies the ChatGLM-specific framing:
//!
//!  - **Prefix tokens** `[gMASK]` (64790) + `sop` (64792) prepended (`build_inputs_with_special_tokens`).
//!  - **Truncation** of the content to `max_length - 2` (reserving the 2 prefix tokens), as HF's
//!    `truncation=True` does (it accounts for the special tokens).
//!  - **Left padding** to `max_length` (256) with pad = unk = 0 (`padding_side="left"`), producing the
//!    matching `attention_mask` (`[0]*pad + [1]*len`) and `position_ids` (`[0]*pad + 0..len`) — the
//!    left-pad restarts real-token positions at 0, and Kolors passes these `position_ids` to the
//!    encoder's RoPE (so the encoder must consume them, not a plain arange).

use std::path::Path;

use mlx_rs::Array;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::{Error, Result};

/// `[gMASK]` prefix token id (appended after the 64789-piece SP vocab).
pub const GMASK_ID: i32 = 64790;
/// `sop` (start-of-prompt) prefix token id.
pub const SOP_ID: i32 = 64792;
/// Pad token id = SentencePiece `unk_id` (0), left-padded by the ChatGLM tokenizer.
pub const PAD_ID: i32 = 0;
/// Kolors' fixed prompt length (`max_sequence_length`).
pub const MAX_LEN: usize = 256;

const PREFIX: [i32; 2] = [GMASK_ID, SOP_ID];

/// One tokenized prompt: `(1, L)` int32 ids + attention mask + position ids, all left-padded to the
/// configured length. `position_ids` is ChatGLM-specific (Kolors threads it into the encoder RoPE).
pub struct KolorsTokens {
    pub input_ids: Array,
    pub attention_mask: Array,
    pub position_ids: Array,
}

/// The Kolors (ChatGLM3) tokenizer.
pub struct KolorsTokenizer {
    inner: TextTokenizer,
    max_len: usize,
}

impl KolorsTokenizer {
    /// Load from a snapshot `tokenizer/` dir containing the materialized `tokenizer.json` (see
    /// `tools/build_kolors_tokenizer.py`). Uses the default [`MAX_LEN`] (256).
    pub fn from_dir(tokenizer_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_file(tokenizer_dir.as_ref().join("tokenizer.json"), MAX_LEN)
    }

    /// Load from an explicit `tokenizer.json` path with a chosen max length.
    pub fn from_file(tokenizer_json: impl AsRef<Path>, max_len: usize) -> Result<Self> {
        // `encode` always prepends the `PREFIX` (GMASK/SOP) and left-pads to `max_len`, so `max_len`
        // must leave room for the prefix plus at least one content token. Below that, `encode`'s
        // `pad = max_len - len` underflows (panic in debug, huge-alloc abort in release). Reject at
        // this public boundary rather than at use (F-142); `from_dir` uses the fixed `MAX_LEN`.
        if max_len < PREFIX.len() + 1 {
            return Err(Error::Msg(format!(
                "kolors tokenizer: max_len must be >= {} (the {}-token GMASK/SOP prefix plus at \
                 least one content token); got {max_len}",
                PREFIX.len() + 1,
                PREFIX.len()
            )));
        }
        // ChatTemplate::None: Kolors tokenizes the raw prompt (no chat wrapping); the SP content path
        // adds no special tokens (prefix/pad are applied here). pad_to_max_length stays false — this
        // wrapper owns the (left-)padding, not core's right-pad.
        let cfg = TokenizerConfig {
            max_length: max_len,
            pad_token_id: PAD_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        };
        Ok(Self {
            inner: TextTokenizer::from_file(tokenizer_json, cfg)?,
            max_len,
        })
    }

    /// Tokenize one prompt → left-padded `(1, max_len)` `input_ids` / `attention_mask` /
    /// `position_ids`, byte-identical to `ChatGLMTokenizer(prompt, padding="max_length",
    /// max_length=max_len, truncation=True)`.
    pub fn encode(&self, prompt: &str) -> Result<KolorsTokens> {
        // SP content ids (no special tokens — the tokenizer.json has no post-processor).
        let mut content = self.inner.encode_ids(prompt, false)?;
        // truncation=True reserves the prefix tokens (HF accounts for num_special_tokens_to_add).
        let keep = self.max_len.saturating_sub(PREFIX.len());
        if content.len() > keep {
            content.truncate(keep);
        }

        let mut ids: Vec<i32> = Vec::with_capacity(PREFIX.len() + content.len());
        ids.extend_from_slice(&PREFIX);
        ids.extend_from_slice(&content);
        let len = ids.len();
        // `len <= max_len`: content is truncated to `max_len - PREFIX.len()` and `from_file`
        // guarantees `max_len >= PREFIX.len() + 1`, so this never underflows (F-142).
        let pad = self.max_len - len;

        let mut input_ids = vec![PAD_ID; pad];
        input_ids.extend_from_slice(&ids);
        let mut attention_mask = vec![0i32; pad];
        attention_mask.resize(self.max_len, 1); // positions pad..max_len = valid (len of them)
        let mut position_ids = vec![0i32; pad];
        position_ids.extend(0..len as i32);

        let shape = [1, self.max_len as i32];
        Ok(KolorsTokens {
            input_ids: Array::from_slice(&input_ids, &shape),
            attention_mask: Array::from_slice(&attention_mask, &shape),
            position_ids: Array::from_slice(&position_ids, &shape),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_file_rejects_max_len_below_prefix_plus_one() {
        // F-142: `max_len < PREFIX.len() + 1` would underflow `encode`'s `pad = max_len - len`. The
        // guard fires before the file is even opened, so a bogus path still surfaces the max_len
        // error (not an IO error) for the rejected values.
        let err_string = |max_len: usize| match KolorsTokenizer::from_file(
            "/nonexistent/tokenizer.json",
            max_len,
        ) {
            Ok(_) => panic!("max_len={max_len} must not load from a bogus path"),
            Err(e) => e.to_string(),
        };
        for bad in [0usize, 1, 2] {
            assert!(
                err_string(bad).contains("max_len"),
                "max_len={bad} not rejected"
            );
        }
        // `max_len == PREFIX.len() + 1` clears the guard (then fails to load the missing file — a
        // different error), confirming the boundary is exactly `PREFIX.len() + 1`.
        assert!(
            !err_string(PREFIX.len() + 1).contains("max_len must be"),
            "max_len = PREFIX.len()+1 should pass the guard"
        );
    }
}
