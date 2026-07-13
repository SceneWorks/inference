//! Anima tokenizes each prompt **twice**: the Qwen2 BPE tokenizer feeds the Qwen3 text encoder
//! (`source_hidden_states`), and the T5 SentencePiece tokenizer's token ids are the conditioner's
//! learned query tokens (`target_input_ids`). Both are the **same** vendored `tokenizer.json`
//! fast-tokenizers as `mlx-gen-anima` (`assets/`), loaded through the backend-neutral
//! `gen_core::tokenizer::TextTokenizer`, so tokenization is **byte-identical** to the MLX lane.
//!
//! **Qwen2 settings are load-bearing** (`Qwen2Tokenizer(padding="longest")`): **no BOS, no EOS**, pad
//! token **151643**. We encode with `add_special_tokens=false` and, batch-1, build an all-ones mask.
//! The **T5** tokenizer keeps its EOS (`add_special_tokens=true`).

use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

use crate::config::QWEN_PAD_TOKEN_ID;

/// Anima's `max_sequence_length` (reference default 512; also the DiT's fixed text length).
const MAX_LEN: usize = 512;

/// The vendored Qwen3/Qwen2 BPE tokenizer (identical asset to the MLX provider).
const QWEN_TOKENIZER_JSON: &str = include_str!("../assets/qwen_tokenizer.json");
/// The vendored T5 SentencePiece tokenizer (google-t5, vocab 32128; shared with mlx/candle chroma).
const T5_TOKENIZER_JSON: &str = include_str!("../assets/t5_tokenizer.json");

/// Both tokenizers for the Anima prompt pipeline.
pub struct AnimaTokenizers {
    qwen: TextTokenizer,
    t5: TextTokenizer,
}

impl AnimaTokenizers {
    /// Load both vendored tokenizers.
    pub fn load() -> Result<Self> {
        let qwen = TextTokenizer::from_json_str(
            QWEN_TOKENIZER_JSON,
            TokenizerConfig {
                max_length: MAX_LEN,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("anima: load qwen tokenizer: {e}")))?;
        let t5 = TextTokenizer::from_json_str(
            T5_TOKENIZER_JSON,
            TokenizerConfig {
                max_length: MAX_LEN,
                pad_token_id: 0,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("anima: load t5 tokenizer: {e}")))?;
        Ok(Self { qwen, t5 })
    }

    /// Qwen2 BPE (no BOS/EOS), truncated to 512, batch-1. Returns `(ids, mask)`. An empty prompt yields
    /// the reference fallback `id=0, mask=0` (a single zero token the mask zeroes out).
    pub fn encode_qwen(&self, prompt: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut ids = self
            .qwen
            .encode_ids(prompt, false)
            .map_err(|e| CandleError::Msg(format!("anima: qwen encode: {e}")))?;
        ids.truncate(MAX_LEN);
        if ids.is_empty() {
            return Ok((vec![0], vec![0]));
        }
        let mask = vec![1i32; ids.len()];
        Ok((ids, mask))
    }

    /// T5 SentencePiece (with EOS), batch-1, padding="longest" (the real sequence only). Returns the
    /// ids. The vendored `t5_tokenizer.json` carries a fixed 512-length padding config, so `encode`
    /// right-pads with the T5 pad token (`0`); Anima's reference uses `padding="longest"` and the
    /// **conditioner** later right-pads its *output* to 512 — so we strip the trailing pad tokens back
    /// to the real sequence (feeding 507 pad-token query rows would corrupt the DiT cross-attention).
    pub fn encode_t5(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut ids = self
            .t5
            .encode_ids(prompt, true)
            .map_err(|e| CandleError::Msg(format!("anima: t5 encode: {e}")))?;
        // Drop the tokenizer's built-in right-padding (T5 pad token id 0); keep >= 1 token.
        while ids.len() > 1 && ids.last() == Some(&0) {
            ids.pop();
        }
        ids.truncate(MAX_LEN);
        if ids.is_empty() {
            // T5 EOS is id 1 (</s>); a truly empty encode shouldn't happen with add_special_tokens.
            ids.push(1);
        }
        Ok(ids)
    }
}
