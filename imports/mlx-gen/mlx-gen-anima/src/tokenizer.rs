//! Anima tokenizes each prompt **twice**: the Qwen2 BPE tokenizer feeds the Qwen3 text encoder
//! (`source_hidden_states`), and the T5 SentencePiece tokenizer's token ids are the conditioner's
//! learned query tokens (`target_input_ids`). Both are vendored `tokenizer.json` fast-tokenizers
//! (`assets/`) loaded through the shared `mlx_gen::tokenizer::TextTokenizer`.
//!
//! **Qwen2 settings are load-bearing** (Anima reference `Qwen2Tokenizer(padding="longest")`): **no
//! BOS, no EOS**, pad token **151643**. We encode with `add_special_tokens=false` and, batch-1, build
//! an all-ones mask (padding="longest" over a single prompt is the sequence itself). The **T5**
//! tokenizer keeps its EOS (`add_special_tokens=true`), matching `T5TokenizerFast`'s default.

use mlx_rs::Array;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::Result;

use crate::config::QWEN_PAD_TOKEN_ID;

/// Anima's `max_sequence_length` (reference default 512; also the DiT's fixed text length).
const MAX_LEN: usize = 512;

/// The vendored Qwen3/Qwen2 BPE tokenizer (Qwen3 lineage; text BPE identical across Qwen2.5/3).
const QWEN_TOKENIZER_JSON: &str = include_str!("../assets/qwen_tokenizer.json");
/// The vendored T5 SentencePiece tokenizer (google-t5, vocab 32128; shared with mlx-gen-chroma).
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
        )?;
        let t5 = TextTokenizer::from_json_str(
            T5_TOKENIZER_JSON,
            TokenizerConfig {
                max_length: MAX_LEN,
                pad_token_id: 0,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        Ok(Self { qwen, t5 })
    }

    /// Qwen2 BPE (no BOS/EOS), truncated to 512, batch-1. Returns `(ids [1,S], mask [1,S])`. An empty
    /// prompt yields the reference fallback `id=0, mask=0` (a single zero token the mask zeroes out).
    pub fn encode_qwen(&self, prompt: &str) -> Result<(Array, Array)> {
        let mut ids = self.qwen.encode_ids(prompt, false)?;
        ids.truncate(MAX_LEN);
        if ids.is_empty() {
            return Ok((
                Array::from_slice(&[0i32], &[1, 1]),
                Array::from_slice(&[0i32], &[1, 1]),
            ));
        }
        let n = ids.len() as i32;
        let mask = vec![1i32; ids.len()];
        Ok((
            Array::from_slice(&ids, &[1, n]),
            Array::from_slice(&mask, &[1, n]),
        ))
    }

    /// T5 SentencePiece (with EOS), batch-1, padding="longest" (i.e. the real sequence only). Returns
    /// `ids [1,S]`. An empty prompt yields the single EOS token (`</s>`), matching `T5TokenizerFast("")`.
    ///
    /// The vendored `t5_tokenizer.json` carries a **fixed 512-length padding** config, so `encode`
    /// right-pads the ids with the T5 pad token (`0`). Anima's reference uses `padding="longest"` on a
    /// single prompt — i.e. the real tokens only — and the **conditioner** later right-pads its *output*
    /// to 512 with zeros. Feeding the raw 512 padded ids instead would make the conditioner process 507
    /// pad-token query rows into 507 non-zero outputs (corrupting/diluting the DiT cross-attention
    /// conditioning), so we strip the trailing pad tokens back to the real sequence.
    pub fn encode_t5(&self, prompt: &str) -> Result<Array> {
        let mut ids = self.t5.encode_ids(prompt, true)?;
        // Drop the tokenizer's built-in right-padding (T5 pad token id 0); keep >= 1 token.
        while ids.len() > 1 && ids.last() == Some(&0) {
            ids.pop();
        }
        ids.truncate(MAX_LEN);
        if ids.is_empty() {
            // T5 EOS is id 1 (</s>); a truly empty encode shouldn't happen with add_special_tokens.
            ids.push(1);
        }
        let n = ids.len() as i32;
        Ok(Array::from_slice(&ids, &[1, n]))
    }
}
