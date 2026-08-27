//! PiD caption encoding — the host-side glue around the Gemma-2 decoder. Faithful port of
//! `pixeldit_model._encode_text_raw`: prepend the fixed **Chi-prompt**, tokenize (`add_special_tokens`
//! → leading `<bos>`) and right-pad/truncate to `num_chi_tokens + model_max_length − 2`, run the Gemma
//! decoder (with the padding mask), then gather `select_index = [0] + range(-(model_max_length−1), 0)`
//! → `caption_embs [1, model_max_length, 2304]`.
//!
//! Note: the `y_norm`/`y_norm_scale_factor` config knob is **never applied** in the reference code
//! (dead config), so we do not scale. PiD's inference net runs **without** a caption mask (the
//! `emb_masks` are discarded), while the shared encoder also exposes the gathered mask for SANA's
//! diffusers-compatible cross-attention path.

use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

use crate::gemma2::Gemma2;

/// PiD's fixed "Chi-prompt" instruction prefix (`experiment/shared_config.py::_CHI_PROMPT`, joined by
/// `\n`). The user caption is appended directly after the trailing `"User Prompt: "`.
///
/// This is the same Complex-Human-Instruction (CHI) template SANA uses; they differ **only** in the
/// quoting around `Enhanced prompt`, which changes the tokenization — so the CHI prompt is
/// parameterized rather than hardcoded (see [`CaptionEncoder::with_chi_prompt`]).
pub const CHI_PROMPT: &str = "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt: ";

const MODEL_MAX_LENGTH: i32 = 300;
const PAD_ID: i32 = 0;

/// The released token-selection policy shared by PiD and SANA: gather `select_index = [0] +
/// range(max_len − (MODEL_MAX_LENGTH − 1), max_len)` — the `<bos>` at position 0 plus the trailing
/// `MODEL_MAX_LENGTH − 1` tokens — from the `max_len`-long Gemma **last-hidden** sequence, yielding
/// exactly `MODEL_MAX_LENGTH` (300) caption tokens.
///
/// Exposed (and unit-tested) so the index math can be confirmed against the SANA reference
/// (`select_index = [0] + range(-(max_sequence_length − 1), 0)` in diffusers
/// `SanaPipeline._get_gemma_prompt_embeds`) without the Gemma weights. `max_len` must be
/// `≥ MODEL_MAX_LENGTH` (the caption tokenizer always right-pads to `num_chi_tokens + 300 − 2`, which
/// is well above 300).
pub fn select_index(max_len: usize) -> Vec<u32> {
    debug_assert!(max_len >= MODEL_MAX_LENGTH as usize);
    let mut sel = Vec::with_capacity(MODEL_MAX_LENGTH as usize);
    sel.push(0u32);
    sel.extend(((max_len as i32 - (MODEL_MAX_LENGTH - 1))..max_len as i32).map(|i| i as u32));
    sel
}

/// Gather a pre-selection padding mask with the exact same released indices as the caption hidden
/// states. The returned values are f32 so a SANA consumer can apply them directly as an additive
/// cross-attention bias (`1.0` real token, `0.0` padding).
fn gather_selected_mask(mask: &[i32]) -> Result<(Vec<u32>, Vec<f32>)> {
    if mask.len() < MODEL_MAX_LENGTH as usize {
        return Err(CandleError::Msg(format!(
            "caption attention mask has {} positions, expected at least {MODEL_MAX_LENGTH}",
            mask.len()
        )));
    }
    let indices = select_index(mask.len());
    let selected = indices
        .iter()
        .map(|&index| mask[index as usize] as f32)
        .collect();
    Ok((indices, selected))
}

/// Gemma-2 caption encoder: tokenizer + CHI-prompt + the released token-selection policy.
pub struct CaptionEncoder {
    gemma: Gemma2,
    tok: TextTokenizer,
    chi_prompt: String,
    num_chi_tokens: i32,
}

impl CaptionEncoder {
    /// Build the PiD caption encoder (uses PiD's [`CHI_PROMPT`]) from a constructed [`Gemma2`] and the
    /// gemma `tokenizer.json` path.
    pub fn new(gemma: Gemma2, tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        Self::with_chi_prompt(gemma, tokenizer_json, CHI_PROMPT)
    }

    /// Build the caption encoder with an explicit CHI-prompt prefix — the reuse seam for SANA, which
    /// shares PiD's entire encoder body but ships a CHI template that differs in quoting.
    pub fn with_chi_prompt(
        gemma: Gemma2,
        tokenizer_json: impl AsRef<Path>,
        chi_prompt: impl Into<String>,
    ) -> Result<Self> {
        let chi_prompt = chi_prompt.into();
        let tok = TextTokenizer::from_file(
            tokenizer_json,
            TokenizerConfig {
                max_length: 4096,
                pad_token_id: PAD_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        // num_chi_tokens counts the CHI-prompt WITH its special tokens (the reference's
        // `tokenizer.encode(chi_prompt_str)` adds the bos).
        let num_chi_tokens = tok.encode_ids(&chi_prompt, true)?.len() as i32;
        Ok(Self {
            gemma,
            tok,
            chi_prompt,
            num_chi_tokens,
        })
    }

    /// Chi-prompt token count (the reference's `_num_chi_tokens`).
    pub fn num_chi_tokens(&self) -> i32 {
        self.num_chi_tokens
    }

    /// The padded `[input_ids, attention_mask]` for a caption — exposed so the tokenizer + Chi-prompt
    /// + length policy can be parity-checked against the reference without the Gemma weights.
    pub fn token_ids(&self, caption: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let max_len = self.num_chi_tokens + MODEL_MAX_LENGTH - 2;
        // The caption is fed RAW (no lowercasing) because PiD was TRAINED on raw-case captions, so raw
        // is what the checkpoint expects (sc-9935). Evidence, not just parity: PiD's training-time
        // caption conditioner (`modules/conditioner.py`) passes captions through unmodified; its
        // reference inference (`_encode_text_raw`) appends `chi_prompt_str + cap` unmodified; the
        // authors' own demo manifest uses mixed-case prompts; and there is no `.lower()` on captions
        // anywhere in the PiD repo. This deliberately differs from SANA, whose reference is the
        // diffusers `SanaPipeline` (`_text_preprocessing` lowercases) — so the Candle and MLX SANA
        // wrappers each apply matching `preprocess()` before this shared encoder (sc-9927).
        // Lowercasing here would feed PiD OOD-cased captions. (The effect is also small — an A/B
        // moved a PiD SR decode 0.034%, weak LQ-dominated conditioning — but the reason to keep raw
        // is correctness, not impact.)
        let mut ids = self
            .tok
            .encode_ids(&format!("{}{caption}", self.chi_prompt), true)?;
        ids.truncate(max_len as usize);
        let real = ids.len();
        ids.resize(max_len as usize, PAD_ID);
        let mask = (0..max_len as usize).map(|i| (i < real) as i32).collect();
        Ok((ids, mask))
    }

    /// Encode one caption to `[1, 300, 2304]` caption embeddings (f32). PiD deliberately discards the
    /// gathered padding mask; SANA uses [`Self::encode_with_mask`] instead.
    pub fn encode(&self, caption: &str) -> Result<Tensor> {
        Ok(self.encode_with_mask(caption)?.0)
    }

    /// Encode one caption to `([1, 300, 2304]` embeddings, `[1, 300]` padding mask`)`. Both tensors
    /// use the same released `select_index = [0] + range(-(300 - 1), 0)`, keeping real/padding slots
    /// aligned for SANA's transformer cross-attention.
    pub fn encode_with_mask(&self, caption: &str) -> Result<(Tensor, Tensor)> {
        let (ids, mask) = self.token_ids(caption)?;
        let max_len = ids.len();
        let device = self.gemma.device();

        let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let ids_arr = Tensor::from_vec(ids_u32, (1, max_len), device)?;
        let mask_f32: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
        let mask_arr = Tensor::from_vec(mask_f32, (1, max_len), device)?;
        // Gemma-2 run in encoder / feature-extraction mode: `forward` returns the decoder's
        // LAST-HIDDEN states `[1, max_len, 2304]` (post final `model.norm`), NOT generation logits —
        // the diffusers reference does `prompt_embeds[0]` (== `last_hidden_state`), never the LM head.
        let hidden = self.gemma.forward(&ids_arr, Some(&mask_arr))?; // [1, max_len, 2304]

        // select_index = [0] + range(max_len-(300-1), max_len), shared by hidden + mask.
        let (sel, selected_mask) = gather_selected_mask(&mask)?;
        let sel_arr = Tensor::from_vec(sel, (MODEL_MAX_LENGTH as usize,), device)?;
        let selected_hidden = hidden.index_select(&sel_arr, 1)?;
        let selected_mask =
            Tensor::from_vec(selected_mask, (1, MODEL_MAX_LENGTH as usize), device)?;
        Ok((selected_hidden, selected_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::{gather_selected_mask, select_index, MODEL_MAX_LENGTH};

    #[test]
    fn select_index_is_bos_plus_trailing_299() {
        // `[0] + range(max_len - 299, max_len)` for a representative padded length.
        let max_len = 555usize; // e.g. num_chi_tokens(257) + 300 - 2
        let sel = select_index(max_len);

        // Exactly MODEL_MAX_LENGTH (300) indices.
        assert_eq!(sel.len(), MODEL_MAX_LENGTH as usize);
        // The <bos> token at position 0 is preserved as the first selected index.
        assert_eq!(sel[0], 0);
        // Then the trailing MODEL_MAX_LENGTH - 1 positions, contiguous, ending at the last token.
        assert_eq!(sel[1], (max_len as u32) - (MODEL_MAX_LENGTH as u32 - 1));
        assert_eq!(*sel.last().unwrap(), max_len as u32 - 1);
        // The tail is a contiguous ascending run (no gaps).
        for w in sel[1..].windows(2) {
            assert_eq!(w[1], w[0] + 1);
        }

        // Byte-identical to the explicit reference construction `[0] + list(range(...))`.
        let mut expect = vec![0u32];
        expect.extend((max_len as u32 - (MODEL_MAX_LENGTH as u32 - 1))..max_len as u32);
        assert_eq!(sel, expect);
    }

    #[test]
    fn gathered_mask_uses_the_embedding_select_index() {
        // With 305 positions the released selection is [0, 6, 7, ..., 304]. Position 5 is marked
        // real but deliberately skipped; a prefix/suffix slice or off-by-one gather changes this row.
        let mut mask = vec![0; 305];
        mask[0] = 1;
        mask[5] = 1;
        mask[6] = 1;
        mask[304] = 1;

        let (indices, gathered) = gather_selected_mask(&mask).unwrap();
        assert_eq!(indices.len(), MODEL_MAX_LENGTH as usize);
        assert_eq!(&indices[..3], &[0, 6, 7]);
        assert_eq!(*indices.last().unwrap(), 304);
        assert_eq!(gathered.len(), MODEL_MAX_LENGTH as usize);
        assert_eq!(gathered.iter().filter(|&&value| value == 1.0).count(), 3);
        assert_eq!(gathered[0], 1.0);
        assert_eq!(gathered[1], 1.0);
        assert_eq!(*gathered.last().unwrap(), 1.0);

        assert!(gather_selected_mask(&vec![1; 299]).is_err());
    }
}
