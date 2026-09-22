//! Qwen3-VL text conditioning for Qwen-Image 2.1 — the port of
//! `QwenImage21Pipeline._get_qwen_prompt_embeds` for the text-only (T2I) path.
//!
//! The prompt is rendered into the **raw** T2I template (not `apply_chat_template`; upstream notes
//! the two tokenize differently and the checkpoint expects this one):
//!
//! ```text
//! <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! tokenized with the snapshot's own `processor/tokenizer.json`, run through the Qwen3-VL **language
//! tower**, and the hidden state of the **last decoder layer before the final RMSNorm** is taken
//! (upstream neutralises `norm` with a forward hook because transformers ≥ 5.0 ties
//! `hidden_states[-1]` to the normalised `last_hidden_state`). The leading system-role tokens are
//! then dropped: upstream derives that count by tokenizing the system message through the chat
//! template, and [`system_prompt_drop_count`] derives it the same way from the loaded tokenizer (14
//! for the released tokenizer, pinned by the ignored real-weight test) instead of hard-coding it.
//!
//! For a text-only prompt the three mRoPE position streams (`mrope_section [24, 20, 20]`,
//! interleaved) are all the plain token index, so the rotary embedding is exactly 1-D RoPE at
//! `rope_theta = 5e6` — which is why the decoder block is **reused** from Z-Image's Qwen3 port
//! ([`mlx_gen_z_image::text_encoder::EncoderLayer`]: per-head `q_norm`/`k_norm`, bias-free GQA,
//! HF half-split RoPE, SwiGLU) rather than ported a third time. The deepstack visual injection and
//! the vision tower only engage with condition images (a later story); their weights
//! (`model.visual.*`) are simply not loaded here.
//!
//! Activations run f32 over the bf16 weight store (the sibling text encoders' policy); the DiT
//! rounds the result to its own compute dtype at `txt_in`.

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_z_image::text_encoder::EncoderLayer;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

use crate::config::{TextEncoderConfig, SYSTEM_PROMPT};

/// Group size for a pre-quantized (packed) Qwen3 tower — the codebase-wide default (64).
/// Single-sourced from [`crate::quant::GROUP_SIZE`]: the converter writes tiers at this group
/// size and the packed-detect loaders must read them back at the same one.
const GROUP_SIZE: i32 = crate::quant::GROUP_SIZE;

/// The system-role prefix of the T2I template — exactly what upstream tokenizes to derive
/// `_drop_idx`.
pub fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// The full raw T2I template for `prompt` (`QwenImage21Pipeline.prompt_template_t2i`). An empty
/// prompt is rendered as a single space, as upstream does ("Qwen has no bos token").
pub fn prompt_template(prompt: &str) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// How many leading template tokens to drop from the hidden states: the tokenized system-role
/// prefix, derived from the loaded tokenizer so it tracks the snapshot rather than a constant.
pub fn system_prompt_drop_count(tokenizer: &TextTokenizer) -> Result<usize> {
    let ids = tokenizer.encode_ids(&system_prefix(), true)?;
    if ids.is_empty() {
        return Err(Error::Msg(
            "qwen_image_2_1: the tokenizer encodes the system prefix to nothing".into(),
        ));
    }
    Ok(ids.len())
}

/// The Qwen3-VL language tower as Qwen-Image 2.1 conditions on it: token embedding → N pre-norm
/// decoder layers → the last layer's **un-normalised** hidden states.
pub struct QwenImage21TextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<EncoderLayer>,
    rope: TextRope,
    hidden_size: usize,
    /// The SwiGLU intermediate width — the second of the two `Linear` input widths
    /// [`Self::quantize`] has to be able to cover at [`GROUP_SIZE`].
    intermediate_size: usize,
}

impl QwenImage21TextEncoder {
    /// Build from a Qwen3-VL checkpoint. `prefix` is the language-tower prefix —
    /// `model.language_model` for the released `Qwen3VLForConditionalGeneration` layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &TextEncoderConfig) -> Result<Self> {
        let join = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        let embed_tokens = mlx_gen::quant::embedding(w, &join("embed_tokens"), GROUP_SIZE)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::from_weights(
                w,
                &join(&format!("layers.{i}")),
                cfg.num_attention_heads as i32,
                cfg.num_key_value_heads as i32,
                cfg.head_dim as i32,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rope: TextRope::new(cfg.head_dim as i32, cfg.rope_theta),
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// `true` iff both decoder `Linear` input widths are multiples of [`crate::quant::GROUP_SIZE`] — i.e. this
    /// geometry can be affine-quantized at the one group size a Qwen-Image 2.1 tier may declare
    /// ([`crate::quant`]). The released Qwen3 tower (4096 / 12288) can; the miniature parity
    /// snapshot (32 / 64) cannot.
    pub fn is_group_aligned(&self) -> bool {
        self.hidden_size.is_multiple_of(GROUP_SIZE as usize)
            && self.intermediate_size.is_multiple_of(GROUP_SIZE as usize)
    }

    /// Quantize every decoder `Linear` to Q4/Q8 at [`crate::quant::GROUP_SIZE`] (the embedding and the norms stay
    /// dense — see [`crate::quant`]'s per-component table), returning whether anything was packed.
    ///
    /// A geometry [`Self::is_group_aligned`] rejects is left **dense** rather than packed at a
    /// second group size: a tier declares exactly one `quantization.group_size`, and a component
    /// packed at another would be decoded at the wrong bit-width on reload. This mirrors
    /// [`crate::transformer::QwenImage21Transformer::quantize`]'s own width fallback, minus its
    /// group-32 arm, which a shippable tier cannot use.
    pub fn quantize(&mut self, bits: i32) -> Result<bool> {
        if !self.is_group_aligned() {
            return Ok(false);
        }
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(true)
    }

    /// `input_ids` `[1, L]` (i32) + `attention_mask` `[1, L]` → the last decoder layer's hidden
    /// states `[1, L, hidden]`, f32, **before** the final norm.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        self.forward_traced(input_ids, attention_mask, false)
            .map(|(hidden, _)| hidden)
    }

    /// [`Self::forward`] that also returns the embedding output and every layer's output
    /// (`embed`, `layer_{i}`) when `trace` is set — the localisation seam the parity tests read.
    pub fn forward_traced(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        trace: bool,
    ) -> Result<(Array, Vec<(String, Array)>)> {
        let shape = input_ids.shape();
        let (b, s) = (shape[0], shape[1]);
        let mut stages = Vec::new();
        let mut x = self
            .embed_tokens
            .forward(input_ids)?
            .as_dtype(Dtype::Float32)?;
        if trace {
            stages.push(("embed".to_string(), x.clone()));
        }
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cos, &sin, &mask)?;
            if trace {
                stages.push((format!("layer_{i}"), x.clone()));
            }
        }
        Ok((x, stages))
    }

    /// Prompt → conditioning `[1, L − drop, hidden]` (f32): render the template, tokenize, run the
    /// tower, drop the `drop` system-prefix tokens.
    pub fn encode_prompt(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        drop: usize,
    ) -> Result<Array> {
        let text = prompt_template(prompt);
        let tokens = tokenizer.tokenize_preformatted(&text)?;
        if tokens.ids.len() <= drop {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: prompt tokenized to {} tokens, not more than the {drop} template tokens to drop",
                tokens.ids.len()
            )));
        }
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tokens);
        let hidden = self.forward(&input_ids, &attention_mask)?;
        let total = hidden.shape()[1];
        Ok(hidden.index((.., drop as i32..total, ..)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_upstream_shape() {
        let t = prompt_template("a fox");
        assert_eq!(
            t,
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(prompt_template("").contains("user\n <|im_end|>"));
        assert!(t.starts_with(&system_prefix()));
    }
}
