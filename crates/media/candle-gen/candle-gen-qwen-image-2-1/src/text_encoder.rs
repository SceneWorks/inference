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
//! template, and [`system_prompt_drop_count`] derives it the same way from the loaded tokenizer
//! instead of hard-coding it.
//!
//! For a text-only prompt the three mRoPE position streams (`mrope_section [24, 20, 20]`,
//! interleaved) are all the plain token index, so the rotary embedding is exactly 1-D RoPE at
//! `rope_theta = 5e6`. The MLX twin reuses `mlx_gen_z_image::text_encoder::EncoderLayer` for this;
//! candle's Z-Image Qwen3 decoder is a private vendored module (`candle_gen_z_image::packed_te`,
//! and its block is additionally pinned to Z-Image's layer[-2]/`model.` conventions), so the same
//! pre-norm decoder block is ported here rather than reached across a crate boundary — the forward
//! math is the stock Qwen3 one (per-head `q_norm`/`k_norm`, bias-free GQA, HF half-split RoPE,
//! SwiGLU). The deepstack visual injection and the vision tower only engage with condition images
//! (a later story); their weights (`model.visual.*`) are simply not loaded here.
//!
//! Activations run f32 (the sibling text encoders' policy); the DiT rounds the result to its own
//! compute dtype at `txt_in`.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_gen::candle_nn::{
    ops::softmax_last_dim, rms_norm, rotary_emb, Embedding, Linear, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::{CandleError as Error, Result};

use crate::config::{TextEncoderConfig, SYSTEM_PROMPT};

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

/// Precomputed 1-D RoPE table (`θ = rope_theta`), half-split as HF/Qwen3 applies it.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f32, max_len: usize, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let half = inv.len();
        let inv = Tensor::from_vec(inv, (1, half), device)?;
        let t = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_len, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Apply to `q`/`k` shaped `[B, H, L, D]`.
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let len = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, len)?;
        let sin = self.sin.narrow(0, 0, len)?;
        Ok((
            rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv_heads, n_rep, len, head_dim))?
        .reshape((b, kv_heads * n_rep, len, head_dim))
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    rotary: Arc<Rotary>,
}

impl Attention {
    fn new(cfg: &TextEncoderConfig, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        let (heads, kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        // Qwen3-VL text attention is bias-free (the config reader refuses `attention_bias: true`).
        let lin = |out: usize, name: &str| -> Result<Linear> {
            Ok(candle_gen::candle_nn::linear_no_bias(
                cfg.hidden_size,
                out,
                vb.pp(name),
            )?)
        };
        Ok(Self {
            q_proj: lin(heads * head_dim, "q_proj")?,
            k_proj: lin(kv_heads * head_dim, "k_proj")?,
            v_proj: lin(kv_heads * head_dim, "v_proj")?,
            o_proj: candle_gen::candle_nn::linear_no_bias(
                heads * head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps as f64, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps as f64, vb.pp("k_norm"))?,
            heads,
            kv_heads,
            kv_groups: heads / kv_heads.max(1),
            head_dim,
            rotary,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, len, _) = x.dims3()?;
        let shape = |t: Tensor, heads: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, len, heads, self.head_dim))?.transpose(1, 2)
        };
        let q = shape(self.q_proj.forward(x)?, self.heads)?;
        let k = shape(self.k_proj.forward(x)?, self.kv_heads)?;
        let v = shape(self.v_proj.forward(x)?, self.kv_heads)?;
        // Per-head RMSNorm (Qwen3).
        let q =
            self.q_norm
                .forward(&q.flatten(0, 2)?)?
                .reshape((b, self.heads, len, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.kv_heads,
            len,
            self.head_dim,
        ))?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let k = repeat_kv(k, self.kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Plain scores here, NOT the chunked `candle_gen::sdpa_budgeted_*`, and that is a bound
        // rather than an oversight: the F-003 guard exists because candle's CUDA kernels index
        // scores with i32, and this tower's sequence is capped by [`crate::loader::MAX_PROMPT_TOKENS`]
        // (4096). Its widest scores tensor is therefore `32 heads · 4096 · 4096 ≈ 5.4e8` elements —
        // a quarter of `i32::MAX`, and under `ATTN_SCORES_BUDGET`, so the planner would return the
        // whole query axis and this exact single pass anyway. The DiT's joint sequence is the one
        // that overflows; see `transformer::block_causal_attention`.
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let ctx = softmax_last_dim(&scores)?.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, len, self.heads * self.head_dim))?;
        Ok(self.o_proj.forward(&ctx)?)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: candle_gen::candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: candle_gen::candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: candle_gen::candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    /// SwiGLU: `down(silu(gate(x)) · up(x))`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate_proj.forward(x)?.silu()? * self.up_proj.forward(x)?)?;
        Ok(self.down_proj.forward(&gated)?)
    }
}

/// One pre-norm Qwen3 decoder block.
struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextEncoderConfig, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps as f64,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps as f64,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.input_layernorm.forward(x)?, mask)?;
        let x = (x + h)?;
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        Ok((&x + h)?)
    }
}

/// The Qwen3-VL language tower as Qwen-Image 2.1 conditions on it: token embedding → N pre-norm
/// decoder layers → the last layer's **un-normalised** hidden states.
pub struct QwenImage21TextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    hidden_size: usize,
    device: Device,
}

impl QwenImage21TextEncoder {
    /// Build from a Qwen3-VL checkpoint. `prefix` is the language-tower prefix —
    /// `model.language_model` for the released `Qwen3VLForConditionalGeneration` layout.
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder, prefix: &str) -> Result<Self> {
        let tower = prefix
            .split('.')
            .filter(|part| !part.is_empty())
            .fold(vb.clone(), |vb, part| vb.pp(part));
        let embed_tokens = candle_gen::candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            tower.pp("embed_tokens"),
        )?;
        let rotary = Arc::new(Rotary::new(
            cfg.head_dim,
            cfg.rope_theta,
            crate::loader::MAX_PROMPT_TOKENS,
            vb.device(),
        )?);
        let vb_layers = tower.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }
        // The final norm (`…norm.weight`) is deliberately NOT loaded — the conditioning is the last
        // layer's output before it.
        Ok(Self {
            embed_tokens,
            layers,
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The additive causal mask for a `len`-token prompt (prompts are never padded, so the
    /// attention mask is all-ones and the block is purely causal).
    fn causal_mask(&self, len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..len)
            .flat_map(|i| (0..len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Ok(Tensor::from_vec(mask, (1, 1, len, len), &self.device)?)
    }

    /// `input_ids` `[1, L]` (u32) → the last decoder layer's hidden states `[1, L, hidden]`, f32,
    /// **before** the final norm.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_traced(input_ids, false).map(|(h, _)| h)
    }

    /// [`Self::forward`] that also returns the embedding output and every layer's output
    /// (`embed`, `layer_{i}`) when `trace` is set — the localisation seam the parity tests read.
    pub fn forward_traced(
        &self,
        input_ids: &Tensor,
        trace: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let (_, len) = input_ids.dims2()?;
        let mut stages = Vec::new();
        let mut x = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        if trace {
            stages.push(("embed".to_string(), x.clone()));
        }
        let mask = self.causal_mask(len)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mask)?;
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
    ) -> Result<Tensor> {
        let tokens = tokenizer.tokenize_preformatted(&prompt_template(prompt))?;
        if tokens.ids.len() <= drop {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: prompt tokenized to {} tokens, not more than the {drop} template tokens to drop",
                tokens.ids.len()
            )));
        }
        let ids = input_ids(&tokens.ids, &self.device)?;
        let hidden = self.forward(&ids)?;
        let total = hidden.dim(1)?;
        Ok(hidden.i((.., drop..total, ..))?.contiguous()?)
    }
}

/// `[1, L]` u32 input ids on `device` from gen-core's host-side token vector.
pub fn input_ids(ids: &[i32], device: &Device) -> Result<Tensor> {
    let host: Vec<u32> = ids.iter().map(|&id| id.max(0) as u32).collect();
    let len = host.len();
    Ok(Tensor::from_vec(host, (1, len), device)?)
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
