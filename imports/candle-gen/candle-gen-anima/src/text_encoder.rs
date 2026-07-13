//! Anima's source text encoder — **Qwen3-0.6B base** (`Qwen3Model`), the candle transcription of
//! `mlx-gen-anima`'s `text_encoder.rs`. Anima consumes the model's **`last_hidden_state`** (the last
//! decoder layer AFTER the final `norm`), which is then mask-multiplied (in the pipeline) and fed to
//! the `AnimaTextConditioner` as `source_hidden_states`.
//!
//! ## GQA 16/8 (the candle-gen-z-image wart the story flags)
//! Qwen3-0.6B is **GQA 16/8**: 16 query heads, 8 KV heads. The candle-gen-z-image *DiT* attention
//! (`dit.rs`) sizes its K/V by `n_kv_heads` but reshapes by `n_heads` — an MHA-only shortcut that
//! `debug_assert`s `n_kv_heads == n_heads` and would collapse a true GQA config. This encoder instead
//! reshapes K/V by `n_kv_heads` and **repeat-expands** them by `kv_groups` (`n_heads / n_kv_heads = 2`)
//! before the attention (query head `i` reads KV head `i / groups`), the same
//! `repeat_kv`/`repeat_interleave` the diffusers Qwen3 + `mlx_gen::nn::repeat_kv` use.
//!
//! ## Masking
//! Qwen3 is a causal decoder LM; the encoder runs a **causal** mask. Anima is batch-1 with
//! `padding="longest"` (no pad tokens in a real prompt), so the reference's causal+padding mask equals
//! a pure causal mask here; the padding of the single empty-uncond token is applied by the pipeline's
//! post-encoder mask-multiply (which also avoids the fully-masked-row NaN a padding-in-the-softmax
//! would produce). For the golden's 18-real-token prompt (mask all ones) this is numerically identical.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_gen::Result;

use crate::config::Qwen3Config;
use crate::nn::{apply_rope_half, lin, rms_norm, sdpa};
use crate::rope::text_rope;

/// Additive causal-mask fill (a large finite negative — avoids `-inf` propagation through the softmax
/// kernel; every causal row keeps its on-diagonal 0, so no row is fully masked).
const MASK_NEG: f32 = -1e30;

/// Expand `[b, hkv, s, d]` → `[b, hkv·groups, s, d]` (GQA): repeat each KV head `groups` times
/// consecutively so query head `i` reads KV head `i / groups`.
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, hkv, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, hkv, groups, s, d))?
        .reshape((b, hkv * groups, s, d))?)
}

/// One Qwen3 decoder layer: pre-RMSNorm GQA self-attn + pre-RMSNorm SwiGLU MLP (both residual).
struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    groups: usize,
    scale: f64,
    eps: f64,
}

impl DecoderLayer {
    fn new(vb: &VarBuilder, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        Ok(Self {
            input_ln: vb.get_unchecked(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: vb.get_unchecked(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: lin(vb, &format!("{attn}.q_proj"))?,
            k_proj: lin(vb, &format!("{attn}.k_proj"))?,
            v_proj: lin(vb, &format!("{attn}.v_proj"))?,
            o_proj: lin(vb, &format!("{attn}.o_proj"))?,
            q_norm: vb.get_unchecked(&format!("{attn}.q_norm.weight"))?,
            k_norm: vb.get_unchecked(&format!("{attn}.k_norm.weight"))?,
            gate: lin(vb, &format!("{mlp}.gate_proj"))?,
            up: lin(vb, &format!("{mlp}.up_proj"))?,
            down: lin(vb, &format!("{mlp}.down_proj"))?,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            groups: cfg.kv_groups(),
            scale: (cfg.head_dim as f64).powf(-0.5),
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let normed = rms_norm(x, &self.input_ln, self.eps)?;

        let q = self
            .q_proj
            .forward(&normed)?
            .reshape((b, s, self.n_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(&normed)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(&normed)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?;

        // Per-head q/k RMSNorm over head_dim (Qwen3 q_norm/k_norm), then half-split RoPE.
        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;
        let q = apply_rope_half(&q, cos, sin)?;
        let k = apply_rope_half(&k, cos, sin)?;

        // [b,s,h,hd] -> [b,h,s,hd], then GQA-expand K/V from n_kv_heads to n_heads.
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = repeat_kv(&k.transpose(1, 2)?.contiguous()?, self.groups)?;
        let v = repeat_kv(&v.transpose(1, 2)?.contiguous()?, self.groups)?;

        let attn = sdpa(&q, &k, &v, self.scale, Some(mask))?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        let x = (x + self.o_proj.forward(&attn)?)?;

        // SwiGLU MLP.
        let normed = rms_norm(&x, &self.post_ln, self.eps)?;
        let gated = (self.gate.forward(&normed)?.silu()? * self.up.forward(&normed)?)?;
        Ok((x + self.down.forward(&gated)?)?)
    }
}

/// The Qwen3-0.6B text tower (token embed → 28 pre-norm decoder layers → final RMSNorm).
pub struct AnimaQwen3 {
    embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    head_dim: usize,
    rope_theta: f64,
    eps: f64,
    device: Device,
}

impl AnimaQwen3 {
    /// `vb` is a VarBuilder rooted at the Qwen3 model (`"model"` for `model.embed_tokens.*`).
    pub fn new(vb: &VarBuilder, cfg: &Qwen3Config) -> Result<Self> {
        let embed = Embedding::new(vb.get_unchecked("embed_tokens.weight")?, cfg.hidden_size);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(DecoderLayer::new(vb, &format!("layers.{i}"), cfg)?);
        }
        Ok(Self {
            embed,
            layers,
            norm: vb.get_unchecked("norm.weight")?,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            eps: cfg.rms_norm_eps,
            device: vb.device().clone(),
        })
    }

    /// Additive causal mask `[1, 1, s, s]` (0 on/below the diagonal, `MASK_NEG` above) at `dtype`.
    fn causal_mask(&self, s: usize, dtype: DType) -> Result<Tensor> {
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                data[i * s + j] = MASK_NEG;
            }
        }
        Ok(Tensor::from_vec(data, (1, 1, s, s), &self.device)?.to_dtype(dtype)?)
    }

    /// `input_ids`: `[B, S]` **U32** token ids. Returns the **last_hidden_state** `[B, S, hidden]` (the
    /// last decoder layer AFTER the final norm). `dtype` is the tower compute dtype (bf16 in production,
    /// f32 for CPU parity). Attention is causal; padding is applied by the pipeline's mask-multiply.
    pub fn forward(&self, input_ids: &Tensor, dtype: DType) -> Result<Tensor> {
        let s = input_ids.dim(1)?;
        let mut h = self.embed.forward(input_ids)?.to_dtype(dtype)?;
        let (cos, sin) = text_rope(s, self.head_dim, self.rope_theta, &self.device)?;
        let mask = self.causal_mask(s, dtype)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        // last_hidden_state = final RMSNorm applied to the last layer.
        rms_norm(&h, &self.norm, self.eps)
    }
}
