//! Generic Llama-family causal decoder (Llama / Mistral / Qwen3).
//!
//! The Candle port of `mlx-llm`'s `LlamaModel`, modelled alongside `candle-gen-sensenova`'s
//! hand-rolled Qwen3 stack. Attention optionally applies per-head q/k RMSNorm (Qwen3); projections
//! are held behind [`Projection`] so a model can be quantized on load. The forward is `&self`; the
//! KV cache is the only mutable state, threaded in as `&mut dyn KvCache`.
//!
//! Shapes are batch-capable (`[batch, seq, …]`). `head_dim` is taken from config and may differ from
//! `hidden_size / num_heads` (e.g. Qwen3-0.6B: hidden 1024, 16 heads, head_dim 128). Compute runs in
//! the device's [`compute_dtype`] (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::config::LlamaConfig;
use crate::device::compute_dtype;
use crate::error::Result;
use crate::primitives::attention::sdpa_causal;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{repeat_kv, ContiguousKvCache, Weights};

/// A loaded causal decoder.
pub struct LlamaModel {
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: Tensor,
    lm_head: Linear,
    rope: Rope,
    cfg: LlamaConfig,
    dtype: DType,
    device: Device,
    quantized: bool,
}

impl LlamaModel {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: LlamaConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: LlamaConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let device = w.device().clone();
        let dtype = compute_dtype(&device);
        let p = |suffix: &str| join(prefix, suffix);
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let proj = |key: String| -> Result<Projection> { Projection::load(req(key)?, quant) };

        let embed_tokens = req(p("model.embed_tokens.weight"))?;
        let norm = req(p("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.clone(), None)
        } else {
            Linear::new(req(p("lm_head.weight"))?, None)
        };

        let qk_norm = cfg.has_qk_norm();
        let groups = cfg.groups() as usize;
        let num_heads = cfg.num_heads as usize;
        let num_kv_heads = cfg.num_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let scale = cfg.attn_scale();
        let eps = cfg.rms_norm_eps as f64;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(prefix, &format!("model.layers.{i}.{suffix}"));
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(req(lp("self_attn.q_norm.weight"))?),
                    Some(req(lp("self_attn.k_norm.weight"))?),
                )
            } else {
                (None, None)
            };
            layers.push(LlamaLayer {
                input_ln: req(lp("input_layernorm.weight"))?,
                post_ln: req(lp("post_attention_layernorm.weight"))?,
                attn: LlamaAttention {
                    q: proj(lp("self_attn.q_proj.weight"))?,
                    k: proj(lp("self_attn.k_proj.weight"))?,
                    v: proj(lp("self_attn.v_proj.weight"))?,
                    o: proj(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    groups,
                    eps,
                },
                mlp: LlamaMlp {
                    gate: proj(lp("mlp.gate_proj.weight"))?,
                    up: proj(lp("mlp.up_proj.weight"))?,
                    down: proj(lp("mlp.down_proj.weight"))?,
                },
                eps,
            });
        }

        let rope = cfg.build_rope();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            cfg,
            dtype,
            device,
            quantized: quant.is_some(),
        })
    }

    /// The model config.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// Whether the projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// Embed token ids `[batch, seq]` (u32) → `[batch, seq, hidden]`.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        embed(&self.embed_tokens, input_ids)
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (number of cached
    /// positions).
    pub fn decode_logits(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`LlamaModel::decode_logits`] but from pre-computed input embeddings — the hook the VLM
    /// path uses to splice image features before the decoder.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let (b, s, _) = input_embeds.dims3()?;
        let (cos, sin) = self
            .rope
            .cos_sin(s as i32, offset, self.dtype, &self.device)?;

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin, cache, i)?;
        }

        let last_h = h.narrow(1, s - 1, 1)?.contiguous()?; // [b, 1, hidden]
        let normed = rms_norm(&last_h, &self.norm, self.cfg.rms_norm_eps as f64)?;
        let logits = self.lm_head.forward(&normed)?; // [b, 1, vocab]
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
    }
}

impl crate::decode::Decode for LlamaModel {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn step(&self, input_ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        self.decode_logits(input_ids, cache, offset)
    }
}

/// One pre-norm transformer block.
struct LlamaLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: LlamaAttention,
    mlp: LlamaMlp,
    eps: f64,
}

impl LlamaLayer {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let attn = self.attn.forward(&normed, cos, sin, cache, layer_idx)?;
        let h = x.broadcast_add(&attn)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        let mlp = self.mlp.forward(&normed2)?;
        Ok(h.broadcast_add(&mlp)?)
    }
}

/// Grouped-query attention with RoPE and optional per-head q/k RMSNorm (Qwen3).
struct LlamaAttention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    groups: usize,
    eps: f64,
}

impl LlamaAttention {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        // Project, then split into heads in [b, s, heads, head_dim] layout.
        let mut q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let mut k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;

        // Qwen3 per-head q/k RMSNorm over the head_dim axis, before RoPE.
        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        // RoPE on q,k (cos/sin broadcast over the head axis), then -> [b, heads, s, head_dim].
        let q = apply_rope(&q, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let k = apply_rope(&k, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let k_all = repeat_kv(&k_all, self.groups)?;
        let v_all = repeat_kv(&v_all, self.groups)?;

        let out = sdpa_causal(&q, &k_all, &v_all, self.scale)?; // [b, heads, s, head_dim]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        self.o.forward(&out)
    }
}

/// SwiGLU feed-forward.
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl LlamaMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        let gated = (gate * up)?;
        self.down.forward(&gated)
    }
}

/// Join a key prefix and suffix (`""` prefix ⇒ the suffix verbatim).
fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_handles_empty_prefix() {
        assert_eq!(join("", "model.norm.weight"), "model.norm.weight");
        assert_eq!(
            join("language_model", "model.norm.weight"),
            "language_model.model.norm.weight"
        );
    }
}
