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
use crate::primitives::attention::{sdpa, AttnMask};
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
        // Phi-3 fuses q‖k‖v into one `qkv_proj` and gate‖up into one `gate_up_proj`; the row spans the
        // split slices below carve out (each `[out, hidden]`, so the split is along axis 0).
        let qd = num_heads * head_dim;
        let kvd = num_kv_heads * head_dim;
        let inter = cfg.intermediate_size as usize;

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
            // Attention projections: a packed `qkv_proj` (Phi-3) is split into q/k/v, else the
            // separate `q_proj`/`k_proj`/`v_proj` are loaded directly.
            let (q, k, v) = {
                let packed = lp("self_attn.qkv_proj.weight");
                if w.contains(&packed) {
                    let qkv = req(packed)?; // [qd + 2*kvd, hidden]
                    (
                        Projection::load(qkv.narrow(0, 0, qd)?.contiguous()?, quant)?,
                        Projection::load(qkv.narrow(0, qd, kvd)?.contiguous()?, quant)?,
                        Projection::load(qkv.narrow(0, qd + kvd, kvd)?.contiguous()?, quant)?,
                    )
                } else {
                    (
                        proj(lp("self_attn.q_proj.weight"))?,
                        proj(lp("self_attn.k_proj.weight"))?,
                        proj(lp("self_attn.v_proj.weight"))?,
                    )
                }
            };
            // MLP gate/up: a packed `gate_up_proj` (Phi-3) is split, else separate projections.
            let (gate, up) = {
                let packed = lp("mlp.gate_up_proj.weight");
                if w.contains(&packed) {
                    let gu = req(packed)?; // [2*inter, hidden]
                    (
                        Projection::load(gu.narrow(0, 0, inter)?.contiguous()?, quant)?,
                        Projection::load(gu.narrow(0, inter, inter)?.contiguous()?, quant)?,
                    )
                } else {
                    (
                        proj(lp("mlp.gate_proj.weight"))?,
                        proj(lp("mlp.up_proj.weight"))?,
                    )
                }
            };
            layers.push(LlamaLayer {
                input_ln: req(lp("input_layernorm.weight"))?,
                post_ln: req(lp("post_attention_layernorm.weight"))?,
                attn: LlamaAttention {
                    q,
                    k,
                    v,
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
                    gate,
                    up,
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

    /// The engine's compute dtype for this model (bf16 on GPU, f32 on CPU) — the batched decode reads
    /// it to match its additive attention mask to the score dtype.
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`) — the **per-sequence** position tables the
    /// batched decode (story 7255) feeds [`LlamaModel::decode_logits_masked`]. Each is
    /// `[rows, cols, head_dim]` in the compute dtype.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.cos_sin_at(positions, self.dtype, &self.device)?; // [1, rows*cols, hd]
        let hd = self.rope.dim();
        Ok((
            cos.reshape((rows as usize, cols as usize, hd))?,
            sin.reshape((rows as usize, cols as usize, hd))?,
        ))
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
        let s = input_embeds.dim(1)? as i32;
        // Single-sequence / uniform batch: positions [offset, offset+s) shared across the batch, with
        // an implicit bottom-right causal mask; cos/sin `[1, s, head_dim]` broadcast over the batch.
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        self.forward_to_last_logits(input_embeds, cache, &cos, &sin, AttnMask::Causal)
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE positions
    /// and an explicit additive attention mask — the decode primitive the dynamic-batch scheduler
    /// (story 7255) runs each step.
    ///
    /// `input_ids` is `[batch, seq]` (u32); `cos`/`sin` are `[batch, seq, head_dim]` (per-row
    /// positions, e.g. from [`LlamaModel::rope_tables`]); `mask` is an additive
    /// `[batch, 1, seq, k_total]` score mask (`0` keep, large-negative block) covering left-padding +
    /// causality. Returns logits for the **last column** `[batch, vocab]` — left-padding right-aligns
    /// every row's last real token to that column, so one slice serves the whole batch.
    pub fn decode_logits_masked(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_to_last_logits(&embeds, cache, cos, sin, AttnMask::Additive(mask))
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`. The shared core of the single and
    /// batched forwards: they differ only in how `cos`/`sin` and `mask` are built.
    fn forward_to_last_logits(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
    ) -> Result<Tensor> {
        let (b, s, _) = input_embeds.dims3()?;

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin, mask, cache, i)?;
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
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let attn = self
            .attn
            .forward(&normed, cos, sin, mask, cache, layer_idx)?;
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
        mask: AttnMask<'_>,
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

        let out = sdpa(&q, &k_all, &v_all, self.scale, mask)?; // [b, heads, s, head_dim]
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
