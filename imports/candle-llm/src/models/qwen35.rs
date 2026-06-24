//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) — the hybrid decoder (story sc-7632,
//! the candle mirror of mlx-llm sc-7628/7629).
//!
//! Unlike the generic [`CausalLm`](super::llama::CausalLm) (all softmax attention over a growing KV
//! cache), this decoder interleaves two mixer types on a fixed schedule (`full_attention_interval`,
//! default 4 → **3 Gated DeltaNet linear-attention layers : 1 gated full-attention layer**):
//!
//! - [`GatedDeltaNet`] — linear attention carrying a fixed-size recurrent state (the verified
//!   primitives in [`crate::primitives::gated_delta`]): a 4-way in-projection → short conv → q/k
//!   L2-norm → gated delta recurrence → gated RMS-norm → out-proj.
//! - [`Qwen35Attention`] — grouped-query attention with **partial RoPE** (`partial_rotary_factor`,
//!   reusing the [`Rope::partial`] path), per-head q/k RMSNorm, and an **output gate** (the queries
//!   projection is doubled into `[queries ‖ gate]`, and the attended output is multiplied by
//!   `sigmoid(gate)` before the output projection).
//!
//! Each decoder layer is `input_layernorm → mixer → residual → post_attention_layernorm → MLP →
//! residual`. The MLP is a dense SwiGLU for the 27B; the 35B MoE bank is wired in a follow-on slice
//! (sc-7632 slice 4). The KV cache (full-attn layers) and the recurrent [`DeltaNetCache`] (linear
//! layers) live side by side in a per-layer [`Qwen35Cache`]. RMSNorm weights follow the Qwen3-Next
//! `(1 + weight)` convention; the recurrence accumulates in f32 (matching the reference GPU kernel)
//! while the rest of the decoder runs in the device compute dtype (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;
use serde_json::Value;

use crate::device::compute_dtype;
use crate::error::{Error, Result};
use crate::primitives::attention::{repeat_kv, sdpa, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::nn::{embed, linear, rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

/// Mixture-of-Experts FFN parameters (`qwen3_5_moe`, the 35B-A3B). Parsed so the loader can detect
/// the MoE variant; the MoE FFN itself is wired in sc-7632 slice 4.
#[derive(Clone, Copy, Debug)]
pub struct MoeParams {
    pub num_experts: i32,
    pub experts_per_tok: usize,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
}

/// Parsed Qwen3.6 (`qwen3_5` / `qwen3_5_moe`) text-decoder configuration. Read from the nested
/// `text_config` of the VLM wrapper (or the top-level config if not wrapped).
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    /// Every `full_attention_interval`-th layer (1-indexed) is full attention; the rest are linear.
    pub full_attention_interval: usize,
    // Linear (Gated DeltaNet) dims.
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    /// MoE FFN parameters when this is the MoE variant (`qwen3_5_moe`, 35B-A3B); `None` ⇒ dense MLP
    /// (`qwen3_5`, 27B).
    pub moe: Option<MoeParams>,
}

impl Qwen35Config {
    /// Parse from a `config.json` value, descending into `text_config` for the VLM wrapper.
    pub fn from_json(v: &Value) -> Result<Self> {
        let c = v.get("text_config").unwrap_or(v);
        let int = |k: &str| -> Option<i32> { c.get(k).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req = |k: &str| -> Result<i32> {
            int(k).ok_or_else(|| Error::Config(format!("qwen3_5 config.json missing `{k}`")))
        };
        let f32o = |k: &str| -> Option<f32> { c.get(k).and_then(|x| x.as_f64()).map(|x| x as f32) };
        // RoPE params moved into a `rope_parameters` sub-object in newer configs (Qwen3.6); read
        // there first, then a legacy top-level field, then the architecture default.
        let rope_f32 = |k: &str| -> Option<f32> {
            c.get("rope_parameters")
                .and_then(|rp| rp.get(k))
                .and_then(|x| x.as_f64())
                .or_else(|| c.get(k).and_then(|x| x.as_f64()))
                .map(|x| x as f32)
        };

        let hidden_size = req("hidden_size")?;
        let num_heads = req("num_attention_heads")?;
        // The MoE variant (`qwen3_5_moe`) has no dense `intermediate_size` — every layer is MoE — so
        // fall back to the per-expert width (unused on the MoE path, but keeps the field valid).
        let intermediate_size = int("intermediate_size")
            .or_else(|| int("moe_intermediate_size"))
            .unwrap_or(0);
        Ok(Self {
            hidden_size,
            num_layers: req("num_hidden_layers")? as usize,
            intermediate_size,
            num_heads,
            num_kv_heads: int("num_key_value_heads").unwrap_or(num_heads),
            head_dim: int("head_dim").unwrap_or(hidden_size / num_heads),
            vocab_size: req("vocab_size")?,
            rms_norm_eps: f32o("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: rope_f32("rope_theta").unwrap_or(10_000_000.0),
            partial_rotary_factor: rope_f32("partial_rotary_factor").unwrap_or(0.25),
            max_position_embeddings: int("max_position_embeddings").unwrap_or(0),
            tie_word_embeddings: c
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            full_attention_interval: int("full_attention_interval").unwrap_or(4).max(1) as usize,
            linear_num_value_heads: req("linear_num_value_heads")?,
            linear_num_key_heads: req("linear_num_key_heads")?,
            linear_key_head_dim: req("linear_key_head_dim")?,
            linear_value_head_dim: req("linear_value_head_dim")?,
            linear_conv_kernel_dim: int("linear_conv_kernel_dim").unwrap_or(4),
            moe: int("num_experts").map(|num_experts| MoeParams {
                num_experts,
                experts_per_tok: int("num_experts_per_tok").unwrap_or(8).max(1) as usize,
                moe_intermediate_size: int("moe_intermediate_size").unwrap_or(intermediate_size),
                shared_expert_intermediate_size: int("shared_expert_intermediate_size")
                    .unwrap_or(intermediate_size),
            }),
        })
    }

    /// Whether layer `i` (0-indexed) is a linear (Gated DeltaNet) layer; otherwise full attention.
    pub fn is_linear(&self, i: usize) -> bool {
        !(i + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Number of head dimensions partial RoPE rotates (even).
    pub fn rotary_dim(&self) -> i32 {
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as i32;
        rd & !1
    }
}

/// L2-normalize over the last axis: `x · rsqrt(Σ x² + eps)` (the FLA `use_qk_l2norm_in_kernel`
/// convention — `eps` is added to the **sum**, not the mean). Computed in `x`'s dtype, matching the
/// reference kernel which normalizes the projected q/k before the recurrence.
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let last = x.rank() - 1;
    let ss = x.sqr()?.sum_keepdim(last)?; // Σ x²  → [.., 1]
    let inv = (ss + eps)?.powf(-0.5)?;
    Ok(x.broadcast_mul(&inv)?)
}

/// The Gated DeltaNet linear-attention layer (`Qwen3_5GatedDeltaNet`).
///
/// The Qwen3.6 checkpoint splits the input projection **four ways** — `in_proj_qkv` (fused q‖k‖v, the
/// only part the short conv mixes), `in_proj_z` (the output gate), and the per-value-head `in_proj_a`
/// / `in_proj_b` (decay / delta-strength). After the conv, q/k/v are a **contiguous** split of the
/// `[key_dim, key_dim, value_dim]` channels (no head interleaving).
struct GatedDeltaNet {
    in_proj_qkv: Projection, // [key_dim·2 + value_dim, hidden] → conv'd
    in_proj_z: Projection,   // [value_dim, hidden]             → output gate
    in_proj_a: Projection,   // [Hv, hidden]                    → decay input
    in_proj_b: Projection,   // [Hv, hidden]                    → delta-strength input
    conv_weight: Tensor,     // [conv_dim, K]
    a_log: Tensor,           // [Hv]
    dt_bias: Tensor,         // [Hv]
    norm_weight: Tensor,     // [Dv] (RMSNormGated; loaded directly, ones-centered)
    out_proj: Projection,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel: usize,
    eps: f64,
}

impl GatedDeltaNet {
    fn forward(&self, x: &Tensor, cache: &mut DeltaNetCache) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        // Four independent in-projections (dtype follows the projection weights).
        let mixed = self.in_proj_qkv.forward(x)?; // [b,s,conv_dim] = q‖k‖v channels
        let dt = mixed.dtype();
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape((b, s, self.num_v_heads, self.head_v_dim))?; // output gate
        let a_in = self.in_proj_a.forward(x)?; // [b,s,Hv] decay input
        let b_in = self.in_proj_b.forward(x)?; // [b,s,Hv] delta-strength input

        // Short conv over the q‖k‖v channels (only these are convolved), seeded by the cache tail,
        // then a *contiguous* split into q [key_dim] ‖ k [key_dim] ‖ v [value_dim] and reshape to heads.
        let conv_state = match &cache.conv_state {
            Some(cs) => cs.clone(),
            None => Tensor::zeros((b, self.conv_kernel - 1, self.conv_dim), dt, x.device())?,
        };
        let (conv_out, new_conv) = causal_depthwise_conv(&mixed, &self.conv_weight, &conv_state)?;
        let qc = conv_out
            .narrow(2, 0, self.key_dim)?
            .contiguous()?
            .reshape((b, s, self.num_k_heads, self.head_k_dim))?;
        let kc = conv_out
            .narrow(2, self.key_dim, self.key_dim)?
            .contiguous()?
            .reshape((b, s, self.num_k_heads, self.head_k_dim))?;
        let vc = conv_out
            .narrow(2, 2 * self.key_dim, self.value_dim)?
            .contiguous()?
            .reshape((b, s, self.num_v_heads, self.head_v_dim))?;

        // L2-normalize q/k (eps 1e-6), then scale q by 1/√head_k_dim — `use_qk_l2norm_in_kernel`.
        let inv = (self.head_k_dim as f64).powf(-0.5);
        let qn = l2norm(&qc, 1e-6)?.affine(inv, 0.0)?;
        let kn = l2norm(&kc, 1e-6)?;

        // The gated delta recurrence, accumulated in f32 (matching the reference kernel). GQA (q/k
        // from Hk key heads → Hv value heads) is handled inside the recurrence primitive.
        let beta = sigmoid(&b_in)?;
        let g = compute_g(&a_in, &self.a_log, &self.dt_bias)?;
        let f = DType::F32;
        let (y, new_ssm) = gated_delta_recurrence(
            &qn.to_dtype(f)?,
            &kn.to_dtype(f)?,
            &vc.to_dtype(f)?,
            &g.to_dtype(f)?,
            &beta.to_dtype(f)?,
            cache.ssm_state.as_ref(),
        )?;

        // Gated RMS-norm with z (back in the layer dtype), then the output projection.
        let out = rms_norm_gated(&y.to_dtype(dt)?, &self.norm_weight, &z, self.eps)?;
        let result = self
            .out_proj
            .forward(&out.reshape((b, s, self.value_dim))?)?;
        cache.update(new_conv, new_ssm, s as i32);
        Ok(result)
    }
}

/// The gated full-attention layer (`Qwen3NextAttention`).
struct Qwen35Attention {
    q_proj: Projection, // out = num_heads · head_dim · 2 (queries ‖ gate)
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    groups: usize,
    scale: f32,
    eps: f64,
}

impl Qwen35Attention {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut AttnKv,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        // q_proj is doubled into [queries ‖ gate]; split along the head_dim axis.
        let qg = self.q_proj.forward(x)?.reshape((b, s, nh, 2 * hd))?;
        let q = qg.narrow(3, 0, hd)?.contiguous()?;
        let gate = qg
            .narrow(3, hd, hd)?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        let q = rms_norm(&q, &self.q_norm, self.eps)?; // [b,s,H,hd]

        let k = rms_norm(
            &self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?,
            &self.k_norm,
            self.eps,
        )?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;

        // Partial RoPE (NeoX), then transpose into head-major [b,H,s,hd].
        let q = apply_rope(&q, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rope(&k, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k_all, v_all) = cache.update(&k, &v)?;
        let k_all = repeat_kv(&k_all, self.groups)?;
        let v_all = repeat_kv(&v_all, self.groups)?;
        let out = sdpa(&q, &k_all, &v_all, self.scale, None, AttnMask::Causal)?; // [b,H,s,hd]
        let merged = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        // Output gate: multiply by sigmoid(gate) before the output projection.
        let gated = merged.broadcast_mul(&sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

/// Dense SwiGLU MLP (`Qwen3_5MLP`) — the 27B FFN.
struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&gate.broadcast_mul(&up)?)
    }
}

enum Mixer {
    Delta(GatedDeltaNet),
    Attn(Qwen35Attention),
}

struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    mixer: Mixer,
    ffn: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Qwen35LayerCache,
    ) -> Result<Tensor> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let r = match (&self.mixer, cache) {
            (Mixer::Delta(d), Qwen35LayerCache::Delta(c)) => d.forward(&normed, c)?,
            (Mixer::Attn(a), Qwen35LayerCache::Attn(c)) => a.forward(&normed, cos, sin, c)?,
            _ => return Err(Error::Msg("qwen3_5: cache/mixer type mismatch".into())),
        };
        let h = x.broadcast_add(&r)?;
        let m = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
        Ok(h.broadcast_add(&m)?)
    }
}

/// A single full-attention layer's growing KV (the linear layers use [`DeltaNetCache`] instead).
#[derive(Clone, Debug, Default)]
pub struct AttnKv {
    kv: Option<(Tensor, Tensor)>,
}

impl AttnKv {
    fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let merged = match self.kv.take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, k], 2)?, Tensor::cat(&[&pv, v], 2)?),
            None => (k.clone(), v.clone()),
        };
        self.kv = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    fn offset(&self) -> i32 {
        self.kv
            .as_ref()
            .map(|(k, _)| k.dims()[2] as i32)
            .unwrap_or(0)
    }
}

/// The per-layer cache slot — a recurrent [`DeltaNetCache`] for linear layers, growing KV for
/// full-attention layers.
#[derive(Clone, Debug)]
pub enum Qwen35LayerCache {
    Delta(DeltaNetCache),
    Attn(AttnKv),
}

/// The hybrid decoder's cache: one slot per decoder layer.
#[derive(Clone, Debug)]
pub struct Qwen35Cache {
    layers: Vec<Qwen35LayerCache>,
}

impl Qwen35Cache {
    /// Positions already cached — the RoPE offset for the next step (read from the first full-attn
    /// layer; all layers advance in lockstep).
    pub fn offset(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|l| match l {
                Qwen35LayerCache::Attn(a) => Some(a.offset()),
                Qwen35LayerCache::Delta(_) => None,
            })
            .unwrap_or(0)
    }

    /// Drop all cached state.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            match l {
                Qwen35LayerCache::Delta(c) => c.reset(),
                Qwen35LayerCache::Attn(a) => a.kv = None,
            }
        }
    }
}

/// A loaded Qwen3.6 (`qwen3_5`) hybrid decoder.
pub struct Qwen35Model {
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
    rope: Rope,
    cfg: Qwen35Config,
    eps: f64,
    dtype: DType,
    device: Device,
    quantized: bool,
}

impl Qwen35Model {
    /// The parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Whether the large projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The compute dtype (bf16 on GPU, f32 on CPU).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// A fresh per-layer cache (linear vs full-attn slot per the schedule).
    pub fn new_cache(&self) -> Qwen35Cache {
        let layers = (0..self.cfg.num_layers)
            .map(|i| {
                if self.cfg.is_linear(i) {
                    Qwen35LayerCache::Delta(DeltaNetCache::new())
                } else {
                    Qwen35LayerCache::Attn(AttnKv::default())
                }
            })
            .collect();
        Qwen35Cache { layers }
    }

    /// Run the decoder stack over `input_ids` `[B, S]` at sequence `offset`, returning the final
    /// hidden states `[B, S, hidden]` (before the final norm / lm_head).
    fn hidden(&self, input_ids: &Tensor, cache: &mut Qwen35Cache, offset: i32) -> Result<Tensor> {
        let mut h = embed(&self.embed_tokens, input_ids)?.to_dtype(self.dtype)?;
        let s = h.dim(1)? as i32;
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, &cos, &sin, slot)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` over hidden states `[B, n, hidden]` → logits `[B, n, vocab]`.
    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(h, &self.norm, self.eps)?;
        linear(&normed, &self.lm_head, None)
    }

    /// Run the decoder over `input_ids` `[B, S]` at sequence `offset`, returning logits for **every**
    /// position `[B, S, vocab]`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Tensor> {
        let h = self.hidden(input_ids, cache, offset)?;
        self.project(&h)
    }

    /// Run the decoder and return logits for the **last** position only, `[B, vocab]` — the decode
    /// contract (prefill + single-token decode).
    pub fn decode_logits(
        &self,
        input_ids: &Tensor,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Tensor> {
        let h = self.hidden(input_ids, cache, offset)?;
        let (b, s, _) = h.dims3()?;
        let last = h.narrow(1, s - 1, 1)?.contiguous()?; // [b,1,hidden]
        let logits = self.project(&last)?; // [b,1,vocab]
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
    }

    /// Build from a loaded checkpoint (dense). See [`Qwen35Model::from_weights_dtype`].
    pub fn from_weights(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the large projections on load. The
    /// compute dtype is the device default ([`compute_dtype`] — bf16 on GPU, f32 on CPU).
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        Self::from_weights_dtype(w, prefix, cfg, quant, compute_dtype(w.device()))
    }

    /// Build from a loaded checkpoint with an explicit compute `dtype`.
    ///
    /// `prefix` is the **decoder root** path: keys are read as `{prefix}.embed_tokens.weight`,
    /// `{prefix}.norm.weight`, `{prefix}.layers.{i}.…`. For the VLM-wrapped Qwen3.6 checkpoint this is
    /// `model.language_model`; `lm_head.weight` lives at the **checkpoint root** (untied), not under
    /// the prefix. `quant` (Q4/Q8) is applied to the big matmuls (in/out projections, attention
    /// q/k/v/o, MLP); the per-head decay/delta projections, conv, `A_log`/`dt_bias`, and all norms
    /// stay dense.
    pub fn from_weights_dtype(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
        dtype: DType,
    ) -> Result<Self> {
        // The MoE FFN (35B-A3B) is wired in a follow-on slice; load the dense (27B) path only.
        if cfg.moe.is_some() {
            return Err(Error::Unsupported(
                "qwen3_5_moe (35B-A3B) FFN is not yet wired (sc-7632 slice 4)".into(),
            ));
        }
        let device = w.device().clone();
        let eps = cfg.rms_norm_eps as f64;
        let join = |s: &str| -> String {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        // Qwen3.6 RMSNorm weights are stored zero-centered → fold in the +1 (the (1 + weight)
        // convention). The gated DeltaNet norm is the exception: it is ones-centered, loaded raw.
        let norm_w = |key: String| -> Result<Tensor> { Ok(req(key)?.affine(1.0, 1.0)?) };
        let proj_q = |key: String| -> Result<Projection> { Projection::load(req(key)?, quant) };
        let proj_dense = |key: String| -> Result<Projection> { Projection::load(req(key)?, None) };

        let embed_tokens = req(join("embed_tokens.weight"))?;
        let norm = norm_w(join("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req("lm_head.weight".to_string())?
        };

        let key_dim = (cfg.linear_key_head_dim * cfg.linear_num_key_heads) as usize;
        let value_dim = (cfg.linear_value_head_dim * cfg.linear_num_value_heads) as usize;
        let conv_dim = key_dim * 2 + value_dim;
        let groups = (cfg.num_heads / cfg.num_kv_heads) as usize;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| join(&format!("layers.{i}.{s}"));
            let mixer = if cfg.is_linear(i) {
                // conv1d.weight is [conv_dim, 1, K] (HF) → squeeze the singleton to [conv_dim, K].
                let conv_weight = req(lp("linear_attn.conv1d.weight"))?
                    .reshape((conv_dim, cfg.linear_conv_kernel_dim as usize))?;
                Mixer::Delta(GatedDeltaNet {
                    in_proj_qkv: proj_q(lp("linear_attn.in_proj_qkv.weight"))?,
                    in_proj_z: proj_q(lp("linear_attn.in_proj_z.weight"))?,
                    in_proj_a: proj_dense(lp("linear_attn.in_proj_a.weight"))?,
                    in_proj_b: proj_dense(lp("linear_attn.in_proj_b.weight"))?,
                    conv_weight,
                    a_log: req(lp("linear_attn.A_log"))?,
                    dt_bias: req(lp("linear_attn.dt_bias"))?,
                    norm_weight: req(lp("linear_attn.norm.weight"))?,
                    out_proj: proj_q(lp("linear_attn.out_proj.weight"))?,
                    num_k_heads: cfg.linear_num_key_heads as usize,
                    num_v_heads: cfg.linear_num_value_heads as usize,
                    head_k_dim: cfg.linear_key_head_dim as usize,
                    head_v_dim: cfg.linear_value_head_dim as usize,
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel: cfg.linear_conv_kernel_dim as usize,
                    eps,
                })
            } else {
                Mixer::Attn(Qwen35Attention {
                    q_proj: proj_q(lp("self_attn.q_proj.weight"))?,
                    k_proj: proj_q(lp("self_attn.k_proj.weight"))?,
                    v_proj: proj_q(lp("self_attn.v_proj.weight"))?,
                    o_proj: proj_q(lp("self_attn.o_proj.weight"))?,
                    q_norm: norm_w(lp("self_attn.q_norm.weight"))?,
                    k_norm: norm_w(lp("self_attn.k_norm.weight"))?,
                    num_heads: cfg.num_heads as usize,
                    num_kv_heads: cfg.num_kv_heads as usize,
                    head_dim: cfg.head_dim as usize,
                    groups,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    eps,
                })
            };
            let ffn = Mlp {
                gate: proj_q(lp("mlp.gate_proj.weight"))?,
                up: proj_q(lp("mlp.up_proj.weight"))?,
                down: proj_q(lp("mlp.down_proj.weight"))?,
            };
            layers.push(DecoderLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp("post_attention_layernorm.weight"))?,
                mixer,
                ffn,
                eps,
            });
        }

        let rope = Rope::partial(cfg.rotary_dim(), cfg.rope_theta, false);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            eps,
            cfg,
            dtype,
            device,
            quantized: quant.is_some(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn cfg_json() -> Value {
        // 4 layers → schedule (interval 4): layers 0,1,2 linear, layer 3 full attention.
        json!({
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "vocab_size": 50, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5, "max_position_embeddings": 128,
                "tie_word_embeddings": false, "full_attention_interval": 4,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4
            },
            "vision_config": { "model_type": "qwen3_5" }
        })
    }

    /// A deterministic small tensor of shape `dims` (finite, non-degenerate), on CPU.
    fn t(map: &mut HashMap<String, Tensor>, key: &str, dims: &[usize]) {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        map.insert(
            key.to_string(),
            Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap(),
        );
    }

    fn synthetic_weights(cfg: &Qwen35Config) -> Weights {
        let h = cfg.hidden_size as usize;
        let key_dim = (cfg.linear_key_head_dim * cfg.linear_num_key_heads) as usize;
        let value_dim = (cfg.linear_value_head_dim * cfg.linear_num_value_heads) as usize;
        let conv_dim = key_dim * 2 + value_dim;
        let kk = cfg.linear_conv_kernel_dim as usize;
        let nh = cfg.num_heads as usize;
        let nkv = cfg.num_kv_heads as usize;
        let hd = cfg.head_dim as usize;
        let hv = cfg.linear_num_value_heads as usize;
        let inter = cfg.intermediate_size as usize;
        let mut m = HashMap::new();
        // Mirror the real VLM-wrapped layout: the text decoder nests under `model.language_model`,
        // with `lm_head.weight` at the checkpoint root.
        let pfx = "model.language_model";
        t(
            &mut m,
            &format!("{pfx}.embed_tokens.weight"),
            &[cfg.vocab_size as usize, h],
        );
        t(&mut m, &format!("{pfx}.norm.weight"), &[h]);
        t(&mut m, "lm_head.weight", &[cfg.vocab_size as usize, h]);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| format!("{pfx}.layers.{i}.{s}");
            t(&mut m, &lp("input_layernorm.weight"), &[h]);
            t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
            t(&mut m, &lp("mlp.gate_proj.weight"), &[inter, h]);
            t(&mut m, &lp("mlp.up_proj.weight"), &[inter, h]);
            t(&mut m, &lp("mlp.down_proj.weight"), &[h, inter]);
            if cfg.is_linear(i) {
                t(
                    &mut m,
                    &lp("linear_attn.in_proj_qkv.weight"),
                    &[conv_dim, h],
                );
                t(&mut m, &lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
                t(&mut m, &lp("linear_attn.in_proj_a.weight"), &[hv, h]);
                t(&mut m, &lp("linear_attn.in_proj_b.weight"), &[hv, h]);
                t(&mut m, &lp("linear_attn.conv1d.weight"), &[conv_dim, 1, kk]);
                t(&mut m, &lp("linear_attn.A_log"), &[hv]);
                t(&mut m, &lp("linear_attn.dt_bias"), &[hv]);
                t(
                    &mut m,
                    &lp("linear_attn.norm.weight"),
                    &[cfg.linear_value_head_dim as usize],
                );
                t(&mut m, &lp("linear_attn.out_proj.weight"), &[h, value_dim]);
            } else {
                t(&mut m, &lp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
                t(&mut m, &lp("self_attn.k_proj.weight"), &[nkv * hd, h]);
                t(&mut m, &lp("self_attn.v_proj.weight"), &[nkv * hd, h]);
                t(&mut m, &lp("self_attn.o_proj.weight"), &[h, nh * hd]);
                t(&mut m, &lp("self_attn.q_norm.weight"), &[hd]);
                t(&mut m, &lp("self_attn.k_norm.weight"), &[hd]);
            }
        }
        Weights::from_map(m, Device::Cpu)
    }

    fn ids(toks: &[u32]) -> Tensor {
        Tensor::from_vec(toks.to_vec(), (1, toks.len()), &Device::Cpu).unwrap()
    }

    fn host(x: &Tensor) -> Vec<f32> {
        x.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn config_parses_and_schedules_3_linear_1_full() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.rotary_dim(), 4); // head_dim 8 * 0.5
        assert!(cfg.moe.is_none());
        // 3 linear : 1 full.
        assert!(cfg.is_linear(0) && cfg.is_linear(1) && cfg.is_linear(2));
        assert!(!cfg.is_linear(3));
    }

    #[test]
    fn assembled_forward_produces_finite_logits() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        let mut cache = model.new_cache();
        let logits = model
            .forward(&ids(&[1, 7, 3, 42, 9]), &mut cache, 0)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size as usize]);
        for x in host(&logits) {
            assert!(x.is_finite(), "non-finite logit: {x}");
        }
        // The full-attention layer (layer 3) advanced the KV cache to 5 positions.
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn decode_after_prefill_advances_cache() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        let mut cache = model.new_cache();
        model.forward(&ids(&[1, 2, 3]), &mut cache, 0).unwrap();
        assert_eq!(cache.offset(), 3);
        // One decode step at offset 3.
        let logits = model.forward(&ids(&[4]), &mut cache, 3).unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size as usize]);
        assert_eq!(cache.offset(), 4);
        assert!(host(&logits).iter().all(|x| x.is_finite()));
    }

    /// A model-level invariant the hybrid cache must satisfy: prefilling a sequence in one pass must
    /// produce the same final-token logits as feeding the tokens one at a time carrying the cache
    /// (conv tail + recurrent SSM state for linear layers, growing KV for full-attention). On CPU
    /// (f32) this is bit-exact; the tolerance allows a hair of float reorder.
    #[test]
    fn prefill_equals_stepwise_decode() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        let toks = [1u32, 7, 3, 42, 9, 2];

        let mut c_pre = model.new_cache();
        let prefill = model.decode_logits(&ids(&toks), &mut c_pre, 0).unwrap();

        let mut c_step = model.new_cache();
        let mut last = None;
        for (i, &tok) in toks.iter().enumerate() {
            last = Some(
                model
                    .decode_logits(&ids(&[tok]), &mut c_step, i as i32)
                    .unwrap(),
            );
        }
        let step = last.unwrap();

        assert_eq!(c_pre.offset(), toks.len() as i32);
        assert_eq!(c_step.offset(), toks.len() as i32);
        let (a, b) = (host(&prefill), host(&step));
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 1e-4,
            "prefill vs stepwise last-token logits diverged: max abs diff {md}"
        );
    }

    /// The whole Gated DeltaNet layer, validated against the numeric oracle from the exact
    /// `Qwen3_5GatedDeltaNet.forward` reference (4-way in-projection → short conv → contiguous q|k|v
    /// split → L2-norm + q-scale → GQA delta recurrence → gated RMS-norm(z) → out-proj). The same
    /// framework-independent fixture the mlx-llm port (sc-7629) used; CPU runs the whole layer in f32
    /// so the match is tight.
    #[test]
    fn deltanet_layer_matches_qwen3_5_reference() {
        let json: Value =
            serde_json::from_str(include_str!("testdata/qwen35_deltanet_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        // Dims mirror the generator (r = Hv/Hk = 2 GQA, single token).
        let (h, hk, hv, dk, dv) = (8usize, 2usize, 4usize, 4usize, 4usize);
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let kk = 4usize;
        let (b, s) = (1usize, 1usize);

        let mk = |k: &str, dims: &[usize]| {
            Tensor::from_vec(arr(k), dims.to_vec(), &Device::Cpu).unwrap()
        };
        let proj = |k: &str, dims: &[usize]| Projection::load(mk(k, dims), None).unwrap();
        let layer = GatedDeltaNet {
            in_proj_qkv: proj("in_proj_qkv", &[conv_dim, h]),
            in_proj_z: proj("in_proj_z", &[value_dim, h]),
            in_proj_a: proj("in_proj_a", &[hv, h]),
            in_proj_b: proj("in_proj_b", &[hv, h]),
            conv_weight: mk("conv_weight", &[conv_dim, kk]),
            a_log: mk("A_log", &[hv]),
            dt_bias: mk("dt_bias", &[hv]),
            norm_weight: mk("norm_weight", &[dv]),
            out_proj: proj("out_proj", &[h, value_dim]),
            num_k_heads: hk,
            num_v_heads: hv,
            head_k_dim: dk,
            head_v_dim: dv,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel: kk,
            eps: 1e-6,
        };

        let x = mk("x", &[b, s, h]);
        let mut cache = DeltaNetCache::new();
        let out = layer.forward(&x, &mut cache).unwrap();
        assert_eq!(out.dims(), &[b, s, h]);

        let got = host(&out);
        let exp = arr("expected_output");
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 2e-4,
            "deltanet layer vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}"
        );

        // The cache advanced and holds both the recurrent and conv state for a follow-on decode step.
        assert_eq!(cache.offset(), s as i32);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
    }
}
