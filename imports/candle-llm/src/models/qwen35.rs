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
//! residual`. The MLP is a dense SwiGLU (the 27B) or a sparse Mixture-of-Experts bank ([`MoeFfn`], the
//! 35B-A3B). The KV cache (full-attn layers) and the recurrent [`DeltaNetCache`] (linear
//! layers) live side by side in a per-layer [`Qwen35Cache`]. RMSNorm weights follow the Qwen3-Next
//! `(1 + weight)` convention; the recurrence accumulates in f32 (matching the reference GPU kernel)
//! while the rest of the decoder runs in the device compute dtype (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;
use serde_json::Value;

use crate::device::compute_dtype;
use crate::error::{Error, Result};
use crate::models::deepstack::{self, deepstack_fused_decoder_layers};
use crate::primitives::attention::{repeat_kv, sdpa, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::nn::{embed, linear, rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{KvCache, Weights};

/// Interleaved M-RoPE output of [`Qwen35Model::mrope_positions`]: the temporal / height / width
/// position rows (each length `S`) plus the `mrope_delta` (`max_position + 1 − len`) for continuing
/// positions after the prompt.
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// Mixture-of-Experts FFN parameters (`qwen3_5_moe`, the 35B-A3B): the routed-expert count / top-k
/// and the per-expert + shared-expert FFN widths that drive the un-fused [`MoeFfn`].
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
    /// Interleaved M-RoPE section `[t, h, w]` (`rope_parameters.mrope_section`, sums to
    /// `rotary_dim/2`); `None` ⇒ the even split from [`Qwen35Config::mrope_section_resolved`]. Drives
    /// the per-channel axis assignment for image (3-D) positions; irrelevant to the text path.
    pub mrope_section: Option<[i32; 3]>,
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
            mrope_section: c
                .get("rope_parameters")
                .and_then(|rp| rp.get("mrope_section"))
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .map(|a| {
                    let g = |i: usize| a[i].as_i64().unwrap_or(0) as i32;
                    [g(0), g(1), g(2)]
                }),
        })
    }

    /// The interleaved M-RoPE section `[t, h, w]`, defaulting to an even split of `rotary_dim/2` when
    /// the config omits it (e.g. text-only checkpoints — where the section is moot). The order biases
    /// the remainder toward `t` then `h` (matching the released `[11, 11, 10]` for `rotary_dim/2 = 32`).
    pub fn mrope_section_resolved(&self) -> [usize; 3] {
        if let Some(s) = self.mrope_section {
            return [
                s[0].max(0) as usize,
                s[1].max(0) as usize,
                s[2].max(0) as usize,
            ];
        }
        let half = (self.rotary_dim() / 2) as usize;
        let base = half / 3;
        let rem = half % 3;
        [base + (rem > 0) as usize, base + (rem > 1) as usize, base]
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

/// Sparse Mixture-of-Experts FFN (`Qwen3_5MoeSparseMoeBlock`, the 35B-A3B): a softmax router over
/// `experts` (top-`experts_per_tok` per token, weights renormalized to sum to 1) plus an always-on
/// **sigmoid-gated** shared expert. Each expert runs only on its routed tokens (gathered, then
/// scatter-added back), so active compute scales with `experts_per_tok` (~3B of 35B). The fused
/// checkpoint tensors (`experts.gate_up_proj` / `experts.down_proj`) are un-fused into per-expert
/// [`Mlp`]s at load. Routing mirrors the generic [`MoeMlp`](super::llama) bank's CPU path.
struct MoeFfn {
    router: Tensor, // [num_experts, hidden]
    experts: Vec<Mlp>,
    shared: Mlp,
    shared_gate: Tensor, // [1, hidden] sigmoid gate
    experts_per_tok: usize,
}

impl MoeFfn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let dtype = x.dtype();
        let device = x.device();
        let xf = x.reshape((t, h))?;
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Router probabilities (f32 softmax for a stable top-k), pulled to host.
        let logits = xf.matmul(&self.router.t()?)?; // [t, E]
        let probs =
            candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?.to_vec2::<f32>()?;

        // Invert the per-token top-k into per-expert (token, weight) lists, renormalized to sum 1.
        let mut routed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_experts];
        for (ti, row) in probs.iter().enumerate() {
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            let top = &idx[..k];
            let denom = top
                .iter()
                .map(|&e| row[e])
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            for &e in top {
                routed[e].push((ti as u32, row[e] / denom));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_vec(
                toks.iter().map(|&(ti, _)| ti).collect::<Vec<u32>>(),
                (n,),
                device,
            )?;
            let wts = Tensor::from_vec(
                toks.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (n, 1),
                device,
            )?
            .to_dtype(dtype)?;
            let xe = xf.index_select(&idx, 0)?; // [n, h]
            let ye = self.experts[e].forward(&xe)?.broadcast_mul(&wts)?; // [n, h]
            out = out.index_add(&idx, &ye, 0)?;
        }

        // Always-on shared expert, gated by sigmoid(x · shared_gateᵀ).
        let shared = self.shared.forward(&xf)?;
        let sg = sigmoid(&xf.matmul(&self.shared_gate.t()?)?)?; // [t, 1]
        Ok((out + shared.broadcast_mul(&sg)?)?.reshape((b, s, h))?)
    }
}

/// The per-layer FFN: a dense SwiGLU (27B) or a sparse MoE block (35B-A3B).
enum Ffn {
    Dense(Mlp),
    Moe(MoeFfn),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
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
    ffn: Ffn,
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
        let h = embed(&self.embed_tokens, input_ids)?.to_dtype(self.dtype)?;
        let s = h.dim(1)? as i32;
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        self.hidden_from_embeds(&h, &cos, &sin, cache)
    }

    /// Run the decoder stack over precomputed input `embeds` `[B, S, hidden]` with the given RoPE
    /// tables, returning the final hidden states `[B, S, hidden]`. The token-id path ([`Self::hidden`])
    /// and the multimodal embeds path ([`Self::decode_logits_from_embeds`]) share this.
    fn hidden_from_embeds(
        &self,
        embeds: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Qwen35Cache,
    ) -> Result<Tensor> {
        let mut h = embeds.clone();
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, cos, sin, slot)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` over hidden states `[B, n, hidden]` → logits `[B, n, vocab]`.
    fn project(&self, h: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(h, &self.norm, self.eps)?;
        linear(&normed, &self.lm_head, None)
    }

    /// Project the **last** position of `h` `[B, S, hidden]` → logits `[B, vocab]`.
    fn project_last(&self, h: &Tensor) -> Result<Tensor> {
        let (b, s, _) = h.dims3()?;
        let last = h.narrow(1, s - 1, 1)?.contiguous()?; // [b,1,hidden]
        let logits = self.project(&last)?; // [b,1,vocab]
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
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
        self.project_last(&h)
    }

    /// Embed token ids `[B, S]` → `[B, S, hidden]` in the compute dtype — the splice point where the
    /// multimodal path overwrites image-token rows with the encoder's projected patch features
    /// ([`Self::splice_image_features`]).
    pub fn embed_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(embed(&self.embed_tokens, input_ids)?.to_dtype(self.dtype)?)
    }

    /// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with `image_features`
    /// `[num_image_tokens, hidden]` (the vision encoder's projected, merged patch rows), in sequence
    /// order. The number of image-token positions must equal the feature-row count.
    pub fn splice_image_features(
        &self,
        embeds: &Tensor,
        input_ids: &[i32],
        image_features: &Tensor,
        image_token_id: i32,
    ) -> Result<Tensor> {
        let hidden = self.cfg.hidden_size as usize;
        let s = embeds.dim(1)?;
        let feats = image_features.to_dtype(self.dtype)?;
        let num_img = input_ids.iter().filter(|&&x| x == image_token_id).count();
        if num_img != feats.dim(0)? {
            return Err(Error::Msg(format!(
                "qwen3_5 splice: {num_img} image tokens != {} feature rows",
                feats.dim(0)?
            )));
        }
        if num_img == 0 {
            return Ok(embeds.clone());
        }
        // Stitch text spans (from `embeds`) and image spans (from `feats`) in order — no scatter.
        let mut pieces: Vec<Tensor> = Vec::new();
        let mut feat_off = 0usize;
        let mut i = 0usize;
        while i < s {
            let is_img = input_ids[i] == image_token_id;
            let mut j = i;
            while j < s && (input_ids[j] == image_token_id) == is_img {
                j += 1;
            }
            let n = j - i;
            if is_img {
                pieces.push(feats.narrow(0, feat_off, n)?.reshape((1, n, hidden))?);
                feat_off += n;
            } else {
                pieces.push(embeds.narrow(1, i, n)?);
            }
            i = j;
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }

    /// Compute the interleaved M-RoPE 3-D position rows (`get_rope_index`, B=1) for `input_ids`
    /// containing `image_grid_thw`-described `image_token_id` runs, plus the `mrope_delta`
    /// (`max_position + 1 − len`) the decode loop adds to continue positions after the prompt.
    ///
    /// Text tokens advance all three axes (t,h,w) by 1; an image run lays its tokens out over the
    /// `(t, h/merge, w/merge)` grid (temporal constant, height = row, width = col, offset by the shared
    /// cursor) and then advances the cursor by `max(h, w) / merge`. `spatial_merge_size` comes from the
    /// vision config. Returns `(t_row, h_row, w_row, mrope_delta)`.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        let merge = spatial_merge_size.max(1);
        let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let mut cur = 0i32;
        let mut gi = 0usize;
        let mut i = 0usize;
        while i < input_ids.len() {
            if input_ids[i] == image_token_id {
                let g = *image_grid_thw.get(gi).ok_or_else(|| {
                    Error::Msg("qwen3_5 mrope: more image runs than image_grid_thw entries".into())
                })?;
                gi += 1;
                let (gt, gh, gw) = (g[0], g[1] / merge, g[2] / merge);
                if gh <= 0 || gw <= 0 || gt <= 0 {
                    return Err(Error::Msg(format!("qwen3_5 mrope: bad image grid {g:?}")));
                }
                let count = (gt * gh * gw) as usize;
                let run = input_ids[i..]
                    .iter()
                    .take_while(|&&x| x == image_token_id)
                    .count();
                if run != count {
                    return Err(Error::Msg(format!(
                        "qwen3_5 mrope: image run length {run} != grid tokens {count}"
                    )));
                }
                let frame = gh * gw;
                for k in 0..count as i32 {
                    t.push(k / frame + cur);
                    let rem = k % frame;
                    h.push(rem / gw + cur);
                    w.push(rem % gw + cur);
                }
                cur += gh.max(gw);
                i += count;
            } else {
                t.push(cur);
                h.push(cur);
                w.push(cur);
                cur += 1;
                i += 1;
            }
        }
        let maxpos = t
            .iter()
            .chain(h.iter())
            .chain(w.iter())
            .copied()
            .max()
            .unwrap_or(-1);
        let delta = maxpos + 1 - input_ids.len() as i32;
        Ok((t, h, w, delta))
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` (text embeds with image
    /// features spliced in) using **interleaved M-RoPE** from the explicit 3-D `positions`
    /// (temporal/height/width rows, each length `S`), returning last-position logits `[1, vocab]`.
    /// With all three rows equal (text-only) this is bit-identical to [`Self::decode_logits`].
    pub fn decode_logits_from_embeds(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
    ) -> Result<Tensor> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            self.dtype,
            &self.device,
        )?;
        let h = self.hidden_from_embeds(&embeds.to_dtype(self.dtype)?, &cos, &sin, cache)?;
        self.project_last(&h)
    }

    /// Like [`Self::decode_logits_from_embeds`] but with **DeepStack** feature fusion: after layer
    /// `i` (for `i < deepstack.len()`) the `i`-th tapped/merged ViT feature set is added to the
    /// visual-token rows (`visual_pos_mask`). `deepstack` is empty for the Qwen3.6 vision path (its
    /// ViT has no DeepStack taps), where this reduces to [`Self::decode_logits_from_embeds`].
    pub fn decode_logits_from_embeds_deepstack(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<Tensor> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            self.dtype,
            &self.device,
        )?;
        let h0 = embeds.to_dtype(self.dtype)?;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            self.layers.len(),
            |i, h| self.layers[i].forward(h, &cos, &sin, &mut cache.layers[i]),
        )?;
        self.project_last(&h)
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
            let ffn = match &cfg.moe {
                // Dense SwiGLU (27B).
                None => Ffn::Dense(Mlp {
                    gate: proj_q(lp("mlp.gate_proj.weight"))?,
                    up: proj_q(lp("mlp.up_proj.weight"))?,
                    down: proj_q(lp("mlp.down_proj.weight"))?,
                }),
                // Sparse MoE (35B-A3B): un-fuse the stacked expert tensors into per-expert SwiGLUs.
                // `experts.gate_up_proj` is [E, 2·moe_inter, hidden] (gate rows ‖ up rows, matching the
                // reference `linear(x, gate_up_proj[e]).chunk(2, -1)`); `experts.down_proj` is
                // [E, hidden, moe_inter].
                Some(moe) => {
                    let mi = moe.moe_intermediate_size as usize;
                    let gate_up = req(lp("mlp.experts.gate_up_proj"))?;
                    let down = req(lp("mlp.experts.down_proj"))?;
                    let mut experts = Vec::with_capacity(moe.num_experts as usize);
                    for e in 0..moe.num_experts as usize {
                        let gu = gate_up.narrow(0, e, 1)?.squeeze(0)?; // [2·mi, hidden]
                        let gate_w = gu.narrow(0, 0, mi)?.contiguous()?;
                        let up_w = gu.narrow(0, mi, mi)?.contiguous()?;
                        let dn = down.narrow(0, e, 1)?.squeeze(0)?.contiguous()?; // [hidden, mi]
                        experts.push(Mlp {
                            gate: Projection::load(gate_w, quant)?,
                            up: Projection::load(up_w, quant)?,
                            down: Projection::load(dn, quant)?,
                        });
                    }
                    Ffn::Moe(MoeFfn {
                        router: req(lp("mlp.gate.weight"))?,
                        experts,
                        shared: Mlp {
                            gate: proj_q(lp("mlp.shared_expert.gate_proj.weight"))?,
                            up: proj_q(lp("mlp.shared_expert.up_proj.weight"))?,
                            down: proj_q(lp("mlp.shared_expert.down_proj.weight"))?,
                        },
                        shared_gate: req(lp("mlp.shared_expert_gate.weight"))?,
                        experts_per_tok: moe.experts_per_tok,
                    })
                }
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

impl KvCache for Qwen35Cache {
    fn offset(&self) -> i32 {
        Qwen35Cache::offset(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|l| match l {
                Qwen35LayerCache::Attn(a) => a.kv.as_ref().map(|(k, _)| k.dims()[0] as i32),
                Qwen35LayerCache::Delta(_) => None,
            })
            .unwrap_or(0)
    }

    fn reset(&mut self) {
        Qwen35Cache::reset(self)
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    // The hybrid cache is driven natively by `Qwen35Model` (which downcasts via `as_any_mut`); the
    // softmax-only trait mutators below are never invoked through the trait object on this path.
    fn update(
        &mut self,
        _layer: usize,
        _keys: &Tensor,
        _values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Err(Error::Msg(
            "Qwen35Cache: generic KvCache::update is not supported (hybrid cache is driven natively)"
                .into(),
        ))
    }

    fn retain_sequences(&mut self, _keep: &[i32]) -> Result<()> {
        Err(Error::Msg(
            "Qwen35Cache: retain_sequences not supported".into(),
        ))
    }

    fn truncate(&mut self, _len: i32) -> Result<()> {
        Err(Error::Msg("Qwen35Cache: truncate not supported".into()))
    }
}

impl crate::decode::Decode for Qwen35Model {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn step(&self, input_ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| Error::Msg("Qwen35Model::step: cache is not a Qwen35Cache".into()))?;
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for Qwen35Model {
    fn embed_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        Qwen35Model::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Tensor,
        input_ids: &[i32],
        vision_features: &Tensor,
        placeholder_tokens: &[i32],
    ) -> Result<Tensor> {
        deepstack::splice_vision_features(embeds, input_ids, vision_features, placeholder_tokens)
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::MropePositions> {
        deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    fn prefill_with_deepstack(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<Tensor> {
        // The hybrid decoder drives its own `Qwen35Cache` (the same downcast `Decode::step` does).
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| Error::Msg("qwen3_5 prefill: expected a Qwen35Cache".into()))?;
        self.decode_logits_from_embeds_deepstack(embeds, positions, cache, visual_pos_mask, deepstack)
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
            match &cfg.moe {
                None => {
                    t(&mut m, &lp("mlp.gate_proj.weight"), &[inter, h]);
                    t(&mut m, &lp("mlp.up_proj.weight"), &[inter, h]);
                    t(&mut m, &lp("mlp.down_proj.weight"), &[h, inter]);
                }
                Some(moe) => {
                    let ne = moe.num_experts as usize;
                    let mi = moe.moe_intermediate_size as usize;
                    let si = moe.shared_expert_intermediate_size as usize;
                    t(&mut m, &lp("mlp.experts.gate_up_proj"), &[ne, 2 * mi, h]);
                    t(&mut m, &lp("mlp.experts.down_proj"), &[ne, h, mi]);
                    t(&mut m, &lp("mlp.gate.weight"), &[ne, h]);
                    t(&mut m, &lp("mlp.shared_expert.gate_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.up_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.down_proj.weight"), &[h, si]);
                    t(&mut m, &lp("mlp.shared_expert_gate.weight"), &[1, h]);
                }
            }
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

    /// A MoE config (`qwen3_5_moe`, the 35B-A3B shape scaled down): 6 experts, top-2, with a shared
    /// expert. Same 4-layer 3:1 mixer schedule as [`cfg_json`].
    fn cfg_json_moe() -> Value {
        let mut v = cfg_json();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("model_type".into(), json!("qwen3_5_moe_text"));
        tc.insert("num_experts".into(), json!(6));
        tc.insert("num_experts_per_tok".into(), json!(2));
        tc.insert("moe_intermediate_size".into(), json!(16));
        tc.insert("shared_expert_intermediate_size".into(), json!(16));
        v
    }

    /// The MoE FFN block, validated against the exact `Qwen3_5MoeSparseMoeBlock.forward` numeric
    /// oracle: softmax router → top-k → renormalize → per-expert SwiGLU (gathered/scattered) →
    /// sigmoid-gated shared expert. Built via the same un-fuse path as the loader. The same
    /// framework-independent fixture as the mlx-llm port (sc-7630); single token so CPU runs exact f32.
    #[test]
    fn moe_ffn_matches_qwen3_5_moe_reference() {
        let json: Value =
            serde_json::from_str(include_str!("testdata/qwen35_moe_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let (h, e, k, mi) = (8usize, 6usize, 2usize, 4usize);
        let mk = |key: &str, dims: &[usize]| {
            Tensor::from_vec(arr(key), dims.to_vec(), &Device::Cpu).unwrap()
        };
        let proj = |a: Tensor| Projection::load(a, None).unwrap();

        // Un-fuse experts.gate_up / down into per-expert SwiGLUs (mirrors the loader).
        let gate_up = mk("gate_up", &[e, 2 * mi, h]);
        let down = mk("down", &[e, h, mi]);
        let mut experts = Vec::new();
        for ei in 0..e {
            let gu = gate_up.narrow(0, ei, 1).unwrap().squeeze(0).unwrap();
            let gate_w = gu.narrow(0, 0, mi).unwrap().contiguous().unwrap();
            let up_w = gu.narrow(0, mi, mi).unwrap().contiguous().unwrap();
            let dn = down
                .narrow(0, ei, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .contiguous()
                .unwrap();
            experts.push(Mlp {
                gate: proj(gate_w),
                up: proj(up_w),
                down: proj(dn),
            });
        }
        let moe = MoeFfn {
            router: mk("router", &[e, h]),
            experts,
            shared: Mlp {
                gate: proj(mk("sh_gate", &[mi, h])),
                up: proj(mk("sh_up", &[mi, h])),
                down: proj(mk("sh_down", &[h, mi])),
            },
            shared_gate: mk("sh_gatew", &[1, h]),
            experts_per_tok: k,
        };

        let out = moe.forward(&mk("x", &[1, 1, h])).unwrap();
        assert_eq!(out.dims(), &[1, 1, h]);
        let got = host(&out);
        let exp = arr("expected_output");
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 2e-4,
            "moe ffn vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}"
        );
    }

    #[test]
    fn moe_model_forward_and_prefill_equals_stepwise() {
        let cfg = Qwen35Config::from_json(&cfg_json_moe()).unwrap();
        assert!(cfg.moe.is_some());
        assert_eq!(cfg.moe.unwrap().num_experts, 6);
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();

        // Multi-token prefill exercises routing/scatter across tokens; logits are finite + shaped.
        let logits = model
            .forward(&ids(&[1, 7, 3, 42, 9]), &mut model.new_cache(), 0)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size as usize]);
        assert!(
            host(&logits).iter().all(|x| x.is_finite()),
            "non-finite MoE logit"
        );

        // Prefill == stepwise decode over the hybrid cache, with the MoE FFN in the loop.
        let pre = model
            .decode_logits(&ids(&[1, 7, 3, 42, 9]), &mut model.new_cache(), 0)
            .unwrap();
        let mut c = model.new_cache();
        let mut last = None;
        for (i, &tok) in [1u32, 7, 3, 42, 9].iter().enumerate() {
            last = Some(model.decode_logits(&ids(&[tok]), &mut c, i as i32).unwrap());
        }
        let (a, b) = (host(&pre), host(&last.unwrap()));
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 1e-4,
            "MoE prefill vs stepwise diverged: max abs diff {md}"
        );
    }

    fn text_model() -> (Qwen35Config, Qwen35Model) {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        (cfg, model)
    }

    /// `mrope_positions` (the `get_rope_index` port) must reproduce the reference 3-D position rows +
    /// `mrope_delta` for an image+text sequence — exact integer index math (oracle gen_mrope.py).
    #[test]
    fn mrope_positions_matches_reference() {
        let j: Value =
            serde_json::from_str(include_str!("testdata/qwen35_mrope_oracle.json")).unwrap();
        let r = &j["rope_index"];
        let ints = |k: &str| -> Vec<i32> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        };
        let toks = ints("input_ids");
        let grid = {
            let g = &r["image_grid_thw"][0];
            vec![[
                g[0].as_i64().unwrap() as i32,
                g[1].as_i64().unwrap() as i32,
                g[2].as_i64().unwrap() as i32,
            ]]
        };
        let img_tok = r["image_token_id"].as_i64().unwrap() as i32;
        let merge = r["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = text_model();
        let (t, h, w, delta) = model.mrope_positions(&toks, &grid, img_tok, merge).unwrap();
        assert_eq!(t, ints("t"));
        assert_eq!(h, ints("h"));
        assert_eq!(w, ints("w"));
        assert_eq!(delta, r["delta"].as_i64().unwrap() as i32);
    }

    /// **The text-path invariant.** Feeding token embeds + equal (text) 3-D positions through
    /// `decode_logits_from_embeds` must be **bit-identical** to the token-id `decode_logits` — the
    /// interleaved M-RoPE collapses to 1D and the embeds path is the same compute. This is the gate
    /// that the multimodal hook doesn't perturb the (verified) text decoder.
    #[test]
    fn decode_from_embeds_text_only_equals_decode_logits() {
        let (_cfg, model) = text_model();
        let toks = [1u32, 7, 3, 42, 9, 2];
        let id_tensor = ids(&toks);

        let a = model
            .decode_logits(&id_tensor, &mut model.new_cache(), 0)
            .unwrap();
        let embeds = model.embed_input_ids(&id_tensor).unwrap();
        let pos: Vec<i32> = (0..toks.len() as i32).collect();
        let b = model
            .decode_logits_from_embeds(&embeds, [&pos, &pos, &pos], &mut model.new_cache())
            .unwrap();

        assert_eq!(a.dims(), b.dims());
        let (av, bv) = (host(&a), host(&b));
        let md = av
            .iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md == 0.0,
            "embeds text path must equal token-id path bit-for-bit; max abs diff {md}"
        );
    }

    /// The splice hook overwrites exactly the image-token rows (in order) with the feature rows and
    /// leaves text rows untouched.
    #[test]
    fn splice_image_features_replaces_image_rows() {
        let (cfg, model) = text_model();
        let hidden = cfg.hidden_size as usize;
        let toks = [7i32, 49, 49, 8, 9]; // two image tokens (id 49) at positions 1,2
                                         // embeds[1,5,hidden]: row r filled with value r.
        let mut e = Vec::new();
        for r in 0..5 {
            e.extend(vec![r as f32; hidden]);
        }
        let embeds = Tensor::from_vec(e, (1, 5, hidden), &Device::Cpu)
            .unwrap()
            .to_dtype(model.compute_dtype())
            .unwrap();
        // feats[2,hidden]: row j filled with 100 + j.
        let mut f = Vec::new();
        for j in 0..2 {
            f.extend(vec![100.0f32 + j as f32; hidden]);
        }
        let feats = Tensor::from_vec(f, (2, hidden), &Device::Cpu).unwrap();

        let out = model
            .splice_image_features(&embeds, &toks, &feats, 49)
            .unwrap();
        assert_eq!(out.dims(), &[1, 5, hidden]);
        let v = host(&out);
        let row = |r: usize| v[r * hidden]; // first element of each row (whole row is constant)
        assert_eq!(
            [row(0), row(1), row(2), row(3), row(4)],
            [0.0, 100.0, 101.0, 3.0, 4.0]
        );
    }

    /// Smoke: the full image+text path (embed → splice features → M-RoPE positions →
    /// decode_logits_from_embeds) runs end to end and yields finite `[1, vocab]` logits.
    #[test]
    fn image_text_decode_from_embeds_runs() {
        let (cfg, model) = text_model();
        let img = 49i32; // within the synthetic vocab (50) so embed gather is in-bounds
        let toks = [1i32, 2, img, img, img, img, 3, 4]; // 2x2 image (4 tokens) between text
        let toks_u32: Vec<u32> = toks.iter().map(|&x| x as u32).collect();
        let grid = vec![[1i32, 4, 4]];
        let id_tensor = ids(&toks_u32);

        let embeds = model.embed_input_ids(&id_tensor).unwrap();
        let feats = Tensor::from_vec(
            (0..4 * cfg.hidden_size)
                .map(|i| (i % 7) as f32 * 0.1 - 0.3)
                .collect::<Vec<_>>(),
            (4, cfg.hidden_size as usize),
            &Device::Cpu,
        )
        .unwrap();
        let spliced = model
            .splice_image_features(&embeds, &toks, &feats, img)
            .unwrap();
        let (t, h, w, _delta) = model.mrope_positions(&toks, &grid, img, 2).unwrap();
        let logits = model
            .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
            .unwrap();
        assert_eq!(logits.dims(), &[1, cfg.vocab_size as usize]);
        assert!(host(&logits).iter().all(|x| x.is_finite()));
    }
}
