//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) — the hybrid decoder (story sc-7632,
//! the candle mirror of mlx-llm sc-7628/7629).
//!
//! Unlike the generic [`CausalLm`](super::llama::CausalLm) (all softmax attention over a growing KV
//! cache), this decoder interleaves two mixer types on a fixed schedule (`full_attention_interval`,
//! default 4 → **3 Gated DeltaNet linear-attention layers : 1 gated full-attention layer**):
//!
//! - `GatedDeltaNet` — linear attention carrying a fixed-size recurrent state (the verified
//!   primitives in [`crate::primitives::gated_delta`]): a 4-way in-projection → short conv → q/k
//!   L2-norm → gated delta recurrence → gated RMS-norm → out-proj.
//! - `Qwen35Attention` — grouped-query attention with **partial RoPE** (`partial_rotary_factor`,
//!   reusing the [`Rope::partial`] path), per-head q/k RMSNorm, and an **output gate** (the queries
//!   projection is doubled into `[queries ‖ gate]`, and the attended output is multiplied by
//!   `sigmoid(gate)` before the output projection).
//!
//! Each decoder layer is `input_layernorm → mixer → residual → post_attention_layernorm → MLP →
//! residual`. The MLP is a dense SwiGLU (the 27B) or a sparse Mixture-of-Experts bank (`MoeFfn`, the
//! 35B-A3B). The KV cache (full-attn layers) and the recurrent [`DeltaNetCache`] (linear
//! layers) live side by side in a per-layer [`Qwen35Cache`]. RMSNorm weights follow the Qwen3-Next
//! `(1 + weight)` convention; the recurrence accumulates in f32 (matching the reference GPU kernel)
//! while the rest of the decoder runs in the device compute dtype (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;
use serde_json::Value;

use crate::decode::step::{LogitsScope, StepModel, StepOutput, StepRequest};
use crate::device::compute_dtype;
use crate::error::{Error, Result};
use crate::models::deepstack::{self, deepstack_fused_decoder_layers};
use crate::primitives::attention::{repeat_kv, sdpa, AttnMask};
use crate::primitives::decode_cache::{tensor_bytes, CacheMemory, DecodeCache};
use crate::primitives::gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::nn::{embed, input_ids, rms_norm, silu};
use crate::primitives::projection::{Projection, ProjectionFormat, QuantSpec, WeightCensus};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{KvCache, PrismRegistry, Weights};

fn checkpoint_norm_weight(weight: Tensor, prism: bool) -> Result<Tensor> {
    if prism {
        Ok(weight)
    } else {
        Ok(weight.affine(1.0, 1.0)?)
    }
}

#[derive(Clone)]
enum QwenEmbedding {
    Dense(Tensor),
    Prism(std::sync::Arc<crate::primitives::PrismPackedWeight>),
}

impl QwenEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(weight) => embed(weight, ids),
            Self::Prism(weight) => weight.embedding(ids),
        }
    }
}

/// Interleaved M-RoPE output of [`Qwen35Model::mrope_positions`]: the temporal / height / width
/// position rows (each length `S`) plus the `mrope_delta` (`max_position + 1 − len`) for continuing
/// positions after the prompt.
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// Mixture-of-Experts FFN parameters (`qwen3_5_moe`, the 35B-A3B): the routed-expert count / top-k
/// and the per-expert + shared-expert FFN widths that drive the un-fused `MoeFfn`.
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
    /// Number of checkpoint-native multi-token-prediction layers. Zero means the checkpoint has no
    /// auxiliary MTP predictor and ordinary autoregressive decoding remains fully valid.
    pub mtp_num_hidden_layers: usize,
    /// Whether the MTP predictor owns an embedding table rather than sharing the target embedding.
    /// Qwen3.8-27B publishes `false` and carries no `mtp.embed_tokens.weight` tensor.
    pub mtp_use_dedicated_embeddings: bool,
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
            mtp_num_hidden_layers: int("mtp_num_hidden_layers").unwrap_or(0).max(0) as usize,
            mtp_use_dedicated_embeddings: c
                .get("mtp_use_dedicated_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
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
    fn record(&self, census: &mut WeightCensus) {
        for p in [
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_a,
            &self.in_proj_b,
            &self.out_proj,
        ] {
            census.projections.record(p);
        }
        for t in [
            &self.conv_weight,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
        ] {
            census.record_tensor(t);
        }
    }

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
    fn record(&self, census: &mut WeightCensus) {
        for p in [&self.gate, &self.up, &self.down] {
            census.projections.record(p);
        }
    }

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
        crate::primitives::host_sync::note_host_sync();
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
    fn record(&self, census: &mut WeightCensus) {
        census.record_tensor(&self.input_ln);
        census.record_tensor(&self.post_ln);
        match &self.mixer {
            Mixer::Delta(d) => d.record(census),
            Mixer::Attn(a) => {
                for p in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] {
                    census.projections.record(p);
                }
                census.record_tensor(&a.q_norm);
                census.record_tensor(&a.k_norm);
            }
        }
        match &self.ffn {
            Ffn::Dense(m) => m.record(census),
            Ffn::Moe(moe) => {
                census.record_tensor(&moe.router);
                census.record_tensor(&moe.shared_gate);
                moe.shared.record(census);
                for e in &moe.experts {
                    e.record(census);
                }
            }
        }
    }
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

    /// Keep positions `0..len` along the sequence axis (the growing-KV half of a rollback). The
    /// narrowed views are made contiguous so the next append copies only what is kept.
    fn truncate(&mut self, len: i32) -> Result<()> {
        let Some((k, v)) = self.kv.take() else {
            return Ok(());
        };
        let cur = k.dims()[2];
        let len = usize::try_from(len).map_err(|_| Error::Msg("AttnKv: negative length".into()))?;
        if len > cur {
            return Err(Error::Msg(format!(
                "AttnKv: cannot truncate to {len} past {cur} cached positions"
            )));
        }
        if len == 0 {
            return Ok(());
        }
        if len == cur {
            self.kv = Some((k, v));
            return Ok(());
        }
        self.kv = Some((
            k.narrow(2, 0, len)?.contiguous()?,
            v.narrow(2, 0, len)?.contiguous()?,
        ));
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.kv
            .as_ref()
            .map(|(k, v)| tensor_bytes(k).saturating_add(tensor_bytes(v)))
            .unwrap_or(0)
    }
}

fn delta_bytes(c: &DeltaNetCache) -> usize {
    c.conv_state
        .as_ref()
        .map(tensor_bytes)
        .unwrap_or(0)
        .saturating_add(c.ssm_state.as_ref().map(tensor_bytes).unwrap_or(0))
}

/// The recurrent (Gated DeltaNet) state of every linear layer at one sequence position — what a
/// rollback restores, because the recurrence cannot be inverted. Only the linear layers are held
/// (the full-attention KV rolls back by narrowing); the tensors are reference-counted clones, so a
/// checkpoint costs no device memory until the live state moves on.
#[derive(Clone, Debug)]
struct DeltaCheckpoint {
    offset: i32,
    /// `(layer index, state)` for each linear layer, in layer order.
    states: Vec<(usize, DeltaNetCache)>,
}

/// Rollback checkpoints retained by a cache built through [`Qwen35Model::new_cache`] (and so
/// `Decode::make_cache`) — the reference and MTP paths the provider runs. Neither calls
/// [`Qwen35Cache::rollback_to`] (the MTP loop restores a clone), so they retain **none**: such a
/// cache holds exactly one recurrent state, which is what request admission charges (E6). The
/// provider's `memory_geometry` prices `1 + REFERENCE_MAX_CHECKPOINTS` recurrent states, so raising
/// this constant raises admission with it.
pub const REFERENCE_MAX_CHECKPOINTS: usize = 0;

/// Rollback checkpoints retained by a cache built through the [`StepModel`] seam
/// ([`StepModel::new_cache`]) — the callers that roll back (the speculative verify of S2): the
/// position the current step started from plus one earlier. Each checkpoint pins every linear
/// layer's recurrent state — on Qwen3.8-27B ~151 MB of SSM state (48 linear layers × 48 value heads
/// × 128 × 128 × 4 B) plus a ~6 MB conv tail — so at steady state a step cache holds three recurrent
/// states (live + two checkpoints). A caller that admits step-seam requests must price
/// `1 + STEP_MAX_CHECKPOINTS` states; deeper history is opt-in via
/// [`Qwen35Cache::set_max_checkpoints`].
pub const STEP_MAX_CHECKPOINTS: usize = 2;

/// The per-layer cache slot — a recurrent [`DeltaNetCache`] for linear layers, growing KV for
/// full-attention layers.
#[derive(Clone, Debug)]
pub enum Qwen35LayerCache {
    Delta(DeltaNetCache),
    Attn(AttnKv),
}

/// The hybrid decoder's cache: one slot per decoder layer, plus the rollback checkpoints that make
/// it a [`DecodeCache`].
///
/// The decoder records a [`checkpoint`](Self::checkpoint) of the linear layers' recurrent state at
/// the start of every forward (i.e. at every position a step began from), keeping the newest
/// [`max_checkpoints`](Self::set_max_checkpoints). [`rollback_to`](Self::rollback_to) can therefore
/// return to any of those positions exactly — which covers what speculative decoding needs (roll
/// back to the position the verify step started from) — and refuses positions it has no checkpoint
/// for rather than approximating them. Per-position checkpoints inside a multi-token forward are the
/// DeltaNet-checkpoint story (S3) and plug in behind the same method.
///
/// Retention depends on who built the cache: [`Qwen35Model::new_cache`] (reference / MTP paths)
/// keeps [`REFERENCE_MAX_CHECKPOINTS`] (none), [`StepModel::new_cache`] keeps
/// [`STEP_MAX_CHECKPOINTS`].
#[derive(Clone, Debug)]
pub struct Qwen35Cache {
    layers: Vec<Qwen35LayerCache>,
    checkpoints: Vec<DeltaCheckpoint>,
    max_checkpoints: usize,
}

impl Qwen35Cache {
    /// Positions already cached — the RoPE offset for the next step (read from the first full-attn
    /// layer, or the first linear layer when the schedule has no full-attention layer; all layers
    /// advance in lockstep).
    pub fn offset(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|l| match l {
                Qwen35LayerCache::Attn(a) => Some(a.offset()),
                Qwen35LayerCache::Delta(_) => None,
            })
            .or_else(|| {
                self.layers.iter().find_map(|l| match l {
                    Qwen35LayerCache::Delta(c) => Some(c.offset()),
                    Qwen35LayerCache::Attn(_) => None,
                })
            })
            .unwrap_or(0)
    }

    /// Drop all cached state and every checkpoint.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            match l {
                Qwen35LayerCache::Delta(c) => c.reset(),
                Qwen35LayerCache::Attn(a) => a.kv = None,
            }
        }
        self.checkpoints.clear();
    }

    /// Change how many rollback checkpoints are retained (newest kept; `0` disables rollback to
    /// anything but the current position and zero).
    pub fn set_max_checkpoints(&mut self, max: usize) {
        self.max_checkpoints = max;
        self.prune_checkpoints();
    }

    /// How many rollback checkpoints this cache retains.
    pub fn max_checkpoints(&self) -> usize {
        self.max_checkpoints
    }

    /// Logical bytes of recurrent (Gated DeltaNet) state referenced by the cache: the live linear
    /// layers' states plus every checkpoint's. The attention KV is excluded — this is the term
    /// admission prices as `recurrent_bytes` (see `REFERENCE_MAX_CHECKPOINTS`).
    pub fn recurrent_bytes(&self) -> usize {
        let live = self.layers.iter().fold(0usize, |acc, l| match l {
            Qwen35LayerCache::Delta(c) => acc.saturating_add(delta_bytes(c)),
            Qwen35LayerCache::Attn(_) => acc,
        });
        live.saturating_add(self.memory().checkpoint_bytes)
    }

    /// The positions [`rollback_to`](Self::rollback_to) can currently return to, ascending
    /// (`0` and the current position are always possible and not listed).
    pub fn checkpoint_offsets(&self) -> Vec<i32> {
        self.checkpoints.iter().map(|c| c.offset).collect()
    }

    /// Record the linear layers' recurrent state at the current position. Idempotent for a
    /// position already checkpointed as the newest entry. Called by the decoder at the start of
    /// every forward.
    pub fn checkpoint(&mut self) {
        let offset = self.offset();
        if self.max_checkpoints == 0 {
            return;
        }
        if self.checkpoints.last().is_some_and(|c| c.offset == offset) {
            return;
        }
        // A forward at an earlier position than a stored checkpoint means the cache was rolled back
        // through a path other than `rollback_to` (a clone restored by the MTP loop); those later
        // checkpoints describe positions that no longer exist.
        self.checkpoints.retain(|c| c.offset < offset);
        let states = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(i, l)| match l {
                Qwen35LayerCache::Delta(c) => Some((i, c.clone())),
                Qwen35LayerCache::Attn(_) => None,
            })
            .collect();
        self.checkpoints.push(DeltaCheckpoint { offset, states });
        self.prune_checkpoints();
    }

    fn prune_checkpoints(&mut self) {
        if self.checkpoints.len() > self.max_checkpoints {
            let drop = self.checkpoints.len() - self.max_checkpoints;
            self.checkpoints.drain(..drop);
        }
    }

    /// Roll the cache back so the next step continues from position `n` (see
    /// [`DecodeCache::rollback_to`]): the full-attention KV is narrowed to `n` and the linear
    /// layers' recurrent state is restored from the checkpoint taken at `n`. `n == offset()` is a
    /// no-op and `n == 0` is a [`reset`](Self::reset); any other `n` without a checkpoint is
    /// [`Error::RollbackUnavailable`] (typed, so a speculative engine can fall back without matching
    /// message text) and leaves the cache untouched. `n` outside `0..=offset()` is [`Error::Msg`].
    pub fn rollback_to(&mut self, n: i32) -> Result<()> {
        let cur = self.offset();
        if n < 0 || n > cur {
            return Err(Error::Msg(format!(
                "Qwen35Cache: cannot roll back to {n} with {cur} positions cached"
            )));
        }
        if n == cur {
            return Ok(());
        }
        if n == 0 {
            self.reset();
            return Ok(());
        }
        let idx = self
            .checkpoints
            .iter()
            .rposition(|c| c.offset == n)
            .ok_or_else(|| Error::RollbackUnavailable {
                n,
                have: self.checkpoint_offsets(),
            })?;
        let checkpoint = self.checkpoints[idx].clone();
        for (i, state) in checkpoint.states {
            match &mut self.layers[i] {
                Qwen35LayerCache::Delta(c) => *c = state,
                Qwen35LayerCache::Attn(_) => {
                    return Err(Error::Msg(
                        "Qwen35Cache: checkpoint/layer type mismatch".into(),
                    ))
                }
            }
        }
        for l in &mut self.layers {
            if let Qwen35LayerCache::Attn(a) = l {
                a.truncate(n)?;
            }
        }
        // The checkpoint at `n` still describes the (restored) state at `n`; later ones do not.
        self.checkpoints.truncate(idx + 1);
        Ok(())
    }

    /// Logical bytes referenced by the live state and by the checkpoints (see
    /// [`CacheMemory`] for what is and is not counted).
    pub fn memory(&self) -> CacheMemory {
        let live_bytes = self.layers.iter().fold(0usize, |acc, l| {
            acc.saturating_add(match l {
                Qwen35LayerCache::Delta(c) => delta_bytes(c),
                Qwen35LayerCache::Attn(a) => a.bytes(),
            })
        });
        let checkpoint_bytes = self
            .checkpoints
            .iter()
            .flat_map(|c| c.states.iter())
            .fold(0usize, |acc, (_, s)| acc.saturating_add(delta_bytes(s)));
        CacheMemory {
            live_bytes,
            checkpoint_bytes,
        }
    }
}

impl DecodeCache for Qwen35Cache {
    fn len(&self) -> i32 {
        self.offset()
    }

    fn rollback_to(&mut self, n: i32) -> Result<()> {
        Qwen35Cache::rollback_to(self, n)
    }

    fn reset(&mut self) {
        Qwen35Cache::reset(self)
    }

    fn memory(&self) -> CacheMemory {
        Qwen35Cache::memory(self)
    }
}

/// A loaded Qwen3.6 (`qwen3_5`) hybrid decoder.
pub struct Qwen35Model {
    embed_tokens: QwenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    lm_head: std::sync::Arc<Projection>,
    rope: Rope,
    cfg: Qwen35Config,
    eps: f64,
    dtype: DType,
    device: Device,
    quantized: bool,
}

/// The checkpoint-native Qwen3.8 multi-token predictor.
///
/// This is an auxiliary draft model only: the target decoder never consults it during ordinary
/// autoregressive inference. Qwen3.8 shares the target token embedding and LM head, then applies
/// `fc([RMS(embed(next_token)), RMS(previous_hidden)])`, one full-attention decoder layer, a final
/// RMS norm, and the shared head. vLLM cycles the published layer when more drafts are requested
/// than the checkpoint's layer count, so `step_idx` selects `layers[step_idx % layers.len()]`.
pub struct Qwen35Mtp {
    embed_tokens: QwenEmbedding,
    lm_head: std::sync::Arc<Projection>,
    pre_fc_norm_embedding: Tensor,
    pre_fc_norm_hidden: Tensor,
    fc: Projection,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    rope: Rope,
    mrope_section: [usize; 3],
    eps: f64,
    dtype: DType,
    device: Device,
    vocab_size: usize,
}

/// Persistent full-attention state for each published MTP layer.
#[derive(Clone, Debug)]
pub struct Qwen35MtpCache {
    layers: Vec<AttnKv>,
}

impl Qwen35MtpCache {
    /// Clear every MTP layer's speculative attention state.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.kv = None;
        }
    }
}

impl Qwen35Mtp {
    /// All tensors required by the frozen Qwen3.8 MTP layout. A checkpoint is advertised as MTP
    /// capable only when every key is present; a partial auxiliary head is never used.
    pub fn required_keys(num_layers: usize) -> Vec<String> {
        let mut keys = vec![
            "mtp.fc.weight".to_string(),
            "mtp.norm.weight".to_string(),
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            "mtp.pre_fc_norm_hidden.weight".to_string(),
        ];
        for i in 0..num_layers {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                keys.push(format!("mtp.layers.{i}.{suffix}"));
            }
        }
        keys
    }

    /// Whether the complete configured MTP tensor set is present.
    pub fn complete_in(w: &Weights, cfg: &Qwen35Config) -> bool {
        cfg.mtp_num_hidden_layers == 1
            && !cfg.mtp_use_dedicated_embeddings
            && Self::required_keys(cfg.mtp_num_hidden_layers)
                .iter()
                .all(|key| w.contains(key))
    }

    /// Load the native MTP predictor, sharing the already loaded target embedding and LM head.
    pub fn from_weights_with(
        w: &Weights,
        target: &Qwen35Model,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let format = quant.map(ProjectionFormat::from);
        Self::from_weights_format(w, target, format.as_ref())
    }

    /// The predictor's own resident weights (its `fc`, layers and norms). The token embedding and
    /// LM head are the target's, shared by `Arc`, and are counted by
    /// [`Qwen35Model::weight_census`] only.
    pub fn weight_census(&self) -> WeightCensus {
        let mut census = WeightCensus::default();
        census.projections.record(&self.fc);
        for t in [
            &self.pre_fc_norm_embedding,
            &self.pre_fc_norm_hidden,
            &self.norm,
        ] {
            census.record_tensor(t);
        }
        for layer in &self.layers {
            layer.record(&mut census);
        }
        census
    }

    /// [`Self::from_weights_with`] for any [`ProjectionFormat`] (NVFP4 included, sc-24135): the
    /// predictor's projections are stored exactly like the target's.
    pub fn from_weights_format(
        w: &Weights,
        target: &Qwen35Model,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        let cfg = &target.cfg;
        if cfg.mtp_num_hidden_layers != 1 {
            return Err(Error::Config(
                "the native Qwen3.8 MTP layout requires exactly one stored predictor layer".into(),
            ));
        }
        if cfg.mtp_use_dedicated_embeddings {
            return Err(Error::Config(
                "qwen3_5 dedicated MTP embeddings are not represented by the frozen Qwen3.8 layout"
                    .into(),
            ));
        }
        let missing: Vec<String> = Self::required_keys(cfg.mtp_num_hidden_layers)
            .into_iter()
            .filter(|key| !w.contains(key))
            .collect();
        if !missing.is_empty() {
            return Err(Error::Msg(format!(
                "qwen3_5 MTP tensor set is incomplete; missing {}",
                missing.join(", ")
            )));
        }
        if cfg.moe.is_some() {
            return Err(Error::Config(
                "qwen3_5 MoE MTP loading requires the checkpoint's sparse MTP FFN layout".into(),
            ));
        }

        let dtype = target.dtype;
        let eps = cfg.rms_norm_eps as f64;
        let req = |key: &str| -> Result<Tensor> { Ok(w.require(key)?.to_dtype(dtype)?) };
        // Qwen3.5/Qwen3.8 RMSNorm parameters are zero-centered (`1 + weight`).
        let norm_w = |key: &str| -> Result<Tensor> { Ok(req(key)?.affine(1.0, 1.0)?) };
        let proj_q =
            |key: &str| -> Result<Projection> { Projection::load_as(req(key)?, None, format) };
        let groups = (cfg.num_heads / cfg.num_kv_heads) as usize;
        let mut layers = Vec::with_capacity(cfg.mtp_num_hidden_layers);
        for i in 0..cfg.mtp_num_hidden_layers {
            let lp = |suffix: &str| format!("mtp.layers.{i}.{suffix}");
            layers.push(DecoderLayer {
                input_ln: norm_w(&lp("input_layernorm.weight"))?,
                post_ln: norm_w(&lp("post_attention_layernorm.weight"))?,
                mixer: Mixer::Attn(Qwen35Attention {
                    q_proj: proj_q(&lp("self_attn.q_proj.weight"))?,
                    k_proj: proj_q(&lp("self_attn.k_proj.weight"))?,
                    v_proj: proj_q(&lp("self_attn.v_proj.weight"))?,
                    o_proj: proj_q(&lp("self_attn.o_proj.weight"))?,
                    q_norm: norm_w(&lp("self_attn.q_norm.weight"))?,
                    k_norm: norm_w(&lp("self_attn.k_norm.weight"))?,
                    num_heads: cfg.num_heads as usize,
                    num_kv_heads: cfg.num_kv_heads as usize,
                    head_dim: cfg.head_dim as usize,
                    groups,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    eps,
                }),
                ffn: Ffn::Dense(Mlp {
                    gate: proj_q(&lp("mlp.gate_proj.weight"))?,
                    up: proj_q(&lp("mlp.up_proj.weight"))?,
                    down: proj_q(&lp("mlp.down_proj.weight"))?,
                }),
                eps,
            });
        }

        Ok(Self {
            embed_tokens: target.embed_tokens.clone(),
            lm_head: target.lm_head.clone(),
            pre_fc_norm_embedding: norm_w("mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: norm_w("mtp.pre_fc_norm_hidden.weight")?,
            fc: proj_q("mtp.fc.weight")?,
            layers,
            norm: norm_w("mtp.norm.weight")?,
            rope: Rope::partial(cfg.rotary_dim(), cfg.rope_theta, false),
            mrope_section: cfg.mrope_section_resolved(),
            eps,
            dtype,
            device: target.device.clone(),
            vocab_size: cfg.vocab_size as usize,
        })
    }

    /// A fresh cache for the auxiliary full-attention layers.
    pub fn new_cache(&self) -> Qwen35MtpCache {
        Qwen35MtpCache {
            layers: (0..self.layers.len()).map(|_| AttnKv::default()).collect(),
        }
    }

    /// Run one autoregressive draft step.
    ///
    /// `input_id` is the token one position to the right of `previous_hidden`; `position` is that
    /// token's absolute text position. The returned hidden state is fed into the next speculative
    /// step, while the logits define the proposal distribution `q` used by exact sampling
    /// acceptance.
    pub fn step(
        &self,
        input_id: i32,
        previous_hidden: &Tensor,
        step_idx: usize,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        if self.layers.is_empty() {
            return Err(Error::Msg("qwen3_5 MTP has no predictor layers".into()));
        }
        let ids = Tensor::from_vec(vec![input_id as i64], (1, 1), &self.device)?;
        let embeddings = self.embed_tokens.forward(&ids)?.to_dtype(self.dtype)?;
        self.step_from_embeddings(&embeddings, previous_hidden, step_idx, position, cache)
    }

    /// Embedding-input twin of [`Self::step`], used when the shifted MTP input row is a fused visual
    /// embedding rather than the image/video placeholder token embedding.
    pub fn step_from_embeddings(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        step_idx: usize,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        let (logits, hidden) = self.forward_sequence_from_embeddings(
            embeddings,
            previous_hidden,
            step_idx,
            position,
            cache,
        )?;
        Ok((
            logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .reshape((1, self.vocab_size))?,
            hidden.narrow(1, hidden.dim(1)? - 1, 1)?,
        ))
    }

    /// Process a contiguous sequence of target-validated token/hidden pairs. This is used both for
    /// prompt prefill and to replace speculative MTP state with target-confirmed state after
    /// acceptance. The frozen Qwen3.8 checkpoint has one MTP layer, so the whole sequence advances
    /// that layer in one causal forward.
    pub fn forward_sequence(
        &self,
        input_ids: &[i32],
        previous_hidden: &Tensor,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        if input_ids.is_empty() {
            return Err(Error::Msg(
                "qwen3_5 MTP sequence input must not be empty".into(),
            ));
        }
        let ids: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let ids = Tensor::from_vec(ids, (1, input_ids.len()), &self.device)?;
        let embeddings = self.embed_tokens.forward(&ids)?.to_dtype(self.dtype)?;
        self.forward_sequence_from_embeddings(&embeddings, previous_hidden, 0, position, cache)
    }

    /// Advance the predictor cache over validated prompt/replay pairs without projecting unused
    /// all-position vocabulary logits.
    pub fn warm_sequence(
        &self,
        input_ids: &[i32],
        previous_hidden: &Tensor,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<()> {
        if input_ids.is_empty() {
            return Err(Error::Msg(
                "qwen3_5 MTP sequence input must not be empty".into(),
            ));
        }
        let ids: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let ids = Tensor::from_vec(ids, (1, input_ids.len()), &self.device)?;
        let embeddings = self.embed_tokens.forward(&ids)?.to_dtype(self.dtype)?;
        self.sequence_hidden_from_embeddings(&embeddings, previous_hidden, 0, position, cache)?;
        Ok(())
    }

    /// Process target-validated fused prompt embeddings with the prompt's explicit interleaved
    /// M-RoPE positions. Qwen3.8 is multimodal, so shifted vision rows must remain vision features
    /// rather than being re-embedded from their placeholder token ids.
    pub fn forward_embeddings_mrope(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.mrope_section,
            self.dtype,
            &self.device,
        )?;
        self.forward_sequence_with_rope(embeddings, previous_hidden, 0, &cos, &sin, cache)
    }

    /// Visual-prompt cache warmup with the same M-RoPE positions and fused embeddings.
    pub fn warm_embeddings_mrope(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35MtpCache,
    ) -> Result<()> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.mrope_section,
            self.dtype,
            &self.device,
        )?;
        self.sequence_hidden_with_rope(embeddings, previous_hidden, 0, &cos, &sin, cache)?;
        Ok(())
    }

    fn forward_sequence_from_embeddings(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        step_idx: usize,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.sequence_hidden_from_embeddings(
            embeddings,
            previous_hidden,
            step_idx,
            position,
            cache,
        )?;
        Ok((self.lm_head.forward(&hidden)?, hidden))
    }

    fn sequence_hidden_from_embeddings(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        step_idx: usize,
        position: i32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<Tensor> {
        let seq = embeddings.dim(1)?;
        if previous_hidden.dim(1)? != seq {
            return Err(Error::Msg(format!(
                "qwen3_5 MTP embedding/hidden sequence mismatch: {} != {}",
                seq,
                previous_hidden.dim(1)?
            )));
        }
        let (cos, sin) = self
            .rope
            .cos_sin(seq as i32, position, self.dtype, &self.device)?;
        self.sequence_hidden_with_rope(embeddings, previous_hidden, step_idx, &cos, &sin, cache)
    }

    fn forward_sequence_with_rope(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        step_idx: usize,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Qwen35MtpCache,
    ) -> Result<(Tensor, Tensor)> {
        let hidden =
            self.sequence_hidden_with_rope(embeddings, previous_hidden, step_idx, cos, sin, cache)?;
        Ok((self.lm_head.forward(&hidden)?, hidden))
    }

    fn sequence_hidden_with_rope(
        &self,
        embeddings: &Tensor,
        previous_hidden: &Tensor,
        step_idx: usize,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Qwen35MtpCache,
    ) -> Result<Tensor> {
        let en = rms_norm(embeddings, &self.pre_fc_norm_embedding, self.eps)?;
        let hn = rms_norm(previous_hidden, &self.pre_fc_norm_hidden, self.eps)?;
        let fused = self.fc.forward(&Tensor::cat(&[&en, &hn], 2)?)?;
        let layer_idx = step_idx % self.layers.len();
        let mut slot = Qwen35LayerCache::Attn(cache.layers[layer_idx].clone());
        let hidden = self.layers[layer_idx].forward(&fused, cos, sin, &mut slot)?;
        let Qwen35LayerCache::Attn(updated) = slot else {
            unreachable!("MTP layers are always full attention")
        };
        cache.layers[layer_idx] = updated;
        let hidden = rms_norm(&hidden, &self.norm, self.eps)?;
        Ok(hidden)
    }
}

impl Qwen35Model {
    /// The parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// The resident weight set by projection kind (sc-24135 load telemetry): every projection the
    /// decoder holds — including the ones a format leaves dense (`in_proj_a/b`, the MoE router is a
    /// plain tensor) — plus embeddings, norms and recurrent parameters under `other`.
    pub fn weight_census(&self) -> WeightCensus {
        let mut census = WeightCensus::default();
        // A dense tied head *is* the embedding tensor; count the storage once.
        let head_is_embedding =
            self.cfg.tie_word_embeddings && matches!(*self.lm_head, Projection::Dense(_));
        match &self.embed_tokens {
            QwenEmbedding::Dense(t) if !head_is_embedding => census.record_tensor(t),
            QwenEmbedding::Dense(_) => {}
            QwenEmbedding::Prism(p) => {
                census.record_unmeasured((p.rows() * p.input_width()) as u64)
            }
        }
        census.projections.record(&self.lm_head);
        census.record_tensor(&self.norm);
        for layer in &self.layers {
            layer.record(&mut census);
        }
        census
    }

    /// Whether the large projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The compute dtype (bf16 on GPU, f32 on CPU).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// A fresh per-layer cache (linear vs full-attn slot per the schedule) for the reference and MTP
    /// paths: it retains [`REFERENCE_MAX_CHECKPOINTS`] (no) rollback checkpoints.
    pub fn new_cache(&self) -> Qwen35Cache {
        self.new_cache_with_checkpoints(REFERENCE_MAX_CHECKPOINTS)
    }

    /// A fresh cache that retains up to `max_checkpoints` rollback checkpoints (see
    /// [`Qwen35Cache::rollback_to`]); each costs one recurrent state of device memory.
    pub fn new_cache_with_checkpoints(&self, max_checkpoints: usize) -> Qwen35Cache {
        let layers = (0..self.cfg.num_layers)
            .map(|i| {
                if self.cfg.is_linear(i) {
                    Qwen35LayerCache::Delta(DeltaNetCache::new())
                } else {
                    Qwen35LayerCache::Attn(AttnKv::default())
                }
            })
            .collect();
        Qwen35Cache {
            layers,
            checkpoints: Vec::new(),
            max_checkpoints,
        }
    }

    /// The device the model's tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run the decoder stack over `input_ids` `[B, S]` at sequence `offset`, returning the final
    /// hidden states `[B, S, hidden]` (before the final norm / lm_head).
    fn hidden(&self, input_ids: &Tensor, cache: &mut Qwen35Cache, offset: i32) -> Result<Tensor> {
        let h = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
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
        cache.checkpoint();
        let mut h = embeds.clone();
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, cos, sin, slot)?;
        }
        Ok(h)
    }

    /// Final target RMSNorm. Its output is both the LM-head input and the target hidden state paired
    /// with the next token embedding by Qwen3.8's MTP predictor.
    fn normalize(&self, h: &Tensor) -> Result<Tensor> {
        rms_norm(h, &self.norm, self.eps)
    }

    /// Shared `lm_head` projection over already normalized hidden states.
    fn project_normalized(&self, h: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(h)
    }

    /// Final RMSNorm + `lm_head` over hidden states `[B, n, hidden]` → logits `[B, n, vocab]`.
    fn project(&self, h: &Tensor) -> Result<Tensor> {
        self.project_normalized(&self.normalize(h)?)
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

    /// Target verification output for Qwen3.8 MTP: logits and final-normalized target hidden states
    /// for every input position. Both tensors retain the `[B, S, ...]` sequence axis.
    pub fn forward_with_hidden(
        &self,
        input_ids: &Tensor,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.normalize(&self.hidden(input_ids, cache, offset)?)?;
        let logits = self.project_normalized(&hidden)?;
        Ok((logits, hidden))
    }

    /// MTP prompt prefill keeps every normalized hidden row for predictor alignment, while only
    /// the final row needs target logits for the first sample.
    pub fn prefill_with_hidden(
        &self,
        input_ids: &Tensor,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.normalize(&self.hidden(input_ids, cache, offset)?)?;
        let (b, s, _) = hidden.dims3()?;
        let last = hidden.narrow(1, s - 1, 1)?.contiguous()?;
        let logits = self
            .project_normalized(&last)?
            .reshape((b, self.cfg.vocab_size as usize))?;
        Ok((logits, hidden))
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
        Ok(self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?)
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
        cache.checkpoint();
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            self.layers.len(),
            |i, h| self.layers[i].forward(h, &cos, &sin, &mut cache.layers[i]),
        )?;
        self.project_last(&h)
    }

    /// Multimodal target prefill for MTP: the ordinary all-position logits plus the final-normalized
    /// hidden row for every fused prompt position. DeepStack fusion and M-RoPE match the regular
    /// multimodal prefill exactly.
    pub fn forward_from_embeds_deepstack_with_hidden(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            self.dtype,
            &self.device,
        )?;
        let h0 = embeds.to_dtype(self.dtype)?;
        cache.checkpoint();
        let hidden = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            self.layers.len(),
            |i, h| self.layers[i].forward(h, &cos, &sin, &mut cache.layers[i]),
        )?;
        let hidden = self.normalize(&hidden)?;
        let logits = self.project_normalized(&hidden)?;
        Ok((logits, hidden))
    }

    /// Multimodal MTP prefill with DeepStack/M-RoPE and only the last target logit row.
    pub fn prefill_from_embeds_deepstack_with_hidden(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            self.dtype,
            &self.device,
        )?;
        let h0 = embeds.to_dtype(self.dtype)?;
        cache.checkpoint();
        let hidden = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            self.layers.len(),
            |i, h| self.layers[i].forward(h, &cos, &sin, &mut cache.layers[i]),
        )?;
        let hidden = self.normalize(&hidden)?;
        let (b, s, _) = hidden.dims3()?;
        let last = hidden.narrow(1, s - 1, 1)?.contiguous()?;
        let logits = self
            .project_normalized(&last)?
            .reshape((b, self.cfg.vocab_size as usize))?;
        Ok((logits, hidden))
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

    /// Build from a loaded checkpoint storing the large projections in `format` (`None` = dense;
    /// NVFP4 included, sc-24135) at the device's compute dtype.
    pub fn from_weights_format(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        Self::from_weights_dtype_impl(w, prefix, cfg, format, compute_dtype(w.device()), None)
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
        let format = quant.map(ProjectionFormat::from);
        Self::from_weights_dtype_impl(w, prefix, cfg, format.as_ref(), dtype, None)
    }

    /// Build Qwen3.5/3.8 with compact Prism matrices while retaining ordinary tensors for norms,
    /// convolution and recurrent parameters. Registry keys are the exact checkpoint tensor names.
    pub fn from_prism_weights(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        prism: &PrismRegistry,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_weights_dtype_impl(w, prefix, cfg, None, dtype, Some(prism))
    }

    fn from_weights_dtype_impl(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        format: Option<&ProjectionFormat>,
        dtype: DType,
        prism: Option<&PrismRegistry>,
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
        // Dense HF Qwen3.6 norms are zero-centered. Frozen Prism/Bonsai artifacts have already
        // converted every ordinary RMSNorm tensor to its direct multiplier and must remain raw.
        let norm_w = |key: String| -> Result<Tensor> {
            let weight = req(key)?;
            checkpoint_norm_weight(weight, prism.is_some())
        };
        let proj_q = |key: String| -> Result<Projection> {
            match prism.and_then(|registry| registry.get(&key)) {
                Some(weight) => Ok(Projection::load_prism(weight.clone())),
                None => Projection::load_as(req(key)?, None, format),
            }
        };
        let proj_dense = |key: String| -> Result<Projection> {
            match prism.and_then(|registry| registry.get(&key)) {
                Some(weight) => Ok(Projection::load_prism(weight.clone())),
                None => Projection::load(req(key)?, None),
            }
        };

        let embed_key = join("embed_tokens.weight");
        let embed_tokens = match prism.and_then(|registry| registry.get(&embed_key)) {
            Some(weight) => QwenEmbedding::Prism(weight.clone()),
            None => QwenEmbedding::Dense(req(embed_key)?),
        };
        let norm = norm_w(join("norm.weight"))?;
        let head_key = if prism.is_some() {
            prefix
                .strip_suffix(".model")
                .map(|root| format!("{root}.lm_head.weight"))
                .unwrap_or_else(|| "lm_head.weight".to_string())
        } else {
            "lm_head.weight".to_string()
        };
        let lm_head = if let Some(weight) = prism.and_then(|registry| registry.get(&head_key)) {
            Projection::load_prism(weight.clone())
        } else if cfg.tie_word_embeddings {
            let QwenEmbedding::Dense(weight) = &embed_tokens else {
                return Err(Error::Config(
                    "Prism tied embeddings require an explicit lm_head packed tensor".into(),
                ));
            };
            Projection::load_as(weight.clone(), None, format)?
        } else {
            Projection::load_as(req(head_key)?, None, format)?
        };
        let lm_head = std::sync::Arc::new(lm_head);

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
                            gate: Projection::load_as(gate_w, None, format)?,
                            up: Projection::load_as(up_w, None, format)?,
                            down: Projection::load_as(dn, None, format)?,
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
            quantized: format.is_some() || prism.is_some(),
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

    // Rollback through the trait object: exact where a checkpoint exists (see `rollback_to`).
    fn truncate(&mut self, len: i32) -> Result<()> {
        Qwen35Cache::rollback_to(self, len)
    }
}

impl StepModel for Qwen35Model {
    type Cache = Qwen35Cache;

    /// A cache that retains [`STEP_MAX_CHECKPOINTS`] rollback checkpoints — the step seam's callers
    /// (the speculative verify) roll back through them.
    fn new_cache(&self) -> Qwen35Cache {
        self.new_cache_with_checkpoints(STEP_MAX_CHECKPOINTS)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size as usize
    }

    fn forward_step(
        &self,
        cache: &mut Qwen35Cache,
        request: StepRequest<'_>,
    ) -> Result<StepOutput> {
        if request.tokens.is_empty() {
            return Err(Error::Msg(
                "Qwen35Model::forward_step: empty token slice".into(),
            ));
        }
        let offset = cache.offset();
        let ids = input_ids(request.tokens, &self.device)?;
        let (logits, hidden) = match (request.scope, request.want_hidden) {
            (LogitsScope::Last, false) => (self.decode_logits(&ids, cache, offset)?, None),
            (LogitsScope::Last, true) => {
                let (logits, hidden) = self.prefill_with_hidden(&ids, cache, offset)?;
                (logits, Some(hidden))
            }
            (LogitsScope::All, false) => (self.forward(&ids, cache, offset)?, None),
            (LogitsScope::All, true) => {
                let (logits, hidden) = self.forward_with_hidden(&ids, cache, offset)?;
                (logits, Some(hidden))
            }
        };
        Ok(StepOutput { logits, hidden })
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
        self.decode_logits_from_embeds_deepstack(
            embeds,
            positions,
            cache,
            visual_pos_mask,
            deepstack,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn published_prism_norm_multiplier_matches_independent_rms_oracle() {
        let source = vec![1.0583496f32, 0.9418945, 1.3125, 1.957_031_3];
        let x = vec![0.25f32, -0.5, 1.25, -2.0];
        let weight = checkpoint_norm_weight(
            Tensor::from_vec(source.clone(), 4, &Device::Cpu).unwrap(),
            true,
        )
        .unwrap();
        let got = rms_norm(
            &Tensor::from_vec(x.clone(), (1, 4), &Device::Cpu).unwrap(),
            &weight,
            1e-6,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let inv = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6)
            .sqrt()
            .recip();
        for (i, value) in got.iter().enumerate() {
            let expected = x[i] * inv * source[i];
            assert!(
                (value - expected).abs() < 1e-5,
                "lane {i}: {value} != {expected}"
            );
        }
        let dense = checkpoint_norm_weight(
            Tensor::from_vec(vec![0.0583496f32], 1, &Device::Cpu).unwrap(),
            false,
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert!((dense[0] - source[0]).abs() < 1e-6);
    }

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

    #[test]
    fn frozen_qwen38_dense_config_matches_qwen35_decoder() {
        let value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/config.json"
        )))
        .unwrap();
        let cfg = Qwen35Config::from_json(&value).unwrap();

        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 64);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.mrope_section_resolved(), [11, 11, 10]);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(!cfg.mtp_use_dedicated_embeddings);
        assert!(cfg.moe.is_none());
    }

    #[test]
    fn frozen_qwen38_mtp_inventory_is_exact_and_fail_closed() {
        let value: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/config.json"
        )))
        .unwrap();
        let cfg = Qwen35Config::from_json(&value).unwrap();
        let mut keys = Qwen35Mtp::required_keys(cfg.mtp_num_hidden_layers);
        keys.sort();
        let expected = [
            "mtp.fc.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ];
        assert_eq!(keys, expected);

        let scalar = || Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
        let complete = Weights::from_map(
            expected
                .iter()
                .map(|key| ((*key).to_string(), scalar()))
                .collect(),
            Device::Cpu,
        );
        assert!(Qwen35Mtp::complete_in(&complete, &cfg));
        let incomplete = Weights::from_map(
            expected[..expected.len() - 1]
                .iter()
                .map(|key| ((*key).to_string(), scalar()))
                .collect(),
            Device::Cpu,
        );
        assert!(!Qwen35Mtp::complete_in(&incomplete, &cfg));
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

    fn synthetic_weights_with_prefix(cfg: &Qwen35Config, pfx: &str) -> Weights {
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
        // The decoder can live under either VLM-wrapped or flat text-only roots; `lm_head` stays
        // at the checkpoint root in both layouts.
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
        if cfg.mtp_num_hidden_layers == 1 {
            t(&mut m, "mtp.fc.weight", &[h, 2 * h]);
            t(&mut m, "mtp.pre_fc_norm_embedding.weight", &[h]);
            t(&mut m, "mtp.pre_fc_norm_hidden.weight", &[h]);
            t(&mut m, "mtp.norm.weight", &[h]);
            let lp = |s: &str| format!("mtp.layers.0.{s}");
            t(&mut m, &lp("input_layernorm.weight"), &[h]);
            t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
            t(&mut m, &lp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
            t(&mut m, &lp("self_attn.k_proj.weight"), &[nkv * hd, h]);
            t(&mut m, &lp("self_attn.v_proj.weight"), &[nkv * hd, h]);
            t(&mut m, &lp("self_attn.o_proj.weight"), &[h, nh * hd]);
            t(&mut m, &lp("self_attn.q_norm.weight"), &[hd]);
            t(&mut m, &lp("self_attn.k_norm.weight"), &[hd]);
            t(&mut m, &lp("mlp.gate_proj.weight"), &[inter, h]);
            t(&mut m, &lp("mlp.up_proj.weight"), &[inter, h]);
            t(&mut m, &lp("mlp.down_proj.weight"), &[h, inter]);
        }
        Weights::from_map(m, Device::Cpu)
    }

    fn synthetic_weights(cfg: &Qwen35Config) -> Weights {
        synthetic_weights_with_prefix(cfg, "model.language_model")
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
    fn flat_text_and_wrapped_qwen35_match_target_and_mtp() {
        let mut wrapped_json = cfg_json();
        wrapped_json["text_config"]["mtp_num_hidden_layers"] = json!(1);
        wrapped_json["text_config"]["mtp_use_dedicated_embeddings"] = json!(false);
        let flat_json = wrapped_json["text_config"].clone();
        let wrapped_cfg = Qwen35Config::from_json(&wrapped_json).unwrap();
        let flat_cfg = Qwen35Config::from_json(&flat_json).unwrap();
        let wrapped_weights = synthetic_weights_with_prefix(&wrapped_cfg, "model.language_model");
        let flat_weights = synthetic_weights_with_prefix(&flat_cfg, "model");
        let wrapped =
            Qwen35Model::from_weights(&wrapped_weights, "model.language_model", wrapped_cfg)
                .unwrap();
        let flat = Qwen35Model::from_weights(&flat_weights, "model", flat_cfg).unwrap();

        let tokens = ids(&[1, 7, 3]);
        let wrapped_logits = wrapped
            .forward(&tokens, &mut wrapped.new_cache(), 0)
            .unwrap();
        let flat_logits = flat.forward(&tokens, &mut flat.new_cache(), 0).unwrap();
        assert_eq!(host(&wrapped_logits), host(&flat_logits));

        let wrapped_mtp = Qwen35Mtp::from_weights_with(&wrapped_weights, &wrapped, None).unwrap();
        let flat_mtp = Qwen35Mtp::from_weights_with(&flat_weights, &flat, None).unwrap();
        let previous = Tensor::zeros((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
        let (wrapped_draft, _) = wrapped_mtp
            .step(2, &previous, 0, 0, &mut wrapped_mtp.new_cache())
            .unwrap();
        let (flat_draft, _) = flat_mtp
            .step(2, &previous, 0, 0, &mut flat_mtp.new_cache())
            .unwrap();
        assert_eq!(host(&wrapped_draft), host(&flat_draft));
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

    pub(crate) fn text_model() -> (Qwen35Config, Qwen35Model) {
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

    /// The load census reports every projection the synthetic decoder holds, by kind, and every
    /// other weight tensor, with f32 (CPU) resident bytes.
    #[test]
    fn dense_weight_census_covers_every_projection_and_tensor() {
        let (cfg, model) = text_model();
        let census = model.weight_census();
        // 3 linear layers × (qkv, z, a, b, out) + 1 attention layer × (q, k, v, o)
        // + 4 layers × (gate, up, down) + lm_head.
        assert_eq!(census.projections.dense.count, 15 + 4 + 12 + 1);
        assert_eq!(census.projections.total().count, 32);
        let total = census.total();
        assert_eq!(total.unmeasured, 0);
        assert_eq!(total.resident_bytes, total.params * 4, "f32 on CPU");
        // The embedding is resident beside the (untied) head.
        assert!(!cfg.tie_word_embeddings);
        assert!(census.other.params >= (cfg.vocab_size * cfg.hidden_size) as u64);
    }

    /// sc-24135: the loader selecting NVFP4 stores every large projection (and the head) as NVFP4,
    /// keeps the per-head decay/delta projections dense exactly as Q4/Q8 do, and the decoder still
    /// produces finite logits that track the dense model's.
    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_format_loads_the_large_projections_as_nvfp4() {
        use crate::primitives::projection::{ProjectionFormat, ProjectionKind};
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let Ok(format) = ProjectionFormat::nvfp4(&device) else {
            eprintln!("skipping: CUDA device below the NVFP4 floor");
            return;
        };
        // The fixture's vocabulary (50) is not a multiple of 16; NVFP4 needs N % 16 == 0.
        let mut v = cfg_json();
        v["text_config"]["vocab_size"] = json!(64);
        let cfg = Qwen35Config::from_json(&v).unwrap();
        let cpu = synthetic_weights(&cfg);
        let on_device = Weights::from_map(
            cpu.keys()
                .map(|k| {
                    (
                        k.to_string(),
                        cpu.require(k).unwrap().to_device(&device).unwrap(),
                    )
                })
                .collect(),
            device.clone(),
        );
        let dense =
            Qwen35Model::from_weights_format(&on_device, "model.language_model", cfg.clone(), None)
                .unwrap();
        let nvfp4 = Qwen35Model::from_weights_format(
            &on_device,
            "model.language_model",
            cfg.clone(),
            Some(&format),
        )
        .unwrap();
        assert!(nvfp4.is_quantized());
        assert_eq!(nvfp4.lm_head.kind(), ProjectionKind::Nvfp4);
        let census = nvfp4.weight_census().projections;
        assert_eq!(census.nvfp4.count, 32 - 6, "all but in_proj_a/b");
        assert_eq!(
            census.dense.count, 6,
            "in_proj_a/b stay dense, as under Q4/Q8"
        );
        assert_eq!(census.ggml.count, 0);
        assert!(
            census.nvfp4.resident_bytes < census.nvfp4.params * 2,
            "packed below bf16"
        );

        let prompt = Tensor::from_vec(vec![1u32, 7, 3, 42, 9], (1, 5), &device).unwrap();
        let logits = |m: &Qwen35Model| {
            let mut cache = m.new_cache();
            m.forward(&prompt, &mut cache, 0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (want, got) = (logits(&dense), logits(&nvfp4));
        assert!(got.iter().all(|x| x.is_finite()));
        let dot: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&got) * norm(&want)).max(1e-30);
        assert!(
            cosine > 0.9,
            "NVFP4 logits diverged from dense: cosine {cosine}"
        );
        // Cosine is scale-invariant; the relative RMS error also pins the logits' magnitude.
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_rms = err / norm(&want).max(1e-30);
        assert!(
            rel_rms <= 0.3,
            "NVFP4 logits diverged from dense: relative RMS {rel_rms}"
        );
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        host(a)
            .iter()
            .zip(&host(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// **AC2 (tiny config).** Decode `m > n` tokens, roll back to `n`, re-decode token `n`: the
    /// logits must equal a fresh decode to `n + 1` — exactly, since the restored recurrent state is
    /// the same tensor and the narrowed KV holds the same values.
    #[test]
    fn rollback_to_then_redecode_matches_fresh_decode() {
        let (_cfg, model) = text_model();
        let toks = [1u32, 7, 3, 42, 9, 2, 11, 5];
        let n = 3usize;
        let m = toks.len();

        // Fresh: prefill toks[..n], then decode toks[n] -> logits for position n.
        let mut fresh = model.new_cache();
        model
            .decode_logits(&ids(&toks[..n]), &mut fresh, 0)
            .unwrap();
        let fresh_logits = model
            .decode_logits(&ids(&[toks[n]]), &mut fresh, n as i32)
            .unwrap();

        // Rolled back: prefill toks[..n], decode toks[n..m] one at a time (m > n), roll back to n.
        // Retention is opt-in beyond the default two: keep every step start of this run.
        let mut cache = model.new_cache();
        cache.set_max_checkpoints(8);
        model
            .decode_logits(&ids(&toks[..n]), &mut cache, 0)
            .unwrap();
        for (i, &t) in toks[n..m].iter().enumerate() {
            model
                .decode_logits(&ids(&[t]), &mut cache, (n + i) as i32)
                .unwrap();
        }
        assert_eq!(cache.offset(), m as i32);
        // Every step start is a checkpoint: the prefill at 0, then each single-token step.
        let mut expected = vec![0i32];
        expected.extend(n as i32..m as i32);
        assert_eq!(cache.checkpoint_offsets(), expected);
        cache.rollback_to(n as i32).unwrap();
        assert_eq!(cache.offset(), n as i32);
        assert_eq!(cache.checkpoint_offsets(), vec![0, n as i32]);
        let replayed = model
            .decode_logits(&ids(&[toks[n]]), &mut cache, n as i32)
            .unwrap();
        assert_eq!(
            max_abs_diff(&fresh_logits, &replayed),
            0.0,
            "rollback_to({n}) then re-decode must equal a fresh decode"
        );
        assert_eq!(cache.offset(), n as i32 + 1);

        // Continuing past the rollback point stays exact too (the conv tail was restored as well).
        let mut fresh2 = fresh;
        let a = model
            .decode_logits(&ids(&[toks[n + 1]]), &mut fresh2, n as i32 + 1)
            .unwrap();
        let b = model
            .decode_logits(&ids(&[toks[n + 1]]), &mut cache, n as i32 + 1)
            .unwrap();
        assert_eq!(max_abs_diff(&a, &b), 0.0);
    }

    /// Rollback refuses positions it has no checkpoint for (inside a multi-token prefill, or pruned
    /// away), and never approximates; `0` and the current position always work.
    #[test]
    fn rollback_is_exact_or_refused() {
        let (_cfg, model) = text_model();
        let mut cache = StepModel::new_cache(&model);
        assert_eq!(cache.max_checkpoints(), STEP_MAX_CHECKPOINTS);
        model
            .decode_logits(&ids(&[1, 7, 3, 42]), &mut cache, 0)
            .unwrap();
        assert!(cache.rollback_to(4).is_ok(), "current position is a no-op");
        assert_eq!(cache.offset(), 4);
        // Refused with the typed variant (the S2 speculative engine matches on it, not on text).
        match cache.rollback_to(2) {
            Err(Error::RollbackUnavailable { n: 2, have }) => assert_eq!(have, vec![0]),
            other => panic!("expected RollbackUnavailable {{ n: 2, have: [0] }}, got {other:?}"),
        }
        assert_eq!(
            cache.offset(),
            4,
            "a refused rollback leaves the cache untouched"
        );
        // Out of range is a plain error, not "no checkpoint".
        assert!(
            matches!(cache.rollback_to(5), Err(Error::Msg(_))),
            "past the end"
        );
        assert!(matches!(cache.rollback_to(-1), Err(Error::Msg(_))));
        assert_eq!(
            cache.checkpoint_offsets(),
            vec![0],
            "one forward so far: only its start is checkpointed"
        );
        cache.rollback_to(0).unwrap();
        assert_eq!(cache.offset(), 0);
        assert!(cache.checkpoint_offsets().is_empty());
        assert_eq!(cache.memory(), CacheMemory::default());

        // Retention: with room for one checkpoint only the newest step start survives.
        cache.set_max_checkpoints(1);
        model.decode_logits(&ids(&[1, 7]), &mut cache, 0).unwrap();
        model.decode_logits(&ids(&[3]), &mut cache, 2).unwrap();
        model.decode_logits(&ids(&[42]), &mut cache, 3).unwrap();
        assert_eq!(cache.checkpoint_offsets(), vec![3]);
        assert!(matches!(
            cache.rollback_to(2),
            Err(Error::RollbackUnavailable { n: 2, .. })
        ));
        cache.rollback_to(3).unwrap();
        assert_eq!(cache.offset(), 3);

        // Through the `KvCache` trait object the same rollback is reachable as `truncate`.
        let dyn_cache: &mut dyn KvCache = &mut cache;
        crate::decode::Decode::step(&model, &ids(&[9]), dyn_cache, 3).unwrap();
        assert!(dyn_cache.truncate(3).is_ok());
        assert_eq!(dyn_cache.offset(), 3);

        // The reference / MTP cache (`new_cache`, what `Decode::make_cache` boxes) retains no
        // checkpoints, so it holds one recurrent state — and refuses any interior rollback.
        let mut reference = model.new_cache();
        assert_eq!(reference.max_checkpoints(), REFERENCE_MAX_CHECKPOINTS);
        model
            .decode_logits(&ids(&[1, 7, 3]), &mut reference, 0)
            .unwrap();
        model.decode_logits(&ids(&[42]), &mut reference, 3).unwrap();
        model.decode_logits(&ids(&[9]), &mut reference, 4).unwrap();
        assert!(reference.checkpoint_offsets().is_empty());
        assert_eq!(reference.memory().checkpoint_bytes, 0);
        match reference.rollback_to(4) {
            Err(Error::RollbackUnavailable { n: 4, have }) => assert!(have.is_empty()),
            other => panic!("expected RollbackUnavailable {{ n: 4, have: [] }}, got {other:?}"),
        }
    }

    /// The memory accounting follows the live tensors and the checkpoints, and a rollback drops
    /// the rolled-back positions' KV bytes.
    #[test]
    fn memory_accounting_tracks_live_state_and_checkpoints() {
        let (cfg, model) = text_model();
        let mut cache = StepModel::new_cache(&model);
        assert_eq!(cache.memory(), CacheMemory::default());
        model
            .decode_logits(&ids(&[1, 7, 3]), &mut cache, 0)
            .unwrap();
        let after_prefill = cache.memory();
        // One full-attention layer: K and V of [1, nkv, 3, hd] f32 each.
        let kv_bytes = 2 * (cfg.num_kv_heads as usize) * 3 * (cfg.head_dim as usize) * 4;
        assert!(
            after_prefill.live_bytes > kv_bytes,
            "KV plus DeltaNet states"
        );
        assert_eq!(
            after_prefill.checkpoint_bytes, 0,
            "the position-0 checkpoint holds empty (None) states"
        );
        model.decode_logits(&ids(&[42]), &mut cache, 3).unwrap();
        let after_step = cache.memory();
        assert_eq!(
            after_step.live_bytes - after_prefill.live_bytes,
            kv_bytes / 3,
            "one more KV position; recurrent states are fixed-size"
        );
        assert!(
            after_step.checkpoint_bytes > 0,
            "the position-3 checkpoint holds real states"
        );
        // The recurrent share: the live linear-layer states plus the one real checkpoint (the
        // position-0 checkpoint is empty) — two states' worth.
        assert_eq!(
            cache.recurrent_bytes(),
            after_step.live_bytes - kv_bytes * 4 / 3 + after_step.checkpoint_bytes
        );
        assert_eq!(
            cache.recurrent_bytes(),
            2 * after_step.checkpoint_bytes,
            "live recurrent state + one checkpoint of the same size"
        );
        cache.rollback_to(3).unwrap();
        assert_eq!(cache.memory().live_bytes, after_prefill.live_bytes);
        assert_eq!(
            cache.memory().total_bytes(),
            after_step.total_bytes() - kv_bytes / 3
        );
    }

    /// A schedule with no full-attention layer still reports its position (from the linear layers).
    #[test]
    fn offset_falls_back_to_linear_layers() {
        let mut v = cfg_json();
        v["text_config"]["full_attention_interval"] = json!(100);
        let cfg = Qwen35Config::from_json(&v).unwrap();
        assert!((0..cfg.num_layers).all(|i| cfg.is_linear(i)));
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        let mut cache = StepModel::new_cache(&model);
        model
            .decode_logits(&ids(&[1, 7, 3]), &mut cache, 0)
            .unwrap();
        assert_eq!(cache.offset(), 3);
        model.decode_logits(&ids(&[9]), &mut cache, 3).unwrap();
        assert_eq!(cache.offset(), 4);
        cache.rollback_to(3).unwrap();
        assert_eq!(cache.offset(), 3);
    }

    /// **AC1 (tiny config).** Greedy generation through the `StepModel` seam is token-identical to
    /// the reference `Decode` loop, and the record says which path ran.
    #[test]
    fn step_model_greedy_matches_reference_decode_loop() {
        use crate::decode::{generate_step, generate_with, CancelFlag, GenerationConfig};
        let (_cfg, model) = text_model();
        let prompt = [1i32, 7, 3, 42, 9];
        let cfg = GenerationConfig {
            max_new_tokens: 24,
            seed: Some(3),
            ..Default::default()
        };
        let reference =
            generate_with(&model, &prompt, &cfg, &CancelFlag::new(), &mut |_| {}, None).unwrap();
        let mut events = 0usize;
        let (step, record) = generate_step(
            &model,
            &prompt,
            &cfg,
            &CancelFlag::new(),
            &mut |e| {
                if matches!(e, crate::decode::StreamEvent::Token { .. }) {
                    events += 1;
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(step.tokens.len(), 24);
        assert_eq!(step.tokens, reference.tokens);
        assert_eq!(step.finish_reason, reference.finish_reason);
        assert_eq!(events, 24);
        assert_eq!(record.path, crate::decode::DecodePath::StepModel);
        assert_eq!(record.target_forwards, 24, "prefill + 23 steps");
        assert_eq!(record.generated_tokens, 24);
        assert_eq!(
            record.host_syncs, 24,
            "one device argmax per token on the plain-greedy path"
        );

        // Sampling (positive temperature) is seed-deterministic and identical across the seams too.
        let mut sampled = cfg.clone();
        sampled.sampling.temperature = 0.9;
        sampled.sampling.top_k = 5;
        let reference = generate_with(
            &model,
            &prompt,
            &sampled,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let (step, _) = generate_step(
            &model,
            &prompt,
            &sampled,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(step.tokens, reference.tokens);
    }

    /// The `StepModel` step shapes: last vs all logits, optional hidden states, and the cache
    /// advancing by the token count.
    #[test]
    fn step_model_shapes_and_cache_advance() {
        let (cfg, model) = text_model();
        let vocab = cfg.vocab_size as usize;
        let hidden = cfg.hidden_size as usize;
        let mut cache = StepModel::new_cache(&model);
        let out = model
            .forward_step(&mut cache, StepRequest::all(&[1, 7, 3]))
            .unwrap();
        assert_eq!(out.logits.dims(), &[1, 3, vocab]);
        assert!(out.hidden.is_none());
        assert_eq!(cache.len(), 3);
        let out = model
            .forward_step(
                &mut cache,
                StepRequest {
                    tokens: &[42, 9],
                    scope: LogitsScope::Last,
                    want_hidden: true,
                },
            )
            .unwrap();
        assert_eq!(out.logits.dims(), &[1, vocab]);
        assert_eq!(out.hidden.unwrap().dims(), &[1, 2, hidden]);
        assert_eq!(cache.len(), 5);
        let out = model
            .forward_step(
                &mut cache,
                StepRequest {
                    tokens: &[2],
                    scope: LogitsScope::All,
                    want_hidden: true,
                },
            )
            .unwrap();
        assert_eq!(out.logits.dims(), &[1, 1, vocab]);
        assert_eq!(out.hidden.unwrap().dims(), &[1, 1, hidden]);
        assert!(model
            .forward_step(&mut cache, StepRequest::last(&[]))
            .is_err());
        assert_eq!(StepModel::vocab_size(&model), vocab);
        // The last-logits step equals the reference `decode_logits` at the same position.
        let mut reference = model.new_cache();
        model
            .decode_logits(&ids(&[1, 7, 3, 42, 9, 2]), &mut reference, 0)
            .unwrap();
        let a = model.decode_logits(&ids(&[11]), &mut reference, 6).unwrap();
        let b = model
            .forward_step(&mut cache, StepRequest::last(&[11]))
            .unwrap()
            .logits;
        assert!(max_abs_diff(&a, &b) < 1e-4);
    }
}
