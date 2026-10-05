//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) — the hybrid decoder (story sc-7628).
//!
//! Unlike the generic [`CausalLm`](super::llama::CausalLm) (all softmax attention over a growing KV
//! cache), this decoder interleaves two mixer types on a fixed schedule (`full_attention_interval`,
//! default 4 → **3 Gated DeltaNet linear-attention layers : 1 gated full-attention layer**):
//!
//! - `GatedDeltaNet` — linear attention carrying a fixed-size recurrent state (the verified
//!   primitives in [`crate::primitives::gated_delta`]): in-proj → short conv → q/k RMS-norm →
//!   gated delta recurrence → gated RMS-norm → out-proj.
//! - `Qwen35Attention` — grouped-query attention with **partial RoPE** (`partial_rotary_factor`,
//!   reusing the [`Rope::partial`] path), per-head q/k RMS-norm, and an **output gate** (the queries
//!   projection is doubled into `[queries ‖ gate]`, and the attended output is multiplied by
//!   `sigmoid(gate)` before the output projection).
//!
//! Each decoder layer is `input_layernorm → mixer → residual → post_attention_layernorm → MLP →
//! residual`. The MLP is a dense SwiGLU for the 27B; the 35B MoE bank is wired in sc-7630. The KV
//! cache (full-attn layers) and the recurrent [`DeltaNetCache`] (linear layers) live side by side in
//! a per-layer [`Qwen35Cache`]. RMSNorm weights follow the Qwen3-Next `(1 + weight)` convention.

use std::cell::Cell;

use mlx_rs::ops::{add, concatenate_axis, multiply, rsqrt, sigmoid, split_sections, sum_axis};
use mlx_rs::transforms::async_eval;
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::models::deepstack::{self, deepstack_fused_decoder_layers};
use crate::primitives::attention::{sdpa_capped, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv_traced, compute_g, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::moe::{MoeRouting, SparseMoe, SwiGlu, SwitchLinear};
use crate::primitives::nn::{embed, rms_norm, silu};
use crate::primitives::prism::PrismEmbedding;
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;
use crate::prism::PrismMlxPack;

fn checkpoint_norm_weight(weight: Array, prism: bool) -> Result<Array> {
    if prism {
        Ok(weight)
    } else {
        Ok(add(
            &weight,
            &Array::from_f32(1.0).as_dtype(weight.dtype())?,
        )?)
    }
}

/// Cached decode runs in bf16 (matching the rest of the engine); the delta recurrence accumulates in
/// f32 (matching the reference GPU kernel) for stability.
const BF16_COMPUTE: Dtype = Dtype::Bfloat16;

#[cfg(test)]
thread_local! {
    /// A test's compute-dtype override ([`with_compute_dtype`]).
    static COMPUTE_OVERRIDE: std::cell::Cell<Option<Dtype>> = const { std::cell::Cell::new(None) };
}

/// The decoder's compute dtype: bf16 ([`BF16_COMPUTE`]) — or, in a test, the dtype
/// [`with_compute_dtype`] runs it in.
#[inline]
fn compute() -> Dtype {
    #[cfg(test)]
    if let Some(dtype) = COMPUTE_OVERRIDE.with(std::cell::Cell::get) {
        return dtype;
    }
    BF16_COMPUTE
}

/// Run `f` — a load and its forwards — with the hybrid decoder computing in `dtype` on this
/// thread (test seam: the f32 twin of a bf16 parity fixture, sc-24446 merge review).
#[cfg(test)]
pub(crate) fn with_compute_dtype<R>(dtype: Dtype, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Dtype>);
    impl Drop for Restore {
        fn drop(&mut self) {
            COMPUTE_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(COMPUTE_OVERRIDE.with(|c| c.replace(Some(dtype))));
    f()
}

/// Interleaved M-RoPE output of [`Qwen35Model::mrope_positions`]: the temporal / height / width
/// position rows (each length `S`) plus the `mrope_delta` (`max_position + 1 − len`) for continuing
/// positions after the prompt.
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// Mixture-of-Experts FFN parameters (`qwen3_5_moe`, the 35B-A3B). Every layer's dense MLP is
/// replaced by a sparse MoE block: a softmax router over `num_experts` experts (top-`experts_per_tok`
/// per token, weights renormalized to sum to 1) plus a sigmoid-gated always-on shared expert.
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
    /// Number of in-checkpoint multi-token predictor layers. Zero means no MTP capability.
    pub mtp_num_hidden_layers: usize,
    /// Whether MTP carries a separate embedding table. Qwen3.8 shares the target embeddings.
    pub mtp_use_dedicated_embeddings: bool,
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
    /// Stored Q4/Q8 projection format, read from either the decoder's nested `text_config` or the
    /// top-level snapshot config. `None` means dense weights (which may still be quantized on load).
    pub quantization: Option<QuantSpec>,
}

impl Qwen35Config {
    /// Parse from a `config.json` value, descending into `text_config` for the VLM wrapper.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
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
        // The MoE variant (`qwen3_5_moe`) has no dense `intermediate_size` — every layer is MoE —
        // so fall back to the per-expert width (unused on the MoE path, but keeps the field valid).
        let intermediate_size = int("intermediate_size")
            .or_else(|| int("moe_intermediate_size"))
            .unwrap_or(0);
        let nested_quant = parse_quantization(c.get("quantization"), "text_config.quantization")?;
        let is_prism =
            v.get("model_type").and_then(|x| x.as_str()) == Some("prism_hadamard_qwen35");
        let top_quant = if std::ptr::eq(c, v) || is_prism {
            None
        } else {
            parse_quantization(v.get("quantization"), "quantization")?
        };
        if nested_quant.is_some() && top_quant.is_some() && nested_quant != top_quant {
            return Err(Error::Config(format!(
                "qwen3_5 config.json has conflicting nested/top-level quantization blocks: \
                 {nested_quant:?} != {top_quant:?}"
            )));
        }
        let quantization = nested_quant.or(top_quant);
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
            mtp_num_hidden_layers: int("mtp_num_hidden_layers").unwrap_or(0).max(0) as usize,
            mtp_use_dedicated_embeddings: c
                .get("mtp_use_dedicated_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            full_attention_interval: int("full_attention_interval").unwrap_or(4).max(1) as usize,
            linear_num_value_heads: req("linear_num_value_heads")?,
            linear_num_key_heads: req("linear_num_key_heads")?,
            linear_key_head_dim: req("linear_key_head_dim")?,
            linear_value_head_dim: req("linear_value_head_dim")?,
            linear_conv_kernel_dim: int("linear_conv_kernel_dim").unwrap_or(4),
            // MoE variant (`qwen3_5_moe`, 35B-A3B) iff `num_experts` is present.
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
            quantization,
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

    /// An upper bound on the bytes a prefix-cache snapshot of the whole cache at `positions`
    /// holds (story sc-24437), at 4 bytes an element: every full-attention layer's KV for
    /// `positions` rounded up to the KV block, every linear layer's conv tail and recurrent state,
    /// and — with `mtp` — the predictor's KV plus one hidden row. `None` on overflow.
    pub fn prefix_snapshot_bytes(&self, positions: usize, mtp: bool) -> Option<u64> {
        let block = crate::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
        let positions = (positions as u64).div_ceil(block).checked_mul(block)?;
        let linear = (0..self.num_layers).filter(|&i| self.is_linear(i)).count() as u64;
        let attention = self.num_layers as u64 - linear;
        let kv_position = (self.num_kv_heads as u64)
            .checked_mul(self.head_dim as u64)?
            .checked_mul(2 * 4)?;
        let conv_dim = (2 * self.linear_num_key_heads as u64)
            .checked_mul(self.linear_key_head_dim as u64)?
            .checked_add(
                (self.linear_num_value_heads as u64)
                    .checked_mul(self.linear_value_head_dim as u64)?,
            )?;
        let recurrent = (self.linear_conv_kernel_dim as u64)
            .saturating_sub(1)
            .checked_mul(conv_dim)?
            .checked_add(
                (self.linear_num_value_heads as u64)
                    .checked_mul(self.linear_value_head_dim as u64)?
                    .checked_mul(self.linear_key_head_dim as u64)?,
            )?
            .checked_mul(4)?;
        let head = if mtp {
            (self.mtp_num_hidden_layers as u64)
                .checked_mul(kv_position)?
                .checked_mul(positions)?
                .checked_add((self.hidden_size as u64).checked_mul(4)?)?
        } else {
            0
        };
        attention
            .checked_mul(kv_position)?
            .checked_mul(positions)?
            .checked_add(linear.checked_mul(recurrent)?)?
            .checked_add(head)
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

fn parse_quantization(value: Option<&serde_json::Value>, label: &str) -> Result<Option<QuantSpec>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let obj = value
        .as_object()
        .ok_or_else(|| Error::Config(format!("{label} must be an object")))?;
    let integer = |field: &str| -> Result<i32> {
        let raw = obj
            .get(field)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Error::Config(format!("{label}.{field} must be an integer")))?;
        i32::try_from(raw)
            .map_err(|_| Error::Config(format!("{label}.{field} is outside the i32 range")))
    };
    let spec = QuantSpec {
        group_size: integer("group_size")?,
        bits: integer("bits")?,
    };
    if !matches!(spec.bits, 4 | 8) {
        return Err(Error::Config(format!(
            "{label}.bits must be 4 or 8, got {}",
            spec.bits
        )));
    }
    if spec.group_size != 64 {
        return Err(Error::Config(format!(
            "{label}.group_size must be 64, got {}",
            spec.group_size
        )));
    }
    Ok(Some(spec))
}

/// Multiply `x` by an f32 scalar (cast to `x`'s dtype).
fn scale(x: &Array, c: f32) -> Result<Array> {
    Ok(multiply(x, &Array::from_f32(c).as_dtype(x.dtype())?)?)
}

/// Zeros `[b, len, c]` in `dtype`.
fn zeros3(b: i32, len: i32, c: i32, dtype: Dtype) -> Result<Array> {
    let n = (b * len * c) as usize;
    Ok(Array::from_slice(&vec![0.0f32; n], &[b, len, c]).as_dtype(dtype)?)
}

/// L2-normalize over the last axis: `x · rsqrt(Σ x² + eps)` (the FLA `use_qk_l2norm_in_kernel`
/// convention — `eps` is added to the **sum**, not the mean). Computed in `x`'s dtype, matching the
/// reference kernel which normalizes the projected q/k before the recurrence.
fn l2norm(x: &Array, eps: f32) -> Result<Array> {
    let ss = sum_axis(&multiply(x, x)?, -1, true)?; // Σ x²  → [.., 1]
    let inv = rsqrt(&add(&ss, &Array::from_f32(eps).as_dtype(ss.dtype())?)?)?;
    Ok(multiply(x, &inv)?)
}

/// The Gated DeltaNet linear-attention layer (`Qwen3_5GatedDeltaNet`).
///
/// The Qwen3.6 checkpoint splits the input projection **four ways** — `in_proj_qkv` (fused q‖k‖v,
/// the only part the short conv mixes), `in_proj_z` (the output gate), and the per-value-head
/// `in_proj_a` / `in_proj_b` (decay / delta-strength) — rather than the fused `in_proj_qkvz` /
/// `in_proj_ba` of the older Qwen3-Next. After the conv, q/k/v are a **contiguous** split of the
/// `[key_dim, key_dim, value_dim]` channels (no head interleaving).
#[derive(Debug)]
struct GatedDeltaNet {
    in_proj_qkv: Projection, // [key_dim·2 + value_dim, hidden]  → conv'd
    in_proj_z: Projection,   // [value_dim, hidden]              → output gate
    in_proj_a: Projection,   // [Hv, hidden]                     → decay input
    in_proj_b: Projection,   // [Hv, hidden]                     → delta-strength input
    conv_weight: Array,      // [conv_dim, K]
    a_log: Array,            // [Hv]
    dt_bias: Array,          // [Hv]
    norm_weight: Array,      // [Dv] (RMSNormGated; loaded directly, ones-centered)
    out_proj: Projection,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel: i32,
    eps: f32,
}

impl GatedDeltaNet {
    fn push_arrays(&self, out: &mut Vec<Array>) {
        for p in [
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_a,
            &self.in_proj_b,
        ] {
            p.push_arrays(out);
        }
        out.extend([
            self.conv_weight.clone(),
            self.a_log.clone(),
            self.dt_bias.clone(),
            self.norm_weight.clone(),
        ]);
        self.out_proj.push_arrays(out);
    }

    fn forward(&self, x: &Array, cache: &mut DeltaNetCache) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        // Four independent in-projections (dtype follows the projection weights).
        let mixed = self.in_proj_qkv.forward(x)?; // [B,S,conv_dim] = q‖k‖v channels
        let dt = mixed.dtype();
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim])?; // output gate
        let a_in = self.in_proj_a.forward(x)?; // [B,S,Hv]  decay input
        let b_in = self.in_proj_b.forward(x)?; // [B,S,Hv]  delta-strength input

        // Short conv over the q‖k‖v channels (only these are convolved), seeded by the cache tail,
        // then a *contiguous* split into q [key_dim] ‖ k [key_dim] ‖ v [value_dim] and reshape to heads.
        let conv_state = match &cache.conv_state {
            Some(cs) => cs.clone(),
            None => zeros3(b, self.conv_kernel - 1, self.conv_dim, dt)?,
        };
        let (conv_out, conv_trace) =
            causal_depthwise_conv_traced(&mixed, &self.conv_weight, &conv_state)?;
        let cp = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], 2)?;
        let qc = cp[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let kc = cp[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let vc = cp[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        // L2-normalize q/k (eps 1e-6), then scale q by 1/√head_k_dim — `use_qk_l2norm_in_kernel`.
        let inv = (self.head_k_dim as f32).powf(-0.5);
        let qn = scale(&l2norm(&qc, 1e-6)?, inv)?;
        let kn = l2norm(&kc, 1e-6)?;

        // The gated delta recurrence, accumulated in f32 (matching the reference kernel). GQA
        // (q/k from Hk key heads → Hv value heads) is handled inside the recurrence primitive.
        // The cache advances itself: an armed (speculative verify) forward also keeps every
        // token's state for the checkpoint ring (sc-24435).
        let beta = sigmoid(&b_in)?;
        let g = compute_g(&a_in, &self.a_log, &self.dt_bias)?;
        let f32 = Dtype::Float32;
        let y = cache.advance(
            conv_trace,
            &qn.as_dtype(f32)?,
            &kn.as_dtype(f32)?,
            &vc.as_dtype(f32)?,
            &g.as_dtype(f32)?,
            &beta.as_dtype(f32)?,
        )?;

        // Gated RMS-norm with z (back in the layer dtype), then the output projection.
        let out = rms_norm_gated(&y.as_dtype(dt)?, &self.norm_weight, &z, self.eps)?;
        self.out_proj
            .forward(&out.reshape(&[b, s, self.value_dim])?)
    }
}

/// The gated full-attention layer (`Qwen3NextAttention`).
#[derive(Debug)]
struct Qwen35Attention {
    q_proj: Projection, // out = num_heads · head_dim · 2 (queries ‖ gate)
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen35Attention {
    fn push_arrays(&self, out: &mut Vec<Array>) {
        for p in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
            p.push_arrays(out);
        }
        out.extend([self.q_norm.clone(), self.k_norm.clone()]);
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, cache: &mut AttnKv) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let qg = self
            .q_proj
            .forward(x)?
            .reshape(&[b, s, self.num_heads, 2 * self.head_dim])?;
        let qp = split_sections(&qg, &[self.head_dim], 3)?;
        let q = rms_norm(&qp[0], &self.q_norm, self.eps)?; // [B,S,H,hd]
        let gate = qp[1].reshape(&[b, s, self.num_heads * self.head_dim])?;

        let k = rms_norm(
            &self
                .k_proj
                .forward(x)?
                .reshape(&[b, s, self.num_kv_heads, self.head_dim])?,
            &self.k_norm,
            self.eps,
        )?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Partial RoPE (NeoX), then transpose into head-major [B,H,S,hd].
        let q = apply_rope(&q, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update(&k, &v)?;
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, None, AttnMask::Causal)?; // [B,H,S,hd]
        let merged =
            out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        // Output gate: multiply by sigmoid(gate) before the output projection.
        let gated = multiply(&merged, &sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

/// Dense SwiGLU MLP (`Qwen3_5MLP`) — the 27B FFN and each 35B expert / shared expert.
#[derive(Debug)]
struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl Mlp {
    fn push_arrays(&self, out: &mut Vec<Array>) {
        for p in [&self.gate, &self.up, &self.down] {
            p.push_arrays(out);
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&gate, &up)?)
    }
}

/// The per-layer FFN: a dense SwiGLU (27B) or a sparse MoE block (35B-A3B).
///
/// The MoE block is the crate's shared [`SparseMoe`] (sc-24440): a softmax router over the stacked
/// experts (top-`experts_per_tok` per token, weights renormalized to sum to 1) plus an always-on
/// **sigmoid-gated** shared expert, routed and dispatched on the device.
#[derive(Debug)]
enum Ffn {
    Dense(Mlp),
    Moe(SparseMoe),
}

impl Ffn {
    fn push_arrays(&self, out: &mut Vec<Array>) {
        match self {
            Ffn::Dense(m) => m.push_arrays(out),
            Ffn::Moe(m) => m.push_arrays(out),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

#[derive(Debug)]
enum Mixer {
    Delta(GatedDeltaNet),
    Attn(Qwen35Attention),
}

#[derive(Debug)]
struct DecoderLayer {
    input_ln: Array,
    post_ln: Array,
    mixer: Mixer,
    ffn: Ffn,
    eps: f32,
}

impl DecoderLayer {
    /// Every array the layer holds, for load-time materialization (sc-24446).
    fn arrays(&self) -> Vec<Array> {
        let mut out = vec![self.input_ln.clone(), self.post_ln.clone()];
        match &self.mixer {
            Mixer::Delta(d) => d.push_arrays(&mut out),
            Mixer::Attn(a) => a.push_arrays(&mut out),
        }
        self.ffn.push_arrays(&mut out);
        out
    }

    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Qwen35LayerCache,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let r = match (&self.mixer, cache) {
            (Mixer::Delta(d), Qwen35LayerCache::Delta(c)) => d.forward(&normed, c)?,
            (Mixer::Attn(a), Qwen35LayerCache::Attn(c)) => a.forward(&normed, cos, sin, c)?,
            _ => return Err(Error::Msg("qwen3_5: cache/mixer type mismatch".into())),
        };
        let h = add(x, &r)?;
        let m = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
        Ok(add(&h, &m)?)
    }
}

/// A single full-attention layer's KV (the linear layers use [`DeltaNetCache`] instead): a
/// one-layer block-allocated [`ContiguousKvCache`], written in place between block growths, so a
/// long generation does not retire a strictly-larger buffer per token per layer (Qwen3.6-27B has
/// 16 of these layers; the per-token concat cost ~370 MB of unreusable buffers per token at 5.6k
/// context).
///
/// `Clone` is a buffer-sharing snapshot — a snapshot rollback (the MTP predictor's `MtpCache`, the
/// engine's generic `SnapshotRollback`) stays correct because the in-place write copies rather
/// than donates while a snapshot holds the buffer. The hybrid target itself rolls back through
/// [`Qwen35Cache::truncate`] and never takes one.
#[derive(Clone, Debug)]
pub struct AttnKv {
    kv: ContiguousKvCache,
}

impl Default for AttnKv {
    fn default() -> Self {
        Self {
            kv: ContiguousKvCache::new(1),
        }
    }
}

impl AttnKv {
    /// A slot growing by `block` positions at a time; tests use a small block to cross growth
    /// boundaries with tiny tensors.
    #[cfg(test)]
    fn with_block_tokens(block: i32) -> Self {
        Self {
            kv: ContiguousKvCache::with_block_tokens(1, block),
        }
    }

    fn update(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        self.kv.update(0, k, v)
    }

    /// Live positions — the slot's write offset, not the padded buffer length.
    fn offset(&self) -> i32 {
        self.kv.offset()
    }

    fn batch_size(&self) -> i32 {
        self.kv.batch_size()
    }

    fn reset(&mut self) -> Result<()> {
        self.kv.reset()
    }

    /// Keep positions `0..len` — bookkeeping only: the rolled-back positions stay in the block
    /// buffer as padding and the next update overwrites them in place.
    fn truncate(&mut self, len: i32) -> Result<()> {
        self.kv.truncate(len)
    }

    /// A copy of positions `0..len` in its own buffers ([`ContiguousKvCache::prefix`]).
    fn prefix(&self, len: i32) -> Result<Self> {
        Ok(Self {
            kv: self.kv.prefix(len)?,
        })
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
///
/// It rolls back without a forward: [`arm_checkpoints`](Self::arm_checkpoints) opens a checkpoint
/// window in every DeltaNet layer (the ring, sc-24435) — the state there and after every token of
/// each forward until the window closes stays restorable — after which
/// [`truncate`](Self::truncate) to any position of the window selects the kept states and
/// truncates the attention KV by offset: the speculative engine's direct rollback. `Clone` is a
/// buffer-sharing snapshot (a clone held across a forward makes the attention KV's in-place write
/// copy its block); the engine never takes one.
#[derive(Debug)]
pub struct Qwen35Cache {
    layers: Vec<Qwen35LayerCache>,
}

impl Clone for Qwen35Cache {
    fn clone(&self) -> Self {
        #[cfg(test)]
        CACHE_CLONES.with(|c| c.set(c.get() + 1));
        Self {
            layers: self.layers.clone(),
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only count of [`Qwen35Cache`] clones on this thread.
    static CACHE_CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Run `f`, returning how many times a [`Qwen35Cache`] was cloned on this thread meanwhile.
#[cfg(test)]
pub(crate) fn counting_cache_clones<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = CACHE_CLONES.with(|c| c.get());
    let out = f();
    (out, CACHE_CLONES.with(|c| c.get()) - before)
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

    /// Drop every attention layer's campaign ownership events
    /// ([`ContiguousKvCache::clear_events`]) — a cache kept as a prefix-cache entry carries none.
    pub(crate) fn clear_kv_events(&mut self) {
        for layer in &mut self.layers {
            if let Qwen35LayerCache::Attn(slot) = layer {
                slot.kv.clear_events();
            }
        }
    }

    /// The dtypes of each full-attention layer's cached keys and values, in layer order, skipping
    /// a layer that has cached nothing yet. Inspection only: since sc-20671 they are the compute
    /// dtype whatever dtype the snapshot stores its quantized scales in.
    pub fn attention_kv_dtypes(&self) -> Result<Vec<(Dtype, Dtype)>> {
        let mut dtypes = Vec::new();
        for layer in &self.layers {
            if let Qwen35LayerCache::Attn(slot) = layer {
                if let Some((keys, values)) = slot.kv.peek(0)? {
                    dtypes.push((keys.dtype(), values.dtype()));
                }
            }
        }
        Ok(dtypes)
    }

    /// Drop all cached state.
    pub fn reset(&mut self) -> Result<()> {
        for l in &mut self.layers {
            match l {
                Qwen35LayerCache::Delta(c) => c.reset(),
                Qwen35LayerCache::Attn(a) => a.reset()?,
            }
        }
        Ok(())
    }

    /// Bytes the cache's arrays hold: every attention layer's KV block buffers plus every linear
    /// layer's resident recurrent bytes ([`DeltaNetCache::resident_bytes`]: the live conv tail
    /// and recurrent state, plus any checkpoint window still open) — what the prefix cache charges
    /// an entry holding this cache (story sc-24437). A stored entry's window is closed, so it is
    /// charged exactly its live state.
    pub fn bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| match l {
                Qwen35LayerCache::Attn(a) => a.kv.bytes(),
                Qwen35LayerCache::Delta(c) => c.resident_bytes() as u64,
            })
            .sum()
    }

    /// Open a checkpoint window at the current position in every DeltaNet layer: the state here
    /// and after every token of each forward until the window closes stays restorable, so
    /// [`truncate`](Self::truncate) can return to any of those positions without a forward. The
    /// speculative engine opens one per verify step (a draft model's proposal spans several
    /// forwards in one window); prefill and plain decode never do, so they keep the
    /// final-state-only recurrence. The window records at most `max_tokens` tokens — the
    /// `width + 1` of a speculative width, what [`Qwen35Model::checkpoint_ring_bytes`] prices: a
    /// forward past that is [`Error::CheckpointWindowFull`], refused by the first DeltaNet layer
    /// (layer 0 of every hybrid schedule) before any layer's cache is written.
    pub fn arm_checkpoints(&mut self, max_tokens: i32) {
        for l in &mut self.layers {
            if let Qwen35LayerCache::Delta(c) = l {
                c.arm_checkpoints(max_tokens);
            }
        }
    }

    /// Tokens the DeltaNet layers' checkpoint windows hold, summed over the layers (`0` when no
    /// layer records).
    #[cfg(test)]
    pub(crate) fn checkpointed_tokens(&self) -> i32 {
        self.layers
            .iter()
            .map(|l| match l {
                Qwen35LayerCache::Delta(c) => c.checkpointed_tokens(),
                Qwen35LayerCache::Attn(_) => 0,
            })
            .sum()
    }

    /// Every DeltaNet layer's live `(conv_state, ssm_state)` on the host, in layer order.
    #[cfg(test)]
    pub(crate) fn delta_states(&self) -> Vec<(Vec<f32>, Vec<f32>)> {
        let host = |a: &Array| {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        self.layers
            .iter()
            .filter_map(|l| match l {
                Qwen35LayerCache::Delta(c) => {
                    c.live_state().map(|(conv, ssm)| (host(conv), host(ssm)))
                }
                Qwen35LayerCache::Attn(_) => None,
            })
            .collect()
    }

    /// Ask the next forward — a prefill, outside any checkpoint window — to capture every
    /// DeltaNet layer's state after `position`, a boundary strictly inside it, so
    /// [`at_boundary`](Self::at_boundary) can copy the cache as it was there without splitting
    /// the prefill into two forwards (sc-24446).
    pub fn capture_boundary(&mut self, position: i32) {
        for l in &mut self.layers {
            if let Qwen35LayerCache::Delta(c) = l {
                c.capture_at(position);
            }
        }
    }

    /// Drop every DeltaNet layer's pending [`capture_boundary`](Self::capture_boundary) — after a
    /// forward that failed part-way, whose later layers never consumed it.
    pub fn clear_boundary_capture(&mut self) {
        for l in &mut self.layers {
            if let Qwen35LayerCache::Delta(c) = l {
                c.clear_capture();
            }
        }
    }

    /// The cache as it was after its first `len` positions, `len` below its length: the attention
    /// KV's first `len` positions in their own buffers (causal: no later position wrote them) and
    /// the DeltaNet states the last forward captured at `len`
    /// ([`capture_boundary`](Self::capture_boundary)) — the cache a prefill that stopped at `len`
    /// would have left, with no checkpoint window. A layer holding no state at `len` is an error.
    pub fn at_boundary(&self, len: i32) -> Result<Self> {
        let layers = self
            .layers
            .iter()
            .map(|l| {
                Ok(match l {
                    Qwen35LayerCache::Delta(c) => {
                        Qwen35LayerCache::Delta(c.at_captured(len).ok_or_else(|| {
                            Error::Msg(format!(
                                "Qwen35Cache: no recurrent state captured at {len} (the cache \
                                 is at {})",
                                c.offset()
                            ))
                        })?)
                    }
                    Qwen35LayerCache::Attn(a) => Qwen35LayerCache::Attn(a.prefix(len)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The captured conv tails are slices of this forward's conv input: evaluate them with the
        // KV copies (`prefix` starts those) so the entry holds no pending work for a later request.
        async_eval(layers.iter().filter_map(|l| match l {
            Qwen35LayerCache::Delta(c) => c.live_state().map(|(conv, _)| conv),
            Qwen35LayerCache::Attn(_) => None,
        }))?;
        Ok(Self { layers })
    }

    /// Every attention layer's live `(keys, values)` on the host, in layer order.
    #[cfg(test)]
    pub(crate) fn attn_states(&self) -> Vec<(Vec<f32>, Vec<f32>)> {
        let host = |a: &Array| {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        self.layers
            .iter()
            .filter_map(|l| match l {
                Qwen35LayerCache::Attn(a) => {
                    a.kv.peek(0).unwrap().map(|(k, v)| (host(&k), host(&v)))
                }
                Qwen35LayerCache::Delta(_) => None,
            })
            .collect()
    }

    /// Drop every DeltaNet layer's checkpoint window, open or closed
    /// ([`DeltaNetCache::discard_checkpoints`]): the cache holds only its live state at
    /// [`offset`](Self::offset) — what a cross-turn prefix-cache entry keeps (sc-24437).
    pub fn discard_checkpoints(&mut self) {
        for l in &mut self.layers {
            if let Qwen35LayerCache::Delta(c) = l {
                c.discard_checkpoints();
            }
        }
    }

    /// Close the checkpoint window: later forwards stop recording (and drop it).
    pub fn close_checkpoints(&mut self) {
        for l in &mut self.layers {
            if let Qwen35LayerCache::Delta(c) = l {
                c.close_checkpoints();
            }
        }
    }

    /// The positions [`truncate`](Self::truncate) can return to besides `0` and the current one:
    /// the checkpoint window's (the ring's depth).
    pub fn restorable(&self) -> std::ops::Range<i32> {
        let offset = self.offset();
        self.layers
            .iter()
            .filter_map(|l| match l {
                Qwen35LayerCache::Delta(c) => Some(c.restorable()),
                Qwen35LayerCache::Attn(_) => None,
            })
            .reduce(|a, b| a.start.max(b.start)..a.end.min(b.end))
            .unwrap_or(0..offset)
    }

    /// Keep positions `0..len` and close the checkpoint window: `len == offset()` only closes it
    /// and `0` is a reset; otherwise every DeltaNet layer restores the state it kept at `len` and
    /// the attention KV drops the positions past it by offset (its buffer is kept and written in
    /// place by the next forward). A `len` outside the checkpoint window
    /// ([`restorable`](Self::restorable)) is [`Error::Unsupported`] and leaves the cache
    /// untouched. The restored states are scheduled for evaluation (asynchronously) so the
    /// checkpoints they were read from free as soon as the next forward drops the window.
    pub fn truncate(&mut self, len: i32) -> Result<()> {
        let offset = self.offset();
        if len < 0 || len > offset {
            return Err(Error::Msg(format!(
                "Qwen35Cache: cannot truncate to {len} with {offset} positions cached"
            )));
        }
        if len == offset {
            self.close_checkpoints();
            return Ok(());
        }
        if len == 0 {
            return self.reset();
        }
        if let Some(Qwen35LayerCache::Delta(c)) = self
            .layers
            .iter()
            .find(|l| matches!(l, Qwen35LayerCache::Delta(c) if !c.can_rollback_to(len)))
        {
            let held = c.restorable();
            return Err(Error::Unsupported(format!(
                "Qwen35Cache: cannot truncate {offset} positions to {len}: the DeltaNet \
                 checkpoint window holds positions {}..{}",
                held.start, held.end
            )));
        }
        let mut restored = Vec::new();
        for l in &mut self.layers {
            match l {
                Qwen35LayerCache::Delta(c) => {
                    c.rollback_to(len)?;
                    if let Some((conv, ssm)) = c.live_state() {
                        restored.extend([conv.clone(), ssm.clone()]);
                    }
                }
                Qwen35LayerCache::Attn(a) => a.truncate(len)?,
            }
        }
        async_eval(&restored)?;
        Ok(())
    }

    /// Bytes of recurrent state the DeltaNet layers hold: every live state plus the checkpoint
    /// ring (the attention KV is excluded) — what admission prices as the request's recurrent
    /// footprint.
    pub fn recurrent_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                Qwen35LayerCache::Delta(c) => c.resident_bytes(),
                Qwen35LayerCache::Attn(_) => 0,
            })
            .sum()
    }

    /// The attention layers' K and V block-buffer addresses, in layer order — equal across steps
    /// exactly when the forwards wrote them in place.
    #[cfg(test)]
    pub(crate) fn attn_buffer_addresses(&self) -> Vec<usize> {
        self.layers
            .iter()
            .filter_map(|l| match l {
                Qwen35LayerCache::Attn(a) => Some(a.kv.buffer_addresses()),
                Qwen35LayerCache::Delta(_) => None,
            })
            .flatten()
            .collect()
    }
}

/// A loaded Qwen3.6 (`qwen3_5`) hybrid decoder.
#[derive(Debug)]
pub struct Qwen35Model {
    embed_tokens: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: Array,
    lm_head: Projection,
    rope: Rope,
    mtp: Option<MtpPredictor>,
    /// Why a configured native MTP head was not built (E2): the snapshot stores none of it, or it
    /// is a variant this runtime does not run. Leads with `mtp:`; the provider names it in
    /// `LoadReport::fallbacks`.
    mtp_fallback: Option<String>,
    cfg: Qwen35Config,
    eps: f32,
    quantized: bool,
    prism: bool,
}

#[derive(Debug)]
enum TokenEmbedding {
    Dense(Array),
    Prism(PrismEmbedding),
}

impl TokenEmbedding {
    fn push_arrays(&self, out: &mut Vec<Array>) {
        match self {
            Self::Dense(weight) => out.push(weight.clone()),
            Self::Prism(embedding) => embedding.push_arrays(out),
        }
    }

    fn forward(&self, input_ids: &Array) -> Result<Array> {
        match self {
            Self::Dense(weight) => embed(weight, input_ids),
            Self::Prism(embedding) => embedding.forward(input_ids),
        }
    }
}

/// Cache for Qwen3.8's in-checkpoint multi-token predictor. Each predictor layer owns a full-
/// attention KV stream; the frozen 27B has one layer, which is cycled for every draft token.
#[derive(Clone, Debug)]
pub struct MtpCache {
    layers: Vec<AttnKv>,
    steps: usize,
}

impl MtpCache {
    /// Bytes the predictor's KV block buffers hold (story sc-24437).
    pub fn bytes(&self) -> u64 {
        self.layers.iter().map(|a| a.kv.bytes()).sum()
    }
}

/// Qwen3.8's optional speculative predictor. Embeddings and the LM head are shared with the target
/// model; all tensors stored under `mtp.*` are owned here and required when config enables MTP.
#[derive(Debug)]
struct MtpPredictor {
    fc: Projection,
    pre_fc_norm_embedding: Array,
    pre_fc_norm_hidden: Array,
    layers: Vec<DecoderLayer>,
    norm: Array,
}

impl MtpPredictor {
    /// The predictor's own non-layer arrays, then one group per predictor layer (sc-24446).
    fn param_groups(&self) -> Vec<Vec<Array>> {
        let mut head = Vec::new();
        self.fc.push_arrays(&mut head);
        head.extend([
            self.pre_fc_norm_embedding.clone(),
            self.pre_fc_norm_hidden.clone(),
            self.norm.clone(),
        ]);
        let mut groups = vec![head];
        groups.extend(self.layers.iter().map(DecoderLayer::arrays));
        groups
    }
}

impl Qwen35Model {
    /// The model's arrays in build order, grouped as load admission prices them (sc-24446): the
    /// arrays outside the layer stack, then one group per decoder layer, then the MTP predictor's
    /// own non-layer arrays and one group per predictor layer. [`Weights::materialize_groups`]
    /// evaluates and releases one group at a time.
    pub fn param_groups(&self) -> Vec<Vec<Array>> {
        let mut top = Vec::new();
        self.embed_tokens.push_arrays(&mut top);
        top.push(self.norm.clone());
        self.lm_head.push_arrays(&mut top);
        let mut groups = vec![top];
        groups.extend(self.layers.iter().map(DecoderLayer::arrays));
        if let Some(mtp) = &self.mtp {
            groups.extend(mtp.param_groups());
        }
        groups
    }

    /// The parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Whether the large projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The decoder's compute dtype (bf16): its activations, logits, and full-attention K/V. The
    /// linear-attention recurrence alone accumulates in f32.
    pub fn compute_dtype(&self) -> Dtype {
        compute()
    }

    /// Whether projections use the packed Prism Hadamard path.
    pub fn is_prism(&self) -> bool {
        self.prism
    }

    /// Whether this snapshot loaded a complete native multi-token predictor.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Why the config's native MTP head was not built, when it was not (E2): the named `mtp:`
    /// load fallback — the snapshot stores none of its tensors, or it is a variant this runtime
    /// does not run. `None` when the head was built or none was configured.
    pub fn mtp_fallback(&self) -> Option<&str> {
        self.mtp_fallback.as_deref()
    }

    /// A fresh MTP attention cache. Returns `None` for ordinary Qwen3.5/3.6 snapshots without the
    /// optional predictor.
    pub fn new_mtp_cache(&self) -> Option<MtpCache> {
        self.mtp.as_ref().map(|mtp| MtpCache {
            layers: (0..mtp.layers.len()).map(|_| AttnKv::default()).collect(),
            steps: 0,
        })
    }

    /// The bytes a speculative run of `width` drafts per step adds to the recurrent footprint
    /// through the DeltaNet checkpoint ring (sc-24435), over the live state: per linear layer,
    /// `width + 2` slots of one recurrent state (`Hv·Dv·Dk`) plus `K` conv rows (`conv_dim`), all
    /// at 4 bytes per element. That covers the most a checkpoint window holds for a verify step of
    /// `width + 1` tokens — the state at the window start, the `width + 1` per-token states, and
    /// the conv tails (`K - 1` rows) plus the forward's conv input (`width + K` rows) — whatever
    /// the conv dtype. `None` on overflow (the caller fails closed).
    pub fn checkpoint_ring_bytes(&self, width: usize) -> Option<u64> {
        let c = &self.cfg;
        let n = |v: i32| u64::try_from(v).ok();
        let linear = (0..c.num_layers).filter(|&i| c.is_linear(i)).count() as u64;
        let state = n(c.linear_num_value_heads)?
            .checked_mul(n(c.linear_value_head_dim)?)?
            .checked_mul(n(c.linear_key_head_dim)?)?;
        let conv_dim = n(c.linear_num_key_heads)?
            .checked_mul(n(c.linear_key_head_dim)?)?
            .checked_mul(2)?
            .checked_add(n(c.linear_num_value_heads)?.checked_mul(n(c.linear_value_head_dim)?)?)?;
        let conv = conv_dim.checked_mul(n(c.linear_conv_kernel_dim)?)?;
        let slots = u64::try_from(width).ok()?.checked_add(2)?;
        linear
            .checked_mul(slots)?
            .checked_mul(state.checked_add(conv)?)?
            .checked_mul(4)
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
    fn hidden(&self, input_ids: &Array, cache: &mut Qwen35Cache, offset: i32) -> Result<Array> {
        let h = self.embed_tokens.forward(input_ids)?.as_dtype(compute())?;
        let s = h.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, compute())?;
        self.hidden_from_embeds(&h, &cos, &sin, cache)
    }

    /// Final-normalized target hidden states and logits for every supplied position. Qwen3.8 MTP
    /// consumes the same final-normalized states the upstream runtime exposes to its predictor.
    pub(crate) fn hidden_and_logits(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<(Array, Array)> {
        let h = self.hidden(input_ids, cache, offset)?;
        let normalized = rms_norm(&h, &self.norm, self.eps)?;
        let logits = self.lm_head.forward(&normalized)?;
        Ok((normalized, logits))
    }

    /// MTP prompt prefill retains every normalized hidden row but projects only the last row.
    pub(crate) fn prefill_hidden_and_last_logits(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<(Array, Array)> {
        let h = self.hidden(input_ids, cache, offset)?;
        let normalized = rms_norm(&h, &self.norm, self.eps)?;
        let s = normalized.shape()[1];
        let last = normalized.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.lm_head.forward(&last)?;
        Ok((
            normalized,
            logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?,
        ))
    }

    /// Final-normalized target hidden states and logits for a fused multimodal prompt.
    ///
    /// The returned hidden rows are the authoritative target states used to seed Qwen3.8's MTP
    /// predictor. Keeping this beside the ordinary token-id path prevents visual placeholders from
    /// being re-embedded as token ids while building the predictor cache.
    #[cfg(test)]
    pub(crate) fn hidden_and_logits_from_embeds_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<(Array, Array)> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            compute(),
        )?;
        let h0 = embeds.as_dtype(compute())?;
        let layers = &self.layers;
        let cache_layers = &mut cache.layers;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &cos, &sin, &mut cache_layers[i]),
        )?;
        let normalized = rms_norm(&h, &self.norm, self.eps)?;
        let logits = self.lm_head.forward(&normalized)?;
        Ok((normalized, logits))
    }

    /// Multimodal MTP prefill with DeepStack/M-RoPE and only the last target logit row.
    pub(crate) fn prefill_hidden_and_last_logits_from_embeds_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<(Array, Array)> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            compute(),
        )?;
        let h0 = embeds.as_dtype(compute())?;
        let layers = &self.layers;
        let cache_layers = &mut cache.layers;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &cos, &sin, &mut cache_layers[i]),
        )?;
        let normalized = rms_norm(&h, &self.norm, self.eps)?;
        let s = normalized.shape()[1];
        let last = normalized.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.lm_head.forward(&last)?;
        Ok((
            normalized,
            logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?,
        ))
    }

    /// Advance the in-checkpoint predictor from a token embedding plus its aligned target/previous
    /// hidden state. Returns the predictor hidden state and next-token logits for the last position.
    pub(crate) fn mtp_step(
        &self,
        input_ids: &Array,
        hidden_states: &Array,
        cache: &mut MtpCache,
        offset: i32,
    ) -> Result<(Array, Array)> {
        let embeds = self.embed_tokens.forward(input_ids)?.as_dtype(compute())?;
        let s = embeds.shape()[1];
        let positions = (offset..offset + s).collect::<Vec<_>>();
        self.mtp_step_from_embeds(
            &embeds,
            hidden_states,
            cache,
            [&positions, &positions, &positions],
        )
    }

    /// Advance the predictor from already-fused token/visual embeddings at explicit M-RoPE
    /// positions. This is the visual-prompt counterpart of [`Self::mtp_step`].
    pub(crate) fn mtp_step_from_embeds(
        &self,
        embeds: &Array,
        hidden_states: &Array,
        cache: &mut MtpCache,
        positions: [&[i32]; 3],
    ) -> Result<(Array, Array)> {
        let normalized = self.mtp_hidden_from_embeds(embeds, hidden_states, cache, positions)?;
        let logits = self.lm_head.forward(&normalized)?;
        let last = logits.take_axis(Array::from_slice(&[normalized.shape()[1] - 1], &[1]), 1)?;
        Ok((
            normalized,
            last.reshape(&[last.shape()[0], self.cfg.vocab_size])?,
        ))
    }

    /// Warm the predictor cache with validated prompt pairs without computing unused logits.
    pub(crate) fn mtp_warm_from_embeds(
        &self,
        embeds: &Array,
        hidden_states: &Array,
        cache: &mut MtpCache,
        positions: [&[i32]; 3],
    ) -> Result<Array> {
        self.mtp_hidden_from_embeds(embeds, hidden_states, cache, positions)
    }

    fn mtp_hidden_from_embeds(
        &self,
        embeds: &Array,
        hidden_states: &Array,
        cache: &mut MtpCache,
        positions: [&[i32]; 3],
    ) -> Result<Array> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::Msg("qwen3_5 MTP requested but predictor is not loaded".into())
        })?;
        if mtp.layers.len() != 1 || cache.layers.len() != 1 {
            return Err(Error::Config(
                "qwen3_5 MTP currently requires exactly one predictor layer".into(),
            ));
        }
        if embeds.shape() != hidden_states.shape() {
            return Err(Error::Msg(format!(
                "qwen3_5 MTP token/hidden shape mismatch: {:?} vs {:?}",
                embeds.shape(),
                hidden_states.shape()
            )));
        }
        let embeds = embeds.as_dtype(compute())?;
        let e = rms_norm(&embeds, &mtp.pre_fc_norm_embedding, self.eps)?;
        let h = rms_norm(hidden_states, &mtp.pre_fc_norm_hidden, self.eps)?;
        let fused = concatenate_axis(&[&e, &h], 2)?;
        let x = mtp.fc.forward(&fused)?;
        let s = x.shape()[1];
        if positions.iter().any(|row| row.len() != s as usize) {
            return Err(Error::Msg(format!(
                "qwen3_5 MTP position length mismatch: sequence {s}, positions [{}, {}, {}]",
                positions[0].len(),
                positions[1].len(),
                positions[2].len()
            )));
        }
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            compute(),
        )?;
        let mut slot = Qwen35LayerCache::Attn(std::mem::take(&mut cache.layers[0]));
        let out = mtp.layers[0].forward(&x, &cos, &sin, &mut slot)?;
        cache.layers[0] = match slot {
            Qwen35LayerCache::Attn(kv) => kv,
            Qwen35LayerCache::Delta(_) => unreachable!("MTP layer is always full attention"),
        };
        cache.steps += s as usize;
        let normalized = rms_norm(&out, &mtp.norm, self.eps)?;
        Ok(normalized)
    }

    /// Run the decoder stack over precomputed input `embeds` `[B, S, hidden]` with the given RoPE
    /// tables, returning the final hidden states `[B, S, hidden]`. The token-id path ([`Self::hidden`])
    /// and the multimodal embeds path ([`Self::decode_logits_from_embeds`]) share this.
    fn hidden_from_embeds(
        &self,
        embeds: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Qwen35Cache,
    ) -> Result<Array> {
        let mut h = embeds.clone();
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, cos, sin, slot)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` over hidden states `[B, n, hidden]` → logits `[B, n, vocab]`.
    fn project(&self, h: &Array) -> Result<Array> {
        let normed = rms_norm(h, &self.norm, self.eps)?;
        self.lm_head.forward(&normed)
    }

    /// Project the **last** position of `h` `[B, S, hidden]` → logits `[B, vocab]`.
    fn project_last(&self, h: &Array) -> Result<Array> {
        let s = h.shape()[1];
        let last = h.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?; // [B,1,hidden]
        let logits = self.project(&last)?; // [B,1,vocab]
        Ok(logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?)
    }

    /// Run the decoder over `input_ids` `[B, S]` at sequence `offset`, returning logits for **every**
    /// position `[B, S, vocab]`.
    pub fn forward(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Array> {
        let h = self.hidden(input_ids, cache, offset)?;
        self.project(&h)
    }

    /// Run the decoder and return logits for the **last** position only, `[B, vocab]` — the
    /// [`crate::decode::Decode::step`] contract (prefill + single-token decode).
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Array> {
        let h = self.hidden(input_ids, cache, offset)?;
        self.project_last(&h)
    }

    /// Embed token ids `[B, S]` → `[B, S, hidden]` in the compute dtype — the splice point where the
    /// multimodal path overwrites image-token rows with the encoder's projected patch features
    /// ([`Self::splice_image_features`]).
    pub fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Ok(self.embed_tokens.forward(input_ids)?.as_dtype(compute())?)
    }

    /// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with `image_features`
    /// `[num_image_tokens, hidden]` (the vision encoder's projected, merged patch rows), in sequence
    /// order. The number of image-token positions must equal the feature-row count.
    pub fn splice_image_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        image_features: &Array,
        image_token_id: i32,
    ) -> Result<Array> {
        self.splice_vision_features(embeds, input_ids, image_features, &[image_token_id])
    }

    /// Replace every row whose id is **any** of `placeholder_tokens` (`<|image_pad|>` and/or
    /// `<|video_pad|>`) with the next `vision_features` row, in sequence order — the multimodal splice
    /// for a mixed image+video prompt. Features must be concatenated in the same order the
    /// placeholders appear. Reduces to [`Self::splice_image_features`] for a single token.
    pub fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        // Invalid-input diagnostics use the shared helper's `vlm splice` prefix; successful output
        // is byte-identical to the former Qwen3.5-local implementation.
        deepstack::splice_vision_features(
            embeds,
            input_ids,
            vision_features,
            placeholder_tokens,
            self.cfg.hidden_size,
            compute(),
        )
    }

    /// Compute the interleaved M-RoPE 3-D position rows (`get_rope_index`, B=1) for `input_ids`
    /// containing `image_grid_thw`-described `image_token_id` runs, plus the `mrope_delta`
    /// (`max_position + 1 − len`) the decode loop adds to continue positions after the prompt.
    ///
    /// The image-only entry point — the multimodal prefill path. See
    /// [`Self::mrope_positions_mm`] for the general (image + video) port; this forwards to it with
    /// no video runs.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        self.mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            &[],
            image_token_id, // unused: no video runs
            spatial_merge_size,
        )
    }

    /// The full Qwen3-VL `get_rope_index` port (B=1), covering **both** image and video vision runs.
    ///
    /// Text tokens advance all three axes (t,h,w) by 1. A vision run lays its tokens over the
    /// `(t, h/merge, w/merge)` grid — temporal index per frame, height = row, width = col — offset by
    /// the shared cursor `st_idx` (`= max(prev block) + 1`), then advances the cursor by
    /// `max(grid_t, h/merge, w/merge)` (the reference's `llm_positions.max() + 1`).
    ///
    /// **Qwen3-VL video is the synthetic time axis.** Unlike a single multi-`t` block, Qwen3-VL
    /// separates frames with timestamp text, so the processor emits one `video_token_id` run **per
    /// frame** and the model splits `video_grid_thw` via `repeat_interleave(t); t ← 1`. Each frame is
    /// thus its own `gt = 1` block, and the temporal index **resets to 0 each frame** (the frames are
    /// ordered only by the advancing cursor). We mirror that by expanding each `[t, h, w]` video grid
    /// into `t` per-frame `[1, h, w]` blocks, one per consecutive video-token run. Image grids are
    /// consumed one run per `image_grid_thw` entry (Qwen3-VL images are always `gt = 1`).
    ///
    /// `spatial_merge_size` comes from the vision config. Returns `(t_row, h_row, w_row, mrope_delta)`.
    pub fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        // Invalid-input diagnostics use the shared helper's `vlm mrope` prefix; successful position
        // rows and continuation delta are identical to the former Qwen3.5-local implementation.
        deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` (text embeds with image
    /// features spliced in) using **interleaved M-RoPE** from the explicit 3-D `positions`
    /// (temporal/height/width rows, each length `S`), returning last-position logits `[1, vocab]`.
    /// With all three rows equal (text-only) this is bit-identical to [`Self::decode_logits`].
    pub fn decode_logits_from_embeds(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
    ) -> Result<Array> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            compute(),
        )?;
        let h = self.hidden_from_embeds(&embeds.as_dtype(compute())?, &cos, &sin, cache)?;
        self.project_last(&h)
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` with interleaved M-RoPE
    /// (the multimodal embeds path) **and DeepStack feature fusion**: after decoder layer `i`, for
    /// `i < deepstack.len()`, the `i`-th tapped/merged ViT feature set is added to the visual-token
    /// rows of the running hidden states (`visual_pos_mask[p]` marks an image-token position).
    ///
    /// This is the Qwen3-VL DeepStack seam (`Qwen3VLTextModel.forward` + `_deepstack_process`):
    /// `deepstack` carries the `deepstack_features` produced by the vision tower at its tap layers,
    /// projected to `out_hidden_size == hidden`, and is injected into the *first* `len(deepstack)`
    /// decoder layers (HF indexes by decoder position, not by the vision tap index). Returns
    /// last-position logits `[1, vocab]`. With an empty `deepstack` this equals
    /// [`Self::decode_logits_from_embeds`].
    pub fn decode_logits_from_embeds_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            compute(),
        )?;
        let h0 = embeds.as_dtype(compute())?;
        let layers = &self.layers;
        let cache_layers = &mut cache.layers;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &cos, &sin, &mut cache_layers[i]),
        )?;
        self.project_last(&h)
    }

    /// Build from a loaded checkpoint (dense). See [`Qwen35Model::from_weights_with`].
    pub fn from_weights(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the large projections on load.
    ///
    /// `prefix` is the **decoder root** path: keys are read as `{prefix}.embed_tokens.weight`,
    /// `{prefix}.norm.weight`, `{prefix}.layers.{i}.…`. For the VLM-wrapped Qwen3.6 checkpoint this
    /// is `model.language_model`; `lm_head.weight` lives at the **checkpoint root** (untied), not
    /// under the prefix. `quant` (Q4/Q8) is applied to the big matmuls (in/out projections,
    /// attention q/k/v/o, MLP); the per-head decay/delta projections, conv, `A_log`/`dt_bias`, and
    /// all norms stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        Self::materialized(w, Self::build_lazy(w, prefix, cfg, quant)?)
    }

    /// Build the Qwen3.5 decoder directly over a validated Prism MLX 2-bit pack.
    pub fn from_prism_weights(w: &Weights, cfg: Qwen35Config, pack: &PrismMlxPack) -> Result<Self> {
        Self::materialized(w, Self::build_prism_lazy(w, cfg, pack)?)
    }

    /// [`Qwen35Model::from_weights_with`] without evaluating anything: every array is a lazy graph
    /// over `w`'s sources. The provider materializes it group by group with
    /// [`Weights::materialize_groups`], releasing each group's consumed sources (sc-24446); a
    /// caller that keeps the map uses [`Qwen35Model::from_weights_with`] instead.
    pub fn build_lazy(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        Self::from_weights_layout(w, prefix, cfg, quant, None)
    }

    /// [`Qwen35Model::from_prism_weights`] without evaluating anything (see
    /// [`Qwen35Model::build_lazy`]).
    pub(crate) fn build_prism_lazy(
        w: &Weights,
        cfg: Qwen35Config,
        pack: &PrismMlxPack,
    ) -> Result<Self> {
        Self::from_weights_layout(w, "language_model.model", cfg, None, Some(pack))
    }

    /// The caller-owned-map load boundary: every source this constructor read is verified
    /// resident (sc-22414) and every derived array evaluated (sc-24446) before the model is
    /// returned, so no forward reads a weight or builds one inside its command stream (sc-24245).
    fn materialized(w: &Weights, model: Self) -> Result<Self> {
        w.verify_accessed_gpu_view()?;
        crate::primitives::weights::eval_groups(&model.param_groups())?;
        Ok(model)
    }

    fn from_weights_layout(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
        prism: Option<&PrismMlxPack>,
    ) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let req = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(compute())?) };
        // Dense HF Qwen3.6 norms are zero-centered. Frozen Prism/Bonsai artifacts have already
        // converted every ordinary RMSNorm tensor to its direct multiplier and must remain raw.
        let norm_w = |key: String| -> Result<Array> {
            let t = req(key)?;
            checkpoint_norm_weight(t, prism.is_some())
        };
        let stored_quant = cfg.quantization;
        if prism.is_some() && (stored_quant.is_some() || quant.is_some()) {
            return Err(Error::Config(
                "Prism packed weights cannot be combined with generic Q4/Q8 quantization".into(),
            ));
        }
        if stored_quant.is_some() && quant.is_some() {
            return Err(Error::Config(
                "qwen3_5 cannot combine stored quantized weights with load-time quantization"
                    .into(),
            ));
        }
        let saw_stored = Cell::new(false);
        let proj_q = |key: String| -> Result<Projection> {
            if let Some(pack) = prism {
                return Ok(Projection::Prism(pack.linear(w, &key, compute())?));
            }
            load_projection(w, &key, stored_quant, quant, &saw_stored)
        };
        let proj_dense = |key: String| -> Result<Projection> {
            Projection::load(w.require(&key)?.as_dtype(compute())?, None)
        };
        let dp = |s: &str| format!("{prefix}.{s}");

        let embed_key = dp("embed_tokens.weight");
        let embed_weight = if prism.is_none() {
            Some(req(embed_key.clone())?)
        } else {
            None
        };
        let embed_tokens = if let Some(pack) = prism {
            TokenEmbedding::Prism(pack.embedding(w, &embed_key)?)
        } else {
            TokenEmbedding::Dense(embed_weight.clone().expect("dense embedding"))
        };
        let norm = norm_w(dp("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            if prism.is_some() {
                return Err(Error::Config(
                    "Prism requires an explicit packed lm_head; tied embeddings are unsupported"
                        .into(),
                ));
            }
            Projection::load(embed_weight.expect("dense embedding"), None)?
        } else if let Some(pack) = prism {
            Projection::Prism(pack.linear(w, "language_model.lm_head.weight", compute())?)
        } else {
            Projection::load(req("lm_head.weight".to_string())?, None)?
        };

        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| dp(&format!("layers.{i}.{s}"));
            let mixer = if cfg.is_linear(i) {
                // conv1d.weight is [conv_dim, 1, K] (HF) → squeeze the singleton to [conv_dim, K].
                let conv_weight = req(lp("linear_attn.conv1d.weight"))?
                    .reshape(&[conv_dim, cfg.linear_conv_kernel_dim])?;
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
                    num_k_heads: cfg.linear_num_key_heads,
                    num_v_heads: cfg.linear_num_value_heads,
                    head_k_dim: cfg.linear_key_head_dim,
                    head_v_dim: cfg.linear_value_head_dim,
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel: cfg.linear_conv_kernel_dim,
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
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    eps,
                })
            };
            let ffn = build_ffn(w, &dp(&format!("layers.{i}.")), &cfg, quant, &proj_q)?;
            layers.push(DecoderLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp("post_attention_layernorm.weight"))?,
                mixer,
                ffn,
                eps,
            });
        }

        // The configured native head, by the rule Candle loads by too (E2, E8): built when
        // complete; a declared head the snapshot does not carry, or a variant this runtime does
        // not run (more than one layer, dedicated embeddings), loads the target plain with the
        // reason named — and the config then prices no head state; a PARTIAL `mtp.*` set still
        // fails the load in `build_mtp_predictor` (a missing tensor is integrity, not absence).
        let mut cfg = cfg;
        let mut mtp_fallback = None;
        let mtp = match core_llm::native_mtp_plan(
            cfg.mtp_num_hidden_layers,
            cfg.mtp_use_dedicated_embeddings,
            w.keys().any(|key| key.starts_with("mtp.")),
        )
        .map_err(|e| Error::Config(e.to_string()))?
        {
            core_llm::NativeMtp::Absent => None,
            // The predictor layer carries the body's FFN choice: the dense MLP (27B) or the
            // sparse-MoE block (35B-A3B), in either expert layout (vLLM `qwen3_5_mtp.py`,
            // sc-24438).
            core_llm::NativeMtp::Build => Some(build_mtp_predictor(
                &cfg,
                "mtp.",
                &|key| proj_q(key),
                &|key| norm_w(key),
                &|lp| build_ffn(w, lp, &cfg, quant, &proj_q),
            )?),
            core_llm::NativeMtp::Fallback(why) => {
                mtp_fallback = Some(why);
                cfg.mtp_num_hidden_layers = 0;
                None
            }
        };

        let rope = Rope::partial(cfg.rotary_dim(), cfg.rope_theta, false);
        let model = Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            mtp,
            mtp_fallback,
            eps,
            cfg,
            quantized: prism.is_some() || quant.is_some() || saw_stored.get(),
            prism: prism.is_some(),
        };
        Ok(model)
    }
}

/// The FFN of the decoder or MTP predictor layer whose tensors live under `lp` (the layer prefix
/// with its trailing dot: `model.language_model.layers.3.`, `mtp.layers.0.`): a dense SwiGLU (27B)
/// or the sparse-MoE block (35B-A3B). The body and the MTP head share it, so a predictor layer
/// takes exactly the body's FFN choice and expert layout (sc-24438). `proj_q` is the loader's
/// projection reader (stored-quantized, load-time-quantized, Prism or dense).
///
/// MoE experts stay stacked for the gathered matmul. The fused layout (Qwen3.6) stores
/// `experts.gate_up_proj` `[E, 2·moe_inter, hidden]` (gate rows ‖ up rows, matching the reference
/// `linear(x, gate_up_proj[e]).chunk(2, -1)`) and `experts.down_proj` `[E, hidden, moe_inter]`.
/// A snapshot that stores each expert under its own key — the bf16 Qwen3.5 release
/// (`experts.{e}.{gate,up,down}_proj.weight`) or a quantized conversion (their `.scales` /
/// `.biases` parts) — is stacked from its per-expert projections instead.
fn build_ffn(
    w: &Weights,
    lp: &str,
    cfg: &Qwen35Config,
    quant: Option<QuantSpec>,
    proj_q: &dyn Fn(String) -> Result<Projection>,
) -> Result<Ffn> {
    let lp = |s: &str| format!("{lp}{s}");
    let req = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(compute())?) };
    let Some(moe) = &cfg.moe else {
        return Ok(Ffn::Dense(Mlp {
            gate: proj_q(lp("mlp.gate_proj.weight"))?,
            up: proj_q(lp("mlp.up_proj.weight"))?,
            down: proj_q(lp("mlp.down_proj.weight"))?,
        }));
    };
    let h = cfg.hidden_size;
    let mi = moe.moe_intermediate_size;
    let expert0 = lp("mlp.experts.0.gate_proj");
    let per_expert = ["weight", "scales", "biases"]
        .iter()
        .any(|part| w.contains(&format!("{expert0}.{part}")));
    let (gate, up, down) = if cfg.quantization.is_some() || per_expert {
        let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
        for e in 0..moe.num_experts {
            gate.push(proj_q(lp(&format!("mlp.experts.{e}.gate_proj.weight")))?);
            up.push(proj_q(lp(&format!("mlp.experts.{e}.up_proj.weight")))?);
            down.push(proj_q(lp(&format!("mlp.experts.{e}.down_proj.weight")))?);
        }
        (
            SwitchLinear::stack(gate)?,
            SwitchLinear::stack(up)?,
            SwitchLinear::stack(down)?,
        )
    } else {
        let e = moe.num_experts;
        // The fused bank stays fused (sc-24446): one gathered matmul over gate‖up rows, its
        // output split per token. Splitting the weights here held a second, gathered copy of
        // the whole bank (~44 GB on Qwen3.6-35B-A3B).
        let gate_up = req(lp("mlp.experts.gate_up_proj"))?;
        if gate_up.shape() != [e, 2 * mi, h] {
            return Err(Error::Config(format!(
                "qwen3_5_moe: `{}` is {:?}; expected [{e}, {}, {h}] from config",
                lp("mlp.experts.gate_up_proj"),
                gate_up.shape(),
                2 * mi
            )));
        }
        return Ok(Ffn::Moe(SparseMoe::fused(
            req(lp("mlp.gate.weight"))?,
            SwitchLinear::load(gate_up, quant)?,
            SwitchLinear::load(req(lp("mlp.experts.down_proj"))?, quant)?,
            shared_expert(&proj_q, &lp)?,
            Some(req(lp("mlp.shared_expert_gate.weight"))?),
            routing(moe),
        )?));
    };
    Ok(Ffn::Moe(SparseMoe::new(
        req(lp("mlp.gate.weight"))?,
        gate,
        up,
        down,
        shared_expert(&proj_q, &lp)?,
        Some(req(lp("mlp.shared_expert_gate.weight"))?),
        routing(moe),
    )?))
}

/// The always-on shared expert of the MoE layer under `lp`.
fn shared_expert(
    proj_q: &dyn Fn(String) -> Result<Projection>,
    lp: &dyn Fn(&str) -> String,
) -> Result<SwiGlu> {
    Ok(SwiGlu {
        gate: proj_q(lp("mlp.shared_expert.gate_proj.weight"))?,
        up: proj_q(lp("mlp.shared_expert.up_proj.weight"))?,
        down: proj_q(lp("mlp.shared_expert.down_proj.weight"))?,
    })
}

/// Qwen3.6-MoE routing: top-k renormalized to sum to 1.
fn routing(moe: &MoeParams) -> MoeRouting {
    MoeRouting {
        experts_per_tok: moe.experts_per_tok,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    }
}

/// Load one `[out, in]` projection that is either a stored MLX affine-quantized triple
/// (`weight`/`scales`/`biases` with `stored_quant` from `config.json`) or a dense matrix
/// (optionally quantized on load with `quant`). `saw_stored` records that a stored triple was used.
fn load_projection(
    w: &Weights,
    key: &str,
    stored_quant: Option<QuantSpec>,
    quant: Option<QuantSpec>,
    saw_stored: &Cell<bool>,
) -> Result<Projection> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    let scales_key = format!("{base}.scales");
    let biases_key = format!("{base}.biases");
    let present = [
        w.contains(key),
        w.contains(&scales_key),
        w.contains(&biases_key),
    ];
    let any_packed_part = present[1] || present[2];
    if any_packed_part || stored_quant.is_some() {
        let spec = stored_quant.ok_or_else(|| {
            Error::Config(format!(
                "snapshot stores quantized parts for `{base}` but config.json has no \
                 `quantization` block"
            ))
        })?;
        if !present.iter().all(|&part| part) {
            let names = ["weight", "scales", "biases"];
            let missing = names
                .iter()
                .zip(present)
                .filter_map(|(name, yes)| (!yes).then_some(*name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::Config(format!(
                "stored quantized projection `{base}` is incomplete; missing {missing}"
            )));
        }
        let weight = w.require(key)?.clone();
        let scales = w.require(&scales_key)?.clone();
        let biases = w.require(&biases_key)?.clone();
        let ws = weight.shape();
        let ss = scales.shape();
        let bs = biases.shape();
        if ws.len() != 2 || ss.len() != 2 || bs != ss {
            return Err(Error::Config(format!(
                "stored quantized projection `{base}` has invalid part shapes: weight \
                 {ws:?}, scales {ss:?}, biases {bs:?}"
            )));
        }
        let input = ss[1].checked_mul(spec.group_size).ok_or_else(|| {
            Error::Config(format!(
                "stored quantized projection `{base}` input overflow"
            ))
        })?;
        let packed_cols = input
            .checked_mul(spec.bits)
            .map(|n| n / 32)
            .ok_or_else(|| {
                Error::Config(format!(
                    "stored quantized projection `{base}` pack overflow"
                ))
            })?;
        if ss[0] <= 0
            || ss[1] <= 0
            || ws[0] != ss[0]
            || ws[1] != packed_cols
            || input % spec.group_size != 0
        {
            return Err(Error::Config(format!(
                "stored quantized projection `{base}` shapes do not match Q{} group {}: \
                 weight {ws:?}, scales {ss:?}, biases {bs:?}",
                spec.bits, spec.group_size
            )));
        }
        saw_stored.set(true);
        Projection::from_quantized(weight, scales, biases, spec, compute())
    } else {
        Projection::load(w.require(key)?.as_dtype(compute())?, quant)
    }
}

/// Build Qwen3.8's one-layer multi-token predictor from tensors named `{prefix}fc.weight`,
/// `{prefix}layers.0.…` etc. The geometry comes from the **target** config: the predictor runs
/// inside the target's residual stream, RoPE and vocabulary.
fn build_mtp_predictor(
    cfg: &Qwen35Config,
    prefix: &str,
    proj: &dyn Fn(String) -> Result<Projection>,
    norm: &dyn Fn(String) -> Result<Array>,
    ffn: &dyn Fn(&str) -> Result<Ffn>,
) -> Result<MtpPredictor> {
    let eps = cfg.rms_norm_eps;
    let lp = |s: &str| format!("{prefix}layers.0.{s}");
    let layer = DecoderLayer {
        input_ln: norm(lp("input_layernorm.weight"))?,
        post_ln: norm(lp("post_attention_layernorm.weight"))?,
        mixer: Mixer::Attn(Qwen35Attention {
            q_proj: proj(lp("self_attn.q_proj.weight"))?,
            k_proj: proj(lp("self_attn.k_proj.weight"))?,
            v_proj: proj(lp("self_attn.v_proj.weight"))?,
            o_proj: proj(lp("self_attn.o_proj.weight"))?,
            q_norm: norm(lp("self_attn.q_norm.weight"))?,
            k_norm: norm(lp("self_attn.k_norm.weight"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
            eps,
        }),
        ffn: ffn(&format!("{prefix}layers.0."))?,
        eps,
    };
    Ok(MtpPredictor {
        fc: proj(format!("{prefix}fc.weight"))?,
        pre_fc_norm_embedding: norm(format!("{prefix}pre_fc_norm_embedding.weight"))?,
        pre_fc_norm_hidden: norm(format!("{prefix}pre_fc_norm_hidden.weight"))?,
        layers: vec![layer],
        norm: norm(format!("{prefix}norm.weight"))?,
    })
}

impl Qwen35Config {
    /// This config projected onto the backend-neutral companion-head contract (sc-24444, E8): the
    /// one geometry check both backends refuse a mismatched head by.
    pub fn companion_mtp_geometry(&self) -> core_llm::CompanionMtpGeometry {
        core_llm::CompanionMtpGeometry {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            rotary_dim: self.rotary_dim(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            mrope_section: self.mrope_section_resolved(),
            mtp_num_hidden_layers: self.mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings: self.mtp_use_dedicated_embeddings,
            moe: self.moe.is_some(),
        }
    }
}

/// The `[out, in]` a stored projection computes, whether dense or stored-quantized.
fn logical_matrix_shape(
    w: &Weights,
    key: &str,
    stored_quant: Option<QuantSpec>,
) -> Option<[i32; 2]> {
    let weight = w.get(key)?;
    let base = key.strip_suffix(".weight").unwrap_or(key);
    match (w.get(&format!("{base}.scales")), stored_quant) {
        (Some(scales), Some(spec)) if scales.ndim() == 2 && weight.ndim() == 2 => Some([
            weight.shape()[0],
            scales.shape()[1].checked_mul(spec.group_size)?,
        ]),
        _ if weight.ndim() == 2 => Some([weight.shape()[0], weight.shape()[1]]),
        _ => None,
    }
}

impl Qwen35Model {
    /// Attach a standalone Qwen3.8 MTP proposal head (a directory holding its `config.json` and
    /// `*.safetensors`, `model_type` [`core_llm::COMPANION_MTP_MODEL_TYPE`]) to a target that has none of its
    /// own — the Prism/Bonsai path, whose packed artifact ships no `mtp.*` tensors.
    ///
    /// The head owns only its predictor layer; it reads token embeddings and projects logits
    /// through the target's own embedding and `lm_head` **modules** (for Prism, the packed
    /// [`PrismEmbedding`] and packed `lm_head`), never a raw `.weight`. Its geometry is checked
    /// against the target's config and every projection's `[out, in]` against the target's
    /// widths before anything is built; a mismatch is an [`Error::Config`] naming each
    /// disagreement, and the target is left exactly as it was. Tensor names may be bare
    /// (`fc.weight`, the published layout) or `mtp.`-prefixed. Stored MLX affine quantization is
    /// read from the head's own `quantization` block; its RMSNorm vectors follow the zero-centred
    /// Qwen3.8 checkpoint convention (`1 + w`) of the official lineage the published heads are
    /// derived from. Every tensor in the head must be consumed.
    pub fn attach_companion_mtp(&mut self, dir: &std::path::Path) -> Result<()> {
        if self.mtp.is_some() {
            return Err(Error::Config(core_llm::COMPANION_MTP_ALREADY_NATIVE.into()));
        }
        if self.cfg.moe.is_some() {
            return Err(Error::Config(core_llm::COMPANION_MTP_MOE_REFUSAL.into()));
        }
        let value =
            core_llm::read_companion_mtp_config(dir).map_err(|e| Error::Config(e.to_string()))?;
        let head_cfg = Qwen35Config::from_json(&value)?;
        let head_geometry = head_cfg.companion_mtp_geometry();
        head_geometry
            .check_against(&self.cfg.companion_mtp_geometry())
            .map_err(Error::Config)?;
        let w = Weights::from_dir(dir)?;
        let prefix = core_llm::companion_mtp_prefix(|key| w.contains(key))
            .map_err(|e| Error::Config(e.to_string()))?;
        let stored = head_cfg.quantization;
        // The shared stored-tensor check (E8: Candle refuses exactly the same heads).
        let as_usize = |shape: &[i32]| -> Option<Vec<usize>> {
            shape.iter().map(|&d| usize::try_from(d).ok()).collect()
        };
        head_geometry
            .check_tensors(
                prefix,
                |key| {
                    let [rows, cols] = logical_matrix_shape(&w, key, stored)?;
                    Some([usize::try_from(rows).ok()?, usize::try_from(cols).ok()?])
                },
                |key| w.get(key).and_then(|a| as_usize(a.shape())),
            )
            .map_err(Error::Config)?;
        let saw_stored = Cell::new(false);
        let proj = |key: String| load_projection(&w, &key, stored, None, &saw_stored);
        let predictor = build_mtp_predictor(
            &self.cfg,
            prefix,
            &proj,
            &|key| checkpoint_norm_weight(w.require(&key)?.as_dtype(compute())?, false),
            // A dense target (MoE targets are refused above) → the dense SwiGLU arm.
            &|lp| build_ffn(&w, lp, &self.cfg, None, &proj),
        )?;
        core_llm::check_companion_unused(w.unused_keys().into_iter().map(Into::into).collect())
            .map_err(Error::Config)?;
        w.verify_accessed_gpu_view()?;
        // Its `1 + w` norms and any cast exist before the first draft step needs them (sc-24446).
        crate::primitives::weights::eval_groups(&predictor.param_groups())?;
        self.mtp = Some(predictor);
        // The target now carries one predictor layer: what its config prices the head's state by
        // (the prefix-cache snapshot's head KV, sc-24437).
        self.cfg.mtp_num_hidden_layers = 1;
        Ok(())
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
                Qwen35LayerCache::Attn(a) => Some(a.batch_size()),
                Qwen35LayerCache::Delta(_) => None,
            })
            .unwrap_or(0)
    }

    fn reset(&mut self) -> Result<()> {
        Qwen35Cache::reset(self)
    }

    // The hybrid cache is driven natively by `Qwen35Model` (which downcasts via `as_any_mut`); the
    // softmax-only trait mutators below are never invoked through the trait object on this path.
    fn update(&mut self, _layer: usize, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
        Err(Error::Msg(
            "Qwen35Cache: generic KvCache::update is not supported (hybrid cache is driven natively)"
                .into(),
        ))
    }

    fn retain_sequences(&mut self, _keep: &[i32]) -> Result<()> {
        Err(Error::Msg(
            "Qwen35Cache: retain_sequences not yet supported".into(),
        ))
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        Qwen35Cache::truncate(self, len)
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl crate::decode::Decode for Qwen35Model {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| Error::Msg("Qwen35Model::step: cache is not a Qwen35Cache".into()))?;
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for Qwen35Model {
    fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Qwen35Model::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        Qwen35Model::splice_vision_features(
            self,
            embeds,
            input_ids,
            vision_features,
            placeholder_tokens,
        )
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        Qwen35Model::mrope_positions_mm(
            self,
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
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        // The hybrid decoder drives its own concrete cache; recover it (the same downcast
        // `Qwen35Model::step` performs) before the deepstack prefill.
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| {
                Error::Msg("Qwen35Model::prefill_with_deepstack: cache is not a Qwen35Cache".into())
            })?;
        self.decode_logits_from_embeds_with_deepstack(
            embeds,
            positions,
            cache,
            visual_pos_mask,
            deepstack,
        )
    }
}

/// Expand a single vision placeholder token into `count` copies — the Qwen3-VL image/video
/// placeholder token expansion (`Qwen3VLProcessor`).
///
/// The chat template renders each visual as `<|vision_start|> <placeholder> <|vision_end|>` with a
/// **single** `placeholder_token` (image `151655` or video `151656`). The processor then replaces
/// that one placeholder with `grid_t · grid_h · grid_w / merge²` copies (`counts[k]` here — the
/// merged-patch count the vision tower emits for visual `k`), leaving the `vision_start` /
/// `vision_end` frame and all surrounding text untouched. The resulting ids line up one-to-one with
/// the spliced patch-feature rows ([`Qwen35Model::splice_image_features`]) and the M-RoPE layout
/// ([`Qwen35Model::mrope_positions_mm`]).
///
/// `counts` must have one entry per placeholder occurrence of `placeholder_token`, in sequence
/// order. Returns the expanded id stream.
pub fn expand_vision_placeholders(
    ids: &[i32],
    placeholder_token: i32,
    counts: &[usize],
) -> Result<Vec<i32>> {
    let n_placeholders = ids.iter().filter(|&&x| x == placeholder_token).count();
    if n_placeholders != counts.len() {
        return Err(Error::Msg(format!(
            "qwen3_5 token-expansion: {n_placeholders} placeholders for token {placeholder_token} \
             but {} counts supplied",
            counts.len()
        )));
    }
    let total: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(ids.len() - n_placeholders + total);
    let mut ci = 0usize;
    for &id in ids {
        if id == placeholder_token {
            out.extend(std::iter::repeat_n(placeholder_token, counts[ci]));
            ci += 1;
        } else {
            out.push(id);
        }
    }
    Ok(out)
}

/// The merged-patch token count for a vision grid `[t, h, w]` (patch units): `t · h · w / merge²`
/// — the number of placeholder tokens the processor emits and the number of feature rows the vision
/// tower produces for that visual.
pub fn vision_merged_token_count(grid_thw: [i32; 3], spatial_merge_size: i32) -> usize {
    let merge = spatial_merge_size.max(1);
    let [t, h, w] = grid_thw;
    (t.max(0) * (h.max(0) / merge) * (w.max(0) / merge)) as usize
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::primitives::kv_cache::testing::{host, on_cpu, tok, ConcatReference};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn attn_kv_matches_concat_reference_across_a_block_boundary() {
        on_cpu(|| {
            // The full-attention slot is a one-layer block cache: 3-token prefill + enough single
            // tokens to cross the block boundary twice, every returned K/V equal to a naive concat,
            // and `offset` reporting live positions rather than the padded buffer length.
            let block = 4;
            let mut slot = AttnKv::with_block_tokens(block);
            let mut reference = ConcatReference::new();
            assert_eq!(slot.offset(), 0);
            assert_eq!(slot.batch_size(), 0);

            let prefill_k = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[1, 1, 3, 2]);
            let prefill_v = Array::from_slice(&[9.0f32, 9.5, 8.0, 8.5, 7.0, 7.5], &[1, 1, 3, 2]);
            let (sk, sv) = slot.update(&prefill_k, &prefill_v).unwrap();
            let (rk, rv) = reference.update(&prefill_k, &prefill_v);
            assert_eq!(host(&sk), host(&rk));
            assert_eq!(host(&sv), host(&rv));
            assert_eq!(slot.offset(), 3);
            assert_eq!(slot.batch_size(), 1);

            for i in 0..7 {
                let k = tok(10.0 + i as f32);
                let v = tok(20.0 + i as f32);
                let (sk, sv) = slot.update(&k, &v).unwrap();
                let (rk, rv) = reference.update(&k, &v);
                assert_eq!(sk.shape(), rk.shape(), "update {i}: shape");
                assert_eq!(host(&sk), host(&rk), "update {i}: keys");
                assert_eq!(host(&sv), host(&rv), "update {i}: values");
                assert_eq!(slot.offset(), 4 + i, "update {i}: offset is live positions");
            }
            // 10 live positions in a 12-position buffer: offset must not read the padding.
            assert_eq!(slot.offset(), 10);
            assert_ne!(slot.offset(), 12);

            slot.reset().unwrap();
            assert_eq!(slot.offset(), 0);
            assert_eq!(slot.batch_size(), 0);
        })
    }

    #[test]
    fn attn_kv_clone_snapshot_survives_in_place_updates_and_rolls_back() {
        on_cpu(|| {
            // A snapshot rollback (the MTP predictor cache, the engine's generic `SnapshotRollback`)
            // restores a `clone()` taken before the trial. While the trial writes in place, the
            // snapshot must stay exactly what it was, and continuing from it must match a reference
            // that never saw the trial — across a block boundary, so the trial both overwrites
            // padding and grows.
            let block = 4;
            let mut slot = AttnKv::with_block_tokens(block);
            let mut reference = ConcatReference::new();
            let prompt = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[1, 1, 3, 2]);
            slot.update(&prompt, &prompt).unwrap();
            reference.update(&prompt, &prompt);

            let snapshot = slot.clone();
            let snapshot_k_before = host(reference.k.as_ref().unwrap());

            // Trial: 3 draft tokens (positions 3..6) — fills the block and grows into a second one.
            for i in 0..3 {
                let d = tok(100.0 + i as f32);
                slot.update(&d, &d).unwrap();
            }
            assert_eq!(slot.offset(), 6);
            assert_eq!(
                snapshot.offset(),
                3,
                "snapshot offset untouched by the trial"
            );
            let (snap_k, _) = snapshot.kv.peek(0).unwrap().unwrap();
            assert_eq!(
                host(&snap_k),
                snapshot_k_before,
                "snapshot contents untouched by the trial's in-place writes"
            );

            // Reject everything: restore the snapshot and replay the accepted path.
            let mut slot = snapshot;
            for i in 0..5 {
                let t = tok(200.0 + i as f32);
                let (sk, sv) = slot.update(&t, &t).unwrap();
                let (rk, rv) = reference.update(&t, &t);
                assert_eq!(host(&sk), host(&rk), "replay {i}: keys");
                assert_eq!(host(&sv), host(&rv), "replay {i}: values");
            }
            assert_eq!(slot.offset(), 8);
        })
    }

    #[test]
    fn published_prism_norm_multiplier_matches_independent_rms_oracle() {
        let source = vec![1.0583496f32, 0.9418945, 1.3125, 1.957_031_3];
        let x = vec![0.25f32, -0.5, 1.25, -2.0];
        let weight = checkpoint_norm_weight(Array::from_slice(&source, &[4]), true).unwrap();
        let got = rms_norm(&Array::from_slice(&x, &[1, 4]), &weight, 1e-6).unwrap();
        mlx_rs::transforms::eval([&got]).unwrap();
        let inv = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6)
            .sqrt()
            .recip();
        for (i, value) in got.as_slice::<f32>().iter().enumerate() {
            let expected = x[i] * inv * source[i];
            assert!(
                (value - expected).abs() < 1e-5,
                "lane {i}: {value} != {expected}"
            );
        }
        let dense =
            checkpoint_norm_weight(Array::from_slice(&[0.0583496f32], &[1]), false).unwrap();
        mlx_rs::transforms::eval([&dense]).unwrap();
        assert!((dense.item::<f32>() - source[0]).abs() < 1e-6);
    }

    pub(crate) fn cfg_json() -> serde_json::Value {
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
        let value: serde_json::Value = serde_json::from_str(include_str!(concat!(
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
        assert!(cfg.quantization.is_none());
    }

    /// A deterministic small tensor `[shape]` (finite, non-degenerate).
    fn t(map: &mut HashMap<String, Array>, key: &str, shape: &[i32]) {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        map.insert(key.to_string(), Array::from_slice(&data, shape));
    }

    fn synthetic_weights_with_prefix(cfg: &Qwen35Config, pfx: &str) -> Weights {
        Weights::from_map(synthetic_tensors_with_prefix(cfg, pfx))
    }

    fn synthetic_tensors_with_prefix(cfg: &Qwen35Config, pfx: &str) -> HashMap<String, Array> {
        let h = cfg.hidden_size;
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let mut m = HashMap::new();
        // The decoder can live under either VLM-wrapped or flat text-only roots; `lm_head` and
        // `mtp.*` remain at the checkpoint root in both layouts.
        t(
            &mut m,
            &format!("{pfx}.embed_tokens.weight"),
            &[cfg.vocab_size, h],
        );
        t(&mut m, &format!("{pfx}.norm.weight"), &[h]);
        t(&mut m, "lm_head.weight", &[cfg.vocab_size, h]);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| format!("{pfx}.layers.{i}.{s}");
            t(&mut m, &lp("input_layernorm.weight"), &[h]);
            t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
            match &cfg.moe {
                None => {
                    t(
                        &mut m,
                        &lp("mlp.gate_proj.weight"),
                        &[cfg.intermediate_size, h],
                    );
                    t(
                        &mut m,
                        &lp("mlp.up_proj.weight"),
                        &[cfg.intermediate_size, h],
                    );
                    t(
                        &mut m,
                        &lp("mlp.down_proj.weight"),
                        &[h, cfg.intermediate_size],
                    );
                }
                Some(moe) => {
                    let mi = moe.moe_intermediate_size;
                    let si = moe.shared_expert_intermediate_size;
                    t(
                        &mut m,
                        &lp("mlp.experts.gate_up_proj"),
                        &[moe.num_experts, 2 * mi, h],
                    );
                    t(
                        &mut m,
                        &lp("mlp.experts.down_proj"),
                        &[moe.num_experts, h, mi],
                    );
                    t(&mut m, &lp("mlp.gate.weight"), &[moe.num_experts, h]);
                    t(&mut m, &lp("mlp.shared_expert.gate_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.up_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.down_proj.weight"), &[h, si]);
                    t(&mut m, &lp("mlp.shared_expert_gate.weight"), &[1, h]);
                }
            }
            if cfg.is_linear(i) {
                // 4-way split projections (real qwen3_5 layout).
                t(
                    &mut m,
                    &lp("linear_attn.in_proj_qkv.weight"),
                    &[conv_dim, h],
                );
                t(&mut m, &lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
                t(
                    &mut m,
                    &lp("linear_attn.in_proj_a.weight"),
                    &[cfg.linear_num_value_heads, h],
                );
                t(
                    &mut m,
                    &lp("linear_attn.in_proj_b.weight"),
                    &[cfg.linear_num_value_heads, h],
                );
                t(
                    &mut m,
                    &lp("linear_attn.conv1d.weight"),
                    &[conv_dim, 1, cfg.linear_conv_kernel_dim],
                );
                t(
                    &mut m,
                    &lp("linear_attn.A_log"),
                    &[cfg.linear_num_value_heads],
                );
                t(
                    &mut m,
                    &lp("linear_attn.dt_bias"),
                    &[cfg.linear_num_value_heads],
                );
                t(
                    &mut m,
                    &lp("linear_attn.norm.weight"),
                    &[cfg.linear_value_head_dim],
                );
                t(&mut m, &lp("linear_attn.out_proj.weight"), &[h, value_dim]);
            } else {
                t(
                    &mut m,
                    &lp("self_attn.q_proj.weight"),
                    &[cfg.num_heads * cfg.head_dim * 2, h],
                );
                t(
                    &mut m,
                    &lp("self_attn.k_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, h],
                );
                t(
                    &mut m,
                    &lp("self_attn.v_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, h],
                );
                t(
                    &mut m,
                    &lp("self_attn.o_proj.weight"),
                    &[h, cfg.num_heads * cfg.head_dim],
                );
                t(&mut m, &lp("self_attn.q_norm.weight"), &[cfg.head_dim]);
                t(&mut m, &lp("self_attn.k_norm.weight"), &[cfg.head_dim]);
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
            t(
                &mut m,
                &lp("self_attn.q_proj.weight"),
                &[cfg.num_heads * cfg.head_dim * 2, h],
            );
            t(
                &mut m,
                &lp("self_attn.k_proj.weight"),
                &[cfg.num_kv_heads * cfg.head_dim, h],
            );
            t(
                &mut m,
                &lp("self_attn.v_proj.weight"),
                &[cfg.num_kv_heads * cfg.head_dim, h],
            );
            t(
                &mut m,
                &lp("self_attn.o_proj.weight"),
                &[h, cfg.num_heads * cfg.head_dim],
            );
            t(&mut m, &lp("self_attn.q_norm.weight"), &[cfg.head_dim]);
            t(&mut m, &lp("self_attn.k_norm.weight"), &[cfg.head_dim]);
            t(
                &mut m,
                &lp("mlp.gate_proj.weight"),
                &[cfg.intermediate_size, h],
            );
            t(
                &mut m,
                &lp("mlp.up_proj.weight"),
                &[cfg.intermediate_size, h],
            );
            t(
                &mut m,
                &lp("mlp.down_proj.weight"),
                &[h, cfg.intermediate_size],
            );
        }
        m
    }

    pub(crate) fn synthetic_weights(cfg: &Qwen35Config) -> Weights {
        synthetic_weights_with_prefix(cfg, "model.language_model")
    }

    pub(crate) fn cfg_json_mtp() -> serde_json::Value {
        let mut v = cfg_json();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("mtp_num_hidden_layers".into(), json!(1));
        tc.insert("mtp_use_dedicated_embeddings".into(), json!(false));
        v
    }

    #[test]
    fn config_parses_and_schedules_3_linear_1_full() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.rotary_dim(), 4); // head_dim 8 * 0.5
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
        // A 5-token prefill.
        let ids = Array::from_slice(&[1i32, 7, 3, 42, 9], &[1, 5]);
        let logits = model.forward(&ids, &mut cache, 0).unwrap();
        assert_eq!(logits.shape(), &[1, 5, cfg.vocab_size]);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite(), "non-finite logit: {x}");
        }
        // The full-attention layer (layer 3) advanced the KV cache to 5 positions.
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn flat_text_and_wrapped_qwen35_match_target_and_mtp() {
        let wrapped_json = cfg_json_mtp();
        let flat_json = wrapped_json["text_config"].clone();
        let wrapped_cfg = Qwen35Config::from_json(&wrapped_json).unwrap();
        let flat_cfg = Qwen35Config::from_json(&flat_json).unwrap();
        let wrapped = Qwen35Model::from_weights(
            &synthetic_weights_with_prefix(&wrapped_cfg, "model.language_model"),
            "model.language_model",
            wrapped_cfg,
        )
        .unwrap();
        let flat = Qwen35Model::from_weights(
            &synthetic_weights_with_prefix(&flat_cfg, "model"),
            "model",
            flat_cfg,
        )
        .unwrap();
        let ids = Array::from_slice(&[1i32, 7, 3], &[1, 3]);
        let wrapped_logits = wrapped.forward(&ids, &mut wrapped.new_cache(), 0).unwrap();
        let flat_logits = flat.forward(&ids, &mut flat.new_cache(), 0).unwrap();
        let values = |a: &Array| {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        assert_eq!(values(&wrapped_logits), values(&flat_logits));

        let shifted_ids = Array::from_slice(&[2i32, 3], &[1, 2]);
        let aligned_hidden = Array::from_slice(&vec![0f32; 2 * 32], &[1, 2, 32]);
        let (_, wrapped_draft) = wrapped
            .mtp_step(
                &shifted_ids,
                &aligned_hidden,
                &mut wrapped.new_mtp_cache().unwrap(),
                0,
            )
            .unwrap();
        let (_, flat_draft) = flat
            .mtp_step(
                &shifted_ids,
                &aligned_hidden,
                &mut flat.new_mtp_cache().unwrap(),
                0,
            )
            .unwrap();
        assert_eq!(values(&wrapped_draft), values(&flat_draft));
    }

    #[test]
    fn mtp_loads_all_frozen_components_and_executes() {
        let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        assert!(model.has_mtp());

        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let mut target_cache = model.new_cache();
        let (target_hidden, _) = model.hidden_and_logits(&ids, &mut target_cache, 0).unwrap();
        let shifted_ids = Array::from_slice(&[2i32, 3], &[1, 2]);
        let aligned_hidden = target_hidden
            .take_axis(Array::from_slice(&[0i32, 1], &[2]), 1)
            .unwrap();
        let mut mtp_cache = model.new_mtp_cache().unwrap();
        let (mtp_hidden, logits) = model
            .mtp_step(&shifted_ids, &aligned_hidden, &mut mtp_cache, 0)
            .unwrap();
        assert_eq!(mtp_hidden.shape(), &[1, 2, cfg.hidden_size]);
        assert_eq!(logits.shape(), &[1, cfg.vocab_size]);
        assert_eq!(mtp_cache.layers[0].offset(), 2);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn mtp_prefill_projects_only_last_row_and_warm_cache_matches_full() {
        let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let model = Qwen35Model::from_weights(
            &synthetic_weights(&cfg),
            "model.language_model",
            cfg.clone(),
        )
        .unwrap();
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let as_f32 = |array: &Array| {
            array
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        let mut full_cache = model.new_cache();
        let (full_hidden, full_logits) = model.hidden_and_logits(&ids, &mut full_cache, 0).unwrap();
        let mut prefill_cache = model.new_cache();
        let (hidden, last_logits) = model
            .prefill_hidden_and_last_logits(&ids, &mut prefill_cache, 0)
            .unwrap();
        assert_eq!(last_logits.shape(), &[1, cfg.vocab_size]);
        assert_eq!(as_f32(&hidden), as_f32(&full_hidden));
        let expected_last = full_logits
            .take_axis(Array::from_slice(&[2i32], &[1]), 1)
            .unwrap();
        assert_eq!(as_f32(&last_logits), as_f32(&expected_last));
        assert_eq!(prefill_cache.offset(), full_cache.offset());

        let embeds = model.embed_input_ids(&ids).unwrap();
        let positions = [0, 1, 2];
        let mut full_visual_cache = model.new_cache();
        let (visual_hidden, visual_logits) = model
            .hidden_and_logits_from_embeds_with_deepstack(
                &embeds,
                [&positions, &positions, &positions],
                &mut full_visual_cache,
                &[false; 3],
                &[],
            )
            .unwrap();
        let mut visual_cache = model.new_cache();
        let (visual_prefill_hidden, visual_last) = model
            .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
                &embeds,
                [&positions, &positions, &positions],
                &mut visual_cache,
                &[false; 3],
                &[],
            )
            .unwrap();
        assert_eq!(as_f32(&visual_prefill_hidden), as_f32(&visual_hidden));
        let visual_expected = visual_logits
            .take_axis(Array::from_slice(&[2i32], &[1]), 1)
            .unwrap();
        assert_eq!(as_f32(&visual_last), as_f32(&visual_expected));
        assert_eq!(visual_cache.offset(), full_visual_cache.offset());

        let shifted = embeds
            .take_axis(Array::from_slice(&[1i32, 2], &[2]), 1)
            .unwrap();
        let aligned = hidden
            .take_axis(Array::from_slice(&[0i32, 1], &[2]), 1)
            .unwrap();
        let seed_positions = [1, 2];
        let mut full_mtp_cache = model.new_mtp_cache().unwrap();
        let (full_mtp_hidden, last_mtp_logits) = model
            .mtp_step_from_embeds(
                &shifted,
                &aligned,
                &mut full_mtp_cache,
                [&seed_positions, &seed_positions, &seed_positions],
            )
            .unwrap();
        let all_mtp_logits = model.lm_head.forward(&full_mtp_hidden).unwrap();
        let expected_mtp_last = all_mtp_logits
            .take_axis(Array::from_slice(&[1i32], &[1]), 1)
            .unwrap();
        assert_eq!(last_mtp_logits.shape(), &[1, cfg.vocab_size]);
        assert_eq!(as_f32(&last_mtp_logits), as_f32(&expected_mtp_last));
        let mut warm_mtp_cache = model.new_mtp_cache().unwrap();
        let warm_hidden = model
            .mtp_warm_from_embeds(
                &shifted,
                &aligned,
                &mut warm_mtp_cache,
                [&seed_positions, &seed_positions, &seed_positions],
            )
            .unwrap();
        assert_eq!(as_f32(&warm_hidden), as_f32(&full_mtp_hidden));
        assert_eq!(warm_mtp_cache.layers[0].offset(), 2);
        assert_eq!(full_mtp_cache.layers[0].offset(), 2);
        let previous = hidden
            .take_axis(Array::from_slice(&[2i32], &[1]), 1)
            .unwrap();
        let next = Array::from_slice(&[1i32], &[1, 1]);
        let (_, full_next) = model
            .mtp_step(&next, &previous, &mut full_mtp_cache, 3)
            .unwrap();
        let (_, warm_next) = model
            .mtp_step(&next, &previous, &mut warm_mtp_cache, 3)
            .unwrap();
        assert_eq!(as_f32(&warm_next), as_f32(&full_next));
    }

    /// E2 (sc-24432 feature-end review): a config declaring a native head the snapshot stores
    /// no tensor of — or a variant this runtime does not run (two predictor layers, dedicated MTP
    /// embeddings) — builds the target plain with a named `mtp:` fallback and prices no head
    /// state; a PARTIAL `mtp.*` set still fails (integrity); `mtp.*` tensors under a config that
    /// disables MTP stay a refused contradiction.
    #[test]
    fn an_mtp_head_the_snapshot_cannot_run_is_a_named_fallback_but_a_partial_one_fails() {
        let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let base_cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let weights = synthetic_weights(&base_cfg);
        let model = Qwen35Model::from_weights(&weights, "model.language_model", cfg.clone())
            .expect("a configured head the snapshot does not carry loads the target plain");
        assert!(!model.has_mtp());
        let why = model.mtp_fallback().expect("the fallback is named");
        assert!(
            why.starts_with("mtp: ") && why.contains("stores no `mtp.*` tensor"),
            "{why}"
        );
        assert_eq!(
            model.config().mtp_num_hidden_layers,
            0,
            "no head state priced"
        );

        let full = synthetic_weights(&cfg);
        let tensors = |keep: &dyn Fn(&str) -> bool| {
            Weights::from_map(
                full.keys()
                    .filter(|k| keep(k))
                    .map(|k| (k.to_string(), full.get(k).unwrap().clone()))
                    .collect(),
            )
        };
        for (edit, named) in [
            (
                (|c: &mut Qwen35Config| c.mtp_num_hidden_layers = 2) as fn(&mut Qwen35Config),
                "declares 2 predictor layers",
            ),
            (
                |c: &mut Qwen35Config| c.mtp_use_dedicated_embeddings = true,
                "dedicated MTP embeddings",
            ),
        ] {
            let mut variant = cfg.clone();
            edit(&mut variant);
            let model =
                Qwen35Model::from_weights(&tensors(&|_| true), "model.language_model", variant)
                    .unwrap_or_else(|e| panic!("{named}: the target still loads (E2): {e}"));
            assert!(!model.has_mtp(), "{named}");
            let why = model.mtp_fallback().unwrap_or_default();
            assert!(why.starts_with("mtp: ") && why.contains(named), "{why}");
        }
        let partial = tensors(&|k| k != "mtp.fc.weight");
        let Err(err) = Qwen35Model::from_weights(&partial, "model.language_model", cfg.clone())
        else {
            panic!("a partial `mtp.*` set fails the load");
        };
        assert!(err.to_string().contains("mtp.fc.weight"), "{err}");
        let complete = Qwen35Model::from_weights(&full, "model.language_model", cfg).unwrap();
        assert!(complete.has_mtp() && complete.mtp_fallback().is_none());

        let mtp_cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let base_cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let err = Qwen35Model::from_weights(
            &synthetic_weights(&mtp_cfg),
            "model.language_model",
            base_cfg,
        )
        .unwrap_err();
        assert!(err.to_string().contains("config disables MTP"), "{err}");
    }

    fn greedy_config(max_new_tokens: usize) -> crate::decode::GenerationConfig {
        crate::decode::GenerationConfig {
            max_new_tokens,
            sampling: crate::primitives::sampler::SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_context: 0,
            },
            seed: Some(7),
            stop_tokens: Vec::new(),
        }
    }

    fn bits(a: &Array) -> Vec<u32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    const PRISM_PROMPT: [i32; 7] = [1, 5, 9, 2, 33, 17, 4];

    /// Story sc-24444 AC1: on a Prism fixture (packed two-bit projections with random codes and
    /// signs, one Gated DeltaNet and one attention layer) the fused rotation leaves the prefill
    /// logits bit-identical and the greedy tokens unchanged.
    #[test]
    fn prism_greedy_tokens_and_logits_are_unchanged_by_the_fused_rotation() {
        use crate::decode::{generate, CancelFlag};
        use crate::primitives::prism::tests::with_unfused_rotation;
        use crate::primitives::prism::{recording_rotation_routes, RotationRoute};

        let fixture = crate::synthetic::prism_qwen35(11);
        let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
        let model = Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
        assert!(model.is_prism());
        let run = || {
            let mut cache = model.new_cache();
            let (hidden, logits) = model
                .hidden_and_logits(&crate::primitives::input_ids(&PRISM_PROMPT), &mut cache, 0)
                .unwrap();
            let out = generate(
                &model,
                &PRISM_PROMPT,
                &greedy_config(24),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
            (bits(&hidden), bits(&logits), out.tokens)
        };
        // Count the routes, not just the numbers: the two paths are bit-identical by
        // construction, so only the route shows the fused kernel actually ran.
        let fused_count = |routes: &[RotationRoute]| {
            routes
                .iter()
                .filter(|r| **r == RotationRoute::Fused)
                .count()
        };
        let (fused, fused_routes) = mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
            recording_rotation_routes(run)
        });
        let (unfused, unfused_routes) =
            mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
                recording_rotation_routes(|| with_unfused_rotation(run))
            });
        assert!(
            fused_count(&fused_routes) > 0,
            "the fused rotation never ran: {fused_routes:?}"
        );
        assert!(
            !fused_routes.contains(&RotationRoute::Unfused),
            "every rotation at the fixture's widths is fused: {fused_routes:?}"
        );
        assert_eq!(
            fused_count(&unfused_routes),
            0,
            "the oracle run used the kernel"
        );
        assert!(!unfused_routes.is_empty());
        assert!(fused.0 == unfused.0, "prefill hidden states differ");
        assert!(fused.1 == unfused.1, "prefill logits differ");
        assert_eq!(fused.2, unfused.2, "greedy tokens differ");
        let distinct: std::collections::BTreeSet<_> = fused.2.iter().collect();
        assert!(
            distinct.len() > 2,
            "fixture must not be degenerate: {:?}",
            fused.2
        );
    }

    /// Story sc-24444 AC2 (model level): a companion MTP head attached to a Prism target borrows
    /// the packed embedding and `lm_head`, drafts, and every depth emits exactly the target's
    /// greedy tokens.
    #[test]
    fn prism_target_with_a_companion_head_drafts_and_keeps_greedy_tokens() {
        use crate::decode::{generate, generate_qwen35_mtp, CancelFlag};

        let fixture = crate::synthetic::prism_qwen35(11);
        let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
        let mut model =
            Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
        assert!(!model.has_mtp());
        assert!(model.new_mtp_cache().is_none());
        let head = crate::test_fixture::Fixture::new("mlx-companion-head-", None);
        crate::synthetic::write_companion_head(&head, &fixture.config["text_config"], 3, None);
        model.attach_companion_mtp(&head).unwrap();
        assert!(model.has_mtp());
        assert!(
            model.is_prism(),
            "the target stays the packed Prism decoder"
        );

        let greedy = greedy_config(24);
        let plain = generate(
            &model,
            &PRISM_PROMPT,
            &greedy,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        for depth in [1, 3, 7] {
            let (speculative, stats) = generate_qwen35_mtp(
                &model,
                &PRISM_PROMPT,
                &greedy,
                depth,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
            )
            .unwrap();
            assert_eq!(speculative.tokens, plain.tokens, "depth {depth}");
            assert!(stats.proposed > 0, "depth {depth}: the head drafted");
        }
    }

    /// Story sc-24444 AC2 / E2: a head whose geometry disagrees with the target is refused with a
    /// reason naming each disagreement, and the target is left exactly as it was — still loaded,
    /// no predictor, same greedy tokens.
    #[test]
    fn a_mismatched_companion_head_is_refused_by_name_and_leaves_the_target_intact() {
        use crate::decode::{generate, CancelFlag};

        let fixture = crate::synthetic::prism_qwen35(11);
        let text = &fixture.config["text_config"];
        let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
        let mut model =
            Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
        let greedy = greedy_config(12);
        let before = generate(
            &model,
            &PRISM_PROMPT,
            &greedy,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();

        // The head's own config disagrees (a head for another model size).
        let mut other = text.clone();
        other["intermediate_size"] = json!(128);
        other["num_key_value_heads"] = json!(2);
        let config_mismatch = crate::test_fixture::Fixture::new("mlx-head-geometry-", None);
        crate::synthetic::write_companion_head(&config_mismatch, &other, 3, None);
        let err = model
            .attach_companion_mtp(&config_mismatch)
            .unwrap_err()
            .to_string();
        assert!(err.contains("geometry does not match"), "{err}");
        assert!(err.contains("intermediate_size 128 != target 256"), "{err}");
        assert!(err.contains("num_key_value_heads 2 != target 1"), "{err}");

        // Each field the predictor computes with but no tensor shape reveals is its own named
        // refusal: `vocab_size` is the only guard against a head for another tokenizer.
        for (field, value, named) in [
            ("vocab_size", json!(65), "vocab_size 65 != target 64"),
            (
                "partial_rotary_factor",
                json!(0.25),
                "rotary_dim 16 != target 32",
            ),
            (
                "rope_theta",
                json!(1000000.0),
                "rope_theta 1000000 != target 10000000",
            ),
            (
                "rms_norm_eps",
                json!(1e-5),
                "rms_norm_eps 0.00001 != target 0.000001",
            ),
        ] {
            let mut other = text.clone();
            other[field] = value;
            let head = crate::test_fixture::Fixture::new("mlx-head-field-", None);
            crate::synthetic::write_companion_head(&head, &other, 3, None);
            let err = model.attach_companion_mtp(&head).unwrap_err().to_string();
            assert!(err.contains("geometry does not match"), "{field}: {err}");
            assert!(err.contains(named), "{field}: {err}");
        }

        // The config matches but a stored tensor does not.
        let tensor_mismatch = crate::test_fixture::Fixture::new("mlx-head-tensors-", None);
        crate::synthetic::write_companion_head(&tensor_mismatch, text, 3, Some(128));
        let err = model
            .attach_companion_mtp(&tensor_mismatch)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`layers.0.self_attn.q_proj` is [128, 128], the target needs [256, 128]"),
            "{err}"
        );

        // Not a companion head at all.
        let wrong_type = crate::test_fixture::Fixture::new("mlx-head-type-", None);
        crate::synthetic::write_companion_head(&wrong_type, text, 3, None);
        let path = wrong_type.join("config.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["model_type"] = json!("qwen3_5");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let err = model
            .attach_companion_mtp(&wrong_type)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be `qwen3_5_mtp`"), "{err}");

        assert!(!model.has_mtp());
        let after = generate(
            &model,
            &PRISM_PROMPT,
            &greedy,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(after.tokens, before.tokens);

        // A target with its own native predictor never takes a second one.
        let mut native = crate::decode::engine::tests::qwen35(true);
        let native_head = crate::test_fixture::Fixture::new("mlx-head-native-", None);
        crate::synthetic::write_companion_head(&native_head, text, 3, None);
        let err = native
            .attach_companion_mtp(&native_head)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("already carries its own MTP predictor"),
            "{err}"
        );
    }

    /// Story sc-24444: a companion head's RMSNorm vectors follow the zero-centred Qwen3.8
    /// checkpoint convention (`1 + w`) whatever the target's own convention — a Prism target's
    /// norms are direct multipliers, the head's are not (the published Qwen3.8 heads'
    /// `pre_fc_norm_embedding` values are all negative, mean ≈ −0.46).
    #[test]
    fn a_companion_heads_norms_are_applied_as_one_plus_w() {
        let fixture = crate::synthetic::prism_qwen35(11);
        let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
        let mut model =
            Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
        let head = crate::test_fixture::Fixture::new("mlx-head-norm-", None);
        crate::synthetic::write_companion_head(&head, &fixture.config["text_config"], 3, None);
        let path = head.join("model.safetensors");
        let mut tensors = Array::load_safetensors(&path).unwrap();
        // The load is lazy: materialize every tensor before overwriting the file it reads.
        mlx_rs::transforms::eval(tensors.values()).unwrap();
        tensors.insert(
            "pre_fc_norm_embedding.weight".into(),
            Array::from_slice(&[-0.5f32; 128], &[128])
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        Array::save_safetensors(tensors.iter().map(|(k, v)| (k.as_str(), v)), None, &path).unwrap();
        model.attach_companion_mtp(&head).unwrap();
        let applied = model
            .mtp
            .as_ref()
            .unwrap()
            .pre_fc_norm_embedding
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(applied.as_slice::<f32>(), [0.5f32; 128]);
    }

    #[test]
    fn mtp_generation_covers_verification_rollback_stop_and_cancel() {
        use crate::decode::{
            generate, generate_qwen35_mtp, generate_speculative, CancelFlag, ConstraintMask,
            EngineOptions, FinishReason, GenerationConfig, MtpProposer, RewindableConstraintMask,
            SpeculativePrompt,
        };
        use crate::primitives::sampler::SamplingParams;

        struct RecordingConstraint {
            allow: Vec<bool>,
            accepted: Vec<i32>,
            rewinds: usize,
        }

        impl ConstraintMask for RecordingConstraint {
            fn allowed(&mut self) -> &[bool] {
                &self.allow
            }

            fn accept(&mut self, token: i32) {
                self.accepted.push(token);
            }
        }

        impl RewindableConstraintMask for RecordingConstraint {
            fn checkpoint(&self) -> usize {
                self.accepted.len()
            }

            fn rewind(&mut self, checkpoint: usize) {
                self.accepted.truncate(checkpoint);
                self.rewinds += 1;
            }
        }

        let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let vocab_size = cfg.vocab_size as usize;
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg)
                .unwrap();
        let greedy = GenerationConfig {
            max_new_tokens: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_context: 0,
            },
            seed: Some(7),
            stop_tokens: Vec::new(),
        };
        let target_only =
            generate(&model, &[1, 2, 3], &greedy, &CancelFlag::new(), &mut |_| {}).unwrap();
        let mut timed = generate_speculative(
            &model,
            &mut MtpProposer::new(),
            SpeculativePrompt::Tokens(&[1, 2, 3]),
            &greedy,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                prefill_clock: Some(std::time::Instant::now()),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let _timings = timed
            .take_timer()
            .expect("a timed run keeps its timer")
            .finish();
        let (speculative, stats) = (timed.output, timed.stats);
        assert_eq!(
            speculative.tokens, target_only.tokens,
            "adversarial MTP drafts must not change greedy target output"
        );
        assert!(stats.proposed > 0);
        assert!(
            stats.accepted < stats.proposed,
            "fixture must exercise rejection and clone/replay rollback: {stats:?}"
        );
        assert!(stats.forwards >= 3, "rejection must add a replay forward");

        let mut constrained = RecordingConstraint {
            allow: vec![true; vocab_size],
            accepted: Vec::new(),
            rewinds: 0,
        };
        let (constrained_greedy, constrained_stats) = generate_qwen35_mtp(
            &model,
            &[1, 2, 3],
            &greedy,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            Some(&mut constrained),
            None,
        )
        .unwrap();
        assert_eq!(constrained_greedy.tokens, target_only.tokens);
        assert!(constrained_stats.accepted < constrained_stats.proposed);
        assert_eq!(constrained.accepted, constrained_greedy.tokens);
        assert!(
            constrained.rewinds >= 2,
            "proposal and verification must both rewind provisional constraint state"
        );

        let mut with_stop = greedy.clone();
        with_stop.stop_tokens = vec![target_only.tokens[2]];
        let (stopped, _) = generate_qwen35_mtp(
            &model,
            &[1, 2, 3],
            &with_stop,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(stopped.tokens, target_only.tokens[..2]);
        assert_eq!(stopped.finish_reason, FinishReason::StopToken);

        let cancel = CancelFlag::new();
        let cancel_from_sink = cancel.clone();
        let (cancelled, _) = generate_qwen35_mtp(
            &model,
            &[1, 2, 3],
            &greedy,
            2,
            &cancel,
            &mut |event| {
                if matches!(event, crate::decode::StreamEvent::Token { step: 0, .. }) {
                    cancel_from_sink.cancel();
                }
            },
            None,
            None,
        )
        .unwrap();
        assert_eq!(cancelled.tokens.len(), 1);
        assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);

        let stochastic = GenerationConfig {
            sampling: SamplingParams {
                temperature: 0.8,
                top_p: 0.95,
                top_k: 20,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_context: 0,
            },
            ..greedy
        };
        let (sampled, sampled_stats) = generate_qwen35_mtp(
            &model,
            &[1, 2, 3],
            &stochastic,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(sampled.tokens.len(), 8);
        assert!(sampled_stats.proposed > 0);
        assert!(sampled_stats.accepted <= sampled_stats.proposed);

        let mut sampled_constraint = RecordingConstraint {
            allow: vec![true; vocab_size],
            accepted: Vec::new(),
            rewinds: 0,
        };
        let (constrained_sampled, _) = generate_qwen35_mtp(
            &model,
            &[1, 2, 3],
            &stochastic,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            Some(&mut sampled_constraint),
            None,
        )
        .unwrap();
        assert_eq!(
            constrained_sampled.tokens, sampled.tokens,
            "a permissive transactional constraint must preserve stochastic p/q sampling"
        );
        assert_eq!(sampled_constraint.accepted, constrained_sampled.tokens);
    }

    #[test]
    fn multimodal_mtp_seed_uses_fused_embeddings_and_matches_text_equivalent() {
        use std::time::Instant;

        use crate::decode::{
            generate_speculative, CancelFlag, EngineOptions, GenerationConfig, MtpProposer,
            Qwen35MtpMultimodalPrompt, SpeculativePrompt,
        };
        use crate::primitives::sampler::SamplingParams;

        let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg)
                .unwrap();
        let ids = [1, 2, 3];
        let embeds = model
            .embed_input_ids(&crate::primitives::input_ids(&ids))
            .unwrap();
        let positions = [0, 1, 2];
        let prompt = Qwen35MtpMultimodalPrompt {
            input_ids: &ids,
            embeddings: &embeds,
            positions: [&positions, &positions, &positions],
            visual_pos_mask: &[false, false, false],
            deepstack: &[],
            continuation_delta: 0,
        };
        let config = GenerationConfig {
            max_new_tokens: 6,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_context: 0,
            },
            seed: Some(17),
            stop_tokens: Vec::new(),
        };
        let clock = || EngineOptions {
            prefill_clock: Some(Instant::now()),
            ..EngineOptions::default()
        };
        let mut text = generate_speculative(
            &model,
            &mut MtpProposer::new(),
            SpeculativePrompt::Tokens(&ids),
            &config,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            clock(),
        )
        .unwrap();
        // The multimodal route: the caller prefills the fused embeddings under explicit M-RoPE
        // positions and the proposer seeds from those same embeddings.
        let mut visual_cache = model.new_cache();
        let (visual_hidden, visual_logits) = model
            .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
                prompt.embeddings,
                prompt.positions,
                &mut visual_cache,
                prompt.visual_pos_mask,
                prompt.deepstack,
            )
            .unwrap();
        let mut visual = generate_speculative(
            &model,
            &mut MtpProposer::multimodal(&prompt),
            SpeculativePrompt::Prefilled {
                cache: &mut visual_cache,
                logits: visual_logits,
                hidden: Some(visual_hidden),
                history: prompt.input_ids,
                position_delta: prompt.continuation_delta,
            },
            &config,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            clock(),
        )
        .unwrap();
        assert_eq!(visual.output.tokens, text.output.tokens);
        assert_eq!(visual.stats, text.stats);
        let _ = text.take_timer().unwrap().finish();
        let _ = visual.take_timer().unwrap().finish();

        // Alter the middle row as an encoded visual feature. Both the target prefill and shifted
        // MTP seed must consume that fused row rather than re-embedding placeholder token id 2.
        let shape = embeds.shape().to_vec();
        let hidden = shape[2] as usize;
        let mut fused_values = embeds
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        for (i, value) in fused_values[hidden..2 * hidden].iter_mut().enumerate() {
            *value += 0.25 + i as f32 * 0.01;
        }
        let fused = Array::from_slice(&fused_values, &shape);
        let mut text_target_cache = model.new_cache();
        let (text_hidden, _) = model
            .hidden_and_logits(
                &crate::primitives::input_ids(&ids),
                &mut text_target_cache,
                0,
            )
            .unwrap();
        let mut fused_target_cache = model.new_cache();
        let (fused_hidden, _) = model
            .hidden_and_logits_from_embeds_with_deepstack(
                &fused,
                [&positions, &positions, &positions],
                &mut fused_target_cache,
                &[false, true, false],
                &[],
            )
            .unwrap();
        let seed_indices = Array::from_slice(&[0i32, 1], &[2]);
        let shifted_indices = Array::from_slice(&[1i32, 2], &[2]);
        let text_aligned = text_hidden.take_axis(&seed_indices, 1).unwrap();
        let fused_aligned = fused_hidden.take_axis(&seed_indices, 1).unwrap();
        let fused_shifted = fused.take_axis(&shifted_indices, 1).unwrap();
        let mut text_mtp_cache = model.new_mtp_cache().unwrap();
        let (_, text_mtp_logits) = model
            .mtp_step(
                &crate::primitives::input_ids(&ids[1..]),
                &text_aligned,
                &mut text_mtp_cache,
                1,
            )
            .unwrap();
        let mut fused_mtp_cache = model.new_mtp_cache().unwrap();
        let seed_positions = [1, 2];
        let (_, fused_mtp_logits) = model
            .mtp_step_from_embeds(
                &fused_shifted,
                &fused_aligned,
                &mut fused_mtp_cache,
                [&seed_positions, &seed_positions, &seed_positions],
            )
            .unwrap();
        let text_logits_f32 = text_mtp_logits.as_dtype(Dtype::Float32).unwrap();
        let fused_logits_f32 = fused_mtp_logits.as_dtype(Dtype::Float32).unwrap();
        let text_logits = text_logits_f32.as_slice::<f32>();
        let fused_logits = fused_logits_f32.as_slice::<f32>();
        assert!(
            text_logits
                .iter()
                .zip(fused_logits)
                .any(|(text, fused)| (text - fused).abs() > 1.0e-5),
            "an encoded visual row must change the MTP seed distribution"
        );
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
        model
            .forward(&Array::from_slice(&[1i32, 2, 3], &[1, 3]), &mut cache, 0)
            .unwrap();
        assert_eq!(cache.offset(), 3);
        // One decode step at offset 3.
        let logits = model
            .forward(&Array::from_slice(&[4i32], &[1, 1]), &mut cache, 3)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, cfg.vocab_size]);
        assert_eq!(cache.offset(), 4);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite());
        }
    }

    /// The whole Gated DeltaNet layer, validated against a numeric oracle from the exact
    /// `Qwen3_5GatedDeltaNet.forward` reference (4-way in-projection → short conv → contiguous q|k|v
    /// split → L2-norm + q-scale → GQA delta recurrence → gated RMS-norm(z) → out-proj).
    ///
    /// **Single token (S=1).** MLX routes f32 matmuls through an exact GEMV when M=1 but a
    /// reduced-precision (bf16-class) GEMM when M>1, so a multi-token f32 oracle floors at ~6e-3
    /// regardless of correctness. With S=1 every projection is a GEMV, so the whole layer runs in
    /// exact f32 and the match is tight (≈1e-5) — pinning the *structural* assembly precisely. The
    /// conv's multi-tap and the recurrence's multi-step carry are covered tightly by the primitive
    /// oracles (sc-7627); cross-token cache carry by `prefill_equals_stepwise_decode`. Regenerate the
    /// fixture with `/tmp/gen_s1.py` (see story sc-7629).
    #[test]
    fn deltanet_layer_matches_qwen3_5_reference() {
        let json: serde_json::Value =
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
        let (h, hk, hv, dk, dv) = (8i32, 2i32, 4i32, 4i32, 4i32);
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let kk = 4i32;
        let (b, s) = (1i32, 1i32);

        let mk = |k: &str, shape: &[i32]| Array::from_slice(&arr(k), shape);
        let proj = |k: &str, shape: &[i32]| Projection::load(mk(k, shape), None).unwrap();
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
        assert_eq!(out.shape(), &[b, s, h]);

        let got = out
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
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
        assert_eq!(cache.offset(), s);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
    }

    /// A model-level invariant the hybrid cache must satisfy: prefilling a sequence in one pass must
    /// produce the same final-token logits as feeding the tokens one at a time carrying the cache
    /// (conv tail + recurrent SSM state for linear layers, growing KV for full-attention). Run on the
    /// production bf16 path (MLX-vs-MLX, so no f32-GEMM-floor concern) — this is what guarantees the
    /// short conv and delta recurrence resume correctly at decode time on real weights.
    #[test]
    fn prefill_equals_stepwise_decode() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        let toks = [1i32, 7, 3, 42, 9, 2];

        // One-shot prefill of the whole sequence; keep the last-token logits.
        let mut c_pre = model.new_cache();
        let prefill = model
            .decode_logits(
                &Array::from_slice(&toks, &[1, toks.len() as i32]),
                &mut c_pre,
                0,
            )
            .unwrap();

        // Token-by-token with cache carry; keep the final step's logits.
        let mut c_step = model.new_cache();
        let mut last = None;
        for (i, &tok) in toks.iter().enumerate() {
            last = Some(
                model
                    .decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32)
                    .unwrap(),
            );
        }
        let step = last.unwrap();

        assert_eq!(c_pre.offset(), toks.len() as i32);
        assert_eq!(c_step.offset(), toks.len() as i32);
        let a = prefill
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let b = step
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // Both bf16 paths; allow small per-op bf16 reorder noise, far below any structural error.
        assert!(
            md < 5e-2,
            "prefill vs stepwise last-token logits diverged: max abs diff {md}"
        );
    }

    /// AC (sc-24443): the Gated DeltaNet dispatch (fused Metal kernel for prefill, decode and
    /// verify) leaves Qwen35 greedy decoding unchanged against the op-by-op reference recurrence —
    /// for a short and a multi-chunk prompt, a speculative-verify-width forward, and the decode
    /// steps after it — on the production bf16 path, with every logit within one bf16 ULP of the
    /// reference's. Runs on a GPU stream (the kernel's route) whatever the process default device.
    #[test]
    fn greedy_tokens_match_the_ops_reference_recurrence() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), greedy_tokens_match_on_the_gpu);
    }

    fn greedy_tokens_match_on_the_gpu() {
        use crate::primitives::gated_delta::{recording_routes, with_ops_reference, Route};
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg)
                .unwrap();
        let argmax_rows = |logits: &Array| -> Vec<i32> {
            let rows = logits.as_dtype(Dtype::Float32).unwrap();
            let v = *rows.shape().last().unwrap() as usize;
            host(&rows)
                .chunks(v)
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
                        .0 as i32
                })
                .collect()
        };
        // Prefill, a 3-token verify block, then 6 greedy decode steps; every row's argmax.
        let run = |prompt: &[i32]| -> (Vec<i32>, Vec<f32>) {
            let mut cache = model.new_cache();
            let n = prompt.len() as i32;
            let mut out = Vec::new();
            let mut logits = Vec::new();
            let pre = model
                .decode_logits(&Array::from_slice(prompt, &[1, n]), &mut cache, 0)
                .unwrap();
            let mut next = argmax_rows(&pre)[0];
            logits.extend(host(&pre.as_dtype(Dtype::Float32).unwrap()));
            let block = [next, (next + 1) % 50, (next + 2) % 50];
            let verify = model
                .forward(&Array::from_slice(&block, &[1, 3]), &mut cache, n)
                .unwrap();
            let rows = argmax_rows(&verify);
            out.extend(&rows);
            logits.extend(host(&verify.as_dtype(Dtype::Float32).unwrap()));
            next = rows[2];
            for i in 0..6 {
                let step = model
                    .decode_logits(&Array::from_slice(&[next], &[1, 1]), &mut cache, n + 3 + i)
                    .unwrap();
                next = argmax_rows(&step)[0];
                out.push(next);
                logits.extend(host(&step.as_dtype(Dtype::Float32).unwrap()));
            }
            (out, logits)
        };
        let long: Vec<i32> = (0..90).map(|i| (i * 7 + 3) % 50).collect();
        for prompt in [&[1i32, 7, 3, 42, 9, 2][..], &long[..]] {
            let ((tokens, logits), routes) = recording_routes(|| run(prompt));
            let ((ref_tokens, ref_logits), ref_routes) =
                recording_routes(|| with_ops_reference(|| run(prompt)));
            // The production run took the fused kernel for every recurrence call and the reference
            // run the op loop — so the comparison is kernel against ops, not a path against itself.
            assert!(
                !routes.is_empty() && routes.iter().all(|r| *r == Route::Kernel),
                "production routes {routes:?}"
            );
            assert!(
                !ref_routes.is_empty() && ref_routes.iter().all(|r| *r == Route::Ops),
                "reference routes {ref_routes:?}"
            );
            let md = logits
                .iter()
                .zip(&ref_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // The logits are bf16: one ULP at the largest reference logit's binade (8 significand
            // bits, so the spacing in [2^e, 2^(e+1)) is 2^(e-7)) bounds a last-bit rounding flip.
            let scale = ref_logits.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let ulp = 2f32.powi(scale.log2().floor() as i32 - 7);
            eprintln!(
                "prompt len {}: max logit diff vs ops reference {md:.2e} (bf16 ulp {ulp:.2e} at \
                 |logit| {scale:.2e})",
                prompt.len()
            );
            assert_eq!(
                tokens,
                ref_tokens,
                "prompt len {}: greedy tokens diverged (max logit diff {md})",
                prompt.len()
            );
            assert!(
                md <= ulp,
                "prompt len {}: max logit diff {md} exceeds one bf16 ulp {ulp}",
                prompt.len()
            );
        }
    }

    /// A MoE config (`qwen3_5_moe`, the 35B-A3B shape, scaled down): 6 experts, top-2, with a shared
    /// expert. Same 4-layer 3:1 mixer schedule as [`cfg_json`].
    pub(crate) fn cfg_json_moe() -> serde_json::Value {
        let mut v = cfg_json();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("model_type".into(), json!("qwen3_5_moe_text"));
        tc.insert("num_experts".into(), json!(6));
        tc.insert("num_experts_per_tok".into(), json!(2));
        tc.insert("moe_intermediate_size".into(), json!(16));
        tc.insert("shared_expert_intermediate_size".into(), json!(16));
        v
    }

    /// [`cfg_json_moe`] with a configured MTP head — the 35B-A3B checkpoint's layout (sc-24438).
    pub(crate) fn cfg_json_moe_mtp() -> serde_json::Value {
        let mut v = cfg_json_moe();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("mtp_num_hidden_layers".into(), json!(1));
        tc.insert("mtp_use_dedicated_embeddings".into(), json!(false));
        v
    }

    /// Synthetic tensors for a sparse-MoE `cfg` with an MTP head whose predictor layer carries a
    /// sparse-MoE FFN (router, experts, shared expert) instead of the dense MLP — as the 35B-A3B
    /// checkpoint ships it (sc-24438) — with every MoE tensor (body and head) seeded **non-zero
    /// random**. `fused` stores the experts as the Qwen3.6 release does (`experts.gate_up_proj` /
    /// `experts.down_proj`); otherwise each expert under its own keys, as the bf16 Qwen3.5 release
    /// does (`experts.{e}.{gate,up,down}_proj.weight`) — the same values either way.
    pub(crate) fn seeded_moe_mtp_tensors(
        cfg: &Qwen35Config,
        fused: bool,
    ) -> HashMap<String, Array> {
        use crate::primitives::sampler::{SplitMix64, TokenRng};
        let moe = cfg.moe.as_ref().expect("a MoE config");
        let h = cfg.hidden_size;
        let (e, mi, si) = (
            moe.num_experts,
            moe.moe_intermediate_size,
            moe.shared_expert_intermediate_size,
        );
        let mut m = synthetic_tensors_with_prefix(cfg, "model.language_model");
        for dense in ["gate_proj", "up_proj", "down_proj"] {
            assert!(m
                .remove(&format!("mtp.layers.0.mlp.{dense}.weight"))
                .is_some());
        }
        let mut rng = SplitMix64::new(0x5EED_24438);
        let mut layers: Vec<String> = (0..cfg.num_layers)
            .map(|i| format!("model.language_model.layers.{i}."))
            .collect();
        layers.push("mtp.layers.0.".into());
        for lp in layers {
            let mut rand = |key: &str, shape: &[i32]| {
                let n: i32 = shape.iter().product();
                let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
                m.insert(format!("{lp}{key}"), Array::from_slice(&data, shape));
                data
            };
            let gate_up = rand("mlp.experts.gate_up_proj", &[e, 2 * mi, h]);
            let down = rand("mlp.experts.down_proj", &[e, h, mi]);
            rand("mlp.gate.weight", &[e, h]);
            rand("mlp.shared_expert.gate_proj.weight", &[si, h]);
            rand("mlp.shared_expert.up_proj.weight", &[si, h]);
            rand("mlp.shared_expert.down_proj.weight", &[h, si]);
            rand("mlp.shared_expert_gate.weight", &[1, h]);
            if !fused {
                m.remove(&format!("{lp}mlp.experts.gate_up_proj"));
                m.remove(&format!("{lp}mlp.experts.down_proj"));
                let (rows, bank) = ((mi * h) as usize, (2 * mi * h) as usize);
                for x in 0..e as usize {
                    let key = |p: &str| format!("{lp}mlp.experts.{x}.{p}_proj.weight");
                    let gu = &gate_up[x * bank..(x + 1) * bank];
                    m.insert(key("gate"), Array::from_slice(&gu[..rows], &[mi, h]));
                    m.insert(key("up"), Array::from_slice(&gu[rows..], &[mi, h]));
                    let dn = &down[x * rows..(x + 1) * rows];
                    m.insert(key("down"), Array::from_slice(dn, &[h, mi]));
                }
            }
        }
        m
    }

    fn load_moe_mtp(cfg: &Qwen35Config, tensors: HashMap<String, Array>) -> Qwen35Model {
        Qwen35Model::from_weights(
            &Weights::from_map(tensors),
            "model.language_model",
            cfg.clone(),
        )
        .expect("a MoE checkpoint with an MTP head loads")
    }

    fn host_f32(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// sc-24438 AC2 (a): a sparse-MoE checkpoint carrying an MTP head loads the head — its
    /// predictor layer the body's sparse-MoE block — and the fused (Qwen3.6) and per-expert
    /// (Qwen3.5) expert layouts of the same seeded non-zero weights give identical target logits
    /// and identical `mtp_step` logits / hidden state.
    #[test]
    fn a_moe_mtp_head_runs_identically_in_both_expert_layouts() {
        let cfg = Qwen35Config::from_json(&cfg_json_moe_mtp()).unwrap();
        let fused = load_moe_mtp(&cfg, seeded_moe_mtp_tensors(&cfg, true));
        let split = load_moe_mtp(&cfg, seeded_moe_mtp_tensors(&cfg, false));
        for model in [&fused, &split] {
            assert!(model.has_mtp());
            let Ffn::Moe(_) = &model.mtp.as_ref().unwrap().layers[0].ffn else {
                panic!("the predictor layer is the sparse-MoE block");
            };
        }
        let ids = Array::from_slice(&[1i32, 7, 3, 42], &[1, 4]);
        let run = |model: &Qwen35Model| {
            let hidden = model.hidden(&ids, &mut model.new_cache(), 0).unwrap();
            let mut cache = model.new_mtp_cache().unwrap();
            let shifted = Array::from_slice(&[7i32, 3, 42, 9], &[1, 4]);
            let (mtp_hidden, mtp_logits) =
                model.mtp_step(&shifted, &hidden, &mut cache, 0).unwrap();
            (
                host_f32(&hidden),
                host_f32(&mtp_hidden),
                host_f32(&mtp_logits),
            )
        };
        let (a, b) = (run(&fused), run(&split));
        assert!(
            a.2.iter().any(|x| x.abs() > 1e-3),
            "non-degenerate MTP logits"
        );
        assert_eq!(a.0, b.0, "target hidden");
        assert_eq!(a.1, b.1, "MTP hidden");
        assert_eq!(a.2, b.2, "MTP logits");
    }

    /// sc-24438 AC2 (b): the MTP predictor layer's FFN, built by the loader under the
    /// `mtp.layers.0.` prefix from the `Qwen3_5MoeSparseMoeBlock.forward` oracle's weights (the
    /// fused layout and the per-expert split of it), reproduces the oracle's output — the check
    /// [`moe_ffn_matches_qwen3_5_moe_reference`] holds the block itself to, at bf16 tolerance.
    #[test]
    fn the_mtp_layer_ffn_matches_the_moe_reference() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_moe_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let (h, e, mi) = (8i32, 6i32, 4i32);
        let mut v = cfg_json_moe_mtp();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("hidden_size".into(), json!(h));
        tc.insert("moe_intermediate_size".into(), json!(mi));
        tc.insert("shared_expert_intermediate_size".into(), json!(mi));
        let cfg = Qwen35Config::from_json(&v).unwrap();
        assert_eq!(
            (
                cfg.moe.unwrap().num_experts,
                cfg.moe.unwrap().experts_per_tok
            ),
            (e, 2)
        );
        for fused in [true, false] {
            let mut m = seeded_moe_mtp_tensors(&cfg, fused);
            let lp = |s: &str| format!("mtp.layers.0.mlp.{s}");
            let mut put = |key: String, data: &[f32], shape: &[i32]| {
                m.insert(key, Array::from_slice(data, shape));
            };
            let (gate_up, down) = (arr("gate_up"), arr("down"));
            if fused {
                put(lp("experts.gate_up_proj"), &gate_up, &[e, 2 * mi, h]);
                put(lp("experts.down_proj"), &down, &[e, h, mi]);
            } else {
                let rows = (mi * h) as usize;
                for x in 0..e as usize {
                    let gu = &gate_up[x * 2 * rows..(x + 1) * 2 * rows];
                    let key = |p: &str| lp(&format!("experts.{x}.{p}_proj.weight"));
                    put(key("gate"), &gu[..rows], &[mi, h]);
                    put(key("up"), &gu[rows..], &[mi, h]);
                    put(key("down"), &down[x * rows..(x + 1) * rows], &[h, mi]);
                }
            }
            put(lp("gate.weight"), &arr("router"), &[e, h]);
            put(
                lp("shared_expert.gate_proj.weight"),
                &arr("sh_gate"),
                &[mi, h],
            );
            put(lp("shared_expert.up_proj.weight"), &arr("sh_up"), &[mi, h]);
            put(
                lp("shared_expert.down_proj.weight"),
                &arr("sh_down"),
                &[h, mi],
            );
            put(lp("shared_expert_gate.weight"), &arr("sh_gatew"), &[1, h]);
            let model = load_moe_mtp(&cfg, m);
            let ffn = &model.mtp.as_ref().unwrap().layers[0].ffn;
            let got = host_f32(
                &ffn.forward(&Array::from_slice(&arr("x"), &[1, 1, h]))
                    .unwrap(),
            );
            let exp = arr("expected_output");
            let md = got
                .iter()
                .zip(&exp)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // The loader stores weights in bf16 (the block test above runs f32): bf16's 2^-8
            // relative step on outputs of magnitude ~0.5, far below a wrong route or expert.
            assert!(
                md < 5e-3,
                "fused {fused}: MTP-layer ffn vs reference: max abs diff {md}\n got {got:?}\n \
                 exp {exp:?}"
            );
        }
    }

    /// The MoE FFN block, validated against a numeric oracle from the exact
    /// `Qwen3_5MoeSparseMoeBlock.forward` reference: softmax router → top-k → renormalize → per-expert
    /// SwiGLU → sigmoid-gated shared expert. Builds the shared [`SparseMoe`] block via the same split
    /// as the loader (`experts.gate_up_proj` → stacked gate / up banks). Single token (S=1) so
    /// MLX runs the exact GEMV path and the match is tight; regenerate with `/tmp/gen_moe.py`.
    #[test]
    fn moe_ffn_matches_qwen3_5_moe_reference() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_moe_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let (h, e, k, mi) = (8i32, 6i32, 2usize, 4i32);
        let mk = |key: &str, shape: &[i32]| Array::from_slice(&arr(key), shape);
        let proj = |a: Array| Projection::load(a, None).unwrap();

        // Split experts.gate_up_proj into the stacked gate / up banks (mirrors the loader).
        let gate_up = mk("gate_up", &[e, 2 * mi, h])
            .reshape(&[e, 2, mi, h])
            .unwrap();
        let half = |i: i32| {
            gate_up
                .take_axis(Array::from_slice(&[i], &[1]), 1)
                .unwrap()
                .reshape(&[e, mi, h])
                .unwrap()
        };
        let bank = |a: Array| SwitchLinear::load(a, None).unwrap();
        let moe = SparseMoe::new(
            mk("router", &[e, h]),
            bank(half(0)),
            bank(half(1)),
            bank(mk("down", &[e, h, mi])),
            SwiGlu {
                gate: proj(mk("sh_gate", &[mi, h])),
                up: proj(mk("sh_up", &[mi, h])),
                down: proj(mk("sh_down", &[h, mi])),
            },
            Some(mk("sh_gatew", &[1, h])),
            MoeRouting {
                experts_per_tok: k,
                norm_topk_prob: true,
                routed_scaling_factor: 1.0,
            },
        )
        .unwrap();

        let out = moe.forward(&mk("x", &[1, 1, h])).unwrap();
        assert_eq!(out.shape(), &[1, 1, h]);
        let got = out
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
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
        let toks = [1i32, 7, 3, 42, 9];
        let mut c_pre = model.new_cache();
        let logits = model
            .forward(
                &Array::from_slice(&toks, &[1, toks.len() as i32]),
                &mut c_pre,
                0,
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, toks.len() as i32, cfg.vocab_size]);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite(), "non-finite MoE logit");
        }

        // Prefill == stepwise decode over the hybrid cache, with the MoE FFN in the loop.
        let pre_last = model
            .decode_logits(
                &Array::from_slice(&toks, &[1, toks.len() as i32]),
                &mut model.new_cache(),
                0,
            )
            .unwrap();
        let mut c_step = model.new_cache();
        let mut last = None;
        for (i, &tok) in toks.iter().enumerate() {
            last = Some(
                model
                    .decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32)
                    .unwrap(),
            );
        }
        let a = pre_last
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let b = last
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let md = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 5e-2,
            "MoE prefill vs stepwise diverged: max abs diff {md}"
        );
    }

    /// sc-24442 AC2: the Qwen35 hybrid decoder (gated full attention with partial M-RoPE
    /// interleaved with linear-attention layers) emits the same greedy tokens under the new `sdpa` routing as under
    /// the pre-sc-24442 one, compared in-process — head dims 8 (unserved), 64 (full kernel) and 256
    /// (vector-only) × GQA 2 / GQA 8 × prompts inside one tile, across tiles, and across many.
    #[test]
    fn greedy_tokens_match_pre_sc24442_sdpa_routing() {
        use crate::primitives::attention::route_override::{
            assert_greedy_matches_pre_sc24442, GreedyComparison,
        };
        use crate::primitives::sampler::{SplitMix64, TokenRng};

        let mut seen = GreedyComparison::default();
        for hd in [8, 64, 256] {
            for (nh, nkv) in [(4, 2), (8, 1)] {
                let mut v = cfg_json();
                let tc = v["text_config"].as_object_mut().unwrap();
                tc.insert("head_dim".into(), json!(hd));
                tc.insert("num_attention_heads".into(), json!(nh));
                tc.insert("num_key_value_heads".into(), json!(nkv));
                // Full attention at layers 1 and 3: an earlier full-attention layer's every row
                // feeds the last-position logits (the last layer's only reaches them via its
                // final row, which no masking or tiling choice changes).
                tc.insert("full_attention_interval".into(), json!(2));
                let cfg = Qwen35Config::from_json(&v).unwrap();
                assert!(!cfg.is_linear(1) && !cfg.is_linear(3));
                // Random (not periodic) values so greedy steps are rarely near-ties. Drawn in key
                // order: a `HashMap`'s iteration order is randomized per process, so drawing in it
                // gave every test process different weights — and the occasional draw whose
                // step-0 drift crosses the bound (the sc-24439 "flake").
                let mut rng = SplitMix64::new(0x2444_2350 + (hd * 16 + nh + nkv) as u64);
                let mut entries: Vec<(String, Array)> =
                    synthetic_weights(&cfg).into_map().into_iter().collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                let weights: HashMap<String, Array> = entries
                    .into_iter()
                    .map(|(key, a)| {
                        let n = a.size();
                        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
                        (key, Array::from_slice(&data, a.shape()))
                    })
                    .collect();
                let model = Qwen35Model::from_weights(
                    &Weights::from_map(weights),
                    "model.language_model",
                    cfg.clone(),
                )
                .unwrap();
                for prompt_len in [5, 20, 70] {
                    let prompt: Vec<i32> = (0..prompt_len)
                        .map(|i| (i * 7 + 3) % cfg.vocab_size)
                        .collect();
                    seen += assert_greedy_matches_pre_sc24442(
                        &format!("qwen35 hd {hd} {nh}/{nkv} prompt {prompt_len}"),
                        &prompt,
                        6,
                        || model.new_cache(),
                        |ids, cache, offset| model.decode_logits(ids, cache, offset).unwrap(),
                    );
                }
            }
        }
        assert!(
            seen.differing_calls > 0,
            "no fixture exercised a routing change"
        );
        assert!(
            seen.compared_steps > seen.tie_steps,
            "most greedy steps must be decisive enough to compare: {seen:?}"
        );
    }

    fn synthetic_model() -> (Qwen35Config, Qwen35Model) {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        (cfg, model)
    }

    /// `mrope_positions` (the `get_rope_index` port) must reproduce the reference 3-D position rows +
    /// `mrope_delta` for an image+text sequence — exact integer index math (oracle /tmp/gen_mrope.py).
    #[test]
    fn mrope_positions_matches_reference() {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_mrope_oracle.json")).unwrap();
        let r = &j["rope_index"];
        let ints = |k: &str| -> Vec<i32> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        };
        let ids = ints("input_ids");
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

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img_tok, merge).unwrap();
        assert_eq!(t, ints("t"));
        assert_eq!(h, ints("h"));
        assert_eq!(w, ints("w"));
        assert_eq!(delta, r["delta"].as_i64().unwrap() as i32);
    }

    // --- Qwen3-VL HF-backed oracles (tools/gen_qwen3vl_mrope_oracle.py) ----------------------------

    fn qwen3vl_mrope_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/qwen3vl_mrope_oracle.json")).unwrap()
    }

    fn ints_of(v: &serde_json::Value) -> Vec<i32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }

    fn grids_of(v: &serde_json::Value) -> Vec<[i32; 3]> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|g| {
                [
                    g[0].as_i64().unwrap() as i32,
                    g[1].as_i64().unwrap() as i32,
                    g[2].as_i64().unwrap() as i32,
                ]
            })
            .collect()
    }

    /// **Interleaved-MRoPE position ids — mixed text/image.** `mrope_positions_mm` must reproduce the
    /// real `Qwen3VLModel.get_rope_index` 3-D rows + `mrope_delta` for a text+image sequence, exact
    /// integer match. This is the Qwen3-VL placement (image block offset by the running cursor, cursor
    /// advanced by `max(t, h/merge, w/merge)`).
    #[test]
    fn qwen3vl_mrope_image_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let r = &j["rope_index_image"];
        let ids = ints_of(&r["input_ids"]);
        let grid = grids_of(&r["image_grid_thw"]);
        let img = j["image_token_id"].as_i64().unwrap() as i32;
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img, merge).unwrap();
        assert_eq!(t, ints_of(&r["t"]), "image t-row vs HF get_rope_index");
        assert_eq!(h, ints_of(&r["h"]), "image h-row vs HF get_rope_index");
        assert_eq!(w, ints_of(&r["w"]), "image w-row vs HF get_rope_index");
        assert_eq!(
            delta,
            r["delta"].as_i64().unwrap() as i32,
            "image mrope_delta"
        );
    }

    /// **Interleaved-MRoPE position ids — synthetic time / multi-frame video axis.** Qwen3-VL splits a
    /// `[t, h, w]` video into `t` per-frame `gt = 1` blocks (timestamps separate frames), so the
    /// temporal index resets per frame and frames are ordered only by the advancing cursor.
    /// `mrope_positions_mm` must reproduce the HF rows + delta exactly for the 2-frame case — the
    /// Qwen3-VL-specific delta a single multi-`t` block would get wrong.
    #[test]
    fn qwen3vl_mrope_video_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let r = &j["rope_index_video"];
        let ids = ints_of(&r["input_ids"]);
        let vgrid = grids_of(&r["video_grid_thw"]);
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let img = j["image_token_id"].as_i64().unwrap() as i32;
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model
            .mrope_positions_mm(&ids, &[], img, &vgrid, vid, merge)
            .unwrap();
        assert_eq!(
            t,
            ints_of(&r["t"]),
            "video t-row vs HF get_rope_index (per-frame reset)"
        );
        assert_eq!(h, ints_of(&r["h"]), "video h-row vs HF get_rope_index");
        assert_eq!(w, ints_of(&r["w"]), "video w-row vs HF get_rope_index");
        assert_eq!(
            delta,
            r["delta"].as_i64().unwrap() as i32,
            "video mrope_delta"
        );
    }

    /// **Interleaved-MRoPE table — end to end at Qwen3-VL config.** Build the partial-rotary RoPE at
    /// the real text config (head_dim 128, `rotary_dim` 128, theta 5e6, `mrope_section [24,20,20]`),
    /// feed it the HF rope rows for the image sequence, and require the interleaved cos/sin tables to
    /// match the reference `apply_interleaved_mrope` within 1e-5 (f32). Closes the loop from
    /// `get_rope_index` rows through the Qwen3-VL interleaving.
    #[test]
    fn qwen3vl_interleaved_cos_sin_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let il = &j["interleaved"];
        let head_dim = j["head_dim"].as_i64().unwrap() as i32;
        let theta = j["rope_theta"].as_f64().unwrap() as f32;
        let section: Vec<usize> = j["mrope_section"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as usize)
            .collect();
        let sections = [section[0], section[1], section[2]];
        let (t, h, w) = (ints_of(&il["t"]), ints_of(&il["h"]), ints_of(&il["w"]));
        let expect = |k: &str| -> Vec<f32> {
            il[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };

        // Qwen3-VL text rope is full-rotary (partial_rotary_factor 1.0 ⇒ rotary_dim == head_dim),
        // NeoX half-split — exactly Rope::partial(head_dim, theta, false).
        let rope = crate::primitives::rope::Rope::partial(head_dim, theta, false);
        let (cos, sin) = rope
            .mrope_interleaved_cos_sin([&t, &h, &w], sections, Dtype::Float32)
            .unwrap();
        assert_eq!(cos.shape(), &[1, t.len() as i32, head_dim]);
        let cmp = |got: &[f32], exp: &[f32]| {
            assert_eq!(got.len(), exp.len(), "table length");
            got.iter()
                .zip(exp)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        let dc = cmp(cos.as_slice::<f32>(), &expect("cos"));
        let ds = cmp(sin.as_slice::<f32>(), &expect("sin"));
        assert!(
            dc < 1e-5,
            "interleaved cos vs HF reference: max abs diff {dc}"
        );
        assert!(
            ds < 1e-5,
            "interleaved sin vs HF reference: max abs diff {ds}"
        );
    }

    /// **Image-placeholder token expansion matches the HF processor.** Given the raw chat ids with a
    /// single `<|image_pad|>` framed by `<|vision_start|>` / `<|vision_end|>`, expanding the
    /// placeholder to `grid.prod() / merge²` copies must reproduce the exact id stream
    /// `Qwen3VLProcessor` emits — same count, same vision framing, surrounding text untouched.
    #[test]
    fn qwen3vl_token_expansion_matches_hf_processor() {
        let j = qwen3vl_mrope_oracle();
        let ex = &j["expand"];
        let expanded_hf = ints_of(&ex["expanded_ids"]);
        let img = ex["image_token_id"].as_i64().unwrap() as i32;
        let vs = ex["vision_start_token_id"].as_i64().unwrap() as i32;
        let ve = ex["vision_end_token_id"].as_i64().unwrap() as i32;
        let g = ints_of(&ex["grid_thw"]); // single [t, h, w]
        let grid = [g[0], g[1], g[2]];
        let merge = ex["merge"].as_i64().unwrap() as i32;
        let expected_count = ex["expected_count"].as_i64().unwrap() as usize;

        // The count the vision tower / processor agree on.
        let count = vision_merged_token_count(grid, merge);
        assert_eq!(
            count, expected_count,
            "merged-token count formula vs HF processor"
        );

        // Reconstruct the *raw* (pre-expansion) chat ids: the single image placeholder framed by
        // vision_start/vision_end, with all surrounding (non-image) tokens preserved in order. The HF
        // expanded ids are that with the placeholder repeated `count` times — so collapsing the image
        // run back to one token recovers the raw stream.
        let mut raw = Vec::new();
        let mut i = 0usize;
        while i < expanded_hf.len() {
            if expanded_hf[i] == img {
                raw.push(img); // one placeholder
                while i < expanded_hf.len() && expanded_hf[i] == img {
                    i += 1;
                }
            } else {
                raw.push(expanded_hf[i]);
                i += 1;
            }
        }

        let expanded = expand_vision_placeholders(&raw, img, &[count]).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded ids vs HF processor");

        // The expansion is framed by exactly one vision_start … vision_end with `count` image tokens
        // between, and the count matches what the merger emits.
        let si = expanded.iter().position(|&x| x == vs).unwrap();
        let ei = expanded.iter().position(|&x| x == ve).unwrap();
        assert_eq!(
            ei - si - 1,
            count,
            "image tokens framed between vision_start/vision_end"
        );
        assert_eq!(
            expanded[si + 1..ei].iter().filter(|&&x| x == img).count(),
            count
        );
    }

    /// **The text-path invariant.** Feeding token embeds + equal (text) 3-D positions through
    /// `decode_logits_from_embeds` must be **bit-identical** to the token-id `decode_logits` — the
    /// interleaved M-RoPE collapses to 1D and the embeds path is the same compute. This is the gate
    /// that the multimodal hook doesn't perturb the (verified) text decoder.
    #[test]
    fn decode_from_embeds_text_only_equals_decode_logits() {
        let (_cfg, model) = synthetic_model();
        let toks = [1i32, 7, 3, 42, 9, 2];
        let ids = Array::from_slice(&toks, &[1, toks.len() as i32]);

        let a = model
            .decode_logits(&ids, &mut model.new_cache(), 0)
            .unwrap();
        let embeds = model.embed_input_ids(&ids).unwrap();
        let pos: Vec<i32> = (0..toks.len() as i32).collect();
        let b = model
            .decode_logits_from_embeds(&embeds, [&pos, &pos, &pos], &mut model.new_cache())
            .unwrap();

        assert_eq!(a.shape(), b.shape());
        let (av, bv) = (
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec(),
            b.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec(),
        );
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
        let (cfg, model) = synthetic_model();
        let hidden = cfg.hidden_size as usize;
        let ids = [7i32, 49, 49, 8, 9]; // two image tokens (id 49) at positions 1,2
                                        // embeds[1,5,hidden]: row r filled with value r.
        let mut e = Vec::new();
        for r in 0..5 {
            e.extend(std::iter::repeat_n(r as f32, hidden));
        }
        let embeds = Array::from_slice(&e, &[1, 5, hidden as i32])
            .as_dtype(compute())
            .unwrap();
        // feats[2,hidden]: row j filled with 100 + j.
        let mut f = Vec::new();
        for j in 0..2 {
            f.extend(std::iter::repeat_n(100.0f32 + j as f32, hidden));
        }
        let feats = Array::from_slice(&f, &[2, hidden as i32]);

        let out = model
            .splice_image_features(&embeds, &ids, &feats, 49)
            .unwrap();
        assert_eq!(out.shape(), &[1, 5, hidden as i32]);
        let v = out
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let row = |r: usize| v[r * hidden]; // first element of each row (whole row is constant)
        assert_eq!(
            [row(0), row(1), row(2), row(3), row(4)],
            [0.0, 100.0, 101.0, 3.0, 4.0]
        );
    }

    /// DeepStack fusion is actually wired through the decoder: feeding non-zero tapped features
    /// through `decode_logits_from_embeds_with_deepstack` must (a) run end to end to finite logits
    /// and (b) **differ** from the same prefill with the fusion disabled (empty taps) — proving the
    /// tapped features are consumed in the decoder layers, not computed and dropped.
    #[test]
    fn deepstack_fused_path_consumes_features_and_differs() {
        let (cfg, model) = synthetic_model();
        let img = 49i32;
        let ids = [1i32, 2, img, img, img, img, 3, 4];
        let grid = vec![[1i32, 4, 4]];
        let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

        let embeds = model.embed_input_ids(&ids_arr).unwrap();
        let feats = Array::from_slice(
            &(0..4 * cfg.hidden_size)
                .map(|i| (i % 7) as f32 * 0.1 - 0.3)
                .collect::<Vec<_>>(),
            &[4, cfg.hidden_size],
        );
        let spliced = model
            .splice_image_features(&embeds, &ids, &feats, img)
            .unwrap();
        let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
        let visual_pos_mask: Vec<bool> = ids.iter().map(|&id| id == img).collect();

        let ds = |scale: f32| {
            Array::from_slice(
                &(0..4 * cfg.hidden_size)
                    .map(|i| (i % 5) as f32 * scale + 0.05)
                    .collect::<Vec<_>>(),
                &[4, cfg.hidden_size],
            )
        };
        let deepstack = [ds(0.2), ds(-0.15)];

        let fused = model
            .decode_logits_from_embeds_with_deepstack(
                &spliced,
                [&t, &h, &w],
                &mut model.new_cache(),
                &visual_pos_mask,
                &deepstack,
            )
            .unwrap();
        let unfused = model
            .decode_logits_from_embeds_with_deepstack(
                &spliced,
                [&t, &h, &w],
                &mut model.new_cache(),
                &visual_pos_mask,
                &[],
            )
            .unwrap();
        let baseline = model
            .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
            .unwrap();

        assert_eq!(fused.shape(), &[1, cfg.vocab_size]);
        let fv = fused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let uv = unfused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let bv = baseline
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(fv.iter().all(|x| x.is_finite()), "non-finite fused logit");

        let unfused_vs_baseline = uv
            .iter()
            .zip(&bv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            unfused_vs_baseline, 0.0,
            "empty-deepstack path must equal plain embeds path"
        );

        let fused_vs_unfused = fv
            .iter()
            .zip(&uv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            fused_vs_unfused > 1e-3,
            "DeepStack fusion did not change the logits (features dropped?): max abs diff {fused_vs_unfused}"
        );

        // The provider drives this decoder through the `VlmDecode` seam, which boxes the cache via
        // `Decode::make_cache` and downcasts it back to `Qwen35Cache` inside `prefill_with_deepstack`.
        // That trait path must reproduce the inherent fused logits bit-for-bit (same model, inputs,
        // and a fresh cache).
        let mut trait_cache = crate::decode::Decode::make_cache(&model);
        let via_trait = crate::models::VlmDecode::prefill_with_deepstack(
            &model,
            &spliced,
            [&t, &h, &w],
            trait_cache.as_mut(),
            &visual_pos_mask,
            &deepstack,
        )
        .unwrap();
        let tv = via_trait
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let trait_vs_inherent = fv
            .iter()
            .zip(&tv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            trait_vs_inherent, 0.0,
            "VlmDecode::prefill_with_deepstack must equal the inherent fused path (cache downcast seam)"
        );
    }

    #[test]
    fn deepstack_seam_is_decoder_agnostic_at_qwen3vl_shapes() {
        let hidden = 4096i32;
        let num_layers = 6usize;
        let taps = [8usize, 16, 24];
        let seq = 8i32;
        let visual_pos_mask: Vec<bool> = (0..seq).map(|i| (2..6).contains(&i)).collect();
        let num_visual = visual_pos_mask.iter().filter(|&&m| m).count() as i32;
        assert_eq!(num_visual, 4);

        let v0 = 0.5f32;
        let h0 = Array::from_slice(&vec![v0; (seq * hidden) as usize], &[1, seq, hidden])
            .as_dtype(compute())
            .unwrap();
        let feat_val = [0.25f32, 0.5, 0.75];
        let deepstack: Vec<Array> = (0..taps.len())
            .map(|t| {
                Array::from_slice(
                    &vec![feat_val[t]; (num_visual * hidden) as usize],
                    &[num_visual, hidden],
                )
                .as_dtype(compute())
                .unwrap()
            })
            .collect();

        let two = Array::from_f32(2.0).as_dtype(compute()).unwrap();
        let mut calls: Vec<usize> = Vec::new();
        let fused = deepstack_fused_decoder_layers(
            &h0,
            &visual_pos_mask,
            &deepstack,
            num_layers,
            |i, h| {
                calls.push(i);
                Ok(multiply(h, &two)?)
            },
        )
        .unwrap();
        assert_eq!(
            calls,
            (0..num_layers).collect::<Vec<_>>(),
            "every decoder layer must run once, in order"
        );

        let unfused =
            deepstack_fused_decoder_layers(&h0, &visual_pos_mask, &[], num_layers, |_i, h| {
                Ok(multiply(h, &two)?)
            })
            .unwrap();

        let scale = 2f32.powi(num_layers as i32);
        let text_expected = v0 * scale;
        let visual_expected: f32 = v0 * scale
            + (0..taps.len())
                .map(|t| feat_val[t] * 2f32.powi((num_layers - 1 - t) as i32))
                .sum::<f32>();

        let fv = fused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let uv = unfused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let row = |buf: &[f32], r: i32| buf[(r * hidden) as usize];
        for r in 0..seq {
            let got = row(&fv, r);
            let want = if visual_pos_mask[r as usize] {
                visual_expected
            } else {
                text_expected
            };
            assert!(
                (got - want).abs() < 1e-1,
                "fused row {r}: got {got}, want {want}"
            );
        }
        assert!((visual_expected - 54.0).abs() < 1e-3);
        assert!((text_expected - 32.0).abs() < 1e-3);

        for r in 0..seq {
            assert!(
                (row(&uv, r) - text_expected).abs() < 1e-1,
                "unfused row {r} must skip injection"
            );
        }
        let fused_vs_unfused = fv
            .iter()
            .zip(&uv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            fused_vs_unfused > 1.0,
            "fused path must differ from non-fused: max abs diff {fused_vs_unfused}"
        );
    }

    /// Smoke: the full image+text path (embed → splice features → M-RoPE positions →
    /// decode_logits_from_embeds) runs end to end and yields finite `[1, vocab]` logits.
    #[test]
    fn image_text_decode_from_embeds_runs() {
        let (cfg, model) = synthetic_model();
        let img = 49i32; // within the synthetic vocab (50) so embed gather is in-bounds
        let ids = [1i32, 2, img, img, img, img, 3, 4]; // 2x2 image (4 tokens) between text
        let grid = vec![[1i32, 4, 4]];
        let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

        let embeds = model.embed_input_ids(&ids_arr).unwrap();
        let feats = Array::from_slice(
            &(0..4 * cfg.hidden_size)
                .map(|i| (i % 7) as f32 * 0.1 - 0.3)
                .collect::<Vec<_>>(),
            &[4, cfg.hidden_size],
        );
        let spliced = model
            .splice_image_features(&embeds, &ids, &feats, img)
            .unwrap();
        let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
        let logits = model
            .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
            .unwrap();
        assert_eq!(logits.shape(), &[1, cfg.vocab_size]);
        assert!(logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .all(|x| x.is_finite()));
    }

    // ---- The DeltaNet checkpoint ring at the model level (sc-24435). ----

    /// Feed `tokens` one forward each, returning the last forward's logits on the host.
    fn step_each(model: &Qwen35Model, cache: &mut Qwen35Cache, tokens: &[i32]) -> Vec<f32> {
        let mut last = Vec::new();
        for &t in tokens {
            let offset = cache.offset();
            let logits = model
                .decode_logits(&crate::primitives::input_ids(&[t]), cache, offset)
                .unwrap();
            last = host(&logits);
        }
        last
    }

    fn prefilled(model: &Qwen35Model, prompt: &[i32]) -> Qwen35Cache {
        let mut cache = model.new_cache();
        let logits = model
            .decode_logits(&crate::primitives::input_ids(prompt), &mut cache, 0)
            .unwrap();
        logits.eval().unwrap();
        cache
    }

    const RING_PROMPT: [i32; 6] = [3, 9, 4, 11, 3, 9];

    /// The draft-model shape of the ring seam: one checkpoint window opened at the step start
    /// spans several single-token forwards (and a multi-token one), and the hybrid cache truncates
    /// to **each** position of it — the DeltaNet layers restore their kept state, the attention KV
    /// drops the rest by offset — after which the next forward's logits are exactly those of a
    /// cache that never saw the dropped tokens.
    #[test]
    fn a_checkpoint_window_over_several_forwards_truncates_to_each_position() {
        let model = crate::decode::engine::tests::qwen35(false);
        let p = RING_PROMPT.len() as i32;
        let drafts = [4, 11, 7, 20];
        let next = 13;
        for j in 0..=drafts.len() {
            let mut cache = prefilled(&model, &RING_PROMPT);
            cache.arm_checkpoints(drafts.len() as i32);
            step_each(&model, &mut cache, &drafts);
            assert_eq!(cache.restorable(), p..p + drafts.len() as i32);
            cache.truncate(p + j as i32).unwrap();
            assert_eq!(cache.offset(), p + j as i32);
            let got = step_each(&model, &mut cache, &[next]);
            let mut reference = prefilled(&model, &RING_PROMPT);
            step_each(&model, &mut reference, &drafts[..j]);
            let want = step_each(&model, &mut reference, &[next]);
            assert_eq!(got, want, "truncated to {j} kept drafts");
        }

        // A multi-token forward inside the window (the verify shape) followed by a single token.
        let mut cache = prefilled(&model, &RING_PROMPT);
        cache.arm_checkpoints(4);
        let offset = cache.offset();
        model
            .decode_logits(
                &crate::primitives::input_ids(&drafts[..3]),
                &mut cache,
                offset,
            )
            .unwrap()
            .eval()
            .unwrap();
        step_each(&model, &mut cache, &drafts[3..]);
        assert_eq!(cache.restorable(), p..p + 4);
        cache.truncate(p + 2).unwrap();
        let got = step_each(&model, &mut cache, &[next]);
        // The reference never saw the third token.
        let mut reference = prefilled(&model, &RING_PROMPT);
        step_each(&model, &mut reference, &drafts[..2]);
        let want = step_each(&model, &mut reference, &[next]);
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // The kept states came out of a 3-token forward, the reference's out of single-token
        // ones: equal up to the multi-row projection's bf16 rounding.
        assert!(max < 5e-2, "restored-from-verify logits differ by {max}");
    }

    /// `truncate` is refused (typed, cache untouched) outside the checkpoint window — with no
    /// window at all, and before the window's start — and a truncation closes the window: the
    /// next forward records nothing and drops it. `0` is a reset and the current length a no-op,
    /// through the [`KvCache`] trait too.
    #[test]
    fn truncate_outside_the_checkpoint_window_is_refused_and_closes_it() {
        let model = crate::decode::engine::tests::qwen35(false);
        let p = RING_PROMPT.len() as i32;
        let mut cache = prefilled(&model, &RING_PROMPT);
        step_each(&model, &mut cache, &[4, 11]);
        let err = cache.truncate(p + 1).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "no window: {err}");
        assert_eq!(cache.offset(), p + 2);

        cache.arm_checkpoints(2);
        step_each(&model, &mut cache, &[7, 20]);
        let err = cache.truncate(p + 1).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "before the window: {err}"
        );
        assert_eq!(
            cache.offset(),
            p + 4,
            "a refused truncate leaves the cache untouched"
        );
        assert!(cache.truncate(p + 5).is_err(), "past the end");

        KvCache::truncate(&mut cache, p + 3).unwrap();
        assert_eq!(cache.offset(), p + 3);
        step_each(&model, &mut cache, &[5]);
        assert!(
            cache.restorable().is_empty(),
            "the truncation closed the window"
        );
        assert!(cache.truncate(p + 3).is_err());

        let offset = cache.offset();
        KvCache::truncate(&mut cache, offset).unwrap();
        assert_eq!(cache.offset(), p + 4);
        KvCache::truncate(&mut cache, 0).unwrap();
        assert_eq!(cache.offset(), 0);
    }

    /// sc-24435: a checkpoint window armed for `2` tokens refuses a 3-token forward (typed)
    /// before any layer runs — the offset, every DeltaNet state and the attention KV are as they
    /// were, so the next forward's logits are a never-refused cache's.
    #[test]
    fn a_forward_past_the_armed_window_is_refused_untouched() {
        let model = crate::decode::engine::tests::qwen35(false);
        let p = RING_PROMPT.len() as i32;
        let mut cache = prefilled(&model, &RING_PROMPT);
        cache.arm_checkpoints(2);
        let before = cache.delta_states();
        let err = model
            .decode_logits(&crate::primitives::input_ids(&[4, 11, 7]), &mut cache, p)
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::CheckpointWindowFull {
                    recorded: 0,
                    requested: 3,
                    max_tokens: 2
                }
            ),
            "{err}"
        );
        assert_eq!(cache.offset(), p);
        assert_eq!(cache.delta_states(), before);
        assert_eq!(cache.checkpointed_tokens(), 0);
        let got = step_each(&model, &mut cache, &[4, 11]);
        let want = step_each(&model, &mut prefilled(&model, &RING_PROMPT), &[4, 11]);
        assert_eq!(got, want);
    }

    /// E7 (sc-24435): the checkpoint ring's resident memory — the window's start state plus every
    /// kept per-token state and conv input, measured on the cache — never exceeds what
    /// [`Qwen35Model::checkpoint_ring_bytes`] prices for the speculative width, for a verify step
    /// (`width + 1` tokens in one forward, the cap the window is armed with) and for a draft
    /// model's window (`width` single-token forwards). Keeping every position — a full acceptance,
    /// `truncate(offset())` — closes the window, so the next forward records nothing and releases
    /// the ring.
    #[test]
    fn the_checkpoint_ring_footprint_is_what_admission_prices() {
        let model = crate::decode::engine::tests::qwen35(false);
        for width in [1usize, 4, 8] {
            let priced = model.checkpoint_ring_bytes(width).unwrap() as usize;
            let cap = width as i32 + 1;
            let tokens: Vec<i32> = (0..cap).map(|i| 4 + i).collect();
            let released = |cache: &mut Qwen35Cache, live: usize, label: &str| {
                let offset = cache.offset();
                cache.truncate(offset).unwrap();
                step_each(&model, cache, &[5]);
                assert!(cache.restorable().is_empty(), "{label}: the window closed");
                assert_eq!(cache.checkpointed_tokens(), 0, "{label}: nothing recorded");
                assert_eq!(cache.recurrent_bytes(), live, "{label}: the ring released");
            };

            let mut cache = prefilled(&model, &RING_PROMPT);
            let live = cache.recurrent_bytes();
            cache.arm_checkpoints(cap);
            step_each(&model, &mut cache, &tokens[..width]);
            let ring = cache.recurrent_bytes() - live;
            assert!(
                ring > 0 && ring <= priced,
                "draft {width}: ring {ring} B, priced {priced} B"
            );
            released(&mut cache, live, &format!("draft {width}"));

            let mut cache = prefilled(&model, &RING_PROMPT);
            cache.arm_checkpoints(cap);
            let offset = cache.offset();
            model
                .decode_logits(&crate::primitives::input_ids(&tokens), &mut cache, offset)
                .unwrap()
                .eval()
                .unwrap();
            let ring = cache.recurrent_bytes() - live;
            assert!(
                ring > 0 && ring <= priced,
                "verify {width}: ring {ring} B, priced {priced} B"
            );
            released(&mut cache, live, &format!("verify {width}"));
        }
        assert!(model.checkpoint_ring_bytes(8) > model.checkpoint_ring_bytes(4));
    }

    /// sc-20671: stored quantized projections whose `scales`/`biases` are F16 (the mlx-community
    /// convention) must not promote the BF16 activations to F32 — the hidden states and the full-
    /// attention K/V stay in the compute dtype. The BF16-scale twin is the unchanged control.
    #[test]
    fn stored_quantized_scales_keep_activations_and_kv_in_compute_dtype() {
        use crate::primitives::quant::QuantizedLinear;
        let mut value = cfg_json();
        // qwen3_5 stores 64-wide groups: widen every stored projection's input to a multiple.
        let text = &mut value["text_config"];
        text["hidden_size"] = json!(64);
        text["intermediate_size"] = json!(128);
        text["head_dim"] = json!(16);
        text["linear_value_head_dim"] = json!(16);
        value["quantization"] = json!({"group_size": 64, "bits": 4});
        let cfg = Qwen35Config::from_json(&value).unwrap();
        let dense = synthetic_weights(&cfg);
        const STORED: [&str; 10] = [
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        for scale_dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let mut map = HashMap::new();
            for key in dense.keys() {
                let array = dense.require(key).unwrap().as_dtype(scale_dtype).unwrap();
                let base = key.strip_suffix(".weight").unwrap_or(key);
                if key.starts_with("model.language_model.layers.")
                    && STORED.iter().any(|p| base.ends_with(p))
                {
                    let q = QuantizedLinear::quantize(&array, 64, 4, None).unwrap();
                    assert_eq!(q.scales.dtype(), scale_dtype);
                    map.insert(format!("{base}.weight"), q.weight);
                    map.insert(format!("{base}.scales"), q.scales);
                    map.insert(format!("{base}.biases"), q.biases);
                } else {
                    map.insert(key.to_string(), array);
                }
            }
            let model = Qwen35Model::from_weights(
                &Weights::from_map(map),
                "model.language_model",
                cfg.clone(),
            )
            .unwrap();
            assert!(model.is_quantized());
            let mut cache = model.new_cache();
            let hidden = model
                .hidden(&Array::from_slice(&[1i32, 2, 3], &[1, 3]), &mut cache, 0)
                .unwrap();
            assert_eq!(hidden.dtype(), compute(), "{scale_dtype:?} hidden");
            let mut attention_layers = 0;
            for layer in &cache.layers {
                if let Qwen35LayerCache::Attn(slot) = layer {
                    let (k, v) = slot.kv.peek(0).unwrap().unwrap();
                    assert_eq!(k.dtype(), compute(), "{scale_dtype:?} keys");
                    assert_eq!(v.dtype(), compute(), "{scale_dtype:?} values");
                    attention_layers += 1;
                }
            }
            assert_eq!(attention_layers, 1);
        }
    }
}
