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

use mlx_rs::ops::{
    add, concatenate_axis, mean_axis, multiply, rsqrt, sigmoid, split_sections, sum_axis,
};
use mlx_rs::transforms::async_eval;
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::models::deepstack::{self, deepstack_fused_decoder_layers};
use crate::primitives::attention::{sdpa_capped, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv_traced, causal_depthwise_conv_traced_f32_accum, compute_g,
    rms_norm_gated, ConvTrace, DeltaNetCache,
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

/// The shared decoder keeps its established MLX-LM arithmetic. MOSS explicitly selects
/// the frozen Transformers intermediate precision without changing BF16 projections/outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PrecisionProfile {
    #[default]
    Native,
    MossFrozenHf,
}

impl PrecisionProfile {
    fn norm_weight(self, weight: Array, prism: bool) -> Result<Array> {
        if self == Self::MossFrozenHf {
            debug_assert!(!prism);
            Ok(add(
                &weight.as_dtype(Dtype::Float32)?,
                Array::from_f32(1.0),
            )?)
        } else {
            checkpoint_norm_weight(weight, prism)
        }
    }

    fn norm(self, x: &Array, weight: &Array, eps: f32) -> Result<Array> {
        if self == Self::MossFrozenHf {
            // Both fused-kernel inputs are F32, so normalization AND multiplication precede
            // the single output narrowing, without relying on mixed-input promotion.
            Ok(rms_norm(&x.as_dtype(Dtype::Float32)?, weight, eps)?.as_dtype(x.dtype())?)
        } else {
            rms_norm(x, weight, eps)
        }
    }

    fn scaled_query(self, normalized: &Array, inverse_head_dim: f32) -> Result<Array> {
        if self == Self::MossFrozenHf {
            scale(&normalized.as_dtype(Dtype::Float32)?, inverse_head_dim)
        } else {
            scale(normalized, inverse_head_dim)
        }
    }

    fn gated_norm(self, x: &Array, weight: &Array, gate: &Array, eps: f32) -> Result<Array> {
        if self == Self::MossFrozenHf {
            // Frozen Qwen3_5RMSNormGated differs from its zero-centered RMSNorm:
            // narrow the normalized values BEFORE the direct weight multiplication,
            // then apply the F32 SiLU gate before the final narrowing.
            let x32 = x.as_dtype(Dtype::Float32)?;
            let variance = mean_axis(&multiply(&x32, &x32)?, -1, true)?;
            let normalized = multiply(&x32, &rsqrt(&add(&variance, Array::from_f32(eps))?)?)?
                .as_dtype(x.dtype())?;
            let weighted = multiply(weight, &normalized)?;
            Ok(multiply(
                &weighted.as_dtype(Dtype::Float32)?,
                &silu(&gate.as_dtype(Dtype::Float32)?)?,
            )?
            .as_dtype(x.dtype())?)
        } else {
            rms_norm_gated(x, weight, gate, eps)
        }
    }

    fn convolve(self, x: &Array, weight: &Array, state: &Array) -> Result<(Array, ConvTrace)> {
        if self == Self::MossFrozenHf {
            causal_depthwise_conv_traced_f32_accum(x, weight, state)
        } else {
            causal_depthwise_conv_traced(x, weight, state)
        }
    }

    fn silu(self, x: &Array) -> Result<Array> {
        if self == Self::MossFrozenHf {
            Ok(silu(&x.as_dtype(Dtype::Float32)?)?.as_dtype(x.dtype())?)
        } else {
            silu(x)
        }
    }

    fn decay(self, a: &Array, a_log: &Array, dt_bias: &Array) -> Result<Array> {
        if self == Self::MossFrozenHf {
            compute_g(&a.as_dtype(Dtype::Float32)?, a_log, dt_bias)
        } else {
            compute_g(a, a_log, dt_bias)
        }
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
        grid: Default::default(),
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
    precision: PrecisionProfile,
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
    fn normalize_output(&self, x: &Array, gate: &Array) -> Result<Array> {
        self.precision
            .gated_norm(x, &self.norm_weight, gate, self.eps)
    }

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
            self.precision
                .convolve(&mixed, &self.conv_weight, &conv_state)?;
        let cp = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], 2)?;
        let qc = cp[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let kc = cp[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let vc = cp[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        // L2-normalize q/k (eps 1e-6), then scale q by 1/√head_k_dim — `use_qk_l2norm_in_kernel`.
        let inv = (self.head_k_dim as f32).powf(-0.5);
        let qn = self.precision.scaled_query(&l2norm(&qc, 1e-6)?, inv)?;
        let kn = l2norm(&kc, 1e-6)?;

        // The gated delta recurrence, accumulated in f32 (matching the reference kernel). GQA
        // (q/k from Hk key heads → Hv value heads) is handled inside the recurrence primitive.
        // The cache advances itself: an armed (speculative verify) forward also keeps every
        // token's state for the checkpoint ring (sc-24435).
        let beta = sigmoid(&b_in)?;
        let g = self.precision.decay(&a_in, &self.a_log, &self.dt_bias)?;
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
        let out = self.normalize_output(&y.as_dtype(dt)?, &z)?;
        self.out_proj
            .forward(&out.reshape(&[b, s, self.value_dim])?)
    }
}

/// The gated full-attention layer (`Qwen3NextAttention`).
#[derive(Debug)]
struct Qwen35Attention {
    precision: PrecisionProfile,
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
        let q = self.precision.norm(&qp[0], &self.q_norm, self.eps)?; // [B,S,H,hd]
        let gate = qp[1].reshape(&[b, s, self.num_heads * self.head_dim])?;

        let k = self.precision.norm(
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

    fn forward(&self, x: &Array, precision: PrecisionProfile) -> Result<Array> {
        let gate = precision.silu(&self.gate.forward(x)?)?;
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

    fn forward(&self, x: &Array, precision: PrecisionProfile) -> Result<Array> {
        match self {
            Ffn::Dense(m) => m.forward(x, precision),
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
    precision: PrecisionProfile,
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
        let normed = self.precision.norm(x, &self.input_ln, self.eps)?;
        let r = match (&self.mixer, cache) {
            (Mixer::Delta(d), Qwen35LayerCache::Delta(c)) => d.forward(&normed, c)?,
            (Mixer::Attn(a), Qwen35LayerCache::Attn(c)) => a.forward(&normed, cos, sin, c)?,
            _ => return Err(Error::Msg("qwen3_5: cache/mixer type mismatch".into())),
        };
        let h = add(x, &r)?;
        let m = self.ffn.forward(
            &self.precision.norm(&h, &self.post_ln, self.eps)?,
            self.precision,
        )?;
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
    precision: PrecisionProfile,
    embed_tokens: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: Array,
    lm_head: Option<Projection>,
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
        if let Some(head) = &self.lm_head {
            head.push_arrays(&mut top);
        }
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

    /// Final RMS-normalized decoder states `[B, S, hidden]`, without a vocabulary projection.
    /// Use an individual unpadded prompt with a fresh cache for text-encoder inference.
    pub fn forward_hidden(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Array> {
        self.forward_hidden_with_block_hook(input_ids, cache, offset, &mut |_, _| Ok(()))
    }

    /// Final-normalized hidden states with a cooperative block execution boundary.
    /// Boundary zero supplies embeddings before the first block; subsequent boundaries supply
    /// each block output, then the final-normalized output. The hook may evaluate each lazy
    /// output before checking cancellation. The ordinary hidden-only entry point uses a no-op
    /// hook and preserves its existing lazy execution. Discard the per-prompt cache after error.
    pub fn forward_hidden_with_block_hook(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
        hook: &mut dyn FnMut(usize, &Array) -> Result<()>,
    ) -> Result<Array> {
        let h = self.embed_tokens.forward(input_ids)?.as_dtype(compute())?;
        let (cos, sin) = self.rope.cos_sin(h.shape()[1], offset, compute())?;
        let h = walk_hidden_blocks(
            h,
            self.layers.iter().zip(cache.layers.iter_mut()),
            |h, (layer, slot)| layer.forward(&h, &cos, &sin, slot),
            &mut *hook,
        )?;
        let normalized = self.precision.norm(&h, &self.norm, self.eps)?;
        hook(self.layers.len() + 1, &normalized)?;
        Ok(normalized)
    }

    fn language_head(&self) -> Result<&Projection> {
        self.lm_head.as_ref().ok_or_else(|| {
            Error::Config(
                "Qwen35Model was loaded headless; vocabulary logits are unavailable".into(),
            )
        })
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
        let normalized = self.precision.norm(&h, &self.norm, self.eps)?;
        let logits = self.language_head()?.forward(&normalized)?;
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
        let normalized = self.precision.norm(&h, &self.norm, self.eps)?;
        let s = normalized.shape()[1];
        let last = normalized.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.language_head()?.forward(&last)?;
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
        let normalized = self.precision.norm(&h, &self.norm, self.eps)?;
        let logits = self.language_head()?.forward(&normalized)?;
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
        let normalized = self.precision.norm(&h, &self.norm, self.eps)?;
        let s = normalized.shape()[1];
        let last = normalized.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.language_head()?.forward(&last)?;
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
        let logits = self.language_head()?.forward(&normalized)?;
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
        let normed = self.precision.norm(h, &self.norm, self.eps)?;
        self.language_head()?.forward(&normed)
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

    /// Build only the text decoder and final norm, without reading or allocating an LM head.
    /// `prefix` is the exact decoder root, including any frozen `language_model` wrapper.
    /// Parse wrapped configurations with [`Qwen35Config::from_json`].
    pub fn from_weights_headless(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        Self::materialized(w, Self::build_headless_lazy(w, prefix, cfg)?)
    }

    /// Headless dense BF16 decoder with the frozen MOSS Transformers precision boundaries.
    /// Norms, normalized-Q scaling, delta decay and depthwise convolution follow the frozen HF
    /// intermediate casts; weights/projections and hidden outputs remain BF16.
    /// Shared constructors retain their native arithmetic. This profile does not support
    /// quantized, MoE or Prism decoders.
    pub fn from_weights_headless_moss_frozen_hf(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
    ) -> Result<Self> {
        Self::materialized(
            w,
            Self::from_weights_layout(
                w,
                prefix,
                cfg,
                None,
                None,
                false,
                PrecisionProfile::MossFrozenHf,
            )?,
        )
    }

    // Keep headless graph construction private: production loads cross the same coherence boundary.
    fn build_headless_lazy(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        Self::from_weights_layout(w, prefix, cfg, None, None, false, PrecisionProfile::Native)
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
        Self::from_weights_layout(w, prefix, cfg, quant, None, true, PrecisionProfile::Native)
    }

    /// [`Qwen35Model::from_prism_weights`] without evaluating anything (see
    /// [`Qwen35Model::build_lazy`]).
    pub(crate) fn build_prism_lazy(
        w: &Weights,
        cfg: Qwen35Config,
        pack: &PrismMlxPack,
    ) -> Result<Self> {
        Self::from_weights_layout(
            w,
            "language_model.model",
            cfg,
            None,
            Some(pack),
            true,
            PrecisionProfile::Native,
        )
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
        with_lm_head: bool,
        precision: PrecisionProfile,
    ) -> Result<Self> {
        if precision == PrecisionProfile::MossFrozenHf
            && (with_lm_head
                || prism.is_some()
                || quant.is_some()
                || cfg.quantization.is_some()
                || cfg.moe.is_some()
                || compute() != Dtype::Bfloat16)
        {
            return Err(Error::Config(
                "MOSS frozen-HF precision requires a dense unquantized BF16 headless decoder"
                    .into(),
            ));
        }
        let eps = cfg.rms_norm_eps;
        let req = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(compute())?) };
        // Dense HF Qwen3.6 norms are zero-centered. Frozen Prism/Bonsai artifacts have already
        // converted every ordinary RMSNorm tensor to its direct multiplier and must remain raw.
        let norm_w = |key: String| -> Result<Array> {
            let t = req(key)?;
            precision.norm_weight(t, prism.is_some())
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
        let dp = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };

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
        let lm_head = if !with_lm_head {
            None
        } else {
            Some(if cfg.tie_word_embeddings {
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
            })
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
                    precision,
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
                    precision,
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
                precision,
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
        let mtp = if !with_lm_head {
            cfg.mtp_num_hidden_layers = 0;
            None
        } else {
            match core_llm::native_mtp_plan(
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
            }
        };

        let rope = Rope::partial(cfg.rotary_dim(), cfg.rope_theta, false);
        let model = Self {
            precision,
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
        precision: PrecisionProfile::Native,
        input_ln: norm(lp("input_layernorm.weight"))?,
        post_ln: norm(lp("post_attention_layernorm.weight"))?,
        mixer: Mixer::Attn(Qwen35Attention {
            precision: PrecisionProfile::Native,
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

// Tensor-independent traversal lets CPU contract tests exercise cancellation short-circuiting
// without initializing a backend device. Production supplies real layer/cache pairs and Arrays.
fn walk_hidden_blocks<T, L>(
    mut hidden: T,
    layers: impl IntoIterator<Item = L>,
    mut forward: impl FnMut(T, L) -> Result<T>,
    mut hook: impl FnMut(usize, &T) -> Result<()>,
) -> Result<T> {
    hook(0, &hidden)?;
    for (index, layer) in layers.into_iter().enumerate() {
        hidden = forward(hidden, layer)?;
        hook(index + 1, &hidden)?;
    }
    Ok(hidden)
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod hidden_block_hook_cpu_tests {
    use super::*;
    #[test]
    fn noop_preserves_values_and_hooks_observe_completed_boundaries() {
        let mut boundaries = Vec::new();
        let output = walk_hidden_blocks(
            1u32,
            [2, 3, 4],
            |v, n| Ok(v + n),
            |index, v| {
                boundaries.push((index, *v));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(output, 10);
        assert_eq!(boundaries, [(0, 1), (1, 3), (2, 6), (3, 10)]);
        assert_eq!(
            walk_hidden_blocks(1u32, [2, 3, 4], |v, n| Ok(v + n), |_, _| Ok(())).unwrap(),
            output
        );
    }
    #[test]
    fn typed_cancel_prevents_later_blocks_and_initial_cancel_runs_none() {
        let mut forwards = 0;
        let result = walk_hidden_blocks(
            0,
            [1, 2, 3],
            |v, n| {
                forwards += 1;
                Ok(v + n)
            },
            |index, _| {
                if index == 1 {
                    Err(Error::Canceled)
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(forwards, 1);
        let result = walk_hidden_blocks(
            0,
            [1, 2, 3],
            |v, n| {
                forwards += 1;
                Ok(v + n)
            },
            |_, _| Err(Error::Canceled),
        );
        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(forwards, 1);
    }
}
