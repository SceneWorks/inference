//! Qwen2.5 self-attention: GQA (28 query / 4 kv heads), **biased** q/k/v projections + bias-less
//! o_proj, HF half-split RoPE, masked SDPA. No per-head q_norm/k_norm (that's Qwen3 / Z-Image).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, split};
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, matmul_t};

pub struct QwenTextAttention {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QwenTextAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            q_b: w.require(&join(prefix, "q_proj.bias"))?.clone(),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            k_b: w.require(&join(prefix, "k_proj.bias"))?.clone(),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            v_b: w.require(&join(prefix, "v_proj.bias"))?.clone(),
            o_w: w.require(&join(prefix, "o_proj.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b,1,s,s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = linear(x, &self.q_w, &self.q_b)?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k =
            linear(x, &self.k_w, &self.k_b)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v =
            linear(x, &self.v_w, &self.v_b)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat each kv head `groups` times to match the query heads.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b,s,h,hd] → [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Match the mask dtype to the query (the fork's `mask.astype(query.dtype)`), else a dtype
        // mismatch can drop the additive mask in SDPA. trailing `None` = MLX ≥0.30 `sinks` arg.
        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        matmul_t(&o, &self.o_w)
    }
}

/// HF half-split RoPE: `x*cos + rotate_half(x)*sin`, `rotate_half(x) = [-x2, x1]`. `cos`/`sin`
/// `[1,s,hd]` → broadcast over heads (axis 2).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?; // [1,s,1,hd]
    let sin = sin.expand_dims(2)?;
    let parts = split(x, 2, 3)?; // halves along the head dim
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
}

/// Expand `[b,s,hkv,hd]` → `[b,s,hkv*groups,hd]`, repeating each kv head `groups` times
/// consecutively (matching `mx.repeat(x, groups, axis=2)`).
fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?; // [b,s,hkv,1,hd]
    let x = broadcast_to(&x, &[b, s, hkv, groups, hd])?;
    Ok(x.reshape(&[b, s, hkv * groups, hd])?)
}
