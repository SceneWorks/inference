//! EVA sub-LN `Attention`. Port of `eva_vit_model.py Attention(subln=True, rope=…, xattn=…)`.
//!
//! subln layout: separate `q_proj`/`k_proj`/`v_proj` (Linear, bias=False) plus standalone `q_bias`
//! and `v_bias` params (k has **no** bias), an `inner_attn_ln` (LayerNorm over all-head-dim) before
//! `proj`. RoPE is applied to the **patch** tokens of q/k only (the CLS token at index 0 is left
//! unrotated). Attention itself: scale q by `head_dim**-0.5`, softmax — the reference's explicit
//! path (xformers absent ⇒ `xattn=False`), reproduced by MLX SDPA (softmax in f32).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::eva_clip::rope::VisionRope;
use crate::eva_clip::{join, EPS};

pub struct Attention {
    q_proj_w: mlx_gen::adapters::AdaptableLinear,

    k_proj_w: mlx_gen::adapters::AdaptableLinear,
    v_proj_w: mlx_gen::adapters::AdaptableLinear,

    inner_ln_w: Array,
    inner_ln_b: Array,
    proj_w: mlx_gen::adapters::AdaptableLinear,

    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            q_proj_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "q_proj"),
                Some(w.require(&join(prefix, "q_bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            k_proj_w: mlx_gen::quant::lin(
                w,
                &join(prefix, "k_proj"),
                false,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            v_proj_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "v_proj"),
                Some(w.require(&join(prefix, "v_bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            inner_ln_w: w.require(&join(prefix, "inner_attn_ln.weight"))?.clone(),
            inner_ln_b: w.require(&join(prefix, "inner_attn_ln.bias"))?.clone(),
            proj_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "proj"),
                Some(w.require(&join(prefix, "proj.bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[B, N, C]` (N = 1 CLS + grid² patches). `rope` is the shared block-invariant table.
    pub fn forward(&self, x: &Array, rope: &VisionRope) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let (h, hd) = (self.num_heads, self.head_dim);

        // subln projections: q/v biased, k unbiased.
        let q = self.q_proj_w.forward(x)?;
        let k = self.k_proj_w.forward(x)?;
        let v = self.v_proj_w.forward(x)?;

        // [B, N, C] -> [B, heads, N, hd]
        let to_heads = |t: &Array| -> Result<Array> {
            Ok(t.reshape(&[b, n, h, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = self.rope_patch_tokens(&to_heads(&q)?, n, rope)?;
        let k = self.rope_patch_tokens(&to_heads(&k)?, n, rope)?;
        let v = to_heads(&v)?;

        let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, h * hd])?;
        let out = layer_norm(&out, Some(&self.inner_ln_w), Some(&self.inner_ln_b), EPS)?;
        self.proj_w.forward(&out)
    }

    /// Apply RoPE to `x[:, :, 1:, :]` (patch tokens) only; the CLS token at index 0 is untouched.
    fn rope_patch_tokens(&self, x: &Array, n: i32, rope: &VisionRope) -> Result<Array> {
        let idx_cls = Array::from_slice(&[0i32], &[1]);
        let idx_pat = Array::from_slice(&(1..n).collect::<Vec<i32>>(), &[n - 1]);
        let cls = x.take_axis(&idx_cls, 2)?;
        let pat = rope.apply(&x.take_axis(&idx_pat, 2)?)?;
        Ok(concatenate_axis(&[&cls, &pat], 2)?)
    }
}
