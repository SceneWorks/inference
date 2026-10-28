//! `PixtralAttention`: **split** bias-less q/k/v/o → per-head split → **`rotate_half` 2-D RoPE (in
//! f32)** → block-diagonal SDPA (one window per reference image, via `cu_seqlens`) → bias-less o.
//! Port of `PixtralAttention`.
//!
//! Same shape as `mlx-gen-qwen-image`'s vision attention except the projections are four separate
//! bias-less Linears (Pixtral) rather than one fused biased `qkv` (Qwen). The RoPE is the
//! **non-interleaved** `rotate_half` form (`[-x2, x1]`), computed in f32 then cast back — distinct
//! from the MMDiT's interleaved-complex RoPE; they are NOT interchangeable.

use mlx_rs::ops::matmul;
use mlx_rs::Array;

use mlx_gen::attention::windowed_sdpa;
use mlx_gen::nn::apply_rope_rotate_half_f32;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct PixtralAttention {
    q_w: Array,
    k_w: Array,
    v_w: Array,
    o_w: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl PixtralAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            o_w: w.require(&join(prefix, "o_proj.weight"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim]`; `cu`: cumulative seqlens
    /// (`[0, …, seq]`). `cu.len() > 2` ⇒ block-diagonal (per-image) attention.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let seq = x.shape()[0];
        let (h, hd) = (self.num_heads, self.head_dim);

        // [seq, embed] → matmul(x, Wᵀ) → [seq, h, hd] → [h, seq, hd] (bias-less projection).
        let to_heads = |proj_w: &Array| -> Result<Array> {
            Ok(matmul(x, proj_w.t())?
                .reshape(&[seq, h, hd])?
                .transpose_axes(&[1, 0, 2])?)
        };
        let q = apply_rope_rotate_half_f32(&to_heads(&self.q_w)?, cos, sin)?;
        let k = apply_rope_rotate_half_f32(&to_heads(&self.k_w)?, cos, sin)?;
        let v = to_heads(&self.v_w)?;

        let attn = windowed_sdpa(&q, &k, &v, cu, self.scale)?; // [h, seq, hd]

        let out = attn.transpose_axes(&[1, 0, 2])?.reshape(&[seq, h * hd])?;
        Ok(matmul(&out, self.o_w.t())?)
    }
}
