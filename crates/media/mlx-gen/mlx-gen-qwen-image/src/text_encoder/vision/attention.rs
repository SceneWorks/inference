//! `VisionAttention`: biased fused `qkv` → per-head split → **`rotate_half` 2-D RoPE (in f32)** →
//! block-diagonal SDPA (one chunk per window, via `cu_seqlens`) → biased `proj`. Port of the fork's
//! `qwen_vision_attention.py`.
//!
//! The RoPE here is the **non-interleaved** `rotate_half` form (`[-x2, x1]`), distinct from the
//! MMDiT's interleaved-complex RoPE — they are NOT interchangeable. Computed in f32 then cast back,
//! matching the fork.

use mlx_rs::ops::split;
use mlx_rs::Array;

use mlx_gen::attention::windowed_sdpa;
use mlx_gen::nn::{apply_rope_rotate_half_f32, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct VisionAttention {
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            qkv_w: w.require(&join(prefix, "qkv.weight"))?.clone(),
            qkv_b: w.require(&join(prefix, "qkv.bias"))?.clone(),
            proj_w: w.require(&join(prefix, "proj.weight"))?.clone(),
            proj_b: w.require(&join(prefix, "proj.bias"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim]`; `cu`: cumulative seqlens for this block's
    /// attention windows (`[0, …, seq]`). `cu.len() > 2` ⇒ block-diagonal (windowed) attention.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let seq = x.shape()[0];
        let (h, hd) = (self.num_heads, self.head_dim);

        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?.reshape(&[seq, 3, h, hd])?;
        let parts = split(&qkv, 3, 1)?; // each [seq, 1, h, hd]
        let to_heads = |p: &Array| -> Result<Array> {
            // [seq, 1, h, hd] → [h, seq, hd]
            Ok(p.reshape(&[seq, h, hd])?.transpose_axes(&[1, 0, 2])?)
        };
        let q = apply_rope_rotate_half_f32(&to_heads(&parts[0])?, cos, sin)?;
        let k = apply_rope_rotate_half_f32(&to_heads(&parts[1])?, cos, sin)?;
        let v = to_heads(&parts[2])?;

        let attn = windowed_sdpa(&q, &k, &v, cu, self.scale)?; // [h, seq, hd]

        let out = attn.transpose_axes(&[1, 0, 2])?.reshape(&[seq, h * hd])?;
        linear(&out, &self.proj_w, &self.proj_b)
    }
}
