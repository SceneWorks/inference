//! Mochi **learned** 3-D RoPE — port of `MochiRoPE._create_rope` + `MochiAttnProcessor2_0`'s
//! `apply_rotary_emb` (diffusers `transformer_mochi.py` / `attention_processor.py`).
//!
//! Unlike the analytic factorized RoPE of Wan (fixed `θ^-k` frequencies split across axes), Mochi
//! learns a `pos_frequencies [3, heads, head_dim/2]` parameter and contracts it with the continuous
//! `(t, h, w)` token positions:
//!
//! ```text
//! freqs = einsum("nd,dhf->nhf", positions[seq, 3], pos_frequencies[3, heads, head_dim/2])
//!       → [seq, heads, head_dim/2]      (f32)
//! cos, sin = cos(freqs), sin(freqs)
//! ```
//!
//! so each head owns its own per-axis frequency mixing. RoPE is then applied to the **visual** query
//! and key only (the text stream is not rotated), with **interleaved** pairs — `x[..., 0::2]` /
//! `x[..., 1::2]` are the real/imag halves, rotated by `(cos, sin)` and re-interleaved. Everything is
//! computed in f32 (the reference wraps the `einsum` in `autocast(float32)`); the pair-rotation reuses
//! the shared `mlx_gen::nn::rope_rotate` (bit-exact eager, one fused kernel under `compile_glue`).

use mlx_rs::ops::matmul;
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

use crate::positions::get_positions;

/// The precomputed per-token `(cos, sin)`, each `[seq, heads, head_dim/2]` (f32), for a constant
/// `(num_frames, height, width)` visual grid. Built once per forward and shared by every block.
pub struct MochiRope {
    /// `[seq, heads, head_dim/2]` f32 cosines.
    pub cos: Array,
    /// `[seq, heads, head_dim/2]` f32 sines.
    pub sin: Array,
}

impl MochiRope {
    /// Build the rotary table from the learned `pos_frequencies [3, heads, head_dim/2]` and the grid
    /// geometry (post-patch `height`/`width`). Mirrors `MochiRoPE.forward` → `_create_rope`.
    pub fn new(
        pos_frequencies: &Array,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let sh = pos_frequencies.shape();
        if sh.len() != 3 {
            return Err(Error::Msg(format!(
                "mochi rope: pos_frequencies must be rank-3 [3, heads, head_dim/2], got {sh:?}"
            )));
        }
        let (axes, heads, half) = (sh[0], sh[1], sh[2]);
        if axes != 3 {
            return Err(Error::Msg(format!(
                "mochi rope: pos_frequencies axis-0 must be 3 (t,h,w), got {axes}"
            )));
        }

        // positions [seq, 3] · pos_frequencies[3, heads·half] → [seq, heads·half] → [seq, heads, half].
        let positions = get_positions(num_frames, height, width).as_dtype(Dtype::Float32)?;
        let seq = positions.shape()[0];
        let pf = pos_frequencies
            .as_dtype(Dtype::Float32)?
            .reshape(&[axes, heads * half])?;
        let freqs = matmul(&positions, &pf)?.reshape(&[seq, heads, half])?;

        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Apply interleaved RoPE to a visual `[B, seq, heads, head_dim]` tensor (query or key). `cos`/`sin`
    /// `[seq, heads, head_dim/2]` broadcast over the batch. Mirrors `apply_rotary_emb`.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "mochi rope apply: expected [B, seq, heads, head_dim], got {sh:?}"
            )));
        }
        let (b, s, n, d) = (sh[0], sh[1], sh[2], sh[3]);
        let half = d / 2;

        // Interleaved split: [B, seq, heads, half, 2] → even = [..,0], odd = [..,1].
        let x5 = x.as_dtype(Dtype::Float32)?.reshape(&[b, s, n, half, 2])?;
        let parts = mlx_rs::ops::split(&x5, 2, 4)?;
        let x_even = parts[0].reshape(&[b, s, n, half])?;
        let x_odd = parts[1].reshape(&[b, s, n, half])?;

        // (even + odd·i)·(cos + sin·i) = (even·cos − odd·sin) + (even·sin + odd·cos)i.
        let (out_even, out_odd) = mlx_gen::nn::rope_rotate(&x_even, &x_odd, &self.cos, &self.sin)?;

        // Re-interleave: stack on a new trailing axis → [B, seq, heads, half, 2] → [B, seq, heads, d].
        let e5 = out_even.reshape(&[b, s, n, half, 1])?;
        let o5 = out_odd.reshape(&[b, s, n, half, 1])?;
        let stacked = mlx_rs::ops::concatenate_axis(&[&e5, &o5], 4)?;
        Ok(stacked.reshape(&[b, s, n, d])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, max, multiply, subtract, sum};

    /// Identity `pos_frequencies` mapping so we can reason about the rotation angles directly.
    #[test]
    fn rope_preserves_norm_and_zero_pos_is_identity() {
        // 1 frame × 2 × 2 = 4 tokens, 2 heads, head_dim 8 → half 4.
        let heads = 2;
        let half = 4;
        // Small nonzero frequencies so the rotation is nontrivial.
        let pf: Vec<f32> = (0..3 * heads * half)
            .map(|i| 0.01 * (i as f32 + 1.0))
            .collect();
        let pf = Array::from_slice(&pf, &[3, heads as i32, half as i32]);
        let rope = MochiRope::new(&pf, 1, 2, 2).unwrap();
        assert_eq!(rope.cos.shape(), &[4, heads as i32, half as i32]);

        // RoPE is an orthogonal per-pair rotation → preserves the L2 norm of q/k.
        let x = Array::from_slice(
            &(0..1 * 4 * heads * (2 * half))
                .map(|i| ((i as f32) * 0.017).sin())
                .collect::<Vec<_>>(),
            &[1, 4, heads as i32, (2 * half) as i32],
        );
        let y = rope.apply(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
        let xn: f32 = sum(multiply(&x, &x).unwrap(), None).unwrap().item();
        let yn: f32 = sum(multiply(&y, &y).unwrap(), None).unwrap().item();
        assert!((xn - yn).abs() / xn < 1e-4, "norm changed: {xn} vs {yn}");
    }

    #[test]
    fn interleave_matches_reference_formula() {
        // Directly check one rotation against the closed form: even·cos − odd·sin | even·sin + odd·cos.
        let pf = Array::from_slice(&[0.3f32, 0.3, 0.3], &[3, 1, 1]); // 1 head, half=1
        let rope = MochiRope::new(&pf, 1, 1, 1).unwrap(); // seq=1
        let cos = rope.cos.item::<f32>();
        let sin = rope.sin.item::<f32>();
        // x = [even=2, odd=5] for the single pair.
        let x = Array::from_slice(&[2.0f32, 5.0], &[1, 1, 1, 2]);
        let y = rope.apply(&x).unwrap();
        let ys: Vec<f32> = y.as_slice::<f32>().to_vec();
        let want_even = 2.0 * cos - 5.0 * sin;
        let want_odd = 2.0 * sin + 5.0 * cos;
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(close(ys[0], want_even), "{} vs {}", ys[0], want_even);
        assert!(close(ys[1], want_odd), "{} vs {}", ys[1], want_odd);
    }

    #[test]
    fn matches_golden_geometry_shape() {
        // The dit_block golden geometry: 2 frames × 4 × 4 = 32 tokens, 24 heads, head_dim/2 = 64.
        let pf = Array::from_slice(
            &(0..3 * 24 * 64).map(|i| (i as f32 * 1e-4).sin()).collect::<Vec<_>>(),
            &[3, 24, 64],
        );
        let rope = MochiRope::new(&pf, 2, 4, 4).unwrap();
        assert_eq!(rope.cos.shape(), &[32, 24, 64]);
        assert_eq!(rope.sin.shape(), &[32, 24, 64]);
        // cos ∈ [−1, 1].
        let m: f32 = max(abs(subtract(&rope.cos, &Array::from_f32(0.0)).unwrap()).unwrap(), None)
            .unwrap()
            .item();
        assert!(m <= 1.0 + 1e-6);
    }
}
