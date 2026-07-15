//! Mochi **learned** 3-D RoPE — port of `MochiRoPE._create_rope` + `MochiAttnProcessor2_0`'s
//! `apply_rotary_emb` (diffusers `transformer_mochi.py`), plus the `_get_positions` / `_centers`
//! visual-grid positions.
//!
//! Unlike the analytic factorized RoPE of Wan, Mochi learns a `pos_frequencies [3, heads, head_dim/2]`
//! parameter and contracts it with the continuous `(t, h, w)` token positions:
//!
//! ```text
//! freqs = einsum("nd,dhf->nhf", positions[seq, 3], pos_frequencies[3, heads, head_dim/2])
//!       → [seq, heads, head_dim/2]      (f32)
//! cos, sin = cos(freqs), sin(freqs)
//! ```
//!
//! so each head owns its own per-axis frequency mixing. RoPE is applied to the **visual** query/key
//! only (the text stream is not rotated), with **interleaved** pairs — `x[..., 0::2]` / `x[..., 1::2]`
//! are the real/imag halves, rotated by `(cos, sin)` and re-interleaved. Everything is f32 (the
//! reference wraps the `einsum` in `autocast(float32)`).
//!
//! The visual sequence is a `(num_frames, post_patch_height, post_patch_width)` grid; each token gets a
//! continuous 3-vector `(t, h, w)`: temporal is the raw frame index, the two spatial axes are
//! **area-normalized cell centers** (scaled so the grid covers the fixed reference area `192·192`
//! regardless of resolution). The stack is row-major over `(t, h, w)`, so `p = ((t·H) + h)·W + w`.
//! Positions are computed in **f64** on the host and cast to f32 at the edge.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::{CandleError, Result};

/// Base reference area for the spatial-axis normalization (`MochiRoPE(base_height=192, base_width=192)`).
const BASE_AREA: f64 = 192.0 * 192.0;

/// `_centers(start, stop, num)`: the `num` midpoints of an evenly-spaced `[start, stop]` partition.
fn centers(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / num as f64;
    (0..num)
        .map(|k| {
            let e0 = start + step * k as f64;
            let e1 = start + step * (k + 1) as f64;
            (e0 + e1) / 2.0
        })
        .collect()
}

/// The per-token `(t, h, w)` positions for a `(num_frames, height, width)` visual grid, row-major over
/// `(t, h, w)` → `[seq, 3]` f32 (`seq = num_frames·height·width`). `height`/`width` are the
/// **post-patch** latent dims.
pub fn get_positions(num_frames: usize, height: usize, width: usize, device: &Device) -> Result<Tensor> {
    let scale = (BASE_AREA / (height as f64 * width as f64)).sqrt();
    let h_centers = centers(-(height as f64) * scale / 2.0, height as f64 * scale / 2.0, height);
    let w_centers = centers(-(width as f64) * scale / 2.0, width as f64 * scale / 2.0, width);

    let seq = num_frames * height * width;
    let mut data = Vec::with_capacity(seq * 3);
    for t in 0..num_frames {
        for &hc in &h_centers {
            for &wc in &w_centers {
                data.push(t as f32);
                data.push(hc as f32);
                data.push(wc as f32);
            }
        }
    }
    Ok(Tensor::from_vec(data, (seq, 3), device)?)
}

/// The precomputed per-token `(cos, sin)`, each `[seq, heads, head_dim/2]` (f32), for a constant
/// `(num_frames, height, width)` visual grid. Built once per forward and shared by every block.
pub struct MochiRope {
    /// `[seq, heads, head_dim/2]` f32 cosines.
    pub cos: Tensor,
    /// `[seq, heads, head_dim/2]` f32 sines.
    pub sin: Tensor,
}

impl MochiRope {
    /// Build the rotary table from the learned `pos_frequencies [3, heads, head_dim/2]` and the grid
    /// geometry (post-patch `height`/`width`). Mirrors `MochiRoPE.forward` → `_create_rope`.
    pub fn new(
        pos_frequencies: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<Self> {
        let sh = pos_frequencies.dims();
        if sh.len() != 3 {
            return Err(CandleError::Msg(format!(
                "mochi rope: pos_frequencies must be rank-3 [3, heads, head_dim/2], got {sh:?}"
            )));
        }
        let (axes, heads, half) = (sh[0], sh[1], sh[2]);
        if axes != 3 {
            return Err(CandleError::Msg(format!(
                "mochi rope: pos_frequencies axis-0 must be 3 (t,h,w), got {axes}"
            )));
        }

        // positions [seq, 3] · pos_frequencies[3, heads·half] → [seq, heads·half] → [seq, heads, half].
        let positions = get_positions(num_frames, height, width, device)?; // f32
        let seq = positions.dim(0)?;
        let pf = pos_frequencies
            .to_dtype(DType::F32)?
            .reshape((axes, heads * half))?;
        let freqs = positions.matmul(&pf)?.reshape((seq, heads, half))?;

        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Build directly from precomputed `(cos, sin)` tables (each `[seq, heads, head_dim/2]`) — the
    /// parity path feeding a golden's captured `image_rotary_emb`.
    pub fn from_parts(cos: Tensor, sin: Tensor) -> Self {
        Self { cos, sin }
    }

    /// Apply **interleaved** RoPE to a visual `[B, seq, heads, head_dim]` tensor (query or key). `cos`/
    /// `sin` `[seq, heads, head_dim/2]` broadcast over the batch. Mirrors `apply_rotary_emb`; computed
    /// in f32.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 4 {
            return Err(CandleError::Msg(format!(
                "mochi rope apply: expected [B, seq, heads, head_dim], got {dims:?}"
            )));
        }
        let (b, s, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let half = d / 2;

        // Interleaved split: [B, seq, heads, half, 2] → even = [..,0], odd = [..,1].
        let x5 = x.to_dtype(DType::F32)?.reshape((b, s, n, half, 2))?;
        let x_even = x5.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?.contiguous()?;
        let x_odd = x5.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?.contiguous()?;

        // cos/sin [seq, heads, half] → [1, seq, heads, half] to broadcast over the batch.
        let cos = self.cos.unsqueeze(0)?;
        let sin = self.sin.unsqueeze(0)?;

        // (even + odd·i)·(cos + sin·i) = (even·cos − odd·sin) + (even·sin + odd·cos)i.
        let out_even = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
        let out_odd = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;

        // Re-interleave: stack on a new trailing axis → [B, seq, heads, half, 2] → [B, seq, heads, d].
        let e5 = out_even.unsqueeze(D::Minus1)?;
        let o5 = out_odd.unsqueeze(D::Minus1)?;
        let stacked = Tensor::cat(&[&e5, &o5], D::Minus1)?;
        Ok(stacked.reshape((b, s, n, d))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centers_are_evenly_spaced_midpoints() {
        // The golden geometry: post-patch 4×4, scale = sqrt(36864/16) = 48 → extent ±96, 4 centers.
        let c = centers(-96.0, 96.0, 4);
        assert_eq!(c, vec![-72.0, -24.0, 24.0, 72.0]);
    }

    #[test]
    fn positions_shape_and_row_major_layout() {
        let dev = Device::Cpu;
        // 2 frames × 4 × 4 = 32 tokens, matching the dit_block golden's visual sequence.
        let pos = get_positions(2, 4, 4, &dev).unwrap();
        assert_eq!(pos.dims(), &[32, 3]);
        let v = pos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&v[0..3], &[0.0, -72.0, -72.0]); // (t0, h0, w0)
        assert_eq!(&v[3..6], &[0.0, -72.0, -24.0]); // (t0, h0, w1) — w advances fastest
        assert_eq!(&v[48..51], &[1.0, -72.0, -72.0]); // frame 1 begins at row 16
    }

    #[test]
    fn interleave_matches_reference_formula() {
        let dev = Device::Cpu;
        // 1 head, half=1, seq=1. pos_frequencies all 0.3.
        let pf = Tensor::from_vec(vec![0.3f32, 0.3, 0.3], (3, 1, 1), &dev).unwrap();
        let rope = MochiRope::new(&pf, 1, 1, 1, &dev).unwrap();
        let cos = rope.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let sin = rope.sin.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        // x = [even=2, odd=5].
        let x = Tensor::from_vec(vec![2.0f32, 5.0], (1, 1, 1, 2), &dev).unwrap();
        let y = rope.apply(&x).unwrap();
        let ys = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want_even = 2.0 * cos - 5.0 * sin;
        let want_odd = 2.0 * sin + 5.0 * cos;
        assert!((ys[0] - want_even).abs() < 1e-5, "{} vs {}", ys[0], want_even);
        assert!((ys[1] - want_odd).abs() < 1e-5, "{} vs {}", ys[1], want_odd);
    }

    #[test]
    fn rope_preserves_norm_and_geometry_shape() {
        let dev = Device::Cpu;
        // dit_block golden geometry: 2 frames × 4 × 4 = 32 tokens, 24 heads, head_dim/2 = 64.
        let n: usize = 3 * 24 * 64;
        let pf_data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
        let pf = Tensor::from_vec(pf_data, (3, 24, 64), &dev).unwrap();
        let rope = MochiRope::new(&pf, 2, 4, 4, &dev).unwrap();
        assert_eq!(rope.cos.dims(), &[32, 24, 64]);
        assert_eq!(rope.sin.dims(), &[32, 24, 64]);

        // RoPE is an orthogonal per-pair rotation → preserves the L2 norm of q/k.
        let xn: usize = 1 * 32 * 24 * 128;
        let xdata: Vec<f32> = (0..xn).map(|i| ((i as f32) * 0.017).sin()).collect();
        let x = Tensor::from_vec(xdata, (1, 32, 24, 128), &dev).unwrap();
        let y = rope.apply(&x).unwrap();
        assert_eq!(y.dims(), x.dims());
        let norm = |t: &Tensor| -> f32 {
            t.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()
        };
        let (xnorm, ynorm) = (norm(&x), norm(&y));
        assert!((xnorm - ynorm).abs() / xnorm < 1e-4, "norm changed: {xnorm} vs {ynorm}");
    }
}
