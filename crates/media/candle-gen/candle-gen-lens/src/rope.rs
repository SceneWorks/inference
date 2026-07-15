//! Lens complex axial RoPE (`LensEmbedRope`) — the DiT positional embedding (sc-5112).
//!
//! A 3-axis (frame / height / width) interleaved RoPE, architecturally a twin of
//! `candle-gen-qwen-image`'s `crate`-sibling `QwenRope`: only the axis widths differ
//! (`axes_dims_rope = (8, 28, 28)`, Σ = 64 = `head_dim`, Σ/2 = 32 complex pairs vs Qwen's 16/56/56).
//! θ = 10000, `scale_rope = True` (height/width positions are centered). Frequencies and angles are
//! computed **host-side** in f32; application is **interleaved** (lanes `2i`/`2i+1` are the real/imag
//! pair) via candle's [`rope_i`], reproducing the reference's complex `view_as_complex · freqs_cis`.
//!
//! - **Image tokens** at grid `(f, h, w)`: the frame axis uses position `f`, height/width use
//!   **centered** positions `h − (lat_h − lat_h/2)` / `w − (lat_w − lat_w/2)`.
//! - **Text tokens** at index `t`: a single scalar position `txt_base + t`
//!   (`txt_base = max(lat_h/2, lat_w/2)`) applied across **all 32** pair-frequencies.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

/// Lens 3-axis RoPE table builder. `θ = 10000`, `axes_dim = (8, 28, 28)`.
pub struct LensRope {
    theta: f32,
    axes_dim: [usize; 3],
    half: usize,
}

impl LensRope {
    pub fn new(theta: f32, axes_dim: [usize; 3]) -> Self {
        Self {
            theta,
            axes_dim,
            half: axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The Lens default: θ=10000, axes `(8, 28, 28)` (Σ = 64 = head_dim, Σ/2 = 32 pairs).
    pub fn lens() -> Self {
        Self::new(10_000.0, [8, 28, 28])
    }

    /// The 32-wide concatenated frequency vector `[ω_frame(4), ω_h(14), ω_w(14)]`,
    /// `ω_d[k] = θ^{-(2k)/d}`.
    fn omega(&self) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.half);
        for &dim in &self.axes_dim {
            for k in 0..dim / 2 {
                all.push(1.0f32 / self.theta.powf((2 * k) as f32 / dim as f32));
            }
        }
        all
    }

    /// Image-token `(cos, sin)` `[frame·lat_h·lat_w, 32]` (row-major over the grid).
    pub fn img_cos_sin(
        &self,
        frame: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let (n_f, n_h) = (self.axes_dim[0] / 2, self.axes_dim[1] / 2); // 4, 14
        let h_center = (lat_h - lat_h / 2) as i64;
        let w_center = (lat_w - lat_w / 2) as i64;
        let seq = frame * lat_h * lat_w;
        let mut cos = vec![0f32; seq * self.half];
        let mut sin = vec![0f32; seq * self.half];
        for f in 0..frame {
            for h in 0..lat_h {
                let hp = h as i64 - h_center;
                for w in 0..lat_w {
                    let wp = w as i64 - w_center;
                    let row = (f * lat_h * lat_w + h * lat_w + w) * self.half;
                    for (j, &om) in omega.iter().enumerate() {
                        // axis by frequency band: frame [0,n_f) → f, height [n_f, n_f+n_h) → hp,
                        // width [n_f+n_h, half) → wp.
                        let pos = if j < n_f {
                            f as i64
                        } else if j < n_f + n_h {
                            hp
                        } else {
                            wp
                        } as f32;
                        let a = pos * om;
                        cos[row + j] = a.cos();
                        sin[row + j] = a.sin();
                    }
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (seq, self.half), device)?,
            Tensor::from_vec(sin, (seq, self.half), device)?,
        ))
    }

    /// Text-token `(cos, sin)` `[txt_seq, 32]`: scalar position `max(lat_h/2, lat_w/2) + t` across
    /// all 32 pair-frequencies.
    pub fn txt_cos_sin(
        &self,
        txt_seq: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let txt_base = (lat_h / 2).max(lat_w / 2) as i64;
        let mut cos = vec![0f32; txt_seq * self.half];
        let mut sin = vec![0f32; txt_seq * self.half];
        for t in 0..txt_seq {
            let pos = (txt_base + t as i64) as f32;
            for (j, &om) in omega.iter().enumerate() {
                let a = pos * om;
                cos[t * self.half + j] = a.cos();
                sin[t * self.half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec(cos, (txt_seq, self.half), device)?,
            Tensor::from_vec(sin, (txt_seq, self.half), device)?,
        ))
    }
}

/// Apply interleaved complex RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`.
/// The rotation is computed in f32 and cast back to `x`'s dtype.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_is_32_wide_with_band_layout() {
        let r = LensRope::lens();
        assert_eq!(r.half, 32);
        let om = r.omega();
        assert_eq!(om.len(), 32);
        // First freq of each band is θ^0 = 1: frame band [0], height band [4], width band [18].
        assert!((om[0] - 1.0).abs() < 1e-6, "frame band base");
        assert!((om[4] - 1.0).abs() < 1e-6, "height band base");
        assert!((om[18] - 1.0).abs() < 1e-6, "width band base");
    }

    #[test]
    fn img_frame_axis_is_zero_angle_for_single_frame() {
        let r = LensRope::lens();
        let (cos, sin) = r.img_cos_sin(1, 4, 4, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[16, 32]);
        // Single frame → frame position 0 → frame band (first 4 lanes) is cos 1 / sin 0 everywhere.
        let cv = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sv = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..16 {
            for j in 0..4 {
                assert!((cv[tok * 32 + j] - 1.0).abs() < 1e-6);
                assert!(sv[tok * 32 + j].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn apply_rope_at_zero_is_identity() {
        let r = LensRope::lens();
        let cos = Tensor::ones((3, 32), DType::F32, &Device::Cpu).unwrap();
        let sin = Tensor::zeros((3, 32), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::arange(0f32, 3.0 * 64.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3, 64))
            .unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
        let _ = &r;
    }
}
