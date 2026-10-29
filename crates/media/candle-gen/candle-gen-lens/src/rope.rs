//! Lens complex axial RoPE (`LensEmbedRope`) — the DiT positional embedding (sc-5112).
//!
//! A 3-axis (frame / height / width) interleaved RoPE, architecturally a twin of
//! `candle-gen-qwen-image`'s `crate`-sibling `QwenRope`: only the axis widths differ
//! (`axes_dims_rope = (8, 28, 28)`, Σ = 64 = `head_dim`, Σ/2 = 32 complex pairs vs Qwen's 16/56/56).
//! θ = 10000, `scale_rope = True` (height/width positions are centered). Frequencies and angles are
//! computed **host-side** in f32; application is **interleaved** (lanes `2i`/`2i+1` are the real/imag
//! pair) via candle's `rope_i`, reproducing the reference's complex `view_as_complex · freqs_cis`.
//!
//! - **Image tokens** at grid `(f, h, w)`: the frame axis uses position `f`, height/width use
//!   **centered** positions `h − (lat_h − lat_h/2)` / `w − (lat_w − lat_w/2)`.
//! - **Text tokens** at index `t`: a single scalar position `txt_base + t`
//!   (`txt_base = max(lat_h/2, lat_w/2)`) applied across **all 32** pair-frequencies.

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::rope::AxialRope3;

/// Lens 3-axis RoPE table builder. `θ = 10000`, `axes_dim = (8, 28, 28)`: the shared [`AxialRope3`]
/// (the builder Qwen-Image uses with its own axes, sc-27286).
pub struct LensRope(AxialRope3);

impl LensRope {
    pub fn new(theta: f32, axes_dim: [usize; 3]) -> Self {
        Self(AxialRope3::new(theta, axes_dim))
    }

    /// The Lens default: θ=10000, axes `(8, 28, 28)` (Σ = 64 = head_dim, Σ/2 = 32 pairs).
    pub fn lens() -> Self {
        Self::new(10_000.0, [8, 28, 28])
    }

    /// Image-token `(cos, sin)` `[frame·lat_h·lat_w, 32]` (row-major over the grid, frame-major): frame
    /// `f` takes frame position `f` — the shared multi-grid builder over `frame` copies of the grid.
    pub fn img_cos_sin(
        &self,
        frame: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.0
            .img_cos_sin_multi(&vec![(lat_h, lat_w); frame], device)
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
        self.0.txt_cos_sin_multi(txt_seq, &[(lat_h, lat_w)], device)
    }
}

/// Apply interleaved complex RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`.
/// The rotation is computed in f32 and cast back to `x`'s dtype.
pub use candle_gen::rope::apply_interleaved_rope as apply_rope;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::DType;

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
