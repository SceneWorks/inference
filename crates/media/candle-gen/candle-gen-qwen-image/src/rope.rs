//! Qwen-Image's **3-axis (frame, height, width) interleaved RoPE** for the MMDiT. Port of
//! `mlx-gen-qwen-image`'s `transformer/rope.rs`. Each axis (`axes_dim = [16, 56, 56]`) contributes
//! `dim/2` frequencies → `8 + 28 + 28 = 64` per token (= `head_dim/2`). θ = 10000, `scale_rope` (the
//! height/width positions are centered). Frequencies and angles are computed **host-side** in f32.
//!
//! - **Image tokens** at grid `(h, w)`: the frame axis uses position 0 (single image), height/width
//!   use **centered** positions `h - (latent_h - latent_h/2)` / `w - (latent_w - latent_w/2)`.
//! - **Text tokens** at index `t`: a single scalar position `txt_base + t`
//!   (`txt_base = max(latent_h/2, latent_w/2)`) applied across **all 64** frequencies.
//!
//! Application is **interleaved** (lanes `2i`/`2i+1` are the real/imag pair), via candle's `rope_i`.

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::rope::{AxialRope3, GeometryCache};

use crate::config::TransformerConfig;

/// The Qwen-Image 3-axis RoPE: the shared [`AxialRope3`] at this config's `rope_theta` / `axes_dim`
/// (`(16, 56, 56)`, Σ/2 = 64 pairs). The builder is shared with Lens (sc-27286).
pub struct QwenRope(AxialRope3);

impl QwenRope {
    pub fn new(cfg: &TransformerConfig) -> Self {
        Self(AxialRope3::new(cfg.rope_theta, cfg.axes_dim))
    }

    /// Image-token `(cos, sin)` `[lat_h·lat_w, 64]` (row-major over the grid). The single-grid (T2I)
    /// case of [`img_cos_sin_multi`](Self::img_cos_sin_multi) — grid index 0 → frame position 0.
    pub fn img_cos_sin(
        &self,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.img_cos_sin_multi(&[(lat_h, lat_w)], device)
    }

    /// Image-token `(cos, sin)` over one-or-more grids (the Qwen-Image-Edit dual-latent path): the
    /// noise grid (index 0) followed by each reference grid (index 1, 2, …) in sequence order. The
    /// grid **index** drives the frame-axis position (so a reference's frame freqs differ from the
    /// noise's), while height/width stay per-grid **centered**. Concatenated `[Σ h_i·w_i, 64]`.
    pub fn img_cos_sin_multi(
        &self,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.0.img_cos_sin_multi(grids, device)
    }

    /// Text-token `(cos, sin)` `[txt_seq, 64]`: scalar position `txt_base + t` across all freqs. The
    /// single-grid case of [`txt_cos_sin_multi`](Self::txt_cos_sin_multi).
    pub fn txt_cos_sin(
        &self,
        txt_seq: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.txt_cos_sin_multi(txt_seq, &[(lat_h, lat_w)], device)
    }

    /// Text-token `(cos, sin)` for the dual-latent path: `txt_base = max_i(max(h_i/2, w_i/2))` over
    /// every grid, then position `txt_base + t` across all freqs.
    pub fn txt_cos_sin_multi(
        &self,
        txt_seq: usize,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.0.txt_cos_sin_multi(txt_seq, grids, device)
    }
}

/// Per-render RoPE-table cache (sc-8992). The image + text `(cos, sin)` tables depend only on
/// the fixed token grids (`[(lat_h, lat_w), …]`) and `txt_seq` — not on σ / the current latent — so
/// they are identical across every denoise step (×2 under CFG, and the control branch re-derives the
/// same grid each step). Cache them keyed on `(grids, txt_seq)` and rebuild only when the geometry
/// changes; hits Arc-clone the stored handles. Byte-identical to recomputing.
///
/// Single-slot ([`GeometryCache`] at capacity 1): the image + text tables, keyed on `(grids, txt_seq)`.
pub struct RopeCache(GeometryCache<(Vec<(usize, usize)>, usize), RopeTables>);

/// `(img_cos, img_sin, txt_cos, txt_sin)`.
type RopeTables = (Tensor, Tensor, Tensor, Tensor);

impl RopeCache {
    pub fn new() -> Self {
        Self(GeometryCache::new(1))
    }

    /// Build (or reuse) the image + text RoPE tables for `grids` / `txt_seq`. Recomputed only when the
    /// geometry changes vs the cached entry. Construction is identical to calling the `QwenRope`
    /// builders inline, so every step is byte-identical.
    pub fn tables(
        &self,
        rope: &QwenRope,
        grids: &[(usize, usize)],
        txt_seq: usize,
        device: &Device,
    ) -> Result<RopeTables> {
        self.0.get_or_build_by(
            |(g, t)| g.as_slice() == grids && *t == txt_seq,
            || (grids.to_vec(), txt_seq),
            || {
                let (img_cos, img_sin) = rope.img_cos_sin_multi(grids, device)?;
                let (txt_cos, txt_sin) = rope.txt_cos_sin_multi(txt_seq, grids, device)?;
                Ok((img_cos, img_sin, txt_cos, txt_sin))
            },
        )
    }
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`.
pub use candle_gen::rope::apply_interleaved_rope as apply_rope;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::DType;

    #[test]
    fn img_frame_axis_is_zero_angle() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        let (cos, sin) = r.img_cos_sin(4, 4, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[16, 64]);
        // Frame band (first 8 lanes) has position 0 → cos 1, sin 0 for every token.
        let cv = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sv = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..16 {
            for j in 0..8 {
                assert!((cv[tok * 64 + j] - 1.0).abs() < 1e-6);
                assert!(sv[tok * 64 + j].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn apply_rope_at_zero_is_identity() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        // txt position 0 with lat 2x2 → txt_base = max(1,1)=1, so not zero; build an explicit zero table.
        let cos = Tensor::ones((3, 64), DType::F32, &Device::Cpu).unwrap();
        let sin = Tensor::zeros((3, 64), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::arange(0f32, 3.0 * 128.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3, 128))
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
