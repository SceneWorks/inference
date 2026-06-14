//! Chroma's **FluxPosEmbed** 3-axis RoPE over `(t, h, w)` position ids — the candle port of
//! `mlx-gen-chroma`'s `build_rope` / `apply_rope_one`.
//!
//! `axes_dims_rope = [16, 56, 56]` (summing to `head_dim = 128`); each axis contributes `dim/2`
//! frequencies, concatenated into a `head_dim/2 = 64`-wide `(cos, sin)` table. Frequencies
//! `ω[j] = theta^{-(2j)/dim}` (θ = 10000) are computed host-side in f32, then the table is
//! materialized on-device.
//!
//! Application is **interleaved** (adjacent-pair): lanes `2i` / `2i+1` of each head are the real /
//! imaginary parts rotated by `(cos[i], sin[i])` — `out0 = r·cos − i·sin`, `out1 = i·cos + r·cos`.
//! candle's [`candle_nn::rotary_emb::rope_i`] implements exactly this, so it is reused for the apply
//! (the same reuse `candle-gen-flux2`'s `pos_embed` makes). The text tokens sit at the RoPE origin
//! `(0,0,0)`; the packed image tokens carry `(0, row, col)` over the `(height/16, width/16)` grid.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

use crate::config::ChromaTransformerConfig;

const ROPE_THETA: f32 = 10000.0;

/// The precomputed `(cos, sin)` RoPE tables `[seq, head_dim/2]` for one ordered position-id list.
pub struct RopeTable {
    pub cos: Tensor,
    pub sin: Tensor,
}

/// Build the `(cos, sin)` tables for the ordered `[t, h, w]` position `ids` (text ids first, then
/// image ids — the same `cat(txt_ids, img_ids)` order the DiT attends). Host-side f32, then moved to
/// `device`.
pub fn build_rope(ids: &[[i64; 3]], axes: [usize; 3], device: &Device) -> Result<RopeTable> {
    let half: usize = axes.iter().map(|d| d / 2).sum();
    let seq = ids.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (s, id) in ids.iter().enumerate() {
        let mut col = 0usize;
        for (axis, &dim) in axes.iter().enumerate() {
            let pos = id[axis] as f32;
            let half_axis = dim / 2;
            for j in 0..half_axis {
                let omega = 1.0f32 / ROPE_THETA.powf((2 * j) as f32 / dim as f32);
                let angle = pos * omega;
                cos[s * half + col] = angle.cos();
                sin[s * half + col] = angle.sin();
                col += 1;
            }
        }
    }
    Ok(RopeTable {
        cos: Tensor::from_vec(cos, (seq, half), device)?,
        sin: Tensor::from_vec(sin, (seq, half), device)?,
    })
}

impl RopeTable {
    /// Apply the interleaved RoPE to `x` `[B, H, S, head_dim]` (the rotation runs in f32, casting back
    /// to `x`'s dtype). `cos`/`sin` are `[S, head_dim/2]`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?.contiguous()?;
        rope_i(&xf, &self.cos, &self.sin)?.to_dtype(dtype)
    }
}

/// FluxPosEmbed image position ids for the packed `(height/16, width/16)` grid: `(0, row, col)`,
/// row-major — diffusers `_prepare_latent_image_ids`. Matches the pack order in
/// [`crate::pipeline`] (which mirrors candle FLUX's `State::new`).
pub fn latent_image_ids(h2: usize, w2: usize) -> Vec<[i64; 3]> {
    let mut ids = Vec::with_capacity(h2 * w2);
    for i in 0..h2 {
        for j in 0..w2 {
            ids.push([0, i as i64, j as i64]);
        }
    }
    ids
}

/// Text position ids `[L]` — all `(0,0,0)` (FluxPosEmbed places every text token at the origin).
pub fn zero_text_ids(l: usize) -> Vec<[i64; 3]> {
    vec![[0, 0, 0]; l]
}

/// The full RoPE id list the DiT attends: `cat(txt_ids, img_ids)` (text first).
pub fn joint_ids(text_len: usize, h2: usize, w2: usize) -> Vec<[i64; 3]> {
    let mut ids = zero_text_ids(text_len);
    ids.extend(latent_image_ids(h2, w2));
    ids
}

/// Build the RoPE table for a `text_len`-token prompt over a `(h2, w2)` packed image grid, against
/// `cfg.axes_dims_rope`.
pub fn build_for(
    cfg: &ChromaTransformerConfig,
    text_len: usize,
    h2: usize,
    w2: usize,
    device: &Device,
) -> Result<RopeTable> {
    build_rope(&joint_ids(text_len, h2, w2), cfg.axes_dims_rope, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_width_is_head_dim_half() {
        let cfg = ChromaTransformerConfig::default();
        let t = build_for(&cfg, 2, 2, 2, &Device::Cpu).unwrap();
        // 16/2 + 56/2 + 56/2 = 8 + 28 + 28 = 64 = head_dim/2.
        assert_eq!(t.cos.dims(), &[2 + 4, 64]);
        assert_eq!(t.sin.dims(), &[2 + 4, 64]);
    }

    #[test]
    fn origin_position_is_identity() {
        // Text tokens (and image token (0,0,0)) sit at the origin → cos 1, sin 0 everywhere.
        let t = build_rope(&[[0, 0, 0]], [16, 56, 56], &Device::Cpu).unwrap();
        let c = t.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = t.sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &c {
            assert!((v - 1.0).abs() < 1e-6);
        }
        for v in &s {
            assert!(v.abs() < 1e-6);
        }
        // rope_i at the origin is the identity.
        let x = Tensor::arange(0f32, 128f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, 128))
            .unwrap();
        let y = t.apply(&x).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn image_ids_are_row_major() {
        let ids = latent_image_ids(2, 3);
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[0], [0, 0, 0]);
        assert_eq!(ids[1], [0, 0, 1]);
        assert_eq!(ids[3], [0, 1, 0]);
        assert_eq!(ids[5], [0, 1, 2]);
    }

    #[test]
    fn first_lane_of_each_axis_uses_base_frequency() {
        // For axis position p and j=0, ω=1, so the first lane of each axis block reflects p directly.
        // Cols: t at 0, h at 8, w at 36.
        let t = build_rope(&[[1, 2, 3]], [16, 56, 56], &Device::Cpu).unwrap();
        let c = t.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((c[0] - 1.0f32.cos()).abs() < 1e-5, "t axis @ pos 1");
        assert!((c[8] - 2.0f32.cos()).abs() < 1e-5, "h axis @ pos 2");
        assert!((c[36] - 3.0f32.cos()).abs() < 1e-5, "w axis @ pos 3");
    }
}
