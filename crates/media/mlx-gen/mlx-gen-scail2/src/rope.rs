//! SCAIL-2 per-chunk 3-axis RoPE.
//!
//! SCAIL-2 packs `[additional_ref | ref | video | pose]` into one self-attention sequence and applies
//! a *different* rotary position to each chunk via **per-axis integer position shifts** (and, for the
//! pose chunk, a 2× spatial **frequency downsample**). This is the `rope_apply_scail` family from
//! upstream `wan/modules/model_scail2.py`:
//!
//! - `rope_apply_ref` / `rope_apply_additional_ref` — the reference frame(s); the `replace_flag`
//!   toggles their **H-shift** between 0 (animation) and 120 (cross-identity replacement).
//! - `rope_apply_video` — the denoised video tokens; T-shifted past the reference frame.
//! - `rope_apply_pose` — the driving-pose tokens; **W-shifted by 120** and **avg-pooled 2× over
//!   (H,W)** because the pose latent is half-resolution.
//!
//! The 3-axis factorization (temporal / height / width frequency lanes) and the inverse-frequency
//! formula are byte-identical to [`mlx_gen_wan::rope::RopeTable`] (which mirrors upstream `rope_params`
//! exactly: head_dim 128 → 22 temporal + 21 height + 21 width = 64 half-lanes, θ = 10000). We build our
//! own tables here because [`mlx_gen_wan::rope::RopeTable`] exposes neither per-axis shifts nor the
//! emitted `(cos, sin)` are shaped `[seq, 1, half_d]`, directly consumable by
//! [`mlx_gen_wan::rope::rope_apply`].

use mlx_gen::Result;
use mlx_rs::Array;

/// RoPE base, matching upstream `rope_params(theta=10000)` and [`mlx_gen_wan`].
const ROPE_THETA: f64 = 10000.0;

/// Per-axis rotary-frequency tables for the SCAIL-2 DiT. Holds only the inverse frequencies; the
/// cos/sin for any (shifted) grid is materialized on demand by [`ScailRope::chunk`].
pub struct ScailRope {
    /// Half the head dimension (= number of complex rotary lanes). 64 for head_dim 128.
    pub half_d: usize,
    /// Temporal-axis lane count (22 for head_dim 128).
    pub temporal_half: usize,
    /// Spatial-axis (height = width) lane count (21 for head_dim 128).
    pub axis_half: usize,
    /// Inverse frequencies for the temporal axis, length `temporal_half`.
    inv_t: Vec<f64>,
    /// Inverse frequencies for the height/width axes, length `axis_half`.
    inv_a: Vec<f64>,
}

impl ScailRope {
    /// Build the tables for a given attention `head_dim` (128 for SCAIL-2's Wan2.1-14B).
    pub fn new(head_dim: usize) -> Self {
        let d6 = head_dim / 6;
        let temporal_dim = head_dim - 4 * d6; // 44
        let axis_dim = 2 * d6; // 42
        let temporal_half = temporal_dim / 2; // 22
        let axis_half = axis_dim / 2; // 21
        let half_d = temporal_half + 2 * axis_half; // 64
        let inv = |dim: usize, n: usize| -> Vec<f64> {
            (0..n)
                .map(|j| ROPE_THETA.powf(-((2 * j) as f64) / dim as f64))
                .collect()
        };
        Self {
            half_d,
            temporal_half,
            axis_half,
            inv_t: inv(temporal_dim, temporal_half),
            inv_a: inv(axis_dim, axis_half),
        }
    }

    /// Write the `half_d` cos/sin lanes for the absolute grid position `(pt, ph, pw)` into the slices.
    /// Lane layout is `[temporal | height | width]`, matching the upstream `freqs.split` order.
    fn fill(&self, cos: &mut [f32], sin: &mut [f32], pt: usize, ph: usize, pw: usize) {
        let t0 = self.temporal_half;
        let t1 = t0 + self.axis_half;
        for (j, &inv) in self.inv_t.iter().enumerate() {
            let a = pt as f64 * inv;
            cos[j] = a.cos() as f32;
            sin[j] = a.sin() as f32;
        }
        for (j, &inv) in self.inv_a.iter().enumerate() {
            let a = ph as f64 * inv;
            cos[t0 + j] = a.cos() as f32;
            sin[t0 + j] = a.sin() as f32;
        }
        for (j, &inv) in self.inv_a.iter().enumerate() {
            let a = pw as f64 * inv;
            cos[t1 + j] = a.cos() as f32;
            sin[t1 + j] = a.sin() as f32;
        }
    }

    /// Per-chunk rotary `(cos, sin)`, each `[seq, 1, half_d]`, for a `grid = (f, h, w)` patch grid with
    /// per-axis integer position shifts `shift = (shift_t, shift_h, shift_w)`.
    ///
    /// When `pose_downsample` is set, the freqs are built at the full `(f, h, w)` grid and then
    /// **avg-pooled 2× over the spatial axes** to `(f, h/2, w/2)` — the exact (non-unit-modulus)
    /// `avg_pool2d` over the complex exponentials that upstream `rope_apply_pose` performs. `h` and `w`
    /// must be even in that case.
    pub fn chunk(
        &self,
        grid: (usize, usize, usize),
        shift: (usize, usize, usize),
        pose_downsample: bool,
    ) -> Result<(Array, Array)> {
        let (f, h, w) = grid;
        let (shift_t, shift_h, shift_w) = shift;
        let hd = self.half_d;
        if !pose_downsample {
            let seq = f * h * w;
            let mut cos = vec![0f32; seq * hd];
            let mut sin = vec![0f32; seq * hd];
            let mut p = 0usize;
            for ti in 0..f {
                for hi in 0..h {
                    for wi in 0..w {
                        let d = p * hd;
                        let (c, s) = (&mut cos[d..d + hd], &mut sin[d..d + hd]);
                        self.fill(c, s, ti + shift_t, hi + shift_h, wi + shift_w);
                        p += 1;
                    }
                }
            }
            let shape = [seq as i32, 1, hd as i32];
            Ok((
                Array::from_slice(&cos, &shape),
                Array::from_slice(&sin, &shape),
            ))
        } else {
            let (ho, wo) = (h / 2, w / 2);
            let seq = f * ho * wo;
            let mut cos = vec![0f32; seq * hd];
            let mut sin = vec![0f32; seq * hd];
            // Scratch for one full-resolution cell before pooling.
            let mut fc = vec![0f32; hd];
            let mut fs = vec![0f32; hd];
            let mut p = 0usize;
            for ti in 0..f {
                for ho_i in 0..ho {
                    for wo_i in 0..wo {
                        let d = p * hd;
                        // Average the 2×2 spatial block of full-res complex freqs (real & imag
                        // independently), matching avg_pool2d(kernel=2, stride=2).
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let hi = ho_i * 2 + dh;
                                let wi = wo_i * 2 + dw;
                                self.fill(
                                    &mut fc,
                                    &mut fs,
                                    ti + shift_t,
                                    hi + shift_h,
                                    wi + shift_w,
                                );
                                for k in 0..hd {
                                    cos[d + k] += fc[k];
                                    sin[d + k] += fs[k];
                                }
                            }
                        }
                        for k in 0..hd {
                            cos[d + k] *= 0.25;
                            sin[d + k] *= 0.25;
                        }
                        p += 1;
                    }
                }
            }
            let shape = [seq as i32, 1, hd as i32];
            Ok((
                Array::from_slice(&cos, &shape),
                Array::from_slice(&sin, &shape),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_split_matches_wan() {
        // head_dim 128 → 22 temporal + 21 height + 21 width = 64 half-lanes (the Wan2.1-14B head_dim).
        let r = ScailRope::new(128);
        assert_eq!(r.temporal_half, 22);
        assert_eq!(r.axis_half, 21);
        assert_eq!(r.half_d, 64);
        assert_eq!(r.inv_t.len(), 22);
        assert_eq!(r.inv_a.len(), 21);
        // Position 0 on every axis → cos 1, sin 0 across all lanes.
        let (cos, sin) = r.chunk((1, 1, 1), (0, 0, 0), false).unwrap();
        assert_eq!(cos.shape(), &[1, 1, 64]);
        let c: Vec<f32> = cos.as_slice().to_vec();
        let s: Vec<f32> = sin.as_slice().to_vec();
        assert!(c.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(s.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn pose_downsample_shapes_and_t_invariance() {
        let r = ScailRope::new(128);
        // Full grid (f=2, h=4, w=4) → pooled to (2, 2, 2) → seq 8.
        let (cos, sin) = r.chunk((2, 4, 4), (1, 0, 120), true).unwrap();
        assert_eq!(cos.shape(), &[8, 1, 64]);
        assert_eq!(sin.shape(), &[8, 1, 64]);
        // The temporal lanes are constant across a pooled spatial block, so pooling leaves them
        // exactly equal to the un-pooled temporal freqs at the shifted frame index.
        let (cos_t, _) = r.chunk((2, 1, 1), (1, 0, 0), false).unwrap();
        let pooled: Vec<f32> = cos.as_slice().to_vec();
        let plain: Vec<f32> = cos_t.as_slice().to_vec();
        // Frame 0, lane 1 (a temporal lane) must match between pooled pose and plain temporal.
        assert!((pooled[1] - plain[1]).abs() < 1e-5);
    }
}
