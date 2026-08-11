//! 3-axis **MM-RoPE** over the packed sequence's `(t, h, w)` coordinates
//! (`MiniMaxH3RotaryPosEmbed`).
//!
//! This is a *different* rotary from [`crate::rope`], which the video VAE decoder uses. Sharing a
//! name and a "3-D rope" description, they disagree on three things, and each disagreement is a
//! silent whole-model error:
//!
//! | | video VAE ([`crate::rope`]) | **DiT (here)** |
//! |---|---|---|
//! | positions | length-normalized cell centres in `(-1, 1)` | raw coordinates, unnormalized |
//! | angle | `2π · pos · inv_freq` (`use_angle = True`) | `pos · inv_freq` — **no `2π`** |
//! | frequency count | `apply_dim / 6` derived from a rotated-dim *ratio* | `rope_freq_dim`, declared; the rotated width is *derived* as `2 · 3 · rope_freq_dim` |
//!
//! # The axis split
//!
//! One `inv_freq` buffer of `rope_freq_dim` entries is shared by all three axes:
//! `inv_freq[i] = rope_theta^(-2i / (2 · rope_freq_dim))`, i.e. `rope_theta^(-i / rope_freq_dim)`.
//!
//! Each axis contributes its own `rope_freq_dim` angles and the three blocks are concatenated **t,
//! then h, then w**, giving `3 · rope_freq_dim`; that block is then concatenated with *itself* so
//! the `rotate_half` convention covers `2 · 3 · rope_freq_dim` channels:
//!
//! ```text
//! channel:  0 ────── F ────── 2F ────── 3F ────── 4F ────── 5F ────── 6F      (F = rope_freq_dim)
//!          │    t    │    h    │    w    │    t    │    h    │    w    │
//!          └──────── first half ─────────┴──────── second half ────────┘
//!                 (x1 of rotate_half)          (x2 of rotate_half)
//! ```
//!
//! At the shipped `rope_freq_dim = 16` that is 96 of the 128 head channels; the trailing **32 pass
//! through unrotated**. Rotating all 128 — or splitting the axes h/w/t, or interleaving them — is
//! shape-identical and a different model.
//!
//! # Audio tokens have no `h`
//!
//! The per-modality coordinate conventions live in [`super::positions`]; the rotary itself is
//! modality-blind and simply consumes `[seq_len, 3]`.

use mlx_rs::ops::{concatenate_axis, cos, multiply, negative, sin};
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

use crate::tensor::slice_axis;

/// Precomputed `(cos, sin)` for one packed sequence, `[seq_len, rotary_dim]` each.
///
/// One row per row of the packed sequence — not per token grid position — because the sequence
/// interleaves three modalities with unrelated coordinate schemes.
#[derive(Debug, Clone)]
pub struct MmRopeTables {
    /// `[seq_len, rotary_dim]`.
    pub cos: Array,
    /// `[seq_len, rotary_dim]`.
    pub sin: Array,
}

/// The DiT's 3-axis rotary embedding.
#[derive(Debug, Clone)]
pub struct MmRope {
    inv_freq: Vec<f32>,
}

impl MmRope {
    /// Build the shared `inv_freq` buffer.
    ///
    /// `freq_dim` is `rope_freq_dim` — the frequency count **per axis**, not the rotated width.
    pub fn new(freq_dim: i32, theta: f32) -> Result<Self> {
        if freq_dim <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 mm-rope: rope_freq_dim must be positive, got {freq_dim}"
            )));
        }
        // `1 / theta ** (arange(0, 2·F, 2) / (2·F))` == `theta ** (-i / F)`.
        let inv_freq = (0..freq_dim)
            .map(|i| theta.powf(-(i as f32) / freq_dim as f32))
            .collect();
        Ok(Self { inv_freq })
    }

    /// Frequencies per axis (`rope_freq_dim`).
    pub fn freq_dim(&self) -> i32 {
        self.inv_freq.len() as i32
    }

    /// Rotated head channels: `2 · 3 · rope_freq_dim`.
    pub fn rotary_dim(&self) -> i32 {
        6 * self.freq_dim()
    }

    /// Build `(cos, sin)` for `position_ids` of shape `[seq_len, 3]`.
    ///
    /// The reference casts the positions to float32 before multiplying (they are built in float64,
    /// which MPS has no dtype for), so the angles are always computed in f32 regardless of the
    /// block stack's precision; `dtype` is what the tables are cast to on the way out.
    pub fn tables(&self, position_ids: &Array, dtype: Dtype) -> Result<MmRopeTables> {
        let shape = position_ids.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 mm-rope: expected position ids as [seq_len, 3], got {shape:?}"
            )));
        }
        let seq = shape[0];
        let f = self.freq_dim();

        // freqs[s, axis, i] = pos[s, axis] · inv_freq[i]
        let inv = Array::from_slice(&self.inv_freq, &[1, 1, f]);
        let pos = position_ids.as_dtype(Dtype::Float32)?.reshape(&[seq, 3, 1])?;
        // `unbind(1)` then `cat(t, h, w)` is exactly a `[seq, 3·F]` row-major flatten of axis 1,
        // because the axes are already stored in `(t, h, w)` order.
        let freqs = multiply(&pos, &inv)?.reshape(&[seq, 3 * f])?;
        let freqs = concatenate_axis(&[freqs.clone(), freqs], -1)?;

        Ok(MmRopeTables {
            cos: cos(&freqs)?.as_dtype(dtype)?,
            sin: sin(&freqs)?.as_dtype(dtype)?,
        })
    }

    /// Rotate the leading `rotary_dim` channels of every head of `x`, shape `[B, S, H, D]`.
    ///
    /// Channels beyond `rotary_dim` are concatenated back **untouched** — the partial-rotary path
    /// the shipped geometry always takes (96 of 128).
    pub fn apply(&self, x: &Array, tables: &MmRopeTables) -> Result<Array> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "minimax-h3 mm-rope: expected [B, S, H, D], got {shape:?}"
            )));
        }
        let (seq, d) = (shape[1], shape[3]);
        let rotary = tables.cos.shape()[1];
        if rotary > d {
            return Err(Error::Msg(format!(
                "minimax-h3 mm-rope: rotary_dim {rotary} exceeds head dim {d}"
            )));
        }
        if tables.cos.shape()[0] != seq {
            return Err(Error::Msg(format!(
                "minimax-h3 mm-rope: rope tables cover {} rows but the sequence has {seq}",
                tables.cos.shape()[0]
            )));
        }

        // `[seq, rotary]` -> `[1, seq, 1, rotary]`, broadcasting over batch and heads.
        let cos_t = tables.cos.as_dtype(x.dtype())?.reshape(&[1, seq, 1, rotary])?;
        let sin_t = tables.sin.as_dtype(x.dtype())?.reshape(&[1, seq, 1, rotary])?;

        let head = slice_axis(x, 3, 0, rotary)?;
        let half = rotary / 2;
        let x1 = slice_axis(&head, 3, 0, half)?;
        let x2 = slice_axis(&head, 3, half, rotary)?;
        let rotated = concatenate_axis(&[negative(&x2)?, x1], -1)?;
        let out = mlx_rs::ops::add(&multiply(&head, &cos_t)?, &multiply(&rotated, &sin_t)?)?;

        if rotary == d {
            return Ok(out);
        }
        let tail = slice_axis(x, 3, rotary, d)?;
        Ok(concatenate_axis(&[out, tail], -1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.61).sin() * 1.7).collect();
        Array::from_slice(&vals, shape)
    }

    /// `inv_freq[i] = theta^(-i/F)`, and the rotated width is DERIVED as `6·F` rather than
    /// declared. A port that read `rope_freq_dim` as the rotated width would rotate 16 channels
    /// instead of 96.
    #[test]
    fn rotary_width_is_six_times_the_declared_frequency_count() {
        let rope = MmRope::new(16, 10_000.0).unwrap();
        assert_eq!(rope.freq_dim(), 16);
        assert_eq!(rope.rotary_dim(), 96);
        assert!(MmRope::new(0, 10_000.0).is_err());

        // inv_freq[0] = 1, inv_freq[F-1] = theta^(-(F-1)/F).
        let ids = Array::from_slice(&[1.0f32, 0.0, 0.0], &[1, 3]);
        let t = rope.tables(&ids, Dtype::Float32).unwrap();
        let c: Vec<f32> = t.cos.as_slice::<f32>().to_vec();
        assert!((c[0] - 1.0f32.cos()).abs() < 1e-6, "inv_freq[0] must be 1");
        let last = 10_000f32.powf(-15.0 / 16.0);
        assert!((c[15] - last.cos()).abs() < 1e-6, "inv_freq[15] = theta^(-15/16)");
    }

    /// **The axis split.** Channels `[0,F)` are `t`, `[F,2F)` are `h`, `[2F,3F)` are `w`, and
    /// `[3F,6F)` repeats that triple. Driven with a one-hot position per axis so a permuted or
    /// interleaved split cannot pass.
    #[test]
    fn axes_split_t_then_h_then_w_and_the_block_repeats() {
        let f = 4;
        let rope = MmRope::new(f, 100.0).unwrap();
        // Three rows, each with a single non-zero axis: t=1, h=1, w=1.
        let ids = Array::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        );
        let tables = rope.tables(&ids, Dtype::Float32).unwrap();
        assert_eq!(tables.cos.shape(), &[3, 6 * f]);
        let sin_v: Vec<f32> = tables.sin.as_slice::<f32>().to_vec();
        let row = |r: usize| &sin_v[r * (6 * f) as usize..(r + 1) * (6 * f) as usize];

        for (r, axis) in [(0usize, 0usize), (1, 1), (2, 2)] {
            let v = row(r);
            for i in 0..f as usize {
                let want = (100f32.powf(-(i as f32) / f as f32)).sin();
                // The axis's own block is non-trivial...
                let at = axis * f as usize + i;
                assert!(
                    (v[at] - want).abs() < 1e-6,
                    "row {r}: channel {at} should carry axis {axis} frequency {i}"
                );
                // ...and it repeats 3F later.
                assert!(
                    (v[at + 3 * f as usize] - want).abs() < 1e-6,
                    "row {r}: the second half must repeat channel {at}"
                );
                // ...while the other two axes' blocks are zero (sin(0) = 0).
                for other in 0..3usize {
                    if other != axis {
                        let idx = other * f as usize + i;
                        assert!(
                            v[idx].abs() < 1e-6,
                            "row {r}: channel {idx} belongs to axis {other}, which is at position 0"
                        );
                    }
                }
            }
        }
    }

    /// **No `2π`.** The video VAE's rotary multiplies positions by `2π` (`use_angle`); this one
    /// does not, and the two are indistinguishable by shape.
    #[test]
    fn angles_carry_no_two_pi_factor() {
        let rope = MmRope::new(1, 100.0).unwrap();
        let ids = Array::from_slice(&[1.0f32, 0.0, 0.0], &[1, 3]);
        let t = rope.tables(&ids, Dtype::Float32).unwrap();
        let c: Vec<f32> = t.cos.as_slice::<f32>().to_vec();
        assert!(
            (c[0] - 1.0f32.cos()).abs() < 1e-6,
            "angle must be pos·inv_freq = 1 rad, got acos {}",
            c[0].acos()
        );
        assert!(
            (c[0] - (2.0 * std::f32::consts::PI).cos()).abs() > 1e-3,
            "an angle of 2π would mean the video VAE's `use_angle` convention leaked in"
        );
    }

    /// The pass-through tail must survive byte-identical, and the rotated head must not.
    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let rope = MmRope::new(2, 100.0).unwrap(); // rotary_dim 12
        let ids = spread(&[5, 3]);
        let tables = rope.tables(&ids, Dtype::Float32).unwrap();
        let x = spread(&[1, 5, 2, 16]);
        let out = rope.apply(&x, &tables).unwrap();
        assert_eq!(out.shape(), x.shape());

        let tail_in = slice_axis(&x, 3, 12, 16).unwrap();
        let tail_out = slice_axis(&out, 3, 12, 16).unwrap();
        assert_eq!(
            tail_in.as_slice::<f32>(),
            tail_out.as_slice::<f32>(),
            "channels beyond 2·3·rope_freq_dim must pass through unrotated"
        );
        let head_in = slice_axis(&x, 3, 0, 12).unwrap();
        let head_out = slice_axis(&out, 3, 0, 12).unwrap();
        assert_ne!(head_in.as_slice::<f32>(), head_out.as_slice::<f32>());
    }

    /// `rotate_half` is norm-preserving per rotated pair; a wrong half split breaks it.
    #[test]
    fn rotation_preserves_the_rotated_norm() {
        let rope = MmRope::new(2, 100.0).unwrap();
        let ids = spread(&[6, 3]);
        let tables = rope.tables(&ids, Dtype::Float32).unwrap();
        let x = spread(&[1, 6, 1, 12]);
        let out = rope.apply(&x, &tables).unwrap();
        let a: f32 = x.square().unwrap().sum(None).unwrap().item();
        let b: f32 = out.square().unwrap().sum(None).unwrap().item();
        assert!((a - b).abs() / a < 1e-4, "rotary must preserve norm: {a} vs {b}");
    }

    /// A tables/sequence length mismatch is an error, not a broadcast that silently reuses row 0.
    #[test]
    fn a_length_mismatch_is_rejected() {
        let rope = MmRope::new(2, 100.0).unwrap();
        let tables = rope.tables(&spread(&[4, 3]), Dtype::Float32).unwrap();
        assert!(rope.apply(&spread(&[1, 7, 1, 12]), &tables).is_err());
        assert!(rope.apply(&spread(&[1, 4, 1, 8]), &tables).is_err());
        assert!(rope.tables(&spread(&[4, 2]), Dtype::Float32).is_err());
    }
}
