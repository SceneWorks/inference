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
//!
//! # This lane builds the tables on the host, in f64
//!
//! The MLX sibling multiplies `[seq, 3, 1]` positions by a `[1, 1, F]` `inv_freq` array on device,
//! at f32. Here the angles are accumulated on the host in **f64** and materialized once, the way
//! [`crate::rope`] already does for the video VAE — so the two lanes' tables agree to f32 round-off
//! rather than to whichever transcendental each accelerator ships, and the cross-backend residual
//! on `rope_cos` is attributable to nothing but the final narrowing. That is a deliberate
//! difference, not an accident: `tests/cross_backend.rs` reports the rope tables' residual
//! separately from the block's for exactly this reason.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

/// Precomputed `(cos, sin)` for one packed sequence, `[seq_len, rotary_dim]` each.
///
/// One row per row of the packed sequence — not per token grid position — because the sequence
/// interleaves three modalities with unrelated coordinate schemes.
#[derive(Debug, Clone)]
pub struct MmRopeTables {
    /// `[seq_len, rotary_dim]`.
    pub cos: Tensor,
    /// `[seq_len, rotary_dim]`.
    pub sin: Tensor,
}

impl MmRopeTables {
    /// Rows the tables cover.
    pub fn seq_len(&self) -> usize {
        self.cos.dims()[0]
    }

    /// Rotated head channels the tables span.
    pub fn rotary_dim(&self) -> usize {
        self.cos.dims()[1]
    }
}

/// The DiT's 3-axis rotary embedding.
#[derive(Debug, Clone)]
pub struct MmRope {
    inv_freq: Vec<f64>,
}

impl MmRope {
    /// Build the shared `inv_freq` buffer.
    ///
    /// `freq_dim` is `rope_freq_dim` — the frequency count **per axis**, not the rotated width.
    pub fn new(freq_dim: usize, theta: f64) -> Result<Self> {
        if freq_dim == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 mm-rope: rope_freq_dim must be positive, got 0".into(),
            ));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 mm-rope: rope_theta must be finite and positive, got {theta}"
            )));
        }
        // `1 / theta ** (arange(0, 2·F, 2) / (2·F))` == `theta ** (-i / F)`.
        let inv_freq = (0..freq_dim)
            .map(|i| theta.powf(-(i as f64) / freq_dim as f64))
            .collect();
        Ok(Self { inv_freq })
    }

    /// Frequencies per axis (`rope_freq_dim`).
    pub fn freq_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Rotated head channels: `2 · 3 · rope_freq_dim`.
    pub fn rotary_dim(&self) -> usize {
        6 * self.freq_dim()
    }

    /// Build `(cos, sin)` for `position_ids` of shape `[seq_len, 3]`.
    ///
    /// The reference casts the positions to float32 before multiplying (they are built in float64,
    /// which MPS has no dtype for). This reproduces the *values*, computing the angles in f64 on
    /// the host from the f32 coordinates, then materializing at `dtype`.
    pub fn tables(&self, position_ids: &Tensor, dtype: DType) -> Result<MmRopeTables> {
        let shape = position_ids.dims();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 mm-rope: expected position ids as [seq_len, 3], got {shape:?}"
            )));
        }
        let seq = shape[0];
        let f = self.freq_dim();
        let pos: Vec<f32> = position_ids
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let width = 6 * f;
        let mut cos = Vec::with_capacity(seq * width);
        let mut sin = Vec::with_capacity(seq * width);
        for row in 0..seq {
            // `unbind(1)` then `cat(t, h, w)` is exactly a `[seq, 3·F]` row-major flatten of axis 1,
            // because the axes are already stored in `(t, h, w)` order. The `3·F` block is then
            // concatenated with ITSELF, which is what `rotate_half` consumes.
            let mut half = Vec::with_capacity(3 * f);
            for axis in 0..3 {
                let p = f64::from(pos[row * 3 + axis]);
                for iv in &self.inv_freq {
                    half.push(p * iv);
                }
            }
            for _ in 0..2 {
                for &angle in &half {
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }

        let device = position_ids.device();
        Ok(MmRopeTables {
            cos: Tensor::from_vec(cos, (seq, width), device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (seq, width), device)?.to_dtype(dtype)?,
        })
    }

    /// Build the tables straight from host-side `[t, h, w]` rows — the path
    /// [`crate::denoise::PackedLayout`] takes, which never materializes the grid as a tensor first.
    pub fn tables_from_rows(
        &self,
        rows: &[[f64; 3]],
        device: &Device,
        dtype: DType,
    ) -> Result<MmRopeTables> {
        let ids = super::positions::to_tensor(rows, device)?;
        self.tables(&ids, dtype)
    }

    /// Rotate the leading `rotary_dim` channels of every head of `x`, shape `[B, S, H, D]`.
    ///
    /// Channels beyond `rotary_dim` are concatenated back **untouched** — the partial-rotary path
    /// the shipped geometry always takes (96 of 128).
    pub fn apply(&self, x: &Tensor, tables: &MmRopeTables) -> Result<Tensor> {
        let shape = x.dims();
        if shape.len() != 4 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 mm-rope: expected [B, S, H, D], got {shape:?}"
            )));
        }
        let (seq, d) = (shape[1], shape[3]);
        let rotary = tables.rotary_dim();
        if rotary > d {
            return Err(CandleError::Msg(format!(
                "minimax-h3 mm-rope: rotary_dim {rotary} exceeds head dim {d}"
            )));
        }
        if tables.seq_len() != seq {
            return Err(CandleError::Msg(format!(
                "minimax-h3 mm-rope: rope tables cover {} rows but the sequence has {seq}",
                tables.seq_len()
            )));
        }

        // `[seq, rotary]` -> `[1, seq, 1, rotary]`, broadcasting over batch and heads.
        let cos_t = tables
            .cos
            .to_dtype(x.dtype())?
            .reshape((1, seq, 1, rotary))?;
        let sin_t = tables
            .sin
            .to_dtype(x.dtype())?
            .reshape((1, seq, 1, rotary))?;

        let head = x.narrow(3, 0, rotary)?;
        let half = rotary / 2;
        let x1 = head.narrow(3, 0, half)?;
        let x2 = head.narrow(3, half, rotary - half)?;
        let rotated = Tensor::cat(&[x2.neg()?, x1], 3)?;
        let out = head
            .broadcast_mul(&cos_t)?
            .add(&rotated.broadcast_mul(&sin_t)?)?;

        if rotary == d {
            return Ok(out.contiguous()?);
        }
        let tail = x.narrow(3, rotary, d - rotary)?;
        Ok(Tensor::cat(&[out, tail], 3)?.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    fn spread(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.61).sin() * 1.7).collect();
        Tensor::from_vec(vals, shape, &dev()).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
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
        assert!(MmRope::new(16, 0.0).is_err());
        assert!(MmRope::new(16, f64::NAN).is_err());

        // inv_freq[0] = 1, inv_freq[F-1] = theta^(-(F-1)/F).
        let ids = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (1, 3), &dev()).unwrap();
        let t = rope.tables(&ids, DType::F32).unwrap();
        let c = flat(&t.cos);
        assert!((c[0] - 1.0f32.cos()).abs() < 1e-6, "inv_freq[0] must be 1");
        let last = 10_000f64.powf(-15.0 / 16.0);
        assert!(
            (f64::from(c[15]) - last.cos()).abs() < 1e-6,
            "inv_freq[15] = theta^(-15/16)"
        );
    }

    /// **The axis split.** Channels `[0,F)` are `t`, `[F,2F)` are `h`, `[2F,3F)` are `w`, and
    /// `[3F,6F)` repeats that triple. Driven with a one-hot position per axis so a permuted or
    /// interleaved split cannot pass.
    #[test]
    fn axes_split_t_then_h_then_w_and_the_block_repeats() {
        let f = 4usize;
        let rope = MmRope::new(f, 100.0).unwrap();
        // Three rows, each with a single non-zero axis: t=1, h=1, w=1.
        let ids = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            (3, 3),
            &dev(),
        )
        .unwrap();
        let tables = rope.tables(&ids, DType::F32).unwrap();
        assert_eq!(tables.cos.dims(), &[3, 6 * f]);
        let sin_v = flat(&tables.sin);
        let row = |r: usize| &sin_v[r * (6 * f)..(r + 1) * (6 * f)];

        for (r, axis) in [(0usize, 0usize), (1, 1), (2, 2)] {
            let v = row(r);
            for i in 0..f {
                let want = (100f64.powf(-(i as f64) / f as f64)).sin() as f32;
                // The axis's own block is non-trivial...
                let at = axis * f + i;
                assert!(
                    (v[at] - want).abs() < 1e-6,
                    "row {r}: channel {at} should carry axis {axis} frequency {i}"
                );
                // ...and it repeats 3F later.
                assert!(
                    (v[at + 3 * f] - want).abs() < 1e-6,
                    "row {r}: the second half must repeat channel {at}"
                );
                // ...while the other two axes' blocks are zero (sin(0) = 0).
                for other in 0..3usize {
                    if other != axis {
                        let idx = other * f + i;
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
        let ids = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (1, 3), &dev()).unwrap();
        let t = rope.tables(&ids, DType::F32).unwrap();
        let c = flat(&t.cos);
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
        let tables = rope.tables(&ids, DType::F32).unwrap();
        let x = spread(&[1, 5, 2, 16]);
        let out = rope.apply(&x, &tables).unwrap();
        assert_eq!(out.dims(), x.dims());

        let tail_in = flat(&x.narrow(3, 12, 4).unwrap().contiguous().unwrap());
        let tail_out = flat(&out.narrow(3, 12, 4).unwrap().contiguous().unwrap());
        assert_eq!(
            tail_in, tail_out,
            "channels beyond 2·3·rope_freq_dim must pass through unrotated"
        );
        let head_in = flat(&x.narrow(3, 0, 12).unwrap().contiguous().unwrap());
        let head_out = flat(&out.narrow(3, 0, 12).unwrap().contiguous().unwrap());
        assert_ne!(head_in, head_out);
    }

    /// `rotate_half` is norm-preserving per rotated pair; a wrong half split breaks it.
    #[test]
    fn rotation_preserves_the_rotated_norm() {
        let rope = MmRope::new(2, 100.0).unwrap();
        let ids = spread(&[6, 3]);
        let tables = rope.tables(&ids, DType::F32).unwrap();
        let x = spread(&[1, 6, 1, 12]);
        let out = rope.apply(&x, &tables).unwrap();
        let a: f32 = flat(&x).iter().map(|v| v * v).sum();
        let b: f32 = flat(&out).iter().map(|v| v * v).sum();
        assert!(
            (a - b).abs() / a < 1e-4,
            "rotary must preserve norm: {a} vs {b}"
        );
    }

    /// A tables/sequence length mismatch is an error, not a broadcast that silently reuses row 0.
    #[test]
    fn a_length_mismatch_is_rejected() {
        let rope = MmRope::new(2, 100.0).unwrap();
        let tables = rope.tables(&spread(&[4, 3]), DType::F32).unwrap();
        assert!(rope.apply(&spread(&[1, 7, 1, 12]), &tables).is_err());
        assert!(rope.apply(&spread(&[1, 4, 1, 8]), &tables).is_err());
        assert!(rope.tables(&spread(&[4, 2]), DType::F32).is_err());
        assert!(rope.apply(&spread(&[4, 1, 12]), &tables).is_err());
    }
}
