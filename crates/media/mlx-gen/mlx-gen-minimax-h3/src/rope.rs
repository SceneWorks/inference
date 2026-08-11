//! 3-D **partial** rotary position embedding (`base_module.py::RotaryEmbeddingND`, `n_dim = 3`,
//! `use_angle = True`) and the normalized token ids it consumes (`func.py::create_token_ids`).
//!
//! Two details separate this from a stock RoPE and either one silently changes every attention
//! score if it is missed:
//!
//! 1. **`use_angle`** multiplies each position by `2π` before the frequency scaling. With the
//!    length-normalized coordinates below (which live in `(-1, 1)` rather than `0..N`) that turns
//!    the embedding into a resolution-independent angle, so the same weights decode any frame size.
//! 2. **`rope_dim_ratio = 0.75`** rotates only the first `int(head_dim · 0.75)` = 48 of 64 head
//!    dims; the remaining 16 are concatenated back **unrotated**. Rotating all 64 (the reference
//!    module's own default) is a different model.

use mlx_rs::ops::{concatenate_axis, cos, multiply, negative, sin};
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

use crate::tensor::slice_axis;

/// Precomputed `(cos, sin)` tables for one token grid.
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// `[1, N, 1, rope_apply_dim]`.
    pub cos: Array,
    /// `[1, N, 1, rope_apply_dim]`.
    pub sin: Array,
}

/// Length-normalized token ids for a `(T, H, W)` latent grid.
///
/// Each axis maps index `i` of `n` to `2·(i + 0.5)/n - 1`, i.e. cell CENTRES spanning `(-1, 1)`;
/// the ids are then meshed in `ij` order and flattened, so token `t·H·W + h·W + w` carries
/// `(t_coord, h_coord, w_coord)`. Returns `[1, T·H·W, 3]`.
pub fn create_token_ids(t: i32, h: i32, w: i32, dtype: Dtype) -> Result<Array> {
    if t <= 0 || h <= 0 || w <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 rope: latent grid must be positive, got ({t}, {h}, {w})"
        )));
    }
    let axis = |n: i32| -> Vec<f32> {
        (0..n)
            .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
            .collect()
    };
    let (tc, hc, wc) = (axis(t), axis(h), axis(w));
    let mut ids = Vec::with_capacity((t * h * w * 3) as usize);
    for &tv in &tc {
        for &hv in &hc {
            for &wv in &wc {
                ids.push(tv);
                ids.push(hv);
                ids.push(wv);
            }
        }
    }
    Ok(Array::from_slice(&ids, &[1, t * h * w, 3]).as_dtype(dtype)?)
}

/// The 3-D rotary embedding.
#[derive(Debug, Clone)]
pub struct Rope3d {
    inv_freq: Vec<f32>,
    /// Rotated head dims.
    pub apply_dim: i32,
}

impl Rope3d {
    /// `apply_dim` must be a positive multiple of `2 · 3` — the reference raises otherwise.
    pub fn new(apply_dim: i32, theta: f32) -> Result<Self> {
        if apply_dim <= 0 || apply_dim % 6 != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: apply_dim {apply_dim} must be a positive multiple of 6"
            )));
        }
        // `1 / theta ** arange(0, 1, 2·n_dim/dim)` — `apply_dim/6` frequencies per position axis.
        let step = 6.0f32 / apply_dim as f32;
        let inv_freq = (0..apply_dim / 6)
            .map(|i| theta.powf(-(i as f32 * step)))
            .collect();
        Ok(Self {
            inv_freq,
            apply_dim,
        })
    }

    /// Frequencies per position axis — `apply_dim / 6`.
    pub fn freqs_per_axis(&self) -> usize {
        self.inv_freq.len()
    }

    /// Build `(cos, sin)` for `ids` of shape `[1, N, 3]`, returning `[1, N, 1, apply_dim]` each.
    pub fn tables(&self, ids: &Array) -> Result<RopeTables> {
        let shape = ids.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: expected ids as [B, N, 3], got {shape:?}"
            )));
        }
        let (b, n) = (shape[0], shape[1]);
        let dtype = ids.dtype();
        let f = self.inv_freq.len() as i32;

        // angles[b, n, axis, f] = 2π · ids[b, n, axis] · inv_freq[f]
        let inv = Array::from_slice(&self.inv_freq, &[1, 1, 1, f]).as_dtype(Dtype::Float32)?;
        let scaled = multiply(
            &ids.as_dtype(Dtype::Float32)?.reshape(&[b, n, 3, 1])?,
            Array::from_f32(2.0 * std::f32::consts::PI),
        )?;
        let angles = multiply(&scaled, &inv)?.reshape(&[b, n, 3 * f])?;
        // `.tile(2)` on the last axis: the half-split rotary consumes [θ, θ].
        let angles =
            concatenate_axis(&[angles.clone(), angles], -1)?.reshape(&[b, n, 1, self.apply_dim])?;
        Ok(RopeTables {
            cos: cos(&angles)?.as_dtype(dtype)?,
            sin: sin(&angles)?.as_dtype(dtype)?,
        })
    }

    /// Apply the rotation to `t` of shape `[B, S, H, D]`.
    ///
    /// When `apply_dim < D` only the leading `apply_dim` dims rotate and the tail is concatenated
    /// back untouched — the partial-rotary path this model always takes.
    pub fn apply(&self, t: &Array, tables: &RopeTables) -> Result<Array> {
        let shape = t.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: expected [B, S, H, D], got {shape:?}"
            )));
        }
        let d = shape[3];
        if self.apply_dim > d {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: apply_dim {} exceeds head dim {d}",
                self.apply_dim
            )));
        }
        let cos_t = tables.cos.as_dtype(t.dtype())?;
        let sin_t = tables.sin.as_dtype(t.dtype())?;

        let rotate = |x: &Array| -> Result<Array> {
            let half = self.apply_dim / 2;
            let x1 = slice_axis(x, 3, 0, half)?;
            let x2 = slice_axis(x, 3, half, self.apply_dim)?;
            Ok(concatenate_axis(&[negative(&x2)?, x1], -1)?)
        };

        if self.apply_dim == d {
            let rotated = multiply(t, &cos_t)?;
            return Ok(mlx_rs::ops::add(&rotated, &multiply(&rotate(t)?, &sin_t)?)?);
        }
        let head = slice_axis(t, 3, 0, self.apply_dim)?;
        let tail = slice_axis(t, 3, self.apply_dim, d)?;
        let rotated = mlx_rs::ops::add(
            &multiply(&head, &cos_t)?,
            &multiply(&rotate(&head)?, &sin_t)?,
        )?;
        Ok(concatenate_axis(&[rotated, tail], -1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic host-side test data. MLX's RNG evaluates on the GPU stream, which is not
    /// available on every `cargo test` worker thread; building from a host slice keeps these unit
    /// tests both stream-free and reproducible.
    fn spread(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 1.3).collect();
        Array::from_slice(&vals, shape)
    }

    #[test]
    fn token_ids_are_cell_centres_spanning_minus_one_to_one() {
        let ids = create_token_ids(2, 1, 3, Dtype::Float32).unwrap();
        assert_eq!(ids.shape(), &[1, 6, 3]);
        let v: Vec<f32> = ids.as_slice::<f32>().to_vec();
        // Temporal axis of length 2 -> centres at -0.5, +0.5.
        assert!((v[0] - -0.5).abs() < 1e-6);
        assert!((v[9] - 0.5).abs() < 1e-6);
        // Width axis of length 3 -> -2/3, 0, +2/3.
        assert!((v[2] - (-2.0 / 3.0)).abs() < 1e-6);
        assert!(v[5].abs() < 1e-6);
        assert!((v[8] - (2.0 / 3.0)).abs() < 1e-6);
        // Height axis of length 1 -> a single centre at 0.
        assert!(v[1].abs() < 1e-6);
    }

    #[test]
    fn frequency_count_is_apply_dim_over_six() {
        let rope = Rope3d::new(48, 100.0).unwrap();
        assert_eq!(rope.freqs_per_axis(), 8);
        let rope = Rope3d::new(12, 100.0).unwrap();
        assert_eq!(rope.freqs_per_axis(), 2);
        assert!(Rope3d::new(32, 100.0).is_err());
        assert!(Rope3d::new(0, 100.0).is_err());
    }

    /// The pass-through tail must be returned BYTE-identical — this is what makes the rotary
    /// partial. A full rotation would perturb the last 16 dims of every head.
    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let rope = Rope3d::new(12, 100.0).unwrap();
        let ids = create_token_ids(1, 1, 4, Dtype::Float32).unwrap();
        let tables = rope.tables(&ids).unwrap();
        let t = spread(&[1, 4, 2, 16]);
        let out = rope.apply(&t, &tables).unwrap();
        assert_eq!(out.shape(), t.shape());

        let tail_in = slice_axis(&t, 3, 12, 16).unwrap();
        let tail_out = slice_axis(&out, 3, 12, 16).unwrap();
        assert_eq!(
            tail_in.as_slice::<f32>(),
            tail_out.as_slice::<f32>(),
            "dims beyond rope_apply_dim must pass through unrotated"
        );
        // ...while the rotated head really did change.
        let head_in = slice_axis(&t, 3, 0, 12).unwrap();
        let head_out = slice_axis(&out, 3, 0, 12).unwrap();
        assert_ne!(head_in.as_slice::<f32>(), head_out.as_slice::<f32>());
    }

    /// Rotation is norm-preserving per rotated pair, which a wrong `rotate_half` split breaks.
    #[test]
    fn rotation_preserves_the_head_norm() {
        let rope = Rope3d::new(12, 100.0).unwrap();
        let ids = create_token_ids(2, 2, 2, Dtype::Float32).unwrap();
        let tables = rope.tables(&ids).unwrap();
        let t = spread(&[1, 8, 1, 12]);
        let out = rope.apply(&t, &tables).unwrap();
        let norm_in: f32 = t.square().unwrap().sum(None).unwrap().item();
        let norm_out: f32 = out.square().unwrap().sum(None).unwrap().item();
        assert!(
            (norm_in - norm_out).abs() / norm_in < 1e-4,
            "rotary must preserve norm: {norm_in} vs {norm_out}"
        );
    }

    #[test]
    fn tables_have_the_duplicated_half_layout() {
        let rope = Rope3d::new(12, 100.0).unwrap();
        let ids = create_token_ids(1, 1, 2, Dtype::Float32).unwrap();
        let tables = rope.tables(&ids).unwrap();
        assert_eq!(tables.cos.shape(), &[1, 2, 1, 12]);
        let c: Vec<f32> = tables.cos.as_slice::<f32>().to_vec();
        // `.tile(2)` means the second half repeats the first.
        for i in 0..6 {
            assert!((c[i] - c[i + 6]).abs() < 1e-6, "half {i} not duplicated");
        }
    }
}
