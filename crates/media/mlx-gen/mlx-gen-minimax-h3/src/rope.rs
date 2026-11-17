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
//!
//! # dtype convention: f32 tables, rotation at the STREAM dtype
//!
//! This is the **same** convention [`crate::dit::rope`] follows, and in both cases it is the
//! reference's, not ours. Checked against `diffusers` 0.40.0.dev0 @ `7564fb01`:
//!
//! * **Tables at f32.** `MiniMaxH3VideoViTDecoder3d.forward`
//!   (`models/autoencoders/autoencoder_kl_minimax_h3.py:461`) builds its position grid with an
//!   explicit `dtype=torch.float32`, and `MiniMaxH3VideoRotaryPosEmbed.__init__` (line 290)
//!   registers `inv_freq` float32, so `angles.cos()` / `angles.sin()` (line 294-296) are f32
//!   whatever the decoder runs at. [`Rope3d::tables`] therefore returns f32 unconditionally and
//!   does not inherit `ids`' dtype.
//! * **Rotation at the stream dtype.** `MiniMaxH3VideoAttnProcessor.__call__` then narrows them —
//!   `cos = cos.to(query.dtype)` / `sin = sin.to(query.dtype)`
//!   (`autoencoder_kl_minimax_h3.py:319-320`) — *before* `query_rotary * cos + query_rotated * sin`
//!   (line 328). The multiply-add runs in the activation's dtype, not f32.
//!
//! So the narrowing in [`Rope3d::apply`] is the port being faithful, not a missed upcast. The DiT's
//! MM-RoPE narrows for the same reason: its own reference module, `_apply_rotary_emb`
//! (`models/transformers/transformer_minimax_h3.py:66-67`), also does `cos.to(hidden_states.dtype)`
//! before the multiply-add while keeping the tables f32 (`:93`, plus `"rope"` in
//! `_keep_in_fp32_modules` at `:448`). The VAE additionally pins the whole decoder in fp32
//! (`_keep_in_fp32_modules = ["encoder", "decoder", ...]`, `autoencoder_kl_minimax_h3.py:532`), so
//! *here* the narrow is a no-op on the shipped recipe. Do not "fix" either rotary into an f32
//! rotation; it would be a different model.

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

/// Length-normalized token ids for an `(H, W)` latent image — the 2-D decoder's grid.
///
/// The same cell-centre mapping as [`create_token_ids`], meshed `ij` over two axes, so token
/// `h·W + w` carries `(h_coord, w_coord)` (`MiniMaxH3ImageViTDecoder2d.forward`). Returns
/// `[1, H·W, 2]`.
pub fn create_token_ids_2d(h: i32, w: i32, dtype: Dtype) -> Result<Array> {
    if h <= 0 || w <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 rope: latent grid must be positive, got ({h}, {w})"
        )));
    }
    let axis = |n: i32| -> Vec<f32> {
        (0..n)
            .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
            .collect()
    };
    let (hc, wc) = (axis(h), axis(w));
    let mut ids = Vec::with_capacity((h * w * 2) as usize);
    for &hv in &hc {
        for &wv in &wc {
            ids.push(hv);
            ids.push(wv);
        }
    }
    Ok(Array::from_slice(&ids, &[1, h * w, 2]).as_dtype(dtype)?)
}

/// The N-axis rotary embedding: 3 axes `(t, h, w)` for the video decoder, 2 axes `(h, w)` for the
/// image decoder (`MiniMaxH3*RotaryPosEmbed(dim, theta, num_axes)`). The name predates the 2-axis
/// mode; the arithmetic is the same module with `num_axes` as a parameter.
#[derive(Debug, Clone)]
pub struct Rope3d {
    inv_freq: Vec<f32>,
    /// Rotated head dims.
    pub apply_dim: i32,
    num_axes: i32,
}

impl Rope3d {
    /// The 3-axis video rotary. `apply_dim` must be a positive multiple of `2 · 3` — the reference
    /// raises otherwise.
    pub fn new(apply_dim: i32, theta: f32) -> Result<Self> {
        Self::with_axes(apply_dim, theta, 3)
    }

    /// A rotary over `num_axes` position axes. `apply_dim` must be a positive multiple of
    /// `2 · num_axes`.
    pub fn with_axes(apply_dim: i32, theta: f32, num_axes: i32) -> Result<Self> {
        if num_axes <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: num_axes {num_axes} must be positive"
            )));
        }
        let per = 2 * num_axes;
        if apply_dim <= 0 || apply_dim % per != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: apply_dim {apply_dim} must be a positive multiple of {per}"
            )));
        }
        // `1 / theta ** arange(0, 1, 2·n_dim/dim)` — `apply_dim / (2·n_dim)` frequencies per
        // position axis. At 3 axes `step` is `6.0 / apply_dim`, the exact f32 the 3-D-only
        // constructor computed, so the video tables are unchanged bit for bit.
        let step = per as f32 / apply_dim as f32;
        let inv_freq = (0..apply_dim / per)
            .map(|i| theta.powf(-(i as f32 * step)))
            .collect();
        Ok(Self {
            inv_freq,
            apply_dim,
            num_axes,
        })
    }

    /// Frequencies per position axis — `apply_dim / (2 · num_axes)`.
    pub fn freqs_per_axis(&self) -> usize {
        self.inv_freq.len()
    }

    /// Position axes the ids carry.
    pub fn num_axes(&self) -> i32 {
        self.num_axes
    }

    /// Build `(cos, sin)` for `ids` of shape `[1, N, num_axes]`, returning
    /// `[1, N, 1, apply_dim]` each.
    ///
    /// Always **f32**, whatever `ids` carries: the reference's position grid is built
    /// `dtype=torch.float32` and `inv_freq` is a float32 buffer
    /// (`autoencoder_kl_minimax_h3.py:461`, `:290`), so its `cos`/`sin` are f32 regardless of the
    /// decoder's compute dtype. [`Rope3d::apply`] performs the single narrow, as the reference does.
    /// The candle twin has always done this; see the module doc.
    pub fn tables(&self, ids: &Array) -> Result<RopeTables> {
        let shape = ids.shape();
        let axes = self.num_axes;
        if shape.len() != 3 || shape[2] != axes {
            return Err(Error::Msg(format!(
                "minimax-h3 rope: expected ids as [B, N, {axes}], got {shape:?}"
            )));
        }
        let (b, n) = (shape[0], shape[1]);
        let f = self.inv_freq.len() as i32;

        // angles[b, n, axis, f] = 2π · ids[b, n, axis] · inv_freq[f]
        let inv = Array::from_slice(&self.inv_freq, &[1, 1, 1, f]).as_dtype(Dtype::Float32)?;
        let scaled = multiply(
            &ids.as_dtype(Dtype::Float32)?.reshape(&[b, n, axes, 1])?,
            Array::from_f32(2.0 * std::f32::consts::PI),
        )?;
        let angles = multiply(&scaled, &inv)?.reshape(&[b, n, axes * f])?;
        // `.tile(2)` on the last axis: the half-split rotary consumes [θ, θ].
        let angles =
            concatenate_axis(&[angles.clone(), angles], -1)?.reshape(&[b, n, 1, self.apply_dim])?;
        Ok(RopeTables {
            cos: cos(&angles)?,
            sin: sin(&angles)?,
        })
    }

    /// Apply the rotation to `t` of shape `[B, S, H, D]`.
    ///
    /// When `apply_dim < D` only the leading `apply_dim` dims rotate and the tail is concatenated
    /// back untouched — the partial-rotary path this model always takes.
    ///
    /// The f32 tables are **narrowed to `t`'s dtype and the rotation runs there**, mirroring
    /// `MiniMaxH3VideoAttnProcessor.__call__` (`autoencoder_kl_minimax_h3.py:319-320`, 328). See
    /// the module doc. [`crate::dit::rope::MmRope::apply`] narrows the same way, for the same
    /// reason: its own reference module narrows explicitly too.
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
        // Reference-faithful narrow, NOT a missed upcast: `cos.to(query.dtype)` at
        // `autoencoder_kl_minimax_h3.py:319-320`.
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

    /// The image decoder's rotary: 32 rotated dims over 2 axes -> 8 frequencies per axis, the
    /// count `MiniMaxH3ImageRotaryPosEmbed(32, num_axes=2)` builds.
    #[test]
    fn two_axis_rotary_has_apply_dim_over_four_frequencies() {
        let rope = Rope3d::with_axes(32, 100.0, 2).unwrap();
        assert_eq!(rope.freqs_per_axis(), 8);
        assert_eq!(rope.num_axes(), 2);
        assert!(Rope3d::with_axes(6, 100.0, 2).is_err());
        assert!(Rope3d::with_axes(32, 100.0, 0).is_err());
    }

    /// `new` is the 3-axis case of `with_axes`, table for table — the video decoder's rotary is
    /// unchanged by the 2-axis generalization.
    #[test]
    fn the_three_axis_constructor_is_unchanged() {
        let a = Rope3d::new(48, 100.0).unwrap();
        let b = Rope3d::with_axes(48, 100.0, 3).unwrap();
        assert_eq!(a.inv_freq, b.inv_freq);
        // The pre-generalization formula, written out: `theta^-(i · 6/48)`.
        let legacy: Vec<f32> = (0..8)
            .map(|i| 100.0f32.powf(-(i as f32 * (6.0f32 / 48.0))))
            .collect();
        assert_eq!(a.inv_freq, legacy);
        let ids = create_token_ids(2, 3, 4, Dtype::Float32).unwrap();
        let (ta, tb) = (a.tables(&ids).unwrap(), b.tables(&ids).unwrap());
        assert_eq!(ta.cos.as_slice::<f32>(), tb.cos.as_slice::<f32>());
        assert_eq!(ta.sin.as_slice::<f32>(), tb.sin.as_slice::<f32>());
    }

    /// 2-D ids mesh `(h, w)` in `ij` order at cell centres.
    #[test]
    fn two_d_token_ids_are_row_major_cell_centres() {
        let ids = create_token_ids_2d(2, 3, Dtype::Float32).unwrap();
        assert_eq!(ids.shape(), &[1, 6, 2]);
        let v: Vec<f32> = ids.as_slice::<f32>().to_vec();
        let expect = [
            (-0.5, -2.0 / 3.0),
            (-0.5, 0.0),
            (-0.5, 2.0 / 3.0),
            (0.5, -2.0 / 3.0),
            (0.5, 0.0),
            (0.5, 2.0 / 3.0),
        ];
        for (k, (h, w)) in expect.iter().enumerate() {
            assert!(
                (v[2 * k] - h).abs() < 1e-6 && (v[2 * k + 1] - w).abs() < 1e-6,
                "token {k}"
            );
        }
        assert!(create_token_ids_2d(0, 3, Dtype::Float32).is_err());
    }

    /// Table layout over 2 axes: `[h-freqs | w-freqs]` tiled twice, angle `2π · coord · f`.
    #[test]
    fn two_axis_tables_follow_the_reference_layout() {
        let rope = Rope3d::with_axes(8, 100.0, 2).unwrap();
        let ids = Array::from_slice(&[0.25f32, -0.5], &[1, 1, 2]);
        let t = rope.tables(&ids).unwrap();
        assert_eq!(t.cos.shape(), &[1, 1, 1, 8]);
        let inv = [1.0f32, 100.0f32.powf(-0.5)];
        let mut want = Vec::new();
        for coord in [0.25f32, -0.5] {
            for f in inv {
                want.push((2.0 * std::f32::consts::PI * coord * f).cos());
            }
        }
        let want: Vec<f32> = want.iter().chain(want.iter()).copied().collect();
        for (g, w) in t.cos.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        // 3-axis ids into a 2-axis rotary is a shape error, not a silent reinterpretation.
        let wrong = create_token_ids(1, 1, 1, Dtype::Float32).unwrap();
        assert!(rope.tables(&wrong).is_err());
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

    /// The dtype convention this rotary ports: **f32 tables, rotation at the STREAM dtype**.
    ///
    /// `MiniMaxH3VideoRotaryPosEmbed` builds its angles from a `dtype=torch.float32` grid and a
    /// float32 `inv_freq` buffer, and `MiniMaxH3VideoAttnProcessor` then narrows with
    /// `cos.to(query.dtype)` *before* the multiply-add
    /// (`autoencoder_kl_minimax_h3.py:461`, `:290`, `:319-320`, `:328`). Both halves are pinned
    /// here so the narrow is not "fixed" into an f32 rotation. [`crate::dit::rope`] pins the same
    /// two halves against its own reference module.
    #[test]
    fn tables_are_f32_and_the_rotation_runs_at_the_stream_dtype() {
        let rope = Rope3d::new(12, 100.0).unwrap();
        // Handed bf16 ids, the tables still come back f32: the reference's grid is always f32.
        let ids = create_token_ids(2, 2, 2, Dtype::Bfloat16).unwrap();
        let tables = rope.tables(&ids).unwrap();
        assert_eq!(tables.cos.dtype(), Dtype::Float32);
        assert_eq!(tables.sin.dtype(), Dtype::Float32);

        let t = spread(&[1, 8, 1, 12]).as_dtype(Dtype::Bfloat16).unwrap();
        let shipped = rope.apply(&t, &tables).unwrap();
        assert_eq!(shipped.dtype(), Dtype::Bfloat16);

        // The measurement that makes the assertion above non-vacuous: rotating at f32 and
        // narrowing afterwards (an f32 rotation, which neither reference module does) is a
        // *different* result, so the
        // choice of convention is load-bearing rather than cosmetic.
        let at_f32 = rope
            .apply(&t.as_dtype(Dtype::Float32).unwrap(), &tables)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let a: Vec<f32> = shipped
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice()
            .to_vec();
        let b: Vec<f32> = at_f32.as_dtype(Dtype::Float32).unwrap().as_slice().to_vec();
        let num = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let den = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            num / den > 1e-4,
            "narrowing before vs after the rotation must be distinguishable at bf16 ({:.3e})",
            num / den
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
