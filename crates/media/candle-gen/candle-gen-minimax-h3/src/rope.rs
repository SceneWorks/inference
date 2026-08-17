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
//! The tables are built on the host in **f64** and materialized as f32, exactly as the MLX lane
//! does, so the two backends' `cos`/`sin` agree to f32 round-off rather than to whichever
//! transcendental each accelerator ships.
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
//!   whatever the decoder runs at. [`Rope3d::tables`] therefore takes no dtype.
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
//! (`_keep_in_fp32_modules = ["encoder", "decoder", ...]`,
//! `autoencoder_kl_minimax_h3.py:532`), so *here* the narrow is a no-op on the shipped recipe. Do
//! not "fix" either rotary into an f32 rotation; it would be a different model.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

/// Precomputed `(cos, sin)` tables for one token grid.
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// `[1, N, 1, rope_apply_dim]`.
    pub cos: Tensor,
    /// `[1, N, 1, rope_apply_dim]`.
    pub sin: Tensor,
}

/// Length-normalized token ids for a `(T, H, W)` latent grid.
///
/// Each axis maps index `i` of `n` to `2·(i + 0.5)/n - 1`, i.e. cell CENTRES spanning `(-1, 1)`;
/// the ids are then meshed in `ij` order and flattened, so token `t·H·W + h·W + w` carries
/// `(t_coord, h_coord, w_coord)`. Returns `[1, T·H·W, 3]` f32.
pub fn create_token_ids(t: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    if t == 0 || h == 0 || w == 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 rope: latent grid must be positive, got ({t}, {h}, {w})"
        )));
    }
    let axis = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0)
            .collect()
    };
    let (tc, hc, wc) = (axis(t), axis(h), axis(w));
    let mut ids = Vec::with_capacity(t * h * w * 3);
    for &tv in &tc {
        for &hv in &hc {
            for &wv in &wc {
                ids.push(tv);
                ids.push(hv);
                ids.push(wv);
            }
        }
    }
    Ok(Tensor::from_vec(ids, (1, t * h * w, 3), device)?)
}

/// The 3-D rotary embedding.
#[derive(Debug, Clone)]
pub struct Rope3d {
    inv_freq: Vec<f32>,
    /// Rotated head dims.
    pub apply_dim: usize,
}

impl Rope3d {
    /// `apply_dim` must be a positive multiple of `2 · 3` — the reference raises otherwise.
    pub fn new(apply_dim: usize, theta: f32) -> Result<Self> {
        if apply_dim == 0 || !apply_dim.is_multiple_of(6) {
            return Err(CandleError::Msg(format!(
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

    /// Build `(cos, sin)` for `ids` of shape `[B, N, 3]`, returning `[B, N, 1, apply_dim]` each.
    ///
    /// The last axis is the axis-major stack `[t-freqs | h-freqs | w-freqs]` **tiled twice**, which
    /// is what the half-split rotary below consumes.
    pub fn tables(&self, ids: &Tensor) -> Result<RopeTables> {
        let shape = ids.dims();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 rope: expected ids as [B, N, 3], got {shape:?}"
            )));
        }
        let (b, n) = (shape[0], shape[1]);
        let f = self.inv_freq.len();
        let device = ids.device();

        // angles[b, n, axis, f] = 2π · ids[b, n, axis] · inv_freq[f]
        let inv = Tensor::from_vec(self.inv_freq.clone(), (1, 1, 1, f), device)?;
        let scaled =
            (ids.to_dtype(DType::F32)?.reshape((b, n, 3, 1))? * (2.0 * std::f64::consts::PI))?;
        let angles = scaled.broadcast_mul(&inv)?.reshape((b, n, 3 * f))?;
        // `.tile(2)` on the last axis: the half-split rotary consumes [θ, θ].
        let angles = Tensor::cat(&[&angles, &angles], 2)?.reshape((b, n, 1, self.apply_dim))?;
        Ok(RopeTables {
            cos: angles.cos()?,
            sin: angles.sin()?,
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
    pub fn apply(&self, t: &Tensor, tables: &RopeTables) -> Result<Tensor> {
        let shape = t.dims();
        if shape.len() != 4 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 rope: expected [B, S, H, D], got {shape:?}"
            )));
        }
        let d = shape[3];
        if self.apply_dim > d {
            return Err(CandleError::Msg(format!(
                "minimax-h3 rope: apply_dim {} exceeds head dim {d}",
                self.apply_dim
            )));
        }
        // Reference-faithful narrow, NOT a missed upcast: `cos.to(query.dtype)` at
        // `autoencoder_kl_minimax_h3.py:319-320`.
        let cos_t = tables.cos.to_dtype(t.dtype())?;
        let sin_t = tables.sin.to_dtype(t.dtype())?;

        let half = self.apply_dim / 2;
        let rotate = |x: &Tensor| -> Result<Tensor> {
            let x1 = x.narrow(3, 0, half)?;
            let x2 = x.narrow(3, half, half)?;
            Ok(Tensor::cat(&[&x2.neg()?, &x1], 3)?)
        };

        let head = t.narrow(3, 0, self.apply_dim)?.contiguous()?;
        let rotated = head
            .broadcast_mul(&cos_t)?
            .add(&rotate(&head)?.broadcast_mul(&sin_t)?)?;
        if self.apply_dim == d {
            return Ok(rotated);
        }
        let tail = t.narrow(3, self.apply_dim, d - self.apply_dim)?;
        Ok(Tensor::cat(&[&rotated, &tail], 3)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 1.3).collect();
        Tensor::from_vec(vals, shape, &Device::Cpu).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn token_ids_are_cell_centres_spanning_minus_one_to_one() {
        let ids = create_token_ids(2, 1, 3, &Device::Cpu).unwrap();
        assert_eq!(ids.dims(), &[1, 6, 3]);
        let v = flat(&ids);
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
        let ids = create_token_ids(1, 1, 4, &Device::Cpu).unwrap();
        let tables = rope.tables(&ids).unwrap();
        let t = spread(&[1, 4, 2, 16]);
        let out = rope.apply(&t, &tables).unwrap();
        assert_eq!(out.dims(), t.dims());

        let tail_in = flat(&t.narrow(3, 12, 4).unwrap().contiguous().unwrap());
        let tail_out = flat(&out.narrow(3, 12, 4).unwrap().contiguous().unwrap());
        assert_eq!(
            tail_in, tail_out,
            "dims beyond rope_apply_dim must pass through unrotated"
        );
        // ...while the rotated head really did change.
        let head_in = flat(&t.narrow(3, 0, 12).unwrap().contiguous().unwrap());
        let head_out = flat(&out.narrow(3, 0, 12).unwrap().contiguous().unwrap());
        assert_ne!(head_in, head_out);
    }

    /// Rotation is norm-preserving per rotated pair, which a wrong `rotate_half` split breaks.
    ///
    /// Deliberately run at a NON-zero grid: a rotary evaluated at position id 0 is an exact
    /// identity and would report PASS for an implementation that did nothing at all.
    #[test]
    fn rotation_preserves_the_head_norm_and_is_not_an_identity() {
        let rope = Rope3d::new(12, 100.0).unwrap();
        let ids = create_token_ids(2, 2, 2, &Device::Cpu).unwrap();
        let tables = rope.tables(&ids).unwrap();
        let t = spread(&[1, 8, 1, 12]);
        let out = rope.apply(&t, &tables).unwrap();
        let norm_in: f32 = flat(&t).iter().map(|v| v * v).sum();
        let norm_out: f32 = flat(&out).iter().map(|v| v * v).sum();
        assert!(
            (norm_in - norm_out).abs() / norm_in < 1e-4,
            "rotary must preserve norm: {norm_in} vs {norm_out}"
        );
        let moved: f32 = flat(&t)
            .iter()
            .zip(flat(&out).iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(moved > 1e-2, "the rotary did nothing ({moved:.3e})");
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
        let ids = create_token_ids(2, 2, 2, &Device::Cpu).unwrap();
        let tables = rope.tables(&ids).unwrap();
        assert_eq!(tables.cos.dtype(), DType::F32);
        assert_eq!(tables.sin.dtype(), DType::F32);

        let t = spread(&[1, 8, 1, 12]).to_dtype(DType::BF16).unwrap();
        let shipped = rope.apply(&t, &tables).unwrap();
        assert_eq!(shipped.dtype(), DType::BF16);

        // The measurement that makes the assertion above non-vacuous: rotating at f32 and
        // narrowing afterwards (an f32 rotation, which neither reference module does) is a
        // *different* result, so the
        // choice of convention is load-bearing rather than cosmetic.
        let at_f32 = rope
            .apply(&t.to_dtype(DType::F32).unwrap(), &tables)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let a = flat(&shipped.to_dtype(DType::F32).unwrap());
        let b = flat(&at_f32.to_dtype(DType::F32).unwrap());
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
        let ids = create_token_ids(1, 1, 2, &Device::Cpu).unwrap();
        let tables = rope.tables(&ids).unwrap();
        assert_eq!(tables.cos.dims(), &[1, 2, 1, 12]);
        let c = flat(&tables.cos);
        // `.tile(2)` means the second half repeats the first.
        for i in 0..6 {
            assert!((c[i] - c[i + 6]).abs() < 1e-6, "half {i} not duplicated");
        }
    }
}
