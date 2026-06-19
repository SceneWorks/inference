//! Boogu DiT 3-axis (t, h, w) unified RoPE — the OmniGen2 `BooguImageDoubleStreamRotaryPosEmbed`.
//!
//! Two things differ from the Qwen3-VL text encoder's RoPE and matter for parity:
//!  1. **Complex *interleaved* rotation** (`apply_rotary_emb(use_real=False)`, the "lumina" branch):
//!     adjacent dims `(2k, 2k+1)` form a complex pair `x[2k] + i·x[2k+1]` rotated by `e^{iθ_k}`
//!     (GPT-J / interleaved), *not* the text encoder's half-split `[x1, x2] → [-x2, x1]`. MLX has no
//!     `view_as_complex`, so we do the real arithmetic directly.
//!  2. **Three position axes**: per token the rotary frequency index `k ∈ [0, 60)` is grouped into
//!     three contiguous blocks of 20 (`axes_dim_rope = [40,40,40]` ⇒ 20 complex freqs each), one per
//!     axis. Text tokens use position `(i, i, i)`; image patch tokens use `(cap_len, row, col)`.
//!
//! Each axis shares the same 20-vector of inverse frequencies `θ^(−2j/40)` (`θ = 10000`). We build the
//! `cos`/`sin` tables on the CPU in f32 (the reference builds the freqs in f32 on MPS) and slice the
//! joint table into its text-only / image-only sub-ranges.

use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// Precomputed `cos`/`sin` rotary tables for one forward pass.
///
/// Layout is `[1, cap_len + img_len, head_dim/2]` (f32). The text tokens occupy `[0, cap_len)` and
/// the image patch tokens `[cap_len, cap_len + img_len)`, matching the joint `[instruct; img]` order.
pub struct RopeTables {
    cos: Array,
    sin: Array,
    cap_len: i32,
}

impl RopeTables {
    /// Build the joint table for a text-to-image forward (no reference images): `cap_len` text
    /// positions followed by an `h_tokens × w_tokens` image grid (row-major, `h` outer).
    pub fn build_t2i(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
    ) -> Self {
        let half_axis = axes_dim / 2; // 20 complex freqs per axis
        let half = half_axis * 3; // 60 for head_dim 120
        let inv: Vec<f32> = (0..half_axis)
            .map(|j| theta.powf(-(2.0 * j as f32) / axes_dim as f32))
            .collect();

        let img_len = h_tokens * w_tokens;
        let total = cap_len + img_len;
        let mut cos = vec![0f32; total * half];
        let mut sin = vec![0f32; total * half];

        for t in 0..total {
            let (p0, p1, p2) = if t < cap_len {
                let i = t as f32;
                (i, i, i)
            } else {
                let it = t - cap_len;
                (
                    cap_len as f32,
                    (it / w_tokens) as f32,
                    (it % w_tokens) as f32,
                )
            };
            for k in 0..half {
                let p = match k / half_axis {
                    0 => p0,
                    1 => p1,
                    _ => p2,
                };
                let angle = p * inv[k % half_axis];
                cos[t * half + k] = angle.cos();
                sin[t * half + k] = angle.sin();
            }
        }

        let shape = [1, total as i32, half as i32];
        Self {
            cos: Array::from_slice(&cos, &shape),
            sin: Array::from_slice(&sin, &shape),
            cap_len: cap_len as i32,
        }
    }

    /// `(cos, sin)` for the text tokens only (`context_refiner`).
    pub fn text(&self) -> Result<(Array, Array)> {
        Ok((
            axis1(&self.cos, 0, self.cap_len)?,
            axis1(&self.sin, 0, self.cap_len)?,
        ))
    }

    /// `(cos, sin)` for the image patch tokens only (`noise_refiner`, `img_self_attn`).
    pub fn image(&self) -> Result<(Array, Array)> {
        let end = self.cos.shape()[1];
        Ok((
            axis1(&self.cos, self.cap_len, end)?,
            axis1(&self.sin, self.cap_len, end)?,
        ))
    }

    /// `(cos, sin)` for the full joint `[text; image]` sequence (double / single stream).
    pub fn joint(&self) -> (Array, Array) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// Slice `[1, L, D]` along the sequence axis (axis 1) to `[start, end)`.
fn axis1(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

/// Apply the complex-interleaved rotary embedding to `x` in `[b, s, heads, head_dim]` layout.
///
/// `cos`/`sin` are `[1, s, head_dim/2]` (broadcast over heads). For each adjacent pair
/// `(x[2k], x[2k+1])`:
///   `out[2k]   = x[2k]·cos_k − x[2k+1]·sin_k`
///   `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k`
/// Computed in f32 (the reference upcasts), then cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let dt = x.dtype();
    let sh = x.shape();
    let (b, s, h, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;

    let cos = cos.as_dtype(Dtype::Float32)?.expand_dims(2)?; // [1, s, 1, half]
    let sin = sin.as_dtype(Dtype::Float32)?.expand_dims(2)?;

    let xr = x.as_dtype(Dtype::Float32)?.reshape(&[b, s, h, half, 2])?;
    let parts = split(&xr, 2, 4)?; // 2 × [b, s, h, half, 1]
    let xe = parts[0].reshape(&[b, s, h, half])?;
    let xo = parts[1].reshape(&[b, s, h, half])?;

    let out_e = subtract(&multiply(&xe, &cos)?, &multiply(&xo, &sin)?)?;
    let out_o = add(&multiply(&xe, &sin)?, &multiply(&xo, &cos)?)?;

    let out = concatenate_axis(&[&out_e.expand_dims(4)?, &out_o.expand_dims(4)?], 4)?;
    Ok(out.reshape(&[b, s, h, hd])?.as_dtype(dt)?)
}
