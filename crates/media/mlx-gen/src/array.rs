//! Small host-readback helpers for integer index/mask arrays.
//!
//! [`mlx_rs::Array::as_slice`] is `try_as_slice().unwrap()`, which **panics** on both a dtype
//! mismatch and a size-0 array. At public boundaries that take caller-supplied ids/masks — which HF
//! tokenizers commonly emit as i64 — that turns a recoverable condition into a process abort.
//! [`host_i32`] reads such an array into a host `Vec<i32>`, converting the dtype when needed and
//! returning a typed [`Error`] instead.

use mlx_rs::{Array, Dtype};

use crate::{Error, Result};

/// Read an integer array (token ids / position ids / attention mask) into a host `Vec<i32>`.
///
/// Unlike `array.as_slice::<i32>()`, this returns an [`Error`] rather than panicking on a dtype
/// mismatch, and accepts any integer dtype by converting to int32 first (so an i64 HF mask is read,
/// not rejected). A size-0 array reads as an empty `Vec`.
pub fn host_i32(a: &Array) -> Result<Vec<i32>> {
    // A size-0 array has no data pointer to borrow (`try_as_slice` returns `Null`); treat an empty
    // input as an empty result rather than an error.
    if a.size() == 0 {
        return Ok(Vec::new());
    }
    let a = a.as_dtype(Dtype::Int32)?;
    a.try_as_slice::<i32>()
        .map(<[i32]>::to_vec)
        .map_err(|e| Error::Msg(format!("host_i32: not a readable int array: {e}")))
}

/// A 1-element `[1]` f32 array — the idiomatic way to lift a host scalar into a broadcastable
/// constant for elementwise ops (`x * scalar(0.5)`).
pub fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a **logically-contiguous** copy so a subsequent [`Array::as_slice`] reads logical (C-)order
/// rather than the physical buffer of a transpose-strided array. A reshape round-trip materializes it.
///
/// **Over-`i32::MAX` safe (sc-12748).** MLX rejects any single tensor dimension outside the `i32`
/// range (`check_shape_dim`, #3524), so the old `reshape(&[-1])` RAISES once the flattened dim crosses
/// the bound (a 1280²×441f VAE output is 2.168e9 elements). This instead flattens to a **2-D** `[a, b]`
/// whose factors are both ≤ `i32::MAX` — a contiguous regrouping of the existing dims, so the row-major
/// bytes are unchanged — then restores the shape. A multi-dim reshape whose *total* exceeds `i32::MAX`
/// while every *dimension* stays within it is int64-safe on this pin (probe-verified in
/// `mlx-gen/tests/mlx_write_bound_probe.rs::reshape_and_contiguous_on_oversized_array`). Below the bound
/// `a == total, b == 1`, byte-identical to the old round-trip.
pub fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    let (a, b) = flat_2d(&shape);
    Ok(x.reshape(&[a, b])?.reshape(&shape)?)
}

/// Split `shape` into two contiguous dimension groups `[a, b]` (`a·b == Π shape`) with each factor
/// ≤ `i32::MAX`. Greedily accumulate leading dims into `a` until the next would overflow `i32`; `b` is
/// the product of the rest. Every real tensor has total ≪ `i32::MAX²` and each single dim already
/// ≤ `i32::MAX` (or it could not exist), so both factors fit. Empty/scalar shapes yield `[1, 1]`.
fn flat_2d(shape: &[i32]) -> (i32, i32) {
    let total: i64 = shape.iter().map(|&d| d as i64).product::<i64>().max(1);
    let mut a: i64 = 1;
    for &d in shape {
        let d = d as i64;
        if a.saturating_mul(d) > i32::MAX as i64 {
            break;
        }
        a *= d;
    }
    let a = a.max(1);
    let b = (total / a).max(1);
    // sc-12926: if no leading group fits (e.g. a hypothetical [46341, 46341, 46341]), `b` overflows
    // i32 and the cast below would silently wrap — turn that into a loud debug failure instead.
    // Unreachable for any real VAE shape (their totals split comfortably), and release builds would
    // still RAISE downstream (MLX rejects the wrapped dim in `reshape`), not corrupt.
    debug_assert!(
        b <= i32::MAX as i64,
        "flat_2d: residual factor {b} > i32::MAX — shape {shape:?} has no contiguous 2-D split \
         with both factors in i32 range"
    );
    (a as i32, b as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_2d_splits_stay_within_i32() {
        // Below the bound: a == total, b == 1 (byte-identical to reshape(-1) round-trip).
        assert_eq!(flat_2d(&[2, 3, 4]), (24, 1));
        // The 1280²×441f VAE output (2.168e9 > i32::MAX): both factors ≤ i32::MAX and multiply back.
        let (a, b) = flat_2d(&[1, 3, 441, 1280, 1280]);
        assert!(a as i64 <= i32::MAX as i64 && b as i64 <= i32::MAX as i64);
        assert_eq!(a as i64 * b as i64, 3i64 * 441 * 1280 * 1280);
        // A Wan 720p×800f output (3·800·1280·720 = 2.21e9): still splits within i32.
        let (a, b) = flat_2d(&[3, 800, 1280, 720]);
        assert!(a as i64 <= i32::MAX as i64 && b as i64 <= i32::MAX as i64);
        assert_eq!(a as i64 * b as i64, 3i64 * 800 * 1280 * 720);
    }

    /// sc-12926: the `b ≤ i32::MAX` debug_assert fires when no contiguous 2-group split exists —
    /// here every grouping of `[46341, 46341, 46341]` leaves one factor over the bound. Debug-only
    /// (tests run in debug), and mutation-discriminating: deleting the assert makes this pass a
    /// silently-wrapped cast instead of panicking.
    #[test]
    #[should_panic(expected = "flat_2d: residual factor")]
    #[cfg(debug_assertions)]
    fn flat_2d_unsplittable_shape_fails_loudly() {
        // 46341² = 2_147_488_281 > i32::MAX, so a = 46341 and b = 46341² overflows the bound.
        let _ = flat_2d(&[46_341, 46_341, 46_341]);
    }

    #[test]
    fn reads_i32_array() {
        let a = Array::from_slice(&[1i32, 2, 3], &[3]);
        assert_eq!(host_i32(&a).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn converts_i64_without_panicking() {
        // HF tokenizers commonly emit i64 masks; `as_slice::<i32>()` would panic here.
        let a = Array::from_slice(&[1i64, 0, 1], &[3]);
        assert_eq!(host_i32(&a).unwrap(), vec![1, 0, 1]);
    }

    #[test]
    fn empty_array_reads_as_empty_vec() {
        // A `[1, 0]` empty prompt array: `as_slice` would panic (`Null`); we return an empty Vec.
        let a = Array::from_slice::<i32>(&[], &[1, 0]);
        assert_eq!(host_i32(&a).unwrap(), Vec::<i32>::new());
    }
}
