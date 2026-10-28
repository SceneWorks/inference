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

/// Multiply `a` by a host `f64` rounded to f32 — the reference's `python_float * mx.array(f32)`,
/// where the weak scalar adopts the array's dtype. The Wan and Krea-Realtime flow schedulers each
/// carried this as a private `fscale` (sc-27448).
pub fn fscale(a: &Array, c: f64) -> Result<Array> {
    Ok(mlx_rs::ops::multiply(a, Array::from_f32(c as f32))?)
}

/// The half-open range `x[.., start..end, ..]` along `axis` (negative counts from the end), as an
/// arange-index `take_axis` gather — the copy the LTX / Wan / MiniMax-H3 / SAM3 / Boogu VAE and
/// sequence helpers each open-coded (sc-27448). The whole axis returns `x` itself (a gather of
/// every index is an exact copy). An out-of-range axis or range is an [`Error`], never a gather
/// past the end.
///
/// **`end`, not `len`:** unlike [`crate::attention::slice_axis`] (`start`, `len`, a zero-copy
/// split), the third argument here is the exclusive end index.
pub fn slice_range(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let rank = x.ndim() as i32;
    let ax = if axis < 0 { axis + rank } else { axis };
    if ax < 0 || ax >= rank {
        return Err(Error::Msg(format!(
            "slice_range: axis {axis} out of range for rank {rank}"
        )));
    }
    let len = x.shape()[ax as usize];
    if start < 0 || end > len || start > end {
        return Err(Error::Msg(format!(
            "slice_range: [{start}, {end}) out of range for axis {ax} of length {len}"
        )));
    }
    if start == 0 && end == len {
        return Ok(x.clone());
    }
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), ax)?)
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
///
/// **Always row-major (sc-20676).** Above the bound the `[a, b]` regrouping can be a pure view: a
/// permutation that swaps whole dim groups (e.g. a transposed `[X, Y]` split at `a = X`) reshapes
/// without a copy and would keep its permuted strides. A `b > 1` result therefore goes through
/// MLX's own `contiguous` op (`mlx_contiguous`), which copies to row-major unless it already is.
pub fn contiguous(x: &Array) -> Result<Array> {
    let (a, b) = flat_2d(x.shape());
    contiguous_via_split(x, a, b)
}

/// [`contiguous`] with an explicit `[a, b]` regrouping (so tests can force a `b > 1` split at a
/// small size).
fn contiguous_via_split(x: &Array, a: i32, b: i32) -> Result<Array> {
    let shape = x.shape().to_vec();
    let regrouped = x.reshape(&[a, b])?.reshape(&shape)?;
    if b == 1 && shape != [a, 1] {
        // `[total, 1]` merges every dim, which MLX can only do without a copy for data already in
        // row-major order: the round-trip is logically contiguous. (A `[total, 1]` input is
        // returned by `reshape` unchanged, strides and all, so it takes the explicit copy below.)
        return Ok(regrouped);
    }
    // Lazy: at evaluation MLX's `Contiguous` shares an already row-contiguous buffer and copies
    // anything else.
    mlx_contiguous(&regrouped)
}

/// Whether an **evaluated** `x`'s strides are the row-major strides of its shape (size-1 dims
/// ignored). An unevaluated array reports default row-major strides.
#[cfg(test)]
fn is_row_major(x: &Array) -> bool {
    let mut expected = 1usize;
    for (&dim, &stride) in x.shape().iter().zip(x.strides()).rev() {
        if dim != 1 && stride != expected {
            return false;
        }
        expected *= dim.max(1) as usize;
    }
    true
}

/// MLX's `contiguous` op (not wrapped by mlx-rs): a row-major copy unless already row-contiguous.
fn mlx_contiguous(x: &Array) -> Result<Array> {
    let stream = mlx_rs::StreamOrDevice::default();
    // SAFETY: on success `mlx_contiguous` writes a new owned handle into `result`, which
    // `Array::from_ptr` takes ownership of; on failure the empty handle is freed here.
    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status =
            mlx_sys::mlx_contiguous(&mut result, x.as_ptr(), false, stream.as_ref().as_ptr());
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Error::Msg(
                "mlx_contiguous failed while making a row-major copy".into(),
            ));
        }
        Ok(Array::from_ptr(result))
    }
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

/// Whether `a` has been evaluated (its buffer is materialized, so it no longer keeps its lazy graph
/// inputs alive). Used to prove retained caches hold only their own bytes.
pub fn is_materialized(a: &Array) -> bool {
    let mut available = false;
    // SAFETY: `_mlx_array_is_available` writes one bool through a valid pointer for a live array.
    unsafe { mlx_sys::_mlx_array_is_available(&mut available, a.as_ptr()) };
    available
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_range_takes_half_open_ranges_and_rejects_out_of_range() {
        let x = Array::from_slice(&(0..10).map(|v| v as f32).collect::<Vec<_>>(), &[2, 5]);
        let mid = slice_range(&x, 1, 1, 4).unwrap();
        assert_eq!(mid.shape(), &[2, 3]);
        // `take_axis` may return a strided view: read logical order through `contiguous`.
        let read = |a: &Array| contiguous(a).unwrap().as_slice::<f32>().to_vec();
        assert_eq!(read(&mid), [1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
        assert_eq!(read(&slice_range(&x, -1, 4, 5).unwrap()), [4.0, 9.0]);
        assert_eq!(slice_range(&x, 0, 0, 2).unwrap().shape(), &[2, 5]);
        assert_eq!(slice_range(&x, 1, 2, 2).unwrap().shape(), &[2, 0]);
        assert!(slice_range(&x, 1, 0, 9).is_err());
        assert!(slice_range(&x, 1, 4, 2).is_err());
        assert!(slice_range(&x, 1, -1, 2).is_err());
        assert!(slice_range(&x, 2, 0, 1).is_err());
        assert!(slice_range(&x, -3, 0, 1).is_err());
    }

    #[test]
    fn fscale_rounds_the_host_scalar_to_f32() {
        let x = Array::from_slice(&[2.0f32, -1.0], &[2]);
        let y = fscale(&x, 0.1).unwrap();
        assert_eq!(y.dtype(), Dtype::Float32);
        assert_eq!(y.as_slice::<f32>(), &[2.0 * 0.1f32, -0.1f32]);
    }

    /// sc-20676: a `b > 1` regrouping of a whole-group permutation is a pure view (the transpose of
    /// `[6, 4]` split as `[4, 6]` reshapes without a copy), so the reshape round-trip alone would
    /// leave it permuted and `as_slice` would read physical order. `contiguous` must still return
    /// logical row-major values.
    #[test]
    fn contiguous_copies_a_permuted_view_the_regrouping_leaves_uncopied() {
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let t = Array::from_slice(&data, &[6, 4])
            .transpose_axes(&[1, 0])
            .unwrap(); // [4, 6]
        let view = t.reshape(&[4, 6]).unwrap().reshape(&[4, 6]).unwrap();
        view.eval().unwrap();
        assert!(
            !is_row_major(&view),
            "precondition: the regrouping is a view"
        );
        let logical: Vec<f32> = (0..4)
            .flat_map(|i| (0..6).map(move |j| (j * 4 + i) as f32))
            .collect();
        let out = contiguous_via_split(&t, 4, 6).unwrap();
        out.eval().unwrap();
        assert!(is_row_major(&out));
        assert_eq!(out.as_slice::<f32>(), logical.as_slice());
        assert_eq!(
            contiguous(&t).unwrap().as_slice::<f32>(),
            logical.as_slice()
        );
    }

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
