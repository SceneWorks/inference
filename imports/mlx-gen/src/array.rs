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

#[cfg(test)]
mod tests {
    use super::*;

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
