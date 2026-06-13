//! Shared SAM3 leaf helpers (F-071): weight-key joining and torch→MLX conv-weight permutes, lifted
//! out of the per-module copies (mirrors `mlx-gen-sam2`'s `util`). The `join` here is the
//! empty-prefix-aware variant — a superset of the plain `format!("{prefix}.{leaf}")` copies that were
//! scattered across the modules: identical for every non-empty prefix, and it returns the bare leaf
//! (rather than a malformed `.leaf`) when the prefix is empty.

use mlx_rs::Array;

use mlx_gen::Result;

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// Permute a torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_w_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Permute a torch transposed-conv weight `[in, out, kH, kW]` (IOHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_transpose_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 3, 0])?)
}
