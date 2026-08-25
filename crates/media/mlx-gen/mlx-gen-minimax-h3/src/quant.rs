//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`] (sc-17150).
//!
//! A `q4` / `q8` tier stores each packed Linear as the triple `{base}.weight` (u32 codes) +
//! `{base}.scales` + `{base}.biases`. [`lin`] **auto-detects** it on the presence of `{base}.scales`
//! and builds the quantized base directly, so a published tier loads packed with **no dense bf16
//! transient** — which at 66.28 GB is the difference between loading and being OOM-killed. A dense
//! `bf16` tier has no `.scales` and loads dense through the identical call, so one loader serves
//! every tier and nothing reads a manifest to decide.
//!
//! # Only two loaders route through here, and that is a correctness property
//!
//! [`crate::convert`] leaves the float32 I/O heads, `context_embedder`, `norm_out.linear` and every
//! norm dense **in every tier**. That dense set is exactly what [`crate::dit::heads`] reads with a
//! raw `Weights::require`, so those loaders must *not* gain a packed path: a `require` that met a
//! packed tensor would load u32 codes where a float is expected and report **no error at all**
//! (sc-14980's Mage `pos_embed` failure). The two loaders that do pack —
//! [`crate::dit::layers::LinearNoBias`] and [`crate::dit::block::AdaLnProjection`] — are therefore
//! the complete set, and `tests/quant_policy.rs` holds the converter's predicate and this loader
//! split against one another so they cannot drift apart.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the tiers are packed at — [`crate::convert::GROUP_SIZE`], the codebase-wide 64.
///
/// Read here rather than re-declared: [`mlx_gen::quant::packed_bits`] *infers* the bit width from
/// the weight/scales column ratio **assuming this group size**, so a loader that assumed a different
/// one would derive an illegal width from a perfectly good artifact rather than fail cleanly.
pub const GROUP_SIZE: i32 = crate::convert::GROUP_SIZE;

/// Load `{base}` as an [`AdaptableLinear`] at the tier [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present, else dense. `bias` additionally loads the dense `{base}.bias` (the quantization's own
/// `{base}.biases` is distinct and always loaded on the packed path).
pub fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Device bytes an [`AdaptableLinear`] holds, packed or dense.
///
/// The AdaLN eviction lever (sc-17145) is sized on this, and its headline 26.02 GB is a **bf16**
/// number: on a `q8` tier the same 50 projections are 13.02 GB and on `q4` 6.52 GB. Summing the
/// packed triple rather than reconstructing the logical shape keeps the accounting honest about what
/// is actually resident, which is the whole point of the measurement.
pub fn nbytes(lin: &AdaptableLinear) -> usize {
    if let Some((wq, scales, biases, bias, _, _)) = lin.quantized_params() {
        wq.nbytes() + scales.nbytes() + biases.nbytes() + bias.map_or(0, |b| b.nbytes())
    } else if let Some((weight, bias)) = lin.dense_weight() {
        weight.nbytes() + bias.map_or(0, |b| b.nbytes())
    } else {
        0
    }
}

/// The dtype an [`AdaptableLinear`]'s base computes in.
///
/// A dense base reports its weight dtype. A packed base has no dense weight to ask, so the **scales**
/// carry it — [`mlx_gen::quant::quantize_map`] casts to bf16 before packing, so the scales are bf16
/// and an activation fed in at f32 would mismatch the quantized matmul.
pub fn compute_dtype(lin: &AdaptableLinear) -> mlx_rs::Dtype {
    lin.weight_dtype().unwrap_or_else(|| {
        lin.quantized_params()
            .map(|(_, scales, ..)| scales.dtype())
            .unwrap_or(mlx_rs::Dtype::Bfloat16)
    })
}
