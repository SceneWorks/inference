//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear / embedding as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The [`lin`] / [`embedding`]
//! loaders **auto-detect** it by the presence of `{base}.scales` (no `quantization` manifest to
//! read) and build the quantized module directly — so a published Q4 snapshot loads packed with no
//! dense bf16 transient and is ~¼ the on-disk size. A dense snapshot (no `.scales`) loads dense
//! exactly as before, so the same loaders serve both.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::Array;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Derive the quant bit-width from the packed shapes (group size [`GROUP_SIZE`]): `scales` is
/// `[out, in/gs]` ⇒ `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒
/// `bits = wq.cols·32/in`. Exact for any group-aligned Q4/Q8 pack, so the bit-width need not be
/// carried in a side manifest.
fn packed_bits(wq: &Array, scales: &Array) -> i32 {
    let in_dim = scales.shape()[1] * GROUP_SIZE;
    wq.shape()[1] * 32 / in_dim
}

/// Load `{base}` as an [`AdaptableLinear`] — packed when `{base}.scales` is present (a pre-quantized
/// snapshot), else dense. `bias` additionally loads the dense `{base}.bias` (the quantization's own
/// `{base}.biases` is distinct and always loaded on the packed path).
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    let bias = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales);
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense.
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales);
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}
