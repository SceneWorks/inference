//! Quantization — group-wise affine Q4/Q8, the mlx-rs equivalent of the Python mflux fork's
//! `nn.quantize(model, bits=bits)` path. mflux never passes `group_size`, so it uses MLX's
//! default of **64**. The actual quantization seam is
//! [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize), which quantizes each
//! `Linear` base in place **with the bf16-parity cast** the fork goldens require — providers route
//! through it (or their per-family loaders), so this module owns only the shared default below.
//!
//! Byte-level packing parity vs the fork (mlx 0.31) is checked in `tests/quant_parity.rs` —
//! the version-drift risk, since the crate links an older bundled MLX (mlx-rs 0.25).
//!
//! This module also owns the **group-size-parametric** packed-weight loaders ([`lin`] / [`embedding`])
//! and the **offline pre-quantization** converter primitives ([`load_dir_map`] / [`save_map`] /
//! [`quantize_map`]) that the provider crates used to copy verbatim (only their `GROUP_SIZE` differed:
//! 64 codebase-default, 32 for Boogu's `3360 = 32·105` DiT). Each provider keeps its own `GROUP_SIZE`
//! constant (and its target predicate) and passes them in (6935).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::adapters::AdaptableLinear;
use crate::nn::TokenEmbedding;
use crate::weights::Weights;
use crate::Result;

/// MLX's default quantization group size; mflux relies on it (never overrides). Used by the real
/// quantization seams ([`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) and
/// the Kolors ChatGLM3 quantizer).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Derive the quant bit-width from the packed shapes at group size `group_size`: `scales` is
/// `[out, in/gs]` ⇒ `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒
/// `bits = wq.cols·32/in`. Exact for any group-aligned Q4/Q8 pack, so the bit-width need not be
/// carried in a side manifest.
fn packed_bits(wq: &Array, scales: &Array, group_size: i32) -> i32 {
    let in_dim = scales.shape()[1] * group_size;
    wq.shape()[1] * 32 / in_dim
}

/// Load `{base}` as an [`AdaptableLinear`] — **packed** (Q4/Q8) when `{base}.scales` is present (a
/// pre-quantized snapshot, bit-width inferred from the shapes at `group_size`), else **dense**. `bias`
/// additionally loads the dense `{base}.bias` (the quantization's own `{base}.biases` is distinct and
/// always loaded on the packed path). Auto-detection means one loader serves both a dense bf16 and a
/// pre-quantized snapshot with no `quantization` manifest to read. Shared by the provider crates'
/// packed loaders — each passes its own `group_size`.
pub fn lin(w: &Weights, base: &str, bias: bool, group_size: i32) -> Result<AdaptableLinear> {
    let bias = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size);
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            group_size,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]).
pub fn embedding(w: &Weights, base: &str, group_size: i32) -> Result<TokenEmbedding> {
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size);
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            group_size,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}

/// Read every tensor of `dir` (possibly sharded safetensors) into an owned key→`Array` map (MLX
/// arrays are ref-counted, so the clone is a handle copy, not a buffer copy). The read side of the
/// offline pre-quantization converters.
pub fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize (`eval`) + write a key→`Array` map to a single `path` safetensors (one file — a packed
/// component is small enough not to need sharding; the loaders glob `*.safetensors`, so one file
/// replaces the source's shards). The write side of the converters.
pub fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Pack every `is_target` `{base}.weight` (group-quantizable 2-D, `in % group_size == 0`,
/// `in >= group_size`) into the triple `{base}.weight` (u32 codes) + `.scales` + `.biases` — the
/// weight cast to **bf16 first** so the pack is byte-identical to the load-time
/// [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) (and the fork's
/// `nn.quantize(bf16)`). Every other tensor (norms, 1-D, non-divisible, non-target) passes through
/// unchanged. `is_target` receives the **base** (the key minus its `.weight` suffix). The shape guard
/// keeps an odd-shaped or 1-D target dense rather than crashing `quantize`.
pub fn quantize_map(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight").filter(|b| is_target(b));
        let packable = base.is_some()
            && v.shape().len() == 2
            && v.shape()[1] % group_size == 0
            && v.shape()[1] >= group_size;
        if let (Some(base), true) = (base, packable) {
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}
