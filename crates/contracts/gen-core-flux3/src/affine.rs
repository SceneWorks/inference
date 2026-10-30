//! Backend-neutral MLX affine (W4/8, A16) packing for native FLUX3 conversion.
//!
//! Layout is MLX's `quantized_matmul` input: per output row, `k·bits/32` little-endian `u32`
//! words holding consecutive codes from the lowest bits up, plus one BF16 `scale` and `bias` per
//! `group` columns, with `w ≈ scale·q + bias`, `q ∈ [0, 2^bits − 1]`. The per-group parameters
//! follow MLX's affine rule (edge-anchored scale so the larger-magnitude extreme is exact), computed
//! in f32 over the BF16-rounded source values and rounded to BF16 once. This is a correct affine
//! pack for MLX's kernel; byte identity with `mx.quantize` is NOT claimed here — the hosted MLX
//! test compares the two on Apple Silicon.
use gen_core::{Error, Result};

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 affine: {detail}"))
}

/// f32 → bf16 bits (round-to-nearest-even, NaN kept quiet) and back: the shared
/// [`gen_core::numeric`] helpers, re-exported at their historical path (sc-27419).
pub use gen_core::numeric::{bf16_bits, bf16_value};

/// Decode a stored little-endian dense payload to f32. Only original dense float storage.
pub fn decode_dense(dtype: &str, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|b| bf16_value(u16::from_le_bytes([b[0], b[1]])))
            .collect()),
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        other => Err(invalid(format!(
            "source stored as {other}; only original BF16/F32 dense weights are packed"
        ))),
    }
}

/// One packed `[n, k]` projection.
#[derive(Clone, Debug, PartialEq)]
pub struct AffinePacked {
    pub codes: Vec<u32>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
}

fn check(n: usize, k: usize, bits: u32, group: u32) -> Result<()> {
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64 | 128) {
        return Err(invalid("unsupported affine bits/group"));
    }
    if n == 0 || k == 0 || !k.is_multiple_of(group as usize) {
        return Err(invalid(format!("width {k} is not whole groups of {group}")));
    }
    Ok(())
}

/// Pack row-major `values` (`n·k`, already the dense source values) at `bits`/`group`.
pub fn quantize(values: &[f32], n: usize, k: usize, bits: u32, group: u32) -> Result<AffinePacked> {
    check(n, k, bits, group)?;
    if values.len() != n * k {
        return Err(invalid("value count does not match [n, k]"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite source weight; refusing to pack"));
    }
    let g = group as usize;
    let per_word = 32 / bits as usize;
    let n_bins = ((1u32 << bits) - 1) as f32;
    let mut out = AffinePacked {
        codes: vec![0; n * k / per_word],
        scales: Vec::with_capacity(n * k / g),
        biases: Vec::with_capacity(n * k / g),
    };
    for (gi, chunk) in values.chunks_exact(g).enumerate() {
        // Quantize what the bf16 compute weight holds (MLX packs the bf16 master).
        let w: Vec<f32> = chunk.iter().map(|&v| bf16_value(bf16_bits(v))).collect();
        let (min, max) = w
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let anchor_min = min.abs() > max.abs();
        let mut scale = ((max - min) / n_bins).max(1e-7);
        if !anchor_min {
            scale = -scale;
        }
        let edge = if anchor_min { min } else { max };
        let q0 = (edge / scale).round_ties_even();
        if q0 != 0.0 {
            scale = edge / q0;
        }
        let bias = edge;
        // Codes from the rounded (stored) parameters, so dequantization reproduces them.
        let (s, b) = (bf16_value(bf16_bits(scale)), bf16_value(bf16_bits(bias)));
        out.scales.push(bf16_bits(scale));
        out.biases.push(bf16_bits(bias));
        for (j, &v) in w.iter().enumerate() {
            let q = ((v - b) / s).round_ties_even().clamp(0.0, n_bins) as u32;
            let element = gi * g + j;
            out.codes[element / per_word] |= q << ((element % per_word) * bits as usize);
        }
    }
    Ok(out)
}

/// The weights MLX's kernel reconstructs from a pack (the test reference).
pub fn dequantize(
    packed: &AffinePacked,
    n: usize,
    k: usize,
    bits: u32,
    group: u32,
) -> Result<Vec<f32>> {
    check(n, k, bits, group)?;
    let per_word = 32 / bits as usize;
    let mask = (1u32 << bits) - 1;
    Ok((0..n * k)
        .map(|e| {
            let q = (packed.codes[e / per_word] >> ((e % per_word) * bits as usize)) & mask;
            let gi = e / group as usize;
            bf16_value(packed.scales[gi]) * q as f32 + bf16_value(packed.biases[gi])
        })
        .collect())
}

/// Little-endian payloads for the `{stem}.weight` / `.scales` / `.biases` triple.
pub fn to_le_bytes(packed: &AffinePacked) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        packed.codes.iter().flat_map(|w| w.to_le_bytes()).collect(),
        packed.scales.iter().flat_map(|w| w.to_le_bytes()).collect(),
        packed.biases.iter().flat_map(|w| w.to_le_bytes()).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        // 1 + 2^-8 is exactly halfway between bf16(1) and the next value: ties to even (1.0).
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        assert!(bf16_value(bf16_bits(f32::NAN)).is_nan());
    }

    /// *Mutations that red this:* packing codes high-bits-first, dropping the edge anchor (zero
    /// extreme no longer exact), or computing codes from unrounded parameters.
    #[test]
    fn pack_dequantizes_within_half_a_step_and_keeps_the_anchored_extreme() {
        let (n, k) = (3, 128);
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.173).sin() * if i % 7 == 0 { 2.0 } else { 0.5 })
            .collect();
        for bits in [4, 8] {
            for group in [32, 64, 128] {
                let packed = quantize(&values, n, k, bits, group).unwrap();
                assert_eq!(packed.codes.len(), n * k * bits as usize / 32);
                assert_eq!(packed.scales.len(), n * k / group as usize);
                let back = dequantize(&packed, n, k, bits, group).unwrap();
                for (gi, chunk) in values.chunks_exact(group as usize).enumerate() {
                    // Half a step from rounding, up to another half step where the BF16-rounded
                    // scale pushes the far extreme into the clamp, plus the BF16 bias rounding.
                    let step = bf16_value(packed.scales[gi]).abs();
                    let bias = bf16_value(packed.biases[gi]).abs();
                    for (j, &v) in chunk.iter().enumerate() {
                        let v = bf16_value(bf16_bits(v));
                        let got = back[gi * group as usize + j];
                        assert!(
                            (got - v).abs() <= step + 4e-3 * bias.max(v.abs()),
                            "bits {bits} group {group}: {v} -> {got} (step {step})"
                        );
                    }
                }
            }
        }
        // A constant group is exact.
        let flat = vec![0.25f32; 64];
        let back = dequantize(&quantize(&flat, 1, 64, 4, 64).unwrap(), 1, 64, 4, 64).unwrap();
        assert!(back.iter().all(|&v| (v - 0.25).abs() < 1e-6), "{back:?}");
    }

    #[test]
    fn invalid_shapes_and_values_are_refused() {
        assert!(quantize(&[0.0; 96], 1, 96, 4, 64).is_err());
        assert!(quantize(&[0.0; 64], 1, 64, 2, 64).is_err());
        assert!(quantize(&[f32::NAN; 64], 1, 64, 4, 64).is_err());
        assert!(decode_dense("F8_E4M3", &[0]).is_err());
        assert_eq!(
            decode_dense("BF16", &0x3f80u16.to_le_bytes()).unwrap(),
            vec![1.0]
        );
    }
}
