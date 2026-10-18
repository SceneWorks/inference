//! FLUX3 publisher-format kernels (sc-26241): the frozen BFL FP8R rowwise W8A8 and static-scale
//! NVFP4 W4A4 activation recipes, plus the scale swizzle their weights are staged with.
//!
//! The device code is `flux3_quant.cu` (nvrtc). Everything in this file outside `cuda` builds on
//! every host: [`FLUX3_QUANT_SRC`] (so the workspace registry names it) and **bit-level CPU mirrors**
//! of each device helper, transcribed operation for operation. Consumers test the mirrors against
//! the backend-neutral oracles in `gen_core::flux3::quantization`; the device kernels are then held
//! to the mirrors by the GPU functional test (one admitted tiny-kernel slot), so a kernel that
//! drifts from the source recipe is caught twice, never assumed.
//!
//! The wrappers that launch these kernels live on `crate::cublaslt::CublasLt` (CUDA builds only)
//! (`flux3_fp8_rowwise_linear`, `flux3_stage_nvfp4_weight`, `flux3_nvfp4_static_linear`).

/// The FLUX3 activation kernels. FP8 E4M3 GEMMs need sm_89; the floor here is the source's own
/// (standard intrinsics only). Callers gate FP8/FP4 compute on the device floors separately.
pub const FLUX3_QUANT_SRC: crate::nvrtc::KernelSource = crate::nvrtc::KernelSource {
    name: "candle_quant_kernels_flux3_quant_v1",
    src: include_str!("flux3_quant.cu"),
    cc_floor: (7, 0),
};

const LN2: f32 = std::f32::consts::LN_2;
const INV_LN2: f32 = std::f32::consts::LOG2_E;

/// `f3_e2m1_code`: nearest E2M1 with even-index ties, sign in bit 3, saturating at 6.
pub fn e2m1_code(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let sign = if v.is_sign_negative() { 0x8 } else { 0 };
    let mag = v.abs().min(6.0);
    let idx = if mag <= 0.25 {
        0
    } else if mag < 0.75 {
        1
    } else if mag <= 1.25 {
        2
    } else if mag < 1.75 {
        3
    } else if mag <= 2.5 {
        4
    } else if mag < 3.5 {
        5
    } else if mag <= 5.0 {
        6
    } else {
        7
    };
    sign | idx
}

/// `f3_e4m3_round`: the non-negative E4M3 grid value nearest `v` (ties to even, saturating 448).
pub fn e4m3_round(v: f32) -> f32 {
    let v = v.clamp(0.0, 448.0);
    let vc = v.clamp(0.015625, 448.0);
    let e = (vc.ln() * INV_LN2).floor();
    let ulp = ((e - 3.0) * LN2).exp();
    let q = (v / ulp).round_ties_even() * ulp;
    q.clamp(0.0, 448.0)
}

/// `f3_e4m3_byte_from_decoded`: the E4M3 byte (0..=254) whose decode is the on-grid `q`.
pub fn e4m3_byte_from_decoded(q: f32) -> u8 {
    if q < (1.0 / 512.0) * 0.5 {
        return 0;
    }
    if q < 0.015625 {
        return (q * 512.0).round().clamp(0.0, 254.0) as u8;
    }
    let e = (q.ln() * INV_LN2).floor();
    let pow_e = (e * LN2).exp();
    let mant = (q / pow_e * 8.0).round() as i32;
    (e as i32 * 8 + 48 + mant).clamp(0, 254) as u8
}

/// `f3_bf16_bits`: f32 → bf16 bits, nearest-even, NaN kept quiet.
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

/// `flux3_fp8_rowwise_quantize_f32` for one row: the row scale and the E4M3 BYTES of the on-grid
/// values the kernel writes (the device writes the decoded value; its F8E4M3 cast is exact).
pub fn fp8_rowwise_row(row: &[f32]) -> (Vec<u8>, f32) {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = (amax / 448.0).max(1e-12);
    let codes = row
        .iter()
        .map(|&x| {
            let v = x / scale;
            let byte = e4m3_byte_from_decoded(e4m3_round(v.abs()));
            if v.is_sign_negative() {
                byte | 0x80
            } else {
                byte
            }
        })
        .collect();
    (codes, scale)
}

/// `flux3_fp8_rowwise_epilogue` for one element.
pub fn fp8_rowwise_epilogue(acc: f32, row_scale: f32, weight_scale: f32) -> u16 {
    bf16_bits(acc * row_scale * weight_scale)
}

/// FlashInfer / cuBLASLt 128×4 scale-factor atom offset (K atoms fastest).
pub fn scale_offset(row: usize, block: usize, sf_cols: usize) -> usize {
    let atom = (row / 128) * (sf_cols / 4) + block / 4;
    atom * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + block % 4
}

/// Row-major `[rows, blocks]` scale bytes into the zero-padded compute atom layout.
pub fn swizzle_scales(values: &[u8], rows: usize, blocks: usize) -> Vec<u8> {
    let sf_rows = rows.div_ceil(128) * 128;
    let sf_cols = blocks.div_ceil(4) * 4;
    let mut out = vec![0u8; sf_rows * sf_cols];
    for r in 0..rows {
        for b in 0..blocks {
            out[scale_offset(r, b, sf_cols)] = values[r * blocks + b];
        }
    }
    out
}

/// `flux3_nvfp4_static_pack_f32` over a full `[m, k]` activation (`k % 16 == 0`): packed nibbles
/// `[m, k/2]` and swizzled scale bytes, with the clamped reciprocal `max(input_scale, 1e-8)`.
pub fn nvfp4_static_pack(x: &[f32], m: usize, k: usize, input_scale: f32) -> (Vec<u8>, Vec<u8>) {
    assert!(
        k.is_multiple_of(16) && x.len() == m * k,
        "whole 16-element blocks"
    );
    let qscale = input_scale.max(1e-8);
    let n_blocks = k / 16;
    let sf_cols = n_blocks.div_ceil(4) * 4;
    let mut packed = vec![0u8; m * k / 2];
    let mut scales = vec![0u8; m.div_ceil(128) * 128 * sf_cols];
    for r in 0..m {
        for blk in 0..n_blocks {
            let block = &x[r * k + blk * 16..r * k + blk * 16 + 16];
            let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
            let decoded = e4m3_round((amax / (6.0 * qscale)).min(448.0));
            let code = e4m3_byte_from_decoded(decoded);
            let divisor = decoded * qscale;
            for j in 0..8 {
                let (lo, hi) = if divisor != 0.0 {
                    (
                        e2m1_code(block[2 * j] / divisor),
                        e2m1_code(block[2 * j + 1] / divisor),
                    )
                } else {
                    (0, 0)
                };
                packed[r * k / 2 + blk * 8 + j] = lo | (hi << 4);
            }
            scales[scale_offset(r, blk, sf_cols)] = code;
        }
    }
    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_is_registered_and_declares_every_entry_point() {
        assert!(crate::NVRTC_SOURCES.contains(&FLUX3_QUANT_SRC));
        for entry in [
            "flux3_fp8_rowwise_quantize_f32",
            "flux3_fp8_rowwise_epilogue",
            "flux3_nvfp4_static_pack_f32",
        ] {
            assert!(
                FLUX3_QUANT_SRC
                    .src
                    .contains(&format!("__global__ void {entry}(")),
                "{entry}"
            );
        }
    }

    #[test]
    fn e4m3_grid_rounding_ties_to_even_and_saturates() {
        assert_eq!(
            e4m3_round(8.5),
            8.0,
            "tie between 8 and 9 goes to the even mantissa"
        );
        assert_eq!(e4m3_round(9.5), 10.0);
        assert_eq!(e4m3_round(1e6), 448.0);
        assert_eq!(e4m3_byte_from_decoded(448.0), 0x7e);
        assert_eq!(e4m3_byte_from_decoded(1.0), 0x38);
        assert_eq!(e4m3_byte_from_decoded(0.0), 0);
        assert_eq!(e4m3_byte_from_decoded(1.0 / 512.0), 1, "smallest subnormal");
        assert_eq!(bf16_bits(1.0), 0x3f80);
    }

    #[test]
    fn swizzle_matches_the_documented_atom_formula() {
        // Row 33, block 5 of a 2-atom-wide grid: atom (0, 1), intra (33%32)*16 + (33/32)*4 + 1.
        assert_eq!(scale_offset(33, 5, 8), 512 + 16 + 4 + 1);
        let s = swizzle_scales(&[1, 2, 3, 4, 5], 1, 5);
        assert_eq!(s.len(), 128 * 8);
        assert_eq!(s[scale_offset(0, 4, 8)], 5);
    }
}
