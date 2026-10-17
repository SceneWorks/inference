//! The FLUX3 publisher-format kernels' bit-level CPU mirrors (`candle_quant_kernels::flux3_quant`)
//! against the backend-neutral source oracles (`gen_core::flux3::quantization`). The device
//! kernels are held to these mirrors by the GPU functional test; this half runs on every host.
use candle_gen::gen_core::flux3::quantization::{
    decode_e4m3, encode_e4m3, quantize_fp8_rows, quantize_nvfp4_static,
};
use candle_quant_kernels::flux3_quant as mirror;

fn values(n: usize, salt: f32, spread: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 * 0.618 + salt).sin() * spread;
            if i % 13 == 0 {
                x * 37.0
            } else {
                x
            }
        })
        .collect()
}

/// Every E4M3 grid point, every midpoint between neighbours (the ties) and a dense sweep: the
/// kernel's ln-based grid rounding picks the oracle's nearest-even code everywhere.
///
/// *Mutation that reds this:* `roundf` (half away from zero) in `f3_e4m3_round`.
#[test]
fn e4m3_kernel_rounding_is_the_oracle_encoder_including_ties() {
    let mut grid: Vec<f32> = (0..=126u8).map(decode_e4m3).collect();
    grid.sort_by(f32::total_cmp);
    let mut probes = grid.clone();
    probes.extend(grid.windows(2).map(|w| (w[0] + w[1]) / 2.0));
    probes.extend((0..20_000).map(|i| i as f32 * 0.0237));
    probes.push(1e9);
    for v in probes {
        let kernel = mirror::e4m3_byte_from_decoded(mirror::e4m3_round(v));
        assert_eq!(kernel, encode_e4m3(v).unwrap(), "value {v}");
    }
}

/// *Mutations that red this:* dropping the 1e-12 clamp, the 448 divisor, or the sign bit.
#[test]
fn fp8_rowwise_activation_mirror_is_the_source_quantizer() {
    for (width, spread) in [(64, 1.0), (3072, 0.01), (96, 300.0)] {
        let rows = values(width * 3, 0.7, spread);
        let (want_codes, want_scales) = quantize_fp8_rows(&rows, width).unwrap();
        for (r, row) in rows.chunks_exact(width).enumerate() {
            let (codes, scale) = mirror::fp8_rowwise_row(row);
            assert_eq!(scale, want_scales[r]);
            assert_eq!(codes, want_codes[r * width..(r + 1) * width]);
        }
    }
    let (zero_codes, zero_scale) = mirror::fp8_rowwise_row(&[0.0; 32]);
    assert_eq!(zero_scale, 1e-12);
    assert!(zero_codes.iter().all(|c| *c == 0));
}

/// Static NVFP4: identical packed nibbles AND swizzled scales to the source oracle, including a
/// zero calibrated scale (clamped reciprocal) and padding atoms.
///
/// *Mutations that red this:* using the raw (unclamped) scale in the quantizer, writing the high
/// nibble first, or the transposed atom order.
#[test]
fn nvfp4_static_activation_mirror_is_the_source_quantizer() {
    for (m, k, input_scale) in [
        (3, 64, 0.02f32),
        (130, 32, 1.5),
        (5, 48, 0.0),
        (2, 16, 1e-12),
    ] {
        let x = values(m * k, 1.3, 2.0);
        let want = quantize_nvfp4_static(&x, k, input_scale).unwrap();
        let (packed, scales) = mirror::nvfp4_static_pack(&x, m, k, input_scale);
        assert_eq!(packed, want.packed, "m{m} k{k} s{input_scale}");
        assert_eq!(scales, want.scales, "m{m} k{k} s{input_scale}");
        assert_eq!(
            want.global_scale, input_scale,
            "alpha keeps the RAW calibrated scale"
        );
    }
}

#[test]
fn weight_scale_swizzle_is_the_source_layout() {
    let (rows, blocks) = (130, 6);
    let raw: Vec<u8> = (0..rows * blocks).map(|i| (i % 251) as u8).collect();
    assert_eq!(
        mirror::swizzle_scales(&raw, rows, blocks),
        candle_gen::gen_core::flux3::quantization::swizzle_scales(&raw, rows, blocks).unwrap()
    );
}

#[test]
fn bf16_epilogue_rounding_is_the_shared_rule() {
    for v in values(4096, 0.1, 50.0) {
        assert_eq!(
            mirror::bf16_bits(v),
            candle_gen::gen_core::flux3::affine::bf16_bits(v)
        );
    }
    assert_eq!(
        mirror::fp8_rowwise_epilogue(2.0, 0.5, 3.0),
        mirror::bf16_bits(3.0)
    );
}
