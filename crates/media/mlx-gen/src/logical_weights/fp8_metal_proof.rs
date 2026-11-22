use gen_core::comfy_quant::ComfyQuantFormat;
use gen_core::comfy_quant::{
    fp8_e4m3fn_to_f32, fp8_e5m2_to_f32, mxfp8_padded_shape, mxfp8_scale_shape,
    mxfp8_swizzled_scale_index,
};
use mlx_layers::fp8::Fp8Linear;
use mlx_rs::{Array, Dtype};

fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff0000)
}
fn codebook(format: ComfyQuantFormat, b: u8) -> f32 {
    if format == ComfyQuantFormat::Float8E5M2 {
        fp8_e5m2_to_f32(b)
    } else {
        fp8_e4m3fn_to_f32(b)
    }
}
// Independent nearest-neighbour rounding over the finite producer codebook. Equal distances
// choose the even low significand bit. No activation quantizer implementation is reused here.
fn nearest(v: f32, format: ComfyQuantFormat) -> f32 {
    let mag = v.abs();
    let max_code = if format == ComfyQuantFormat::Float8E5M2 {
        0x7b
    } else {
        0x7e
    };
    let selected = (0..=max_code)
        .min_by(|&a, &b| {
            (codebook(format, a) - mag)
                .abs()
                .total_cmp(&(codebook(format, b) - mag).abs())
                .then_with(|| (a & 1).cmp(&(b & 1)))
        })
        .unwrap();
    codebook(format, selected).copysign(v)
}
fn ordered_bf16(v: f32) -> i32 {
    let b = (v.to_bits() >> 16) as i32;
    if b & 0x8000 != 0 {
        0x8000 - (b & 0x7fff)
    } else {
        0x8000 + b
    }
}

fn check_weight_only_scalar_projection() {
    use ComfyQuantFormat::*;

    for (format, unit, non_fp8) in [
        (Float8E4M3Fn, 0x38_u8, 1.0625_f32),
        (Float8E5M2, 0x3c_u8, 1.125_f32),
    ] {
        // The exact midpoint is representable in every compute dtype, but FP8 rounds it
        // down to one. A raw unit-scale source must preserve it through the projection.
        let raw = Fp8Linear::from_source(
            format,
            Array::from_slice(&[unit], &[1, 1]),
            None,
            None,
            &[1, 1],
            None,
            true,
        )
        .unwrap();
        let calibrated =
            Fp8Linear::new(format, raw.codes().clone(), None, None, &[1, 1], None).unwrap();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let x = Array::from_slice(&[non_fp8], &[1, 1])
                .as_dtype(dtype)
                .unwrap();
            let actual = raw.forward(&x).unwrap();
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>(),
                &[non_fp8]
            );
            let rounded = calibrated
                .forward(&x)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            assert_eq!(rounded.as_slice::<f32>(), &[1.]);
        }

        // Multiple K tiles and partial M/N tiles with independent inputs, weight codes,
        // and a bias. The dense oracle rounds the scaled coefficient to the compute dtype
        // before matmul and adds bias only after the output cast.
        const M: usize = 17;
        const N: usize = 19;
        const K: usize = 67;
        let codes: Vec<_> = (0..N * K)
            .map(|i| unit + ((i / K * 5 + i % K * 3) % 7) as u8)
            .collect();
        let values: Vec<_> = (0..M * K)
            .map(|i| ((i * 7 + 5) % 17 + 1) as f32 / 16.)
            .collect();
        let biases: Vec<_> = (0..N).map(|i| (i % 5) as f32 / 16. - 0.125).collect();
        let sw = 1. + 3. / 1024.;
        let weight: Vec<_> = codes
            .iter()
            .map(|&byte| codebook(format, byte) * sw)
            .collect();
        let weight = Array::from_slice(&weight, &[N as i32, K as i32]);
        let bias = Array::from_slice(&biases, &[N as i32]);
        let op = Fp8Linear::from_source(
            format,
            Array::from_slice(&codes, &[N as i32, K as i32]),
            Some(Array::from_f32(sw)),
            Some(Array::from_f32(0.3)),
            &[N, K],
            Some(bias.clone()),
            true,
        )
        .unwrap();
        assert_eq!(op.codes().dtype(), Dtype::Uint8);
        assert_eq!(op.codes().nbytes(), N * K);
        assert_eq!(op.scales().as_slice::<f32>(), &[sw]);
        assert_eq!(op.input_scale().as_slice::<f32>(), &[0.3]);
        for (dtype, relative) in [
            (Dtype::Float32, 1e-6),
            (Dtype::Float16, 1. / 1024.),
            (Dtype::Bfloat16, 1. / 128.),
        ] {
            let input = Array::from_slice(&values, &[M as i32, K as i32])
                .as_dtype(dtype)
                .unwrap();
            let rounded = weight.as_dtype(dtype).unwrap().transpose().unwrap();
            let expected = mlx_rs::ops::matmul(&input, &rounded).unwrap();
            let expected = mlx_rs::ops::add(&expected, bias.as_dtype(dtype).unwrap())
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let actual = op.forward(&input).unwrap();
            assert_eq!(actual.shape(), [M as i32, N as i32]);
            assert_eq!(actual.dtype(), dtype);
            let actual = actual.as_dtype(Dtype::Float32).unwrap();
            for (&got, &want) in actual
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
            {
                assert!(
                    (got - want).abs() <= relative * want.abs().max(1.),
                    "{format:?} {dtype:?}: {got} vs {want}"
                );
            }
        }
    }

    // The BF16-cast weight subnormal makes the exact accumulator a rounding midpoint.
    // Applying the scale after matmul would push the output above that midpoint.
    let op = Fp8Linear::from_source(
        Float8E4M3Fn,
        Array::from_slice(&[0x38_u8, 0x02], &[1, 2]),
        Some(Array::from_f32(1. + 1. / 512.)),
        None,
        &[1, 2],
        Some(Array::from_slice(&[-1.0_f32], &[1])),
        true,
    )
    .unwrap();
    let x = Array::from_slice(&[1.0_f32, 1.0], &[1, 2])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let y = op.forward(&x).unwrap().as_dtype(Dtype::Float32).unwrap();
    assert_eq!(y.as_slice::<f32>(), &[0.]);
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn metal_projection_matches_fixed_codebook_reference_and_dtype_rounding() {
    use ComfyQuantFormat::*;
    assert!(crate::metal_available());
    assert!(matches!(
        mlx_rs::Device::try_default().unwrap().get_type().unwrap(),
        mlx_rs::DeviceType::Gpu
    ));
    for format in [Float8E4M3Fn, Float8E5M2, Mxfp8, Fp8Rowwise] {
        for (m, n, k) in [(1usize, 1usize, 32usize), (3, 17, 35)] {
            let mx = format == Mxfp8;
            let rowwise = format == Fp8Rowwise;
            let stored = if mx {
                mxfp8_padded_shape([n, k])
            } else {
                [n, k]
            };
            let swshape = mxfp8_scale_shape(stored);
            let mut scales = vec![127u8; swshape.iter().product()];
            let mut codes = vec![0u8; stored.iter().product()];
            let mut input = vec![0f32; m * k];
            let scalar_scale = 0.8125f32;
            for row in 0..n {
                for col in 0..k {
                    let base = if format == Float8E5M2 { 0x3c } else { 0x38 };
                    codes[row * stored[1] + col] = if n == 1 {
                        if col == 0 {
                            base
                        } else {
                            0
                        }
                    } else {
                        base + ((row * 7 + col * 11) % 19) as u8
                            + (if (row + col) % 7 == 0 { 128 } else { 0 })
                    };
                    if mx {
                        scales[mxfp8_swizzled_scale_index(stored, row, col / 32)] = if n == 1 {
                            127
                        } else {
                            (126 + (row + col / 32) % 3) as u8
                        };
                    }
                }
            }
            for (i, x) in input.iter_mut().enumerate() {
                *x = if n == 1 {
                    if i == 0 {
                        1.875
                    } else {
                        0.
                    }
                } else {
                    ((i * 29 + 7) % 113) as f32 / 32.0 - 1.0
                };
            }
            let row_scales: Vec<_> = (0..n).map(|row| scalar_scale + row as f32 / 16.).collect();
            let scale = if rowwise {
                Array::from_slice(&row_scales, &[n as i32])
            } else if mx {
                Array::from_slice(&scales, &[swshape[0] as i32, swshape[1] as i32])
            } else {
                Array::from_f32(scalar_scale)
            };
            let weight = Array::from_slice(&codes, &[stored[0] as i32, stored[1] as i32]);
            let op = Fp8Linear::new(format, weight, Some(scale), None, &[n, k], None).unwrap();
            let x = Array::from_slice(&input, &[m as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let actual = op.forward(&x).unwrap().as_dtype(Dtype::Float32).unwrap();
            actual.eval().unwrap();
            for r in 0..m {
                for c in 0..n {
                    let mut sum = 0f64;
                    for (b, block) in input[r * k..(r + 1) * k].chunks(32).enumerate() {
                        let max = block
                            .iter()
                            .copied()
                            .map(round_bf16)
                            .map(f32::abs)
                            .fold(0., f32::max);
                        let act_scale = if mx && max > 0. {
                            2f32.powi((max / 448.).log2().ceil() as i32)
                        } else if rowwise {
                            let peak = input[r * k..(r + 1) * k]
                                .iter()
                                .copied()
                                .map(round_bf16)
                                .map(f32::abs)
                                .fold(0., f32::max);
                            (peak / 448.).max(1e-12)
                        } else {
                            1.
                        };
                        let ws = if mx {
                            2f64.powi(
                                i32::from(scales[mxfp8_swizzled_scale_index(stored, c, b)]) - 127,
                            )
                        } else {
                            f64::from(if rowwise { row_scales[c] } else { scalar_scale })
                        };
                        for (j, &a) in block.iter().enumerate() {
                            let q = nearest(round_bf16(a) / act_scale, format) * act_scale;
                            let w = codebook(format, codes[c * stored[1] + b * 32 + j]);
                            sum += f64::from(q) * f64::from(w) * ws;
                        }
                    }
                    let expected = round_bf16(sum as f32);
                    let measured = actual.as_slice::<f32>()[r * n + c];
                    let distance = ordered_bf16(expected).abs_diff(ordered_bf16(measured));
                    assert!(distance<=if n==1{1}else{2},"{format:?} row={r} out={c}: {measured} vs {expected} ({distance} BF16 ULP)");
                    if mx && n == 1 {
                        assert_eq!(measured,1.875,"Kitchen rounds 1.875/448 up to E8M0 byte 120 and encodes the activation as 0x77");
                    }
                }
            }
        }
    }
    // The sum is exactly 1+2^-9; whether it survives subtraction of a unit bias
    // distinguishes each output dtype's mandated pre-bias rounding.
    let codes = Array::from_slice(&[0x38u8, 0x01], &[1, 2]);
    let op = Fp8Linear::new(
        Float8E4M3Fn,
        codes,
        None,
        None,
        &[1, 2],
        Some(Array::from_slice(&[-1.0f32], &[1])),
    )
    .unwrap();
    for (dtype, expected) in [
        (Dtype::Bfloat16, 0.),
        (Dtype::Float16, 1. / 512.),
        (Dtype::Float32, 1. / 512.),
    ] {
        let x = Array::from_slice(&[1.0f32, 1.0], &[1, 2])
            .as_dtype(dtype)
            .unwrap();
        let y = op.forward(&x).unwrap();
        assert_eq!(y.dtype(), dtype);
        assert_eq!(
            y.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>(),
            [expected]
        );
    }
    let rowwise = Fp8Linear::new(
        Fp8Rowwise,
        Array::from_slice(&[0x38u8], &[1, 1]),
        Some(Array::from_slice(&[1.0f32 + 1.0 / 512.0], &[1])),
        None,
        &[1, 1],
        Some(Array::from_slice(&[-1.0f32], &[1])),
    )
    .unwrap();
    for (dtype, expected) in [
        (Dtype::Bfloat16, 0.),
        (Dtype::Float16, 1. / 512.),
        (Dtype::Float32, 1. / 512.),
    ] {
        let x = Array::from_slice(&[1.0f32], &[1, 1])
            .as_dtype(dtype)
            .unwrap();
        let y = rowwise.forward(&x).unwrap();
        assert_eq!(y.dtype(), dtype);
        assert_eq!(
            y.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>(),
            [expected]
        );
    }
    // A gathered coefficient rounds once to the embedding consumer’s dtype.
    let mut weights = crate::weights::Weights::empty();
    let desc = br#"{"format":"float8_e4m3fn"}"#;
    weights.insert(
        "table.comfy_quant",
        Array::from_slice(desc, &[desc.len() as i32]),
    );
    weights.insert("table.weight", Array::from_slice(&[0x38u8, 0x40], &[2, 1]));
    let scale = 1.0 + 2.0f32.powi(-9);
    weights.insert("table.weight_scale", Array::from_f32(scale));
    let ids = Array::from_slice(&[0i32], &[1]);
    for (dtype, expected) in [
        (Dtype::Float32, scale),
        (Dtype::Float16, scale),
        (Dtype::Bfloat16, 1.),
    ] {
        let rows = crate::quant::embedding_compute(&weights, "table", dtype)
            .unwrap()
            .rows(&ids)
            .unwrap();
        assert_eq!(rows.dtype(), dtype);
        let actual = rows.as_dtype(Dtype::Float32).unwrap();
        actual.eval().unwrap();
        assert_eq!(actual.as_slice::<f32>(), &[expected]);
    }
    check_weight_only_scalar_projection();
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED logical_weights::fp8_metal_proof::metal_projection_matches_fixed_codebook_reference_and_dtype_rounding");
}
