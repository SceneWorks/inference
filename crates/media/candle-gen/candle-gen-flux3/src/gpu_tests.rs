//! CUDA functional tests for the FLUX3 publisher-format kernels (sc-26241). Synthetic tiny tensors
//! only; no real weights, no timing or memory claims. Each test skips with a printed reason when the
//! device is below the format's compute floor.
//!
//! Two kinds of check:
//!
//! * **Exact kernel-vs-mirror.** With an identity (FP8R) or one-hot (NVFP4) weight every output is
//!   ONE product, so the f32 accumulate is exact and the device output exposes the kernel's own
//!   activation quantization. It must equal the bit-level CPU mirror's prediction bit for bit.
//! * **Layer parity vs the gen-core oracle** on random weights. The only allowed difference is the
//!   f32 accumulation order of a K-term dot product followed by one BF16 rounding: the tolerance is
//!   two BF16 ulps of the output peak (2^-7 · peak). BFL's `use_fast_accum` FP8 path is NOT
//!   reproduced (we accumulate fully in f32), which is why this tolerance is stated here and not
//!   widened anywhere else.
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::flux3::quantization::{
    decode_e4m3, dequantize_fp8_rows, quantize_fp8_rows, quantize_nvfp4_static,
};
use candle_quant_kernels::cublaslt::CublasLt;
use candle_quant_kernels::flux3_quant as mirror;

fn device() -> Device {
    Device::new_cuda(0).expect("CUDA device 0 (CUDA_VISIBLE_DEVICES selects GPU1)")
}

fn seeded(n: usize, salt: f32, spread: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 * 0.618 + salt).sin() * spread;
            if i % 11 == 0 {
                x * 9.0
            } else {
                x
            }
        })
        .collect()
}

fn bf16_round(v: f32) -> f32 {
    f32::from_bits((mirror::bf16_bits(v) as u32) << 16)
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn bf16_input(values: &[f32], m: usize, k: usize, dev: &Device) -> (Tensor, Vec<f32>) {
    let rounded: Vec<f32> = values.iter().map(|v| bf16_round(*v)).collect();
    let t = Tensor::from_vec(rounded.clone(), (m, k), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    (t, rounded)
}

fn within_two_bf16_ulps(got: &[f32], want: &[f32], what: &str) {
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("{what}: max |gpu - oracle| = {worst:e}, peak {peak:e}");
    assert!(
        worst <= peak / 128.0,
        "{what}: {worst} exceeds two bf16 ulps of the peak {peak}"
    );
}

/// FP8R with an identity weight: y[m, n] = bf16(e4m3(x/sa)[m, n] · sa[m]) exactly.
#[test]
fn fp8r_kernels_match_the_cpu_mirror_bit_for_bit() {
    let dev = device();
    let lt = CublasLt::new(&dev).unwrap();
    if !lt.meets_fp8_floor().unwrap() {
        eprintln!(
            "SKIP: {:?} is below the FP8 floor",
            lt.compute_cap().unwrap()
        );
        return;
    }
    let (m, k) = (5, 64);
    let (x, xr) = bf16_input(&seeded(m * k, 0.3, 3.0), m, k, &dev);
    let identity: Vec<u8> = (0..k * k)
        .map(|i| if i / k == i % k { 0x38 } else { 0 })
        .collect();
    let weight = lt.stage_fp8_bytes(&identity, k, k).unwrap();
    let scales = lt.stage_f32(&vec![1.0; k]).unwrap();
    let y = host(&lt.flux3_fp8_rowwise_linear(&x, &weight, &scales).unwrap());
    for (r, row) in xr.chunks_exact(k).enumerate() {
        let (codes, sa) = mirror::fp8_rowwise_row(row);
        for c in 0..k {
            let want = f32::from_bits(
                (mirror::fp8_rowwise_epilogue(decode_e4m3(codes[c]), sa, 1.0) as u32) << 16,
            );
            // f32 equality: exact, with +0 == -0 (a zero product's sign is not a kernel fact).
            assert!(
                y[r * k + c] == want,
                "row {r} col {c}: {} vs {want}",
                y[r * k + c]
            );
        }
    }
}

/// FP8R layer parity on random weights vs the gen-core rowwise oracle.
#[test]
fn fp8r_layer_matches_the_source_oracle() {
    let dev = device();
    let lt = CublasLt::new(&dev).unwrap();
    if !lt.meets_fp8_floor().unwrap() {
        eprintln!("SKIP: below the FP8 floor");
        return;
    }
    let (m, k, n) = (7, 96, 48);
    let (x, xr) = bf16_input(&seeded(m * k, 1.1, 2.0), m, k, &dev);
    let w = seeded(n * k, 2.7, 0.05);
    let (wc, ws) = quantize_fp8_rows(&w, k).unwrap();
    let weight = lt.stage_fp8_bytes(&wc, n, k).unwrap();
    let scales = lt.stage_f32(&ws).unwrap();
    let y = host(&lt.flux3_fp8_rowwise_linear(&x, &weight, &scales).unwrap());
    let (xc, xs) = quantize_fp8_rows(&xr, k).unwrap();
    let xd = dequantize_fp8_rows(&xc, &xs, k).unwrap();
    let wd = dequantize_fp8_rows(&wc, &ws, k).unwrap();
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            let acc: f64 = (0..k)
                .map(|j| xd[r * k + j] as f64 * wd[c * k + j] as f64)
                .sum();
            bf16_round(acc as f32)
        })
        .collect();
    within_two_bf16_ulps(&y, &want, "fp8r");
}

fn unswizzle(swizzled: &[u8], rows: usize, blocks: usize) -> Vec<u8> {
    let sf_cols = blocks.div_ceil(4) * 4;
    (0..rows * blocks)
        .map(|i| swizzled[mirror::scale_offset(i / blocks, i % blocks, sf_cols)])
        .collect()
}

fn e2m1_value(nibble: u8) -> f32 {
    const E2M1: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let v = E2M1[(nibble & 7) as usize];
    if nibble & 8 != 0 {
        -v
    } else {
        v
    }
}

/// Element (r, c) of a packed static-NVFP4 operand WITHOUT its global scale.
fn nvfp4_element(packed: &[u8], block_scales: &[u8], k: usize, r: usize, c: usize) -> f32 {
    let byte = packed[r * k / 2 + c / 2];
    let nibble = if c.is_multiple_of(2) {
        byte & 15
    } else {
        byte >> 4
    };
    e2m1_value(nibble) * decode_e4m3(block_scales[r * (k / 16) + c / 16])
}

/// NVFP4 with a one-hot weight (E2M1 1.0, block scale 1.0, weight_scale_2 1.0): y[m, n] is the
/// device-packed activation element times the RAW input scale, exactly.
#[test]
fn nvfp4_static_kernel_matches_the_cpu_mirror_bit_for_bit() {
    let dev = device();
    let lt = CublasLt::new(&dev).unwrap();
    if !lt.meets_nvfp4_floor().unwrap() {
        eprintln!(
            "SKIP: {:?} is below the NVFP4 floor",
            lt.compute_cap().unwrap()
        );
        return;
    }
    let (m, k) = (5, 64);
    for input_scale in [0.37f32, 0.0] {
        let (x, xr) = bf16_input(&seeded(m * k, 0.9, 4.0), m, k, &dev);
        let mut packed = vec![0u8; k * k / 2];
        for n in 0..k {
            packed[n * k / 2 + n / 2] |= 2 << ((n % 2) * 4);
        }
        let block_scales = vec![0x38u8; k * k / 16];
        let weight = lt
            .flux3_stage_nvfp4_weight(&packed, &block_scales, k, k, 1.0)
            .unwrap();
        let y = host(
            &lt.flux3_nvfp4_static_linear(&x, &weight, input_scale)
                .unwrap(),
        );
        let (xp, xs) = mirror::nvfp4_static_pack(&xr, m, k, input_scale);
        let xs = unswizzle(&xs, m, k / 16);
        for r in 0..m {
            for c in 0..k {
                let want = bf16_round(nvfp4_element(&xp, &xs, k, r, c) * input_scale);
                assert!(
                    y[r * k + c] == want,
                    "scale {input_scale} row {r} col {c}: {} vs {want}",
                    y[r * k + c]
                );
            }
        }
    }
}

/// Static NVFP4 layer parity on random weights vs the gen-core static oracle.
#[test]
fn nvfp4_static_layer_matches_the_source_oracle() {
    let dev = device();
    let lt = CublasLt::new(&dev).unwrap();
    if !lt.meets_nvfp4_floor().unwrap() {
        eprintln!("SKIP: below the NVFP4 floor");
        return;
    }
    let (m, k, n) = (7, 128, 64);
    let input_scale = 0.11f32;
    let weight_scale_2 = 0.0042f32;
    let (x, xr) = bf16_input(&seeded(m * k, 1.9, 3.0), m, k, &dev);
    // A packed weight from the same static recipe (any valid E2M1/E4M3 bytes are a weight).
    let w = quantize_nvfp4_static(&seeded(n * k, 0.4, 2.0), k, weight_scale_2).unwrap();
    let w_scales = unswizzle(&w.scales, n, k / 16);
    let weight = lt
        .flux3_stage_nvfp4_weight(&w.packed, &w_scales, n, k, weight_scale_2)
        .unwrap();
    let y = host(
        &lt.flux3_nvfp4_static_linear(&x, &weight, input_scale)
            .unwrap(),
    );
    let xq = quantize_nvfp4_static(&xr, k, input_scale).unwrap();
    let x_scales = unswizzle(&xq.scales, m, k / 16);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            let acc: f64 = (0..k)
                .map(|j| {
                    nvfp4_element(&xq.packed, &x_scales, k, r, j) as f64
                        * nvfp4_element(&w.packed, &w_scales, k, c, j) as f64
                })
                .sum();
            bf16_round((acc as f32) * (input_scale * weight_scale_2))
        })
        .collect();
    within_two_bf16_ulps(&y, &want, "nvfp4-static");
}
