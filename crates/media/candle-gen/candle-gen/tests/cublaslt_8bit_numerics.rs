//! sc-9299 numerics gate for the cuBLASLt 8-bit GEMM leg. CUDA-only (the wrapper needs a live Lt
//! handle); a graceful no-op on CPU/Metal builds. Two checks the story calls out:
//!
//! 1. **Parity vs f32** for well-conditioned inputs — fp8 and int8 both reconstruct `X·Wᵀ` within
//!    their per-tensor-quant error budget (fp8 ≈ a few %, int8 tighter for in-range data).
//! 2. **The sc-7702 activation-outlier scenario** — one feature blown up to ±1e4 (the gpt-oss
//!    outlier). Per-tensor int8 sets the whole scale off that single value and zeros the co-located
//!    in-range channels → large error. This is EXPECTED to degrade WITHOUT rotation; it is the
//!    motivation for the ConvRot consume story (sc-9300), recorded here, not "fixed".

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::cublaslt::CublasLt;

/// splitmix64-hashed deterministic pseudo-random in ~[-1, 1] (launch-portable; no device RNG).
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Mean relative RMS error of `got` vs `reference` (both flattened f32).
fn rel_rms(got: &[f32], reference: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        num += (*g as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

fn f32_reference(x: &Tensor, w: &Tensor) -> Vec<f32> {
    // (M,K) · (N,K)ᵀ = (M,N)
    let y = x.matmul(&w.t().unwrap().contiguous().unwrap()).unwrap();
    y.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn wellconditioned_parity_vs_f32() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9299] no CUDA device; skipping numerics gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();
    eprintln!(
        "[sc-9299] device compute cap = {:?}",
        lt.compute_cap().unwrap()
    );

    // Small, 16-aligned DiT-ish shape.
    let (m, k, n) = (32usize, 256usize, 256usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 1), (m, k), &dev).unwrap();
    let w = Tensor::from_vec(pseudo_random(n * k, 2), (n, k), &dev).unwrap();
    let reference = f32_reference(&x, &w);

    // fp8 path.
    let qw = candle_gen::quant::quantize_weight_fp8(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_fp8(&x).unwrap();
    let y_fp8 = lt
        .matmul_fp8(&qw.q, qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_fp8 = rel_rms(&y_fp8, &reference);
    eprintln!("[sc-9299] fp8  rel-RMS vs f32 = {err_fp8:.4}");

    // int8 path.
    let qwi = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let qxi = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let y_i8 = lt
        .matmul_int8(&qwi.q, qwi.scale, &qxi.q, qxi.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_i8 = rel_rms(&y_i8, &reference);
    eprintln!("[sc-9299] int8 rel-RMS vs f32 = {err_i8:.4}");

    // Per-tensor 8-bit on well-conditioned data lands well under 10% for both. fp8 E4M3 (3 mantissa
    // bits) is the looser of the two here; int8 (7-bit magnitude) is tighter for in-range data.
    assert!(err_fp8 < 0.10, "fp8 rel-RMS too high: {err_fp8}");
    assert!(err_i8 < 0.10, "int8 rel-RMS too high: {err_i8}");
}

#[test]
fn int8_per_channel_parity_vs_f32() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9300] no CUDA device; skipping per-channel numerics gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();

    // A weight whose output rows span very different magnitudes — the case per-channel exists for.
    let (m, k, n) = (32usize, 256usize, 256usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 5), (m, k), &dev).unwrap();
    let mut wv = pseudo_random(n * k, 6);
    for row in 0..n {
        let mag = 1.0 + (row as f32) * 0.5; // rows 1x .. ~128x
        for j in 0..k {
            wv[row * k + j] *= mag;
        }
    }
    let w = Tensor::from_vec(wv, (n, k), &dev).unwrap();
    let reference = f32_reference(&x, &w);

    // Per-channel int8 weight quant (the ConvRot granularity), on-device IGEMM + per-row dequant.
    let qw = candle_gen::quant::quantize_weight_int8_per_channel(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let y_pc = lt
        .matmul_int8_per_channel(&qw.q, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_pc = rel_rms(&y_pc, &reference);

    // Per-TENSOR int8 on the SAME ragged weight — the scalar scale is dominated by the largest row,
    // so the small-magnitude rows lose precision and the matmul error is materially worse.
    let qw_pt = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let y_pt = lt
        .matmul_int8(&qw_pt.q, qw_pt.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let err_pt = rel_rms(&y_pt, &reference);

    eprintln!("[sc-9300] int8 X·Wᵀ rel-RMS vs f32: per-channel={err_pc:.4} per-tensor={err_pt:.4}");
    // Per-channel lands well under 10% on the ragged weight (per-tensor does not).
    assert!(err_pc < 0.10, "per-channel int8 rel-RMS too high: {err_pc}");
    assert!(
        err_pc < err_pt,
        "per-channel ({err_pc}) must beat per-tensor ({err_pt}) on a magnitude-ragged weight"
    );
}

/// sc-9601 perf: the **on-device** per-channel dequant (`matmul_int8_per_channel_staged_ondevice`,
/// int32 IGEMM + on-device `i32 → f32` cast + candle float fold) must match the exact int32→host fold
/// (`matmul_int8_per_channel_staged`) — same numbers, no host round-trip. Also asserts the on-device
/// path is actually available (the vendored `cast_i32_f32` kernel is present), so the fast path is the
/// one that runs, not a silent fallback to the host fold.
#[test]
fn int8_per_channel_ondevice_matches_host_fold() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9601] no CUDA device; skipping on-device dequant gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();
    assert!(
        lt.supports_ondevice_int8_dequant(),
        "sc-9601: this device must cast i32→f32 on-device (vendored cast_i32_f32) for the fast path"
    );

    let (m, k, n) = (48usize, 512usize, 384usize);
    let x = Tensor::from_vec(pseudo_random(m * k, 11), (m, k), &dev).unwrap();
    let mut wv = pseudo_random(n * k, 12);
    for row in 0..n {
        let mag = 1.0 + (row as f32) * 0.25;
        for j in 0..k {
            wv[row * k + j] *= mag;
        }
    }
    let w = Tensor::from_vec(wv, (n, k), &dev).unwrap();

    let qw = candle_gen::quant::quantize_weight_int8_per_channel(&w).unwrap();
    let qx = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    let staged = lt.stage_int8(&qw.q).unwrap();

    let y_host = lt
        .matmul_int8_per_channel_staged(&staged, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let y_dev = lt
        .matmul_int8_per_channel_staged_ondevice(&staged, &qw.scale, &qx.q, qx.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let err = rel_rms(&y_dev, &y_host);
    eprintln!("[sc-9601] on-device vs host int8 fold rel-RMS = {err:.2e} (both bf16 output)");
    // Both fold the SAME exact int32 accumulate (one on host, one via the f32-output epilogue) and cast
    // to bf16 — they must agree to bf16 rounding (~1e-3), not merely "close".
    assert!(
        err < 5e-3,
        "on-device dequant must match the host fold to bf16 rounding, got rel-RMS {err}"
    );
}

#[test]
fn int8_degrades_on_activation_outlier_no_rotation() {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-9299] no CUDA device; skipping outlier gate");
            return;
        }
    };
    let lt = CublasLt::new(&dev).unwrap();

    let (m, k, n) = (32usize, 256usize, 256usize);
    let base = pseudo_random(m * k, 3); // the in-range signal (~±1)
    let mut xv = base.clone();
    // sc-7702 scenario: a single feature blown up to ±1e4 (the gpt-oss activation outlier), in
    // column 0 of every token row so the per-tensor activation scale is dominated by it.
    for row in 0..m {
        xv[row * k] = if row % 2 == 0 { 1.0e4 } else { -1.0e4 };
    }
    // The in-range-only activation: outlier column zeroed. This is the co-located signal a correct
    // (rotated) path preserves — the contribution per-tensor int8 is expected to annihilate.
    let mut inrange = base.clone();
    for row in 0..m {
        inrange[row * k] = 0.0;
    }

    let x = Tensor::from_vec(xv, (m, k), &dev).unwrap();
    let x_inrange = Tensor::from_vec(inrange, (m, k), &dev).unwrap();
    // The outlier-only activation (everything but column 0 zeroed) — the part int8 DOES keep.
    let mut only_out = vec![0f32; m * k];
    for row in 0..m {
        only_out[row * k] = if row % 2 == 0 { 1.0e4 } else { -1.0e4 };
    }
    let x_only_outlier = Tensor::from_vec(only_out, (m, k), &dev).unwrap();
    let w = Tensor::from_vec(pseudo_random(n * k, 4), (n, k), &dev).unwrap();

    // True in-range contribution: what the co-located features SHOULD add to the output.
    let true_inrange = f32_reference(&x_inrange, &w);
    // The pure outlier-column contribution (what int8 preserves faithfully).
    let outlier_contrib = f32_reference(&x_only_outlier, &w);

    let qwi = candle_gen::quant::quantize_weight_int8(&w).unwrap();
    let qxi = candle_gen::quant::quantize_activation_int8(&x).unwrap();
    // absmax/127 ≈ 1e4/127 ≈ 79 — every in-range (±1) feature rounds to 0 after dividing by ~79.
    eprintln!(
        "[sc-9299] int8 activation scale under outlier = {:.2} (in-range features round to 0)",
        qxi.scale
    );
    let y_i8 = lt
        .matmul_int8(&qwi.q, qwi.scale, &qxi.q, qxi.scale)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Recovered in-range contribution = int8 output − the outlier-column contribution. Because the
    // in-range channels quantized to 0, this residual is ~0, NOT the true in-range signal → the
    // relative error against `true_inrange` is ~100%. That collapse is the sc-7702 failure and the
    // whole reason ConvRot (sc-9300) rotates the outlier energy across channels before quantizing.
    let recovered_inrange: Vec<f32> = y_i8
        .iter()
        .zip(&outlier_contrib)
        .map(|(y, o)| y - o)
        .collect();
    let err = rel_rms(&recovered_inrange, &true_inrange);
    eprintln!("[sc-9299] int8 in-range rel-RMS under activation outlier = {err:.4} (EXPECTED ~1.0 — in-range signal annihilated; motivates sc-9300 ConvRot)");

    // Assert the DEGRADATION. If this ever came back small, per-tensor int8 would be outlier-safe
    // and the ConvRot motivation would be void. Recorded, not fixed.
    assert!(
        err > 0.5,
        "expected int8 to annihilate the in-range signal under the sc-7702 outlier (got rel-RMS \
         {err}); if this fails, per-tensor int8 is outlier-safe and ConvRot is unnecessary"
    );
}

/// Checkpoint input, stock Candle constructor, immutable packed projection, and additive adapter.
/// These fixed reference products come from the independent host float8 implementation.
#[test]
#[ignore = "dedicated CUDA checkpoint-numerics proof"]
fn checkpoint_fp8_projection_exact_bf16() -> Result<(), Box<dyn std::error::Error>> {
    use candle_gen::candle_nn::Module;
    use candle_gen::quant::AdaptLinear;
    use float8::{F8E4M3, F8E5M2};
    use half::bf16;
    use safetensors::{tensor::TensorView, Dtype};
    let dev = Device::new_cuda(0)?;
    let tmp = tempfile::tempdir()?;
    let scalar = |e5: bool, v: f32| {
        if e5 {
            F8E5M2::from_f32(v).to_f32()
        } else {
            F8E4M3::from_f32(v).to_f32()
        }
    };
    for e5 in [false, true] {
        for sx in [None, Some(0.5f32)] {
            let format = if e5 { "float8_e5m2" } else { "float8_e4m3fn" };
            let dtype = if e5 { Dtype::F8_E5M2 } else { Dtype::F8_E4M3 };
            let mut codes = vec![0u8; 16 * 16];
            let mut w = Vec::new();
            for row in 0..16 {
                let v = if row % 2 == 0 { 1.5 } else { -0.75 };
                codes[row * 16 + row] = if e5 {
                    F8E5M2::from_f32(v).to_bits()
                } else {
                    F8E4M3::from_f32(v).to_bits()
                };
                w.push(v);
            }
            let desc = format!(r#"{{"format":"{format}"}}"#);
            let sw = 2f32.to_le_bytes();
            let input_scale = sx.unwrap_or(1.).to_le_bytes();
            let mut tensors = vec![
                (
                    "linear.weight",
                    TensorView::new(dtype, vec![16, 16], &codes)?,
                ),
                (
                    "linear.comfy_quant",
                    TensorView::new(Dtype::U8, vec![desc.len()], desc.as_bytes())?,
                ),
                (
                    "linear.weight_scale",
                    TensorView::new(Dtype::F32, vec![], &sw)?,
                ),
            ];
            if sx.is_some() {
                tensors.push((
                    "linear.input_scale",
                    TensorView::new(Dtype::F32, vec![], &input_scale)?,
                ));
            }
            let path = tmp
                .path()
                .join(format!("scalar-{e5}-{}.safetensors", sx.is_some()));
            safetensors::tensor::serialize_to_file(tensors, None, &path)?;
            let vb = candle_gen::mmap_var_builder(&[path], DType::BF16, &dev)?;
            let lin = candle_gen::candle_nn::linear_no_bias(16, 16, vb.pp("linear"))?;
            assert_eq!(
                lin.native_format_id(),
                Some(if e5 {
                    "fp8-e5m2-scalar-v1"
                } else {
                    "fp8-e4m3-scalar-v1"
                })
            );
            let input: Vec<bf16> = (0..18 * 16)
                .map(|i| {
                    bf16::from_f32(
                        [0.5, -1.375, 0.001953125, 300.0, 0.375, -0.25, 0.125, 1.0][i % 8],
                    )
                })
                .collect();
            let x = Tensor::from_slice(&input, (18, 16), &dev)?.narrow(0, 1, 17)?;
            let got = __model_proof_output!(lin.forward(&x))?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let mut adapted = AdaptLinear::from_dense(lin, 16, 16);
            let mut a = vec![bf16::ZERO; 16];
            a[0] = bf16::ONE;
            adapted.push_lora_checked(
                Tensor::from_slice(&a, (16, 1), &dev)?,
                Tensor::ones((1, 16), DType::BF16, &dev)?,
                1.0,
            )?;
            let got_adapted = __model_proof_output!(adapted.forward(&x))?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            for row in 0..17 {
                for j in 0..16 {
                    let xij = input[(row + 1) * 16 + j].to_f32();
                    let s = sx.unwrap_or(1.);
                    let sat = if e5 { 57344.0 } else { 448.0 };
                    let expected =
                        bf16::from_f32(scalar(e5, (xij / s).clamp(-sat, sat)) * s * w[j] * 2.);
                    assert_eq!(
                        got[row * 16 + j],
                        expected,
                        "{format} scale {sx:?} [{row},{j}]"
                    );
                    assert_eq!(
                        got_adapted[row * 16 + j],
                        bf16::from_f32(expected.to_f32() + input[(row + 1) * 16].to_f32())
                    );
                }
            }
        }
    }
    // One term distinguishes native F16/F32 output from a BF16 intermediate and bias-before-rounding.
    for (format, dtype, unit) in [
        (
            "float8_e4m3fn",
            Dtype::F8_E4M3,
            F8E4M3::from_f32(1.).to_bits(),
        ),
        (
            "float8_e5m2",
            Dtype::F8_E5M2,
            F8E5M2::from_f32(1.).to_bits(),
        ),
    ] {
        let path = tmp.path().join(format!("{format}-bias.safetensors"));
        let mut codes = [0u8; 16 * 16];
        for row in 0..16 {
            codes[row * 16 + row] = unit;
        }
        let scale = (1.0_f32 + 1.0 / 512.).to_le_bytes();
        let bias = (-1.0_f32).to_le_bytes().repeat(16);
        let desc = format!(r#"{{"format":"{format}"}}"#);
        safetensors::tensor::serialize_to_file(
            [
                (
                    "linear.weight",
                    TensorView::new(dtype, vec![16, 16], &codes)?,
                ),
                (
                    "linear.weight_scale",
                    TensorView::new(Dtype::F32, vec![], &scale)?,
                ),
                ("linear.bias", TensorView::new(Dtype::F32, vec![16], &bias)?),
                (
                    "linear.comfy_quant",
                    TensorView::new(Dtype::U8, vec![desc.len()], desc.as_bytes())?,
                ),
            ],
            None,
            &path,
        )?;
        for output in [DType::BF16, DType::F16, DType::F32] {
            let vb = candle_gen::mmap_var_builder(std::slice::from_ref(&path), output, &dev)?;
            let op = candle_gen::candle_nn::linear(16, 16, vb.pp("linear"))?;
            assert!(op.native_format_id().is_some());
            let y = op.forward(&Tensor::ones((1, 16), output, &dev)?)?;
            assert_eq!(y.dtype(), output);
            let expected = if output == DType::BF16 { 0. } else { 1. / 512. };
            assert_eq!(
                y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                vec![expected; 16]
            );
        }
    }
    let (m, n, k) = (17usize, 23usize, 61usize);
    let mut codes = vec![0u8; 32 * 64];
    let mut scales = vec![127u8; 128 * 4];
    let mut factors = Vec::new();
    for row in 0..n {
        let col = (row * 7) % k;
        let code = F8E4M3::from_f32(if row % 2 == 0 { 1.5 } else { -0.75 });
        let exponent = (row % 5) as i32 - 2;
        codes[row * 64 + col] = code.to_bits();
        scales[(row % 32) * 16 + (row / 32) * 4 + col / 32] = (exponent + 127) as u8;
        factors.push((col, code.to_f32() * 2f32.powi(exponent)));
    }
    let desc = br#"{"format":"mxfp8"}"#;
    let path = tmp.path().join("mx.safetensors");
    safetensors::tensor::serialize_to_file(
        [
            (
                "linear.weight",
                TensorView::new(Dtype::F8_E4M3, vec![32, 64], &codes)?,
            ),
            (
                "linear.weight_scale",
                TensorView::new(Dtype::U8, vec![128, 4], &scales)?,
            ),
            (
                "linear.comfy_quant",
                TensorView::new(Dtype::U8, vec![desc.len()], desc)?,
            ),
        ],
        None,
        &path,
    )?;
    let vb = candle_gen::mmap_var_builder(&[path], DType::BF16, &dev)?;
    let lin = candle_gen::candle_nn::linear_no_bias(k, n, vb.pp("linear"))?;
    assert_eq!(lin.native_format_id(), Some("mxfp8-v1"));
    let input: Vec<bf16> = (0..(m + 1) * k)
        .map(|i| bf16::from_f32([0.25, 0.5, -1., 1.875][i % 4]))
        .collect();
    let x = Tensor::from_slice(&input, (m + 1, k), &dev)?.narrow(0, 1, m)?;
    let got = __model_proof_output!(lin.forward(&x))?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    for row in 0..m {
        for (j, &(col, w)) in factors.iter().enumerate() {
            assert_eq!(
                got[row * n + j],
                bf16::from_f32(input[(row + 1) * k + col].to_f32() * w),
                "MX [{row},{j}]"
            );
        }
    }
    // Publisher FP8R scales its activation per token and the weight per output.
    let k = 16;
    let mut codes = vec![0; k * k];
    let mut scales = Vec::new();
    let mut scale_bytes = Vec::new();
    for col in 0..k {
        codes[col * k + col] = F8E4M3::from_f32(1.5).to_bits();
        let scale = 0.25 + col as f32 * 0.03125;
        scales.push(scale);
        scale_bytes.extend(scale.to_le_bytes());
    }
    let path = tmp.path().join("rowwise.safetensors");
    safetensors::tensor::serialize_to_file(
        [
            (
                "proj.weight",
                TensorView::new(Dtype::F8_E4M3, vec![k, k], &codes)?,
            ),
            (
                "proj.weight_scale",
                TensorView::new(Dtype::F32, vec![k], &scale_bytes)?,
            ),
        ],
        None,
        &path,
    )?;
    let vb = candle_gen::mmap_var_builder(&[path], DType::BF16, &dev)?;
    let lin = candle_gen::candle_nn::linear_no_bias(k, k, vb.pp("proj"))?;
    assert_eq!(lin.native_format_id(), Some("fp8-e4m3-rowwise-v1"));
    let inputs: Vec<_> = (0..17 * k)
        .map(|i| bf16::from_f32(((i / k) as f32 - 8.) * ((i % k) as f32 - 7.5) * 0.125))
        .collect();
    let x = Tensor::from_slice(&inputs, (17, k), &dev)?;
    let outputs = __model_proof_output!(lin.forward(&x))?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    for (row, (input, output)) in inputs
        .chunks_exact(k)
        .zip(outputs.chunks_exact(k))
        .enumerate()
    {
        let max = input.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
        let sx = (max / 448.).max(1e-12);
        for col in 0..k {
            let code = F8E4M3::from_f32((input[col].to_f32() / sx).clamp(-448., 448.)).to_f32();
            assert_eq!(
                output[col],
                bf16::from_f32((code * 1.5 * sx) * scales[col]),
                "FP8R [{row},{col}]"
            );
        }
    }
    // The BFL activation is BF16 before row quantization. Preserve the consumer's
    // F16/F32 scale product before adding its bias, including near rounding ties.
    let mut identity = vec![0; k * k];
    let row_scales: Vec<f32> = (0..k).map(|j| 1. + (5 + 2 * j) as f32 / 4096.).collect();
    for i in 0..k {
        identity[i * k + i] = F8E4M3::from_f32(1.).to_bits();
    }
    let scale_bytes: Vec<u8> = row_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let bias = (-1.0f32).to_le_bytes().repeat(k);
    let path = tmp.path().join("rowwise-typed.safetensors");
    safetensors::tensor::serialize_to_file(
        [
            (
                "proj.weight",
                TensorView::new(Dtype::F8_E4M3, vec![k, k], &identity)?,
            ),
            (
                "proj.weight_scale",
                TensorView::new(Dtype::F32, vec![k], &scale_bytes)?,
            ),
            ("proj.bias", TensorView::new(Dtype::F32, vec![k], &bias)?),
        ],
        None,
        &path,
    )?;
    for output in [DType::F16, DType::F32] {
        let vb = candle_gen::mmap_var_builder(std::slice::from_ref(&path), output, &dev)?;
        let op = candle_gen::candle_nn::linear(k, k, vb.pp("proj"))?;
        assert_eq!(op.native_format_id(), Some("fp8-e4m3-rowwise-v1"));
        let x = Tensor::from_slice(&vec![1.0f32 + 1. / 512.; k], (1, k), &dev)?.to_dtype(output)?;
        let y = __model_proof_output!(op.forward(&x))?;
        assert_eq!(y.dtype(), output);
        assert_eq!(y.dims(), [1, k]);
        let expected: Vec<f32> = row_scales
            .iter()
            .map(|&scale| {
                let product = (448.0f32 * (1.0f32 / 448.0)) * scale;
                match output {
                    DType::F16 => half::f16::from_f32(product).to_f32() - 1.,
                    _ => product - 1.,
                }
            })
            .collect();
        assert_eq!(
            y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            expected
        );
    }
    // The H3 Veda predictor has a separate per-head calibration and a frozen serial scorer.
    // The first coefficient rounds to one in BF16 only after applying its stored FP32 scale.
    use candle_gen::quant::veda_score::{self, Projection};
    let valid = [127usize, 2];
    let qs = [1., -2., 0.5, 4.];
    let ks = [0.25, 3., -1., 2.];
    let tiles = |values: &[f32; 4]| -> candle_core::Result<Tensor> {
        let mut rows = vec![0f32; 4 * 128];
        for (i, &value) in values.iter().enumerate() {
            rows[i * 128..i * 128 + valid[i % 2]].fill(value);
        }
        Tensor::from_vec(rows, (2, 2, 128, 1), &dev)
    };
    let qcodes = [0x38, 0, 0, 0x3c, 0, 0];
    let qscales = [1. + 1. / 512., 0.75];
    let kcodes = [0x38, 0, 0, 0xb8, 0, 0];
    let kscales = [0.5, 1.25];
    let got = veda_score::tile_logits_weights(
        &tiles(&qs)?,
        &tiles(&ks)?,
        &valid,
        2,
        Projection::Fp8 {
            codes: &qcodes,
            scales: &qscales,
        },
        Projection::Fp8 {
            codes: &kcodes,
            scales: &kscales,
        },
    )?;
    for head in 0..2 {
        let q =
            bf16::from_f32(F8E4M3::from_bits(qcodes[3 * head]).to_f32() * qscales[head]).to_f32();
        let k =
            bf16::from_f32(F8E4M3::from_bits(kcodes[3 * head]).to_f32() * kscales[head]).to_f32();
        for row in 0..2 {
            let scores = &got[head * 4 + row * 2..head * 4 + row * 2 + 2];
            let expected = [0usize, 1].map(|col| {
                (qs[head * 2 + row] * q + qs[head * 2 + row])
                    * (ks[head * 2 + col] * k + ks[head * 2 + col])
            });
            assert_eq!(scores, expected);
            assert_eq!(scores[0] > scores[1], expected[0] > expected[1]);
        }
    }
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED cublaslt_8bit_numerics::checkpoint_fp8_projection_exact_bf16");
    Ok(())
}
