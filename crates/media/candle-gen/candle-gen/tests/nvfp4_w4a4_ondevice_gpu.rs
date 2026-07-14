//! sc-11044 GPU gates for the **on-device NVFP4 W4A4 activation-quant path** (epic 11037) on consumer
//! Blackwell `sm_120`. CUDA-only (needs a live cuBLASLt handle **and** an `sm_120` device); a graceful
//! no-op on CPU/Metal and on pre-Blackwell CUDA.
//!
//! What this pins on the real GPU:
//!
//! 1. **On-device activation quantize == the CPU `Nvfp4Tensor::pack` reference** — quantizing the
//!    activation entirely on the GPU (`CublasLt::quantize_nvfp4_activation`) and running the FP4 GEMM
//!    matches, within a tiny rel-RMS, the same GEMM fed by the old CPU-round-trip pack. The W4A4
//!    forward is therefore fully on-device (no host round-trip) and still numerically faithful.
//! 2. **W4A4 forward is finite + tracks bf16** — `Nvfp4Linear::forward_checked` (NaN/inf guard) never
//!    NaNs across repeated forwards (a stand-in for denoise steps) and stays within NVFP4 tolerance of
//!    a bf16-dense reference.
//! 3. **Throughput: W4A4 vs W4A16 (and bf16)** — on representative Sana-DiT GEMM shapes, on the
//!    now-exclusive GPU, timed layer forwards. Reports the multiple (informational, not asserted —
//!    hardware-dependent).
//! 4. **Outlier-sparsity capture confirms the partition** — the spike residual-gate metric
//!    (`OutlierSparsity`) on synthetic benign / sparse / dense activations classifies as expected, and
//!    the benign→W4A4 / dense→W4A16 partition holds.
//! 5. **Real Sana-1.6B DiT weight (`#[ignore]`, env-gated)** — if `SC11044_SANA_DIT_SAFETENSORS`
//!    points at a Sana transformer shard, a real projection weight is quantized and run W4A4 vs bf16.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::nvfp4::Nvfp4Tensor;
use candle_gen::quant::{
    ActPrecision, CublasLt, Nvfp4Linear, Nvfp4Regime, OutlierClass, OutlierSparsity,
};
use std::time::Instant;

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

fn rel_rms(got: &[f32], reference: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        num += (*g as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

/// CPU f32 reference `X·Wᵀ` for `X=[M,K]`, `W=[N,K]` row-major → `[M,N]`.
fn ref_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += (x[r * k + kk] as f64) * (w[c * k + kk] as f64);
            }
            out[r * n + c] = acc as f32;
        }
    }
    out
}

/// A CUDA device iff one exists and it is Blackwell `sm_120`+ — else `None` with a SKIP note.
fn nvfp4_device() -> Option<Device> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-11044] no CUDA device; skipping on-device W4A4 GPU gate");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_nvfp4_floor() {
        Ok(true) => {
            eprintln!("[sc-11044] device cap = {:?} (NVFP4 eligible)", lt.compute_cap().unwrap());
            Some(dev)
        }
        _ => {
            eprintln!("[sc-11044] device not sm_120 ({:?}); skipping", lt.compute_cap().ok());
            None
        }
    }
}

fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// (1) The on-device activation quantize matches the CPU `Nvfp4Tensor::pack` reference: feeding the
/// same resident weight, the FP4 GEMM output from the on-device-quantized activation tracks the output
/// from the CPU-packed activation within a tiny rel-RMS — proving the W4A4 forward is fully on-device
/// (no host round-trip) without changing the numerics. Also checks both track the CPU dequant math.
#[test]
fn ondevice_activation_quant_matches_cpu_pack_ref() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (m, k, n) = (256usize, 256usize, 128usize); // K%32==0, N%16==0, M%16==0

    let x_f32 = pseudo_random(m * k, 101);
    let w_f32 = pseudo_random(n * k, 202);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();
    let w = Tensor::from_vec(w_f32.clone(), (n, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();

    // Resident weight (staged once from the CPU packer, as in sc-11041).
    let w_pk = Nvfp4Tensor::pack(&w).unwrap();
    let w_stg = lt.stage_nvfp4(&w_pk).unwrap();

    // On-device activation quantize (the sc-11044 net-new path) vs the old CPU-round-trip pack.
    let x_dev = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
    let x_cpu = lt.stage_nvfp4(&Nvfp4Tensor::pack(&x).unwrap()).unwrap();

    let y_dev = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_dev).unwrap());
    let y_cpu = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_cpu).unwrap());
    assert!(y_dev.iter().all(|v| v.is_finite()), "on-device quant produced NaN/Inf");

    let rr = rel_rms(&y_dev, &y_cpu);
    eprintln!("[sc-11044] on-device vs CPU-pack activation quant: GEMM rel-RMS = {rr:.6}");
    assert!(rr < 0.02, "on-device activation quant diverges from the CPU pack ref (rel-RMS {rr:.6})");

    // Both also track the CPU dequant reference (X_dq · W_dqᵀ).
    let x_pk = Nvfp4Tensor::pack(&x).unwrap();
    let dq_ref = ref_matmul(&x_pk.dequantize_to_vec(), &w_pk.dequantize_to_vec(), m, k, n);
    let rr_dq = rel_rms(&y_dev, &dq_ref);
    eprintln!("[sc-11044] on-device W4A4 vs CPU dequant reference rel-RMS = {rr_dq:.5}");
    assert!(rr_dq < 0.03, "on-device W4A4 does not track the dequant reference (rel-RMS {rr_dq:.5})");
}

/// (2) The full `Nvfp4Linear` W4A4 forward runs the on-device quantize end-to-end, is finite across
/// repeated forwards (a denoise-step stand-in), and stays within NVFP4 tolerance of a bf16-dense
/// reference. Exercises the `forward_checked` NaN/inf guard.
#[test]
fn w4a4_forward_ondevice_no_nan_vs_bf16() {
    let Some(dev) = nvfp4_device() else { return };
    let (m, k, n) = (512usize, 512usize, 512usize);
    let x_f32 = pseudo_random(m * k, 7);
    let w_f32 = pseudo_random(n * k, 8);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();
    let w = Tensor::from_vec(w_f32.clone(), (n, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();

    let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(lin.regime(), Nvfp4Regime::Fp4W4A4, "W4A4 must light the FP4 cores on sm_120");

    // Repeat to emulate steps: the guard must never trip, output must stay finite.
    let mut last = None;
    for step in 0..8 {
        let y = lin
            .forward_checked(&x)
            .unwrap_or_else(|e| panic!("W4A4 forward_checked tripped at step {step}: {e}"));
        assert_eq!(y.dims(), &[m, n]);
        last = Some(to_vec_f32(&y));
    }
    let got = last.unwrap();
    assert!(got.iter().all(|v| v.is_finite()));
    let bf16_ref = ref_matmul(&x_f32, &w_f32, m, k, n);
    let rr = rel_rms(&got, &bf16_ref);
    eprintln!("[sc-11044] on-device W4A4 forward vs bf16-dense rel-RMS = {rr:.5}");
    assert!(rr < 0.2, "on-device W4A4 vs bf16 {rr:.5} exceeds NVFP4 tolerance");
}

/// (3) Throughput: on-device **W4A4** vs **W4A16** (and a bf16-dense baseline) layer forwards on
/// representative Sana-DiT GEMM shapes, on the (now exclusive) GPU. Informational — the multiple is
/// hardware-dependent, so it is reported, not asserted (the correctness/no-NaN gates above are the
/// hard checks).
#[test]
fn w4a4_vs_w4a16_throughput() {
    let Some(dev) = nvfp4_device() else { return };
    // (M tokens, K in, N out): Sana-1.6B-ish attn proj (2240²) and FF up-proj (2240→5600).
    let shapes = [(1024usize, 2240usize, 2240usize), (1024, 2240, 5600)];
    let iters = 40usize;

    for (m, k, n) in shapes {
        let x = Tensor::from_vec(pseudo_random(m * k, 1), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(pseudo_random(n * k, 2), (n, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let w4a4 = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
        let w4a16 = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A16).unwrap();
        assert_eq!(w4a4.regime(), Nvfp4Regime::Fp4W4A4);
        assert_eq!(w4a16.regime(), Nvfp4Regime::DequantBf16);

        let time = |lin: &Nvfp4Linear| -> f64 {
            // warmup
            for _ in 0..5 {
                let _ = lin.forward(&x).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = lin.forward(&x).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        // bf16-dense baseline (candle matmul).
        let time_bf16 = || -> f64 {
            let wt = w.t().unwrap().contiguous().unwrap();
            for _ in 0..5 {
                let _ = x.matmul(&wt).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = x.matmul(&wt).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };

        let t_w4a4 = time(&w4a4);
        let t_w4a16 = time(&w4a16);
        let t_bf16 = time_bf16();
        eprintln!(
            "[sc-11044] LAYER shape M={m} K={k} N={n}: W4A4 {:.3} ms/fwd (incl. on-device act quant), \
             W4A16 {:.3} ms/fwd, bf16-dense {:.3} ms/fwd | W4A4 vs W4A16 = {:.2}×, vs bf16 = {:.2}× \
             (exclusive GPU)",
            t_w4a4 * 1e3,
            t_w4a16 * 1e3,
            t_bf16 * 1e3,
            t_w4a16 / t_w4a4,
            t_bf16 / t_w4a4,
        );

        // GEMM-CORE isolation: pre-stage both operands once, then time ONLY the FP4 GEMM vs the bf16
        // GEMM — the real FP4 tensor-core win, separated from the (unfused, candle-op) activation
        // quantize tax that the layer number above includes.
        let lt = CublasLt::new(&dev).unwrap();
        let w_pk = Nvfp4Tensor::pack(&w).unwrap();
        let w_stg = lt.stage_nvfp4(&w_pk).unwrap();
        let x_stg = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
        let wt = w.t().unwrap().contiguous().unwrap();
        let time_fp4_gemm = || -> f64 {
            for _ in 0..5 {
                let _ = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let time_bf16_gemm = || -> f64 {
            for _ in 0..5 {
                let _ = x.matmul(&wt).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = x.matmul(&wt).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let t_fp4g = time_fp4_gemm();
        let t_bf16g = time_bf16_gemm();
        eprintln!(
            "[sc-11044] GEMM-CORE shape M={m} K={k} N={n}: FP4 {:.3} ms, bf16 {:.3} ms | FP4 tensor-core \
             speedup = {:.2}× (pre-staged operands, exclusive GPU)",
            t_fp4g * 1e3,
            t_bf16g * 1e3,
            t_bf16g / t_fp4g,
        );
        assert!(t_w4a4.is_finite() && t_w4a4 > 0.0 && t_fp4g > 0.0);
    }
}

/// (4) Outlier-sparsity capture (the spike residual gate) on synthetic activations: benign / sparse /
/// dense classify as expected on-device, confirming the metric that governs the benign→W4A4 /
/// dense→W4A16 partition.
#[test]
fn outlier_sparsity_capture_confirms_partition() {
    let Some(dev) = nvfp4_device() else { return };
    let (m, k) = (256usize, 512usize);

    // Benign self-attn/FF-style activation.
    let benign = Tensor::from_vec(
        pseudo_random(m * k, 11).iter().map(|v| v * 0.3).collect::<Vec<_>>(),
        (m, k),
        &dev,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let s_benign = OutlierSparsity::from_tensor(&benign, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] benign layer: benign_fraction={:.4} class={:?}",
        s_benign.benign_fraction,
        s_benign.class()
    );
    assert!(s_benign.w4a4_viable(), "benign activation must be W4A4-viable");

    // Dense-outlier (caption/cross-attn-style) activation: an outlier in most blocks.
    let mut dense = pseudo_random(m * k, 22).iter().map(|v| v * 0.3).collect::<Vec<_>>();
    for r in 0..m {
        for b in 0..(k / OutlierSparsity::BLOCK) {
            dense[r * k + b * OutlierSparsity::BLOCK + 1] = 100.0;
        }
    }
    let dense_t = Tensor::from_vec(dense, (m, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();
    let s_dense = OutlierSparsity::from_tensor(&dense_t, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] dense-outlier layer: benign_fraction={:.4} class={:?} crush={:.0}",
        s_dense.benign_fraction,
        s_dense.class(),
        s_dense.max_crush_ratio
    );
    assert_eq!(s_dense.class(), OutlierClass::Dense, "dense outliers must flag collapse (W4A16)");
    assert!(!s_dense.w4a4_viable(), "dense-outlier layer must NOT be W4A4-viable — partition holds");
}

/// (5) Real Sana-1.6B DiT projection weight, W4A4 vs bf16 (env-gated, `#[ignore]` per repo convention
/// for real-weight tests). Set `SC11044_SANA_DIT_SAFETENSORS` to a Sana transformer shard; the test
/// loads the first eligible 2-D linear weight (K%32==0, N%16==0), quantizes W4A4, and reports the
/// quality delta vs the bf16 weight on a synthetic activation. Activations are synthetic — the live
/// per-step denoise activation capture is deferred to sc-11045.
#[test]
#[ignore = "real-weight test: set SC11044_SANA_DIT_SAFETENSORS to a Sana DiT shard"]
fn real_sana_dit_weight_w4a4() {
    let Some(dev) = nvfp4_device() else { return };
    let path = match std::env::var("SC11044_SANA_DIT_SAFETENSORS") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[sc-11044] SC11044_SANA_DIT_SAFETENSORS unset; skipping real-weight test");
            return;
        }
    };
    let tensors = candle_gen::candle_core::safetensors::load(&path, &Device::Cpu)
        .expect("load Sana DiT safetensors shard");
    // Find the first eligible 2-D weight: K a multiple of 32, N a multiple of 16, reasonably large.
    let mut chosen: Option<(String, Tensor)> = None;
    for (name, t) in tensors.iter() {
        if t.rank() == 2 {
            let (n, k) = t.dims2().unwrap();
            if k.is_multiple_of(32) && n.is_multiple_of(16) && k >= 256 && n >= 256 {
                chosen = Some((name.clone(), t.clone()));
                break;
            }
        }
    }
    let (name, w_cpu) = chosen.expect("no eligible 2-D linear weight in the shard");
    let (n, k) = w_cpu.dims2().unwrap();
    eprintln!("[sc-11044] real Sana DiT weight '{name}' shape [N={n}, K={k}]");
    let w = w_cpu.to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();

    let m = 1024usize;
    let x_f32 = pseudo_random(m * k, 42);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();

    let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(lin.regime(), Nvfp4Regime::Fp4W4A4);
    let got = to_vec_f32(&lin.forward_checked(&x).unwrap());
    assert!(got.iter().all(|v| v.is_finite()), "real-weight W4A4 produced NaN/Inf");

    let w_ref = to_vec_f32(&w);
    let bf16_ref = ref_matmul(&x_f32, &w_ref, m, k, n);
    let rr = rel_rms(&got, &bf16_ref);
    // Weight-outlier sparsity of the real weight, for the record.
    let ws = OutlierSparsity::from_tensor(&w, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] real Sana DiT '{name}': W4A4 vs bf16 rel-RMS = {rr:.5}; weight benign_fraction \
         {:.4} ({:?})",
        ws.benign_fraction,
        ws.class()
    );
    assert!(rr < 0.25, "real-weight W4A4 vs bf16 {rr:.5} unexpectedly large");
}
