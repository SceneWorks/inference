//! sc-11041 GPU gates for [`Nvfp4Linear`] — the NVFP4 FP4 linear path (`Nvfp4Linear` +
//! `MatmulStrategy::Nvfp4`) on consumer Blackwell `sm_120`. CUDA-only (needs a live cuBLASLt handle
//! **and** an `sm_120` device); a graceful no-op on CPU/Metal and on pre-Blackwell CUDA.
//!
//! What this pins on the real GPU:
//!
//! 1. **W4A4 output vs a bf16 reference linear** — `Nvfp4Linear` (default W4A4, FP4 cores lit) matches
//!    `x·Wᵀ` within NVFP4 tolerance, and tightly matches the CPU dequant reference.
//! 2. **SC#6 packed-forward (resident VRAM == NVFP4 footprint)** — the resident W4A4 weight occupies
//!    the ~4.5-eff-bit NVFP4 footprint on-device, **not** the bf16 size; proven by the staged
//!    device-byte accounting and cross-checked against a `mem_get_info` delta.
//! 3. **Non-aligned M handled** — arbitrary token counts (M∈{1,7,17,100}) forward without a cuBLASLt
//!    `NOT_SUPPORTED` (the layer pads M to `NVFP4_M_ALIGN` and slices back).
//! 4. **W4A16 override on capable hardware** — an explicit W4A16 request on `sm_120` still takes the
//!    dequant→bf16 regime (the outlier-class fallback), proving the policy flag is honored.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::nvfp4::Nvfp4Tensor;
use candle_gen::quant::{ActPrecision, Nvfp4Linear, Nvfp4Regime};

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
    use candle_gen::quant::cublaslt::CublasLt;
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-11041] no CUDA device; skipping Nvfp4Linear GPU gate");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_nvfp4_floor() {
        Ok(true) => {
            eprintln!(
                "[sc-11041] device compute cap = {:?} (NVFP4 eligible)",
                lt.compute_cap().unwrap()
            );
            Some(dev)
        }
        _ => {
            eprintln!(
                "[sc-11041] device cap {:?} < 12.0 (not sm_120); skipping Nvfp4Linear GPU gate",
                lt.compute_cap().ok()
            );
            None
        }
    }
}

/// (1) W4A4 `Nvfp4Linear` output matches a bf16 reference linear within NVFP4 tolerance, and tightly
/// matches the CPU dequant reference (the exact value the FP4 cores approximate).
#[test]
fn nvfp4_linear_w4a4_matches_bf16_reference() {
    let Some(dev) = nvfp4_device() else { return };
    let (m, k, n) = (256usize, 256usize, 256usize);

    let x_f32 = pseudo_random(m * k, 11);
    let w_f32 = pseudo_random(n * k, 22);
    let x_bf16 = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w_bf16 = Tensor::from_vec(w_f32.clone(), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let lin = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(
        lin.regime(),
        Nvfp4Regime::Fp4W4A4,
        "W4A4 on sm_120 must light up the FP4 cores, not fall back to bf16"
    );
    assert!(lin.lights_up_fp4());

    let y = lin.forward(&x_bf16).unwrap();
    assert_eq!(y.dims(), &[m, n]);
    let got = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "Nvfp4Linear produced NaN/Inf");

    // Tight: vs the CPU dequant reference (X_dq · W_dqᵀ) — what the FP4 GEMM computes.
    let w_pk = Nvfp4Tensor::pack(&w_bf16).unwrap();
    let x_pk = Nvfp4Tensor::pack(&x_bf16).unwrap();
    let dq_ref = ref_matmul(&x_pk.dequantize_to_vec(), &w_pk.dequantize_to_vec(), m, k, n);
    let rr_dq = rel_rms(&got, &dq_ref);
    eprintln!("[sc-11041] Nvfp4Linear vs CPU-dequant reference rel-RMS = {rr_dq:.5}");
    assert!(rr_dq < 0.02, "Nvfp4Linear does not track the dequant reference (rel-RMS {rr_dq:.5})");

    // Looser: vs the original bf16 dense matmul — within NVFP4 tolerance.
    let bf16_ref = ref_matmul(&x_f32, &w_f32, m, k, n);
    let rr_bf16 = rel_rms(&got, &bf16_ref);
    eprintln!("[sc-11041] Nvfp4Linear vs bf16-dense reference rel-RMS = {rr_bf16:.5}");
    assert!(rr_bf16 < 0.2, "Nvfp4Linear vs bf16 dense {rr_bf16:.5} exceeds NVFP4 tolerance");
}

/// (2) **SC#6 packed-forward.** The resident W4A4 weight occupies the NVFP4 footprint on-device
/// (packed nibbles + UE4M3 block scales), NOT the bf16 size. Proven by staged device-byte accounting
/// and cross-checked against a `mem_get_info` free-memory delta across the resident stage.
#[test]
fn nvfp4_linear_resident_vram_is_nvfp4_footprint() {
    use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
    let Some(dev) = nvfp4_device() else { return };
    // A large weight so the mem delta is well above allocator noise and the scale-atom padding is
    // negligible relative to the ~4.5-bit ideal.
    let (out_dim, in_dim) = (4096usize, 4096usize);
    let w_bf16 = Tensor::from_vec(pseudo_random(out_dim * in_dim, 5), (out_dim, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    dev.synchronize().unwrap();
    let (free_before, _total) = cuda::mem_get_info().unwrap();

    let lin = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(lin.regime(), Nvfp4Regime::Fp4W4A4);
    dev.synchronize().unwrap();
    let (free_after, _total) = cuda::mem_get_info().unwrap();

    let resident = lin
        .resident_device_bytes()
        .expect("W4A4 regime reports its resident FP4 device bytes");
    let nvfp4 = lin.nvfp4_footprint_bytes();
    let bf16 = lin.bf16_footprint_bytes();

    eprintln!(
        "[sc-11041] SC#6 resident VRAM: staged {resident} B, NVFP4 footprint {nvfp4} B, bf16 would be \
         {bf16} B (ratio {:.3})",
        resident as f64 / bf16 as f64
    );

    // The staged device weight is exactly the packed nibble + block-scale byte count (no bf16 expansion).
    assert_eq!(
        resident, nvfp4,
        "resident device bytes must equal the NVFP4 packed footprint (no dequant/expansion)"
    );
    // And that footprint is far below the bf16 size (~4.5 vs 16 bit) — the whole point of the format.
    assert!(
        (resident as f64) < 0.32 * bf16 as f64,
        "resident VRAM {resident} B is not ≈ the NVFP4 footprint vs bf16 {bf16} B — SC#6 violated"
    );

    // Driver cross-check (informational): the free-memory drop across construction. This includes the
    // cuBLASLt handle's 32 MiB workspace + allocator rounding, so it is NOT asserted (the deterministic
    // SC#6 proof is the byte-accounting above); it is reported to show the real VRAM movement is on the
    // order of the NVFP4 footprint + workspace, not a bf16 expansion of the weight.
    let free_drop = free_before.saturating_sub(free_after);
    eprintln!(
        "[sc-11041] SC#6 mem_get_info free drop across resident stage = {free_drop} B (NVFP4 weight \
         {nvfp4} B + ~32 MiB cuBLASLt workspace; a bf16 weight alone would be {bf16} B)"
    );
}

/// (3) **Non-aligned M handled.** Arbitrary token counts forward without a cuBLASLt `NOT_SUPPORTED`
/// (the layer pads M to `NVFP4_M_ALIGN` and slices the padding back off), and the real rows match the
/// dequant reference.
#[test]
fn nvfp4_linear_handles_non_aligned_m() {
    let Some(dev) = nvfp4_device() else { return };
    let (k, n) = (256usize, 128usize); // K_pad % 32 == 0, N % 16 == 0
    let w_bf16 = Tensor::from_vec(pseudo_random(n * k, 22), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let lin = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(lin.regime(), Nvfp4Regime::Fp4W4A4);
    let w_pk = Nvfp4Tensor::pack(&w_bf16).unwrap();

    for &m in &[1usize, 7, 17, 100] {
        let x_f32 = pseudo_random(m * k, 300 + m as u64);
        let x_bf16 = Tensor::from_vec(x_f32, (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let y = lin
            .forward(&x_bf16)
            .unwrap_or_else(|e| panic!("non-aligned M={m} forward failed (M-align not handled): {e}"));
        assert_eq!(y.dims(), &[m, n], "M={m} output shape");
        let got = y
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "M={m} produced NaN/Inf");

        // Real rows match the dequant reference (padding rows are sliced off and must not leak in).
        let x_pk = Nvfp4Tensor::pack(&x_bf16).unwrap();
        let dq_ref = ref_matmul(&x_pk.dequantize_to_vec(), &w_pk.dequantize_to_vec(), m, k, n);
        let rr = rel_rms(&got, &dq_ref);
        eprintln!("[sc-11041] non-aligned M={m:>3}: forward OK, rel-RMS vs dequant ref = {rr:.5}");
        assert!(rr < 0.03, "M={m} real rows do not match the dequant reference (rel-RMS {rr:.5})");
    }
}

/// (4) An explicit **W4A16** override on `sm_120` still takes the dequant→bf16 regime (the outlier-class
/// fallback), proving the mixed-precision policy flag is honored even where W4A4 is available.
#[test]
fn nvfp4_linear_w4a16_override_forces_dequant_on_sm120() {
    let Some(dev) = nvfp4_device() else { return };
    let (out_dim, in_dim) = (128usize, 256usize);
    let w_bf16 = Tensor::from_vec(pseudo_random(out_dim * in_dim, 9), (out_dim, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let lin = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A16).unwrap();
    assert_eq!(
        lin.regime(),
        Nvfp4Regime::DequantBf16,
        "W4A16 override must run the dequant→bf16 path (no FP4 compute), even on sm_120"
    );
    assert!(!lin.lights_up_fp4());
    assert!(lin.resident_device_bytes().is_none(), "W4A16 has no staged FP4 weight");

    // It still forwards coherently.
    let x = Tensor::from_vec(pseudo_random(4 * in_dim, 3), (4, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, out_dim]);
    assert!(y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}
