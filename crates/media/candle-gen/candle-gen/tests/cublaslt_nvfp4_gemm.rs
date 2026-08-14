//! sc-11039 device round-trip + throughput gate for the **NVFP4 block-scaled FP4 GEMM**
//! (`CublasLt::matmul_nvfp4`), the primary FP4 compute path of epic 11037. CUDA-only (needs a live
//! cuBLASLt handle **and** an `sm_120` device); a graceful no-op on CPU/Metal, on pre-Blackwell CUDA,
//! and where cuBLASLt does not dispatch an FP4 kernel.
//!
//! What this pins on the real GPU:
//!
//! 1. **Round-trip correctness** — pack a bf16 weight `[N,K]` + activation `[M,K]` to NVFP4
//!    (sc-11040 [`Nvfp4Tensor`]), run the FP4 GEMM, and compare against the **CPU dequant reference**
//!    (`X_dq · W_dqᵀ`, the exact value the FP4 cores approximate). A tight match confirms the GEMM
//!    consumes the packed nibbles + the row-major-atom UE4M3 scale tensor + the folded FP32 per-tensor
//!    scale correctly — including the **multi-row-atom** shapes (>128 rows) that are the only regime
//!    exercising the scale-atom tiling order (**handoff item (a)**). Also asserts no NaN/Inf.
//! 2. **NVFP4 end-to-end error** vs the original bf16 dense matmul stays within NVFP4 tolerance.
//! 3. **The staged, algo-cached FP4 path** on a compute-bound DiT shape — that cuBLASLt dispatches
//!    an FP4 kernel for it at all, and that the result is still numerically right there, gated
//!    both in aggregate (rel-RMS) and pointwise (rel-max-abs, which is what can see a fault
//!    localised to a single scale atom). The throughput multiple against bf16 dense is
//!    **reported, not gated** (the spike saw 1.9–3.7×); see the note in
//!    `nvfp4_gemm_throughput_vs_bf16` for why a wall-clock ratio cannot be an assertion
//!    (sc-19505). ⚠️ That demotion **gives up** the "is it faster" half of this item — a
//!    driver/algo regression returning a working-but-slow FP4 algo is now gated nowhere, and
//!    needs an algo-identity instrument to recover (sc-19556).
//! 4. **K-alignment** (**handoff item (b)**) — the GPU-confirmed requirement is K a multiple of 32
//!    (K∈{16,48} → `NOT_SUPPORTED`; K∈{32,64,128} accepted); this pins the enforced bound.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::cublaslt::CublasLt;
use candle_gen::quant::nvfp4::Nvfp4Tensor;

/// splitmix64-hashed deterministic pseudo-random in ~[-1, 1] (no device RNG; launch-portable).
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

/// CPU f32 reference `X · Wᵀ` for `X=[M,K]`, `W=[N,K]` given as flat row-major slices → `[M,N]`.
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

/// A CUDA device + handle iff one exists and it is Blackwell `sm_120`+ (cap ≥ 12.0). Returns `None`
/// (with a SKIP note) otherwise — the NVFP4 FP4 path is only defined there.
fn nvfp4_device() -> Option<(Device, CublasLt)> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-11039] no CUDA device; skipping NVFP4 GEMM gate");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_nvfp4_floor() {
        Ok(true) => {
            eprintln!(
                "[sc-11039] device compute cap = {:?} (NVFP4 eligible)",
                lt.compute_cap().unwrap()
            );
            Some((dev, lt))
        }
        _ => {
            eprintln!(
                "[sc-11039] device cap {:?} < 12.0 (not sm_120); skipping NVFP4 GEMM gate",
                lt.compute_cap().ok()
            );
            None
        }
    }
}

/// Pack + run the FP4 GEMM for one shape and return `(got, dequant_ref)`.
fn run_gemm(dev: &Device, lt: &CublasLt, m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let x_bf16 = Tensor::from_vec(pseudo_random(m * k, 11), (m, k), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w_bf16 = Tensor::from_vec(pseudo_random(n * k, 22), (n, k), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w_pk = Nvfp4Tensor::pack(&w_bf16).unwrap();
    let x_pk = Nvfp4Tensor::pack(&x_bf16).unwrap();
    let d = lt.matmul_nvfp4(&w_pk, &x_pk).unwrap();
    let got = d
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let dq_ref = ref_matmul(
        &x_pk.dequantize_to_vec(),
        &w_pk.dequantize_to_vec(),
        m,
        k,
        n,
    );
    (got, dq_ref)
}

#[test]
fn nvfp4_gemm_roundtrip_vs_bf16_dense() {
    let Some((dev, lt)) = nvfp4_device() else {
        return;
    };

    // Shapes chosen to exercise: a single scale atom, >128 weight rows (2 row-atoms on A), and >128
    // activation rows (2 row-atoms on B) — the multi-atom regime that pins the scale tiling order.
    for &(m, k, n) in &[
        (64usize, 256usize, 128usize), // single 128-row atom
        (64, 256, 256),                // weight has 2 row-atoms (handoff a)
        (256, 256, 256),               // both operands have 2 row-atoms
        (256, 512, 128),               // 4 col-atoms on K
    ] {
        let (got, dq_ref) = run_gemm(&dev, &lt, m, k, n);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "NVFP4 GEMM produced NaN/Inf at ({m},{k},{n})"
        );
        let rr = rel_rms(&got, &dq_ref);
        eprintln!("[sc-11039] round-trip ({m}x{k}x{n}) vs CPU-dequant reference rel-RMS = {rr:.5}");
        assert!(
            rr < 0.02,
            "NVFP4 GEMM does not match the dequant reference at ({m},{k},{n}) (rel-RMS {rr:.5}) — \
             the block-scale atom order / layout is being consumed incorrectly (handoff item a)"
        );
    }

    // End-to-end NVFP4 error vs the original bf16 dense matmul — within NVFP4 tolerance.
    let (m, k, n) = (256usize, 256usize, 256usize);
    let (got, _) = run_gemm(&dev, &lt, m, k, n);
    let bf16_ref = ref_matmul(
        &pseudo_random(m * k, 11),
        &pseudo_random(n * k, 22),
        m,
        k,
        n,
    );
    let rr_e2e = rel_rms(&got, &bf16_ref);
    eprintln!("[sc-11039] NVFP4 GEMM vs bf16-dense end-to-end rel-RMS = {rr_e2e:.5}");
    // Uniform-random [-1,1] is a worst-case for NVFP4 (per-block dynamic range is large); the tight
    // gate is the dequant-reference check above. This bound just asserts "not garbage / not scrambled".
    assert!(
        rr_e2e < 0.2,
        "NVFP4 end-to-end error {rr_e2e:.5} exceeds NVFP4 tolerance vs bf16 dense"
    );
}

#[test]
fn nvfp4_gemm_throughput_vs_bf16() {
    let Some((dev, lt)) = nvfp4_device() else {
        return;
    };
    use std::time::Instant;

    // A compute-bound DiT projection shape.
    let (m, k, n) = (1024usize, 4096usize, 4096usize);
    let x_bf16 = Tensor::from_vec(pseudo_random(m * k, 7), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w_bf16 = Tensor::from_vec(pseudo_random(n * k, 8), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Pre-stage the FP4 operands (weight resident, activation staged once — the honest compute path).
    let w_stg = lt
        .stage_nvfp4(&Nvfp4Tensor::pack(&w_bf16).unwrap())
        .unwrap();
    let x_stg = lt
        .stage_nvfp4(&Nvfp4Tensor::pack(&x_bf16).unwrap())
        .unwrap();

    let iters = 100;
    // Warm up both paths (the FP4 warmup also primes the per-shape algo cache so the loop times the
    // kernel, not the one-off cuBLASLt heuristic search).
    let _ = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
    let _ = x_bf16.matmul(&w_bf16.t().unwrap()).unwrap();
    dev.synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
    }
    dev.synchronize().unwrap();
    let fp4_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = x_bf16.matmul(&w_bf16.t().unwrap()).unwrap();
    }
    dev.synchronize().unwrap();
    let bf16_s = t1.elapsed().as_secs_f64();

    let mult = bf16_s / fp4_s.max(1e-9);
    eprintln!(
        "[sc-11039] throughput ({m}x{k}x{n}, {iters} iters): FP4 {:.3} ms/it, bf16 {:.3} ms/it → {mult:.2}× faster",
        1e3 * fp4_s / iters as f64,
        1e3 * bf16_s / iters as f64
    );
    // `mult` is REPORTED, never asserted on (sc-19505). It is a ratio of two `Instant::elapsed()`
    // values, and the old gate — a bare `mult > 1.0` — had no margin whatsoever: any contended
    // runner that happened to deschedule the FP4 loop rather than the bf16 one flips it, and the
    // failure lands on whatever PR is in flight. "Robust across driver/algo variation" is what a
    // threshold with zero headroom looks like from the inside; sc-19452 measured the same shape one
    // repo over and found it fails in *both* directions, false-red and false-green alike.
    //
    // The failure this test names — "cuBLASLt may not be dispatching the FP4 tensor-core kernel" —
    // is already gated deterministically, and by the lines above rather than by the clock:
    // `matmul_nvfp4_staged` runs the cuBLASLt heuristic search against the FP4 descriptor, and an
    // empty result surfaces as an `Err` (`quant/cublaslt.rs`, the `get_matmul_algo_heuristic`
    // arm) — so a device that cannot deliver an FP4 kernel panics at those `.unwrap()`s and never
    // reaches a timer. `nvfp4_device()` has already screened out the pre-Blackwell case above.
    //
    // ⚠️ But one real claim DOES lapse here, and it is not noise. `mult > 1.0` also covered a
    // driver/algo regression in which cuBLASLt returns *an* FP4 algo that is SLOWER than bf16
    // dense. The `.unwrap()`s above cannot see that — a slow algo is still an algo, and the
    // heuristic search succeeds. That is this file's own published item 3, "is it faster"
    // (sc-11039), and after this change it is gated NOWHERE. It is given up deliberately, not
    // covered by something else: re-gating it needs an instrument that names what the heuristic
    // actually selected — an algo-identity / kernel-name assertion — rather than a throughput
    // ratio, which is unfixable on a shared runner at any threshold. Tracked in sc-19556, and not
    // attempted here because no lane can execute this file to validate a replacement (below).
    //
    // What the clock did carry, and the unwraps do not, is that this staged/algo-cached path is
    // still numerically right at a DiT shape — the round-trip test only reaches 256³ through the
    // *unstaged* entry point. So gate that directly instead, against the bf16 dense output on the
    // same operands, at the same NVFP4 end-to-end tolerance the round-trip test uses.
    let fp4 = lt
        .matmul_nvfp4_staged(&w_stg, &x_stg)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let dense = x_bf16
        .matmul(&w_bf16.t().unwrap())
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(fp4.len(), m * n, "staged FP4 GEMM returned the wrong shape");
    assert!(
        fp4.iter().all(|v| v.is_finite()),
        "staged NVFP4 GEMM produced NaN/Inf at ({m},{k},{n})"
    );
    // A mis-dispatched or no-op kernel most plausibly returns a constant (usually zero) buffer, and
    // a relative bound alone would accept that whenever the reference is small.
    let lo = fp4.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = fp4.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi > lo,
        "staged NVFP4 GEMM returned a constant buffer ({lo}) at ({m},{k},{n}) — the kernel ran but \
         computed nothing"
    );
    // rel-RMS alone is the BLIND instrument for the defect this shape exists to expose. It is a
    // global averaging norm over m*n = 4.19M elements, so a fault confined to one 64-column scale
    // atom out of 4096 — precisely the multi-row-atom / scale-atom tiling-order regime of handoff
    // item (a) — touches only f = 64/4096 = 1/64 of the output and moves rel-RMS by at most
    // sqrt(f) = 0.125 if that atom is zeroed, or sqrt(2f) = 0.177 if it holds garbage at reference
    // magnitude. Both clear the 0.2 bound below while that atom is entirely wrong.
    //
    // So gate the POINTWISE error alongside it, on the epic's convention — max|diff| normalised by
    // the reference's own peak (`real_weights.rs`'s `rel_max` / `peak`). Correct NVFP4 noise is
    // spread homogeneously and lands near rel-RMS; a localised atom fault puts this at ~0.85,
    // because the bad block's extreme and the matrix's extreme are drawn from the same
    // distribution. The two bounds fail on disjoint defect shapes, which is the point of keeping
    // both.
    let rr = rel_rms(&fp4, &dense);
    let peak = dense.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel_max = fp4
        .iter()
        .zip(&dense)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / peak;
    eprintln!(
        "[sc-11039] staged FP4 vs bf16 dense ({m}x{k}x{n}): rel-RMS = {rr:.5}, rel-max-abs = {rel_max:.5}"
    );
    // ⚠️ NEITHER BOUND IS MEASURED AT THIS SHAPE, and no CI lane can measure them. The 0.2 is this
    // file's own NVFP4 end-to-end tolerance from `nvfp4_gemm_round_trip_and_error`, where it was
    // calibrated at 256x256x256; reusing it at 1024x4096x4096 rests on NVFP4's error being
    // per-block and therefore not growing with K — an argument, not a measurement. The 0.5 is
    // derived from the separation reasoned out above (correct path near rel-RMS, atom fault ~0.85),
    // also not from hardware. The `windows-cuda-check` lane compiles and clippies `--features cuda`
    // but has no sm_120 device, so `nvfp4_device()` returns `None` there and this body never runs.
    // The first real execution will be someone's Blackwell box. CONFIRM AND RETUNE BOTH ON THAT
    // FIRST sm_120 RUN (sc-19556) — a red there is as likely to be an unvalidated constant as a
    // regression, and must not be read as the latter without checking.
    assert!(
        rr < 0.2,
        "staged NVFP4 GEMM differs from bf16 dense by rel-RMS {rr:.5} at ({m},{k},{n}) — beyond \
         NVFP4 tolerance, so the algo-cached staged path is not computing this shape correctly"
    );
    assert!(
        rel_max < 0.5,
        "staged NVFP4 GEMM's worst pointwise error is {rel_max:.5} of the reference peak at \
         ({m},{k},{n}) (rel-RMS {rr:.5} — if that one is small, the error is LOCALISED, which is \
         the scale-atom tiling-order signature of handoff item (a))"
    );
}

/// **Handoff item (b): operand-K alignment.** Pins the GPU-confirmed requirement — cuBLASLt's FP4
/// block-scaled path accepts K∈{32,64,128} (bit-accurate) and rejects K∈{16,48}
/// (`CUBLAS_STATUS_NOT_SUPPORTED`), i.e. **K must be a multiple of 32** (two NVFP4 blocks), not the
/// single 16-element block nor the 64-element scale atom. The wrapper enforces this (`NVFP4_K_ALIGN`),
/// so K∈{16,48} now surface as a clear wrapper error before the cuBLASLt call; K∈{32,64,128} run and
/// match the dequant reference.
#[test]
fn nvfp4_k_alignment_probe() {
    let Some((dev, lt)) = nvfp4_device() else {
        return;
    };
    let (m, n) = (32usize, 64usize);
    let mut accepted = Vec::new();
    for &k in &[16usize, 32, 48, 64, 128] {
        let x_bf16 = Tensor::from_vec(pseudo_random(m * k, 100 + k as u64), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_bf16 = Tensor::from_vec(pseudo_random(n * k, 200 + k as u64), (n, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_pk = Nvfp4Tensor::pack(&w_bf16).unwrap();
        let x_pk = Nvfp4Tensor::pack(&x_bf16).unwrap();
        match lt.matmul_nvfp4(&w_pk, &x_pk) {
            Ok(d) => {
                let got = d
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let dq_ref = ref_matmul(
                    &x_pk.dequantize_to_vec(),
                    &w_pk.dequantize_to_vec(),
                    m,
                    k,
                    n,
                );
                let rr = rel_rms(&got, &dq_ref);
                let ok = got.iter().all(|v| v.is_finite()) && rr < 0.05;
                eprintln!(
                    "[sc-11039] K={k:>3}: ACCEPTED (rel-RMS vs dequant {rr:.5}, {})",
                    if ok { "correct" } else { "WRONG numerics" }
                );
                if ok {
                    accepted.push(k);
                }
            }
            Err(e) => {
                eprintln!("[sc-11039] K={k:>3}: REJECTED ({e})");
            }
        }
    }
    eprintln!("[sc-11039] NVFP4 accepted-K set (correct numerics) = {accepted:?}");
    assert_eq!(
        accepted,
        vec![32, 64, 128],
        "NVFP4 K-alignment must be multiples of 32 (K=16,48 rejected); accepted = {accepted:?}"
    );
}
