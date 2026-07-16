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
use candle_gen::quant::{ActPrecision, Nvfp4Context, Nvfp4Linear, Nvfp4Regime};

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

// ==============================================================================================
// (5) sc-12274 — the per-layer cuBLASLt workspace, measured in isolation.
// ==============================================================================================

/// **sc-12274 measurement: one `CublasLt` handle costs 32 MiB of VRAM, and `Nvfp4Linear` builds one
/// per layer.**
///
/// sc-12274 asserts, from the code alone, that `Nvfp4Linear::try_build_fp4` calls `CublasLt::new` per
/// layer and that each handle eagerly allocates a `CublasLt::WORKSPACE` (32 MiB) buffer it holds for
/// life — so a blanket-W4A4 Krea trunk's 260 W4A4 projections carry ~8.1 GiB of duplicated workspace
/// that the SC#6 byte-accounting (`resident_weight_bytes`, weights only) cannot see.
///
/// That is arithmetic off a code read. **This test is the reading**, isolated from any model: it grows
/// the handle count on a fixed, tiny weight and measures the driver's free-memory response. Two legs:
///
/// 1. **Bare handles** — `N × CublasLt::new` with no layers at all. Isolates the handle's own cost
///    from anything weight-related; the per-handle slope is the quantity sc-12274 multiplies by 260.
/// 2. **`Nvfp4Linear` layers** — the real construction path. Each layer's *weight* is deliberately
///    small (`[512, 512]` → ~144 KiB packed) so that if the per-layer VRAM cost lands near 32 MiB, the
///    only thing it can be is the workspace: the weight is ~0.4% of it.
///
/// 3. **The fix** — the same layers built through `from_dense_in` against **one** shared
///    [`Nvfp4Context`]. Per-layer VRAM must collapse to the packed weight alone, with a single 32 MiB
///    workspace for the whole set however many layers there are.
///
/// Weights-free (a fixed pseudo-random weight, no snapshot), so it runs in the normal CUDA lane. Legs
/// 2 and 3 together are the sc-12274 regression gate: leg 2 pins what the private-handle convenience
/// constructor still costs (by design — it is for one-off layers and tests), leg 3 pins that the
/// constructor a trunk loader uses does not.
#[test]
fn nvfp4_linear_shares_one_cublaslt_workspace_across_layers() {
    use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
    use candle_gen::quant::cublaslt::CublasLt;
    let Some(dev) = nvfp4_device() else { return };

    /// `CublasLt::WORKSPACE` — private to the handle, restated here as the *predicted* per-handle cost
    /// this test exists to confirm or refute against the driver.
    const WORKSPACE: usize = 32 * 1024 * 1024;
    const N: usize = 32;
    let mib = |b: f64| b / (1024.0 * 1024.0);

    let free_at = || {
        dev.synchronize().unwrap();
        let (free, _total) = cuda::mem_get_info().unwrap();
        free
    };

    eprintln!("\n[sc-12274] ===== PER-LAYER cuBLASLt WORKSPACE (isolated, weights-free) =====");
    eprintln!("[sc-12274] predicted CublasLt::WORKSPACE = {} B (32 MiB)", WORKSPACE);

    // --- Leg 1: bare handles ------------------------------------------------------------------
    let base = free_at();
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        handles.push(CublasLt::new(&dev).expect("cuBLASLt handle"));
        if i == 0 || i == N - 1 {
            let used = base.saturating_sub(free_at()) as f64;
            eprintln!(
                "[sc-12274] leg1 bare handles: {:>3} handle(s) → {:>9.2} MiB used ({:>7.2} MiB/handle)",
                i + 1,
                mib(used),
                mib(used) / (i + 1) as f64
            );
        }
    }
    let leg1_used = base.saturating_sub(free_at()) as f64;
    let leg1_per_handle = leg1_used / N as f64;
    drop(handles);
    let leg1_reclaimed = free_at() >= base.saturating_sub(WORKSPACE);
    eprintln!(
        "[sc-12274] leg1 TOTAL: {N} bare handles = {:.2} MiB → {:.2} MiB/handle (predicted {:.2}); \
         freed-on-drop = {leg1_reclaimed}",
        mib(leg1_used),
        mib(leg1_per_handle),
        mib(WORKSPACE as f64)
    );

    // --- Leg 2: Nvfp4Linear layers (the real path) --------------------------------------------
    // A deliberately TINY weight: [512, 512] bf16 packs to ~144 KiB (0.5625 B/wt). If the measured
    // per-layer cost is ~32 MiB, the weight cannot account for it — the workspace is the only
    // candidate. K=512 and N=512 clear the FP4 shape gate (K_pad % 32 == 0, N % 16 == 0).
    let (out_dim, in_dim) = (512usize, 512usize);
    let w_bf16 = Tensor::from_vec(pseudo_random(out_dim * in_dim, 77), (out_dim, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let base2 = free_at();
    let mut layers = Vec::with_capacity(N);
    for _ in 0..N {
        let lin = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A4).unwrap();
        assert_eq!(
            lin.regime(),
            Nvfp4Regime::Fp4W4A4,
            "the measurement is only meaningful if the layers actually take the FP4 path"
        );
        layers.push(lin);
    }
    let leg2_used = base2.saturating_sub(free_at()) as f64;
    let leg2_per_layer = leg2_used / N as f64;

    let packed_per_layer = layers[0].resident_device_bytes().expect("W4A4 stages a packed weight");
    let accounted = (packed_per_layer * N) as f64; // what SC#6's `resident_weight_bytes` would sum
    let unaccounted = leg2_used - accounted;

    eprintln!(
        "[sc-12274] leg2 Nvfp4Linear x{N} ([{out_dim},{in_dim}] each): {:.2} MiB used → {:.2} MiB/layer",
        mib(leg2_used),
        mib(leg2_per_layer)
    );
    eprintln!(
        "[sc-12274] leg2   of which SC#6 byte-accounting SEES (packed weights): {:.2} MiB ({:.2} MiB/layer)",
        mib(accounted),
        mib(packed_per_layer as f64)
    );
    eprintln!(
        "[sc-12274] leg2   of which SC#6 byte-accounting MISSES:               {:.2} MiB ({:.2} MiB/layer)",
        mib(unaccounted),
        mib(unaccounted / N as f64)
    );
    eprintln!(
        "[sc-12274] leg2   workspace share of resident VRAM = {:.1}% — the SC#6 number counts the other {:.1}%",
        100.0 * unaccounted / leg2_used,
        100.0 * accounted / leg2_used
    );

    // --- The sc-12274 extrapolation, stated with the measured slope ---------------------------
    let krea_blanket = leg2_per_layer * 260.0 - (packed_per_layer * 260) as f64;
    eprintln!(
        "[sc-12274] EXTRAPOLATION: at this per-layer overhead, a 260-projection blanket-W4A4 Krea \
         trunk would carry {:.2} GiB of workspace invisible to SC#6 (story asserts ~8.12 GiB; the \
         real trunk measures ~6.6 GiB — see nvfp4_krea_dit_sc6_cublaslt_workspace_gap)",
        krea_blanket / (1024.0 * 1024.0 * 1024.0)
    );
    drop(layers);

    // --- Leg 3: THE FIX — the same N layers against ONE shared context -------------------------
    let base3 = free_at();
    let ctx = Nvfp4Context::new(&dev).expect("shared cuBLASLt context");
    assert!(
        ctx.is_fp4(),
        "an sm_120 device must yield a live FP4 context, else leg 3 proves nothing"
    );
    let mut shared: Vec<Nvfp4Linear> = Vec::with_capacity(N);
    for _ in 0..N {
        let lin = Nvfp4Linear::from_dense_in(&w_bf16, None, &dev, ActPrecision::W4A4, &ctx).unwrap();
        assert_eq!(
            lin.regime(),
            Nvfp4Regime::Fp4W4A4,
            "the shared-context layers must still light the FP4 cores — a fix that silently dropped \
             every layer to the bf16 fallback would also 'save' the workspace"
        );
        shared.push(lin);
    }
    let leg3_used = base3.saturating_sub(free_at()) as f64;
    let leg3_per_layer = leg3_used / N as f64;
    // One handle for the whole set, not one each.
    let leg3_workspace = leg3_used - accounted;

    eprintln!(
        "[sc-12274] leg3 SHARED ctx x{N}: {:.2} MiB used → {:.2} MiB/layer (vs {:.2} MiB/layer private)",
        mib(leg3_used),
        mib(leg3_per_layer),
        mib(leg2_per_layer)
    );
    eprintln!(
        "[sc-12274] leg3   non-weight VRAM for ALL {N} layers: {:.2} MiB (one shared workspace; \
         private-handle path would be {:.2} MiB)",
        mib(leg3_workspace),
        mib(unaccounted)
    );
    eprintln!(
        "[sc-12274] leg3   SAVED: {:.2} MiB over {N} layers ({:.1}× less non-weight VRAM)",
        mib(leg2_used - leg3_used),
        unaccounted / leg3_workspace.max(1.0)
    );

    // --- Assertions: the mechanism, not the exact byte count -----------------------------------
    // Leg 1 pins the handle's own cost at ~WORKSPACE. Tolerance is generous (±25%) because the handle
    // also allocates cuBLASLt-internal state and the driver rounds allocations; the claim under test is
    // "each handle costs a 32 MiB-scale buffer", not a byte-exact figure.
    assert!(
        leg1_per_handle > 0.75 * WORKSPACE as f64 && leg1_per_handle < 1.25 * WORKSPACE as f64,
        "a bare CublasLt handle measured {:.2} MiB, expected ~{:.2} MiB (CublasLt::WORKSPACE). \
         sc-12274's arithmetic rests on this being ~32 MiB — if it is not, the story's headline is wrong.",
        mib(leg1_per_handle),
        mib(WORKSPACE as f64)
    );
    // Leg 2: the private-handle constructor still costs a whole workspace per layer. That is BY DESIGN
    // (`from_dense` is for a one-off layer / a test) — pinned so the cost stays visible and nobody
    // reaches for it in a loader.
    assert!(
        leg2_per_layer > 0.75 * WORKSPACE as f64,
        "`from_dense` builds a private handle per layer, so it should still measure ~32 MiB/layer; \
         got {:.2} MiB. If this dropped, the private/shared distinction has been lost.",
        mib(leg2_per_layer)
    );
    assert!(
        unaccounted > 10.0 * accounted,
        "sc-12274's claim is that a per-layer workspace DWARFS the packed weights it serves; measured \
         unaccounted {:.2} MiB vs accounted {:.2} MiB",
        mib(unaccounted),
        mib(accounted)
    );
    // Leg 3: THE FIX. `from_dense_in` must add at most ONE workspace for the whole set, however many
    // layers — so total non-weight VRAM stays ~32 MiB rather than N × 32 MiB. This is the assertion
    // that fails if anyone reintroduces a per-layer `CublasLt::new`.
    assert!(
        leg3_workspace < 1.5 * WORKSPACE as f64,
        "a shared Nvfp4Context must cost ONE {:.2} MiB workspace for all {N} layers, but the \
         non-weight VRAM was {:.2} MiB (≈{:.1} workspaces) — the handle is not actually being shared \
         (sc-12274 regression).",
        mib(WORKSPACE as f64),
        mib(leg3_workspace),
        leg3_workspace / WORKSPACE as f64
    );
    assert!(
        leg3_per_layer < 0.25 * leg2_per_layer,
        "sharing the handle must collapse per-layer VRAM toward the weight alone: shared \
         {:.2} MiB/layer vs private {:.2} MiB/layer is not a real saving",
        mib(leg3_per_layer),
        mib(leg2_per_layer)
    );

    // --- Leg 4: sharing must not change a single bit --------------------------------------------
    // The one genuinely shared MUTABLE resource is the 32 MiB scratch that every matmul on the handle
    // writes. Every handle on a device already resolves to the same stream (`CublasLt::new` takes
    // `device.cuda_stream()`), and a stream serializes its kernels, so reuse *should* be safe — but
    // that is a reasoning chain, and this is the check. Drive all N shared layers through a forward so
    // the scratch is genuinely reused across layers, and require every output to be bit-identical to a
    // fresh private-handle layer holding the same weight.
    let x = Tensor::from_vec(pseudo_random(64 * in_dim, 5150), (64, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bits = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let private = Nvfp4Linear::from_dense(&w_bf16, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(private.regime(), Nvfp4Regime::Fp4W4A4);
    let reference = bits(&private.forward(&x).unwrap());
    assert!(
        reference.iter().all(|v| v.is_finite()) && reference.iter().any(|v| *v != 0.0),
        "the reference forward must produce real finite output, else bit-equality is vacuous"
    );
    for (i, lin) in shared.iter().enumerate() {
        let got = bits(&lin.forward(&x).unwrap());
        assert_eq!(
            got, reference,
            "shared-workspace layer {i}/{N} diverged from the private-handle reference. Identical \
             weight + identical input MUST give identical output; a difference here means the shared \
             32 MiB cuBLASLt scratch is being clobbered across layers (sc-12274)."
        );
    }
    eprintln!(
        "[sc-12274] leg4 NUMERICS: all {N} shared-workspace layers bit-identical to the \
         private-handle reference ({} elems) — scratch reuse across layers is safe on one stream",
        reference.len()
    );
    drop(shared);
}
