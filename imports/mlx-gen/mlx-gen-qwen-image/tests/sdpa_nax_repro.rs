//! sc-2770: tripwire verifying the NAX 16-bit fast SDPA kernel is correct on the pinned build.
//!
//! BACKGROUND. On the pmetal NAX builds (macOS-26 / Metal-400), upstream MLX's
//! `fast::scaled_dot_product_attention` — the fused-attention analogue of the sc-2714 dense-GEMM
//! bug — returned GARBAGE for 16-bit q/k/v (both bf16 *and* f16): mean-relative error vs an f32
//! reference ~1.1 (right scale, uncorrelated, no NaN). f32 was always correct. Validated same-build:
//! the bug is DTYPE-driven, not layout-driven — f32 is correct in BOTH contiguous and transposed-view
//! layouts; 16-bit is garbage in BOTH; shape-independent (L = 64..1024 identical). A manual
//! `softmax(q·kᵀ·scale)·v` reference (using the sc-2714-fixed matmul + softmax) is correct at every
//! dtype, so the garbage is the fused KERNEL, not the reference. Root cause:
//! `scaled_dot_product_attention.cpp` routed all 16-bit into the broken `sdpa_full_self_attention_nax`
//! (`get_steel_attention_nax_kernel`) via the same `enable_tf32() || dtype != float32` dispatch as the
//! GEMM bug. See memory `pmetal-mlx-bf16-matmul-bug`.
//!
//! FIX (sc-2772, root cause). The garbage was NOT the dispatch and NOT the MLX version — it was the
//! metal-kernel COMPILE TARGET. The NAX kernels use Apple's matrix-unit intrinsic
//! `mpp::tensor_ops::matmul2d`, valid only for macOS >= 26.2 (the same floor `is_nax_available()`
//! checks). The kernels were being compiled with `-mmacosx-version-min=26.0` (the old
//! MACOSX_DEPLOYMENT_TARGET), below that floor, so metalfe miscompiled the tensor-op to garbage for
//! 16-bit while the host still dispatched to it. Building the kernels at 26.2 (mlx-gen's
//! .cargo/config.toml) makes the NAX 16-bit SDPA correct — proven byte-for-byte vs Apple's official
//! wheel metallib. The earlier sc-2770 dispatch-gate patch (route 16-bit off NAX) was a WORKAROUND and
//! is now REMOVED; 16-bit uses the correct, faster NAX matrix unit.
//!
//! THIS TEST is the per-build guarantor that the NAX 16-bit SDPA kernel is correct: it sweeps
//! representative attention shapes × layouts × dtypes and asserts the 16-bit `fast` kernel matches an
//! f32 ground-truth manual attention (≈16-bit rounding, not garbage). It FAILS on a NAX build whose
//! metal kernels were compiled below the 26.2 floor (e.g. MACOSX_DEPLOYMENT_TARGET regressed to 26.0).
//! On non-NAX builds 16-bit SDPA uses correct fallback kernels, so it passes there too. Needs no
//! weights, only MLX. Run: `cargo test -p mlx-gen-qwen-image --release --test sdpa_nax_repro
//! -- --nocapture`.

use mlx_gen::array::scalar;
use mlx_rs::{
    fast::scaled_dot_product_attention,
    ops::{matmul, multiply, softmax_axis},
    random, Array, Dtype,
};

/// Mean-absolute relative error: sum|a-b| / sum|b|, computed in f32.
fn rel(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).sum();
    let den: f64 = b.iter().map(|y| y.abs() as f64).sum();
    num / den
}

/// manual attention on logical `[1,N,L,D]`: softmax((q @ kᵀ) * scale, -1) @ v. Uses the
/// sc-2714-fixed matmul + softmax, so it is the correct reference at every dtype.
fn manual_attn(q: &Array, k: &Array, v: &Array, scale: f32) -> Array {
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let scores = multiply(matmul(q, &kt).unwrap(), scalar(scale)).unwrap();
    let probs = softmax_axis(&scores, -1, true).unwrap();
    matmul(&probs, v).unwrap()
}

// Always-on guard: with the kernels compiled at macOS >= 26.2 (sc-2772), 16-bit `fast` SDPA is
// correct, so this asserts correctness on every build (NAX or non-NAX). On a NAX build whose metal
// kernels were compiled below 26.2 it (rightly) FAILS. Mirrors
// `bf16_matmul_sweep::nax_16bit_dense_gemm_is_correct`.
#[test]
fn nax_16bit_sdpa_is_correct() {
    // (heads N, seq L, head_dim D) — cover D=128 and D=64, short and long sequences.
    let shapes = [(24i32, 64i32, 128i32), (8, 256, 64), (16, 1024, 64)];
    let dtypes = [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16];

    let mut worst_16bit = 0.0f64; // 16-bit `fast` vs f32 ground truth — must be LOW (patched)
    let mut worst_f32 = 0.0f64; // f32 `fast` vs f32 ground truth — NAX f32 path, always correct
    let mut worst_manual = 0.0f64; // any-dtype manual vs f32 ground truth — validates the reference

    println!(
        "  fast/man vs f32 ground truth (rel). fast 16-bit > 0.1 => NAX fast-SDPA kernel wrong."
    );
    println!("  shape[N,L,D]   layout       dtype       fast      man");
    for (n, l, d) in shapes {
        let scale = (d as f32).powf(-0.5);
        let (kq, kk, kv) = (
            random::key(0).unwrap(),
            random::key(1).unwrap(),
            random::key(2).unwrap(),
        );
        for layout in ["contiguous", "transp-view"] {
            // Build f32 q,k,v with logical shape [1,N,L,D] in each physical layout.
            let (qf, kf, vf) = if layout == "contiguous" {
                let g = |key| random::normal::<f32>(&[1, n, l, d], None, None, Some(key)).unwrap();
                (g(&kq), g(&kk), g(&kv))
            } else {
                // physical [1,L,N,D] -> transpose to logical [1,N,L,D] (strided view)
                let g = |key| {
                    random::normal::<f32>(&[1, l, n, d], None, None, Some(key))
                        .unwrap()
                        .transpose_axes(&[0, 2, 1, 3])
                        .unwrap()
                };
                (g(&kq), g(&kk), g(&kv))
            };
            let gt = manual_attn(&qf, &kf, &vf, scale); // f32 ground truth
            for dt in dtypes {
                let (q, k, v) = (
                    qf.as_dtype(dt).unwrap(),
                    kf.as_dtype(dt).unwrap(),
                    vf.as_dtype(dt).unwrap(),
                );
                let fast = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
                let man = manual_attn(&q, &k, &v, scale);
                let (r_fast, r_man) = (rel(&fast, &gt), rel(&man, &gt));
                println!(
                    "  [{n:>2},{l:>4},{d:>3}]  {layout:<11}  {dt:?}  {r_fast:>9.4}  {r_man:>7.4}"
                );
                worst_manual = worst_manual.max(r_man);
                if dt == Dtype::Float32 {
                    worst_f32 = worst_f32.max(r_fast);
                } else {
                    worst_16bit = worst_16bit.max(r_fast);
                }
            }
        }
    }
    println!(
        "max 16-bit fast rel: {worst_16bit:.4}   max f32 fast rel: {worst_f32:.4}   max manual rel: {worst_manual:.4}"
    );

    // The manual reference must itself be sound at every dtype (matmul+softmax correct, sc-2714).
    assert!(
        worst_manual < 0.05,
        "manual softmax(q·kᵀ)·v reference diverged ({worst_manual:.4} ≥ 0.05) — the f32 ground \
         truth is unreliable; re-characterize before trusting the fast-SDPA verdict."
    );
    // The f32 NAX SDPA path was always correct and is untouched by the gate.
    assert!(
        worst_f32 < 0.05,
        "f32 fast SDPA diverged ({worst_f32:.4} ≥ 0.05) — unexpected; the NAX f32 attention path \
         regressed."
    );
    // GUARANTOR: 16-bit fast SDPA is now correct. If this fails on a NAX build, the metal kernels were
    // compiled below the macOS 26.2 NAX floor (sc-2772) — check MACOSX_DEPLOYMENT_TARGET in
    // .cargo/config.toml is >= 26.2 (a clean rebuild of pmetal-mlx-sys is needed after a change there).
    assert!(
        worst_16bit < 0.05,
        "NAX 16-bit fast SDPA is GARBAGE ({worst_16bit:.4} ≥ 0.05): the metal kernels were compiled \
         below the macOS 26.2 NAX floor. Verify MACOSX_DEPLOYMENT_TARGET >= 26.2 (sc-2772)."
    );
}
