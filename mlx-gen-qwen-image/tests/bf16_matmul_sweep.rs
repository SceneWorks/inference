//! sc-2517: characterization + tripwire for the NAX 16-bit DENSE Metal GEMM bug.
//!
//! On the pinned pmetal MLX builds (BOTH 0.30.6 and 0.31.1, on macOS-26 / Metal-400 / NAX) a
//! `matmul(16bit, 16bit)` — both operands bf16 *or* both f16 — returns GARBAGE (mean-relative
//! error vs an f32 reference ~0.3–2.3: right scale, uncorrelated, no NaN) for `M >= 2` with
//! `K <= 512`, plus very large `M` (=1024) at any `K`. `M = 1` (the gemv path), f32, and
//! `quantized_matmul` (fp32 accumulation, mlx#963) are all correct. Root cause: the NAX
//! `steel_gemm_fused_nax_*` matrix-unit kernels (lmstudio#1356, mlx#3196/#3337). The bump to
//! MLX 0.31.1 (sc-2517) did NOT fix it — confirmed identical to 0.30.6 by this very sweep.
//!
//! Why it's harmless today: the shipped models keep activations in f32 (MLX promotes
//! `matmul(f32, bf16-weight)` to f32), so the 16-bit×16-bit dense GEMM is never executed; the
//! `adapters.rs` bf16→f32 upcast keeps the quant path off it too. See memory
//! `pmetal-mlx-bf16-matmul-bug` / `mlx-031-bump-plan`.
//!
//! This test is a **tripwire**: it asserts (a) the safe cells are correct, and (b) the bug is
//! STILL present. If a future MLX bump actually FIXES the NAX GEMM, assertion (b) flips and this
//! test FAILS — which is *good news*: the `adapters.rs` upcast and the f32-activation invariant
//! can then be retired. Needs no weights, only MLX.

use mlx_rs::{ops::matmul, random, Array, Dtype};

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

// `#[ignore]`d: this characterizes a bug in the **NAX** matrix-unit kernels, which only exist on the
// macOS-26 / Metal-400 build (`MACOSX_DEPLOYMENT_TARGET=26.0`). CI builds non-NAX (Metal 320,
// `-DMLX_METAL_NO_NAX`), where the 16-bit GEMM uses correct fallback kernels and assertion (b) below
// would (rightly) fail. Run on the NAX machine: `cargo test -p mlx-gen-qwen-image --release \
// --test bf16_matmul_sweep -- --ignored --nocapture`.
#[test]
#[ignore = "needs the NAX (macOS-26/Metal-400) build; CI is non-NAX so the bug does not reproduce"]
fn nax_16bit_gemm_bug_characterization() {
    // Distinct keys for the two operands so no (M,K)==(K,N) cell degenerates to A == B.
    let ka = random::key(0).unwrap();
    let kb = random::key(1).unwrap();
    let n = 1024i32; // out_features; the bug is independent of N.
    let ms = [1, 2, 4, 16, 256, 1024];
    let ks = [64, 128, 256, 512, 1024, 3072];

    // Garbage zone = M>=2 AND (K<=512 OR M==1024). Track the worst cell in each region.
    let mut worst_garbage = f64::INFINITY; // min over garbage cells — must stay HIGH (bug present)
    let mut worst_safe = 0.0f64; // max over safe cells — must stay LOW (kernel correct)
    println!("  *=garbage-zone (M>=2 & (K<=512 or M==1024))   N={n}");
    println!("    M     K      bf16      f16");
    for &m in &ms {
        for &k in &ks {
            let a = random::normal::<f32>(&[m, k], None, None, Some(&ka)).unwrap();
            let b = random::normal::<f32>(&[k, n], None, None, Some(&kb)).unwrap();
            let reff = matmul(&a, &b).unwrap();
            let bf16 = matmul(
                a.as_dtype(Dtype::Bfloat16).unwrap(),
                b.as_dtype(Dtype::Bfloat16).unwrap(),
            )
            .unwrap();
            let f16 = matmul(
                a.as_dtype(Dtype::Float16).unwrap(),
                b.as_dtype(Dtype::Float16).unwrap(),
            )
            .unwrap();
            let r_bf16 = rel(&bf16, &reff);
            let r_f16 = rel(&f16, &reff);
            let garbage_zone = m >= 2 && (k <= 512 || m == 1024);
            let mark = if garbage_zone { '*' } else { ' ' };
            println!("{mark} {m:5} {k:5}  {r_bf16:8.4}  {r_f16:8.4}");
            // bf16 and f16 fail together; take the lower of the two as the conservative signal.
            let cell = r_bf16.min(r_f16);
            if garbage_zone {
                worst_garbage = worst_garbage.min(cell);
            } else {
                worst_safe = worst_safe.max(r_bf16.max(r_f16));
            }
        }
    }
    println!("min garbage-zone rel: {worst_garbage:.4}   max safe-zone rel: {worst_safe:.4}");

    // (a) The kernel is correct where the shipped models actually use it (M=1, large-K).
    assert!(
        worst_safe < 0.05,
        "a 16-bit GEMM cell OUTSIDE the known garbage zone diverged ({worst_safe:.4} ≥ 0.05): \
         the NAX bug footprint changed — re-characterize before trusting any 16-bit GEMM."
    );
    // (b) TRIPWIRE: the garbage zone is still garbage. If this fails, the underlying NAX
    // 16-bit GEMM has been FIXED by an MLX bump — retire the `adapters.rs` bf16→f32 upcast and
    // the f32-activation invariant, then delete this assertion.
    assert!(
        worst_garbage > 0.2,
        "GOOD NEWS: the NAX 16-bit dense GEMM appears FIXED (worst garbage-zone rel {worst_garbage:.4} ≤ 0.2). \
         Retire the adapters.rs bf16→f32 upcast + f32-activation invariant and update this tripwire."
    );
}
