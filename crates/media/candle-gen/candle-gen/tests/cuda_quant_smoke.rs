//! Self-contained CUDA quantized-matmul regression smoke (sc-7544).
//!
//! candle-kernels compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, the Q4_0/Q8_0/k-quant matmuls)
//! into a **static `libmoe.a`** via cudaforge `build_lib()` → `nvcc -c -gencode=arch=…,code=sm_XX`,
//! i.e. **SASS, no PTX**. Built at the old `CUDA_COMPUTE_CAP=80` packaging baseline the archive holds
//! *only* sm_80 cubin; on a Blackwell sm_120 GPU there is no compatible code and nothing to JIT, so
//! the quant matmul **silently no-ops to zeros/garbage** (cos≈0 vs the CPU reference) while dense
//! (PTX) kernels JIT up fine. The packaging fix is a **multi-arch fatbin** that embeds native sm_120
//! SASS alongside the sm_80 baseline + forward-JIT PTX (see README "Packaging"); with it CUDA matches
//! the CPU reference (cos≈1).
//!
//! This test is the canary so that regression can't return silently. It is **weightless** (no
//! checkpoints) and fast. On a CPU/Metal build it is a graceful no-op — the bug only exists on the
//! CUDA backend, so the check is meaningful only when `default_device()` resolves to CUDA.
//!
//! ## sc-19545 — the gate was scale-blind, and it has never actually run
//!
//! 1. **It gated on cosine, which cannot see this failure class in general.** Cosine is
//!    scale-invariant: a kernel returning `2 * reference`, or the right values under a wrong
//!    dequant scale, scores 1.0 and passes. The gate is now a relative max-abs-diff against the
//!    CPU reference plus an explicit all-zeros check; cosine is still printed, but nothing asserts
//!    on it. (Cosine did catch *exact* zeros — cos 0 — so the sc-7544 regression itself would have
//!    been caught. Everything adjacent to it would not have been.)
//!
//! 2. **It had never executed in CI.** The claim above that it runs "in the local CUDA gate"
//!    points at `scripts/check-cuda.ps1`, which **no workflow invokes** (`grep -rn check-cuda
//!    .github/` is empty). The only automated CUDA lane, `ci.yml`'s `windows-cuda-check`, compiles
//!    with `--no-run` — it builds this binary and throws it away. The lane that would run it,
//!    `ci.yml`'s `windows-cuda`, is `workflow_dispatch`-only and was skipped in all 25 most recent
//!    ci.yml runs. So the canary guarding a *silent* failure mode was itself silent for its whole
//!    life. It is now executed by the `candle-minimax-h3` real-weights CUDA job, which asserts it
//!    reached a CUDA device rather than trusting the exit code.
//!
//! `the_fatbin_covers_this_runners_gpu` adds the structural half: the runner's actual compute
//! capability, read off the device, checked against the `-gencode` ladder parsed out of
//! `vendor/candle-kernels/build.rs`. Note the invariant people expect here is the wrong one —
//! `CUDA_COMPUTE_CAP` is **not** supposed to match the GPU. It names the bottom rung of the arch
//! ladder, and build.rs appends sm_90 / sm_120 / `compute_120` PTX above it. What must hold is that
//! the runner's arch is served by *some* rung; see `scripts/ci/cuda_arch_support.py`, which applies
//! the same predicate statically across all 15 CI sites.

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{Device, Module, Tensor};
use candle_gen::default_device;

/// Deterministic, launch-portable pseudo-random f32 in roughly [-1, 1] (splitmix64-style hash of the
/// index). Avoids a device RNG so the CPU reference and the CUDA result quantize byte-identical data.
fn pseudo_random(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // map the top 24 bits to [-1, 1)
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Cosine similarity of two tensors over all elements (flattened, on the CPU).
fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
}

/// `max|a|` over all elements — the scale the relative error is measured against.
fn max_abs(t: &Tensor) -> f32 {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
}

/// Relative max-absolute-difference, `max|a - b| / max|b|`, with `b` the reference (sc-19545).
///
/// **This, not cosine, is the gate.** Cosine is scale-invariant: a kernel returning exactly
/// `2 * reference`, or the right values under a wrong dequant scale, scores 1.0 and sails through.
/// Norm/cosine/checksum comparisons have been blind to real defects in this family repeatedly, so
/// the assertions below bound the worst single element and cosine survives only as a diagnostic
/// print.
fn relative_max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let scale = max_abs(b);
    assert!(
        scale > 0.0,
        "the CPU reference is itself all-zeros — the fixture is broken, not the CUDA kernel"
    );
    max_abs(&(a - b).expect("elementwise difference")) / scale
}

/// The compute capability of the CUDA device as a packed `major * 10 + minor` (sm_120 → `120`).
///
/// candle keeps `CudaDevice::context` private, so `cuda_stream()` is the only public route to the
/// cudarc context. Mirrors `candle_gen::quant::CublasLt::compute_cap` without paying for that
/// helper's 32 MiB cuBLASLt workspace — this smoke is deliberately allocation-light.
#[cfg(feature = "cuda")]
fn device_compute_cap(device: &Device) -> Option<u32> {
    use candle_gen::candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let Device::Cuda(cuda) = device else {
        return None;
    };
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let major = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    Some(major as u32 * 10 + minor as u32)
}

/// `(native SASS arches, PTX floors)` for `libmoe.a`, read out of the vendored `build.rs` itself.
///
/// The flags that decide coverage live in exactly one file, so this parses that file rather than
/// restating its ladder — a restatement would keep reporting coverage after the real flags were
/// removed, which is precisely the drift this guard exists to catch. Anchored on the `.arg("…")`
/// call and not the bare `-gencode` text, because the same file quotes the ladder in prose.
///
/// Mirrors `scripts/ci/cuda_arch_support.py`, which applies the identical predicate statically in
/// CI; this copy is what runs on the box, where the hardware is finally observable.
fn fatbin_arches(build_rs: &str) -> (Vec<u32>, Vec<u32>) {
    let (mut sass, mut ptx) = (Vec::new(), Vec::new());
    for emitted in build_rs.split(".arg(\"").skip(1) {
        let Some(flag) = emitted.split('"').next() else {
            continue;
        };
        let Some(code) = flag
            .strip_prefix("-gencode=")
            .and_then(|rest| rest.split("code=").nth(1))
        else {
            continue;
        };
        if let Some(cap) = code.strip_prefix("sm_").and_then(|c| c.parse().ok()) {
            sass.push(cap);
        } else if let Some(cap) = code.strip_prefix("compute_").and_then(|c| c.parse().ok()) {
            ptx.push(cap);
        }
    }
    (sass, ptx)
}

/// The GGUF `QMatMul` kernels can actually run on `device_cap`.
///
/// SASS is binary-compatible upward **within one major version only** (an sm_80 cubin serves
/// sm_86/sm_89 but never sm_90); PTX JITs forward across majors. False here is the silent-zeros
/// condition — no compatible cubin and nothing to JIT means the launch does not fail, it just
/// produces nothing.
///
/// Gated with its only caller: without a CUDA device there is no capability to test it against, and
/// an ungated copy would be dead code under `-D warnings` on the CPU/Metal lanes.
#[cfg(feature = "cuda")]
fn quant_path_covers(device_cap: u32, sass: &[u32], ptx: &[u32]) -> bool {
    sass.iter()
        .any(|&c| c / 10 == device_cap / 10 && c % 10 <= device_cap % 10)
        || ptx.iter().any(|&floor| floor <= device_cap)
}

/// The GGUF Q4_0/Q8_0 `QMatMul` on the CUDA device matches the CPU reference, all-finite.
///
/// On the broken sm_80-SASS-only packaging the CUDA result is all-zeros/garbage — this fails loudly.
/// With the multi-arch fatbin (native sm_120 cubin) it passes. The comparison is a relative
/// max-abs-diff, not cosine; see `relative_max_abs_diff` for why that distinction matters.
#[test]
fn cuda_qmatmul_matches_cpu() {
    let device = default_device().expect("default device");
    if !device.is_cuda() {
        eprintln!("SKIP cuda_qmatmul_matches_cpu: default_device()={device:?} is not CUDA");
        return;
    }
    eprintln!("[quant-smoke] device={device:?}");

    // out=N, in=K, rows=M. K is a multiple of 32 (Q4_0/Q8_0 block) and 256 (k-quant QK_K), so the
    // shapes are valid for every GGUF dtype should we extend the sweep later.
    let (n, k, m) = (512usize, 1024usize, 8usize);
    let w_cpu = Tensor::from_vec(pseudo_random(n * k), (n, k), &Device::Cpu).expect("w");
    let x_cpu = Tensor::from_vec(pseudo_random(m * k), (m, k), &Device::Cpu).expect("x");

    // Relative max-abs-diff ceilings, NOT cosine floors (sc-19545). These bound the worst single
    // element against `max|reference|`, so a uniform scale error — invisible to cosine — trips them.
    // The values are deliberately loose: the failure being gated is a kernel that returns ZEROS
    // (relative error 1.0) or garbage, not a fifth-digit accumulation-order difference between the
    // CPU and CUDA reduction trees. Q4_0's 4-bit grid is a wider noise floor than Q8_0's 8-bit one,
    // hence the split. Tighten once a CUDA run has printed the measured values — the observed
    // numbers are logged on every run for exactly that purpose.
    for (dtype, max_rel, label) in [
        (GgmlDType::Q8_0, 0.05f32, "Q8_0"),
        (GgmlDType::Q4_0, 0.25f32, "Q4_0"),
    ] {
        // CPU reference: quantize + matmul entirely on the CPU.
        let mm_cpu = QMatMul::from_qtensor(QTensor::quantize(&w_cpu, dtype).expect("cpu quantize"))
            .expect("cpu qmatmul");
        let y_cpu = mm_cpu.forward(&x_cpu).expect("cpu forward");

        // CUDA: quantize the SAME cpu source straight onto the device, matmul on the device.
        let mm_cuda = QMatMul::from_qtensor(
            QTensor::quantize_onto(&w_cpu, dtype, &device).expect("cuda quantize_onto"),
        )
        .expect("cuda qmatmul");
        let x_cuda = x_cpu.to_device(&device).expect("x->cuda");
        let y_cuda = mm_cuda
            .forward(&x_cuda)
            .expect("cuda forward")
            .to_device(&Device::Cpu)
            .expect("y->cpu");

        let rel = relative_max_abs_diff(&y_cuda, &y_cpu);
        let cuda_scale = max_abs(&y_cuda);
        let cpu_scale = max_abs(&y_cpu);
        let finite = all_finite(&y_cuda);
        // Cosine is printed, never asserted on — see `relative_max_abs_diff`.
        eprintln!(
            "[quant-smoke] {label}: rel_max_abs_diff={rel:.6} max|cuda|={cuda_scale:.6} \
             max|cpu|={cpu_scale:.6} all_finite={finite} (diagnostic cos={:.5})",
            cosine(&y_cpu, &y_cuda)
        );

        assert!(
            finite,
            "{label} CUDA QMatMul produced non-finite values — likely no compatible cubin for this \
             arch (sm_80-SASS-only build on a newer GPU). Rebuild with the multi-arch fatbin."
        );
        // The zeros check, stated separately from the tolerance so the failure message names the
        // actual sc-7544 symptom instead of reading as a precision regression. A thresholded
        // comparison alone would report "0.999 > 0.05" here and bury the diagnosis.
        assert!(
            cuda_scale > 0.0,
            "{label} CUDA QMatMul returned ALL ZEROS (max|cuda|=0, max|cpu|={cpu_scale:.6}). This \
             is the sc-7544 silent-zeros signature: candle-kernels' libmoe.a holds no cubin for \
             this GPU's arch and no PTX to JIT, so the kernel launched and wrote nothing. Check \
             the -gencode ladder in vendor/candle-kernels/build.rs — see README \"Packaging\"."
        );
        assert!(
            rel <= max_rel,
            "{label} CUDA QMatMul does not match the CPU reference: relative max-abs-diff \
             {rel:.6} > {max_rel} (max|cuda|={cuda_scale:.6}, max|cpu|={cpu_scale:.6}). A ratio \
             near 1.0 with a non-zero max|cuda| means garbage rather than zeros — still an arch \
             coverage problem. A small uniform ratio means a dequant SCALE error, which the \
             cosine gate this replaced could not see at all."
        );
    }
}

/// The GPU this runner actually has is covered by the fatbin the build actually emitted (sc-19545).
///
/// The invariant is NOT "`CUDA_COMPUTE_CAP` equals the hardware" — it is not supposed to, it names
/// the ladder's bottom rung so the shipped worker still runs on older customer GPUs. The invariant
/// is that the runner's arch is served by *some* rung. When it is not, nothing fails: quantized
/// matmuls return zeros and the job goes green.
///
/// `cuda_qmatmul_matches_cpu` proves this by value for the two dtypes it exercises. This test
/// answers the same question for the whole arch ladder and, unlike a value check, names the arch
/// and the ladder in its failure message — so a future GPU swap into this pool reports *why* rather
/// than as an unexplained numeric drift. Datacenter Blackwell sm_100 (B100/B200) is the live
/// example: build.rs targets sm_80/sm_90/sm_120 plus `compute_120` PTX, none of which serve major
/// 10, and build.rs says so deliberately.
#[test]
fn the_fatbin_covers_this_runners_gpu() {
    let device = default_device().expect("default device");
    if !device.is_cuda() {
        eprintln!("SKIP the_fatbin_covers_this_runners_gpu: device={device:?} is not CUDA");
        return;
    }

    // `CARGO_MANIFEST_DIR` is `crates/media/candle-gen/candle-gen`; the vendored fork is its
    // sibling. Read at run time rather than `include_str!`ed so that editing build.rs without
    // rebuilding this test cannot leave a stale ladder baked into the binary.
    let build_rs =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../vendor/candle-kernels/build.rs");
    let source = std::fs::read_to_string(&build_rs)
        .unwrap_or_else(|e| panic!("read {}: {e}", build_rs.display()));
    let (sass, ptx) = fatbin_arches(&source);
    // Printed unconditionally, so a `--features cuda` build that lands on a GPU-less host still puts
    // the ladder it was built with in the log. (It also keeps `ptx` read on the non-CUDA build,
    // where the comparison below is compiled out — `-D warnings` is an error on that lane.)
    eprintln!("[quant-smoke] libmoe.a ladder from build.rs: SASS={sass:?} PTX floors={ptx:?}");

    assert!(
        !sass.is_empty(),
        "vendor/candle-kernels/build.rs emits no `code=sm_NN` gencode at all. sc-7544's multi-arch \
         fatbin has been reverted — most likely by a candle pin bump that re-copied build.rs \
         without re-applying the block (see vendor/candle-kernels/VENDORED.md)."
    );

    #[cfg(feature = "cuda")]
    {
        let cap = device_compute_cap(&device).expect("compute capability of the CUDA device");
        // The declared cap is a build-time input to the SAME job that runs this test, so the job
        // env is the honest source. Absent (a local `cargo test` without the recipe), report only
        // the rungs build.rs adds explicitly rather than inventing a baseline.
        let declared: Option<u32> = std::env::var("CUDA_COMPUTE_CAP")
            .ok()
            .and_then(|v| v.trim().parse().ok());
        let mut effective = sass.clone();
        if let Some(cap) = declared {
            effective.push(cap);
        }
        effective.sort_unstable();
        eprintln!(
            "[quant-smoke] device sm_{cap}; CUDA_COMPUTE_CAP={declared:?}; libmoe.a SASS={effective:?} PTX floors={ptx:?}"
        );
        assert!(
            quant_path_covers(cap, &effective, &ptx),
            "this runner's GPU is sm_{cap}, which NO rung of the quantized-kernel fatbin serves: \
             SASS {effective:?}, PTX floors {ptx:?}. SASS is compatible only upward within one \
             major version and PTX only JITs to archs at or above its floor, so every Q4/Q8 \
             QMatMul on this box returns ZEROS while every job stays green. Add \
             `-gencode=arch=compute_{cap},code=sm_{cap}` to vendor/candle-kernels/build.rs. Do NOT \
             instead raise CUDA_COMPUTE_CAP to {cap}: that deletes the ladder's bottom rung and \
             breaks every older GPU (see scripts/ci/cuda_arch_support.py)."
        );
    }
}
