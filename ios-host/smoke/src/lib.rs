//! On-device MLX smoke test.
//!
//! Answers the two questions no build can settle, which the iOS epic tracks as S3.3 and R11:
//!
//! 1. **Does the bundled metallib resolve inside the app sandbox?** `~/.cache/pmetal/lib` is
//!    unreadable there and the compiled-in `METAL_PATH` points into the cargo target directory,
//!    so resolution must fall through to `load_colocated_library` finding `mlx.metallib` next to
//!    the executable. If it does not, the first Metal op throws.
//! 2. **Are the cross-compiled kernels numerically correct?** Precedent for "compiles, runs,
//!    returns garbage" is sc-2772, where NAX kernels built at the wrong deployment target
//!    miscompiled silently. A metallib that loads proves nothing about what it computes.
//!
//! Each check runs a real GPU kernel and verifies the result against a value known independently
//! of MLX. The FFI entry point returns a report string so the app can display it and an XCTest
//! can assert on it.

use std::ffi::{c_char, CString};

use mlx_rs::ops::{matmul, ones, softmax_axis};
use mlx_rs::{Array, Dtype};

/// One check's outcome. `detail` carries the observed value, so a failure report is diagnosable
/// without a debugger attached.
struct Check {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Sum of `x` over all axes, read back to the host.
///
/// `try_item` evaluates, so a kernel fault surfaces here as an `Err` rather than a panic — which
/// matters because this runs behind an FFI boundary.
fn sum_to_host(x: &Array) -> Result<f32, String> {
    x.sum(None)
        .map_err(|e| e.to_string())?
        .try_item::<f32>()
        .map_err(|e| e.to_string())
}

/// The metallib loaded at all: the simplest possible GPU op.
///
/// This is the S3.3 check. If the bundled metallib is missing or unreadable in the sandbox, MLX
/// throws while loading the default library and this returns the error rather than a wrong value.
fn check_metallib_resolves() -> Check {
    const NAME: &str = "metallib resolves + elementwise kernel";
    let run = || -> Result<f32, String> {
        let a = ones::<f32>(&[4, 4]).map_err(|e| e.to_string())?;
        sum_to_host(&a)
    };
    // A 4x4 of ones sums to 16 by construction, not by MLX's say-so.
    match run() {
        Ok(v) if approx(v, 16.0, 1e-5) => Check {
            name: NAME,
            passed: true,
            detail: format!("sum(ones[4,4]) = {v}"),
        },
        Ok(v) => Check {
            name: NAME,
            passed: false,
            detail: format!("expected 16.0, got {v}"),
        },
        Err(e) => Check {
            name: NAME,
            passed: false,
            detail: format!("kernel dispatch failed: {e}"),
        },
    }
}

/// GEMM correctness in f32, then bf16 — the steel-GEMM path.
///
/// A 64x64 matmul of ones has every output element equal to 64, so the sum is 64^3 = 262144.
/// Large enough to exercise a real tiled kernel rather than a trivial one, and the expected value
/// is arithmetic rather than an MLX reference.
fn check_gemm(dtype: Dtype, label: &'static str) -> Check {
    const N: i32 = 64;
    let expected = (N as f32).powi(3);

    let build = || -> Result<f32, String> {
        let a = ones::<f32>(&[N, N])
            .map_err(|e| e.to_string())?
            .as_dtype(dtype)
            .map_err(|e| e.to_string())?;
        let b = ones::<f32>(&[N, N])
            .map_err(|e| e.to_string())?
            .as_dtype(dtype)
            .map_err(|e| e.to_string())?;
        let c = matmul(&a, &b)
            .map_err(|e| e.to_string())?
            .as_dtype(Dtype::Float32)
            .map_err(|e| e.to_string())?;
        sum_to_host(&c)
    };

    match build() {
        // bf16 has ~8 bits of mantissa; 262144 is exactly representable, but accumulate slack.
        Ok(v) if approx(v, expected, expected * 1e-3) => Check {
            name: label,
            passed: true,
            detail: format!("sum({N}x{N} matmul) = {v} (expected {expected})"),
        },
        Ok(v) => Check {
            name: label,
            passed: false,
            detail: format!("expected {expected}, got {v} -- kernels miscompiled (cf. sc-2772)"),
        },
        Err(e) => Check {
            name: label,
            passed: false,
            detail: e,
        },
    }
}

/// Softmax: a reduction kernel with a known analytic answer.
///
/// Over a uniform row of width 8 every probability is 1/8, so each row sums to 1 and the whole
/// [4, 8] array sums to 4. Catches a reduction that dispatches but reduces along the wrong axis.
fn check_softmax() -> Check {
    let run = || -> Result<f32, String> {
        let x = ones::<f32>(&[4, 8]).map_err(|e| e.to_string())?;
        let p = softmax_axis(&x, -1, None).map_err(|e| e.to_string())?;
        sum_to_host(&p)
    };

    match run() {
        Ok(v) if approx(v, 4.0, 1e-4) => Check {
            name: "softmax reduction kernel",
            passed: true,
            detail: format!("sum(softmax(ones[4,8])) = {v}"),
        },
        Ok(v) => Check {
            name: "softmax reduction kernel",
            passed: false,
            detail: format!("expected 4.0, got {v}"),
        },
        Err(e) => Check {
            name: "softmax reduction kernel",
            passed: false,
            detail: e,
        },
    }
}

/// Run every check and render a human- and test-readable report.
///
/// The first line is `SMOKE: PASS` or `SMOKE: FAIL` so an XCTest can assert on a prefix without
/// parsing the body.
pub fn run_report() -> String {
    let checks = vec![
        check_metallib_resolves(),
        check_gemm(Dtype::Float32, "f32 GEMM (steel)"),
        check_gemm(Dtype::Bfloat16, "bf16 GEMM (steel)"),
        check_softmax(),
    ];

    let failed = checks.iter().filter(|c| !c.passed).count();
    let mut out = String::new();
    out.push_str(if failed == 0 {
        "SMOKE: PASS\n"
    } else {
        "SMOKE: FAIL\n"
    });
    for c in &checks {
        out.push_str(&format!(
            "  [{}] {} -- {}\n",
            if c.passed { "ok" } else { "XX" },
            c.name,
            c.detail
        ));
    }
    out
}

/// C entry point. Returns an owned UTF-8 string; free it with [`ios_smoke_free`].
///
/// # Safety
/// The returned pointer must be passed to [`ios_smoke_free`] exactly once and not used after.
#[no_mangle]
pub extern "C" fn ios_smoke_run() -> *mut c_char {
    // A panic across an FFI boundary is UB, and any check can panic in principle. Catch it and
    // report it as a failure instead.
    let report = std::panic::catch_unwind(run_report)
        .unwrap_or_else(|_| "SMOKE: FAIL\n  [XX] panic inside the smoke test\n".to_string());
    CString::new(report)
        .unwrap_or_else(|_| CString::new("SMOKE: FAIL\n  [XX] report contained a NUL\n").unwrap())
        .into_raw()
}

/// Free a string returned by [`ios_smoke_run`].
///
/// # Safety
/// `ptr` must have come from [`ios_smoke_run`] and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn ios_smoke_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

#[cfg(test)]
mod tests {
    /// The same checks must pass on the host, so a failure on device is attributable to the iOS
    /// build rather than to a bug in the checks themselves.
    #[test]
    fn smoke_passes_on_the_host() {
        let report = super::run_report();
        assert!(report.starts_with("SMOKE: PASS"), "{report}");
    }
}
