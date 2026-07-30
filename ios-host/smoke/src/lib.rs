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
use std::path::Path;
use std::time::Instant;

use mlx_rs::ops::{matmul, ones, softmax_axis};
use mlx_rs::{Array, Dtype};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};
use runtime_ios::core_llm::{LoadSpec, Message, Sampling, TextLlmRequest};

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

/// Peak resident memory, in MiB — the number that decides whether a model fits the per-app cap.
///
/// `ru_maxrss` is bytes on Darwin (pages on Linux), and it is a high-water mark, so it captures
/// the transient load spike rather than only the steady state.
fn peak_rss_mib() -> f64 {
    // SAFETY: getrusage writes into a fully-initialized struct we own.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0.0;
        }
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
}

/// End-to-end generation through the `runtime-ios` bundle, from a snapshot in the app container.
///
/// This is the check that matters for the product: not "does MLX dispatch a kernel" but "does the
/// composed runtime load a real model on this device and emit correct tokens". It runs through
/// the bundle's registry and the `TextLlm` contract, so provider selection and the capability
/// descriptor are exercised too, not just the engine.
///
/// Skips (rather than fails) when no snapshot is present, so the kernel checks above stay useful
/// on a device that has not been provisioned yet.
fn check_generation(model_dir: Option<&Path>) -> Check {
    const NAME: &str = "runtime-ios generation";
    let Some(dir) = model_dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no snapshot in Documents/ (see docs/ios-epics.md S3.4)".to_string(),
        };
    };

    let before = peak_rss_mib();
    let started = Instant::now();

    let run = || -> Result<(String, String, u32, f64, f64), String> {
        let llm = runtime_ios::llm::load_for_model(&LoadSpec::dense(
            dir.to_string_lossy().to_string(),
        ))
        .map_err(|e| format!("load failed: {e}"))?;
        let descriptor = llm.descriptor();
        let id = descriptor.id.clone();
        let tools = descriptor.capabilities.supports_tools;
        let loaded_at = started.elapsed().as_secs_f64();
        let after_load_rss = peak_rss_mib();

        // Greedy + fixed seed: the answer is then a property of the weights, not of sampling luck,
        // so a wrong result means the kernels are wrong rather than the dice.
        let request = TextLlmRequest {
            messages: vec![Message::user("What is the capital of France? Answer in one word.")],
            sampling: Sampling::greedy(),
            max_new_tokens: 12,
            seed: Some(0),
            ..Default::default()
        };
        let out = llm
            .complete(&request)
            .map_err(|e| format!("generate failed: {e}"))?;
        let answer_secs = started.elapsed().as_secs_f64() - loaded_at;

        // A second, longer request for the throughput number. The correctness prompt above stops
        // at EOS after one word, so its tok/s is dominated by first-token latency (weights fault
        // in lazily, so the real load cost lands on the first forward pass, not on `load`) and
        // says nothing about steady-state decode.
        let bench = TextLlmRequest {
            messages: vec![Message::user("Count from one to twenty in words.")],
            sampling: Sampling::greedy(),
            max_new_tokens: 64,
            seed: Some(0),
            ..Default::default()
        };
        let bench_started = Instant::now();
        let bench_out = llm
            .complete(&bench)
            .map_err(|e| format!("throughput run failed: {e}"))?;
        let bench_secs = bench_started.elapsed().as_secs_f64();
        let bench_tokens = bench_out.usage.generated_tokens;

        let tokens = out.usage.generated_tokens;
        let detail = format!(
            "id={id} tools={tools} | load {loaded_at:.1}s, first answer {tokens} tok in \
             {answer_secs:.1}s | steady {bench_tokens} tok in {bench_secs:.1}s ({:.1} tok/s) | \
             RSS after load {after_load_rss:.0} MiB, peak {:.0} MiB (+{:.0} from {before:.0}) | {:?}",
            bench_tokens as f64 / bench_secs.max(1e-6),
            peak_rss_mib(),
            peak_rss_mib() - before,
            out.text.trim(),
        );
        Ok((detail, out.text.to_lowercase(), tokens, 0.0, 0.0))
    };

    match run() {
        // "Paris" is checkable without a reference implementation, which is the point: a metallib
        // that loads and computes garbage would produce tokens but not this token.
        Ok((detail, lowered, tokens, _, _)) if tokens > 0 && lowered.contains("paris") => Check {
            name: NAME,
            passed: true,
            detail,
        },
        Ok((detail, _, _, _, _)) => Check {
            name: NAME,
            passed: false,
            detail: format!("expected an answer containing \"Paris\" -- {detail}"),
        },
        Err(e) => Check {
            name: NAME,
            passed: false,
            detail: e,
        },
    }
}

/// Sustained decode: the memory and throughput question E4 actually owns.
///
/// The short generation above says nothing about this. A single 64-token run has a KV cache of
/// negligible size, so it cannot show whether memory grows without bound or whether throughput
/// degrades as the context fills — and on a phone with a hard per-app cap, an OOM kill (jetsam)
/// looks like the app "just closing", with no crash log tying it to inference.
///
/// So: several back-to-back generations, sampling peak RSS and per-segment throughput as they go.
/// What we want to see is RSS holding flat across repeated work and steady tok/s rather than a
/// decay curve as the device warms.
///
/// **Scope, precisely.** `TextLlm::generate` allocates a fresh KV cache per call
/// (`decode/stream.rs`), so this measures *repeated independent generations* — it proves the
/// runtime does not leak across calls and does not throttle over ~30 s of continuous GPU work. It
/// does **not** measure a single long context: KV growth within one generation needs a
/// prefix-cached or multi-turn path, and is a separate measurement.
///
/// Reported, not asserted. A threshold invented from one device on one thermal state would be
/// noise; these numbers are the baseline a threshold can later be set from (S4.6).
fn check_sustained_decode(model_dir: Option<&Path>) -> Check {
    const NAME: &str = "sustained decode (memory + throughput)";
    const TOTAL_TOKENS: u32 = 512;
    const SEGMENTS: usize = 4;

    let Some(dir) = model_dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no snapshot in Documents/".to_string(),
        };
    };

    let run = || -> Result<String, String> {
        let llm = runtime_ios::llm::load_for_model(&LoadSpec::dense(
            dir.to_string_lossy().to_string(),
        ))
        .map_err(|e| format!("load failed: {e}"))?;

        let baseline_rss = peak_rss_mib();
        let mut segments = Vec::with_capacity(SEGMENTS);
        let per_segment = TOTAL_TOKENS / SEGMENTS as u32;

        for i in 0..SEGMENTS {
            let request = TextLlmRequest {
                messages: vec![Message::user(
                    "Write a detailed account of a long sea voyage, with many specific incidents.",
                )],
                sampling: Sampling::greedy(),
                max_new_tokens: per_segment,
                // Vary the seed so segments are not identical work; greedy still keeps each one
                // deterministic and comparable run to run.
                seed: Some(i as u64),
                ..Default::default()
            };
            let started = Instant::now();
            let out = llm
                .complete(&request)
                .map_err(|e| format!("segment {i} failed: {e}"))?;
            let secs = started.elapsed().as_secs_f64();
            let tokens = out.usage.generated_tokens;
            segments.push((
                tokens,
                tokens as f64 / secs.max(1e-6),
                peak_rss_mib(),
                out.usage.prompt_tokens,
            ));
        }

        let first = segments.first().expect("SEGMENTS > 0");
        let last = segments.last().expect("SEGMENTS > 0");
        // Throughput retention across the run. Well under 1.0 means thermal throttling; this is
        // the headline number for E4/S4.4. Note the FIRST segment is usually the slowest (weights
        // fault in lazily on the first forward pass), so retention above 100% is expected and is
        // not evidence of speed-up.
        let retention = last.1 / first.1.max(1e-6);
        let rss_growth = last.2 - first.2;
        let total: u32 = segments.iter().map(|s| s.0).sum();

        let per_seg = segments
            .iter()
            .map(|(t, tps, rss, _)| format!("{t}tok@{tps:.1}t/s/{rss:.0}MiB"))
            .collect::<Vec<_>>()
            .join(" ");

        Ok(format!(
            "{total} tok over {SEGMENTS} segments | {per_seg} | retention {:.0}% \
             (first {:.1} -> last {:.1} tok/s) | RSS {:.0} -> {:.0} MiB (growth {rss_growth:.0}, \
             baseline {baseline_rss:.0})",
            retention * 100.0,
            first.1,
            last.1,
            first.2,
            last.2,
        ))
    };

    match run() {
        Ok(detail) => Check {
            name: NAME,
            passed: true,
            detail,
        },
        Err(e) => Check {
            name: NAME,
            passed: false,
            detail: e,
        },
    }
}

/// The full backend-neutral conformance suite, on device (S3.5).
///
/// This is the check that makes "conformant on iOS" mean the same thing it means on macOS: the
/// identical `core_llm_testkit::textllm_conformance` the other lanes run, over all eight always-on
/// checks — descriptor, validate, streaming, cancellation, mid-stream cancel, seed determinism,
/// multimodal, video, thinking, tools.
///
/// The suite signals failure by panicking with an aggregated message, which is fine in a test
/// harness but not across an FFI boundary, so it runs inside `catch_unwind` and the panic payload
/// becomes the report detail. `AssertUnwindSafe` is sound here: on the failure path the provider
/// is dropped and nothing observes it afterwards.
fn check_conformance(model_dir: Option<&Path>) -> Check {
    const NAME: &str = "core-llm conformance suite";
    let Some(dir) = model_dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no snapshot in Documents/".to_string(),
        };
    };

    let dir = dir.to_string_lossy().to_string();
    let started = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        textllm_conformance(
            &|| {
                runtime_ios::llm::load_for_model(&LoadSpec::dense(dir.clone()))
                    .expect("load provider for conformance")
            },
            &TextLlmProfile::cheap(),
        );
    }));

    match result {
        Ok(()) => Check {
            name: NAME,
            passed: true,
            detail: format!(
                "all always-on checks passed in {:.1}s",
                started.elapsed().as_secs_f64()
            ),
        },
        Err(payload) => {
            // The suite aggregates every failure into one panic message, so this is the whole
            // diagnostic rather than just the first failing check.
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "conformance panicked with a non-string payload".to_string());
            Check {
                name: NAME,
                passed: false,
                // Collapse to one line: the report is read from a device console and a file.
                detail: message.replace('\n', " | "),
            }
        }
    }
}

/// Locate a prepared snapshot inside the app's Documents container.
///
/// A snapshot is any directory holding `config.json`. Documents itself is checked first, because
/// `devicectl device copy to --source <dir> --destination Documents/` **flattens** the directory —
/// the files land in `Documents/` rather than in `Documents/<dir>/`. Both layouts are accepted so
/// a snapshot pushed either way (or side-loaded through the Files app) is found.
///
/// This is how weights reach the device: the workspace never fetches, so a caller provisions every
/// path (`WeightsSource::Dir`).
fn find_snapshot() -> Option<std::path::PathBuf> {
    let docs = dirs_documents()?;
    if docs.join("config.json").is_file() {
        return Some(docs);
    }
    std::fs::read_dir(&docs)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("config.json").is_file())
}

/// The app's Documents directory. `NSHomeDirectory` is the container root inside the sandbox; on
/// the host (where the same checks run under `cargo test`) `HOME` serves the same role.
fn dirs_documents() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let docs = std::path::PathBuf::from(home).join("Documents");
    docs.is_dir().then_some(docs)
}

/// Run every check and render a human- and test-readable report.
///
/// The first line is `SMOKE: PASS` or `SMOKE: FAIL` so an XCTest can assert on a prefix without
/// parsing the body.
pub fn run_report() -> String {
    let snapshot = find_snapshot();
    let checks = vec![
        check_metallib_resolves(),
        check_gemm(Dtype::Float32, "f32 GEMM (steel)"),
        check_gemm(Dtype::Bfloat16, "bf16 GEMM (steel)"),
        check_softmax(),
        check_generation(snapshot.as_deref()),
        check_conformance(snapshot.as_deref()),
        check_sustained_decode(snapshot.as_deref()),
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
