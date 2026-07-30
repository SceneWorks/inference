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

use core_llm_testkit::{textllm_conformance, TextLlmProfile};
use mlx_rs::ops::{matmul, ones, softmax_axis};
use mlx_rs::{Array, Dtype};
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

/// The per-app memory the OS will still let this process allocate, in MiB — or `None` off iOS.
///
/// Every memory conclusion on this branch has been measured against an *assumed* cap: "~6 GB on a
/// 12 GB device, ~4 GB on 8 GB". Those are folklore. `os_proc_available_memory` is the number iOS
/// actually enforces, read from the process it applies to, and it is the only way to know whether a
/// measured peak has headroom or is one allocation from a jetsam kill.
///
/// It reports memory *remaining*, not the total limit, so it must be sampled at a known point —
/// here, at the start of the report, before anything large is resident. Sampling it again after a
/// large allocation gives the drop, which is a cross-check on MLX's own accounting.
#[cfg(target_os = "ios")]
fn available_memory_mib() -> Option<f64> {
    extern "C" {
        fn os_proc_available_memory() -> libc::size_t;
    }
    // SAFETY: no arguments, no pointers; returns 0 when unavailable (e.g. an unsupported platform).
    let bytes = unsafe { os_proc_available_memory() };
    (bytes > 0).then(|| bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(not(target_os = "ios"))]
fn available_memory_mib() -> Option<f64> {
    // macOS has no per-app cap, so there is no honest number to report and the host run says so
    // rather than inventing one.
    None
}

/// Report the OS-enforced per-app memory limit, so every other number has a denominator.
fn check_memory_headroom() -> Check {
    const NAME: &str = "per-app memory limit (OS-reported)";
    match available_memory_mib() {
        Some(mib) => Check {
            name: NAME,
            passed: true,
            detail: format!(
                "os_proc_available_memory = {mib:.0} MiB available at start of run \
                 ({:.2} GiB) -- this is the real ceiling every peak below is measured against, \
                 not an assumed one",
                mib / 1024.0
            ),
        },
        None => Check {
            name: NAME,
            passed: true,
            detail: "not applicable on this platform (no per-app cap)".to_string(),
        },
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
        let llm =
            runtime_ios::llm::load_for_model(&LoadSpec::dense(dir.to_string_lossy().to_string()))
                .map_err(|e| format!("load failed: {e}"))?;
        let descriptor = llm.descriptor();
        let id = descriptor.id.clone();
        let tools = descriptor.capabilities.supports_tools;
        let loaded_at = started.elapsed().as_secs_f64();
        let after_load_rss = peak_rss_mib();

        // Greedy + fixed seed: the answer is then a property of the weights, not of sampling luck,
        // so a wrong result means the kernels are wrong rather than the dice.
        let request = TextLlmRequest {
            messages: vec![Message::user(
                "What is the capital of France? Answer in one word.",
            )],
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
        let llm =
            runtime_ios::llm::load_for_model(&LoadSpec::dense(dir.to_string_lossy().to_string()))
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

/// A sustained soak, long enough for thermal behaviour to show (S4.4).
///
/// The 512-token sustained check runs ~30 s — enough to prove throughput does not collapse
/// immediately, not enough for a phone to actually heat up. Thermal throttling on passively-cooled
/// hardware takes minutes, so a short run measures the best case and calls it the steady state.
///
/// This runs for a configurable wall-clock duration (default 5 min, the figure E4/S4.4 names) and
/// reports throughput per minute plus the OS thermal state. Off by default — it is the slowest
/// check by an order of magnitude — and enabled with `IOS_SMOKE_SOAK_SECS`.
///
/// Energy is captured separately by `xctrace` on the host while this runs; `scripts/ios/soak.sh`
/// drives both together.
fn check_thermal_soak(model_dir: Option<&Path>) -> Check {
    const NAME: &str = "thermal soak";

    let Some(secs) = std::env::var("IOS_SMOKE_SOAK_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- set IOS_SMOKE_SOAK_SECS=300 to run".to_string(),
        };
    };
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

        let started = Instant::now();
        let mut buckets: Vec<(f64, u32, f64)> = Vec::new();
        let mut bucket_started = Instant::now();
        let mut bucket_tokens = 0u32;
        let mut total_tokens = 0u32;
        let mut round = 0u64;

        while started.elapsed().as_secs() < secs {
            let request = TextLlmRequest {
                messages: vec![Message::user(
                    "Write a detailed account of a long sea voyage, with many specific incidents.",
                )],
                sampling: Sampling::greedy(),
                max_new_tokens: 128,
                seed: Some(round),
                ..Default::default()
            };
            let out = llm
                .complete(&request)
                .map_err(|e| format!("soak round {round} failed: {e}"))?;
            bucket_tokens += out.usage.generated_tokens;
            total_tokens += out.usage.generated_tokens;
            round += 1;

            // Bucket by minute so a decay curve is visible rather than averaged away.
            if bucket_started.elapsed().as_secs() >= 60 || started.elapsed().as_secs() >= secs {
                let elapsed = bucket_started.elapsed().as_secs_f64();
                buckets.push((
                    started.elapsed().as_secs_f64(),
                    bucket_tokens,
                    bucket_tokens as f64 / elapsed.max(1e-6),
                ));
                bucket_started = Instant::now();
                bucket_tokens = 0;
            }
        }

        let per_bucket = buckets
            .iter()
            .map(|(at, tok, tps)| format!("{:.0}s:{tok}tok@{tps:.1}t/s", at))
            .collect::<Vec<_>>()
            .join(" ");
        let first = buckets.first().map(|b| b.2).unwrap_or(0.0);
        let last = buckets.last().map(|b| b.2).unwrap_or(0.0);
        let retention = if first > 0.0 { last / first } else { 0.0 };

        Ok(format!(
            "{}s soak, {total_tokens} tok | {per_bucket} | retention {:.0}% \
             (first {first:.1} -> last {last:.1} tok/s) | peak RSS {:.0} MiB",
            started.elapsed().as_secs(),
            retention * 100.0,
            peak_rss_mib(),
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

/// The staged load/unload seam: does dropping a provider actually give the memory back? (S4.5)
///
/// A 17 Pro Max does not need this — a 2.6 GiB model and the image stack are co-resident under
/// its ~6 GB cap. An 8 GB device (~4 GB cap) cannot hold both, so it must unload one to load the
/// other. The seam is built and measured **now, while it is not needed**, because retrofitting it
/// into a pipeline that assumed co-residency is the expensive version
/// (`docs/architecture/ios-project-spec.md` §0.1).
///
/// The question is not "does `drop` compile" but "does the memory come back". Measured on device:
/// **`drop` alone returns everything** — active memory goes 2693 MiB → 0 *before* `clear_cache` is
/// called, so MLX's buffer cache is not holding the weights after the provider dies. The
/// `clear_cache` call is kept as belt-and-braces (free when the cache is already empty), but it is
/// not what does the work.
///
/// This reads MLX's own accounting (`get_active_memory`), not RSS: `ru_maxrss` is a high-water
/// mark that by definition never falls, so it cannot observe a release at all — measuring with it
/// would have reported "nothing freed" and been entirely wrong.
fn check_unload_seam(model_dir: Option<&Path>) -> Check {
    const NAME: &str = "staged load/unload seam";
    let Some(dir) = model_dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no snapshot in Documents/".to_string(),
        };
    };

    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let spec = LoadSpec::dense(dir.to_string_lossy().to_string());

    // Baseline with nothing loaded, so the delta is attributable to the model.
    mlx_rs::memory::clear_cache();
    let idle = mlx_rs::memory::get_active_memory();

    let loaded = {
        let llm = match runtime_ios::llm::load_for_model(&spec) {
            Ok(llm) => llm,
            Err(e) => {
                return Check {
                    name: NAME,
                    passed: false,
                    detail: format!("load failed: {e}"),
                }
            }
        };
        // Weights fault in lazily, so a load alone touches little. Generate to force them
        // resident — otherwise "unload" would be reclaiming memory never actually used.
        let request = TextLlmRequest {
            messages: vec![Message::user("Hello")],
            sampling: Sampling::greedy(),
            max_new_tokens: 4,
            seed: Some(0),
            ..Default::default()
        };
        if let Err(e) = llm.complete(&request) {
            return Check {
                name: NAME,
                passed: false,
                detail: format!("generate failed: {e}"),
            };
        }
        // Sampled BEFORE the scope ends, so this is allocation with the provider still alive —
        // which is the whole point of the comparison below.
        mlx_rs::memory::get_active_memory()
        // `llm` drops here.
    };

    // Measured at ~0: `drop` returns the weights on its own, so the clear below is a guard
    // rather than the mechanism.
    let after_drop = mlx_rs::memory::get_active_memory();
    mlx_rs::memory::clear_cache();
    let after_clear = mlx_rs::memory::get_active_memory();

    let held = loaded.saturating_sub(idle);
    let reclaimed = loaded.saturating_sub(after_clear);
    let fraction = if held > 0 {
        reclaimed as f64 / held as f64
    } else {
        0.0
    };

    let detail = format!(
        "idle {:.0} -> loaded {:.0} -> dropped {:.0} -> cleared {:.0} MiB | reclaimed {:.0} MiB \
         ({:.0}% of {:.0} MiB held)",
        mib(idle),
        mib(loaded),
        mib(after_drop),
        mib(after_clear),
        mib(reclaimed),
        fraction * 100.0,
        mib(held),
    );

    // 90%: the seam has to actually work for a smaller device to be viable. Anything less means
    // a second model could not be loaded after unloading the first, which is the whole point.
    Check {
        name: NAME,
        passed: fraction >= 0.9,
        detail: if fraction >= 0.9 {
            detail
        } else {
            format!(
                "only {:.0}% reclaimed -- unload does not free the model | {detail}",
                fraction * 100.0
            )
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
            || {
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

/// The SANA snapshot, if one was pushed. A diffusers multi-component tree, so it is identified by
/// its component directories rather than by a root `config.json` — which it does not have, and which
/// is also what keeps it from being mistaken for the LLM snapshot by [`find_snapshot`] above.
#[cfg(feature = "media")]
fn find_media_snapshot() -> Option<std::path::PathBuf> {
    let docs = dirs_documents()?;
    // The component tree ALONE is not enough to identify SANA any more. Z-Image's q4 tier has the
    // identical diffusers shape (transformer/ vae/ text_encoder/), so once both are pushed this
    // predicate matches both and `find` takes whichever readdir yields first — which handed SANA
    // z-image's weights and failed with "Path must point to a local file", a load error that reads
    // nothing like a snapshot mix-up.
    //
    // Both finders must be specific, not just one. The z-image finder was written to require
    // "zimage" in the name precisely so it could not claim SANA's directory; leaving this one
    // unqualified defended a single direction of a symmetric problem.
    let is_sana = |p: &std::path::Path| {
        p.file_name()
            .is_some_and(|n| n.to_string_lossy().to_lowercase().contains("sana"))
            && p.join("transformer").is_dir()
            && p.join("vae").is_dir()
            && p.join("text_encoder").is_dir()
    };
    if is_sana(&docs) {
        return Some(docs);
    }
    std::fs::read_dir(&docs)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir() && is_sana(p))
}

/// The app's Documents directory. `NSHomeDirectory` is the container root inside the sandbox; on
/// the host (where the same checks run under `cargo test`) `HOME` serves the same role.
fn dirs_documents() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let docs = std::path::PathBuf::from(home).join("Documents");
    docs.is_dir().then_some(docs)
}

/// Append one line to a file in Documents, flushing immediately.
///
/// Best-effort by design: this is diagnostic breadcrumbing for a process that may be killed without
/// warning, so a failure to write it must never fail the check it is instrumenting.
// Both image lanes write breadcrumbs, so this cannot be gated on `media` alone: `--features zimage`
// by itself failed to compile because of it. That combination had never been built — every device run
// so far passed BOTH features — which is precisely why it went unnoticed, and why the z-image lane
// could not be exercised in isolation to rule the SANA graph out of a z-image failure.
#[cfg(any(feature = "media", feature = "zimage"))]
fn append_breadcrumb(name: &str, line: &str) {
    use std::io::Write;
    let Some(docs) = dirs_documents() else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(docs.join(name))
    {
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

/// **E5: SANA image generation on device.** The deciding measurement for S5.2.
///
/// The host harness (`mlx-gen-ios-catalog`'s `image_budget`) puts the sequential peak at 3294-4773
/// MiB depending on resolution and tiling, and it measures the *same* allocator with the *same*
/// weights — but it measures it against `set_memory_limit`, which is backpressure. Jetsam is a kill.
/// So the host number proves the working set fits; only this proves iOS lets the app keep it.
///
/// Reported per configuration rather than asserted against a single threshold: the peak that matters
/// is device-class-specific (a 12 GB phone caps near 6 GB, an 8 GB one near 4), and a threshold set
/// from one device would be noise. What *is* asserted is that an image comes back, at the requested
/// size, and is not degenerate — a decode that silently produces a constant image would otherwise
/// pass every shape and memory check while being worthless.
#[cfg(feature = "media")]
fn check_image_generation(media_dir: Option<&Path>) -> Check {
    const NAME: &str = "SANA image generation (E5)";
    let Some(dir) = media_dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no SANA snapshot in Documents/ (see docs/ios-epics.md E5)"
                .to_string(),
        };
    };

    use runtime_ios::gen_core::{
        GenerationOutput, GenerationRequest, LoadSpec as GenLoadSpec, OffloadPolicy, Progress,
        WeightsSource,
    };

    /// The base (non-Sprint) SANA id. Pinned as a literal on purpose: if the catalog ever stops
    /// registering it, this check must fail loudly rather than quietly pick whatever is first.
    const SANA_ID: &str = "sana_1600m";

    // (label, edge, decode tile in output px). `None` would decode whole-image; nothing here does.
    //
    // BOTH configurations tile, and that is a device finding rather than a preference. An earlier
    // revision ran `512 untiled` — 4773 MiB on the host, comfortably inside a 6136 MiB cap on
    // paper — and it was **jetsam-killed on device** while 1024-at-128px-tiles completed at 2839
    // MiB. So tiling is not merely how large images fit; it is what makes SANA survivable here at
    // all, and the untiled path is not a configuration this harness should be asking the phone for.
    //
    // Why the host number misled: `image_budget` measures under `set_memory_limit`, which applies
    // backpressure and makes MLX evict rather than grow. The device has no such limit — it has
    // jetsam, which kills — so an untiled decode is free to allocate past what the host run
    // recorded. That is the exact gap between "the working set fits" and "iOS lets the app live",
    // and it is why this check exists.
    //
    // Still ordered by ASCENDING measured peak, so a kill leaves the cheaper configuration proven.
    //
    // **512 is a MEMORY data point, not a quality one.** This is the `Sana_1600M_1024px`
    // checkpoint, and 512 is outside its training resolution: the composition collapses (tiny
    // subject, washed-out sky, mushy texture) whether or not the decode tiles. Verified by
    // rendering 512 whole-image on the host, which looks equally wrong — so nothing here is
    // evidence against the tiled decode. The second config stays because a second resolution
    // exercises a different allocation shape, and it is labelled so nobody reads its PNG as a
    // regression.
    const SHIPPING: &[(&str, u32, Option<u32>, bool)] = &[
        ("1024 tile128", 1024, Some(128), false),
        ("512 tile256 (off-distribution: 1024px checkpoint)", 512, Some(256), false),
    ];

    // Under `IOS_SMOKE_IMAGE_ONLY` the untiled configuration comes back — the one that died.
    //
    // It is deliberately NOT in the shipping list: running it is what killed the app, and a harness
    // that routinely asks the phone for a fatal render is a bad harness. Here it IS the experiment.
    // The LLM checks are skipped in this mode, so it runs against a cold process, which separates
    // the two live explanations for that death: a decode genuinely too large, or ~2.9 GB of LLM-era
    // memory still resident underneath it (MLX's own accounting cannot see the difference).
    //
    // The untiled config runs FIRST here, which is the opposite of the shipping list's
    // ascending-peak ordering, and deliberately so.
    //
    // Ordering by ascending peak is a safety property: a kill leaves the cheaper configs proven.
    // But it makes the fatal config untestable, because headroom does not survive a render.
    // Measured on a cold process: `os_proc_available_memory` fell 4664 -> 1223 MiB across a single
    // 512px tiled render, while MLX reported a 3093 MiB peak and RSS only 2068 — so ~3.4 GB left
    // the available pool and did not come back. Any config that runs third starts with ~1.2 GB and
    // dies whatever its own demand, which tells us nothing about that demand.
    //
    // To measure the untiled decode we must give it the headroom a real first render would have.
    // If it survives at full headroom, its death in every previous run was accumulation, not size.
    // Third element `Some(0)` is untiled; fourth forces a GPU sync between decoder stages.
    //
    // The eval-stages row is the discriminating experiment. Plain untiled dies here even first and
    // cold, so total work is not the question any more — the question is whether what kills it is
    // the SUM of the decode's allocations or how many of them are live AT ONCE. MLX is lazy, so an
    // untiled decode can materialize most of a stage graph together; `MLX_GEN_DCAE_EVAL_STAGES`
    // serializes it without changing the work or the tiling.
    //
    // On the host that flag moves the peak by 2 MiB — nothing. If it is the difference between
    // living and dying on device, then jetsam's per-process-limit is being crossed by an
    // instantaneous spike that host accounting never shows, and "how the memory is grabbed" matters
    // as much as how much. That would also explain the direction flip: host over-reads tiled
    // configs by ~16% and under-reads this one by >1.4 GB.
    const DIAGNOSTIC: &[(&str, u32, Option<u32>, bool)] = &[
        ("512 UNTILED + stage-eval (serialized, not tiled)", 512, Some(0), true),
        ("1024 tile128", 1024, Some(128), false),
        ("512 tile256 (off-distribution: 1024px checkpoint)", 512, Some(256), false),
    ];

    let configs: &[(&str, u32, Option<u32>, bool)] =
        if std::env::var_os("IOS_SMOKE_IMAGE_ONLY").is_some() {
            DIAGNOSTIC
        } else {
            SHIPPING
        };

    // Truncate any breadcrumb from a previous run FIRST. A stale one would be read as this run's
    // progress and point the blame at the wrong configuration — the same class of mistake as the
    // stale `smoke-report.txt` that once made a fixed bug look like it persisted.
    if let Some(docs) = dirs_documents() {
        let _ = std::fs::remove_file(docs.join("sana-progress.txt"));
    }

    let mut details: Vec<String> = Vec::new();
    for &(label, edge, tile, stage_eval) in configs {
        // ALWAYS set it explicitly; `0` is the provider's "no tiling" control.
        //
        // This used to `remove_var` for the untiled case, which silently stopped meaning
        // whole-image the moment SANA made tiling the default under `Sequential`: an unset variable
        // now selects the provider default, so a config labelled UNTILED tiled anyway and reported
        // 2566 MiB — a number that would have exonerated the untiled decode on the strength of a
        // run that never performed one. Leaving the variable unset is no longer a way to express
        // anything; only `0` is.
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", tile.unwrap_or(0).to_string());
        // Serialize the decoder's stages: a GPU sync after each, so their transients cannot be
        // live together. Costs a sync per stage and changes no arithmetic.
        if stage_eval {
            std::env::set_var("MLX_GEN_DCAE_EVAL_STAGES", "1");
        } else {
            std::env::remove_var("MLX_GEN_DCAE_EVAL_STAGES");
        }

        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let started = Instant::now();

        let run = || -> Result<String, String> {
            // Sequential is the whole point: it is what sheds the Gemma encoder after conditioning
            // and (since the staged-decode change) the DiT before decode.
            let spec = GenLoadSpec {
                offload_policy: OffloadPolicy::Sequential,
                ..GenLoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
            };
            // Through the REGISTRY, not a direct loader — the same reasoning as the LLM checks
            // taking the bundle rather than `mlx-llm`. This is the path a product takes, so it
            // exercises catalog composition and provider resolution, not just the engine.
            let registry = runtime_ios::media::provider_registry()
                .map_err(|e| format!("{label}: registry build failed: {e}"))?;
            let generator = registry
                .load(SANA_ID, &spec)
                .map_err(|e| format!("{label}: load failed: {e}"))?;

            let request = GenerationRequest {
                prompt: "a lighthouse on a rocky coast at dawn".to_string(),
                width: edge,
                height: edge,
                count: 1,
                steps: Some(4),
                seed: Some(0),
                ..Default::default()
            };
            let mut noop = |_: Progress| {};
            let out = generator
                .generate(&request, &mut noop)
                .map_err(|e| format!("{label}: generate failed: {e}"))?;

            let image = match out {
                GenerationOutput::Images(mut v) if !v.is_empty() => v.remove(0),
                _ => return Err(format!("{label}: generator returned no image")),
            };
            if (image.width, image.height) != (edge, edge) {
                return Err(format!(
                    "{label}: got {}x{}, expected {edge}x{edge}",
                    image.width, image.height
                ));
            }
            // Degeneracy guard. A decode that returns a constant (black, grey, saturated) image has
            // the right shape and the right memory profile and is still worthless, which is exactly
            // the failure a shape check cannot see. Real content spans a wide range.
            let (lo, hi) = image
                .pixels
                .iter()
                .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
            if hi - lo < 32 {
                return Err(format!(
                    "{label}: image is near-constant (range {lo}..{hi}) -- decode produced no content"
                ));
            }

            // Save it. The checks above prove the decode produced *content* (a wide pixel range);
            // only the image proves it produced the *right* content, and it is the artifact worth
            // having — a render made on a phone. Written into Documents so `devicectl copy from`
            // can pull it, named per configuration so neither overwrites the other. Best-effort:
            // failing to encode a PNG must not fail the generation check.
            if let Some(docs) = dirs_documents() {
                let name = format!("sana-{}.png", label.replace(' ', "-"));
                match image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()) {
                    Some(buf) => {
                        if let Err(e) = buf.save(docs.join(&name)) {
                            eprintln!("could not write {name}: {e}");
                        }
                    }
                    None => eprintln!("{name}: pixel buffer does not match {edge}x{edge}"),
                }
            }

            let secs = started.elapsed().as_secs_f64();
            // MLX's own accounting is the number to read, and it is reset per configuration above.
            //
            // RSS is reported alongside it but is NOT the gauge here, for two reasons. It is a
            // process-lifetime high-water mark with no reset, so by the time this runs the LLM
            // checks have already set it and the delta is meaningless. More importantly, on macOS
            // it was measured *below* MLX's peak (2962 vs 4773 MiB) — `getrusage` is not capturing
            // Metal buffer allocations. On iOS those allocations do count toward the footprint
            // jetsam reads, so the divergence is a host artifact; it is printed so the two can be
            // compared on device, where it should not appear.
            let mlx_peak = mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0);
            // Headroom LEFT after the largest allocation of the run. This is the number that says
            // whether the configuration is comfortable or one step from a kill, and unlike a
            // percentage-of-assumed-cap it comes from the OS.
            let headroom = available_memory_mib()
                .map(|m| format!(", {m:.0} MiB still available"))
                .unwrap_or_default();
            Ok(format!(
                "{label}: {secs:.1}s, MLX peak {mlx_peak:.0} MiB, process RSS peak {:.0} MiB{headroom}, \
                 pixel range {lo}..{hi}",
                peak_rss_mib(),
            ))
        };

        match run() {
            Ok(detail) => {
                // Persisted BEFORE the next configuration starts.
                //
                // Jetsam is a kill, not an exception: if the next config exceeds the per-app limit
                // the process dies where it stands and `run_report`'s string — every check in it,
                // including the LLM ones that already passed — is lost with it. The device harness
                // then reports "no report was produced", which is indistinguishable from a launch
                // failure. This breadcrumb survives, so a kill is diagnosable as "died during the
                // configuration after this one" rather than as a mystery.
                append_breadcrumb("sana-progress.txt", &detail);
                details.push(detail);
            }
            Err(e) => {
                std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
                append_breadcrumb("sana-progress.txt", &format!("FAILED {e}"));
                return Check {
                    name: NAME,
                    passed: false,
                    detail: e,
                };
            }
        }
    }
    std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");

    Check {
        name: NAME,
        passed: true,
        detail: details.join(" | "),
    }
}

/// **Z-Image-Turbo on device (E5 go/no-go).** The second image model, and a different bet from SANA.
///
/// # Why this one, and why it might work where SANA's untiled path did not
///
/// Z-Image already carries the whole `gen_core::memory_strategy` ladder (rungs 0-4), which SANA does
/// not. Rung 4 streams the 30-block DiT instead of holding it, so its denoise phase is 1.795 GiB
/// rather than 4.653, and its request peak becomes decode-bound at **4.363 GiB** — measured on host
/// against this exact q4 tier, with the full ladder engaged
/// (`mlx-gen-z-image/tests/block_residency_real_weights.rs`).
///
/// That is 73% of this device's measured 6135 MiB cap. It is also, unlike SANA's fatal
/// configuration, the SHIPPING shape: the number and the thing that runs are the same thing.
///
/// # What would make it fail anyway
///
/// Host readings are shape-dependently wrong in both directions — they over-read SANA's tiled
/// configs by ~16% and under-read its untiled one by >1.4 GB, and the untiled decode died because a
/// *single stage* needed ~1 GB more than the host charged. So 73% is a reason to try, not a
/// prediction. If z-image dies here, the next question is whether one of its stages has that same
/// shape, which the breadcrumb + a stage trace would answer.
///
/// Untiled is never attempted: z-image untiled is **19.172 GiB** at 1024², three times the cap.
#[cfg(feature = "zimage")]
fn check_zimage_generation(dir: Option<&Path>) -> Check {
    const NAME: &str = "Z-Image-Turbo generation (E5 go/no-go)";
    let Some(dir) = dir else {
        return Check {
            name: NAME,
            passed: true,
            detail: "skipped -- no Z-Image q4 tier in Documents/ (push with scripts/ios/push_model.sh)"
                .to_string(),
        };
    };

    use runtime_ios::gen_core::{
        GenerationMemory, GenerationOutput, GenerationRequest, LoadSpec as GenLoadSpec,
        OffloadPolicy, Progress, WeightsSource,
    };

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();

    let run = || -> Result<String, String> {
        // `Sequential` + a `Dir` source are what make rung 4 AVAILABLE at all — z-image declares it
        // Missing for single-file/ComfyUI loads, because streaming rebuilds blocks from the snapshot
        // per window and an in-memory `Weights` has no re-openable source.
        let spec = GenLoadSpec {
            offload_policy: OffloadPolicy::Sequential,
            ..GenLoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        };
        // Resolution is a knob because the 1024px answer was "killed" and the next question is
        // where the boundary actually sits. Z-Image is native at 1024, so anything below is a
        // fit-finding measurement, not a shipping proposal. Bound before the load so the load's own
        // breadcrumbs can name it.
        let edge: u32 = std::env::var("IOS_SMOKE_ZIMAGE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        append_breadcrumb(
            "zimage-progress.txt",
            &format!(
                "  [{edge}px] before load — {} MiB avail",
                available_memory_mib().map(|m| format!("{m:.0}")).unwrap_or_default()
            ),
        );
        let generator =
            mlx_gen_z_image::load(&spec).map_err(|e| format!("load failed: {e}"))?;
        append_breadcrumb(
            "zimage-progress.txt",
            &format!(
                "  [{edge}px] after load — {} MiB avail",
                available_memory_mib().map(|m| format!("{m:.0}")).unwrap_or_default()
            ),
        );

        // Availability is not engagement: the ladder rungs are request-scoped, so they must be asked
        // for. `transformer_window_size: Some(1)` is the published production window.
        let request = GenerationRequest {
            prompt: "a lighthouse on a rocky coast at dawn".to_string(),
            width: edge,
            height: edge,
            count: 1,
            steps: Some(4),
            seed: Some(0),
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                chunk_attention: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                // The decode tile edge, and the reason it is a knob.
                //
                // z-image's published ladder marks every edge below 512 "rejected", because the
                // REQUEST peak pinned at 4.898 GiB however small the tiles got — finer tiles cost
                // fidelity and bought no admission. But that ladder was measured WITHOUT rung 4.
                // With rung 4 carrying denoise down to 1.795 GiB, the decode becomes the binding
                // phase, so shrinking it moves the request peak for the first time:
                //
                //   edge 512 -> decode 4.363 GiB      edge 384 -> 3.896      edge 256 -> 3.544
                //
                // The phase trace puts the kill inside a decode that had 5778 MiB available, so the
                // device charges >1.3 GB more than the host's 4468 for this phase. A smaller edge is
                // the only lever that acts on that phase without changing the model.
                decode_tile_edge: std::env::var("IOS_SMOKE_ZIMAGE_TILE")
                    .ok()
                    .and_then(|v| v.parse().ok()),
                ..Default::default()
            }),
            ..Default::default()
        };

        // What the pipeline DECIDED, not what it was asked for.
        //
        // The 1024px kill happens inside a decode entered with 6007 MiB free, while the identical
        // request on the host peaks at 3102 MiB for the whole run. A >6 GB delta in one phase is not
        // a bounded decode overrunning; it is the shape of this VAE's ~14 GiB untiled 1024²
        // transient. The cheapest explanation is that the bounded configuration never engaged --
        // an unset or unparseable `IOS_SMOKE_ZIMAGE_TILE` resolves to `None`, and twice today a knob
        // that read correctly in source was not the knob the run used. Reading the request back
        // would only restate what was just constructed, so this asks the provider instead.
        append_breadcrumb(
            "zimage-progress.txt",
            &format!(
                "  [{edge}px] decode plan — IOS_SMOKE_ZIMAGE_TILE={:?}, resolved {}",
                std::env::var("IOS_SMOKE_ZIMAGE_TILE").ok(),
                match mlx_gen_z_image::pipeline::resolved_decode_plan(&request, true) {
                    Some((e, o)) => format!("TILED edge={e} overlap={o}"),
                    None => "UNTILED (whole-image) -- ~14 GiB at 1024², expect jetsam".to_string(),
                }
            ),
        );

        // What MLX believes its budget is, against what iOS will actually allow.
        //
        // MLX sizes its memory and cache limits from the system's recommended working-set, which on a
        // Mac is the right denominator. An iOS app is not bounded by the machine's RAM but by a
        // per-process jetsam limit far below it, and nothing tells MLX that. If these two numbers
        // come back near device RAM rather than near `os_proc_available_memory`, then MLX is holding
        // reclaimable buffers up to a ceiling that does not exist, and the app is killed while
        // sitting on memory it would have freed on request.
        append_breadcrumb(
            "zimage-progress.txt",
            &format!(
                "  [{edge}px] MLX budget — memory_limit={:.0} MiB, cache_limit(probe)={:.0} MiB, \
                 os_available={} MiB",
                mlx_rs::memory::get_memory_limit() as f64 / (1024.0 * 1024.0),
                {
                    // No getter for the cache limit: set it to read it, then restore what was there.
                    let prev = mlx_rs::memory::set_cache_limit(0);
                    mlx_rs::memory::set_cache_limit(prev);
                    prev as f64 / (1024.0 * 1024.0)
                },
                available_memory_mib().map(|m| format!("{m:.0}")).unwrap_or_default()
            ),
        );

        // A concurrent sampler, because the interesting failure kills the process mid-decode.
        //
        // Every breadcrumb so far is written by the progress callback, which fires on phase CHANGE --
        // so a phase that dies partway through reports only the headroom it STARTED with, and the
        // trajectory that would explain the death is exactly the part never written. `Decoding` at
        // 6007 MiB free is not evidence the decode is cheap; it is the last thing observed before a
        // 6 GB excursion nobody sampled.
        //
        // 100 ms is chosen against the failure, not the runtime: the host decodes in ~3 s, so a kill
        // partway leaves tens of samples, while the cost is a `writeln` + flush per sample against a
        // phase doing full VAE forwards. Tiles-decoded rides along because footprint alone cannot
        // separate "never tiled" from "tiled and never released" -- the count does.
        mlx_gen::vae_tiling::reset_tiles_decoded();
        let sampling = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let sampler = {
            let sampling = std::sync::Arc::clone(&sampling);
            std::thread::spawn(move || {
                let t0 = Instant::now();
                while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                    // `cache` is not decoration. The 1024px kill ran the decode with 4 MiB of
                    // headroom while MLX reported a 2901 MiB peak — so ~3.2 GB of the process
                    // footprint was invisible to `active`/`peak`. MLX retains freed buffers in a
                    // reuse cache that those two metrics exclude but `phys_footprint` (and therefore
                    // jetsam) counts in full. Without this column the trace shows a bounded decode
                    // being killed for no reason.
                    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
                    append_breadcrumb(
                        "zimage-progress.txt",
                        &format!(
                            "    t={:>6.2}s avail={:>6} MiB active={:>6.0} cache={:>6.0} peak={:>6.0} MiB tiles={}",
                            t0.elapsed().as_secs_f64(),
                            available_memory_mib()
                                .map(|m| format!("{m:.0}"))
                                .unwrap_or_else(|| "n/a".to_string()),
                            mib(mlx_rs::memory::get_active_memory()),
                            mib(mlx_rs::memory::get_cache_memory()),
                            mib(mlx_rs::memory::get_peak_memory()),
                            mlx_gen::vae_tiling::tiles_decoded(),
                        ),
                    );
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            })
        };
        // Stop AND join the sampler on EVERY exit from the generate, including the `?` paths below.
        // Joining rather than detaching matters for the file: a detached sampler can append one more
        // line after the render's own summary, and a breadcrumb file whose tail is out of order is
        // the kind of artifact that gets read as evidence of a phase that never happened.
        struct StopOnDrop(
            std::sync::Arc<std::sync::atomic::AtomicBool>,
            Option<std::thread::JoinHandle<()>>,
        );
        impl Drop for StopOnDrop {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Relaxed);
                if let Some(h) = self.1.take() {
                    let _ = h.join();
                }
            }
        }
        let _stop = StopOnDrop(std::sync::Arc::clone(&sampling), Some(sampler));

        // Phase-level breadcrumbs, because the coarse one is not localizing the kill.
        //
        // Both 1024 and 768 died with NO breadcrumb at all, and the check only writes one after the
        // whole generate returns. Since halving the pixel count changed nothing, the decode peak is
        // probably not what kills it — but "probably not the decode" is not an answer. The provider
        // already emits phase progress; discarding it into a no-op threw away exactly the signal
        // needed. Each distinct phase now appends a line WITH the live headroom, so the last line
        // written names the phase that died and what was left when it started.
        //
        // Written on phase CHANGE, not per step: a 4-step denoise would otherwise write four
        // identical lines and a 30-block window sweep far more.
        let mut last_phase = String::new();
        let mut on_progress = |pr: Progress| {
            let phase = match pr {
                Progress::Step { .. } => "Step".to_string(),
                other => format!("{other:?}"),
            };
            if phase != last_phase {
                let avail = available_memory_mib()
                    .map(|m| format!("{m:.0} MiB avail"))
                    .unwrap_or_else(|| "n/a".to_string());
                append_breadcrumb(
                    "zimage-progress.txt",
                    &format!(
                        "  [{edge}px] phase {phase} — {avail}, MLX active {:.0} MiB",
                        mlx_rs::memory::get_active_memory() as f64 / (1024.0 * 1024.0)
                    ),
                );
                last_phase = phase;
            }
        };
        let out = generator
            .generate(&request, &mut on_progress)
            .map_err(|e| format!("generate failed: {e}"))?;
        let image = match out {
            GenerationOutput::Images(mut v) if !v.is_empty() => v.remove(0),
            _ => return Err("generator returned no image".into()),
        };
        if (image.width, image.height) != (edge, edge) {
            return Err(format!("got {}x{}, expected {edge}x{edge}", image.width, image.height));
        }
        let (lo, hi) = image
            .pixels
            .iter()
            .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        if hi - lo < 32 {
            return Err(format!("near-constant image (range {lo}..{hi}) -- decode produced nothing"));
        }

        if let Some(docs) = dirs_documents() {
            match image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()) {
                Some(buf) => {
                    if let Err(e) = buf.save(docs.join(format!("zimage-{edge}.png"))) {
                        eprintln!("could not write zimage-{edge}.png: {e}");
                    }
                }
                None => eprintln!("zimage pixel buffer does not match dimensions"),
            }
        }

        let secs = started.elapsed().as_secs_f64();
        let mlx_peak = mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0);
        // WHERE the memory went, not just how much peaked.
        //
        // `os_proc_available_memory` falls ~2.5 GB across a render and does not come back, while
        // MLX reports a much smaller peak and `clear_cache()` is already called between configs.
        // Two candidates, and they need opposite fixes:
        //
        //   - MLX's buffer CACHE is holding it -> `set_cache_limit(0)` stops the retention.
        //   - it is WIRED -> `set_wired_limit` is the lever, and `clear_cache` cannot help
        //     because wired pages are not evictable and still count against the footprint.
        //
        // `get_cache_memory()` distinguishes them. Sampled after the render and again after an
        // explicit clear, so the clear's actual effect is visible rather than assumed.
        let cache_before = mlx_rs::memory::get_cache_memory() as f64 / (1024.0 * 1024.0);
        let active_before = mlx_rs::memory::get_active_memory() as f64 / (1024.0 * 1024.0);
        mlx_rs::memory::clear_cache();
        let cache_after = mlx_rs::memory::get_cache_memory() as f64 / (1024.0 * 1024.0);
        let active_after = mlx_rs::memory::get_active_memory() as f64 / (1024.0 * 1024.0);
        let headroom = available_memory_mib()
            .map(|m| format!(", {m:.0} MiB still available"))
            .unwrap_or_default();
        let line = format!(
            "{edge}px full ladder (rung 4 w=1): {secs:.1}s, MLX peak {mlx_peak:.0} MiB, \
             process RSS peak {:.0} MiB{headroom}, pixel range {lo}..{hi} \
             [host predicted 4468 MiB] | cache {cache_before:.0}->{cache_after:.0} MiB, \
             active {active_before:.0}->{active_after:.0} MiB after clear_cache",
            peak_rss_mib(),
        );
        // Breadcrumb before returning: if this model dies, it dies HERE, and the report never gets
        // written. Same reasoning as the SANA lane.
        append_breadcrumb("zimage-progress.txt", &line);
        Ok(line)
    };

    match run() {
        Ok(detail) => Check { name: NAME, passed: true, detail },
        Err(e) => {
            append_breadcrumb("zimage-progress.txt", &format!("FAILED {e}"));
            Check { name: NAME, passed: false, detail: e }
        }
    }
}

/// The Z-Image q4 tier, if pushed. Identified by its diffusers component tree, like SANA's, but
/// under a distinct directory name so the two snapshots coexist without either finder claiming the
/// other's.
#[cfg(feature = "zimage")]
fn find_zimage_snapshot() -> Option<std::path::PathBuf> {
    let docs = dirs_documents()?;
    let looks_right = |p: &std::path::Path| {
        p.join("transformer").is_dir() && p.join("vae").is_dir() && p.join("text_encoder").is_dir()
    };
    std::fs::read_dir(&docs)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.is_dir()
                && p.file_name().is_some_and(|n| n.to_string_lossy().contains("zimage"))
                && looks_right(p)
        })
}

/// Run every check and render a human- and test-readable report.
///
/// The first line is `SMOKE: PASS` or `SMOKE: FAIL` so an XCTest can assert on a prefix without
/// parsing the body.
pub fn run_report() -> String {
    // `IOS_SMOKE_IMAGE_ONLY=1` skips every LLM check and runs the image lane against a cold
    // process. This exists to settle one specific question, not as a convenience.
    //
    // The 512px untiled render died on device at a host-measured 4773 MiB, under a measured 6135
    // MiB cap — which does not add up on its own. It DOES add up if the LLM checks that ran first
    // left their memory in the process: they take RSS to ~2964 MiB, and 2.9 + 4.77 is over the cap
    // while 2.9 + 2.75 (the tiled config, which survived) is not. Every peak this harness reports
    // is MLX's own accounting, which cannot see memory MLX has already released to its cache or
    // that the OS has not reclaimed, so the report cannot distinguish the two explanations.
    //
    // Running the image lane alone does. If the untiled config survives a cold process, the kill
    // was residue and the decode is exonerated; if it dies anyway, the decode really is too large
    // and the residue hypothesis is dead. Either answer is worth one device run.
    let image_only = std::env::var_os("IOS_SMOKE_IMAGE_ONLY").is_some();

    // `IOS_SMOKE_ONLY=sana|zimage` runs ONE model lane and skips the rest.
    //
    // This is not a convenience. Headroom does not survive a render — measured, twice: available
    // memory fell 4664 -> 1223 MiB across one 512px SANA render, and 4620 -> 2093 across the two
    // shipping configs. A model running after another therefore starts with whatever the previous
    // one left, and a model needing more than that dies without ever reporting a number about
    // ITSELF. Z-Image (host: ~4468 MiB) ran third with 2093 MiB available and died before writing
    // a single breadcrumb — which says nothing about Z-Image.
    //
    // Any model whose demand approaches the cap must be measured alone, first, at full headroom.
    let only = std::env::var("IOS_SMOKE_ONLY").unwrap_or_default();
    let run_sana = only.is_empty() || only == "sana";
    let run_zimage = only.is_empty() || only == "zimage";

    let snapshot = if image_only || !only.is_empty() {
        None
    } else {
        find_snapshot()
    };
    #[allow(unused_mut)]
    let mut checks = vec![
        // First: it is the denominator for every memory number below, and it must be sampled
        // before anything large is resident.
        check_memory_headroom(),
        check_metallib_resolves(),
        check_gemm(Dtype::Float32, "f32 GEMM (steel)"),
        check_gemm(Dtype::Bfloat16, "bf16 GEMM (steel)"),
        check_softmax(),
        check_generation(snapshot.as_deref()),
        check_conformance(snapshot.as_deref()),
        check_sustained_decode(snapshot.as_deref()),
        check_unload_seam(snapshot.as_deref()),
        check_thermal_soak(snapshot.as_deref()),
    ];
    // Last, deliberately: it is the largest allocation in the run, so putting it after the LLM
    // checks means a jetsam kill during image generation cannot be mistaken for one during them.
    #[cfg(feature = "media")]
    if run_sana {
        checks.push(check_image_generation(find_media_snapshot().as_deref()));
    }
    // Last of all: the largest single allocation in the run — which is exactly why it usually needs
    // `IOS_SMOKE_ONLY=zimage` to be measurable at all.
    #[cfg(feature = "zimage")]
    if run_zimage {
        checks.push(check_zimage_generation(find_zimage_snapshot().as_deref()));
    }

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
        // Printed, not just asserted. Most of these checks report measurements (throughput, peak
        // RSS, MLX peak) whose VALUE is the output — a bare pass tells you the harness ran and
        // nothing about what it found, and the host numbers are the baseline the device numbers
        // are read against. Visible under `--nocapture`.
        println!("{report}");
        assert!(report.starts_with("SMOKE: PASS"), "{report}");
    }
}
