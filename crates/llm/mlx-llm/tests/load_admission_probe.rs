//! Guarded real-weight load-admission probe (sc-24446): one load of `LOAD_PROBE_SOURCE` (at
//! `LOAD_PROBE_QUANTIZE` = `q4` | `q8`, or unquantized) through the production
//! [`LlamaProvider::load`] admission, then a one-token request, reporting the process's peak
//! `phys_footprint` sampled in-process every 100 ms next to MLX's own peak active memory. The
//! peak growth over the pre-load footprint is what `load_memory`'s estimate must cover. Run one
//! model at a time under the memory guard; never signal-kill it.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use core_llm::{LoadSpec, Message, Quantize, Sampling, TextLlm, TextLlmRequest, ThinkingMode};
use mlx_llm::LlamaProvider;

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
}

/// This process's `ri_phys_footprint` (`rusage_info_v2`: a 16-byte UUID, then seven `u64`
/// counters before it).
fn phys_footprint() -> u64 {
    let mut info = [0u64; 64];
    // SAFETY: `info` outlives the call and is larger than `rusage_info_v2`.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 2, info.as_mut_ptr()) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    info[9]
}

#[test]
#[ignore = "guarded real-weight probe: set LOAD_PROBE_SOURCE (and LOAD_PROBE_QUANTIZE=q4|q8)"]
fn load_admission_probe() {
    let source = std::env::var("LOAD_PROBE_SOURCE").expect("set LOAD_PROBE_SOURCE");
    let quantize = match std::env::var("LOAD_PROBE_QUANTIZE").as_deref() {
        Ok("q4") => Some(Quantize::Q4),
        Ok("q8") => Some(Quantize::Q8),
        Ok("") | Err(_) => None,
        Ok(other) => panic!("unknown LOAD_PROBE_QUANTIZE {other}"),
    };
    let mut spec = LoadSpec::dense(&source);
    spec.quantize = quantize;
    // No cross-turn prefix snapshot: the probe measures the load and the first forward only.
    spec.prefix_cache_bytes = Some(0);

    // Initialize the Metal device and load MLX's kernel library first: a process-wide one-time
    // cost (the metallib alone is ~170 MB), not a cost of this load.
    let one = mlx_rs::Array::from_slice(&[1.0f32], &[1]);
    mlx_rs::ops::add(&one, &one).unwrap().eval().unwrap();
    let baseline = phys_footprint();
    let peak = Arc::new(AtomicU64::new(baseline));
    let done = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (peak, done) = (peak.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                peak.fetch_max(phys_footprint(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(100));
            }
        })
    };
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let provider = LlamaProvider::load(&spec).unwrap_or_else(|e| panic!("load refused: {e}"));
    let load_secs = started.elapsed().as_secs_f64();
    let after_load_peak = peak.load(Ordering::Relaxed).max(phys_footprint());
    let mlx_load_peak = mlx_rs::memory::get_peak_memory();
    let mlx_after_load = mlx_rs::memory::get_active_memory();
    let req = TextLlmRequest {
        messages: vec![Message::user("Hi")],
        sampling: Sampling::greedy(),
        max_new_tokens: 1,
        thinking: ThinkingMode::Disabled,
        seed: Some(0),
        ..Default::default()
    };
    let out = provider
        .generate(&req, &mut |_| {})
        .expect("one-token request");
    let peak_first = peak.load(Ordering::Relaxed).max(phys_footprint());
    let mlx_peak_first = mlx_rs::memory::get_peak_memory();
    // A second identical request on the now-materialized model isolates the request's own
    // working set (what request admission prices) from the first forward's load-derived arrays.
    mlx_rs::memory::clear_cache();
    let before_second = mlx_rs::memory::get_active_memory();
    let footprint_before_second = phys_footprint();
    peak.store(footprint_before_second, Ordering::Relaxed);
    mlx_rs::memory::reset_peak_memory();
    provider
        .generate(&req, &mut |_| {})
        .expect("second one-token request");
    let mlx_request_working_set = mlx_rs::memory::get_peak_memory() - before_second;
    let footprint_request_growth =
        peak.load(Ordering::Relaxed).max(phys_footprint()) - footprint_before_second;
    done.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    let peak = peak_first;
    println!(
        "LOAD_PROBE {}",
        serde_json::json!({
            "source": source,
            "quantize": quantize.map(|q| format!("{q:?}")),
            "baseline_footprint": baseline,
            "peak_footprint": peak,
            "peak_growth": peak - baseline,
            "after_load_peak_growth": after_load_peak - baseline,
            "mlx_load_peak_active": mlx_load_peak,
            "mlx_active_after_load": mlx_after_load,
            "mlx_peak_active": mlx_peak_first,
            "mlx_active_materialized": before_second,
            "mlx_request_working_set": mlx_request_working_set,
            "footprint_request_growth": footprint_request_growth,
            "mlx_active_end": mlx_rs::memory::get_active_memory(),
            "mlx_cache_end": mlx_rs::memory::get_cache_memory(),
            "load_secs": load_secs,
            "tokens": out.usage.generated_tokens,
        })
    );
}
