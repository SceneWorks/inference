//! Guarded real-weight load-admission probe (sc-24446): one load of `LOAD_PROBE_SOURCE` (at
//! `LOAD_PROBE_QUANTIZE` = `q4` | `q8`, or unquantized) through the production
//! [`LlamaProvider::load`] admission, then one-token requests, reporting **exact** `phys_footprint`
//! peaks — the kernel's interval maximum (`crate::common::footprint`), never a sample — next to
//! MLX's own active peaks. Every figure is growth over a baseline taken after the Metal device and
//! MLX's kernel library initialized and the driver settled.
//!
//! The load's share of the peak, by the one method `load_memory::tests::Measured::peak` uses:
//! `max(load_peak, first_request_peak - request_footprint)`. Run one model at a time under the
//! memory guard; never signal-kill it.

use std::time::Instant;

use core_llm::{LoadSpec, Message, Quantize, Sampling, TextLlm, TextLlmRequest, ThinkingMode};
use mlx_llm::LlamaProvider;

use crate::common::footprint;

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
    footprint::settle();
    let baseline = footprint::current();

    footprint::reset_peak();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let provider = LlamaProvider::load(&spec).unwrap_or_else(|e| panic!("load refused: {e}"));
    let load_secs = started.elapsed().as_secs_f64();
    let load_peak = footprint::peak_since_reset().saturating_sub(baseline);
    let after_load = footprint::current().saturating_sub(baseline);
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
    footprint::reset_peak();
    mlx_rs::memory::reset_peak_memory();
    let out = provider
        .generate(&req, &mut |_| {})
        .expect("one-token request");
    let first_request_peak = footprint::peak_since_reset().saturating_sub(baseline);
    let mlx_first_request_peak = mlx_rs::memory::get_peak_memory();

    // A second identical request on the now-materialized model isolates the request's own
    // working set (what request admission prices): exact footprint growth after the driver
    // settled, and MLX's active growth.
    let request_footprint = footprint::peak_growth(|| {
        provider
            .generate(&req, &mut |_| {})
            .expect("second one-token request");
    });
    mlx_rs::memory::clear_cache();
    let before_active = mlx_rs::memory::get_active_memory();
    mlx_rs::memory::reset_peak_memory();
    provider
        .generate(&req, &mut |_| {})
        .expect("third one-token request");
    let request_active = mlx_rs::memory::get_peak_memory() - before_active;

    // Optional decode-speed sample: `LOAD_PROBE_DECODE_TOKENS` greedy tokens.
    let decode_tokens_per_sec = std::env::var("LOAD_PROBE_DECODE_TOKENS")
        .ok()
        .and_then(|n| n.parse::<u32>().ok())
        .map(|n| {
            let req = TextLlmRequest {
                messages: vec![Message::user(
                    "Write a long story about a lighthouse keeper.",
                )],
                max_new_tokens: n,
                ..req.clone()
            };
            let started = Instant::now();
            let out = provider.generate(&req, &mut |_| {}).expect("decode sample");
            f64::from(out.usage.generated_tokens) / started.elapsed().as_secs_f64()
        });
    println!(
        "LOAD_PROBE {}",
        serde_json::json!({
            "source": source,
            "quantize": quantize.map(|q| format!("{q:?}")),
            "sampling": "exact",
            "baseline_footprint": baseline,
            "load_peak": load_peak,
            "after_load": after_load,
            "first_request_peak": first_request_peak,
            "request_footprint": request_footprint,
            "request_active": request_active,
            "load_share": load_peak.max(first_request_peak.saturating_sub(request_footprint)),
            "lifetime_peak_growth": footprint::lifetime_peak().saturating_sub(baseline),
            "mlx_load_peak_active": mlx_load_peak,
            "mlx_active_after_load": mlx_after_load,
            "mlx_first_request_peak_active": mlx_first_request_peak,
            "load_secs": load_secs,
            "tokens": out.usage.generated_tokens,
            "decode_tokens_per_sec": decode_tokens_per_sec,
        })
    );
}
