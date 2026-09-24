//! `load_residency` — where a provider load's device memory goes, format by format (sc-24140
//! terminal measurement, E6).
//!
//! One `#[ignore]`d test that loads a snapshot through [`LlamaProvider::load`] in one weight
//! format, generates a greedy fixture, and records the device memory at each stage next to the
//! figure admission priced the load at ([`LlamaProvider::load_memory_estimate`], the function the
//! load itself admits with) and the loaded weight census. Every sample is taken after a
//! device-wide synchronize and reads both the driver (`cuMemGetInfo` total - free: device-wide,
//! so the CUDA context, loaded kernels and any co-tenant are included) and the device's current
//! CUDA memory pool — the stream-ordered allocator cudarc serves every candle allocation from:
//! `reserved` (what the pool holds from the driver) against `used` (live allocations), so idle
//! pool capacity that the owning process reuses is told apart from live tensors. Peaks are the
//! pool's high watermarks (reset before the load and again before the generation) and a
//! background `cuMemGetInfo` poll over the load (device-wide).
//!
//! | variable                    | meaning                                                      |
//! |-----------------------------|--------------------------------------------------------------|
//! | `LOAD_RESIDENCY_SNAPSHOT`   | snapshot directory (config.json, tokenizer*.json, shards)    |
//! | `LOAD_RESIDENCY_FORMAT`     | `bf16` (default, dense), `q8`, `q4` or `nvfp4` — `LoadSpec::quantize` |
//! | `LOAD_RESIDENCY_NEW_TOKENS` | greedy tokens generated (default 256)                        |
//! | `LOAD_RESIDENCY_OUTPUT`     | JSON path to write (optional; the document is always printed) |
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=1 LOAD_RESIDENCY_SNAPSHOT=E:\...\models--Qwen--Qwen3-8B\snapshots\<rev> \
//!   LOAD_RESIDENCY_FORMAT=q8 cargo test --release --features cuda -p candle-llm \
//!   --test load_residency -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaContext};
use candle_llm::device::select_device;
use candle_llm::primitives::ProjectionTally;
use candle_llm::LlamaProvider;
use core_llm::{
    LoadSpec, Message, MtpMode, Quantize, Sampling, StreamEvent, TextLlm, TextLlmRequest,
};
use serde_json::{json, Value};

const PROMPT: &str = "Write a detailed, multi-paragraph explanation of how transformer language \
    models generate text. Cover tokenization, self-attention, the key/value cache, and greedy \
    versus sampled decoding, and finish with the trade-offs of speculative decoding.";

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The device's current memory pool (the one `cuMemAllocAsync` on the default stream serves from).
fn pool(ctx: &CudaContext) -> sys::CUmemoryPool {
    ctx.bind_to_thread().unwrap();
    // SAFETY: the live context is bound to this thread and `cu_device` is owned by it.
    unsafe { result::device::get_mem_pool(ctx.cu_device()) }.unwrap()
}

fn pool_attr(ctx: &CudaContext, attr: sys::CUmemPool_attribute) -> u64 {
    let mut value = 0u64;
    // SAFETY: every attribute read here is a u64 counter of the live pool.
    unsafe { result::mem_pool::get_attribute(pool(ctx), attr, (&mut value as *mut u64).cast()) }
        .unwrap();
    value
}

/// Reset the pool's high watermarks (a set to zero resets them to the current value).
fn reset_pool_peaks(ctx: &CudaContext) {
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    ] {
        let mut zero = 0u64;
        // SAFETY: both high-watermark attributes take a u64; zero is the documented reset.
        unsafe { result::mem_pool::set_attribute(pool(ctx), attr, (&mut zero as *mut u64).cast()) }
            .unwrap();
    }
}

/// One synchronized sample: device-wide used, and the pool's reserved / used and their peaks.
fn sample(ctx: &CudaContext) -> Value {
    use sys::CUmemPool_attribute as A;
    ctx.synchronize().unwrap();
    let (free, total) = ctx.mem_get_info().unwrap();
    json!({
        "device_used_bytes": (total - free) as u64,
        "pool_reserved_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        "pool_used_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        "pool_reserved_peak_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
        "pool_used_peak_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_USED_MEM_HIGH),
    })
}

/// Poll `cuMemGetInfo` from a background thread until `stop`, keeping the device-wide maximum.
fn poll_device_peak(
    ctx: Arc<CudaContext>,
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if let Ok((free, total)) = ctx.mem_get_info() {
                peak.fetch_max((total - free) as u64, Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })
}

fn tally_json(t: &ProjectionTally) -> Value {
    json!({ "count": t.count, "params": t.params, "resident_bytes": t.resident_bytes })
}

#[test]
#[ignore = "needs a snapshot via LOAD_RESIDENCY_SNAPSHOT and a CUDA GPU"]
fn load_residency_by_format() {
    let snapshot = std::env::var("LOAD_RESIDENCY_SNAPSHOT").expect("set LOAD_RESIDENCY_SNAPSHOT");
    let format = env_or("LOAD_RESIDENCY_FORMAT", "bf16");
    let quantize = match format.as_str() {
        "bf16" => None,
        "q8" => Some(Quantize::Q8),
        "q4" => Some(Quantize::Q4),
        "nvfp4" => Some(Quantize::Nvfp4),
        other => panic!("LOAD_RESIDENCY_FORMAT must be bf16, q8, q4 or nvfp4, got {other:?}"),
    };
    let new_tokens: u32 = env_or("LOAD_RESIDENCY_NEW_TOKENS", "256").parse().unwrap();
    let spec = LoadSpec {
        quantize,
        cuda_graphs: Some(false),
        ..LoadSpec::dense(snapshot.clone())
    };

    // A handle on the same primary context and pool the provider's device will use.
    let device = select_device().unwrap();
    let ctx = device
        .as_cuda_device()
        .unwrap()
        .cuda_stream()
        .context()
        .clone();
    let estimate = LlamaProvider::load_memory_estimate(&spec, true).unwrap();
    let payload = estimate.payload_bytes;
    // The device bound at R (e9faeabdb) for comparison: payload + 25 % headroom, plus the packed
    // copy for NVFP4 only — the GGML copy was unpriced there.
    let admission_at_r = payload
        + payload / 4
        + if quantize == Some(Quantize::Nvfp4) {
            payload * 9 / 32
        } else {
            0
        };

    let before_load = sample(&ctx);
    reset_pool_peaks(&ctx);
    let stop = Arc::new(AtomicBool::new(false));
    let polled_peak = Arc::new(AtomicU64::new(0));
    let poller = poll_device_peak(ctx.clone(), stop.clone(), polled_peak.clone());
    let started = Instant::now();
    let provider = LlamaProvider::load(&spec).expect("load");
    let load_secs = started.elapsed().as_secs_f64();
    let after_load = sample(&ctx);
    stop.store(true, Ordering::Relaxed);
    poller.join().unwrap();
    let load_device_peak = polled_peak.load(Ordering::Relaxed);

    let record = provider.load_record();
    let census = record.census.expect("a census");
    let total = census.total();

    reset_pool_peaks(&ctx);
    let request = TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        max_new_tokens: new_tokens,
        seed: Some(0),
        mtp: MtpMode::Off,
        ..Default::default()
    };
    // Sampled at every streamed token (the stream can merge a token into its neighbour's text
    // event, so the last event is not always index `new_tokens - 1`): the first decode step
    // (index 0 comes from the prefill) and the last event, while the request's cache is alive.
    let mut after_first_step = Value::Null;
    let mut at_last_token = Value::Null;
    let mut events = 0usize;
    let out = provider
        .generate(&request, &mut |event| {
            if let StreamEvent::Token { index, .. } = event {
                events += 1;
                let now = sample(&ctx);
                if index == 1 {
                    after_first_step = now.clone();
                }
                at_last_token = json!({ "index": index, "sample": now });
            }
        })
        .expect("generate");
    let after_generate = sample(&ctx);
    let decode_record = provider.last_decode_record();
    drop(provider);
    let after_drop = sample(&ctx);
    // SAFETY: trimming the live pool to zero idle bytes only returns unused reservations.
    unsafe { result::mem_pool::trim_to(pool(&ctx), 0) }.unwrap();
    let after_drop_trimmed = sample(&ctx);

    let doc = json!({
        "label": "RTX Pro 6000 / sm_120",
        "snapshot": snapshot,
        "format": format,
        "new_tokens": new_tokens,
        "generated_tokens": out.usage.generated_tokens,
        "token_events": events,
        "finish": format!("{:?}", out.finish_reason),
        "decode_record": format!("{decode_record:?}"),
        "load_secs": load_secs,
        "admission": {
            "payload_bytes": payload,
            "host_required_bytes": estimate.host_required_bytes,
            "device_required_bytes": estimate.device_required_bytes,
            "quantized_copy_bytes": estimate.quantized_copy_bytes,
            "device_required_bytes_at_r": admission_at_r,
        },
        "census": {
            "total_resident_bytes": total.resident_bytes,
            "dense": tally_json(&census.projections.dense),
            "ggml": tally_json(&census.projections.ggml),
            "nvfp4": tally_json(&census.projections.nvfp4),
            "other": tally_json(&census.other),
        },
        "samples": {
            "before_load": before_load,
            "after_load": after_load,
            "load_device_peak_polled_bytes": load_device_peak,
            "after_first_decode_step": after_first_step,
            "at_last_token": at_last_token,
            "after_generate": after_generate,
            "after_provider_drop": after_drop,
            "after_provider_drop_trimmed": after_drop_trimmed,
        },
    });
    let text = serde_json::to_string_pretty(&doc).unwrap();
    eprintln!("[load_residency] {text}");
    if let Ok(path) = std::env::var("LOAD_RESIDENCY_OUTPUT") {
        std::fs::write(&path, &text).unwrap();
    }
    assert_eq!(
        out.usage.generated_tokens, new_tokens,
        "the fixture ran to length"
    );
}
