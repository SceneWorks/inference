//! Throughput bench (`#[ignore]` — needs a model on disk), story 7263 (CUDA perf).
//!
//! Reports prefill and decode tokens/s for a reference decoder across compute dtypes (bf16 vs f16 on
//! GPU) and quantized loads (Q8), so dtype / quant perf is *measured*, not assumed. Built **without**
//! `flash-attn` the attention runs the eager softmax SDPA; built **with** `--features flash-attn` the
//! same bench exercises the fused kernel — run it both ways to read the flash-vs-eager speedup (the
//! header line prints which path this build took). Point `CANDLE_LLM_TEST_MODEL` at a Llama-family
//! snapshot and run (CUDA is the intended target):
//!
//! ```text
//! CANDLE_LLM_TEST_MODEL=/path/SmolLM2-135M-Instruct \
//!   cargo test --features cuda           --test bench -- --ignored --nocapture   # eager
//! CANDLE_LLM_TEST_MODEL=/path/SmolLM2-135M-Instruct \
//!   cargo test --features flash-attn     --test bench -- --ignored --nocapture   # fused
//! ```
//!
//! Token *values* don't affect the compute, so the prompt is synthesized (ids `0..vocab`) — no
//! tokenizer needed. Each region is timed with a warmup pass and a `device.synchronize()` on both
//! sides so the wall-clock reflects kernel time, not Candle's async dispatch.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use candle_llm::config::ModelConfig;
use candle_llm::device::{compute_dtype, select_device};
use candle_llm::models::CausalLm;
use candle_llm::primitives::{input_ids, QuantSpec, Weights};

const PREFILL_TOKENS: usize = 256;
const DECODE_STEPS: usize = 128;
const WARMUP_DECODE_STEPS: usize = 8;

/// One load configuration to time.
struct Variant {
    label: &'static str,
    dtype: DType,
    quant: Option<QuantSpec>,
}

/// A `[1, n]` u32 id tensor with ids in `0..vocab` (values are irrelevant to the timing).
fn synth_ids(n: usize, vocab: usize, device: &Device) -> Tensor {
    let ids: Vec<i32> = (0..n).map(|i| (i % vocab.max(1)) as i32).collect();
    input_ids(&ids, device).unwrap()
}

/// Assert the last-position logits are finite (the forward actually produced numbers).
fn assert_finite(logits: &Tensor) {
    let v = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        v.first().is_some_and(|x| x.is_finite()),
        "logits must be finite"
    );
}

/// Time prefill (one P-token forward) and decode (single-token steps from the prefilled cache) for a
/// freshly loaded variant, and print tokens/s.
fn bench_variant(dir: &str, device: &Device, v: &Variant) {
    let cfg = ModelConfig::from_dir(dir).unwrap();
    let vocab = cfg.vocab_size as usize;
    let weights = Weights::from_dir(dir, device).unwrap();
    let model = CausalLm::from_weights_dtype(&weights, "", cfg, v.quant, v.dtype).unwrap();

    // ---- Prefill: a single forward over PREFILL_TOKENS, after a warmup pass. ----
    {
        let mut cache = model.new_cache();
        let ids = synth_ids(PREFILL_TOKENS, vocab, device);
        let _ = model.decode_logits(&ids, &mut cache, 0).unwrap();
        device.synchronize().unwrap();
    }
    let mut cache = model.new_cache();
    let ids = synth_ids(PREFILL_TOKENS, vocab, device);
    device.synchronize().unwrap();
    let t = Instant::now();
    let logits = model.decode_logits(&ids, &mut cache, 0).unwrap();
    device.synchronize().unwrap();
    let prefill_tps = PREFILL_TOKENS as f64 / t.elapsed().as_secs_f64();
    assert_finite(&logits);

    // ---- Decode: continue the prefilled cache one token at a time. ----
    let mut offset = PREFILL_TOKENS as i32;
    for _ in 0..WARMUP_DECODE_STEPS {
        let one = synth_ids(1, vocab, device);
        let _ = model.decode_logits(&one, &mut cache, offset).unwrap();
        offset += 1;
    }
    device.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..DECODE_STEPS {
        let one = synth_ids(1, vocab, device);
        let _ = model.decode_logits(&one, &mut cache, offset).unwrap();
        offset += 1;
    }
    device.synchronize().unwrap();
    let decode_tps = DECODE_STEPS as f64 / t.elapsed().as_secs_f64();

    println!(
        "{:<12} prefill {:>9.1} tok/s   decode {:>8.1} tok/s",
        v.label, prefill_tps, decode_tps
    );
    assert!(
        prefill_tps > 0.0 && decode_tps > 0.0,
        "{}: throughput must be positive",
        v.label
    );
}

/// Sweep the dtype / quant variants appropriate for the selected device and print a tokens/s table.
#[test]
#[ignore = "needs a Llama-family snapshot via CANDLE_LLM_TEST_MODEL"]
fn throughput_sweep() {
    let Some(dir) = std::env::var("CANDLE_LLM_TEST_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
        return;
    };
    let device = select_device().unwrap();
    let default_dt = compute_dtype(&device);

    println!(
        "# candle-llm throughput — device {device:?}, flash-attn feature: {}, prefill {PREFILL_TOKENS} / decode {DECODE_STEPS} tokens",
        cfg!(feature = "flash-attn")
    );

    // On the GPU compare the half-precision dtypes; on CPU only f32 is sensible. Q8_0 quantize-on-load
    // applies broadly (block size 32); Q4_K needs 256-aligned in-dims, so it's left out of the sweep.
    let variants: Vec<Variant> = if device.is_cpu() {
        vec![
            Variant {
                label: "f32",
                dtype: DType::F32,
                quant: None,
            },
            Variant {
                label: "f32+Q8",
                dtype: DType::F32,
                quant: Some(QuantSpec::q8()),
            },
        ]
    } else {
        vec![
            Variant {
                label: "bf16",
                dtype: DType::BF16,
                quant: None,
            },
            Variant {
                label: "f16",
                dtype: DType::F16,
                quant: None,
            },
            Variant {
                label: default_dt_label(default_dt),
                dtype: default_dt,
                quant: Some(QuantSpec::q8()),
            },
        ]
    };

    for v in &variants {
        bench_variant(&dir, &device, v);
    }
}

/// Label for the quantized row: it dequantizes to the device's default compute dtype.
fn default_dt_label(dt: DType) -> &'static str {
    match dt {
        DType::F16 => "f16+Q8",
        _ => "bf16+Q8",
    }
}
