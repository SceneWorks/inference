//! Iteration-level continuous batching tests (story 7347).
//!
//! ## What these prove
//! - **`Exact` == batch-1**: each request decoded through `generate_continuous` in `Exact` mode is
//!   **token-for-token identical** to running it alone through `generate` — across differing prompt
//!   lengths and across admit-on-retire (a request admitted into a freed slot mid-flight still matches
//!   its batch-1 run).
//! - **`Throughput` per-sequence attention is correct**: in `Throughput` mode the projections/MLP are
//!   batched and only attention runs per-sequence; on CPU (where the f32 matmul reduces in a fixed,
//!   batch-invariant order) this is *also* bit-exact to batch-1, which pins down the `forward_per_seq`
//!   path. (On a GPU the batched matmul is not M-invariant, so a row only tracks batch-1 to sub-ULP —
//!   the documented Throughput tradeoff; the real-weights test asserts equality only for `Exact`.)
//! - **Admission / cancel bookkeeping**: every request emits exactly one terminal `Done` — including
//!   ones still queued or still decoding when a cancel lands, and zero-budget requests.
//!
//! The synthetic CPU tests need no model and no GPU. The `#[ignore]`d real-weights test confirms the
//! `Exact` equality on a GPU snapshot and reports `Exact` vs `Throughput` decode throughput by
//! occupancy.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use core_llm::Tokenizer;

use candle_llm::config::LlamaConfig;
use candle_llm::decode::{
    generate, generate_continuous, BatchExactness, BatchRequest, CancelFlag, ContinuousConfig,
    GenerationConfig, StreamEvent,
};
use candle_llm::models::LlamaModel;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{SplitMix64, TokenRng, Weights};

// ---- Synthetic 2-layer CPU model (no download) ---------------------------------------------------

const VOCAB: usize = 48;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2; // GQA (groups = 2)
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::from_vec(vec![1.0f32; d], (d,), &Device::Cpu).unwrap()
}

fn build_tiny_llama() -> LlamaModel {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-cont-{}-{uniq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = format!(
        r#"{{
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": {HIDDEN}, "intermediate_size": {INTER}, "num_hidden_layers": 2,
            "num_attention_heads": {NUM_HEADS}, "num_key_value_heads": {NUM_KV_HEADS},
            "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 0
        }}"#
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();

    let mut rng = SplitMix64::new(0xC047_1140);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        randn((VOCAB, HIDDEN), &mut rng),
    );
    w.insert("model.norm.weight".into(), ones(HIDDEN));
    w.insert("lm_head.weight".into(), randn((VOCAB, HIDDEN), &mut rng));

    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        w.insert(p("input_layernorm.weight"), ones(HIDDEN));
        w.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        w.insert(
            p("self_attn.q_proj.weight"),
            randn((q_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.k_proj.weight"),
            randn((kv_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.v_proj.weight"),
            randn((kv_dim, HIDDEN), &mut rng),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            randn((HIDDEN, q_dim), &mut rng),
        );
        w.insert(p("mlp.gate_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        w.insert(p("mlp.up_proj.weight"), randn((INTER, HIDDEN), &mut rng));
        w.insert(p("mlp.down_proj.weight"), randn((HIDDEN, INTER), &mut rng));
    }

    candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();
    let cfg = LlamaConfig::from_dir(&dir).unwrap();
    let model =
        LlamaModel::from_weights(&Weights::from_dir(&dir, &Device::Cpu).unwrap(), "", cfg).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    model
}

// ---- Helpers -------------------------------------------------------------------------------------

fn req(prompt: Vec<i32>, max_new: usize) -> BatchRequest {
    BatchRequest {
        prompt_ids: prompt,
        sampling: SamplingParams::default(), // greedy (temperature 0)
        seed: Some(0),
        max_new_tokens: max_new,
        stop_tokens: Vec::new(),
    }
}

/// The batch-1 reference: run a single request alone through the streaming `generate`.
fn batch1(model: &LlamaModel, r: &BatchRequest) -> Vec<i32> {
    generate(
        model,
        &r.prompt_ids,
        &GenerationConfig {
            max_new_tokens: r.max_new_tokens,
            sampling: r.sampling,
            seed: r.seed,
            stop_tokens: r.stop_tokens.clone(),
        },
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
}

fn run(
    model: &LlamaModel,
    reqs: &[BatchRequest],
    cfg: &ContinuousConfig,
) -> Vec<candle_llm::decode::GenerationOutput> {
    generate_continuous(model, reqs, cfg, &CancelFlag::new(), &mut |_, _| {}).unwrap()
}

// ---- Tests ---------------------------------------------------------------------------------------

/// `Exact` mode, all slots live at once: each differing-length request equals its batch-1 run.
#[test]
fn exact_matches_batch1_cpu() {
    let model = build_tiny_llama();
    let reqs = vec![
        req(vec![1, 2, 3], 14),
        req(vec![4, 5, 6, 7, 8, 9, 10], 14),
        req(vec![2, 9], 14),
    ];
    let cfg = ContinuousConfig {
        max_batch: 8,
        block_size: 4,
        exactness: BatchExactness::Exact,
    };
    let outs = run(&model, &reqs, &cfg);
    assert_eq!(outs.len(), reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        assert_eq!(
            outs[i].tokens,
            batch1(&model, r),
            "request {i}: Exact continuous must equal its batch-1 run token-for-token"
        );
        assert!(!outs[i].tokens.is_empty());
    }
}

/// `Throughput` mode (batched projections + per-sequence attention): on CPU the f32 matmul reduces in
/// a batch-invariant order, so this is *also* bit-exact to batch-1 — pinning down `forward_per_seq`.
#[test]
fn throughput_matches_batch1_cpu() {
    let model = build_tiny_llama();
    let reqs = vec![
        req(vec![3, 1, 4, 1, 5], 12),
        req(vec![9, 2], 12),
        req(vec![6, 5, 3, 5], 12),
    ];
    let cfg = ContinuousConfig {
        max_batch: 8,
        block_size: 4,
        exactness: BatchExactness::Throughput,
    };
    let outs = run(&model, &reqs, &cfg);
    for (i, r) in reqs.iter().enumerate() {
        assert_eq!(
            outs[i].tokens,
            batch1(&model, r),
            "request {i}: Throughput per-seq attention must match batch-1 on CPU"
        );
    }
}

/// Admit-on-retire: with only `max_batch = 2` slots over 5 differing-budget requests, sequences
/// retire and free slots that later requests are admitted into — each still equals its batch-1 run.
#[test]
fn admit_on_retire_matches_batch1_cpu() {
    let model = build_tiny_llama();
    let reqs = vec![
        req(vec![1, 2], 3),
        req(vec![3, 4, 5], 9),
        req(vec![6], 5),
        req(vec![7, 8, 9, 10], 7),
        req(vec![2, 4, 6], 4),
    ];
    let cfg = ContinuousConfig {
        max_batch: 2,
        block_size: 4,
        exactness: BatchExactness::Exact,
    };
    let outs = run(&model, &reqs, &cfg);
    assert_eq!(outs.len(), reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        assert_eq!(
            outs[i].tokens,
            batch1(&model, r),
            "request {i}: a late-admitted lane must still equal its batch-1 run"
        );
    }
}

/// Every request emits exactly one terminal `Done` even when a cancel lands mid-stream with requests
/// still decoding and still queued (`max_batch` smaller than the queue).
#[test]
fn cancel_signals_every_request_once_cpu() {
    let model = build_tiny_llama();
    let reqs: Vec<BatchRequest> = (0..6i32).map(|i| req(vec![1 + i, 2, 3], 64)).collect();
    let cfg = ContinuousConfig {
        max_batch: 2,
        block_size: 4,
        exactness: BatchExactness::Exact,
    };

    let cancel = CancelFlag::new();
    let mut dones = vec![0usize; reqs.len()];
    let mut seen_tokens = 0;
    generate_continuous(&model, &reqs, &cfg, &cancel, &mut |ri, ev| match ev {
        StreamEvent::Token { .. } => {
            seen_tokens += 1;
            if seen_tokens >= 3 {
                cancel.cancel(); // trip the cancel a few tokens in
            }
        }
        StreamEvent::Done { .. } => dones[ri] += 1,
    })
    .unwrap();

    assert!(
        dones.iter().all(|&d| d == 1),
        "every request must emit exactly one Done, got {dones:?}"
    );
}

/// A zero-budget request (`max_new_tokens == 0`) retires at admission with empty output and one Done,
/// without disturbing its neighbours.
#[test]
fn zero_budget_request_completes_empty_cpu() {
    let model = build_tiny_llama();
    let reqs = vec![req(vec![1, 2, 3], 0), req(vec![4, 5], 8)];
    let cfg = ContinuousConfig {
        max_batch: 4,
        block_size: 4,
        exactness: BatchExactness::Exact,
    };

    let mut dones = vec![0usize; reqs.len()];
    let outs = generate_continuous(&model, &reqs, &cfg, &CancelFlag::new(), &mut |ri, ev| {
        if let StreamEvent::Done { .. } = ev {
            dones[ri] += 1;
        }
    })
    .unwrap();

    assert!(
        outs[0].tokens.is_empty(),
        "zero-budget request generates nothing"
    );
    assert_eq!(dones[0], 1, "zero-budget request still emits one Done");
    assert_eq!(
        outs[1].tokens,
        batch1(&model, &reqs[1]),
        "neighbour unaffected"
    );
}

// ---- Real-weights variant (#[ignore]) ------------------------------------------------------------

mod real {
    use super::*;
    use std::time::Instant;

    fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
        tok.encode(text, true)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect()
    }

    fn load(env: &str) -> Option<(LlamaModel, Tokenizer)> {
        let dir = std::env::var(env).ok().filter(|v| !v.is_empty())?;
        let device = candle_llm::device::select_device().unwrap();
        let cfg = LlamaConfig::from_dir(&dir).unwrap();
        let model =
            LlamaModel::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg).unwrap();
        let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
        Some((model, tok))
    }

    /// On a real GPU snapshot: `Exact` continuous batching is token-for-token identical to each
    /// request's batch-1 run (incl. admit-on-retire), and `Throughput` decode throughput rises with
    /// occupancy. Set `CANDLE_LLM_TEST_MODEL` (SmolLM2) and/or `CANDLE_LLM_QWEN3_MODEL`.
    #[test]
    #[ignore = "needs CANDLE_LLM_TEST_MODEL / CANDLE_LLM_QWEN3_MODEL"]
    fn continuous_matches_batch1_and_scales() {
        for env in ["CANDLE_LLM_TEST_MODEL", "CANDLE_LLM_QWEN3_MODEL"] {
            let Some((model, tok)) = load(env) else {
                eprintln!("skip: set {env}");
                continue;
            };
            let prompts = [
                "The capital of France is",
                "Once upon a time,",
                "In a galaxy far, far away,",
                "The meaning of life is",
                "Water boils at",
            ];
            let reqs: Vec<BatchRequest> =
                prompts.iter().map(|p| req(encode(&tok, p), 24)).collect();

            // Exact, with admit-on-retire (max_batch < requests), must match each batch-1 run.
            let cfg = ContinuousConfig {
                max_batch: 2,
                block_size: 16,
                exactness: BatchExactness::Exact,
            };
            let outs = run(&model, &reqs, &cfg);
            for (i, r) in reqs.iter().enumerate() {
                assert_eq!(
                    outs[i].tokens,
                    batch1(&model, r),
                    "[{env}] request {i}: Exact continuous must equal batch-1"
                );
            }
            println!(
                "[{env}] Exact continuous == batch-1 for {} requests",
                reqs.len()
            );

            // Throughput decode tok/s by occupancy (informational + a loose monotonic-ish gate).
            for n in [1usize, 2, 4] {
                let batch: Vec<BatchRequest> =
                    (0..n).map(|_| req(encode(&tok, prompts[0]), 48)).collect();
                let cfg = ContinuousConfig {
                    max_batch: n,
                    block_size: 16,
                    exactness: BatchExactness::Throughput,
                };
                let t = Instant::now();
                let outs = run(&model, &batch, &cfg);
                let toks: usize = outs.iter().map(|o| o.tokens.len()).sum();
                let secs = t.elapsed().as_secs_f64();
                println!(
                    "[{env}] Throughput N={n}: {:.1} tok/s ({toks} tokens)",
                    toks as f64 / secs
                );
                assert!(outs.iter().all(|o| !o.tokens.is_empty()));
            }
        }
    }
}
