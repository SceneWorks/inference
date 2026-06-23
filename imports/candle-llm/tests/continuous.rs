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

use candle_llm::config::ModelConfig;
use candle_llm::decode::{
    generate, generate_batch, generate_continuous, BatchExactness, BatchRequest, CancelFlag,
    ContinuousConfig, GenerationConfig, StreamEvent,
};
use candle_llm::models::CausalLm;
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

fn build_tiny_llama() -> CausalLm {
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
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model =
        CausalLm::from_weights(&Weights::from_dir(&dir, &Device::Cpu).unwrap(), "", cfg).unwrap();
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
fn batch1(model: &CausalLm, r: &BatchRequest) -> Vec<i32> {
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
    model: &CausalLm,
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

    fn load(env: &str) -> Option<(CausalLm, Tokenizer)> {
        let dir = std::env::var(env).ok().filter(|v| !v.is_empty())?;
        let device = candle_llm::device::select_device().unwrap();
        let cfg = ModelConfig::from_dir(&dir).unwrap();
        let model =
            CausalLm::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg).unwrap();
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

    /// The measurement harness behind the sc-7258 split into sc-7351 (varlen) + sc-7258 (custom
    /// kernel). Over **uniform-length** sequences it pits `generate_batch` (which already batches
    /// attention into one masked SDPA — the throughput ceiling) against continuous `Throughput` (the
    /// per-sequence attention path this story optimizes), reporting tok/s and the gap by occupancy.
    ///
    /// Before sc-7351 the per-sequence loop flatlined (~50 tok/s on the RTX PRO 6000) while
    /// `generate_batch` scaled to 263–453 at N=16 — a 5–9× gap. With `--features flash-attn` the
    /// per-sequence loop is now one `flash_attn_varlen` call, which should close most of that gap; run
    /// the same build with and without `flash-attn` to see the before/after. Ratios are the signal, so
    /// a debug build is fine. Set `CANDLE_LLM_TEST_MODEL` (SmolLM2) and/or `CANDLE_LLM_QWEN3_MODEL`.
    #[test]
    #[ignore = "needs CANDLE_LLM_TEST_MODEL / CANDLE_LLM_QWEN3_MODEL (CUDA bench)"]
    fn attention_bottleneck_bound() {
        // ~11 tokens; short prompt so decode (not prefill) dominates the timing.
        const PROMPT: &str = "The quick brown fox jumps over the lazy dog near the";
        const NEW: usize = 64;
        for env in ["CANDLE_LLM_TEST_MODEL", "CANDLE_LLM_QWEN3_MODEL"] {
            let Some((model, tok)) = load(env) else {
                eprintln!("skip: set {env}");
                continue;
            };
            let prompt = encode(&tok, PROMPT);
            println!(
                "[{env}] attention_bottleneck_bound (prompt {} tok, {NEW} new):",
                prompt.len()
            );
            for n in [1usize, 2, 4, 8, 16] {
                let reqs: Vec<BatchRequest> = (0..n).map(|_| req(prompt.clone(), NEW)).collect();

                // generate_batch: attention batched into one masked SDPA (no gather, no per-seq loop).
                let t = Instant::now();
                let outs =
                    generate_batch(&model, &reqs, &CancelFlag::new(), &mut |_, _| {}).unwrap();
                let batch_toks: usize = outs.iter().map(|o| o.tokens.len()).sum();
                let batch_tps = batch_toks as f64 / t.elapsed().as_secs_f64();

                // continuous Throughput: the per-sequence (now varlen) attention path.
                let cfg = ContinuousConfig {
                    max_batch: n,
                    block_size: 16,
                    exactness: BatchExactness::Throughput,
                };
                let t = Instant::now();
                let outs = run(&model, &reqs, &cfg);
                let cont_toks: usize = outs.iter().map(|o| o.tokens.len()).sum();
                let cont_tps = cont_toks as f64 / t.elapsed().as_secs_f64();

                let gap = 100.0 * (1.0 - cont_tps / batch_tps);
                println!(
                    "  N={n:<2} generate_batch {batch_tps:7.1} tok/s | continuous Throughput \
                     {cont_tps:7.1} tok/s | gap {gap:4.0}%"
                );
                assert!(outs.iter().all(|o| !o.tokens.is_empty()));
            }
        }
    }

    /// **sc-7477 ragged + admit-on-retire serving bench.** `attention_bottleneck_bound` uses
    /// **uniform** lengths and budgets — `generate_batch`'s best case (zero padding waste, one full
    /// batch) and continuous batching's worst relative case — so its 14–39% gap is an upper bound. The
    /// real continuous-batching win is **ragged lengths + ragged budgets under a concurrency cap**:
    /// `generate_batch` cannot admit a new request mid-flight, so to serve `M > max_batch` requests at
    /// `≤ max_batch` concurrency it must run them in **chunks** that each drain to completion (a short
    /// request idles its slot until the chunk's longest sequence finishes); continuous **admits on
    /// retire**, refilling each freed slot immediately and keeping the pipe full.
    ///
    /// This pits the two at equal concurrency (`max_batch`) over `M` requests, in a **uniform** and a
    /// **ragged** scenario, reporting realized tok/s and the gap. The hypothesis (sc-7477): the gap is
    /// largest uniform and **shrinks or inverts** ragged. Also prints `generate_batch` as one
    /// `M`-wide batch (its raw-throughput best case, ignoring the no-mid-flight-admission constraint) for
    /// reference. Set `CANDLE_LLM_TEST_MODEL` (SmolLM2) and/or `CANDLE_LLM_QWEN3_MODEL`.
    #[test]
    #[ignore = "needs CANDLE_LLM_TEST_MODEL / CANDLE_LLM_QWEN3_MODEL (CUDA bench)"]
    fn ragged_churn_serving_bench() {
        const M: usize = 48; // total requests served
        const MAX_BATCH: usize = 16; // concurrency cap for both engines
                                     // Ragged prompt lengths + budgets (cycled across the M requests).
        const PLENS: [usize; 6] = [8, 16, 32, 64, 96, 128];
        const BUDGETS: [usize; 5] = [16, 32, 48, 64, 96];

        for env in ["CANDLE_LLM_TEST_MODEL", "CANDLE_LLM_QWEN3_MODEL"] {
            let Some((model, tok)) = load(env) else {
                eprintln!("skip: set {env}");
                continue;
            };
            // A long real-token pool to slice ragged prompt lengths from (repeat to clear 128 tokens).
            let big =
                "The quick brown fox jumps over the lazy dog near the old riverbank, while a \
                       curious cat watched the clouds drift slowly across the wide afternoon sky. "
                    .repeat(12);
            let pool = encode(&tok, &big);
            assert!(pool.len() >= 128, "token pool too short: {}", pool.len());

            // Sum of generated tokens over a set of generate_batch chunks, capped to MAX_BATCH each.
            let gb_chunked = |reqs: &[BatchRequest]| -> (usize, f64) {
                let t = Instant::now();
                let mut toks = 0usize;
                for chunk in reqs.chunks(MAX_BATCH) {
                    let outs =
                        generate_batch(&model, chunk, &CancelFlag::new(), &mut |_, _| {}).unwrap();
                    toks += outs.iter().map(|o| o.tokens.len()).sum::<usize>();
                }
                (toks, t.elapsed().as_secs_f64())
            };
            let cont = |reqs: &[BatchRequest]| -> (usize, f64) {
                let cfg = ContinuousConfig {
                    max_batch: MAX_BATCH,
                    block_size: 16,
                    exactness: BatchExactness::Throughput,
                };
                let t = Instant::now();
                let outs = run(&model, reqs, &cfg);
                let toks: usize = outs.iter().map(|o| o.tokens.len()).sum();
                (toks, t.elapsed().as_secs_f64())
            };
            let report = |label: &str, reqs: &[BatchRequest]| {
                let (gt, gs) = gb_chunked(reqs);
                let (ct, cs) = cont(reqs);
                let (gtps, ctps) = (gt as f64 / gs, ct as f64 / cs);
                println!(
                    "  [{label}] M={M} cap={MAX_BATCH}: generate_batch(chunked) {gtps:7.1} tok/s \
                     ({gt} tok {gs:.2}s) | continuous {ctps:7.1} tok/s ({ct} tok {cs:.2}s) | \
                     gap {:4.0}%",
                    100.0 * (1.0 - ctps / gtps)
                );
            };

            println!("\n[{env}] ragged_churn_serving_bench:");
            // UNIFORM control: every request the same (longest) prompt + (largest) budget.
            let uniform: Vec<BatchRequest> = (0..M)
                .map(|_| {
                    req(
                        pool[..*PLENS.last().unwrap()].to_vec(),
                        *BUDGETS.last().unwrap(),
                    )
                })
                .collect();
            report("uniform", &uniform);

            // RAGGED: prompt lengths and budgets cycle across requests.
            let ragged: Vec<BatchRequest> = (0..M)
                .map(|i| {
                    req(
                        pool[..PLENS[i % PLENS.len()]].to_vec(),
                        BUDGETS[i % BUDGETS.len()],
                    )
                })
                .collect();
            report("ragged ", &ragged);

            // Reference: generate_batch as ONE M-wide batch over the ragged set (its raw-throughput
            // best case — assumes all M present up front, ignoring the no-mid-flight-admission limit).
            let t = Instant::now();
            let outs = generate_batch(&model, &ragged, &CancelFlag::new(), &mut |_, _| {}).unwrap();
            let big_toks: usize = outs.iter().map(|o| o.tokens.len()).sum();
            println!(
                "  [ref] generate_batch one {M}-wide batch (ragged): {:7.1} tok/s ({big_toks} tok {:.2}s)",
                big_toks as f64 / t.elapsed().as_secs_f64(),
                t.elapsed().as_secs_f64()
            );
        }
    }
}
