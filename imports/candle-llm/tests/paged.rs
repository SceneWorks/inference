//! Paged KV-cache tests (story 7257).
//!
//! ## What these prove
//! - **Drop-in parity**: a sequence decoded through a `PagedKvCache` is **token-for-token identical**
//!   to the same sequence through the contiguous cache — gather returns the same per-position KV, in
//!   order, attended over by the same kernels. A fresh paged cache runs the *same forward shapes* as
//!   the contiguous path, so this holds bit-exactly even on a GPU.
//! - **Ragged batches**: sequences of differing lengths, each on its own paged cache over one shared
//!   pool, each decode correctly (bit-exact vs their own cold run) — what the left-padded contiguous
//!   batch can only approximate at sub-ULP.
//! - **Bounded memory + copy-on-write prefix sharing**: a sequence reserves `ceil(len/block_size)`
//!   blocks, never a `max_context` slab; two divergent requests sharing a system prefix point at the
//!   *same physical blocks* (refcounted, not copied).
//!
//! The synthetic CPU tests run with no download and assert full bit-exactness. The `#[ignore]`d
//! real-weights tests confirm the drop-in parity and reservation/sharing accounting on a GPU snapshot;
//! the *shared-suffix* output is checked only by its greedy first token, since the reused prefix was
//! computed in a different-length prefill and a bf16 GEMM is not perfectly `M`-invariant (the same
//! sub-ULP limitation the prefix cache documents) — full shared-suffix bit-exactness is proven on CPU.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use core_llm::Tokenizer;

use candle_llm::config::ModelConfig;
use candle_llm::decode::{generate, generate_with_cache, CancelFlag, GenerationConfig};
use candle_llm::device::select_device;
use candle_llm::models::CausalLm;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{BlockPool, KvCache, PagedKvCache, SplitMix64, TokenRng, Weights};

const BLOCK_SIZE: usize = 4;

// ---- Shared helpers ------------------------------------------------------------------------------

/// Greedy (temperature 0 ⇒ deterministic), fixed seed, no stop tokens — bit-exactness of paged vs
/// contiguous is the point, and the full budget keeps the assertion content-independent.
fn config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

/// Cold single-sequence run through the default (contiguous) cache.
fn cold(model: &CausalLm, prompt: &[i32], max_new: usize) -> Vec<i32> {
    generate(
        model,
        prompt,
        &config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
}

/// Run through a caller-provided (paged) cache.
fn paged(model: &CausalLm, prompt: &[i32], max_new: usize, cache: &mut PagedKvCache) -> Vec<i32> {
    generate_with_cache(
        model,
        prompt,
        cache,
        &config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
}

// ---- Synthetic CPU model (no download) -----------------------------------------------------------

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

/// Write a tiny 2-layer `llama` snapshot from deterministic random weights and load it on CPU.
///
/// A per-call atomic sequence keeps the temp dir unique so the concurrently-running synthetic tests
/// never share (and delete) one another's snapshot.
fn build_tiny_llama() -> CausalLm {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-paged-{}-{uniq}", std::process::id()));
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

    let mut rng = SplitMix64::new(0x50A6_ED10);
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
    let weights = Weights::from_dir(&dir, &Device::Cpu).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    model
}

#[test]
fn paged_is_bitexact_dropin_cpu() {
    let model = build_tiny_llama();
    // A prompt long enough to span several blocks (block_size 4) plus a partial tail.
    let prompt: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let base = cold(&model, &prompt, 12);
    assert!(!base.is_empty());

    let mut cache = model.new_paged_cache(BLOCK_SIZE);
    let out = paged(&model, &prompt, 12, &mut cache);
    assert_eq!(
        out, base,
        "paged cache must be token-for-token identical to contiguous"
    );

    // Reservation is ~length, never a max-context slab.
    let len = prompt.len() + out.len();
    assert!(
        cache.reserved_tokens() <= len + BLOCK_SIZE,
        "paged reserves ~len ({}), got {}",
        len,
        cache.reserved_tokens()
    );
}

#[test]
fn ragged_lengths_decode_correctly_cpu() {
    let model = build_tiny_llama();
    // Three sequences of *different* prompt lengths — a ragged batch the contiguous cache can't hold
    // without left-padding. Each is driven on its own paged cache over one shared pool.
    let prompts: [Vec<i32>; 3] = [
        vec![3, 1, 4],
        vec![2, 7, 1, 8, 2, 8, 1],
        vec![9, 9, 5, 6, 5],
    ];
    let max_new = 10;
    let colds: Vec<Vec<i32>> = prompts.iter().map(|p| cold(&model, p, max_new)).collect();

    let pool = BlockPool::new(BLOCK_SIZE);
    let nl = model.config().num_layers;
    let mut outs = Vec::new();
    for p in &prompts {
        let mut cache = PagedKvCache::with_pool(pool.clone(), nl);
        outs.push(paged(&model, p, max_new, &mut cache));
    }
    for (i, (out, base)) in outs.iter().zip(&colds).enumerate() {
        assert_eq!(
            out,
            base,
            "ragged sequence {i} (len {}) must match its cold run",
            prompts[i].len()
        );
    }
}

#[test]
fn shared_prefix_cow_is_bitexact_cpu() {
    let model = build_tiny_llama();
    let nl = model.config().num_layers;
    // A block-aligned shared prefix (8 tokens = 2 blocks at block_size 4), then two suffixes that
    // diverge on their first token.
    let sys: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let q1: Vec<i32> = vec![10, 11, 12];
    let q2: Vec<i32> = vec![20, 21, 22];
    let p1: Vec<i32> = sys.iter().chain(&q1).copied().collect();
    let p2: Vec<i32> = sys.iter().chain(&q2).copied().collect();
    let max_new = 10;
    let cold1 = cold(&model, &p1, max_new);
    let cold2 = cold(&model, &p2, max_new);

    let pool = BlockPool::new(BLOCK_SIZE);
    // Sequence 1 populates the shared system-prefix blocks.
    let mut c1 = PagedKvCache::with_pool(pool.clone(), nl);
    let out1 = paged(&model, &p1, max_new, &mut c1);
    assert_eq!(out1, cold1, "paged seq 1 must match its cold run");

    // Sequence 2 adopts seq 1's whole system-prefix blocks (copy-on-write, no recompute, no copy).
    let shared = c1.shareable_prefix_blocks(sys.len());
    assert_eq!(
        shared.len(),
        sys.len() / BLOCK_SIZE,
        "the system prefix is whole blocks"
    );
    let mut c2 = PagedKvCache::new_seeded(pool.clone(), nl, &shared);
    assert_eq!(
        c2.offset(),
        (shared.len() * BLOCK_SIZE) as i32,
        "seeded past the shared prefix"
    );
    let out2 = paged(&model, &p2, max_new, &mut c2);
    assert_eq!(
        out2, cold2,
        "paged seq 2 sharing seq 1's prefix blocks must match its cold run"
    );

    // The shared blocks are physically shared (refcount > 1) and counted once.
    {
        let p = pool.borrow();
        assert_eq!(
            p.shared_blocks(),
            shared.len(),
            "the whole system prefix is shared"
        );
        assert_eq!(
            p.live_blocks(),
            c1.blocks() + c2.blocks() - shared.len(),
            "shared prefix blocks are referenced, not duplicated"
        );
        assert!(
            p.live_blocks() < c1.blocks() + c2.blocks(),
            "sharing reduces the block count"
        );
    }
    drop(c2);
    assert_eq!(
        pool.borrow().shared_blocks(),
        0,
        "dropping seq 2 releases the shared references"
    );
}

// ---- Real-weights variants (#[ignore]) -----------------------------------------------------------

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
    max_ctx: i32,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok().filter(|p| !p.is_empty())?;
    let device = select_device().unwrap();
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let max_ctx = cfg.max_position_embeddings;
    let model =
        CausalLm::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some(Fixture {
        model,
        tok,
        max_ctx,
    })
}

fn encode(tok: &Tokenizer, text: &str, bos: bool) -> Vec<i32> {
    tok.encode(text, bos)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

fn run_real(fx: Fixture) {
    let max_new = 24;
    let nl = fx.model.config().num_layers;

    // ---- Drop-in parity: a fresh paged cache is token-for-token identical to the contiguous cache
    // (same forward shapes, so bit-exact even on GPU). ----
    let prompt = encode(&fx.tok, "The capital of France is", true);
    let base = cold(&fx.model, &prompt, max_new);
    assert!(!base.is_empty());
    let mut paged_cache = fx.model.new_paged_cache(8);
    let out = paged(&fx.model, &prompt, max_new, &mut paged_cache);
    assert_eq!(
        out, base,
        "paged cache must be a token-for-token drop-in for the contiguous cache"
    );

    // ---- Near-zero reservation waste vs a naive max-context slab. ----
    let len = prompt.len() + out.len();
    let reserved = paged_cache.reserved_tokens();
    let naive = fx.max_ctx.max(2048) as usize;
    assert!(reserved <= len + 8, "paged reserves ~len, not a fixed max");
    assert!(
        reserved * 8 < naive,
        "paged reservation must be a small fraction of naive max-context"
    );
    println!(
        "reservation: paged {reserved} tokens for a {len}-token sequence vs naive {naive} ({:.1}x less)",
        naive as f64 / reserved as f64
    );

    // ---- Shared-prefix sequences share blocks (copy-on-write). ----
    let sys = encode(
        &fx.tok,
        "You are a helpful, knowledgeable, and meticulous assistant. Always answer accurately and \
         concisely, reason step by step when a question requires it, cite concrete facts, and avoid \
         unnecessary repetition, filler, or hedging in your responses.\n\n",
        true,
    );
    let q1 = encode(&fx.tok, "What is the capital of France?", false);
    let q2 = encode(&fx.tok, "Name three primary colors.", false);
    assert_ne!(q1.first(), q2.first());
    let p1: Vec<i32> = sys.iter().chain(&q1).copied().collect();
    let p2: Vec<i32> = sys.iter().chain(&q2).copied().collect();

    let pool = BlockPool::new(8);
    let mut c1 = PagedKvCache::with_pool(pool.clone(), nl);
    let out1 = paged(&fx.model, &p1, max_new, &mut c1);
    assert_eq!(
        out1,
        cold(&fx.model, &p1, max_new),
        "paged seq 1 matches its cold run (fresh cache)"
    );

    let shared = c1.shareable_prefix_blocks(sys.len());
    assert!(
        !shared.is_empty(),
        "the system prefix should span at least one block"
    );
    let mut c2 = PagedKvCache::new_seeded(pool.clone(), nl, &shared);
    let out2 = paged(&fx.model, &p2, max_new, &mut c2);
    // The shared-suffix output reuses a prefix from a different-length prefill, so on GPU it is only
    // sub-ULP-near (full bit-exactness is the CPU `shared_prefix_cow_is_bitexact_cpu` test); the
    // greedy first token still matches, and the block sharing is exact.
    let cold2 = cold(&fx.model, &p2, max_new);
    assert_eq!(
        out2.first(),
        cold2.first(),
        "shared-prefix seq 2 picks the same greedy first token"
    );
    {
        let p = pool.borrow();
        assert_eq!(
            p.shared_blocks(),
            shared.len(),
            "the whole system prefix is shared"
        );
        assert_eq!(
            p.live_blocks(),
            c1.blocks() + c2.blocks() - shared.len(),
            "shared prefix blocks are referenced, not duplicated"
        );
        assert!(
            p.live_blocks() < c1.blocks() + c2.blocks(),
            "sharing reduces the block count"
        );
        println!(
            "sharing: {} blocks shared; pool holds {} live blocks (vs {} if independent)",
            shared.len(),
            p.live_blocks(),
            c1.blocks() + c2.blocks()
        );
    }
    drop(c2);
    assert_eq!(
        pool.borrow().shared_blocks(),
        0,
        "dropping seq 2 releases the shared references"
    );
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_TEST_MODEL"]
fn paged_cache_real_weights_llama() {
    let Some(fx) = load_from("CANDLE_LLM_TEST_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
        return;
    };
    run_real(fx);
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_QWEN3_MODEL"]
fn paged_cache_real_weights_qwen3() {
    let Some(fx) = load_from("CANDLE_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_QWEN3_MODEL");
        return;
    };
    run_real(fx);
}
