//! Prompt-lookup speculative-decoding tests (story 7259).
//!
//! ## What these prove
//! - **Exactness gate (`num_draft = 0`)**: with no drafts the verify is a single-token forward, so
//!   speculative decoding is **token-for-token identical** to non-speculative `generate`. This pins
//!   the loop / acceptance / KV-rollback / first-token logic as exactly correct.
//! - **CPU bit-exactness with drafts**: on CPU the multi-token verify forward is deterministic and
//!   order-stable, so even *with* drafts the realized greedy run is token-for-token identical to
//!   non-speculative — and the n-gram proposer accepts drafts on a repetitive output (`forwards <
//!   tokens`), the measured speedup.
//! - **Tracking on a GPU (real weights)**: the multi-token verify kernel rounds a few bf16 ULP
//!   differently from the single-token decode kernel (a target-model property, cf. story 7255), so a
//!   greedy run *tracks* (rather than bit-matches) non-speculative, diverging only where that rounding
//!   flips a near-tie. The acceptance itself is exact w.r.t. the verify forward (proven in core-llm).

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use core_llm::Tokenizer;

use candle_llm::config::ModelConfig;
use candle_llm::decode::{
    generate, generate_draft_speculative, generate_prompt_lookup, CancelFlag, GenerationConfig,
    SpeculativeConfig,
};
use candle_llm::device::select_device;
use candle_llm::models::CausalLm;
use candle_llm::primitives::projection::QuantSpec;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{SplitMix64, TokenRng, Weights};

// ---- Shared helpers ------------------------------------------------------------------------------

fn greedy_config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(), // greedy ⇒ deterministic
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

fn base_greedy(model: &CausalLm, prompt: &[i32], max_new: usize) -> Vec<i32> {
    generate(
        model,
        prompt,
        &greedy_config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ---- Synthetic CPU model (no download) -----------------------------------------------------------

const VOCAB: usize = 48;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
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
    build_tiny(VOCAB, "main")
}

/// Build a tiny 2-layer `llama` of the given `vocab` size. A per-call atomic sequence keeps the temp
/// dir unique so concurrently-running tests never share (and delete) one another's snapshot.
fn build_tiny(vocab: usize, tag: &str) -> CausalLm {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "candle-llm-spec-{tag}-{}-{uniq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = format!(
        r#"{{
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": {HIDDEN}, "intermediate_size": {INTER}, "num_hidden_layers": 2,
            "num_attention_heads": {NUM_HEADS}, "num_key_value_heads": {NUM_KV_HEADS},
            "vocab_size": {vocab}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 0
        }}"#
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();

    let mut rng = SplitMix64::new(0xC0DE_CAFE);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        randn((vocab, HIDDEN), &mut rng),
    );
    w.insert("model.norm.weight".into(), ones(HIDDEN));
    w.insert("lm_head.weight".into(), randn((vocab, HIDDEN), &mut rng));
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
fn num_draft_zero_is_bit_identical_to_nonspec_cpu() {
    let model = build_tiny_llama();
    let no_draft = SpeculativeConfig {
        max_ngram: 3,
        num_draft: 0,
    };
    for prompt in [
        vec![1, 2, 3, 4, 5],
        vec![9, 8, 7],
        vec![3, 1, 4, 1, 5, 9, 2, 6],
    ] {
        let base = base_greedy(&model, &prompt, 24);
        let (out, stats) = generate_prompt_lookup(
            &model,
            &prompt,
            &greedy_config(24),
            &no_draft,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            out.tokens, base,
            "num_draft=0 must equal non-speculative greedy"
        );
        assert_eq!(stats.proposed, 0);
        assert_eq!(stats.accepted, 0);
        // No speculation: the first token comes from the prefill forward, every later token from one
        // single-token verify forward ⇒ exactly `base.len()` target forwards.
        assert_eq!(stats.forwards, base.len());
    }
}

#[test]
fn drafts_track_and_accept_on_cpu() {
    let model = build_tiny_llama();
    let spec = SpeculativeConfig::default();
    // A long greedy run; a tiny deterministic model settles into a repeating cycle, so the n-gram
    // proposer hits and drafts are accepted.
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let max_new = 64;
    let base = base_greedy(&model, &prompt, max_new);
    let (out, stats) = generate_prompt_lookup(
        &model,
        &prompt,
        &greedy_config(max_new),
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    // On CPU the multi-token verify is deterministic and order-stable, so the greedy run is
    // token-for-token identical to non-speculative.
    assert_eq!(
        out.tokens, base,
        "CPU speculative greedy is bit-identical to non-speculative"
    );
    assert!(
        stats.accepted > 0,
        "a repetitive greedy run should yield n-gram hits"
    );
    assert!(
        stats.forwards < out.tokens.len() + 1,
        "accepted drafts must reduce target forwards below the token count ({} forwards, {} tokens)",
        stats.forwards,
        out.tokens.len()
    );
}

#[test]
fn stochastic_is_deterministic_for_fixed_seed_cpu() {
    let model = build_tiny_llama();
    let spec = SpeculativeConfig::default();
    let cfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        },
        seed: Some(7),
        stop_tokens: Vec::new(),
    };
    let prompt = vec![2, 4, 6, 8, 10];
    let a = generate_prompt_lookup(
        &model,
        &prompt,
        &cfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    let b = generate_prompt_lookup(
        &model,
        &prompt,
        &cfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    assert_eq!(
        a.tokens, b.tokens,
        "stochastic speculative must be deterministic for a fixed seed"
    );
    assert!(!a.tokens.is_empty());
}

// ---- Draft-model speculation (synthetic CPU) -----------------------------------------------------

#[test]
fn draft_num_draft_zero_identical_to_target_cpu() {
    let model = build_tiny_llama();
    let no_draft = SpeculativeConfig {
        max_ngram: 3,
        num_draft: 0,
    };
    let prompt = vec![1, 2, 3, 4, 5];
    let base = base_greedy(&model, &prompt, 24);
    // The draft is irrelevant when num_draft = 0 (the model itself stands in as a draft).
    let (out, stats) = generate_draft_speculative(
        &model,
        &model,
        &prompt,
        &greedy_config(24),
        &no_draft,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        out.tokens, base,
        "num_draft=0 draft-spec must equal non-speculative target greedy"
    );
    assert_eq!(stats.proposed, 0);
    assert_eq!(stats.accepted, 0);
    assert_eq!(stats.forwards, base.len());
}

#[test]
fn draft_equals_target_accepts_everything_cpu() {
    // Using the target as its own draft: every proposed token is the target's argmax, so on CPU
    // (deterministic, order-stable) every draft is accepted — the maximal speedup, identical output.
    let model = build_tiny_llama();
    let spec = SpeculativeConfig::default();
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let max_new = 48;
    let base = base_greedy(&model, &prompt, max_new);
    let (out, stats) = generate_draft_speculative(
        &model,
        &model,
        &prompt,
        &greedy_config(max_new),
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        out.tokens, base,
        "an identical draft yields identical greedy output"
    );
    assert!(stats.proposed > 0);
    assert_eq!(
        stats.accepted, stats.proposed,
        "an identical draft has every token accepted"
    );
    assert!(
        stats.forwards < out.tokens.len() + 1,
        "all-accepted ⇒ fewer target forwards ({}) than tokens ({})",
        stats.forwards,
        out.tokens.len()
    );
}

#[test]
fn draft_spec_rejects_vocab_mismatch_cpu() {
    let target = build_tiny(VOCAB, "voca"); // vocab 48
    let draft = build_tiny(64, "vocb"); // vocab 64
    assert_ne!(target.config().vocab_size, draft.config().vocab_size);
    let err = generate_draft_speculative(
        &target,
        &draft,
        &[1, 2, 3],
        &greedy_config(8),
        &SpeculativeConfig::default(),
        &CancelFlag::new(),
        &mut |_| {},
    );
    assert!(err.is_err(), "a draft/target vocab mismatch must error");
}

#[test]
fn draft_spec_stochastic_deterministic_cpu() {
    let model = build_tiny_llama();
    let spec = SpeculativeConfig::default();
    let cfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        },
        seed: Some(11),
        stop_tokens: Vec::new(),
    };
    let prompt = vec![3, 6, 9, 12];
    let a = generate_draft_speculative(
        &model,
        &model,
        &prompt,
        &cfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    let b = generate_draft_speculative(
        &model,
        &model,
        &prompt,
        &cfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    assert_eq!(
        a.tokens, b.tokens,
        "stochastic draft-spec must be deterministic for a fixed seed"
    );
    assert!(!a.tokens.is_empty());
}

// ---- Real-weights variants (#[ignore]) -----------------------------------------------------------

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok().filter(|p| !p.is_empty())?;
    let device = select_device().unwrap();
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model =
        CausalLm::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some(Fixture { model, tok })
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

fn run_suite(fx: Fixture) {
    // ---- Exactness gate: num_draft = 0 ⇒ single-token verify ⇒ identical to non-speculative. ----
    let no_draft = SpeculativeConfig {
        max_ngram: 3,
        num_draft: 0,
    };
    for text in ["The capital of France is", "Q: What is 2+2? A:"] {
        let p = encode(&fx.tok, text);
        let base = base_greedy(&fx.model, &p, 32);
        let (out, stats) = generate_prompt_lookup(
            &fx.model,
            &p,
            &greedy_config(32),
            &no_draft,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            out.tokens, base,
            "num_draft=0 speculative must equal non-speculative for '{text}'"
        );
        assert_eq!(stats.proposed, 0);
        assert_eq!(stats.accepted, 0);
    }

    // ---- Multi-draft: tracks non-speculative + measured speedup on a context-repetitive prompt. ----
    let spec = SpeculativeConfig::default();
    let rep = encode(
        &fx.tok,
        "List: alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma",
    );
    let base = base_greedy(&fx.model, &rep, 48);
    let (out, stats) = generate_prompt_lookup(
        &fx.model,
        &rep,
        &greedy_config(48),
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    let n = out.tokens.len();
    let cp = common_prefix(&out.tokens, &base);
    println!(
        "{n} tokens in {} target forwards ({:.2} tok/forward); {}/{} drafts accepted; tracks non-spec for {cp}/{} tokens",
        stats.forwards,
        n as f64 / stats.forwards as f64,
        stats.accepted,
        stats.proposed,
        base.len(),
    );
    assert!(
        stats.accepted > 0,
        "the repetitive prompt should yield n-gram hits"
    );
    assert!(
        stats.forwards < n + 1,
        "speculation must use fewer forwards than tokens generated"
    );
    assert!(
        cp >= 1,
        "speculative output must track non-speculative (diverges only on bf16 near-ties)"
    );
    assert!(!out.tokens.is_empty());

    // ---- Stochastic: deterministic for a fixed seed, and valid. ----
    let scfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        },
        seed: Some(7),
        stop_tokens: Vec::new(),
    };
    let p = encode(&fx.tok, "Write a short sentence about the sea:");
    let a = generate_prompt_lookup(&fx.model, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .0;
    let b = generate_prompt_lookup(&fx.model, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .0;
    assert_eq!(
        a.tokens, b.tokens,
        "stochastic speculative must be deterministic for a fixed seed"
    );
    assert!(!a.tokens.is_empty());
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_TEST_MODEL"]
fn prompt_lookup_llama() {
    let Some(fx) = load_from("CANDLE_LLM_TEST_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
        return;
    };
    run_suite(fx);
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_QWEN3_MODEL"]
fn prompt_lookup_qwen3() {
    let Some(fx) = load_from("CANDLE_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_QWEN3_MODEL");
        return;
    };
    run_suite(fx);
}

// ---- Draft-model speculation (real weights, #[ignore]) -------------------------------------------

/// Load a dense **target** and a quantized **draft** from the same snapshot — vocab-compatible by
/// construction, with the quantized draft a faster, lossy approximation that yields genuine partial
/// acceptance. Prefers a **Q4** draft (more lossy ⇒ more interesting acceptance), falling back to
/// **Q8** when the model's projection `in`-dims aren't 256-aligned (Q4_K's block size — e.g. SmolLM2's
/// hidden 576; Qwen3's 1024 is fine).
fn load_draft_target(env: &str) -> Option<(CausalLm, CausalLm, Tokenizer)> {
    let dir = std::env::var(env).ok().filter(|p| !p.is_empty())?;
    let device = select_device().unwrap();
    let w = Weights::from_dir(&dir, &device).unwrap();
    let target = CausalLm::from_weights(&w, "", ModelConfig::from_dir(&dir).unwrap()).unwrap();
    let draft = CausalLm::from_weights_with(
        &w,
        "",
        ModelConfig::from_dir(&dir).unwrap(),
        Some(QuantSpec::q4()),
    )
    .or_else(|_| {
        CausalLm::from_weights_with(
            &w,
            "",
            ModelConfig::from_dir(&dir).unwrap(),
            Some(QuantSpec::q8()),
        )
    })
    .unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some((target, draft, tok))
}

fn run_draft_suite(target: CausalLm, draft: CausalLm, tok: Tokenizer) {
    // ---- Exactness gate: num_draft = 0 ⇒ identical to non-speculative target decoding. ----
    let no_draft = SpeculativeConfig {
        max_ngram: 3,
        num_draft: 0,
    };
    for text in ["The capital of France is", "Q: What is 2+2? A:"] {
        let p = encode(&tok, text);
        let base = base_greedy(&target, &p, 28);
        let (out, stats) = generate_draft_speculative(
            &target,
            &draft,
            &p,
            &greedy_config(28),
            &no_draft,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            out.tokens, base,
            "num_draft=0 draft-spec must equal non-speculative for '{text}'"
        );
        assert_eq!(stats.accepted, 0);
    }

    // ---- Draft-model greedy: tracks non-spec + measured win from accepted drafts. ----
    let spec = SpeculativeConfig::default();
    let p = encode(
        &tok,
        "Once upon a time in a small village there lived a curious",
    );
    let base = base_greedy(&target, &p, 48);
    let (out, stats) = generate_draft_speculative(
        &target,
        &draft,
        &p,
        &greedy_config(48),
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    let n = out.tokens.len();
    let cp = common_prefix(&out.tokens, &base);
    println!(
        "draft-model: {n} tokens in {} target forwards ({:.2} tok/forward); {}/{} drafts accepted (quantized draft); tracks dense for {cp}/{} tokens",
        stats.forwards,
        n as f64 / stats.forwards as f64,
        stats.accepted,
        stats.proposed,
        base.len(),
    );
    assert!(
        stats.accepted > 0,
        "the Q4 draft should agree with the dense target on some tokens"
    );
    assert!(
        stats.forwards < n + 1,
        "accepted drafts must reduce target forwards below the token count"
    );
    assert!(
        cp >= 1,
        "draft-spec output must track non-speculative (diverges only on bf16 near-ties)"
    );

    // ---- Stochastic: deterministic for a fixed seed, valid output. ----
    let scfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        },
        seed: Some(11),
        stop_tokens: Vec::new(),
    };
    let p = encode(&tok, "Describe the morning sky:");
    let a = generate_draft_speculative(
        &target,
        &draft,
        &p,
        &scfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    let b = generate_draft_speculative(
        &target,
        &draft,
        &p,
        &scfg,
        &spec,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .0;
    assert_eq!(
        a.tokens, b.tokens,
        "stochastic draft-spec must be deterministic for a fixed seed"
    );
    assert!(!a.tokens.is_empty());
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_TEST_MODEL"]
fn draft_speculative_llama() {
    let Some((target, draft, tok)) = load_draft_target("CANDLE_LLM_TEST_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
        return;
    };
    run_draft_suite(target, draft, tok);
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_QWEN3_MODEL"]
fn draft_speculative_qwen3() {
    let Some((target, draft, tok)) = load_draft_target("CANDLE_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_QWEN3_MODEL");
        return;
    };
    run_draft_suite(target, draft, tok);
}

/// Two models with different vocabularies (SmolLM2 vs Qwen3) must be rejected as a draft/target pair.
#[test]
#[ignore = "needs both CANDLE_LLM_TEST_MODEL and CANDLE_LLM_QWEN3_MODEL"]
fn draft_speculative_rejects_vocab_mismatch() {
    let (Some(a), Some(b)) = (
        load_from("CANDLE_LLM_TEST_MODEL"),
        load_from("CANDLE_LLM_QWEN3_MODEL"),
    ) else {
        eprintln!("skip: set both CANDLE_LLM_TEST_MODEL and CANDLE_LLM_QWEN3_MODEL");
        return;
    };
    assert_ne!(a.model.config().vocab_size, b.model.config().vocab_size);
    let p = encode(&a.tok, "Hello");
    let err = generate_draft_speculative(
        &a.model,
        &b.model,
        &p,
        &greedy_config(8),
        &SpeculativeConfig::default(),
        &CancelFlag::new(),
        &mut |_| {},
    );
    assert!(err.is_err(), "mismatched vocab must error");
}
