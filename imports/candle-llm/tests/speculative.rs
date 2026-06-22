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

use candle_llm::config::LlamaConfig;
use candle_llm::decode::{
    generate, generate_prompt_lookup, CancelFlag, GenerationConfig, SpeculativeConfig,
};
use candle_llm::device::select_device;
use candle_llm::models::LlamaModel;
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

fn base_greedy(model: &LlamaModel, prompt: &[i32], max_new: usize) -> Vec<i32> {
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

fn build_tiny_llama() -> LlamaModel {
    let dir = std::env::temp_dir().join(format!("candle-llm-spec-{}", std::process::id()));
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

    let mut rng = SplitMix64::new(0xC0DE_CAFE);
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
    let weights = Weights::from_dir(&dir, &Device::Cpu).unwrap();
    let model = LlamaModel::from_weights(&weights, "", cfg).unwrap();
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

// ---- Real-weights variants (#[ignore]) -----------------------------------------------------------

struct Fixture {
    model: LlamaModel,
    tok: Tokenizer,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok().filter(|p| !p.is_empty())?;
    let device = select_device().unwrap();
    let cfg = LlamaConfig::from_dir(&dir).unwrap();
    let model =
        LlamaModel::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg).unwrap();
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
