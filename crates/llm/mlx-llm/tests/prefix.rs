//! Prefix-cache tests, story 7168: real-weights suites (`#[ignore]` — needs a model on disk) plus
//! a synthetic no-download regression test for the budget-finish off-by-one (sc-12455).
//!
//! Point `MLX_LLM_TEST_MODEL` (Llama-family) and/or `MLX_LLM_QWEN3_MODEL` (Qwen3) at a Hugging Face
//! snapshot and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M-Instruct \
//!   cargo test --test prefix -- --ignored --nocapture
//! ```
//!
//! ## What these prove
//! Shared-prefix KV reuse is **exact**, not approximate: [`generate_cached`] is token-for-token
//! identical to a cold [`generate`] of the same prompt, because a causal decoder's KV at position `i`
//! depends only on tokens `0..=i` — the reused span is the same values the cold path would compute,
//! attended over by the same kernels. The reuse is then *measured*: a second request sharing a fixed
//! system-prompt prefix runs only its suffix through the model (the shared span's prefill is skipped),
//! and that saved-token count is asserted against [`PrefixStats`].

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, generate_cached, CancelFlag, GenerationConfig, PrefixCache};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some(Fixture { model, tok })
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true).unwrap().into_iter().map(|id| id as i32).collect()
}

/// Encode without the BOS so a fragment can be appended after a prefix without a spurious token.
fn encode_no_bos(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, false).unwrap().into_iter().map(|id| id as i32).collect()
}

fn config(_fx: &Fixture, max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(), // greedy ⇒ deterministic
        seed: Some(0),
        // No stop tokens: bit-exactness (cold == cached) is what matters, and running the full
        // budget keeps the test model-agnostic (some models greedily emit EOS first on a raw,
        // non-chat-templated prompt, which would otherwise produce an empty baseline).
        stop_tokens: Vec::new(),
    }
}

fn cold(fx: &Fixture, prompt: &[i32], max_new: usize) -> Vec<i32> {
    generate(&fx.model, prompt, &config(fx, max_new), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens
}

fn cached(fx: &Fixture, prompt: &[i32], max_new: usize, pc: &mut PrefixCache) -> Vec<i32> {
    generate_cached(&fx.model, prompt, &config(fx, max_new), &CancelFlag::new(), &mut |_| {}, pc)
        .unwrap()
        .tokens
}

/// Build a [`system prefix tokens, suffix tokens]` prompt by concatenating token ids (the BOS lives
/// in the system span only), so the shared prefix is *exactly* the system span — no tokenizer
/// boundary merging between the two requests.
fn prompt(sys: &[i32], suffix: &[i32]) -> Vec<i32> {
    let mut p = sys.to_vec();
    p.extend_from_slice(suffix);
    p
}

/// The body shared by the per-model tests.
fn run_suite(fx: Fixture) {
    let sys = encode(&fx.tok, "You are a helpful assistant. Answer concisely and accurately.\n\n");
    // Two questions that diverge on their very first token after the shared system prefix.
    let q1 = encode_no_bos(&fx.tok, "What is the capital of France?");
    let q2 = encode_no_bos(&fx.tok, "Name three primary colors.");
    assert_ne!(q1.first(), q2.first(), "questions must diverge immediately after the prefix");

    let p1 = prompt(&sys, &q1);
    let p2 = prompt(&sys, &q2);

    // Cold baselines (plain single-sequence loop).
    let base1 = cold(&fx, &p1, 24);
    let base2 = cold(&fx, &p2, 24);
    assert!(!base1.is_empty() && !base2.is_empty(), "baselines should generate");

    let mut pc = PrefixCache::new(16);

    // First cached request: cold (nothing stored yet) but must equal the baseline, and it stores
    // p1 + base1 for reuse.
    let out1 = cached(&fx, &p1, 24, &mut pc);
    assert_eq!(out1, base1, "first cached run must equal the cold baseline (bit-exact)");
    let s = pc.stats();
    assert_eq!(s.hits, 0, "nothing to reuse on the first request");
    assert_eq!(s.computed_prefill_tokens, p1.len());

    // Second cached request shares exactly the system prefix: KV for `sys` is reused, only `q2` is
    // prefilled — and the output is still bit-exact vs the cold baseline.
    let out2 = cached(&fx, &p2, 24, &mut pc);
    assert_eq!(out2, base2, "shared-prefix cached run must equal the cold baseline (bit-exact)");
    let s = pc.stats();
    assert_eq!(s.hits, 1, "second request hits the cached system prefix");
    assert_eq!(s.reused_prefix_tokens, sys.len(), "the whole system span is reused");
    // Total computed prefill = p1 (cold) + q2 only (suffix after the shared sys).
    assert_eq!(s.computed_prefill_tokens, p1.len() + q2.len());
    println!(
        "reused {} of {} prompt tokens on the shared-prefix request ({} computed)",
        sys.len(),
        p2.len(),
        p2.len() - sys.len()
    );

    // Re-issuing p1 verbatim now matches the *whole* stored prompt; reuse clamps to recompute only
    // the final token, and the output is unchanged.
    let out1_again = cached(&fx, &p1, 24, &mut pc);
    assert_eq!(out1_again, base1, "whole-prompt reuse must still be bit-exact");
    let s = pc.stats();
    assert_eq!(s.hits, 2);
    // This hit reused p1.len() - 1 positions (recomputing only the last prompt token).
    assert_eq!(s.reused_prefix_tokens, sys.len() + (p1.len() - 1));
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn prefix_reuse_is_bit_exact_llama() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    run_suite(fx);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL"]
fn prefix_reuse_is_bit_exact_qwen3() {
    let Some(fx) = load_from("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    run_suite(fx);
}

// ---- Synthetic no-download regression (sc-12455) -------------------------------------------------
//
// Same tiny deterministic Llama as `tests/streaming.rs` — runs in CI with no weights.

use std::collections::HashMap;

use mlx_rs::Array;

use mlx_llm::decode::FinishReason;
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};

fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        architecture: mlx_llm::config::Architecture::Llama,
        max_position_embeddings: 0,
        quantization: None,
        moe: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        query_pre_attn_scalar: None,
        partial_rotary_factor: 1.0,
        mla: None,
        yarn: None,
        mrope_section: None,
    }
}

fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

fn tiny_model(cfg: &ModelConfig) -> CausalLm {
    let mut rng = SplitMix64::new(0xC0FFEE);
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qd = cfg.num_heads * cfg.head_dim;
    let kvd = cfg.num_kv_heads * cfg.head_dim;

    let mut m: HashMap<String, Array> = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng));
    m.insert("model.norm.weight".into(), Array::ones::<f32>(&[h]).unwrap());
    m.insert("lm_head.weight".into(), randn(&[v, h], &mut rng));
    for i in 0..cfg.num_layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), Array::ones::<f32>(&[h]).unwrap());
        m.insert(
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng));
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng));
        m.insert(p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng));
    }

    CausalLm::from_weights(&Weights::from_map(m), "", cfg.clone()).unwrap()
}

fn greedy(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(), // greedy ⇒ deterministic
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

/// Regression (sc-12455): on a `MaxTokens` finish `decode_loop` breaks *before* feeding the last
/// generated token's KV, so the cache holds `prompt + n - 1` positions while the stored index entry
/// used to claim `prompt + n` tokens. A follow-up prompt that **extends** the stored sequence (the
/// module's multi-turn use case) then matched one position past the stored tensors' sequence dim.
/// On MLX this was *worse* than an error: `slice_layers` used an unchecked `take_axis` gather, so
/// the seeded KV was silent garbage — the bit-exact assertion against the cold baseline below is
/// what catches it.
#[test]
fn budget_finished_entry_supports_extension() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let prompt: Vec<i32> = vec![1, 2, 3, 4, 5];
    let max_new = 6;

    let mut pc = PrefixCache::new(16);
    let out1 = generate_cached(
        &model,
        &prompt,
        &greedy(max_new),
        &CancelFlag::new(),
        &mut |_| {},
        &mut pc,
    )
    .unwrap();
    assert_eq!(
        out1.finish_reason,
        FinishReason::MaxTokens,
        "the priming run must finish on the token budget"
    );
    assert_eq!(out1.tokens.len(), max_new);

    // Multi-turn continuation: previous prompt + everything generated + the next user turn.
    let mut extended = prompt.clone();
    extended.extend_from_slice(&out1.tokens);
    extended.extend_from_slice(&[9, 8, 7]);

    let base = generate(
        &model,
        &extended,
        &greedy(4),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap()
    .tokens;
    let out2 = generate_cached(
        &model,
        &extended,
        &greedy(4),
        &CancelFlag::new(),
        &mut |_| {},
        &mut pc,
    )
    .unwrap()
    .tokens;
    assert_eq!(
        out2, base,
        "extending a budget-finished entry must reuse the prefix bit-exactly (garbage KV from an \
         out-of-range gather would diverge here)"
    );
    let s = pc.stats();
    assert_eq!(s.hits, 1, "the extension must hit the stored entry");
    // The budget finish never fed the last generated token's KV, so the reusable span is exactly
    // one short of `prompt + generated`.
    assert_eq!(s.reused_prefix_tokens, prompt.len() + max_new - 1);
    assert_eq!(
        s.computed_prefill_tokens,
        prompt.len() + (extended.len() - (prompt.len() + max_new - 1))
    );
}
