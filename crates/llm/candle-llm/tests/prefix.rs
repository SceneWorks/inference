//! Shared-prefix KV-reuse tests (story 7256).
//!
//! ## What these prove
//! Reuse is **exact**, not approximate: [`generate_cached`] is token-for-token identical to a cold
//! [`generate`] of the same prompt, because a causal decoder's KV at position `i` depends only on
//! tokens `0..=i` — the reused span is the same values the cold path would compute, attended over by
//! the same kernels (the bottom-right causal mask already aligns a suffix prefill against the seeded
//! keys). The reuse is then *measured*: a second request sharing a fixed system-prompt prefix runs
//! only its suffix through the model, and that saved-token count is asserted against [`PrefixStats`].
//!
//! The headline [`prefix_reuse_is_bit_exact_cpu`] test runs on a tiny synthetic model on CPU with no
//! download — so it guards the wiring in CI and is genuinely *bit-exact*: `generate_cached` is
//! token-for-token identical to a cold `generate` (CPU f32 is order-stable). The `#[ignore]`d
//! real-weights tests check the same mechanic on a GPU snapshot, where the reuse is only *near*-exact:
//! the reused KV is computed in a different-length prefill than the cold path's, and a bf16 GEMM is
//! not perfectly invariant to its `M` dimension, so first-token logits drift at sub-ULP (the same
//! batch-invariance limitation the batched decode documents). They therefore assert the first-token
//! logits agree within a small tolerance, the greedy first token is identical, and the [`PrefixStats`]
//! reuse accounting (derived from token ids, not tensors — exactly reproducible) is correct.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use core_llm::Tokenizer;

use candle_llm::config::ModelConfig;
use candle_llm::decode::{generate, generate_cached, CancelFlag, GenerationConfig, PrefixCache};
use candle_llm::device::select_device;
use candle_llm::models::CausalLm;
use candle_llm::primitives::sampler::SamplingParams;
use candle_llm::primitives::{SplitMix64, TokenRng, Weights};

// ---- Shared helpers ------------------------------------------------------------------------------

/// Greedy (temperature 0 ⇒ deterministic), fixed seed, no stop tokens: bit-exactness of cold vs
/// cached is what matters, and running the full budget keeps the assertion content-independent.
fn config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

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

fn cached(model: &CausalLm, prompt: &[i32], max_new: usize, pc: &mut PrefixCache) -> Vec<i32> {
    generate_cached(
        model,
        prompt,
        &config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
        pc,
    )
    .unwrap()
    .tokens
}

/// `[sys ‖ suffix]` token ids concatenated, so the shared prefix is *exactly* the `sys` span.
fn prompt(sys: &[i32], suffix: &[i32]) -> Vec<i32> {
    let mut p = sys.to_vec();
    p.extend_from_slice(suffix);
    p
}

/// The body shared by the synthetic and real-weights runs: cold baselines, then three cached calls
/// exercising a cold miss, a shared-prefix hit, and a whole-prompt (clamped) hit — each bit-exact vs
/// its baseline, with [`PrefixStats`] asserting exactly which spans were reused.
fn run_suite(model: &CausalLm, sys: &[i32], q1: &[i32], q2: &[i32], max_new: usize) {
    assert_ne!(
        q1.first(),
        q2.first(),
        "questions must diverge immediately after the shared prefix"
    );
    let p1 = prompt(sys, q1);
    let p2 = prompt(sys, q2);

    // Cold baselines (the plain single-sequence loop).
    let base1 = cold(model, &p1, max_new);
    let base2 = cold(model, &p2, max_new);
    assert!(
        !base1.is_empty() && !base2.is_empty(),
        "baselines should generate"
    );

    let mut pc = PrefixCache::new(16);

    // First cached request: a cold miss (nothing stored yet), but it must equal the baseline, and it
    // stores `p1 + base1` for reuse.
    let out1 = cached(model, &p1, max_new, &mut pc);
    assert_eq!(
        out1, base1,
        "first cached run must equal the cold baseline (bit-exact)"
    );
    let s = pc.stats();
    assert_eq!(s.lookups, 1);
    assert_eq!(s.hits, 0, "nothing to reuse on the first request");
    assert_eq!(s.reused_prefix_tokens, 0);
    assert_eq!(s.computed_prefill_tokens, p1.len());

    // Second cached request shares exactly the `sys` prefix: its KV is reused, only `q2` is prefilled
    // — and the output is still bit-exact vs the cold baseline.
    let out2 = cached(model, &p2, max_new, &mut pc);
    assert_eq!(
        out2, base2,
        "shared-prefix cached run must equal the cold baseline (bit-exact)"
    );
    let s = pc.stats();
    assert_eq!(s.hits, 1, "second request hits the cached system prefix");
    assert_eq!(
        s.reused_prefix_tokens,
        sys.len(),
        "the whole system span is reused"
    );
    // Total computed prefill = p1 (cold) + q2 only (the suffix after the shared `sys`).
    assert_eq!(s.computed_prefill_tokens, p1.len() + q2.len());

    // Re-issuing p1 verbatim now matches the *whole* stored prompt; reuse clamps to recompute only
    // the final token, and the output is unchanged.
    let out1_again = cached(model, &p1, max_new, &mut pc);
    assert_eq!(
        out1_again, base1,
        "whole-prompt reuse must still be bit-exact"
    );
    let s = pc.stats();
    assert_eq!(s.hits, 2);
    // This hit reused p1.len() - 1 positions (recomputing only the last prompt token).
    assert_eq!(s.reused_prefix_tokens, sys.len() + (p1.len() - 1));
}

// ---- Synthetic CPU model (no download) -----------------------------------------------------------

const VOCAB: usize = 48;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2; // GQA (groups = 2), to exercise repeat_kv on the reuse path
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
fn build_tiny_llama() -> CausalLm {
    let dir = std::env::temp_dir().join(format!("candle-llm-prefix-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let config = format!(
        r#"{{
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": {HIDDEN}, "intermediate_size": {INTER}, "num_hidden_layers": 2,
            "num_attention_heads": {NUM_HEADS}, "num_key_value_heads": {NUM_KV_HEADS},
            "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "eos_token_id": 0
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();

    let mut rng = SplitMix64::new(0x9E37_79B9);
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
fn prefix_reuse_is_bit_exact_cpu() {
    let model = build_tiny_llama();
    // A shared "system" prefix, then two suffixes that diverge on their first token.
    let sys: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let q1: Vec<i32> = vec![10, 11, 12];
    let q2: Vec<i32> = vec![20, 21, 22];
    run_suite(&model, &sys, &q1, &q2, 12);
}

/// Regression (sc-12455): on a `MaxTokens` finish `decode_loop` breaks *before* feeding the last
/// generated token's KV, so the cache holds `prompt + n - 1` positions while the stored index entry
/// used to claim `prompt + n` tokens. A follow-up prompt that **extends** the stored sequence (the
/// module's multi-turn use case) then matched one position past the stored tensors' sequence dim and
/// the whole request failed with a bounds error. The `.min(prompt_len - 1)` clamp only bounds by the
/// *query* length, which is a no-op for an extending prompt — re-issuing the same prompt (what the
/// other tests do) masks the bug.
#[test]
fn budget_finished_entry_supports_extension() {
    let model = build_tiny_llama();
    let prompt: Vec<i32> = vec![1, 2, 3, 4, 5];
    let max_new = 6;

    let mut pc = PrefixCache::new(16);
    let out1 = generate_cached(
        &model,
        &prompt,
        &config(max_new),
        &CancelFlag::new(),
        &mut |_| {},
        &mut pc,
    )
    .unwrap();
    assert_eq!(
        out1.finish_reason,
        candle_llm::decode::FinishReason::MaxTokens,
        "the priming run must finish on the token budget"
    );
    assert_eq!(out1.tokens.len(), max_new);

    // Multi-turn continuation: previous prompt + everything generated + the next user turn.
    let mut extended = prompt.clone();
    extended.extend_from_slice(&out1.tokens);
    extended.extend_from_slice(&[9, 8, 7]);

    let base = cold(&model, &extended, 4);
    let out2 = cached(&model, &extended, 4, &mut pc);
    assert_eq!(
        out2, base,
        "extending a budget-finished entry must reuse the prefix bit-exactly"
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

// ---- Real-weights variants (#[ignore]) -----------------------------------------------------------
//
// On a GPU the reuse is numerically *near*-exact rather than bit-exact: the reused `sys` KV is
// computed inside a different-length prefill than the cold path's, and a bf16 GEMM is not perfectly
// invariant to its `M` dimension, so the first-token logits differ at sub-ULP — occasionally enough
// to flip a greedy *near-tie* a few tokens in (the same batch-invariance limitation documented for
// the batched decode). So these tests assert the property the GPU *does* guarantee: the first-token
// logits match within a tiny tolerance, and the [`PrefixStats`] reuse accounting (which is computed
// from token ids, not tensors, hence exactly reproducible) is correct. Full token-for-token bit
// exactness is proven on CPU by [`prefix_reuse_is_bit_exact_cpu`].

use candle_core::DType;

use candle_llm::primitives::{input_ids, KvCache};

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

fn encode(tok: &Tokenizer, text: &str, bos: bool) -> Vec<i32> {
    tok.encode(text, bos)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

/// First-position logits `[1, vocab]` (f32) for a cold prefill of `prompt`.
fn cold_logits(model: &CausalLm, prompt: &[i32]) -> Tensor {
    let mut cache = model.new_cache();
    let ids = input_ids(prompt, model.device()).unwrap();
    model
        .decode_logits(&ids, &mut cache, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
}

/// First-position logits `[1, vocab]` (f32) for the reuse path: prefill `sys`, then prefill `suffix`
/// at offset `sys.len()` over the cached `sys` KV — the exact mechanic [`generate_cached`] runs on a
/// shared-prefix hit.
fn reused_logits(model: &CausalLm, sys: &[i32], suffix: &[i32]) -> Tensor {
    let mut cache = model.new_cache();
    let sys_ids = input_ids(sys, model.device()).unwrap();
    model.decode_logits(&sys_ids, &mut cache, 0).unwrap();
    assert_eq!(cache.offset(), sys.len() as i32, "sys KV seeded");
    let suffix_ids = input_ids(suffix, model.device()).unwrap();
    model
        .decode_logits(&suffix_ids, &mut cache, sys.len() as i32)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
}

fn argmax(logits: &Tensor) -> i32 {
    logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn run_real(fx: Fixture) {
    // The BOS lives in the `sys` span only, so the shared prefix is exactly the system tokens — no
    // tokenizer boundary merging between the two requests.
    let sys = encode(
        &fx.tok,
        "You are a helpful assistant. Answer concisely and accurately.\n\n",
        true,
    );
    let q2 = encode(&fx.tok, "Name three primary colors.", false);
    let p2 = prompt(&sys, &q2);

    // First-token logits: cold (sys+q2 in one prefill) vs reuse (sys cached, then q2). They agree to
    // within a small bf16 tolerance, and the greedy first token is identical.
    let cold = cold_logits(&fx.model, &p2);
    let reused = reused_logits(&fx.model, &sys, &q2);
    let diff = max_abs_diff(&cold, &reused);
    println!(
        "first-token logits max|Δ| = {diff:.3e} over {} shared-prefix tokens ({} suffix computed)",
        sys.len(),
        q2.len()
    );
    // Observed ~0.3–0.8 on SmolLM2 / Qwen3 (bf16, ~30 layers): the sub-ULP GEMM drift from reusing
    // KV computed in a different-length prefill. A real offset/mask bug would diverge by 10s–100s, so
    // a 2.0 ceiling guards the wiring while tolerating bf16 non-invariance. (Bit-exact on CPU above.)
    assert!(
        diff <= 2.0,
        "reused first-token logits should match cold within bf16 tolerance, got max|Δ| = {diff}"
    );
    assert_eq!(
        argmax(&cold),
        argmax(&reused),
        "reuse must pick the same greedy first token as the cold path"
    );

    // Reuse accounting through the public path: prime with p1, then a shared-prefix request reuses
    // exactly the `sys` span (this is exact — derived from token ids, not tensors).
    let q1 = encode(&fx.tok, "What is the capital of France?", false);
    let p1 = prompt(&sys, &q1);
    let mut pc = PrefixCache::new(16);
    cached(&fx.model, &p1, 8, &mut pc);
    cached(&fx.model, &p2, 8, &mut pc);
    let s = pc.stats();
    assert_eq!(
        s.hits, 1,
        "the shared-prefix request hits the cached system prefix"
    );
    assert_eq!(
        s.reused_prefix_tokens,
        sys.len(),
        "the whole system span is reused"
    );
    assert_eq!(s.computed_prefill_tokens, p1.len() + q2.len());
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_TEST_MODEL"]
fn prefix_reuse_real_weights_llama() {
    let Some(fx) = load_from("CANDLE_LLM_TEST_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_TEST_MODEL");
        return;
    };
    run_real(fx);
}

#[test]
#[ignore = "needs a real snapshot via CANDLE_LLM_QWEN3_MODEL"]
fn prefix_reuse_real_weights_qwen3() {
    let Some(fx) = load_from("CANDLE_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set CANDLE_LLM_QWEN3_MODEL");
        return;
    };
    run_real(fx);
}
