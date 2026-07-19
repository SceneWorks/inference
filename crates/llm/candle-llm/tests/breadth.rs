//! Real-weights model-breadth tests (`#[ignore]` — need models on disk), story 7261.
//!
//! Each non-Llama architecture added to the `config.json` dispatch must stream coherent text through
//! the backend-neutral `core_llm::TextLlm` from a real HF snapshot. Point the per-family env var at a
//! snapshot dir and run (add `--features cuda` for the GPU path):
//!
//! ```text
//! CANDLE_LLM_PHI3_MODEL=/path/Phi-3-mini-4k-instruct \
//!   cargo test --features cuda --test breadth -- --ignored --nocapture
//! ```

use candle_llm::load_for_model;
use core_llm::{LoadSpec, Message, Sampling, StreamEvent, TextLlmRequest};

/// Load the snapshot at `$env` **by model** (story 7406: `load_for_model`, naming no provider id /
/// family / backend), check its reported family tag, and assert it streams coherent, word-bearing
/// text (the streamed deltas reconstructing the final output).
fn assert_streams_coherent(env: &str, family: &str) {
    let Some(dir) = std::env::var(env).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: set {env}");
        return;
    };
    let spec = LoadSpec::dense(dir);
    // The weightless probe must accept the snapshot, and model-first resolution must route it to the
    // single generic candle text provider purely by architecture — the family is only known
    // post-load, so this exercises the `can_load`-not-`descriptor.family` resolution the story exists
    // for (e.g. Gemma2/GLM4/DeepSeek behind `candle-llama`).
    assert!(
        candle_llm::provider::can_load(&spec),
        "{family}: can_load must accept the snapshot"
    );
    let provider = load_for_model(&spec).expect("resolve + load provider by model");
    assert_eq!(
        provider.descriptor().id,
        "candle-llama",
        "resolved provider id"
    );
    assert_eq!(provider.descriptor().family, family, "reported family tag");

    let req = TextLlmRequest {
        messages: vec![Message::user("The capital of France is")],
        sampling: Sampling::greedy(),
        max_new_tokens: 24,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { text, .. } = ev {
                streamed.push_str(&text);
            }
        })
        .expect("generate");

    println!("[{family}] {}", out.text.replace('\n', " "));
    assert!(!out.text.trim().is_empty(), "{family}: produced no text");
    assert_eq!(
        streamed, out.text,
        "{family}: streamed deltas must reconstruct the final text"
    );
    assert!(
        out.text.chars().any(|c| c.is_alphabetic()),
        "{family}: output should contain words, not just punctuation"
    );
}

/// Phi-3: the Llama decoder shape with a packed `qkv_proj` + `gate_up_proj` (split at load).
#[test]
#[ignore = "needs a Phi-3 snapshot via CANDLE_LLM_PHI3_MODEL"]
fn phi3_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_PHI3_MODEL", "phi3");
}

/// Qwen2-MoE: Qwen2 attention (q/k/v bias) + a sparse MoE FFN (router + top-k experts + shared).
#[test]
#[ignore = "needs a Qwen2-MoE snapshot via CANDLE_LLM_QWEN2MOE_MODEL"]
fn qwen2_moe_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_QWEN2MOE_MODEL", "qwen2_moe");
}

/// Gemma-2: `(1+weight)` norms, embedding ×√hidden, GeGLU, soft-capped attention + final logits,
/// 4-norm sandwich block.
#[test]
#[ignore = "needs a Gemma-2 snapshot via CANDLE_LLM_GEMMA2_MODEL"]
fn gemma2_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_GEMMA2_MODEL", "gemma2");
}

/// GLM-4: 4-norm sandwich (standard RMSNorm), q/k/v bias, packed gate_up, and partial + interleaved
/// RoPE.
#[test]
#[ignore = "needs a GLM-4 snapshot via CANDLE_LLM_GLM4_MODEL"]
fn glm4_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_GLM4_MODEL", "glm4");
}

/// DeepSeek-V2: Multi-head Latent Attention (low-rank KV path + decoupled YaRN RoPE) and a fine-
/// grained MoE FFN (many routed experts + shared experts, a leading dense layer). Verified on
/// `deepseek-ai/DeepSeek-V2-Lite-Chat` (15.7B, fits 96GB).
#[test]
#[ignore = "needs a DeepSeek-V2 snapshot via CANDLE_LLM_DEEPSEEK_MODEL"]
fn deepseek_v2_streams_coherent_text() {
    assert_streams_coherent("CANDLE_LLM_DEEPSEEK_MODEL", "deepseek_v2");
}
