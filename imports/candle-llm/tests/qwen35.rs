//! Real-weights end-to-end tests for Qwen3.6 (`qwen3_5`), the hybrid Gated-DeltaNet /
//! gated-full-attention text decoder (story sc-7632). The same tests cover both variants — point
//! `CANDLE_LLM_QWEN35_MODEL` at the 27B (dense) or the 35B-A3B (MoE) snapshot:
//!
//! ```text
//! CANDLE_LLM_QWEN35_MODEL=D:\repos\models\Qwen3.6-27B \
//!   cargo test --features cuda --test qwen35 -- --ignored --nocapture
//! ```
//!
//! These are the slice-3 acceptance gate: model-first resolution (the VLM-wrapped checkpoint routes
//! to the `candle-llama` **text** provider, not the LLaVA vision provider), dispatch (family
//! `qwen3_5`), coherent greedy text on real weights, the thinking / no-think split driven by the
//! model's own chat template, quantize-on-load, and the full `core-llm-testkit` conformance suite.

use candle_llm::provider::{can_load, PROVIDER_ID};
use candle_llm::LlamaProvider;
use core_llm::{
    load_for_model, Channel, LoadSpec, Message, Quantize, Sampling, StreamEvent, TextLlm,
    TextLlmOutput, TextLlmRequest, ThinkingMode,
};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};

fn model_dir() -> String {
    std::env::var("CANDLE_LLM_QWEN35_MODEL").expect("set CANDLE_LLM_QWEN35_MODEL")
}

fn req(prompt: &str, mode: ThinkingMode, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(prompt)],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        thinking: mode,
        ..Default::default()
    }
}

/// Run a request, reconstructing the per-channel text from the streamed deltas.
fn run(p: &dyn TextLlm, r: &TextLlmRequest) -> (TextLlmOutput, String, String) {
    let (mut think, mut content) = (String::new(), String::new());
    let out = p
        .generate(r, &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                match channel {
                    Channel::Thinking => think.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .expect("generate");
    (out, think, content)
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot via CANDLE_LLM_QWEN35_MODEL"]
fn qwen35_resolves_dispatch_and_coherent_text() {
    let spec = LoadSpec::dense(model_dir());

    // Model-first resolution (story 7406): the VLM-wrapped Qwen3.6 checkpoint carries `text_config` +
    // `vision_config`, so the LLaVA probe would normally claim it — but it is the hybrid Gated-DeltaNet
    // decoder, served as TEXT. The text provider's `can_load` accepts it (the LLaVA probe declines), so
    // `load_for_model` resolves uniquely to `candle-llama`.
    assert!(
        can_load(&spec),
        "candle-llama must accept the Qwen3.6 snapshot"
    );
    let p = load_for_model(&spec).expect("resolve + load provider by model");
    assert_eq!(
        p.descriptor().id,
        PROVIDER_ID,
        "resolved to the text provider"
    );
    assert_eq!(
        p.descriptor().family,
        "qwen3_5",
        "must dispatch to the qwen3_5 hybrid decoder"
    );
    assert!(p.descriptor().capabilities.max_context_tokens > 0);
    assert!(
        !p.descriptor().capabilities.supports_vision,
        "text-only here"
    );

    // Coherence gate: greedy, no-think → a direct factual answer. A wrong architecture (4-way split,
    // L2-norm, schedule, partial RoPE, EOS…) produces token soup, not "Paris".
    let (out, _think, content) = run(
        &*p,
        &req(
            "What is the capital of France? Answer with just the city name.",
            ThinkingMode::Disabled,
            24,
        ),
    );
    println!("\n=== qwen3.6 NO-THINK ===\n[answer] {:?}\n", out.text);
    assert!(!content.trim().is_empty(), "must produce a direct answer");
    assert!(
        content.to_lowercase().contains("paris"),
        "greedy answer should be coherent and name Paris, got: {content:?}"
    );
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot via CANDLE_LLM_QWEN35_MODEL"]
fn qwen35_thinking_and_nothink() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6");
    assert!(
        p.descriptor().capabilities.supports_thinking,
        "Qwen3.6's chat template gates enable_thinking → supports_thinking must be on"
    );
    for mode in [
        ThinkingMode::Auto,
        ThinkingMode::Enabled,
        ThinkingMode::Disabled,
    ] {
        p.validate(&req("hi", mode, 8))
            .unwrap_or_else(|e| panic!("validate {mode:?}: {e}"));
    }

    // Thinking: a <think>…</think> block is emitted and split into output.thinking; the answer excludes
    // the reasoning and the markers.
    let (out, think, content) = run(
        &p,
        &req("What is 2+2? Reply briefly.", ThinkingMode::Enabled, 512),
    );
    println!(
        "\n=== qwen3.6 THINK ===\n[reasoning]\n{think}\n[answer]\n{}\n",
        out.text
    );
    assert!(
        out.thinking
            .as_deref()
            .is_some_and(|t| !t.trim().is_empty()),
        "thinking run must produce a reasoning block"
    );
    assert!(
        !out.text.contains("<think>") && !out.text.contains("</think>"),
        "markers must be stripped from the answer: {:?}",
        out.text
    );
    assert_eq!(
        content, out.text,
        "content-channel deltas reconstruct output.text"
    );
    assert_eq!(think, out.thinking.clone().unwrap_or_default());

    // No-think: the empty <think></think> echo is injected, so the model answers directly.
    let (nout, nthink, ncontent) = run(
        &p,
        &req("What is 2+2? Reply briefly.", ThinkingMode::Disabled, 64),
    );
    println!("=== qwen3.6 NO-THINK ===\n[answer]\n{}\n", nout.text);
    assert!(
        nthink.is_empty(),
        "no-think must emit no Thinking-channel tokens"
    );
    assert!(
        nout.thinking.is_none(),
        "no-think output.thinking must be None"
    );
    assert!(
        !ncontent.trim().is_empty(),
        "no-think must produce a direct answer"
    );
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot via CANDLE_LLM_QWEN35_MODEL (Q8 quantize-on-load)"]
fn qwen35_quantize_on_load_q8() {
    let q8 = LlamaProvider::load(&LoadSpec {
        source: model_dir(),
        quantize: Some(Quantize::Q8),
    })
    .expect("load q8");
    assert!(q8.is_quantized(), "Q8 load must report quantized");
    let (_out, _think, content) = run(
        &q8,
        &req(
            "Name a primary color. One word.",
            ThinkingMode::Disabled,
            16,
        ),
    );
    println!("\n=== qwen3.6 Q8 ===\n[answer] {content:?}\n");
    assert!(
        !content.trim().is_empty(),
        "quantized model must still generate text"
    );
}

/// A conformance profile tuned for a strong 27B. The default `cheap()` profile ("Hello", 16 tokens,
/// temperature 1.0) is near-argmax for a model this capable, so two seeds produce identical output and
/// `check_seed_determinism` falsely reads the seed as ignored. An open-ended prompt, a longer budget,
/// and nucleus sampling give the (genuinely seed-driven) sampler room to diverge across seeds. (The
/// candle sampler is the same one Qwen3-0.6B passes `cheap()` with — see tests/conformance.rs.)
fn qwen35_profile() -> TextLlmProfile {
    let mut p = TextLlmProfile::cheap();
    p.prompt = "Write a few sentences of an imaginative story about a curious robot.".to_string();
    p.max_new_tokens = 64;
    p.determinism_sampling = Sampling {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context: 0,
    };
    p
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot via CANDLE_LLM_QWEN35_MODEL"]
fn qwen35_passes_core_llm_conformance() {
    let spec = LoadSpec::dense(model_dir());
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load qwen3.6 provider")),
        &qwen35_profile(),
    );
}
