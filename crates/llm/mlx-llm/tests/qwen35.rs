//! Real-weights end-to-end tests for Qwen3.6 (`qwen3_5` / `qwen3_5_moe`), the hybrid Gated-DeltaNet /
//! gated-full-attention decoder (stories sc-7627…sc-7630). The same tests cover both variants — point
//! `MLX_LLM_QWEN35_MODEL` at the 27B (dense) or the 35B-A3B (MoE) snapshot:
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B-or-35B-A3B \
//!   cargo test --test integration -- qwen35:: --ignored --nocapture
//! ```
//!
//! These are the acceptance gate for sc-7629 (27B dense) and sc-7630 (35B-A3B MoE): dispatch (family
//! `qwen3_5`), coherent greedy text on real weights, and the thinking / no-think split driven by the
//! model's own chat template.

use core_llm::{
    Channel, LoadSpec, Message, MtpMode, Quantize, Sampling, StreamEvent, TextLlm, TextLlmOutput,
    TextLlmRequest, ThinkingMode, ToolSpec,
};
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::LlamaProvider;
use mlx_rs::Array;

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

fn model_dir() -> String {
    std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL")
}

/// A tiny Qwen3.8-shaped snapshot using the exact frozen tokenizer/template. Set
/// `QWEN38_TOKENIZER_JSON` to the pinned tokenizer file; weights are synthetic and small except for
/// the shared 248,320-row vocabulary tables.
fn write_tiny_qwen38_snapshot() -> tempfile::TempDir {
    let tokenizer = std::env::var_os("QWEN38_TOKENIZER_JSON")
        .expect("set QWEN38_TOKENIZER_JSON to the frozen Qwen3.8 tokenizer.json");
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(tokenizer, dir.path().join("tokenizer.json")).unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/tokenizer_config.json"
        )),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/generation_config.json"
        )),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
          "architectures":["Qwen3_5ForConditionalGeneration"], "model_type":"qwen3_5",
          "text_config":{
            "model_type":"qwen3_5_text", "hidden_size":8, "num_hidden_layers":1,
            "intermediate_size":16, "num_attention_heads":2, "num_key_value_heads":1,
            "head_dim":4, "vocab_size":248320, "rms_norm_eps":0.000001,
            "rope_theta":10000000.0, "partial_rotary_factor":1.0,
            "max_position_embeddings":512, "tie_word_embeddings":false,
            "full_attention_interval":1, "linear_num_value_heads":2,
            "linear_num_key_heads":1, "linear_key_head_dim":4,
            "linear_value_head_dim":4, "linear_conv_kernel_dim":4,
            "mtp_num_hidden_layers":1, "mtp_use_dedicated_embeddings":false
          }
        }"#,
    )
    .unwrap();

    let z = |shape: &[i32]| Array::zeros::<f32>(shape).unwrap();
    let (h, vocab, inter, q, kv) = (8, 248_320, 16, 16, 4);
    let mut tensors: Vec<(String, Array)> = vec![
        (
            "model.language_model.embed_tokens.weight".into(),
            z(&[vocab, h]),
        ),
        ("model.language_model.norm.weight".into(), z(&[h])),
        ("lm_head.weight".into(), z(&[vocab, h])),
    ];
    let mut layer = |prefix: &str| {
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), z(&[h])),
            (format!("{prefix}.post_attention_layernorm.weight"), z(&[h])),
            (format!("{prefix}.self_attn.q_proj.weight"), z(&[q, h])),
            (format!("{prefix}.self_attn.k_proj.weight"), z(&[kv, h])),
            (format!("{prefix}.self_attn.v_proj.weight"), z(&[kv, h])),
            (format!("{prefix}.self_attn.o_proj.weight"), z(&[h, 8])),
            (format!("{prefix}.self_attn.q_norm.weight"), z(&[4])),
            (format!("{prefix}.self_attn.k_norm.weight"), z(&[4])),
            (format!("{prefix}.mlp.gate_proj.weight"), z(&[inter, h])),
            (format!("{prefix}.mlp.up_proj.weight"), z(&[inter, h])),
            (format!("{prefix}.mlp.down_proj.weight"), z(&[h, inter])),
        ]);
    };
    layer("model.language_model.layers.0");
    layer("mtp.layers.0");
    tensors.extend([
        ("mtp.fc.weight".into(), z(&[h, 2 * h])),
        ("mtp.pre_fc_norm_embedding.weight".into(), z(&[h])),
        ("mtp.pre_fc_norm_hidden.weight".into(), z(&[h])),
        ("mtp.norm.weight".into(), z(&[h])),
    ]);
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
    dir
}

#[test]
#[ignore = "requires frozen Qwen3.8 tokenizer via QWEN38_TOKENIZER_JSON"]
fn frozen_qwen38_tokenizer_runs_tiny_native_text_and_mtp() {
    let dir = write_tiny_qwen38_snapshot();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string())).unwrap();
    let mtp = provider.descriptor().capabilities.mtp.unwrap();
    assert_eq!(mtp.recommended_draft_tokens, 3);
    assert_eq!(mtp.max_draft_tokens, u32::MAX);

    let ar = provider
        .generate(
            &req("ordinary autoregressive route", ThinkingMode::Disabled, 1),
            &mut |_| {},
        )
        .unwrap();
    assert!(ar.mtp.is_none(), "MTP must remain opt-in by default");
    assert!(
        ar.timings.is_some(),
        "native AR must report measured phases"
    );

    let mut request = req("What is 2+2?", ThinkingMode::Disabled, 4);
    request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    let (output, thinking, content) = run(&provider, &request);
    assert_eq!(output.usage.generated_tokens, 4);
    assert!(thinking.is_empty());
    assert_eq!(content, output.text);
    let stats = output.mtp.expect("MTP stats");
    assert!(
        output.timings.is_some(),
        "native MTP must report measured phases"
    );
    assert!(stats.proposed_tokens > 0);
    assert!(stats.accepted_tokens <= stats.proposed_tokens);
    assert!(stats.target_forwards >= 2);

    // The same native provider route accepts the frozen template's tool prompt while MTP is active.
    let mut with_tool = req("weather in Paris?", ThinkingMode::Disabled, 2);
    with_tool.mtp = MtpMode::Auto;
    with_tool.tools = vec![ToolSpec::new(
        "get_weather",
        "Get the weather",
        serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}}),
    )];
    let out = provider.generate(&with_tool, &mut |_| {}).unwrap();
    assert!(out.mtp.is_some());

    let mut constrained = req("return json", ThinkingMode::Disabled, 2);
    constrained.mtp = MtpMode::Enabled { draft_tokens: 1 };
    constrained.constraint = Some(core_llm::Constraint::Json);
    let constrained_mtp = provider.generate(&constrained, &mut |_| {}).unwrap();
    assert!(
        constrained_mtp.mtp.is_some(),
        "explicit MTP must preserve native JSON-constrained generation"
    );

    constrained.mtp = MtpMode::Auto;
    let auto = provider.generate(&constrained, &mut |_| {}).unwrap();
    assert!(
        auto.mtp.is_some(),
        "Auto must retain MTP for native JSON-constrained generation"
    );
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_dispatch_and_coherent_text() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6");

    // Architecture dispatch: the hybrid decoder loads and reports itself as `qwen3_5`.
    assert_eq!(p.descriptor().id, PROVIDER_ID);
    assert_eq!(
        p.descriptor().family,
        "qwen3_5",
        "must dispatch to the qwen3_5 hybrid decoder"
    );
    assert!(p.descriptor().capabilities.max_context_tokens > 0);
    assert!(!p.is_quantized());

    // Coherence gate: greedy, no-think → a direct factual answer. A wrong architecture (split,
    // l2-norm, schedule, RoPE…) produces token soup, not "Paris".
    let (out, _think, content) = run(
        &p,
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
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
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

    // Thinking: a <think>…</think> block is emitted and split into output.thinking; the answer
    // excludes the reasoning and the markers.
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
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_quantize_on_load_q8() {
    let dir = model_dir();
    let q8 = LlamaProvider::load(&LoadSpec {
        source: dir,
        projector_source: None,
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
