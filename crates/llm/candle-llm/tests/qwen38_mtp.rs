//! Frozen Qwen3.8 tokenizer/template and native Candle MTP provider acceptance.
//!
//! The tokenizer is intentionally external: set `QWEN38_TOKENIZER_JSON` to the tokenizer pinned by
//! `docs/reference/qwen38/tokenizer_oracle.json`. The checked-in template and exact prompt/token-id
//! oracle remain immutable and reviewable; the executable model is a small CPU fixture with the
//! released 248,320-token vocabulary and exact 15-key MTP layout.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_llm::decode::DecodePath;
use candle_llm::primitives::KvCacheKind;
use candle_llm::LlamaProvider;
use core_llm::{
    Channel, ChatTemplate, FinishReason, JinjaChatTemplate, LoadSpec, Message, MtpMode,
    ProposerKind, ReasoningEffort, RenderOptions, Sampling, StreamEvent, TextLlm, TextLlmRequest,
    ThinkingMode, Tokenizer, ToolSpec,
};
use serde_json::Value;

const VOCAB: usize = 248_320;
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 16;
const EOS: usize = 248_046;

fn frozen_tokenizer() -> std::path::PathBuf {
    std::env::var_os("QWEN38_TOKENIZER_JSON")
        .map(Into::into)
        .expect("QWEN38_TOKENIZER_JSON must point to the frozen tokenizer.json")
}

fn oracle() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/reference/qwen38/tokenizer_oracle.json"
    )))
    .unwrap()
}

fn weather_tool() -> ToolSpec {
    ToolSpec::new(
        "get_weather",
        "Get the weather",
        serde_json::json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }),
    )
}

fn assert_oracle_case(
    name: &str,
    template: &JinjaChatTemplate,
    tokenizer: &Tokenizer,
    messages: &[Message],
    options: RenderOptions<'_>,
) {
    let oracle = oracle();
    assert_eq!(
        oracle["source_revision"],
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    assert_eq!(
        oracle["tokenizer_sha256"],
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    );
    let case = &oracle["cases"][name];
    let rendered = template.render_with(messages, &options).unwrap();
    assert_eq!(
        rendered,
        case["text"].as_str().unwrap(),
        "prompt case {name}"
    );
    let expected = case["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    assert_eq!(
        tokenizer.encode(&rendered, false).unwrap(),
        expected,
        "token ids for prompt case {name}"
    );
}

#[test]
#[ignore = "requires QWEN38_TOKENIZER_JSON from the frozen Qwen3.8 snapshot"]
fn frozen_template_and_tokenizer_match_all_request_controls() {
    let tokenizer_path = frozen_tokenizer();
    let template = JinjaChatTemplate::from_tokenizer_config_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/reference/qwen38/tokenizer_config.json"
    ))
    .unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let arithmetic = [Message::user("What is 2+2?")];

    assert_oracle_case(
        "default_xhigh",
        &template,
        &tokenizer,
        &arithmetic,
        RenderOptions::generation(),
    );
    assert_oracle_case(
        "low",
        &template,
        &tokenizer,
        &arithmetic,
        RenderOptions::generation().with_reasoning_effort(Some(ReasoningEffort::Low)),
    );
    assert_oracle_case(
        "disabled",
        &template,
        &tokenizer,
        &arithmetic,
        RenderOptions::generation().with_enable_thinking(Some(false)),
    );

    let tools = [weather_tool()];
    assert_oracle_case(
        "tools_low",
        &template,
        &tokenizer,
        &[Message::user("weather in Paris?")],
        RenderOptions::generation()
            .with_reasoning_effort(Some(ReasoningEffort::Low))
            .with_tools(&tools),
    );

    let history = [
        Message::user("What is 2+2?"),
        Message::assistant("Four.").with_thinking("Add two and two."),
        Message::user("And 3+3?"),
    ];
    assert_oracle_case(
        "history_stripped",
        &template,
        &tokenizer,
        &history,
        RenderOptions::generation().with_preserve_thinking(Some(false)),
    );
}

fn zeros(map: &mut HashMap<String, Tensor>, key: impl Into<String>, dims: (usize, usize)) {
    map.insert(
        key.into(),
        Tensor::zeros(dims, candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
}

fn zero_vector(map: &mut HashMap<String, Tensor>, key: impl Into<String>, len: usize) {
    map.insert(
        key.into(),
        Tensor::zeros(len, candle_core::DType::F32, &Device::Cpu).unwrap(),
    );
}

/// Write a tiny full-attention Qwen3.8 target plus the released one-layer MTP tensor layout. Every
/// target embedding points along dimension zero. The LM head selects either ordinary token 1 or the
/// frozen EOS token, while the zero MTP fusion predicts token 0 and exercises target rejection.
fn write_snapshot(tokenizer_path: &std::path::Path, stop_first: bool) -> tempfile::TempDir {
    write_snapshot_with(tokenizer_path, stop_first, true)
}

/// [`write_snapshot`] with or without the MTP head (`with_mtp`): a checkpoint without one
/// advertises no MTP capability, so `MtpMode::Auto` must decode normally and say `proposer=none`.
fn write_snapshot_with(
    tokenizer_path: &std::path::Path,
    stop_first: bool,
    with_mtp: bool,
) -> tempfile::TempDir {
    let guard = tempfile::Builder::new()
        .prefix("candle-qwen38-mtp-")
        .tempdir()
        .unwrap();
    let dir = guard.path();
    std::fs::copy(tokenizer_path, dir.join("tokenizer.json")).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/tokenizer_config.json"
        )),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/generation_config.json"
        )),
    )
    .unwrap();
    std::fs::write(
        dir.join("config.json"),
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
            "mtp_num_hidden_layers":MTP_LAYERS, "mtp_use_dedicated_embeddings":false
          }
        }"#
        .replace("MTP_LAYERS", if with_mtp { "1" } else { "0" }),
    )
    .unwrap();

    let mut tensors = HashMap::new();
    let mut embeddings = vec![0.0f32; VOCAB * HIDDEN];
    for row in embeddings.chunks_exact_mut(HIDDEN) {
        row[0] = 1.0;
    }
    tensors.insert(
        "model.language_model.embed_tokens.weight".into(),
        Tensor::from_vec(embeddings, (VOCAB, HIDDEN), &Device::Cpu).unwrap(),
    );
    zero_vector(&mut tensors, "model.language_model.norm.weight", HIDDEN);
    let mut head = vec![0.0f32; VOCAB * HIDDEN];
    head[if stop_first { EOS } else { 1 } * HIDDEN] = 2.0;
    tensors.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(head, (VOCAB, HIDDEN), &Device::Cpu).unwrap(),
    );

    let mut add_layer = |prefix: &str| {
        zero_vector(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            HIDDEN,
        );
        zero_vector(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            HIDDEN,
        );
        zeros(
            &mut tensors,
            format!("{prefix}.self_attn.q_proj.weight"),
            (16, HIDDEN),
        );
        zeros(
            &mut tensors,
            format!("{prefix}.self_attn.k_proj.weight"),
            (4, HIDDEN),
        );
        zeros(
            &mut tensors,
            format!("{prefix}.self_attn.v_proj.weight"),
            (4, HIDDEN),
        );
        zeros(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            (HIDDEN, 8),
        );
        zero_vector(&mut tensors, format!("{prefix}.self_attn.q_norm.weight"), 4);
        zero_vector(&mut tensors, format!("{prefix}.self_attn.k_norm.weight"), 4);
        zeros(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            (INTERMEDIATE, HIDDEN),
        );
        zeros(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            (INTERMEDIATE, HIDDEN),
        );
        zeros(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            (HIDDEN, INTERMEDIATE),
        );
    };
    add_layer("model.language_model.layers.0");
    if with_mtp {
        add_layer("mtp.layers.0");
        zeros(&mut tensors, "mtp.fc.weight", (HIDDEN, HIDDEN * 2));
        zero_vector(&mut tensors, "mtp.pre_fc_norm_embedding.weight", HIDDEN);
        zero_vector(&mut tensors, "mtp.pre_fc_norm_hidden.weight", HIDDEN);
        zero_vector(&mut tensors, "mtp.norm.weight", HIDDEN);
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
    guard
}

fn request(prompt: &str, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(prompt)],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(7),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

#[test]
#[ignore = "requires QWEN38_TOKENIZER_JSON from the frozen Qwen3.8 snapshot"]
fn configured_mtp_rejects_every_partial_tensor_set_and_disabled_config_contradiction() {
    let tokenizer_path = frozen_tokenizer();
    let required = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ];
    for missing in required {
        let snapshot = write_snapshot(&tokenizer_path, false);
        let model_path = snapshot.path().join("model.safetensors");
        let mut tensors = candle_core::safetensors::load(&model_path, &Device::Cpu).unwrap();
        tensors.remove(missing).expect("required fixture tensor");
        candle_core::safetensors::save(&tensors, &model_path).unwrap();
        let error = LlamaProvider::load(&LoadSpec::dense(snapshot.path().display().to_string()))
            .err()
            .unwrap_or_else(|| panic!("provider accepted configured MTP without {missing}"));
        assert!(error.to_string().contains(missing), "{missing}: {error}");
    }

    let contradictory = write_snapshot(&tokenizer_path, false);
    let config_path = contradictory.path().join("config.json");
    let config = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace("\"mtp_num_hidden_layers\":1", "\"mtp_num_hidden_layers\":0");
    std::fs::write(config_path, config).unwrap();
    let error = LlamaProvider::load(&LoadSpec::dense(contradictory.path().display().to_string()))
        .err()
        .expect("MTP tensors with disabled config must fail");
    assert!(error.to_string().contains("config disables MTP"), "{error}");
}

#[test]
#[ignore = "requires QWEN38_TOKENIZER_JSON from the frozen Qwen3.8 snapshot"]
fn frozen_qwen38_provider_executes_ar_mtp_tools_and_stops() {
    let tokenizer_path = frozen_tokenizer();
    let snapshot = write_snapshot(&tokenizer_path, false);
    let provider = LlamaProvider::load(&LoadSpec::dense(snapshot.path().display().to_string()))
        .expect("load tiny Qwen3.8 Candle provider");
    let capabilities = &provider.descriptor().capabilities;
    assert!(capabilities.supports_thinking);
    assert!(capabilities.supports_reasoning_effort);
    assert!(capabilities.supports_preserve_thinking);
    assert!(capabilities.supports_tools);
    assert_eq!(
        capabilities.mtp,
        Some(core_llm::MtpCapabilities {
            max_draft_tokens: u32::MAX,
            recommended_draft_tokens: 3,
        })
    );

    let ar = provider
        .generate(&request("ordinary autoregressive route", 2), &mut |_| {})
        .unwrap();
    assert!(ar.mtp.is_none(), "MTP remains opt-in by default");
    assert!(
        ar.timings.is_some(),
        "native Qwen AR reports synchronized phase timings"
    );
    assert_eq!(ar.usage.generated_tokens, 2);
    let ar_record = provider.last_decode_record().unwrap();
    assert_eq!(ar_record.path, DecodePath::Reference);
    assert_eq!(ar_record.proposer, ProposerKind::None);

    let mut mtp_request = request("What is 2+2?", 4);
    mtp_request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    let (mut thinking, mut content) = (String::new(), String::new());
    let mtp = provider
        .generate(&mtp_request, &mut |event| {
            if let StreamEvent::Token { text, channel, .. } = event {
                match channel {
                    Channel::Thinking => thinking.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .unwrap();
    assert!(
        mtp.timings.is_some(),
        "native Qwen MTP reports synchronized phase timings"
    );
    assert!(thinking.is_empty());
    assert_eq!(content, mtp.text);
    assert_eq!(mtp.usage.generated_tokens, 4);
    let stats = mtp.mtp.expect("native MTP stats");
    assert_eq!(stats.accepted_tokens, 0, "fixture forces MTP rejection");
    assert!(stats.proposed_tokens > 0);
    assert!(stats.target_forwards >= 2);
    // The unified engine ran it (sc-24130): the record names the proposer, the static step cache
    // and one host sync per verify step (AC2).
    let mtp_record = provider.last_decode_record().unwrap();
    assert_eq!(mtp_record.path, DecodePath::Mtp { drafts: 3 });
    assert_eq!(mtp_record.proposer, ProposerKind::Mtp);
    assert_eq!(mtp_record.kv_cache, KvCacheKind::Static);
    assert_eq!(mtp_record.proposed_tokens, u64::from(stats.proposed_tokens));
    assert_eq!(mtp_record.target_forwards, u64::from(stats.target_forwards));
    assert!(mtp_record.verify_steps > 0);
    assert_eq!(mtp_record.host_syncs_per_verify_step(), Some(1.0));

    let mut tool_request = request("weather in Paris?", 2);
    tool_request.thinking = ThinkingMode::Auto;
    tool_request.reasoning_effort = Some(ReasoningEffort::Low);
    tool_request.tools = vec![weather_tool()];
    tool_request.mtp = MtpMode::Auto;
    let tool_output = provider.generate(&tool_request, &mut |_| {}).unwrap();
    assert!(tool_output.mtp.is_some());
    assert!(tool_output.tool_calls.is_empty());

    drop(provider);
    drop(snapshot);
    let stop_snapshot = write_snapshot(&tokenizer_path, true);
    let stop_provider =
        LlamaProvider::load(&LoadSpec::dense(stop_snapshot.path().display().to_string())).unwrap();
    let mut stop_request = request("stop on the frozen EOS", 4);
    stop_request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    let mut token_events = 0usize;
    let stopped = stop_provider
        .generate(&stop_request, &mut |event| {
            if matches!(event, StreamEvent::Token { .. }) {
                token_events += 1;
            }
        })
        .unwrap();
    assert_eq!(stopped.finish_reason, Some(FinishReason::Stop));
    assert_eq!(stopped.usage.generated_tokens, 0);
    assert!(stopped.text.is_empty());
    assert_eq!(token_events, 0, "EOS is never surfaced as a token delta");
    let stop_stats = stopped.mtp.expect("MTP route still reports its prefill");
    assert_eq!(stop_stats.proposed_tokens, 0);

    // AC3 (sc-24130): the same checkpoint without an MTP head advertises no MTP; `Auto` decodes
    // normally and the record says `proposer=none`; `Enabled` is refused rather than downgraded.
    drop(stop_provider);
    drop(stop_snapshot);
    let plain_snapshot = write_snapshot_with(&tokenizer_path, false, false);
    let plain_provider = LlamaProvider::load(&LoadSpec::dense(
        plain_snapshot.path().display().to_string(),
    ))
    .unwrap();
    assert!(plain_provider.descriptor().capabilities.mtp.is_none());
    let mut auto_request = request("auto without a head", 3);
    auto_request.mtp = MtpMode::Auto;
    let auto_output = plain_provider.generate(&auto_request, &mut |_| {}).unwrap();
    assert!(auto_output.mtp.is_none());
    assert_eq!(auto_output.usage.generated_tokens, 3);
    let auto_record = plain_provider.last_decode_record().unwrap();
    assert_eq!(auto_record.path, DecodePath::Reference);
    assert_eq!(auto_record.proposer, ProposerKind::None);
    assert_eq!(auto_record.proposer.label(), "none");
    assert_eq!(auto_record.proposed_tokens, 0);
    let mut enabled_request = request("enabled without a head", 3);
    enabled_request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    assert!(matches!(
        plain_provider.generate(&enabled_request, &mut |_| {}),
        Err(core_llm::Error::Unsupported(_))
    ));
    assert_eq!(stop_stats.accepted_tokens, 0);
    assert_eq!(stop_stats.target_forwards, 1);
}
