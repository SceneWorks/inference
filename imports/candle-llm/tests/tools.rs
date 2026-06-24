//! Tool ("function") calling contract conformance (story 7636 — candle parity with mlx-llm sc-7756).
//!
//! Drives the registered `LlamaProvider` through the tools-aware `core-llm` conformance suite with a
//! synthetic snapshot whose chat template renders a `tools` section and round-trips `<tool_call>`
//! blocks — so the provider advertises tool calling (`supports_tools = true`). This exercises the
//! `supports_tools = true` branch of `check_tools` in CI (a tools request validates and generates;
//! the streamed Content deltas reconstruct `out.text`; no `<tool_call>` markup leaks into the text;
//! parsed calls are well-formed) WITHOUT needing model weights. The synthetic model's vocabulary is
//! `t0..t31`, so it can never emit the literal `<tool_call>` markup — it is a tool-capable model that
//! simply did not call a tool. A model that *actually emits and parses* a call is the gated
//! `tests/qwen35_tools.rs` real-weight acceptance gate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::{DType, Device, Tensor};

use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::LlamaProvider;
use core_llm::{LoadSpec, Message, Sampling, TextLlm, TextLlmRequest, ToolSpec};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};

const VOCAB: usize = 32;
static SEQ: AtomicU32 = AtomicU32::new(0);

/// A ChatML-style Jinja template that renders an OpenAI-shaped `<tools>` section when `tools` are
/// offered and re-renders a prior assistant turn's `tool_calls` as the Qwen3.6 `<function=…>` /
/// `<parameter=…>` XML. Because its source mentions `tool_call`, `load_chat_template` flips
/// `supports_tools` on. It does NOT gate `enable_thinking`, so `supports_thinking` stays false —
/// isolating the tool path. (This is the structure of the real Qwen3.6 tool template, minus the
/// format-instruction prose.)
const TOOLS_TEMPLATE: &str = "{%- if tools %}{{- '<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools>\\n' }}{%- endif %}{%- for message in messages %}{%- if message.tool_calls %}{{- '<|im_start|>assistant\\n' }}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- if tool_call.arguments is defined %}{%- for args_name, args_value in tool_call.arguments|items %}{{- '<parameter=' + args_name + '>\\n' }}{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}{{- args_value }}{{- '\\n</parameter>\\n' }}{%- endfor %}{%- endif %}{{- '</function>\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- else %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

fn randn(shape: (usize, usize), rng: &mut SplitMix64) -> Tensor {
    let data: Vec<f32> = (0..shape.0 * shape.1)
        .map(|_| (rng.next_f32() - 0.5) * 0.4)
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn ones(d: usize) -> Tensor {
    Tensor::ones((d,), DType::F32, &Device::Cpu).unwrap()
}

/// A WordLevel `tokenizer.json` whose vocab is `t0..t{VOCAB-1}`, so distinct token ids decode to
/// distinct pieces (template markup tokenizes to the `t0` unk).
fn tokenizer_json() -> String {
    let entries: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{ "version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null, "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        entries.join(", ")
    )
}

fn write_tools_snapshot() -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-tools-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{ "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 32,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 999 }"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    // The tool-rendering chat template (read from tokenizer_config.json) is what flips supports_tools
    // on for the loaded provider.
    let tok_cfg = serde_json::json!({ "chat_template": TOOLS_TEMPLATE });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string(&tok_cfg).unwrap(),
    )
    .unwrap();

    let (h, inter, qd, kvd) = (8usize, 16usize, 8usize, 4usize);
    let mut rng = SplitMix64::new(0xC0FFEE);
    let mut arrays: HashMap<String, Tensor> = HashMap::new();
    arrays.insert(
        "model.embed_tokens.weight".into(),
        randn((VOCAB, h), &mut rng),
    );
    arrays.insert("model.norm.weight".into(), ones(h));
    arrays.insert("lm_head.weight".into(), randn((VOCAB, h), &mut rng));
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.insert(p("input_layernorm.weight"), ones(h));
        arrays.insert(p("post_attention_layernorm.weight"), ones(h));
        arrays.insert(p("self_attn.q_proj.weight"), randn((qd, h), &mut rng));
        arrays.insert(p("self_attn.k_proj.weight"), randn((kvd, h), &mut rng));
        arrays.insert(p("self_attn.v_proj.weight"), randn((kvd, h), &mut rng));
        arrays.insert(p("self_attn.o_proj.weight"), randn((h, qd), &mut rng));
        arrays.insert(p("mlp.gate_proj.weight"), randn((inter, h), &mut rng));
        arrays.insert(p("mlp.up_proj.weight"), randn((inter, h), &mut rng));
        arrays.insert(p("mlp.down_proj.weight"), randn((h, inter), &mut rng));
    }
    candle_core::safetensors::save(&arrays, dir.join("model.safetensors")).unwrap();
    dir
}

fn weather_tool() -> ToolSpec {
    ToolSpec::new(
        "get_weather",
        "Get the current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
        }),
    )
}

#[test]
fn tools_provider_passes_core_llm_conformance() {
    let dir = write_tools_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // The template renders tool calls, so the provider advertises tool calling (and no reasoning).
    let p = LlamaProvider::load(&spec).expect("load tools provider");
    assert!(
        p.descriptor().capabilities.supports_tools,
        "a template rendering tool calls must set supports_tools"
    );
    assert!(
        !p.descriptor().capabilities.supports_thinking,
        "this template does not gate enable_thinking, so supports_thinking stays false"
    );
    drop(p);

    // The tools-aware suite drives the supports_tools=true branch of check_tools end-to-end (the
    // profile offers a get_weather tool): a tools request validates + generates, the streamed Content
    // deltas reconstruct out.text, no <tool_call> markup leaks, and parsed calls are well-formed.
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load tools provider")),
        &TextLlmProfile::cheap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tools request validates and generates without error on a tools-capable provider, and never
/// leaks raw `<tool_call>` markup into the answer text (the synthetic model cannot emit it, so the
/// answer is plain content and no calls are parsed — proving the tool wiring is transparent when no
/// call is made).
#[test]
fn tools_request_generates_without_leaking_markup() {
    let dir = write_tools_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());
    let p = LlamaProvider::load(&spec).expect("load tools provider");

    let req = TextLlmRequest {
        messages: vec![Message::user("What is the weather in Paris?")],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        tools: vec![weather_tool()],
        ..Default::default()
    };
    p.validate(&req)
        .expect("a tools request validates on a tools-capable provider");

    let out = p.generate(&req, &mut |_| {}).expect("generate with tools");
    assert!(
        !out.text.contains("<tool_call>") && !out.text.contains("<function="),
        "no tool-call markup may leak into output.text: {:?}",
        out.text
    );
    // The synthetic model's t0..t31 vocabulary cannot spell `<tool_call>`, so it makes no call.
    assert!(out.tool_calls.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
