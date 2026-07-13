//! Thinking-mode contract conformance (story 7707).
//!
//! Drives the registered `LlamaProvider` through the thinking-aware `core-llm` conformance suite with
//! a synthetic snapshot whose chat template gates an `enable_thinking` kwarg — so the provider
//! advertises a controllable reasoning mode (`supports_thinking = true`). This exercises the
//! `supports_thinking = true` branch of `check_thinking` in CI (Auto/Enabled/Disabled all validate;
//! Content deltas reconstruct `out.text`, Thinking deltas reconstruct `out.thinking`; a Disabled
//! request emits no reasoning) without needing model weights. The synthetic model never emits a
//! `<think>` block, so its reasoning is empty — a valid thinking-capable model that simply did not
//! reason. A model that *actually reasons* is covered by the gated `real_qwen3_produces_reasoning`
//! below (and by the Qwen3 conformance run in `conformance.rs`, now thinking-aware).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::{DType, Device, Tensor};

use candle_llm::primitives::sampler::{SplitMix64, TokenRng};
use candle_llm::LlamaProvider;
use core_llm::{Channel, LoadSpec, Message, StreamEvent, TextLlm, TextLlmRequest, ThinkingMode};
use core_llm_testkit::{textllm_conformance, TextLlmProfile};

const VOCAB: usize = 32;
static SEQ: AtomicU32 = AtomicU32::new(0);

/// A ChatML-style Jinja template that **gates `enable_thinking`** (so the provider advertises a
/// thinking mode) and, in no-think mode, injects the model's closed `<think></think>` generation
/// prompt — the transformers convention `load_chat_template` detects by source.
const THINKING_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}";

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

fn write_thinking_snapshot() -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("candle-llm-thinking-{}-{n}", std::process::id()));
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
    // The chat template (with its `enable_thinking` gate) is read from tokenizer_config.json — this
    // is what flips supports_thinking on for the loaded provider.
    let tok_cfg = serde_json::json!({ "chat_template": THINKING_TEMPLATE });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string(&tok_cfg).unwrap(),
    )
    .unwrap();

    let (h, inter, qd, kvd) = (8usize, 16usize, 8usize, 4usize);
    let mut rng = SplitMix64::new(0xBEEF);
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

#[test]
fn thinking_provider_passes_core_llm_conformance() {
    let dir = write_thinking_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // The template gates `enable_thinking`, so the provider advertises a thinking mode.
    let p = LlamaProvider::load(&spec).expect("load thinking provider");
    assert!(
        p.descriptor().capabilities.supports_thinking,
        "a template gating enable_thinking must set supports_thinking"
    );
    drop(p);

    // The thinking-aware suite drives the supports_thinking=true branch end-to-end.
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load thinking provider")),
        &TextLlmProfile::cheap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model that *actually reasons*: Qwen3's chat template gates `enable_thinking`, so an Enabled
/// request produces `<think>…</think>` reasoning. Asserts the provider advertises thinking, the
/// streamed channels reconstruct `out.text` / `out.thinking`, and reasoning is non-empty.
#[test]
#[ignore = "needs a Qwen3 snapshot via CANDLE_LLM_QWEN3_MODEL (real reasoning)"]
fn real_qwen3_produces_reasoning() {
    let dir = std::env::var("CANDLE_LLM_QWEN3_MODEL").expect("set CANDLE_LLM_QWEN3_MODEL");
    let p = LlamaProvider::load(&LoadSpec::dense(dir)).expect("load qwen3");
    assert!(
        p.descriptor().capabilities.supports_thinking,
        "Qwen3's chat template gates enable_thinking -> supports_thinking"
    );

    let mut req = TextLlmRequest::new(vec![Message::user("What is 17 + 25? Think briefly.")], 160);
    req.thinking = ThinkingMode::Enabled;
    req.seed = Some(0);

    let mut content = String::new();
    let mut thinking = String::new();
    let out = p
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                match channel {
                    Channel::Thinking => thinking.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .expect("generate");

    assert_eq!(
        content, out.text,
        "Content deltas must reconstruct out.text"
    );
    assert_eq!(
        thinking,
        out.thinking.clone().unwrap_or_default(),
        "Thinking deltas must reconstruct out.thinking"
    );
    assert!(
        out.thinking.as_deref().is_some_and(|t| !t.is_empty()),
        "Qwen3 in Enabled mode should produce reasoning"
    );
    eprintln!(
        "[qwen3 thinking] reasoning={:?}\n[qwen3 answer] {:?}",
        out.thinking, out.text
    );
}
