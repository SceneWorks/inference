//! Round-trips a streaming generation through the backend-neutral `core-llm` contract (story 7154).
//!
//! Builds a tiny on-disk snapshot (config.json + model.safetensors + tokenizer.json), loads it both
//! directly (`LlamaProvider::load`) and through the MLX registry (`mlx_llm::load_textllm`), and streams
//! a generation entirely via the `core_llm::TextLlm` trait — no model weights needed, runs in CI.

use mlx_rs::Array;

use core_llm::{
    Content, Error as CoreError, ImageRef, LoadSpec, Message, Sampling, StreamEvent, TextLlm,
    TextLlmRequest,
};
use mlx_llm::load_textllm;
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::LlamaProvider;

mod common;
use common::{assert_fixture_is_self_removing, Fixture};

const TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "WordLevel",
        "vocab": { "<unk>": 0, "hello": 1, "world": 2, "<eos>": 3 },
        "unk_token": "<unk>"
    }
}"#;

// eos_token_id = 99 is outside the 4-token vocab, so generation always runs to max_new_tokens
// (deterministic token count for the assertions) rather than randomly stopping.
const CONFIG_JSON: &str = r#"{
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 4,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false,
    "eos_token_id": 99
}"#;

fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

/// Write a tiny but complete snapshot directory and return it, guard and all.
///
/// sc-17768: the returned [`Fixture`] owns a `TempDir`, so the snapshot leaves on `Drop` even out of
/// a panicking test. Bind it for as long as a provider loaded from it is used — MLX's
/// `load_safetensors` is lazy, so the provider still reads this directory.
fn write_tiny_snapshot() -> Fixture {
    let dir = Fixture::new("mlx-llm-contract-", None);
    std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();

    // hidden=8, vocab=4, intermediate=16, q dim = heads*head_dim = 8, kv dim = kv_heads*head_dim = 4.
    let (h, v, inter, qd, kvd) = (8, 4, 16, 8, 4);
    let mut rng = SplitMix64::new(0xC0FFEE);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    arrays.push(("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng)));
    arrays.push((
        "model.norm.weight".into(),
        Array::ones::<f32>(&[h]).unwrap(),
    ));
    arrays.push(("lm_head.weight".into(), randn(&[v, h], &mut rng)));
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.push((
            p("input_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        arrays.push((
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        arrays.push((p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng)));
        arrays.push((p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng)));
        arrays.push((p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng)));
        arrays.push((p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng)));
        arrays.push((p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng)));
        arrays.push((p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng)));
        arrays.push((p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng)));
    }
    let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    dir
}

fn text_request(max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user("hello world")],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        ..Default::default()
    }
}

#[test]
fn provider_loads_and_streams_through_the_contract() {
    let dir = write_tiny_snapshot();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap())).unwrap();

    // Descriptor identity.
    assert_eq!(provider.descriptor().id, PROVIDER_ID);
    assert_eq!(provider.descriptor().backend, "mlx");

    // Stream a generation through the trait.
    let req = text_request(6);
    let mut tokens = 0usize;
    let mut streamed = String::new();
    let mut saw_done = false;
    let out = provider
        .generate(&req, &mut |ev| match ev {
            StreamEvent::Token { text, .. } => {
                tokens += 1;
                streamed.push_str(&text);
            }
            StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                saw_done = true;
                assert_eq!(finish_reason, core_llm::FinishReason::Length);
                assert_eq!(usage.generated_tokens, 6);
            }
        })
        .unwrap();

    assert!(saw_done);
    assert_eq!(tokens, 6);
    assert_eq!(out.usage.generated_tokens, 6);
    assert!(out.usage.prompt_tokens > 0);
    assert_eq!(out.finish_reason, Some(core_llm::FinishReason::Length));
    // The incremental token deltas reconstruct the final text.
    assert_eq!(streamed, out.text);
}

#[test]
fn registry_routes_to_the_provider() {
    let dir = write_tiny_snapshot();
    // Go through the explicit MLX catalog by id.
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    assert_eq!(provider.descriptor().id, PROVIDER_ID);

    let out = provider.complete(&text_request(4)).unwrap();
    assert_eq!(out.usage.generated_tokens, 4);
}

#[test]
fn greedy_generation_is_reproducible_through_the_contract() {
    let dir = write_tiny_snapshot();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    let a = provider.complete(&text_request(8)).unwrap();
    let b = provider.complete(&text_request(8)).unwrap();
    assert_eq!(a.text, b.text);
}

#[test]
fn already_cancelled_request_errors_before_inference() {
    let dir = write_tiny_snapshot();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    let req = text_request(8);
    req.cancel.cancel();
    let res = provider.generate(&req, &mut |_| {});
    assert!(matches!(res, Err(CoreError::Canceled)));
}

#[test]
fn validate_rejects_unsupported_vision_input() {
    let dir = write_tiny_snapshot();
    let provider = LlamaProvider::load(&LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    let img = ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
    let req = TextLlmRequest {
        messages: vec![Message {
            role: core_llm::Role::User,
            content: vec![Content::Text("look".into()), Content::Image(img)],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        max_new_tokens: 4,
        ..Default::default()
    };
    assert!(matches!(
        provider.validate(&req),
        Err(CoreError::Unsupported(_))
    ));
}

/// Drop-regression for this suite's fixture helper: the snapshot tree leaves with the value. Flip
/// [`Fixture::new`]'s builder to `disable_cleanup(true)` and this goes RED.
#[test]
fn contract_fixture_is_self_removing() {
    assert_fixture_is_self_removing(write_tiny_snapshot());
}
