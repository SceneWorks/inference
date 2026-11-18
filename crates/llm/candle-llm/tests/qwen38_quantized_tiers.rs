//! SW-7201 real-weight check: a fully quantized Qwen3.8-27B Candle tier (`prepare_snapshot ...
//! q8-full` or `q4-full`) against today's tier of the same body bits (`q8` / `q4`).
//!
//! ```text
//! QWEN38_FULL_TIER_SNAPSHOT=/path/to/q8-full QWEN38_PLAIN_TIER_SNAPSHOT=/path/to/q8 \
//!   cargo test -p candle-llm --features cuda --test qwen38_quantized_tiers -- --ignored --nocapture
//! ```
//!
//! Text: today's tier decodes a greedy reply and records each step's top-two logit margin; the full
//! tier is teacher-forced on that reply, and wherever its greedy choice differs today's margin must
//! be a near-tie ([`NEAR_TIE_LOGITS`]). Image prompts: both tiers load through the provider and must
//! ground a solid-color image; their greedy replies are printed beside each other. Each tier is
//! loaded alone and dropped before the next.

use candle_core::{DType, Device, Tensor};
use candle_llm::models::{Qwen35Config, Qwen35Model};
use candle_llm::primitives::projection::{ProjectionFormat, QuantSpec};
use candle_llm::primitives::Weights;
use candle_llm::LlamaProvider;
use core_llm::{
    Channel, Content, ImageRef, LoadSpec, Message, Role, Sampling, StreamEvent, TextLlm,
    TextLlmRequest, ThinkingMode, Tokenizer,
};

const FULL: &str = "QWEN38_FULL_TIER_SNAPSHOT";
const PLAIN: &str = "QWEN38_PLAIN_TIER_SNAPSHOT";
/// A greedy choice may differ from today's tier only where today's top two logits are this close.
const NEAR_TIE_LOGITS: f32 = 1.0;
const REPLY_TOKENS: usize = 48;

fn snapshot(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| panic!("set {var}"))
}

/// The decoder in the tier's own stored-block format.
fn load_model(dir: &str, device: &Device) -> Qwen35Model {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap())
            .unwrap();
    let block = &config["quantization"];
    assert_eq!(
        block["storage"], "ggml",
        "{dir} is a prepared stored-block tier"
    );
    let spec = QuantSpec::from_bits(block["bits"].as_u64().unwrap() as u32).unwrap();
    let weights = Weights::from_dir(dir, device).expect("read the tier");
    Qwen35Model::from_weights_format(
        &weights,
        "model.language_model",
        Qwen35Config::from_json(&config).unwrap(),
        Some(&ProjectionFormat::Ggml(spec)),
    )
    .expect("build the tier")
}

fn prompt_ids(dir: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let prompt = "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n\
                  <|im_start|>assistant\n<think>\n\n</think>\n\n";
    tokenizer.encode(prompt, false).unwrap()
}

/// Every position's logits `[positions][vocab]` for `ids` on a fresh cache.
fn logits(model: &Qwen35Model, ids: &[u32], device: &Device) -> Vec<Vec<f32>> {
    let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), device).unwrap();
    model
        .forward(&input, &mut model.new_cache(), 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
}

fn argmax(row: &[f32]) -> (u32, f32) {
    let mut sorted: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    (sorted[0].0 as u32, sorted[0].1 - sorted[1].1)
}

fn image_reply(provider: &dyn TextLlm, rgb: [u8; 3]) -> String {
    let request = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![
                Content::Image(ImageRef::new(256, 256, rgb.repeat(256 * 256)).unwrap()),
                Content::text("What is the dominant color of this image? Answer with one word."),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    };
    let mut content = String::new();
    provider
        .generate(&request, &mut |event| {
            if let StreamEvent::Token { text, channel, .. } = event {
                if channel == Channel::Content {
                    content.push_str(&text);
                }
            }
        })
        .expect("generate");
    content
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn qwen38_full_tier_replies_agree_with_the_plain_tier() {
    let (full, plain) = (snapshot(FULL), snapshot(PLAIN));
    let device = candle_llm::device::select_device().unwrap();
    let prompt = prompt_ids(&plain);
    assert_eq!(prompt, prompt_ids(&full), "both tiers share one tokenizer");

    // Today's tier: a greedy reply and each step's top-two margin.
    let (reply, margins) = {
        let model = load_model(&plain, &device);
        let mut ids = prompt.clone();
        let mut margins = Vec::new();
        for _ in 0..REPLY_TOKENS {
            let (token, margin) = argmax(logits(&model, &ids, &device).last().unwrap());
            ids.push(token);
            margins.push(margin);
        }
        (ids[prompt.len()..].to_vec(), margins)
    };

    // The full tier, teacher-forced on that reply.
    let model = load_model(&full, &device);
    let mut forced = prompt.clone();
    forced.extend_from_slice(&reply[..REPLY_TOKENS - 1]);
    let rows = logits(&model, &forced, &device);
    drop(model);
    let mut agree = 0;
    for (i, row) in rows[prompt.len() - 1..].iter().enumerate() {
        let (choice, _) = argmax(row);
        if choice == reply[i] {
            agree += 1;
        } else {
            assert!(
                margins[i] <= NEAR_TIE_LOGITS,
                "step {i}: the full tier chose {choice} where today's tier chose {} by a margin \
                 of {} logits (not a near-tie)",
                reply[i],
                margins[i]
            );
        }
    }
    println!("text: full tier agrees on {agree}/{REPLY_TOKENS} greedy steps of today's tier");

    // Image prompts through the provider: both tiers ground each color.
    let mut replies = Vec::new();
    for dir in [&plain, &full] {
        let provider = LlamaProvider::load(&LoadSpec::dense(dir.clone())).expect("load the tier");
        assert!(provider.descriptor().capabilities.supports_vision);
        let tier: Vec<String> = [[205u8, 35, 35], [35u8, 70, 200]]
            .into_iter()
            .map(|rgb| image_reply(&provider, rgb))
            .collect();
        drop(provider);
        replies.push(tier);
    }
    for (i, want) in ["red", "blue"].into_iter().enumerate() {
        println!(
            "image ({want}): today's tier {:?}, full tier {:?}",
            replies[0][i], replies[1][i]
        );
        for tier in &replies {
            assert!(
                tier[i].to_lowercase().contains(want),
                "{want}: {:?}",
                tier[i]
            );
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED qwen38_full_tier_replies_agree_with_the_plain_tier");
}
