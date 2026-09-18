//! Terminal real-weight acceptance for native Bonsai image, multi-image, and video generation.
//!
//! This gate is explicit and ignored in cache-independent CI. Run it once per shipped format:
//!
//! ```text
//! CANDLE_LLM_BONSAI_MODEL=<mlx snapshot> \
//!   cargo test --features cuda --test bonsai_multimodal -- --ignored --nocapture
//! CANDLE_LLM_BONSAI_MODEL=<language.gguf> \
//! CANDLE_LLM_BONSAI_PROJECTOR=<mmproj-Q8_0.gguf> \
//!   cargo test --features cuda --test bonsai_multimodal -- --ignored --nocapture
//! ```

use candle_llm::LlamaProvider;
use core_llm::{
    Content, ImageRef, LoadSpec, Message, Role, Sampling, TextLlm, TextLlmRequest, ThinkingMode,
    VideoRef,
};

fn solid_image(rgb: [u8; 3]) -> ImageRef {
    let mut pixels = Vec::with_capacity(256 * 256 * 3);
    for _ in 0..256 * 256 {
        pixels.extend_from_slice(&rgb);
    }
    ImageRef::new(256, 256, pixels).unwrap()
}

fn request(content: Vec<Content>, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

fn run(provider: &dyn TextLlm, content: Vec<Content>) -> String {
    let output = provider
        .generate(&request(content, 48), &mut |_| {})
        .expect("native Bonsai multimodal generation");
    assert!(output.usage.prompt_tokens > 0);
    assert!(output.usage.generated_tokens > 0);
    assert!(!output.text.trim().is_empty());
    output.text.to_lowercase()
}

#[test]
#[ignore = "requires CANDLE_LLM_BONSAI_MODEL and, for GGUF, CANDLE_LLM_BONSAI_PROJECTOR"]
fn bonsai_images_multiimage_and_video_ground_in_document_order() {
    let model = std::env::var("CANDLE_LLM_BONSAI_MODEL")
        .expect("CANDLE_LLM_BONSAI_MODEL must point to the frozen MLX snapshot or language GGUF");
    let spec = match std::env::var("CANDLE_LLM_BONSAI_PROJECTOR") {
        Ok(projector) => LoadSpec::dense(model).with_projector(projector),
        Err(_) => LoadSpec::dense(model),
    };
    let provider = LlamaProvider::load(&spec).expect("load complete native Bonsai vision stack");
    let caps = &provider.descriptor().capabilities;
    assert!(caps.supports_vision);
    assert!(caps.supports_video);

    let red = solid_image([205, 35, 35]);
    let blue = solid_image([35, 70, 205]);
    let green = solid_image([40, 180, 60]);

    let answer = run(
        &provider,
        vec![
            Content::Image(red.clone()),
            Content::text("What is the dominant color? Answer with one word."),
        ],
    );
    assert!(answer.contains("red"), "single-image grounding: {answer:?}");

    let answer = run(
        &provider,
        vec![
            Content::text("First image: "),
            Content::Image(blue.clone()),
            Content::text(" Second image: "),
            Content::Image(green.clone()),
            Content::text(" Name their colors in order, two words only."),
        ],
    );
    let blue_at = answer.find("blue").expect("multi-image answer names blue");
    let green_at = answer
        .find("green")
        .expect("multi-image answer names green");
    assert!(blue_at < green_at, "interleaved image order: {answer:?}");

    let video = VideoRef::new(
        vec![red.clone(), red, blue.clone(), blue],
        vec![0.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let answer = run(
        &provider,
        vec![
            Content::Video(video),
            Content::text(
                "Name the color at the start, then the color at the end. Two words only.",
            ),
        ],
    );
    let red_at = answer.find("red").expect("video answer names starting red");
    let blue_at = answer.find("blue").expect("video answer names ending blue");
    assert!(red_at < blue_at, "temporal video order: {answer:?}");
}
