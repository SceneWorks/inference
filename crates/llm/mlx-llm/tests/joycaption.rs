//! Real-weights JoyCaption parity test (`#[ignore]` — needs the 16GB snapshot), story 7157.
//!
//! Point `MLX_LLM_JOYCAPTION_SNAPSHOT` at a cached
//! `fancyfeast/llama-joycaption-beta-one-hf-llava` directory and run:
//!
//! ```text
//! MLX_LLM_JOYCAPTION_SNAPSHOT=/path/to/snapshot \
//!   cargo test --release --test joycaption -- --ignored --nocapture
//! ```
//!
//! ## Parity
//! Greedy decoding (temperature 0, no repetition penalty) is deterministic, so the converted VLM
//! must reproduce the reference engine's caption **token-for-token**. The golden below was captured
//! from the mlx-gen JoyCaption path on the same fixture (a solid gray 384×384 image, "Write a very
//! short caption.", 16 tokens). A divergence means the vision tower / projector / splice / decode
//! port differs numerically from the reference.

use core_llm::{
    Content, ImageRef, LoadSpec, Message, PrepareSpec, Quantize, Role, Sampling, TextLlm,
    TextLlmRequest, Tokenizer,
};

use mlx_llm::decode::CancelFlag;
use mlx_llm::joycaption::{build_chat_text, JoyCaptionModel, JoyCaptionProvider, STOP_TOKENS};
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::{load_for_model, prepare_snapshot};

/// Pure-greedy golden tokens for the gray-384 fixture + "Write a very short caption." (16 tokens).
const GOLDEN: &[i32] = &[
    53304, 3257, 315, 264, 6573, 11, 10269, 11, 18004, 4092, 449, 912, 9621, 6302, 11, 30953,
];
const GOLDEN_TEXT: &str =
    "Photograph of a solid, flat, gray background with no visible objects, textures";

fn snapshot() -> Option<String> {
    std::env::var("MLX_LLM_JOYCAPTION_SNAPSHOT").ok()
}

fn gray_image() -> (Vec<u8>, u32, u32) {
    (vec![127u8; 384 * 384 * 3], 384, 384)
}

/// A deterministic 512×384 RGB gradient — non-square, so captioning it exercises the bicubic resize
/// to the model's 384 input. Must be byte-identical to the fixture the reference golden was captured
/// from.
fn gradient_image() -> (Vec<u8>, u32, u32) {
    let mut px = Vec::with_capacity(512 * 384 * 3);
    for y in 0..384u32 {
        for x in 0..512u32 {
            px.push(((x + y) % 256) as u8);
            px.push(((x + 2 * y) % 256) as u8);
            px.push(((2 * x + y) % 256) as u8);
        }
    }
    (px, 512, 384)
}

struct PreparedSnapshot(std::path::PathBuf);

impl Drop for PreparedSnapshot {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Release gate for stored-quantized VLM snapshots: prepare the pinned full JoyCaption source,
/// prove only the Llama text projections were packed, reload the prepared directory, and execute
/// SigLIP → projector → image splice → quantized Llama generation with an actual RGB image.
#[test]
#[ignore = "needs pinned MLX_LLM_JOYCAPTION_SNAPSHOT and several GiB of temporary disk"]
fn prepared_q4_snapshot_runs_full_vlm() {
    let source = std::env::var("MLX_LLM_JOYCAPTION_SNAPSHOT").expect(
        "MLX_LLM_JOYCAPTION_SNAPSHOT must name the verified pinned JoyCaption source snapshot",
    );
    let source = std::path::Path::new(&source);
    assert!(
        source.is_dir(),
        "JoyCaption source snapshot is not a directory: {}",
        source.display()
    );

    let out = std::env::temp_dir().join(format!("mlx-llm-joycaption-q4-{}", std::process::id()));
    std::fs::remove_dir_all(&out).ok();
    let prepared = PreparedSnapshot(out);
    let report = prepare_snapshot(&PrepareSpec::quantized(source, &prepared.0, Quantize::Q4))
        .expect("registered quantize-prepare of the pinned JoyCaption source must succeed");
    assert_eq!(report.quantized, Some(Quantize::Q4));
    assert!(
        report.num_tensors > 0,
        "JoyCaption prepare must report written tensors"
    );

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(prepared.0.join("config.json"))
            .expect("prepared config.json must exist"),
    )
    .expect("prepared config.json must remain valid JSON");
    assert_eq!(config["quantization"]["bits"], 4);
    assert_eq!(config["text_config"]["quantization"]["bits"], 4);

    let weights = Weights::from_dir(&prepared.0).expect("prepared weights must reload");
    for base in [
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.mlp.gate_proj",
    ] {
        assert!(
            weights.contains(&format!("{base}.weight")),
            "missing {base}.weight"
        );
        assert!(
            weights.contains(&format!("{base}.scales")),
            "missing {base}.scales"
        );
        assert!(
            weights.contains(&format!("{base}.biases")),
            "missing {base}.biases"
        );
    }
    for base in [
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "multi_modal_projector.linear_1",
    ] {
        assert!(
            weights.contains(&format!("{base}.weight")),
            "missing dense {base}.weight"
        );
        assert!(
            !weights.contains(&format!("{base}.scales")),
            "{base} must remain dense"
        );
    }

    let provider = load_for_model(&LoadSpec::dense(prepared.0.to_string_lossy().to_string()))
        .expect("registered provider selection must load the prepared JoyCaption VLM");
    assert_eq!(provider.descriptor().id, "mlx-joycaption");
    assert!(provider.descriptor().capabilities.supports_vision);
    let (pixels, width, height) = gradient_image();
    let request = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![
                Content::Image(
                    ImageRef::new(width, height, pixels).expect("gradient image must be valid RGB"),
                ),
                Content::text("Write a very short caption."),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 2,
        seed: Some(0),
        ..Default::default()
    };
    let mut streamed_tokens = 0u32;
    let mut streamed_text = String::new();
    let mut done_usage = None;
    let generated = provider
        .generate(&request, &mut |event| match event {
            core_llm::StreamEvent::Token { text, .. } => {
                streamed_tokens += 1;
                streamed_text.push_str(&text);
            }
            core_llm::StreamEvent::Done { usage, .. } => {
                assert!(
                    done_usage.is_none(),
                    "public streaming contract emitted Done twice"
                );
                done_usage = Some(usage);
            }
        })
        .expect("registered JoyCaption provider must generate from the real image request");
    assert!(
        generated.usage.generated_tokens > 0,
        "prepared JoyCaption generation stopped before producing a token"
    );
    assert!(
        streamed_tokens > 0,
        "public streaming contract must emit Token"
    );
    assert_eq!(
        streamed_tokens, generated.usage.generated_tokens,
        "streamed Token count must equal returned generated-token usage"
    );
    assert!(
        !streamed_text.is_empty(),
        "public streaming contract must emit non-empty caption text"
    );
    assert_eq!(
        streamed_text, generated.text,
        "concatenated Token text must equal returned generation text"
    );
    assert_eq!(
        done_usage,
        Some(generated.usage),
        "terminal Done usage must equal returned generation usage"
    );
}

/// Model-level parity: vision tower → projector → splice → Llama decode reproduces the reference
/// engine's greedy caption token-for-token.
#[test]
#[ignore = "needs MLX_LLM_JOYCAPTION_SNAPSHOT"]
fn joycaption_model_matches_golden_tokens() {
    let Some(snap) = snapshot() else {
        eprintln!("skip: set MLX_LLM_JOYCAPTION_SNAPSHOT");
        return;
    };
    let model = JoyCaptionModel::from_dir(&snap).unwrap();
    let tok = Tokenizer::from_file(format!("{snap}/tokenizer.json")).unwrap();

    let (pixels, w, h) = gray_image();
    let features = model
        .image_features(&pixels, w as usize, h as usize)
        .unwrap();

    let chat = build_chat_text("Write a very short caption.");
    let prompt_ids: Vec<i32> = tok
        .encode(&chat, false)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect();

    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context: 0,
    };
    let gen = model
        .generate(
            &prompt_ids,
            &features,
            &greedy,
            16,
            Some(0),
            STOP_TOKENS,
            &CancelFlag::new(),
            &mut |_, _| {},
        )
        .unwrap();

    let text = tok
        .decode(
            &gen.tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            true,
        )
        .unwrap();
    println!("tokens = {:?}", gen.tokens);
    println!("text   = {text:?}");
    assert_eq!(
        gen.tokens, GOLDEN,
        "greedy tokens must match the reference engine"
    );
    assert_eq!(text.trim(), GOLDEN_TEXT);
}

/// Resize-path parity: a non-square (512×384) image is bicubic-resized to the model's 384 input and
/// must still reproduce the reference engine's greedy caption token-for-token — the proof the ported
/// PIL fixed-point resampler is bit-exact with the reference's.
#[test]
#[ignore = "needs MLX_LLM_JOYCAPTION_SNAPSHOT"]
fn joycaption_resize_path_matches_golden() {
    const GRAD: &[i32] = &[
        39212, 8278, 5497, 16850, 38336, 11, 34966, 11, 20779, 43546, 304, 43120,
    ];
    const GRAD_TEXT: &str =
        "Digital abstract pattern featuring diagonal, colorful, gradient triangles in vivid";

    let Some(snap) = snapshot() else {
        eprintln!("skip: set MLX_LLM_JOYCAPTION_SNAPSHOT");
        return;
    };
    let model = JoyCaptionModel::from_dir(&snap).unwrap();
    let tok = Tokenizer::from_file(format!("{snap}/tokenizer.json")).unwrap();

    let (pixels, w, h) = gradient_image();
    let features = model
        .image_features(&pixels, w as usize, h as usize)
        .unwrap();
    let chat = build_chat_text("Write a very short caption.");
    let prompt_ids: Vec<i32> = tok
        .encode(&chat, false)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect();
    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context: 0,
    };
    let gen = model
        .generate(
            &prompt_ids,
            &features,
            &greedy,
            12,
            Some(0),
            STOP_TOKENS,
            &CancelFlag::new(),
            &mut |_, _| {},
        )
        .unwrap();
    let text = tok
        .decode(
            &gen.tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            true,
        )
        .unwrap();
    println!("resize tokens = {:?}\nresize text = {text:?}", gen.tokens);
    assert_eq!(
        gen.tokens, GRAD,
        "resized-image greedy tokens must match the reference engine"
    );
    assert_eq!(text.trim(), GRAD_TEXT);
}

/// Provider-level parity through the `core-llm` multimodal contract: a `Content::Image` + text
/// request streams the same caption, with token events and a terminal `Done`.
#[test]
#[ignore = "needs MLX_LLM_JOYCAPTION_SNAPSHOT"]
fn joycaption_provider_streams_caption_through_contract() {
    let Some(snap) = snapshot() else {
        eprintln!("skip: set MLX_LLM_JOYCAPTION_SNAPSHOT");
        return;
    };
    let provider = JoyCaptionProvider::load(&LoadSpec::dense(snap)).unwrap();
    assert!(provider.descriptor().capabilities.supports_vision);

    let (pixels, w, h) = gray_image();
    let req = TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![
                Content::Image(ImageRef::new(w, h, pixels).unwrap()),
                Content::text("Write a very short caption."),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let mut tokens = 0usize;
    let mut saw_done = false;
    let out = provider
        .generate(&req, &mut |ev| match ev {
            core_llm::StreamEvent::Token { text, .. } => {
                streamed.push_str(&text);
                tokens += 1;
            }
            core_llm::StreamEvent::Done { .. } => saw_done = true,
        })
        .unwrap();

    println!("streamed = {streamed:?}");
    assert_eq!(
        out.text.trim(),
        GOLDEN_TEXT,
        "provider caption must match the reference engine"
    );
    assert_eq!(
        streamed.trim(),
        GOLDEN_TEXT,
        "streamed deltas must reconstruct the caption"
    );
    assert!(saw_done);
    assert_eq!(out.usage.generated_tokens, 16);
    assert!(tokens >= 1);

    // A request with no image is rejected.
    let no_image = TextLlmRequest {
        messages: vec![Message::user("Write a caption.")],
        max_new_tokens: 4,
        ..Default::default()
    };
    assert!(provider.generate(&no_image, &mut |_| {}).is_err());
}
