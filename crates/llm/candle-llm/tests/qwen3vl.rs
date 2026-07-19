//! Real-weights end-to-end tests for Qwen3-VL **vision** (`qwen3_vl` VLM) — the epic acceptance
//! gate. Point `CANDLE_LLM_QWEN3VL_MODEL` at a `Qwen/Qwen3-VL-8B-Instruct` snapshot (the one carrying
//! `model.visual.*`):
//!
//! ```text
//! CANDLE_LLM_QWEN3VL_MODEL=<snapshot dir> \
//!   cargo test --features cuda --test qwen3vl -- --ignored --nocapture
//! ```
//!
//! - **Image (sc-8080):** a solid-color image asked for its dominant color exercises the whole stack
//!   — preprocess → ViT encoder + DeepStack taps → merger → splice into the (generic `CausalLm`)
//!   decoder embeds → interleaved M-RoPE + DeepStack fusion → greedy decode. If any stage is wrong
//!   (preprocess layout, encoder numerics, DeepStack fusion, splice, M-RoPE positions), the model
//!   cannot reliably name the color.
//! - **Video (the candle peer of mlx-llm sc-8081):** a short synthetic video whose color changes over
//!   time additionally exercises the multi-frame ViT path, per-frame `<|video_pad|>` expansion, the
//!   Text–Timestamp-Alignment timestamps, and the interleaved-M-RoPE per-frame time axis. Asking
//!   which color comes first is a temporal-grounding check; the reversed-order pair (blue→green) makes
//!   a fixed/order-insensitive answer impossible.

use candle_llm::LlamaProvider;
use core_llm::{
    Channel, Content, ImageRef, LoadSpec, Message, Role, Sampling, StreamEvent, TextLlm,
    TextLlmOutput, TextLlmRequest, ThinkingMode, VideoRef,
};

fn model_dir() -> String {
    std::env::var("CANDLE_LLM_QWEN3VL_MODEL").expect("set CANDLE_LLM_QWEN3VL_MODEL")
}

/// A solid-color RGB image.
fn solid_image(w: u32, h: u32, rgb: [u8; 3]) -> ImageRef {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        px.extend_from_slice(&rgb);
    }
    ImageRef::new(w, h, px).expect("image bytes")
}

/// An image + text user turn.
fn image_request(img: ImageRef, prompt: &str) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Image(img), Content::text(prompt)],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 32,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

fn run(p: &dyn TextLlm, r: &TextLlmRequest) -> (TextLlmOutput, String) {
    let mut content = String::new();
    let out = p
        .generate(r, &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                if channel == Channel::Content {
                    content.push_str(&text);
                }
            }
        })
        .expect("generate");
    (out, content)
}

#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct snapshot (model.visual.*) via CANDLE_LLM_QWEN3VL_MODEL"]
fn qwen3vl_vision_grounds_on_image() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load Qwen3-VL VLM");
    assert!(
        p.descriptor().capabilities.supports_vision,
        "a qwen3_vl checkpoint carrying model.visual.* must advertise vision"
    );

    // Two solid colors: a correct vision path names each; a broken one (preprocess / encoder /
    // DeepStack / splice / M-RoPE) cannot ground reliably, and certainly not on both.
    for (rgb, want, label) in [
        ([205u8, 35, 35], "red", "red"),
        ([35u8, 70, 200], "blue", "blue"),
    ] {
        let img = solid_image(256, 256, rgb);
        let (out, content) = run(
            &p,
            &image_request(
                img,
                "What is the dominant color of this image? Answer with one word.",
            ),
        );
        println!(
            "\n=== Qwen3-VL VISION ({label}) ===\n[answer] {:?}\n",
            out.text
        );
        assert!(
            !content.trim().is_empty(),
            "{label}: must produce an answer"
        );
        assert!(
            content.to_lowercase().contains(want),
            "{label}: greedy answer must ground on the image and name '{want}', got: {content:?}"
        );
    }
}

/// Build a sampled video from solid-color frames at `fps`, with the per-frame timestamps the host
/// would derive from the sampled frame indices (`idx / fps`). 64×64 keeps the video preprocess
/// resize a no-op (a clean, fast grid) while still exercising the full multi-frame ViT path.
fn solid_video(colors: &[[u8; 3]], fps: f32) -> VideoRef {
    let frames: Vec<ImageRef> = colors.iter().map(|&c| solid_image(64, 64, c)).collect();
    let timestamps: Vec<f32> = (0..colors.len()).map(|i| i as f32 / fps).collect();
    VideoRef::new(frames, timestamps).expect("video frames + timestamps")
}

/// A video + text user turn.
fn video_request(video: VideoRef, prompt: &str, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Video(video), Content::text(prompt)],
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

const VIDEO_PROMPT: &str =
    "This is a short video. What color is shown at the start, and what color is shown at the end? \
     Answer with two color words: first the starting color, then the ending color.";

/// **Video AC #1 — a short video prompt produces a temporally-grounded answer.** Load the real
/// Qwen3-VL-8B snapshot end-to-end; confirm it advertises **video** (the config carries
/// `video_token_id`); then feed a synthetic video whose color changes over time (red → blue) and ask
/// which color comes first. A correct end-to-end video path (frame sampling layout → multi-frame ViT
/// → per-frame `<|video_pad|>` expansion → Text–Timestamp-Alignment timestamps → interleaved-M-RoPE
/// per-frame time axis → DeepStack fusion → decode) is what lets the model order the frames; a break
/// anywhere makes the temporal answer unreliable. The model's exact phrasing is logged.
#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct snapshot (model.visual.*) via CANDLE_LLM_QWEN3VL_MODEL"]
fn qwen3vl_video_grounds_temporal_order() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load Qwen3-VL VLM");
    assert!(
        p.descriptor().capabilities.supports_video,
        "a loaded Qwen3-VL checkpoint must advertise video (config carries video_token_id)"
    );

    // Solid RED for the first half then solid BLUE for the second half. Four sampled frames at 1 fps
    // fold (temporal_patch_size=2) into **two** merged temporal patches — patch 0 = red+red (≈0.5s),
    // patch 1 = blue+blue (≈2.5s) — so the model sees two distinct timestamped vision frames in order.
    let video = solid_video(
        &[[205, 35, 35], [205, 35, 35], [35, 70, 200], [35, 70, 200]],
        1.0,
    );
    let (out, content) = run(&p, &video_request(video, VIDEO_PROMPT, 48));
    println!(
        "\n=== Qwen3-VL VIDEO temporal-order ===\n[answer] {:?}\n",
        out.text
    );
    assert!(
        out.usage.prompt_tokens > 0 && out.usage.generated_tokens > 0,
        "must run the video prefill"
    );
    assert!(!content.trim().is_empty(), "must produce an answer");
    let lc = content.to_lowercase();
    // The video path must ground on *both* frames' colors. Temporal ordering (red before blue) is the
    // strong claim; if the model names them out of order on synthetic input we still require both
    // colors to appear (proving both frames reached the model with distinct content).
    assert!(
        lc.contains("red") && lc.contains("blue"),
        "video answer must ground on both frames (red and blue), got: {content:?}"
    );
    if let (Some(r), Some(b)) = (lc.find("red"), lc.find("blue")) {
        println!(
            "temporal order {}: red@{r} blue@{b}",
            if r < b {
                "CORRECT (red before blue)"
            } else {
                "out-of-order"
            }
        );
        assert!(
            r < b,
            "temporally-grounded answer must name the starting color (red) before the ending color \
             (blue), got: {content:?}"
        );
    }
}

/// **Video — temporal grounding on a second, independent video (order reversed).** Colors in the
/// *opposite* order (blue first, then green) so a model that simply always emits a fixed pair cannot
/// pass both this and the red→blue test. The four frames fold into two merged temporal patches
/// (blue@~0.5s, green@~2.5s); a temporally-grounded answer names blue before green. Same
/// Text–Timestamp-Alignment path, different content.
#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct snapshot (model.visual.*) via CANDLE_LLM_QWEN3VL_MODEL"]
fn qwen3vl_video_temporal_order_reversed() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load Qwen3-VL VLM");

    // Solid BLUE for the first half, then solid GREEN for the second half.
    let video = solid_video(
        &[[35, 70, 200], [35, 70, 200], [40, 180, 60], [40, 180, 60]],
        1.0,
    );
    let (out, content) = run(&p, &video_request(video, VIDEO_PROMPT, 48));
    println!(
        "\n=== Qwen3-VL VIDEO temporal-order (reversed) ===\n[answer] {:?}\n",
        out.text
    );
    assert!(out.usage.generated_tokens > 0);
    let lc = content.to_lowercase();
    assert!(!lc.trim().is_empty(), "must produce an answer");
    assert!(
        lc.contains("blue") && lc.contains("green"),
        "video answer must ground on both frames (blue and green), got: {content:?}"
    );
    if let (Some(b), Some(g)) = (lc.find("blue"), lc.find("green")) {
        println!(
            "temporal order {}: blue@{b} green@{g}",
            if b < g {
                "CORRECT (blue before green)"
            } else {
                "out-of-order"
            }
        );
        assert!(
            b < g,
            "temporally-grounded answer must name the starting color (blue) before the ending color \
             (green), got: {content:?}"
        );
    }
}
