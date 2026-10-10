//! Qwen3-VL conditioning vs upstream `Qwen3VLTextEncoder` (real constructor, miniature
//! `Qwen3VLForConditionalGeneration` + WordLevel tokenizer): the separately-tokenized template
//! pieces, the window with caption truncation and suffix survival, the masks, and the stacked
//! selected-layer states with pad rows zeroed — the MLX twin's fixture, through the shared
//! `candle_llm::CausalLm` tower.
//!
//! Tolerance: the oracle runs the tower in bf16 (the release's `text_encoder.dtype`) over the
//! 36-block miniature tower with weights pre-rounded to bf16. The Candle CPU backend has no
//! half-precision GEMM, so here the tower runs f32 (`tower_dtype`) — the measured distance (max
//! |Δ| 2.0e-2–2.5e-2 of peak) is therefore upstream's own bf16-vs-fp32 envelope on this snapshot
//! (2.2 %). The bound is the MLX twin's 4e-2 of peak; a structural error (wrong layer, template,
//! RoPE or mask) is O(peak). Ids and masks are exact. `windows_masks_and_layer_states_match_upstream`
//! runs on the build's device, so on the CUDA lane the tower runs the release's bf16 on the real
//! kernels against the same fixture and bound.

use candle_gen_iris::text_encoder::caption_overflow_warning;
use candle_gen_iris::IrisTextEncoder;

use candle_gen::candle_core::Device;

use crate::common::{
    assert_close, cpu, device, fixture, host_f32, host_i32, tiny_config, tiny_text_encoder,
};

fn encoder() -> IrisTextEncoder {
    encoder_on(&cpu())
}

fn encoder_on(device: &Device) -> IrisTextEncoder {
    IrisTextEncoder::load(&tiny_text_encoder(), &tiny_config().text_encoder, device)
        .expect("tiny Qwen3-VL snapshot loads")
}

const PROMPTS: [(&str, &str); 3] = [
    ("short", "a red fox in the snow"),
    (
        "overflow",
        "a red fox , a red fox , golden hour , snow in the",
    ),
    ("empty", ""),
];

#[test]
fn template_pieces_tokenize_like_upstream() {
    let te = encoder();
    let golden = fixture("iris_text_golden.safetensors");
    assert_eq!(te.prefix_ids(), host_i32(golden.require("prefix_ids")));
    assert_eq!(te.suffix_ids(), host_i32(golden.require("suffix_ids")));
}

#[test]
fn windows_masks_and_layer_states_match_upstream() {
    // The build's device: on the CUDA lane the tower runs the release's bf16 on the real kernels.
    let te = encoder_on(&device());
    let golden = fixture("iris_text_golden.safetensors");
    for (name, prompt) in PROMPTS {
        let window = te.window(prompt).unwrap();
        let want_caption = host_i32(golden.require(&format!("{name}/caption_ids")));
        let caption_len = if want_caption == [-1] {
            0
        } else {
            want_caption.len()
        };
        let kept = window.window_tokens() - te.suffix_ids().len();
        assert_eq!(
            &window.input_ids[te.prefix_ids().len()..te.prefix_ids().len() + kept],
            &want_caption[..kept.min(caption_len)],
            "{name}: caption ids"
        );
        assert_eq!(
            window.truncated_tokens,
            caption_len - kept,
            "{name}: truncation"
        );
        let out = te.encode_window(&window).unwrap();
        assert_eq!(
            out.mask,
            host_i32(golden.require(&format!("{name}/mask"))),
            "{name}: mask"
        );
        let want = golden.require(&format!("{name}/embeddings"));
        // The miniature tower is 36 blocks deep and selects the release's 12 post-block states
        // ([2, 5, …, 35]), so the stacking law under test is the shipped one.
        assert_eq!(want.dim(1).unwrap(), 12, "{name}: golden selected layers");
        let got = out.states.squeeze(0).unwrap();
        assert_eq!(got.dims(), want.dims(), "{name}: conditioning shape");
        assert_close(&format!("{name}/embeddings"), &got, want, 4e-2);
        // pad rows are exactly zero
        let real = out.mask.iter().filter(|m| **m == 1).count();
        if real < out.mask.len() {
            let tail = got.narrow(0, real, out.mask.len() - real).unwrap();
            assert!(host_f32(&tail).iter().all(|v| *v == 0.0), "{name}: pads");
        }
    }
}

#[test]
fn empty_negative_prompt_is_the_training_null() {
    let te = encoder();
    let golden = fixture("iris_text_golden.safetensors");
    let null = te.encode("").unwrap();
    assert_eq!(null.mask, host_i32(golden.require("null/mask")));
    assert_close(
        "null/embeddings",
        &null.states.squeeze(0).unwrap(),
        golden.require("null/embeddings"),
        4e-2,
    );
}

#[test]
fn caption_overflow_is_warned_under_the_release_policy() {
    let te = encoder();
    // Upstream's oracle log for this prompt: "longest 14 tokens, budget 7".
    let overflow = te.window(PROMPTS[1].1).unwrap();
    let msg = caption_overflow_warning(&overflow, overflow.mask.len(), te.suffix_ids().len())
        .expect("an overflowing caption is warned");
    assert!(
        msg.contains("7 caption tokens dropped (caption 14 tokens, budget 7"),
        "{msg}"
    );
    let short = te.window(PROMPTS[0].1).unwrap();
    assert_eq!(
        caption_overflow_warning(&short, short.mask.len(), te.suffix_ids().len()),
        None
    );
}
