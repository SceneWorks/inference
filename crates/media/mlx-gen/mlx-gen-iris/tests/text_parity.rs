//! Qwen3-VL conditioning vs upstream `Qwen3VLTextEncoder` (real constructor, miniature
//! `Qwen3VLForConditionalGeneration` + WordLevel tokenizer): the separately-tokenized template
//! pieces, the window with caption truncation and suffix survival, the masks, and the stacked
//! selected-layer states with pad rows zeroed.
//!
//! Tolerance: the tower runs in bf16 on both sides (the release's `text_encoder.dtype`), with the
//! miniature weights pre-rounded to bf16 so only activation rounding differs; MLX and torch-CPU bf16
//! kernels round differently. Measured: max |Δ| = 0.125–0.156 at peak ≈ 10.6, i.e. 1–2 bf16 ulps
//! at that magnitude (mean ≈ 2e-2), identical on the MLX CPU and GPU streams — so the bound is
//! bf16-scale, 2e-2 of peak (≈ 3 ulps). A structural error (wrong layer, template, RoPE or mask)
//! is O(peak). Ids and masks are exact.

use mlx_gen_iris::IrisTextEncoder;
use mlx_rs::Array;

use crate::common::{assert_close, fixture, host_i32, tiny_config, tiny_text_encoder};

fn encoder() -> IrisTextEncoder {
    IrisTextEncoder::load(&tiny_text_encoder(), &tiny_config().text_encoder)
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
    assert_eq!(
        te.prefix_ids(),
        host_i32(golden.require("prefix_ids").unwrap())
    );
    assert_eq!(
        te.suffix_ids(),
        host_i32(golden.require("suffix_ids").unwrap())
    );
}

#[test]
fn windows_masks_and_layer_states_match_upstream() {
    let te = encoder();
    let golden = fixture("iris_text_golden.safetensors");
    for (name, prompt) in PROMPTS {
        let window = te.window(prompt).unwrap();
        let want_caption = host_i32(golden.require(&format!("{name}/caption_ids")).unwrap());
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
            host_i32(golden.require(&format!("{name}/mask")).unwrap()),
            "{name}: mask"
        );
        let want: &Array = golden.require(&format!("{name}/embeddings")).unwrap();
        let got = out.states.squeeze_axes(&[0]).unwrap();
        assert_close(&format!("{name}/embeddings"), &got, want, 2e-2);
        // pad rows are exactly zero
        let real = out.mask.iter().filter(|m| **m == 1).count();
        if real < out.mask.len() {
            let tail = got.split_axis(&[real as i32], 0).unwrap().swap_remove(1);
            assert!(
                crate::common::host_f32(&tail).iter().all(|v| *v == 0.0),
                "{name}: pads"
            );
        }
    }
}

#[test]
fn empty_negative_prompt_is_the_training_null() {
    let te = encoder();
    let golden = fixture("iris_text_golden.safetensors");
    let null = te.encode("").unwrap();
    assert_eq!(null.mask, host_i32(golden.require("null/mask").unwrap()));
    assert_close(
        "null/embeddings",
        &null.states.squeeze_axes(&[0]).unwrap(),
        golden.require("null/embeddings").unwrap(),
        2e-2,
    );
}
