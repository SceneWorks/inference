//! Qwen3-VL conditioning vs upstream `TextEncoder.encode` (real class, miniature
//! `Qwen3VLForConditionalGeneration` + WordLevel tokenizer carrying the PINNED chat template):
//! the single-string template tokenization, the pad-to-multiple window with the clamp, the pad
//! position ids, the padded attention and the selected-layer stacking.
//!
//! Tolerance: the tower runs in bf16 on both sides (the release's `text_encoder.dtype`), with the
//! miniature weights pre-rounded to bf16 so only activation rounding differs; MLX and torch-CPU
//! bf16 kernels round differently and over the full 36-block depth that rounding compounds. The
//! bound is 4e-2 of peak (the `mlx-gen-iris` measurement on the same tower shape: max 2.5 % of
//! peak); a structural error (wrong layer, template, position ids or mask) is O(peak). Ids, masks
//! and positions are exact.

use mlx_gen_flux3::Flux3TextEncoder;
use mlx_rs::Array;

use crate::common::{assert_close, cpu_process, fixture, host_i32, tiny_config, tiny_text_encoder};

const PROMPTS: [(&str, &str); 4] = [
    ("short", "a red fox in the snow"),
    ("exact", "a red fox in the snow , a cat on a roof"),
    (
        "long",
        "a red fox in the snow , a cat on a roof with a hat , golden hour , blue sky and sun of the",
    ),
    ("empty", ""),
];

fn encoder() -> Flux3TextEncoder {
    let config = tiny_config();
    Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .expect("tiny Qwen3-VL snapshot loads")
}

#[test]
fn windows_positions_and_stacked_states_match_upstream() {
    cpu_process(
        "text_parity::windows_positions_and_stacked_states_match_upstream",
        windows_positions_and_stacked_states_match_upstream_body,
    );
}

fn windows_positions_and_stacked_states_match_upstream_body() {
    let te = encoder();
    let golden = fixture("flux3_text_golden.safetensors");
    assert_eq!(te.pad_id(), host_i32(golden.require("pad_id").unwrap())[0]);
    let mut lengths = Vec::new();
    for (name, prompt) in PROMPTS {
        let out = te.encode(prompt).unwrap();
        let w = &out.window;
        assert_eq!(
            w.input_ids,
            host_i32(golden.require(&format!("{name}/input_ids")).unwrap()),
            "{name}: input ids"
        );
        assert_eq!(
            w.mask,
            host_i32(golden.require(&format!("{name}/mask")).unwrap()),
            "{name}: mask"
        );
        assert_eq!(
            w.positions,
            host_i32(golden.require(&format!("{name}/positions")).unwrap()),
            "{name}: positions"
        );
        let want: &Array = golden.require(&format!("{name}/tokens")).unwrap();
        assert_eq!(
            out.tokens.shape(),
            want.shape(),
            "{name}: conditioning shape"
        );
        assert_eq!(out.tokens.dtype(), mlx_rs::Dtype::Bfloat16);
        assert_close(&format!("{name}/tokens"), &out.tokens, want, 4e-2);
        let ids: Vec<f32> = out.ids.iter().flatten().copied().collect();
        let want_ids = golden.require(&format!("{name}/ids")).unwrap();
        assert_eq!(ids, crate::common::host_f32(want_ids), "{name}: ids");
        lengths.push((name, w.real, w.length(), w.truncated_tokens));
    }
    // The window law under test: pad to the multiple (8 here), clamp at the max (24 here).
    assert_eq!(
        lengths,
        [
            ("short", 11, 16, 0),
            ("exact", 17, 24, 0),
            ("long", 24, 24, 6),
            ("empty", 5, 8, 0),
        ]
    );
}

/// The pad rows are real conditioning (the pad token's state at position 1 over the real keys),
/// identical to each other, and never zero — upstream feeds them to the DiT as text tokens.
///
/// *Mutation that reds this:* zeroing or dropping the pad rows, or giving the pads their
/// sequential positions (the rows then differ from each other).
#[test]
fn pad_rows_are_identical_states_not_zeros() {
    cpu_process(
        "text_parity::pad_rows_are_identical_states_not_zeros",
        pad_rows_are_identical_states_not_zeros_body,
    );
}

fn pad_rows_are_identical_states_not_zeros_body() {
    let te = encoder();
    let out = te.encode("a red fox in the snow").unwrap();
    let real = out.window.real as i32;
    let length = out.window.length() as i32;
    assert!(
        length > real + 1,
        "the short prompt pads by more than one row"
    );
    let rows = out.tokens.split_axis(&[real], 0).unwrap().swap_remove(1);
    let pads: Vec<Array> = rows.split(length - real, 0).unwrap();
    let first = crate::common::host_f32(&pads[0]);
    assert!(first.iter().any(|v| *v != 0.0), "pad rows carry state");
    for pad in &pads[1..] {
        assert_eq!(
            crate::common::host_f32(pad),
            first,
            "every pad row is the same state"
        );
    }
    // And the real rows are unaffected by the pads: the same prompt padded further (a window of
    // 24 by a longer tokenization is not available here, so compare against the golden instead).
    let golden = fixture("flux3_text_golden.safetensors");
    let want = golden.require("short/tokens").unwrap();
    let real_rows = out.tokens.split_axis(&[real], 0).unwrap().swap_remove(0);
    let want_rows = want.split_axis(&[real], 0).unwrap().swap_remove(0);
    assert_close("short/real rows", &real_rows, &want_rows, 4e-2);
}
