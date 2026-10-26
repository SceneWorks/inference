//! Frozen training teacher precision against actual tiny HF5.20 CPU reference.
use crate::common::{fixture, host_f32, tiny_config, tiny_package};
use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::flux3::{assemble_window, TextWindow};
use candle_gen_flux3::Flux3TextEncoder;
fn prompt(name: &str) -> &'static str {
    match name {
    "short"=>"a red fox in the snow", "exact"=>"a red fox in the snow , a cat on a roof",
    "long"=>"a red fox in the snow , a cat on a roof with a hat , golden hour , blue sky and sun of the", "empty"=>"", _=>panic!("fixture prompt"),
}
}
fn close(
    name: &str,
    got: &candle_gen::candle_core::Tensor,
    expected: &candle_gen::candle_core::Tensor,
) {
    assert_eq!(got.dims(), expected.dims(), "{name}");
    let (actual, reference) = (host_f32(got), host_f32(expected));
    let peak = reference
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let error = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(actual.iter().all(|x| x.is_finite()));
    assert!(
        error <= 2e-5 + 1e-4 * peak,
        "{name}: abs {error}, peak {peak}"
    );
}
#[test]
fn explicit_f32_teacher_is_not_bf16_output_upcast_and_honors_training_window() {
    let cfg = tiny_config();
    let root = tiny_package().join("text_encoder");
    let encoder = Flux3TextEncoder::load_for_training(
        &root,
        &cfg.text_encoder,
        cfg.dit.text_in_channels,
        &Device::Cpu,
        DType::F32,
        24,
    )
    .unwrap();
    assert_eq!(encoder.dtype(), DType::F32);
    assert_eq!(encoder.output_dtype(), DType::F32);
    let golden = fixture("flux3_text_training_golden.safetensors");
    for name in ["short", "exact", "long", "empty"] {
        let result = encoder.encode(prompt(name)).unwrap();
        assert_eq!(result.tokens.dtype(), DType::F32);
        close(
            name,
            &result.tokens,
            golden.require(&format!("f32/{name}/tokens")),
        );
    }
    let short = encoder.window(prompt("short")).unwrap();
    let empty = encoder.window("").unwrap();
    let empty = assemble_window(
        &empty.input_ids[..empty.real],
        encoder.pad_id(),
        short.length(),
        24,
    )
    .unwrap();
    close(
        "matched-window empty",
        &encoder.encode_window(empty).unwrap().tokens,
        golden.require("f32/empty_at_short_window/tokens"),
    );
    let cap16 = Flux3TextEncoder::load_for_training(
        &root,
        &cfg.text_encoder,
        cfg.dit.text_in_channels,
        &Device::Cpu,
        DType::F32,
        16,
    )
    .unwrap();
    let result = cap16.encode(prompt("long")).unwrap();
    assert_eq!(result.window.length(), 16);
    assert!(result.window.truncated_tokens > 0);
    close(
        "training cap16",
        &result.tokens,
        golden.require("f32/long_cap16/tokens"),
    );
    assert_ne!(
        encoder.precision_profile_key(),
        cap16.precision_profile_key()
    );
    let input_ids = golden
        .require("f32/rotary_probe/input_ids")
        .to_vec1::<i32>()
        .unwrap();
    let mask = golden
        .require("f32/rotary_probe/mask")
        .to_vec1::<i32>()
        .unwrap();
    let positions = golden
        .require("f32/rotary_probe/positions")
        .to_vec1::<i32>()
        .unwrap();
    close(
        "explicit positions",
        &encoder
            .encode_window(TextWindow {
                input_ids,
                mask,
                positions,
                real: 4,
                truncated_tokens: 0,
            })
            .unwrap()
            .tokens,
        golden.require("f32/rotary_probe/tokens"),
    );
}
#[test]
fn cpu_bf16_teacher_is_actionably_rejected_not_silently_f32() {
    let cfg = tiny_config();
    let result = Flux3TextEncoder::load_for_training(
        std::path::Path::new("missing-never-loaded"),
        &cfg.text_encoder,
        cfg.dit.text_in_channels,
        &Device::Cpu,
        DType::BF16,
        512,
    );
    assert!(matches!(
        result,
        Err(candle_gen::CandleError::Unsupported(_))
    ));
}
