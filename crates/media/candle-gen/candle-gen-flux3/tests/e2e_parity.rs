//! End to end vs upstream on the miniature package — the same `flux3_e2e_golden.safetensors` the
//! MLX twin's `e2e_parity` reads: `TextEncoder.encode` → `denoise_flow_map` over
//! `timesteps(4, 6.93)` from the same injected bf16-valued noise (FP32 DiT) → `AutoEncoder.decode`
//! → `to_pil`.
//!
//! Tolerance: the DiT, sampler and autoencoder run FP32 on Candle CPU (true f32, like the oracle);
//! the text tower is f32 here against the oracle's bf16 tower and differs by bf16 ulps (see
//! `text_parity`), which flows through four network evaluations. The bound on the packed tokens
//! and the decoded image is 5e-2 of peak (the MLX twin's bound); the RGB8 conversion is exact up
//! to the one-level truncation boundary (±1).
//!
//! The second case runs the same loop at `Precision::Bf16` — the release's sampling policy (bf16
//! DiT, `dt · velocity` rounded to bf16 per step) — against the oracle's own bf16-autocast loop,
//! on the build's device. Its bound is the larger of 5e-2 and 1.5× the oracle's recorded
//! bf16-vs-f32 distance: two bf16 implementations of one loop may not agree more closely than
//! bf16 agrees with f32.

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::CancelFlag;
use candle_gen_flux3::{
    decode_tokens, denoise_flow_map, float_image_to_rgb8, image_ids, load_autoencoder, load_dit,
    Flux3TextEncoder,
};

use crate::common::{
    assert_close, batched_nchw, cpu, device, errors, fixture, host_f32, host_i32, tiny_autoencoder,
    tiny_config, tiny_package, tiny_text_encoder,
};

fn text_encoder(device: &Device) -> Flux3TextEncoder {
    let config = tiny_config();
    Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &config.text_encoder,
        config.dit.text_in_channels,
        device,
    )
    .unwrap()
}

#[test]
fn generate_matches_upstream_from_injected_noise() {
    let device = cpu();
    let config = tiny_config();
    let te = text_encoder(&device);
    let dit = load_dit(
        &tiny_package().join("model.safetensors"),
        &config,
        DType::F32,
        &device,
    )
    .unwrap();
    let ae = load_autoencoder(&tiny_autoencoder(), &device).unwrap();
    let golden = fixture("flux3_e2e_golden.safetensors");
    let text = te.encode("a red fox in the snow").unwrap();
    let (h, w) = (2usize, 3usize);
    let ids = image_ids(h, w, config.dit.image_rope_scale as f32);
    let schedule = host_f32(golden.require("schedule"));
    assert_eq!(schedule.len(), 5);
    assert_eq!(
        schedule,
        gen_core_flux3::flow_map_schedule(4, 6.93).unwrap(),
        "the oracle's schedule is the contract's"
    );
    let mut steps = Vec::new();
    let tokens = denoise_flow_map(
        &dit,
        &text,
        golden.require("noise"),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |i| steps.push(i),
    )
    .unwrap();
    assert_eq!(steps, [1, 2, 3, 4]);
    assert_eq!(tokens.dtype(), DType::F32, "the Euler state is f32");
    assert_close("tokens", &tokens, golden.require("tokens"), 5e-2);

    let image = ae.decode(&tokens, h, w).unwrap();
    let want = batched_nchw(golden.require("image"));
    assert_close("image", &image, &want, 5e-2);

    // The RGB8 conversion of the ORACLE's image is exact to the truncation boundary.
    let rgb8 = float_image_to_rgb8(&want).unwrap();
    assert_eq!((rgb8.width, rgb8.height), (48, 32));
    let want_pixels: Vec<u8> = host_i32(golden.require("rgb8"))
        .into_iter()
        .map(|v| v as u8)
        .collect();
    assert_eq!(rgb8.pixels.len(), want_pixels.len());
    let off: Vec<(usize, u8, u8)> = rgb8
        .pixels
        .iter()
        .zip(&want_pixels)
        .enumerate()
        .filter(|(_, (a, b))| a.abs_diff(**b) > 1)
        .map(|(i, (a, b))| (i, *a, *b))
        .collect();
    assert!(
        off.is_empty(),
        "RGB8 differs by more than one level: {off:?}"
    );
    // And the whole pipeline's RGB8, through the same conversion, lands within a few levels of
    // the oracle's everywhere.
    let ours = decode_tokens(&ae, &tokens, h, w).unwrap();
    let far = ours
        .pixels
        .iter()
        .zip(&want_pixels)
        .filter(|(a, b)| a.abs_diff(**b) > 8)
        .count();
    assert_eq!(
        far, 0,
        "pipeline RGB8 within 8 levels of the oracle everywhere"
    );
    // A tripped cancel mid-loop is the typed cancellation, never a partial state.
    let cancel = CancelFlag::new();
    let flag = cancel.clone();
    match denoise_flow_map(
        &dit,
        &text,
        golden.require("noise"),
        &ids,
        &schedule,
        &cancel,
        |i| {
            if i == 2 {
                flag.cancel();
            }
        },
    ) {
        Err(candle_gen::CandleError::Canceled) => {}
        other => panic!("expected Canceled, got {:?}", other.map(|_| ())),
    }
}

/// The release's bf16 sampling policy vs upstream's bf16-autocast loop on the same noise, on the
/// build's device.
#[test]
fn bf16_release_policy_matches_upstream_autocast_loop() {
    let device = device();
    let config = tiny_config();
    let te = text_encoder(&device);
    let dit = load_dit(
        &tiny_package().join("model.safetensors"),
        &config,
        DType::BF16,
        &device,
    )
    .unwrap();
    let golden = fixture("flux3_e2e_golden.safetensors");
    let text = te.encode("a red fox in the snow").unwrap();
    let (h, w) = (2usize, 3usize);
    let ids = image_ids(h, w, config.dit.image_rope_scale as f32);
    let schedule = host_f32(golden.require("schedule"));
    let tokens = denoise_flow_map(
        &dit,
        &text,
        &golden.require("noise").to_device(&device).unwrap(),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |_| {},
    )
    .unwrap();
    assert_eq!(tokens.dtype(), DType::F32, "the Euler state is f32");
    let want_bf16 = golden.require("tokens_bf16");
    let want_f32 = golden.require("tokens");
    let oracle_distance: f32 = golden.meta("e2e_bf16_distance").parse().unwrap();
    let (max_abs, peak, _) = errors(want_bf16, want_f32);
    assert!(
        (max_abs / peak.max(1.0) - oracle_distance).abs() < 1e-6,
        "metadata agrees with the stored tensors ({max_abs:.3e} / {peak:.3e} vs {oracle_distance:.3e})"
    );
    assert!(
        oracle_distance > 1e-4,
        "bf16 autocast is not a no-op on the oracle"
    );
    let bound = (1.5 * oracle_distance).max(5e-2);
    assert_close("tokens (bf16 policy)", &tokens, want_bf16, bound);
    assert_close(
        "tokens (bf16 policy vs f32 oracle)",
        &tokens,
        want_f32,
        bound,
    );
}
