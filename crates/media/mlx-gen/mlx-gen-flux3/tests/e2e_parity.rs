//! End to end vs upstream on the miniature package: `TextEncoder.encode` (bf16, as released) →
//! `denoise_flow_map` over `timesteps(4, 6.93)` from the same injected bf16-valued noise (FP32
//! DiT) → `AutoEncoder.decode` → `to_pil`.
//!
//! Tolerance: the DiT, sampler and autoencoder run FP32 on the MLX CPU stream (true f32, like the
//! oracle); the text tower is bf16 on both sides and MLX vs torch-CPU differ by 1–2 bf16 ulps
//! there (see `text_parity`), which flows through four network evaluations. The bound on the
//! packed tokens and the decoded image is 5e-2 of peak; the RGB8 conversion is exact up to the
//! one-level truncation boundary (±1).
//!
//! The second case runs the same loop at `Precision::Bf16` — the release's sampling policy (bf16
//! DiT, `dt · velocity` rounded to bf16 per step) — against the oracle's own bf16-autocast loop.
//! Its bound is the larger of 5e-2 and 1.5× the oracle's recorded bf16-vs-f32 distance: two
//! bf16 implementations of one loop may not agree more closely than bf16 agrees with f32.

use mlx_gen::CancelFlag;
use mlx_gen_flux3::{
    decode_tokens, denoise_flow_map, float_image_to_rgb8, image_ids, load_autoencoder, load_dit,
    Flux3TextEncoder,
};
use mlx_rs::Dtype;

use crate::common::{
    assert_close, cpu_process, errors, fixture, host_f32, nchw_to_nhwc, nhwc_to_nchw,
    tiny_autoencoder, tiny_config, tiny_package, tiny_text_encoder,
};

#[test]
fn rgb8_materializes_strided_non_square_nhwc_in_pixel_order() {
    cpu_process(
        "e2e_parity::rgb8_materializes_strided_non_square_nhwc_in_pixel_order",
        rgb8_materializes_strided_non_square_nhwc_in_pixel_order_body,
    );
}

fn rgb8_materializes_strided_non_square_nhwc_in_pixel_order_body() {
    let interleaved = [
        -2.0f32, 0.0, 2.0, -0.5, 0.5, -0.25, 0.25, -0.75, 0.75, 2.0, -2.0, 0.0, 0.5, -0.25, -0.5,
        -0.75, 0.75, 0.25,
    ];
    let expected = [
        0u8, 127, 255, 63, 191, 95, 159, 31, 223, 255, 0, 127, 191, 95, 63, 31, 223, 159,
    ];
    let contiguous = mlx_rs::Array::from_slice(&interleaved, &[1, 2, 3, 3]);
    let mut planar = Vec::with_capacity(interleaved.len());
    for channel in 0..3 {
        for pixel in interleaved.chunks_exact(3) {
            planar.push(pixel[channel]);
        }
    }
    let nchw = mlx_rs::Array::from_slice(&planar, &[1, 3, 2, 3]);
    let strided = nchw_to_nhwc(&nchw);
    assert_eq!(strided.shape(), contiguous.shape());
    assert_eq!(host_f32(&strided), interleaved);
    for image in [&contiguous, &strided] {
        let rgb = float_image_to_rgb8(image).unwrap();
        assert_eq!((rgb.width, rgb.height), (3, 2));
        assert_eq!(
            rgb.pixels, expected,
            "logical HWC order, clamp and truncation"
        );
    }
    // Removing pipeline::float_image_to_rgb8's reshape(-1) reads planar storage from `strided`;
    // the explicit channel-distinct bytes above must then fail, not merely the two outputs agree.
}

#[test]
fn generate_matches_upstream_from_injected_noise() {
    cpu_process(
        "e2e_parity::generate_matches_upstream_from_injected_noise",
        run,
    );
}

/// The release's bf16 sampling policy vs upstream's bf16-autocast loop on the same noise.
#[test]
fn bf16_release_policy_matches_upstream_autocast_loop() {
    cpu_process(
        "e2e_parity::bf16_release_policy_matches_upstream_autocast_loop",
        run_bf16,
    );
}

fn run_bf16() {
    let config = tiny_config();
    let te = Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let dit = load_dit(
        &tiny_package().join("model.safetensors"),
        &config,
        Dtype::Bfloat16,
    )
    .unwrap();
    let golden = fixture("flux3_e2e_golden.safetensors");
    let text = te.encode("a red fox in the snow").unwrap();
    let (h, w) = (2usize, 3usize);
    let ids = image_ids(h, w, config.dit.image_rope_scale as f32);
    let schedule = host_f32(golden.require("schedule").unwrap());
    let tokens = denoise_flow_map(
        &dit,
        &text,
        golden.require("noise").unwrap(),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |_| {},
    )
    .unwrap();
    assert_eq!(tokens.dtype(), Dtype::Float32, "the Euler state is f32");
    let want_bf16 = golden.require("tokens_bf16").unwrap();
    let want_f32 = golden.require("tokens").unwrap();
    let oracle_distance: f32 = golden
        .metadata("e2e_bf16_distance")
        .expect("the golden records the oracle's bf16-vs-f32 distance")
        .parse()
        .unwrap();
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
    // And no closer to the f32 oracle than to the bf16 one would be suspicious only if it were
    // exact; what is required is that the bf16 run stays within the same bound of the f32 truth.
    assert_close(
        "tokens (bf16 policy vs f32 oracle)",
        &tokens,
        want_f32,
        bound,
    );
}

fn run() {
    let config = tiny_config();
    let te = Flux3TextEncoder::load(
        &tiny_text_encoder(),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let dit = load_dit(
        &tiny_package().join("model.safetensors"),
        &config,
        Dtype::Float32,
    )
    .unwrap();
    let ae = load_autoencoder(&tiny_autoencoder()).unwrap();
    let golden = fixture("flux3_e2e_golden.safetensors");
    let text = te.encode("a red fox in the snow").unwrap();
    let (h, w) = (2usize, 3usize);
    let ids = image_ids(h, w, config.dit.image_rope_scale as f32);
    let schedule = host_f32(golden.require("schedule").unwrap());
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
        golden.require("noise").unwrap(),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |i| steps.push(i),
    )
    .unwrap();
    assert_eq!(steps, [1, 2, 3, 4]);
    assert_close("tokens", &tokens, golden.require("tokens").unwrap(), 5e-2);

    let image = ae.decode(&tokens, h as i32, w as i32).unwrap();
    let want = nchw_to_nhwc(golden.require("image").unwrap());
    assert_close("image", &image, &want, 5e-2);
    let _ = nhwc_to_nchw(&image);

    // The RGB8 conversion of the ORACLE's image is exact to the truncation boundary.
    let rgb8 = float_image_to_rgb8(&want).unwrap();
    let want_rgb8 = golden.require("rgb8").unwrap();
    assert_eq!((rgb8.width, rgb8.height), (48, 32));
    let want_pixels: Vec<u8> = want_rgb8
        .as_dtype(Dtype::Int32)
        .unwrap()
        .reshape(&[-1])
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|v| *v as u8)
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
    // And the whole pipeline's RGB8, through the same conversion, lands within a level of the
    // oracle's wherever the float images agree to a level.
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
}
