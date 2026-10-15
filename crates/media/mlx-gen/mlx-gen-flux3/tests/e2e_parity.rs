//! End to end vs upstream on the miniature package: `TextEncoder.encode` (bf16, as released) →
//! `denoise_flow_map` over `timesteps(4, 6.93)` from the same injected bf16-valued noise (FP32
//! DiT) → `AutoEncoder.decode` → `to_pil`.
//!
//! Tolerance: the DiT, sampler and autoencoder run FP32 on the MLX CPU stream (true f32, like the
//! oracle); the text tower is bf16 on both sides and MLX vs torch-CPU differ by 1–2 bf16 ulps
//! there (see `text_parity`), which flows through four network evaluations. The bound on the
//! packed tokens and the decoded image is 5e-2 of peak; the RGB8 conversion is exact up to the
//! one-level truncation boundary (±1).

use mlx_gen::CancelFlag;
use mlx_gen_flux3::{
    decode_tokens, denoise_flow_map, float_image_to_rgb8, image_ids, load_autoencoder, load_dit,
    Flux3TextEncoder,
};
use mlx_rs::Dtype;

use crate::common::{
    assert_close, cpu_process, fixture, host_f32, nchw_to_nhwc, nhwc_to_nchw, tiny_autoencoder,
    tiny_config, tiny_package, tiny_text_encoder,
};

#[test]
fn generate_matches_upstream_from_injected_noise() {
    cpu_process(
        "e2e_parity::generate_matches_upstream_from_injected_noise",
        run,
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
        mlx_gen::gen_core::flux3::flow_map_schedule(4, 6.93).unwrap(),
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
