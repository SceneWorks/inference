//! Real-CUDA acceptance for the bespoke SDXL-family image-detail provider.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{AdapterKind, AdapterSpec, WeightsSource};
use candle_gen_sdxl::{SdxlDetail, SdxlDetailPaths, SdxlDetailRequest};

fn env_dir(name: &str) -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var(name).unwrap_or_else(|_| panic!("set {name}")),
    ))
}

#[test]
#[ignore = "requires explicitly scheduled CUDA and local SDXL-family/tile-ControlNet components"]
fn sdxl_family_tile_detail_real_cuda() {
    let adapters = std::env::var_os("SDXL_DETAIL_ADAPTER")
        .map(|path| {
            vec![AdapterSpec::new(
                PathBuf::from(path),
                1.0,
                AdapterKind::Lora,
            )]
        })
        .unwrap_or_default();
    let model = SdxlDetail::load(&SdxlDetailPaths {
        sdxl_base: PathBuf::from(std::env::var("SDXL_DETAIL_BASE").expect("set SDXL_DETAIL_BASE")),
        tokenizer_clip_l: env_dir("SDXL_TOKENIZER_CLIP_L_DIR"),
        tokenizer_clip_bigg: env_dir("SDXL_TOKENIZER_CLIP_BIGG_DIR"),
        vae_fp16_fix: env_dir("SDXL_VAE_FP16_FIX_DIR"),
        tile_controlnet: env_dir("SDXL_TILE_CONTROLNET_DIR"),
        adapters,
    })
    .expect("load SDXL/RealVisXL/Illustrious tile detail provider");

    let width = 512;
    let height = 512;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[(x % 256) as u8, (y % 256) as u8, 96]);
        }
    }
    let tile = candle_gen::gen_core::Image {
        width,
        height,
        pixels,
    };
    let plain = model
        .generate(
            &SdxlDetailRequest {
                width,
                height,
                steps: 2,
                strength: 0.55,
                control_scale: 0.0,
                seed: 7,
                ..Default::default()
            },
            &tile,
            &tile,
            &mut |_| {},
        )
        .expect("zero-scale tile-ControlNet img2img render succeeds");
    let output = model
        .generate(
            &SdxlDetailRequest {
                width,
                height,
                steps: 2,
                strength: 0.55,
                control_scale: 0.7,
                seed: 7,
                ..Default::default()
            },
            &tile,
            &tile,
            &mut |_| {},
        )
        .expect("positive-scale tile-ControlNet img2img render succeeds");
    assert_eq!((output.width, output.height), (width, height));
    assert_eq!(output.pixels.len(), (width * height * 3) as usize);
    let (min, max) = output
        .pixels
        .iter()
        .fold((u8::MAX, u8::MIN), |(lo, hi), &value| {
            (lo.min(value), hi.max(value))
        });
    assert!(max > min, "detail render must not be a flat image");
    assert_ne!(
        plain.pixels, output.pixels,
        "positive tile-ControlNet scale must change deterministic img2img output"
    );
}
