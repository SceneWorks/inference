//! Reference latents encoded on Candle-Metal match the CPU's.
//!
//! The VAE encoder's `AvgDown` shortcut averages a 5-D `[B, C, group, H', W']` view over dim 2. At
//! the pinned candle revision that reduction took candle-metal-kernels' rank > 4 strided-index
//! fallback, which decoded the wrong dims and read past the source buffer, so every reference /
//! edit render on Metal conditioned on wrong — and, once the text encoder's vision tower had left
//! NaN-bearing buffers in the pool, non-finite — reference latents (sc-25909; fix and provenance in
//! `crates/media/candle-gen/vendor/candle-metal-kernels/VENDORED.md`). This runs the production
//! order — reference text conditioning first, then the VAE encode — on both devices.
//!
//! Metal-only: on a CPU-only build there is no second device to compare.

#![cfg(feature = "metal")]

use candle_core::Device;
use candle_gen::gen_core::{Image, RgbaImage};
use candle_gen_qwen_image_2_1::{
    encode_references, load_text_encoder, load_tokenizer, load_vae,
    prepare_conditioning_references, system_prompt_drop_count,
};

use crate::common::{host_f32, tiny_snapshot};

/// A 64×64 gradient: smooth enough that a mis-indexed average is far from the true one.
fn reference() -> RgbaImage {
    let edge = 64u32;
    let pixels = (0..edge)
        .flat_map(|y| (0..edge).flat_map(move |x| [(x * 4) as u8, (y * 4) as u8, 32]))
        .collect();
    RgbaImage::from_rgb(&Image {
        width: edge,
        height: edge,
        pixels,
    })
    .unwrap()
}

/// The packed reference latents `device` produces after the reference text conditioning ran.
fn reference_latents(device: &Device) -> Vec<f32> {
    let root = tiny_snapshot();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root, device).unwrap();
    let vae = load_vae(&root, device).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prepared = prepare_conditioning_references(&encoder, &[reference()]).unwrap();
    encoder
        .encode_conditioning(&tokenizer, "a red fox in the forest", drop, &prepared)
        .unwrap();
    let latents = encode_references(&vae, &prepared).unwrap();
    assert_eq!(latents.len(), 1, "one reference in, one latent block out");
    host_f32(&latents[0])
}

#[test]
fn reference_vae_latents_on_metal_match_cpu_after_the_text_encoder() {
    let metal = reference_latents(&Device::new_metal(0).unwrap());
    let cpu = reference_latents(&Device::Cpu);
    assert_eq!(metal.len(), cpu.len());
    let nonfinite = metal.iter().filter(|v| !v.is_finite()).count();
    let worst = metal
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let peak = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "metal vs cpu reference latents: nonfinite={nonfinite} max|D|={worst:.3e} peak={peak:.3e}"
    );
    assert_eq!(nonfinite, 0, "non-finite reference latents on Metal");
    assert!(
        worst <= 1e-2 * peak,
        "Metal reference latents disagree with the CPU: max|D|={worst:.3e} at peak {peak:.3e}"
    );
}
