//! Reference latents encoded on Candle-Metal match the CPU's.
//!
//! The VAE encoder's `AvgDown` shortcut used to average a 5-D `[B, C, group, H', W']` view over
//! dim 2. At the pinned candle revision that reduction took candle-metal-kernels' rank > 4
//! strided-index fallback, which decoded the wrong dims and read past the source buffer, so every
//! reference / edit render on Metal conditioned on wrong — and, once the text encoder's vision
//! tower had left NaN-bearing buffers in the pool, non-finite — reference latents (sc-25909; fix
//! and provenance in `crates/media/candle-gen/vendor/candle-metal-kernels/VENDORED.md`). It now
//! averages a rank-4 `[B, C, group, H'·W']` view (a defensive second guard; the kernel fix is
//! pinned by `candle-gen`'s `metal_strided_reduce`), so this is the **pipeline-level** pin: any
//! device-specific defect in the reference encode, whichever op causes it. It runs the production
//! order — reference text conditioning first, then the VAE encode — on both devices.
//!
//! Metal-only: on a CPU-only build there is no second device to compare.

#![cfg(feature = "metal")]

use candle_core::{DType, Device};
use candle_gen::gen_core::{Image, RgbaImage};
use candle_gen_qwen_image_2_1::{
    encode_references, load_text_encoder, load_tokenizer, load_vae,
    prepare_conditioning_references, system_prompt_drop_count, QwenImage21Vae, VaeConfig,
};

use crate::common::{host_f32, tiny_snapshot};

/// bf16's unit roundoff, 2⁻⁸ (8 significand bits).
const BF16_UNIT_ROUNDOFF: f32 = 1.0 / 256.0;
/// The production Metal encode runs at bf16 (`compute_dtype_on` a GPU device): weights, input and
/// every activation are rounded to bf16 through the encoder, while the CPU reference is f32. That
/// rounding is the whole residual (the f32 leg below pins the rest at f32 ulps). Measured 3.6e-3
/// at peak 6.5e-1, i.e. 1.4 bf16 unit roundoffs of the peak; 4 roundoffs (1.0e-2 at that peak)
/// leaves ~3x headroom and stays 20-37x below the unfixed kernel's pool-dependent 2.1e-1-3.7e-1.
const BF16_ROUNDOFFS: f32 = 4.0;
/// The same VAE materialized at f32 on Metal differs from the CPU only by reassociation of f32 sums
/// (Metal's conv / reduction order differs from the CPU's): measured 1.8e-7 at peak 6.5e-1
/// (2.7e-7 relative, a few f32 ulps). That isolates the bf16 term above as the whole production
/// residual. 1e-5 of the peak leaves ~35x headroom; the unfixed kernel misses it by ~4 orders.
const F32_RELATIVE: f32 = 1e-5;

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

/// The packed reference latents `device` produces after the reference text conditioning ran:
/// through the production VAE (the device's compute dtype — bf16 on Metal) and, when
/// `f32_vae` is set, through the same VAE materialized at f32.
fn reference_latents(device: &Device, f32_vae: bool) -> Vec<Vec<f32>> {
    let root = tiny_snapshot();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root, device).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prepared = prepare_conditioning_references(&encoder, &[reference()]).unwrap();
    encoder
        .encode_conditioning(&tokenizer, "a red fox in the forest", drop, &prepared)
        .unwrap();
    let mut vaes = vec![load_vae(&root, device).unwrap()];
    if f32_vae {
        let cfg = VaeConfig::from_json_file(&root.join("vae").join("config.json")).unwrap();
        let vb =
            candle_gen::loader::component_vb(&root, "vae", DType::F32, device, "test").unwrap();
        vaes.push(QwenImage21Vae::new(&cfg, vb).unwrap());
    }
    vaes.iter()
        .map(|vae| {
            let latents = encode_references(vae, &prepared).unwrap();
            assert_eq!(latents.len(), 1, "one reference in, one latent block out");
            host_f32(&latents[0])
        })
        .collect()
}

/// `(non-finite count, max |got − want|, max |want|)`.
fn compare(got: &[f32], want: &[f32]) -> (usize, f32, f32) {
    assert_eq!(got.len(), want.len());
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let worst = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    (nonfinite, worst, peak)
}

#[test]
fn reference_vae_latents_on_metal_match_cpu_after_the_text_encoder() {
    let metal = reference_latents(&Device::new_metal(0).unwrap(), true);
    let cpu = reference_latents(&Device::Cpu, false).remove(0);
    for (label, got, bound) in [
        ("bf16", &metal[0], BF16_ROUNDOFFS * BF16_UNIT_ROUNDOFF),
        ("f32", &metal[1], F32_RELATIVE),
    ] {
        let (nonfinite, worst, peak) = compare(got, &cpu);
        eprintln!(
            "{label} metal vs f32 cpu reference latents: nonfinite={nonfinite} \
             max|D|={worst:.3e} peak={peak:.3e}"
        );
        assert_eq!(
            nonfinite, 0,
            "non-finite {label} reference latents on Metal"
        );
        assert!(
            worst <= bound * peak,
            "{label} Metal reference latents disagree with the CPU: max|D|={worst:.3e} at \
             peak {peak:.3e} (bound {bound:.1e} of the peak)"
        );
    }
}
