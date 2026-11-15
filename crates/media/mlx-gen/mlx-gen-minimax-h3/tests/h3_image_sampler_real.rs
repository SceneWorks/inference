//! SW-7106: the H3-Image sampler A/B on real weights. The registered `minimax_h3_image` generator
//! renders the same two prompts at two seeds, 2048² bf16, once with the released velocity Euler
//! and once with ComfyUI's stochastic LCM (`s_noise` 1.1 → 0.9). Every render must be
//! non-degenerate, and each LCM render must differ from its Euler twin. The images are written to
//! `MINIMAX_H3_IMAGE_SAMPLER_AB_OUT` for review.
//!
//! `#[ignore]`d: it needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`) and MiniMax-H3's
//! `text_encoder/` (`MINIMAX_H3_SNAPSHOT`). The `minimax-h3-image` real-weights profile runs it on
//! the `mlx-minimax-h3-image` job and uploads the images as `minimax-h3-image-sampler-ab-mlx`.

use std::path::PathBuf;

use mlx_gen::gen_core::h3_image::H3ImageSampler;
use mlx_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_minimax_h3::h3_image::IMAGE_DIT_COMPONENT;
use mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT;

const PROMPTS: [&str; 2] = [
    "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves breaking on the \
     rocks, detailed photograph",
    "A portrait of an elderly fisherman mending a net on a wooden dock, morning fog, soft window \
     light, 85mm photograph",
];
const SEEDS: [u64; 2] = [7106, 71060];
const LCM: H3ImageSampler = H3ImageSampler::Lcm {
    s_noise: 1.1,
    s_noise_end: 0.9,
    noise_clip_std: 0.0,
};

/// `(mean, std)` of the 8-bit pixels.
fn pixel_stats(pixels: &[u8]) -> (f64, f64) {
    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
    let var = pixels
        .iter()
        .map(|&p| (f64::from(p) - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

/// Mean absolute per-channel difference between two equal-size renders, in 8-bit levels.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / a.len() as f64
}

#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); eight 2048² renders"]
fn h3_image_sampler_ab_renders_2048_bf16() {
    let root = PathBuf::from(std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default());
    assert!(
        root.join(IMAGE_DIT_COMPONENT).join("config.json").is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} must hold a MiniMax-H3-Image snapshot. This test is \
         #[ignore]d and asserts rather than skips.",
        root.display()
    );
    let h3 = PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default());
    assert!(
        h3.join("text_encoder").join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT={} must hold MiniMax-H3's text_encoder/",
        h3.display()
    );
    let out_dir = std::env::var("MINIMAX_H3_IMAGE_SAMPLER_AB_OUT")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from);
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.to_owned(),
        WeightsSource::Dir(h3.join("text_encoder")),
    );
    let registry = mlx_gen_minimax_h3::h3_image::register_providers(
        mlx_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    let registration = registry
        .generators()
        .find(|g| (g.descriptor)().id == "minimax_h3_image")
        .expect("minimax_h3_image is registered");
    let generator = (registration.load)(&spec).unwrap();
    let mut failures = Vec::new();
    for (p, prompt) in PROMPTS.iter().enumerate() {
        for seed in SEEDS {
            let mut renders = Vec::new();
            for (name, sampler) in [("euler", H3ImageSampler::Euler), ("lcm", LCM)] {
                let req = GenerationRequest {
                    prompt: (*prompt).into(),
                    width: 2048,
                    height: 2048,
                    count: 1,
                    seed: Some(seed),
                    h3_image_sampler: Some(sampler),
                    ..Default::default()
                };
                generator.validate(&req).unwrap();
                let started = std::time::Instant::now();
                let mut steps = 0u32;
                let out = generator
                    .generate(&req, &mut |progress| {
                        if let mlx_gen::gen_core::Progress::Step { current, .. } = progress {
                            steps = current;
                        }
                    })
                    .unwrap();
                let GenerationOutput::Images(mut images) = out else {
                    panic!("minimax_h3_image must return images");
                };
                assert_eq!(images.len(), 1);
                let image = images.remove(0);
                assert_eq!((image.width, image.height), (2048, 2048));
                assert_eq!(steps, 6, "both samplers run the six fixed steps");
                let (mean, std) = pixel_stats(&image.pixels);
                println!(
                    "sampler-ab mlx prompt {p} seed {seed} {name}: {:.1} s, pixel mean {mean:.2}, \
                     std {std:.2}",
                    started.elapsed().as_secs_f64()
                );
                if !(mean.is_finite()
                    && std.is_finite()
                    && std > 10.0
                    && (5.0..250.0).contains(&mean))
                {
                    failures.push(format!(
                        "prompt {p} seed {seed} {name}: degenerate (mean {mean:.2}, std {std:.2})"
                    ));
                }
                if let Some(dir) = &out_dir {
                    let path = dir.join(format!("mlx-prompt{p}-seed{seed}-{name}.png"));
                    image::save_buffer(
                        &path,
                        &image.pixels,
                        image.width,
                        image.height,
                        image::ColorType::Rgb8,
                    )
                    .unwrap();
                    println!("wrote {}", path.display());
                }
                renders.push(image.pixels);
            }
            let diff = mean_abs_diff(&renders[0], &renders[1]);
            println!("sampler-ab mlx prompt {p} seed {seed}: euler vs lcm mean |diff| {diff:.2}");
            if diff < 0.5 {
                failures.push(format!(
                    "prompt {p} seed {seed}: lcm is identical to euler (mean |diff| {diff:.3})"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "sampler A/B failed: {failures:#?}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_sampler_real::h3_image_sampler_ab_renders_2048_bf16");
}
