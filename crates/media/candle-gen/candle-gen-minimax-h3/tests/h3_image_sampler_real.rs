//! SW-7106: the H3-Image sampler A/B on real weights (candle/CUDA). The registered
//! `minimax_h3_image` generator renders the same two prompts at two seeds, 2048² bf16, once with
//! the released velocity Euler and once with ComfyUI's stochastic LCM (`s_noise` 1.1 → 0.9).
//! Every render must be non-degenerate, and each LCM render must differ from its Euler twin. The
//! images are written to `MINIMAX_H3_IMAGE_SAMPLER_AB_OUT` for review.
//!
//! `#[ignore]`d: it needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`), MiniMax-H3's
//! `text_encoder/` (`MINIMAX_H3_SNAPSHOT`) and CUDA. The `minimax-h3-image` real-weights profile
//! runs it on the `candle-minimax-h3-image` job and uploads the images as
//! `minimax-h3-image-sampler-ab-candle`.

const PROMPTS: [&str; 2] = [
    "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves breaking on the \
     rocks, detailed photograph",
    "A portrait of an elderly fisherman mending a net on a wooden dock, morning fog, soft window \
     light, 85mm photograph",
];
const SEEDS: [u64; 2] = [7106, 71060];

/// `(mean, std)` of the 8-bit pixels.
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / a.len() as f64
}

#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT), MiniMax-H3's text encoder (MINIMAX_H3_SNAPSHOT) and CUDA; eight 2048x2048 renders"]
fn image_sampler_ab_renders_2048_on_cuda() {
    let _ = (PROMPTS, SEEDS);
    #[cfg(not(feature = "cuda"))]
    panic!("the sampler A/B runs the 66 GB DiT on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen::gen_core::h3_image::H3ImageSampler;
        use candle_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
        use std::path::PathBuf;

        let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
        let root = PathBuf::from(&raw);
        assert!(
            !raw.is_empty()
                && root
                    .join("transformer_diffusers")
                    .join("config.json")
                    .is_file(),
            "MINIMAX_H3_IMAGE_SNAPSHOT={raw} must hold a MiniMax-H3-Image snapshot; this test is \
             #[ignore]d and asserts rather than skips."
        );
        let out_dir = std::env::var("MINIMAX_H3_IMAGE_SAMPLER_AB_OUT")
            .ok()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from);
        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir).expect("create the A/B output directory");
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        // The conditioner: this snapshot's own `text_encoder/` when present, else MiniMax-H3's
        // byte-identical copy staged as the `text_encoder` component.
        if !root.join("text_encoder").is_dir() {
            let h3 = std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default();
            assert!(
                !h3.is_empty(),
                "the image snapshot has no text_encoder/; MINIMAX_H3_SNAPSHOT must point at a \
                 MiniMax-H3 snapshot holding text_encoder/ shards 1-12"
            );
            spec.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(PathBuf::from(h3).join("text_encoder")),
            );
        }
        let generator = candle_gen_minimax_h3::image_model::load_image(&spec)
            .expect("load the minimax_h3_image generator");
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let mut failures = Vec::new();
        for (p, prompt) in PROMPTS.iter().enumerate() {
            for seed in SEEDS {
                let mut renders = Vec::new();
                for (name, sampler) in [("euler", H3ImageSampler::Euler), ("lcm", lcm)] {
                    let req = GenerationRequest {
                        prompt: (*prompt).into(),
                        width: 2048,
                        height: 2048,
                        seed: Some(seed),
                        h3_image_sampler: Some(sampler),
                        ..Default::default()
                    };
                    generator.validate(&req).expect("validate");
                    let started = std::time::Instant::now();
                    let mut steps = Vec::new();
                    let output = generator
                        .generate(&req, &mut |progress| {
                            if let candle_gen::gen_core::Progress::Step { current, .. } = progress {
                                steps.push(current)
                            }
                        })
                        .expect("render");
                    assert_eq!(steps, vec![1, 2, 3, 4, 5, 6], "one forward per fixed step");
                    let GenerationOutput::Images(mut images) = output else {
                        panic!("an image generator must return images")
                    };
                    assert_eq!(images.len(), 1);
                    let img = images.remove(0);
                    assert_eq!((img.width, img.height), (2048, 2048));
                    let (mean, std) = pixel_stats(&img.pixels);
                    println!(
                        "sampler-ab candle prompt {p} seed {seed} {name}: {:.1} s, pixel mean \
                         {mean:.2}, std {std:.2}",
                        started.elapsed().as_secs_f64()
                    );
                    if !(mean.is_finite()
                        && std.is_finite()
                        && std > 10.0
                        && (5.0..250.0).contains(&mean))
                    {
                        failures.push(format!(
                            "prompt {p} seed {seed} {name}: degenerate (mean {mean:.2}, std \
                             {std:.2})"
                        ));
                    }
                    if let Some(dir) = &out_dir {
                        let path = dir.join(format!("candle-prompt{p}-seed{seed}-{name}.png"));
                        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                            .expect("rgb")
                            .save(&path)
                            .expect("write the render");
                        println!("wrote {}", path.display());
                    }
                    renders.push(img.pixels);
                }
                let diff = mean_abs_diff(&renders[0], &renders[1]);
                println!(
                    "sampler-ab candle prompt {p} seed {seed}: euler vs lcm mean |diff| {diff:.2}"
                );
                if diff < 0.5 {
                    failures.push(format!(
                        "prompt {p} seed {seed}: lcm is identical to euler (mean |diff| \
                         {diff:.3})"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "sampler A/B failed: {failures:#?}");
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_sampler_real::image_sampler_ab_renders_2048_on_cuda"
        );
    }
}
