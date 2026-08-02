//! Fresh-process CUDA measurement harness for SC-15256 / SC-16170.
//!
//! Run once per hosted Q4/Q8/BF16 tier and policy. Sequential runs emit physical text, denoise, and
//! decode peaks from the provider's `Progress::Loading` boundaries; every run records the overall
//! `VramProbe` report and writes the fixed-seed image for resident-vs-sequential comparison.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationMemory, GenerationOutput, GenerationRequest,
    Image, LoadSpec, Progress, WeightsSource,
};

fn reference_fixture(width: u32, height: u32) -> Image {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[
                (x * 255 / width.max(1)) as u8,
                (y * 255 / height.max(1)) as u8,
                ((x + y) * 255 / (width + height).max(1)) as u8,
            ]);
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn seam_ratio(image: &candle_gen::gen_core::Image, seam_x: usize) -> f64 {
    let width = image.width as usize;
    let height = image.height as usize;
    if seam_x == 0 || seam_x >= width {
        return 0.0;
    }
    let edge_delta = |x: usize| -> f64 {
        let mut total = 0.0;
        for y in 0..height {
            for channel in 0..3 {
                let left = image.pixels[(y * width + x - 1) * 3 + channel] as f64;
                let right = image.pixels[(y * width + x) * 3 + channel] as f64;
                total += (right - left).abs();
            }
        }
        total / (height * 3) as f64
    };
    let seam = edge_delta(seam_x);
    let lo = seam_x.saturating_sub(16).max(1);
    let hi = (seam_x + 16).min(width - 1);
    let neighborhood = (lo..=hi)
        .filter(|&x| x != seam_x)
        .map(edge_delta)
        .sum::<f64>()
        / (hi - lo) as f64;
    seam / neighborhood.max(1e-9)
}

fn mutation_metrics(left: &Image, right: &Image) -> (u8, f64) {
    assert_eq!((left.width, left.height), (right.width, right.height));
    assert_eq!(left.pixels.len(), right.pixels.len());
    let mut maximum = 0u8;
    let mut total = 0u64;
    for (&lhs, &rhs) in left.pixels.iter().zip(&right.pixels) {
        let delta = lhs.abs_diff(rhs);
        maximum = maximum.max(delta);
        total += u64::from(delta);
    }
    (maximum, total as f64 / left.pixels.len() as f64)
}

fn generated_image(output: GenerationOutput) -> Image {
    match output {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.remove(0)
        }
        other => panic!("expected image output, got {other:?}"),
    }
}

#[test]
#[ignore = "needs Z_IMAGE_TIER_DIR + CUDA; run in a fresh otherwise-idle process"]
fn measure_z_image_tier() {
    let tier_dir = PathBuf::from(
        std::env::var("Z_IMAGE_TIER_DIR")
            .expect("set Z_IMAGE_TIER_DIR to a hosted q4/q8/bf16 tier snapshot"),
    );
    let tier = std::env::var("Z_IMAGE_TIER_NAME").unwrap_or_else(|_| "unknown".into());
    let provider = std::env::var("Z_IMAGE_PROVIDER").unwrap_or_else(|_| "z_image_turbo".into());
    let policy_name = std::env::var("Z_IMAGE_POLICY").unwrap_or_else(|_| "staged".into());
    let memory = match policy_name.as_str() {
        "resident" => GenerationMemory::default(),
        "staged" | "request-staged" => GenerationMemory {
            stage_residency: true,
            ..Default::default()
        },
        "decode" => GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            ..Default::default()
        },
        "attention" => GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            chunk_attention: true,
            ..Default::default()
        },
        "transformer" => GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            chunk_attention: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(1),
            ..Default::default()
        },
        other => panic!(
            "Z_IMAGE_POLICY must be resident, staged, decode, attention, or transformer; got {other}"
        ),
    };
    let repeats: usize = std::env::var("Z_IMAGE_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(repeats > 0);

    let width = std::env::var("Z_IMAGE_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let height = std::env::var("Z_IMAGE_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let default_steps = if provider == "z_image" { 50 } else { 8 };
    let steps = std::env::var("Z_IMAGE_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default_steps);

    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir));
    if let Ok(path) = std::env::var("Z_IMAGE_ADAPTER") {
        spec = spec.with_adapters(vec![AdapterSpec::new(
            PathBuf::from(path),
            1.0,
            AdapterKind::Lora,
        )]);
    }
    let style_variation = std::env::var("Z_IMAGE_STYLE_VARIATION").as_deref() == Ok("1");
    let base_request = GenerationRequest {
        prompt: "a rusty robot holding a lit candle, cinematic studio lighting, highly detailed"
            .into(),
        width,
        height,
        steps: Some(steps),
        count: 1,
        seed: Some(42),
        memory: Some(memory),
        ..Default::default()
    };
    let request = GenerationRequest {
        conditioning: style_variation
            .then(|| Conditioning::Reference {
                image: reference_fixture(width, height),
                strength: Some(0.5),
            })
            .into_iter()
            .collect(),
        ..base_request.clone()
    };

    let mut probe = candle_gen::testkit::VramProbe::start_rendered();
    let handle_phase = probe.phase();
    let generator = candle_gen_z_image::provider_registry()
        .expect("registry")
        .load(&provider, &spec)
        .expect("load generator handle");
    probe.end_load(handle_phase);

    let generate_phase = probe.phase();
    let mut phase_peaks = Vec::new();
    let mut first_pixels = None;
    let mut final_image = None;
    for repeat in 0..repeats {
        let mut observed = None;
        let mut phase_index = 0usize;
        let output = generator
            .generate(&request, &mut |progress| {
                if matches!(progress, Progress::Loading(_)) {
                    if let Some((name, phase)) = observed.take() {
                        phase_peaks.push((repeat, name, probe.end_observed(phase)));
                    }
                    let name = match phase_index {
                        0 => "text",
                        1 => "denoise",
                        _ => "decode",
                    };
                    phase_index += 1;
                    observed = Some((name, probe.phase()));
                }
            })
            .expect("generate");
        if let Some((name, phase)) = observed.take() {
            phase_peaks.push((repeat, name, probe.end_observed(phase)));
        }
        let image = generated_image(output);
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width * height * 3) as usize);
        let min = *image.pixels.iter().min().unwrap();
        let max = *image.pixels.iter().max().unwrap();
        assert!(max > min + 16, "render is degenerate: [{min}, {max}]");
        if let Some(expected) = &first_pixels {
            assert_eq!(
                &image.pixels, expected,
                "fixed-seed repeat {repeat} changed output"
            );
        } else {
            first_pixels = Some(image.pixels.clone());
        }
        final_image = Some(image);
    }
    let image = final_image.unwrap();

    if style_variation {
        let plain = generated_image(
            generator
                .generate(&base_request, &mut |_| {})
                .expect("generate plain comparison for style mutation"),
        );
        let (maximum, mean) = mutation_metrics(&image, &plain);
        eprintln!(
            "ZIMAGE_MUTATION tier={tier} policy={policy_name} kind=style_reference_vs_plain max_rgb8={maximum} mean_rgb8={mean:.6}"
        );
        assert!(maximum > 0 && mean > 0.0, "style reference was ignored");
    }

    if std::env::var("Z_IMAGE_ADAPTER").is_ok() {
        drop(generator);
        let plain_generator = candle_gen_z_image::provider_registry()
            .expect("registry")
            .load(
                &provider,
                &LoadSpec::new(WeightsSource::Dir(PathBuf::from(
                    std::env::var("Z_IMAGE_TIER_DIR").expect("tier dir"),
                ))),
            )
            .expect("load unadapted comparison generator");
        let plain = generated_image(
            plain_generator
                .generate(&base_request, &mut |_| {})
                .expect("generate unadapted comparison"),
        );
        let (maximum, mean) = mutation_metrics(&image, &plain);
        eprintln!(
            "ZIMAGE_MUTATION tier={tier} policy={policy_name} kind=lora_vs_plain max_rgb8={maximum} mean_rgb8={mean:.6}"
        );
        assert!(maximum > 0 && mean > 0.0, "LoRA adapter was ignored");
    }

    probe.end_gen(generate_phase);
    let report = probe.report().assert_trustworthy(1.0);

    for (repeat, phase, peak_gb) in phase_peaks {
        eprintln!(
            "ZIMAGE_PHASE tier={tier} policy={policy_name} repeat={repeat} phase={phase} peak_gb={peak_gb:.3}"
        );
    }
    for seam in [width as usize * 3 / 8, width as usize * 3 / 4] {
        let ratio = seam_ratio(&image, seam);
        eprintln!(
            "ZIMAGE_SEAM tier={tier} policy={policy_name} x={seam} local_delta_ratio={ratio:.3}"
        );
        assert!(
            ratio < 3.0,
            "tile boundary x={seam} is an outlier (local delta ratio {ratio:.3})"
        );
    }
    eprintln!(
        "ZIMAGE_VRAM provider={provider} tier={tier} mode={} policy={policy_name} gpu={} {width}x{height} steps={steps} count=1 cold=true repeats={repeats} | {report}",
        if style_variation { "style_variations" } else { "text_to_image" },
        candle_gen::testkit::probe_gpu()
    );

    let output_path = std::env::var("Z_IMAGE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("z_image_{tier}_{policy_name}.png")));
    image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .expect("RGB geometry")
        .save(&output_path)
        .expect("write probe image");
    eprintln!("ZIMAGE_OUT {}", output_path.display());
}
