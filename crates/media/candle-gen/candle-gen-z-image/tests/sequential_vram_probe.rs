//! Fresh-process CUDA measurement harness for SC-15256.
//!
//! Run once per hosted Q4/Q8/BF16 tier and policy. Sequential runs emit physical text, denoise, and
//! decode peaks from the provider's `Progress::Loading` boundaries; every run records the overall
//! `VramProbe` report and writes the fixed-seed image for resident-vs-sequential comparison.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};

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

#[test]
#[ignore = "needs Z_IMAGE_TIER_DIR + CUDA; run in a fresh otherwise-idle process"]
fn measure_z_image_tier() {
    let tier_dir = PathBuf::from(
        std::env::var("Z_IMAGE_TIER_DIR")
            .expect("set Z_IMAGE_TIER_DIR to a hosted q4/q8/bf16 tier snapshot"),
    );
    let tier = std::env::var("Z_IMAGE_TIER_NAME").unwrap_or_else(|_| "unknown".into());
    let policy_name = std::env::var("Z_IMAGE_POLICY").unwrap_or_else(|_| "sequential".into());
    let policy = match policy_name.as_str() {
        "resident" => OffloadPolicy::Resident,
        "sequential" => OffloadPolicy::Sequential,
        other => panic!("Z_IMAGE_POLICY must be resident or sequential, got {other}"),
    };
    let repeats: usize = std::env::var("Z_IMAGE_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(repeats > 0);

    let spec = LoadSpec::new(WeightsSource::Dir(tier_dir)).with_offload_policy(policy);
    let request = GenerationRequest {
        prompt: "a rusty robot holding a lit candle, cinematic studio lighting, highly detailed"
            .into(),
        width: 1024,
        height: 1024,
        steps: Some(8),
        count: 1,
        seed: Some(42),
        ..Default::default()
    };

    let mut probe = candle_gen::testkit::VramProbe::start_rendered();
    let handle_phase = probe.phase();
    let generator = candle_gen_z_image::provider_registry()
        .expect("registry")
        .load("z_image_turbo", &spec)
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
        let image = match output {
            GenerationOutput::Images(mut images) => {
                assert_eq!(images.len(), 1);
                images.remove(0)
            }
            other => panic!("expected image output, got {other:?}"),
        };
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.pixels.len(), 1024 * 1024 * 3);
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
    probe.end_gen(generate_phase);
    let report = probe.report().assert_trustworthy(1.0);
    let image = final_image.unwrap();

    for (repeat, phase, peak_gb) in phase_peaks {
        eprintln!(
            "ZIMAGE_PHASE tier={tier} policy={policy_name} repeat={repeat} phase={phase} peak_gb={peak_gb:.3}"
        );
    }
    for seam in [384usize, 768] {
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
        "ZIMAGE_VRAM tier={tier} policy={policy_name} gpu={} 1024x1024 steps=8 count=1 cold=true repeats={repeats} | {report}",
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
