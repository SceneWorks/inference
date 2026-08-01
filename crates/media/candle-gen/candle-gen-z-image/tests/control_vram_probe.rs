//! Fresh-process CUDA calibration for the eager Z-Image base-control provider (SC-16170).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{GenerationMemory, Image, Progress};
use candle_gen_z_image::{ZImageControl, ZImageControlPaths, ZImageControlRequest};

fn assert_coherent(image: &Image, tier: &str) {
    assert_eq!(
        image.pixels.len(),
        (image.width * image.height * 3) as usize
    );
    let min = *image.pixels.iter().min().unwrap();
    let max = *image.pixels.iter().max().unwrap();
    let mean = image
        .pixels
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>()
        / image.pixels.len() as f64;
    let variance = image
        .pixels
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / image.pixels.len() as f64;
    assert!(max > min + 16, "{tier} control render is near-constant");
    assert!(
        variance.sqrt() > 8.0,
        "{tier} control render has insufficient pixel spread"
    );
}

fn control_fixture(width: u32, height: u32) -> Image {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let mut set = |x: u32, y: u32| {
        let offset = ((y * width + x) * 3) as usize;
        pixels[offset..offset + 3].fill(255);
    };
    for y in height / 8..height * 7 / 8 {
        for dx in 0..3 {
            set(width / 2 + dx, y);
        }
    }
    for x in width / 4..width * 3 / 4 {
        for dy in 0..3 {
            set(x, height / 3 + dy);
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_TIER + Z_IMAGE_BASE_CONTROL + CUDA; run in a fresh process"]
fn measure_z_image_base_control_tier() {
    let snapshot = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_TIER").expect("set Z_IMAGE_BASE_TIER to the tier directory"),
    );
    let control = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_CONTROL")
            .expect("set Z_IMAGE_BASE_CONTROL to the control checkpoint"),
    );
    let tier = std::env::var("Z_IMAGE_TIER_NAME").unwrap_or_else(|_| "unknown".into());
    let width = std::env::var("Z_IMAGE_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let height = std::env::var("Z_IMAGE_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let steps = std::env::var("Z_IMAGE_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(50);
    let repeats = std::env::var("Z_IMAGE_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2usize);
    let strategy = std::env::var("Z_IMAGE_MEMORY").unwrap_or_else(|_| "resident".into());
    let memory = match strategy.as_str() {
        "resident" => GenerationMemory::default(),
        "staged" => GenerationMemory {
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
        other => panic!("unknown Z_IMAGE_MEMORY strategy {other}"),
    };

    let mut probe = candle_gen::testkit::VramProbe::start_rendered();
    let load_phase = probe.phase();
    let paths = ZImageControlPaths {
        snapshot,
        control,
        base: true,
    };
    let model = ZImageControl::load_with_memory(&paths, memory)
        .expect("load Z-Image base-control provider");
    probe.end_load(load_phase);

    let control_image = control_fixture(width, height);
    let request = ZImageControlRequest {
        prompt: "a studio photograph of a dancer, full body, crisp details".into(),
        width,
        height,
        steps,
        control_scale: 1.0,
        guidance: Some(4.0),
        negative_prompt: Some("blurry, malformed".into()),
        seed: 16170,
        use_pid: false,
        memory,
        cancel: candle_gen::gen_core::CancelFlag::new(),
    };
    let mut reference = None;
    for repeat in 0..repeats {
        let generate_phase = probe.phase();
        let mut predecode_phase = Some(probe.phase());
        let mut predecode_peak = None;
        let mut decode_phase = None;
        let image = model
            .generate(&request, &control_image, &mut |progress| {
                if matches!(progress, Progress::Decoding) {
                    predecode_peak = predecode_phase
                        .take()
                        .map(|phase| probe.end_observed(phase));
                    decode_phase = Some(probe.phase());
                }
            })
            .expect("base-control generate");
        if let Some(phase) = predecode_phase.take() {
            predecode_peak = Some(probe.end_observed(phase));
        }
        let decode_peak = decode_phase.map(|phase| probe.end_observed(phase));
        probe.end_gen(generate_phase);
        eprintln!(
            "ZIMAGE_CONTROL_PHASE tier={tier} strategy={strategy} repeat={repeat} phase=predecode peak_gb={:.3}",
            predecode_peak.unwrap()
        );
        if let Some(decode_peak) = decode_peak {
            eprintln!(
                "ZIMAGE_CONTROL_PHASE tier={tier} strategy={strategy} repeat={repeat} phase=decode peak_gb={decode_peak:.3}"
            );
        }
        assert_coherent(&image, &tier);
        if let Some(expected) = &reference {
            assert_eq!(&image, expected, "fixed-seed warm control repeat changed");
        } else {
            reference = Some(image);
        }
    }

    let report = probe.report().assert_trustworthy(1.0);
    eprintln!(
        "ZIMAGE_CONTROL_VRAM tier={tier} strategy={strategy} gpu={} {width}x{height} steps={steps} guidance=4 control_scale=1 count=1 cold=true repeats={repeats} | {report}",
        candle_gen::testkit::probe_gpu()
    );
    let image = reference.unwrap();
    let output_path = std::env::var("Z_IMAGE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!("z_image_control_{tier}_{strategy}.png"))
        });
    image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .expect("RGB geometry")
        .save(&output_path)
        .expect("write probe image");
    eprintln!("ZIMAGE_CONTROL_OUT {}", output_path.display());
}
