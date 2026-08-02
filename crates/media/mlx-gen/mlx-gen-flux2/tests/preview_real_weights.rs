//! Real-weight active-preview acceptance for FLUX.2 Klein.
//!
//! ```sh
//! MLX_GEN_FLUX2_SNAPSHOT=/path/to/snapshot \
//! FLUX2_PREVIEW_ARTIFACT_DIR=/path/to/output \
//! cargo test --locked --release -p mlx-gen-flux2 --test preview_real_weights \
//!   -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, PreviewSink, WeightsSource};

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("MLX_GEN_FLUX2_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set MLX_GEN_FLUX2_SNAPSHOT to the required real FLUX.2 snapshot")
    }))
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights and a Metal-capable macOS host"]
fn active_preview_emits_exact_outer_step_strip_without_changing_rgb() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    let generator = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(
            mlx_gen_flux2::FLUX2_KLEIN_9B_ID,
            &LoadSpec::new(WeightsSource::Dir(snapshot())),
        )
        .unwrap();

    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
    let active_request = GenerationRequest {
        prompt: "a red fox beside a turquoise alpine lake at golden hour".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(1663088),
        steps: Some(8),
        preview: sink,
        ..Default::default()
    };
    let active = generator.generate(&active_request, &mut |_| {}).unwrap();
    let GenerationOutput::Images(active_images) = active else {
        panic!("expected active image output")
    };

    let inert_request = GenerationRequest {
        prompt: active_request.prompt.clone(),
        width: active_request.width,
        height: active_request.height,
        count: active_request.count,
        seed: active_request.seed,
        steps: active_request.steps,
        ..Default::default()
    };
    let inert = generator.generate(&inert_request, &mut |_| {}).unwrap();
    let GenerationOutput::Images(inert_images) = inert else {
        panic!("expected inert image output")
    };
    assert_eq!(
        active_images, inert_images,
        "active PreviewSink changed final RGB8 bytes"
    );

    let frames = frames.lock().unwrap();
    assert_eq!(
        frames.len(),
        8,
        "one frame is required per outer Euler step"
    );
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, 8));
        assert_eq!((frame.image.width, frame.image.height), (32, 32));
    }

    if let Ok(path) = std::env::var("FLUX2_PREVIEW_ARTIFACT_DIR") {
        let dir = PathBuf::from(path);
        std::fs::create_dir_all(&dir).unwrap();
        let frame_width = frames[0].image.width as usize;
        let frame_height = frames[0].image.height as usize;
        let mut strip = vec![0u8; frame_width * frames.len() * frame_height * 3];
        for (frame_index, frame) in frames.iter().enumerate() {
            for y in 0..frame_height {
                let src = y * frame_width * 3;
                let dst = (y * frame_width * frames.len() + frame_index * frame_width) * 3;
                strip[dst..dst + frame_width * 3]
                    .copy_from_slice(&frame.image.pixels[src..src + frame_width * 3]);
            }
        }
        image::save_buffer(
            dir.join("flux2_preview_strip.png"),
            &strip,
            (frame_width * frames.len()) as u32,
            frame_height as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();
        image::save_buffer(
            dir.join("flux2_preview_final.png"),
            &active_images[0].pixels,
            active_images[0].width,
            active_images[0].height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
}
