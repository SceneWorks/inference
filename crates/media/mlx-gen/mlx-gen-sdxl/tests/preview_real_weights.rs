//! Real-weight acceptance producer for registered SDXL previews.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, PreviewSink, WeightsSource};

fn save_evidence(frames: &[mlx_gen::PreviewFrame], final_image: &mlx_gen::Image) {
    let Ok(root) = std::env::var("SDXL_PREVIEW_ARTIFACT_DIR") else {
        return;
    };
    let root = PathBuf::from(root);
    std::fs::create_dir_all(&root).unwrap();
    for frame in frames {
        image::save_buffer(
            root.join(format!("sdxl_frame_{:02}.png", frame.current)),
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
    let width = frames[0].image.width;
    let height = frames[0].image.height;
    let mut strip = vec![0u8; (width * frames.len() as u32 * height * 3) as usize];
    for (index, frame) in frames.iter().enumerate() {
        for row in 0..height as usize {
            let source =
                &frame.image.pixels[row * width as usize * 3..(row + 1) * width as usize * 3];
            let destination_width = width as usize * frames.len() * 3;
            let start = row * destination_width + index * width as usize * 3;
            strip[start..start + source.len()].copy_from_slice(source);
        }
    }
    image::save_buffer(
        root.join("sdxl_preview_strip.png"),
        &strip,
        width * frames.len() as u32,
        height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    image::save_buffer(
        root.join("sdxl_final.png"),
        &final_image.pixels,
        final_image.width,
        final_image.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs real SDXL weights and a Metal-capable macOS host"]
fn registered_ancestral_route_emits_one_numbered_frame_per_step() {
    let snapshot = PathBuf::from(std::env::var("SDXL_SNAPSHOT").expect("set SDXL_SNAPSHOT"));
    let generator = mlx_gen_sdxl::provider_registry()
        .unwrap()
        .load("sdxl", &LoadSpec::new(WeightsSource::Dir(snapshot)))
        .unwrap();
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let request = GenerationRequest {
        prompt: "a glass greenhouse in a snowy pine forest at sunrise, warm light and cool shadows"
            .into(),
        width: 512,
        height: 512,
        steps: Some(8),
        guidance: Some(5.0),
        seed: Some(16633),
        preview: PreviewSink::new(move |frame| captured.lock().unwrap().push(frame)),
        ..Default::default()
    };
    let output = generator.generate(&request, &mut |_| {}).unwrap();
    let final_image = match output {
        GenerationOutput::Images(images) => images.into_iter().next().unwrap(),
        other => panic!("expected image output, got {other:?}"),
    };
    let frames = frames.lock().unwrap();
    assert_eq!(frames.len(), 8);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, 8));
        assert_eq!((frame.image.width, frame.image.height), (64, 64));
    }
    save_evidence(&frames, &final_image);
}

#[test]
#[ignore = "needs real SDXL weights and a Metal-capable macOS host"]
fn registered_curated_route_keeps_the_first_frame_off_the_rails() {
    let snapshot = PathBuf::from(std::env::var("SDXL_SNAPSHOT").expect("set SDXL_SNAPSHOT"));
    let generator = mlx_gen_sdxl::provider_registry()
        .unwrap()
        .load("sdxl", &LoadSpec::new(WeightsSource::Dir(snapshot)))
        .unwrap();
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let request = GenerationRequest {
        prompt: "a glass greenhouse in a snowy pine forest at sunrise, warm light and cool shadows"
            .into(),
        width: 512,
        height: 512,
        steps: Some(8),
        sampler: Some("euler".into()),
        guidance: Some(5.0),
        seed: Some(17181),
        preview: PreviewSink::new(move |frame| captured.lock().unwrap().push(frame)),
        ..Default::default()
    };
    let output = generator.generate(&request, &mut |_| {}).unwrap();
    let final_image = match output {
        GenerationOutput::Images(images) => images.into_iter().next().unwrap(),
        other => panic!("expected image output, got {other:?}"),
    };
    let frames = frames.lock().unwrap();
    assert_eq!(frames.len(), 8);
    let first = &frames[0].image.pixels;
    let rail_fraction = first
        .iter()
        .filter(|&&value| value == 0 || value == 255)
        .count() as f32
        / first.len() as f32;
    eprintln!("[sc-17181] corrected curated first-frame rail fraction: {rail_fraction:.6}");
    assert!(
        rail_fraction < 0.10,
        "corrected curated first frame still clips {rail_fraction:.3} of RGB values"
    );
    save_evidence(&frames, &final_image);
}
