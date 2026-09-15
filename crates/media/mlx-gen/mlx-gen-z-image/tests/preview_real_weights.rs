//! sc-16631 on-device acceptance for every registered Z-Image preview route.
//!
//! Run one route at a time so the two 6.7 GB control overlays can be downloaded, verified, used,
//! and removed sequentially on a space-constrained host:
//!
//! ```sh
//! ZIMAGE_PREVIEW_ROUTE=z_image_turbo ZIMAGE_TURBO_SNAPSHOT=/path/to/q4 ...
//! ZIMAGE_PREVIEW_ROUTE=z_image ZIMAGE_BASE_SNAPSHOT=/path/to/q8 ...
//! ZIMAGE_PREVIEW_ROUTE=z_image_turbo_control ZIMAGE_TURBO_SNAPSHOT=/path/to/q4 \
//!   ZIMAGE_CONTROL_WEIGHTS=/path/to/full-turbo-control.safetensors ...
//! ZIMAGE_PREVIEW_ROUTE=z_image_control ZIMAGE_BASE_SNAPSHOT=/path/to/q8 \
//!   ZIMAGE_BASE_CONTROL_WEIGHTS=/path/to/full-base-control.safetensors ...
//! ```
//!
//! The sc-16631 acceptance run used `SceneWorks/z-image-turbo-mlx` revision
//! `bb2bc9893b3c49ae96c813350775f791a2e8bc80` (`q4`) and `SceneWorks/z-image-mlx`
//! revision `c74f74c2ad193294fc9ff3f8a5be71daa00d22ab` (`q8`). Control routes used the
//! following pinned full-precision overlay artifacts (6,712,485,600 bytes each):
//!
//! - `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` revision
//!   `5155fc56d17821007d6f62ac192c09e0f0e72016`, file
//!   `Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors`, SHA-256
//!   `7f611e6d52b133f64b84bef2549fcb84589a766b8255954f96ea34684f52b633`.
//! - `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` revision
//!   `755999a934909bd5832e20718bb7c639d2a63eb9`, file
//!   `Z-Image-Fun-Controlnet-Union-2.1.safetensors`, SHA-256
//!   `2393b0c58c52a12134f6ffd96ff9b6ea3c80bb233665fb2c3b9aebcee71ae3e4`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame,
    PreviewSink, WeightsSource,
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: u32 = 3;

fn required(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn synthetic_control() -> Image {
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 3) as usize];
    for y in HEIGHT / 4..3 * HEIGHT / 4 {
        for x in WIDTH / 3..2 * WIDTH / 3 {
            let offset = ((y * WIDTH + x) * 3) as usize;
            pixels[offset..offset + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    Image {
        width: WIDTH,
        height: HEIGHT,
        pixels,
    }
}

fn request(route: &str, preview: PreviewSink) -> GenerationRequest {
    let conditioning = route.ends_with("control").then(|| Conditioning::Control {
        image: synthetic_control(),
        kind: ControlKind::Canny,
        scale: Some(0.8),
    });
    GenerationRequest {
        prompt: "a red ceramic teapot on a blue table, soft window light".into(),
        negative_prompt: (route == "z_image" || route == "z_image_control")
            .then(|| "blurry".into()),
        guidance: (route == "z_image" || route == "z_image_control").then_some(1.0),
        width: WIDTH,
        height: HEIGHT,
        steps: Some(STEPS),
        seed: Some(16631),
        conditioning: conditioning.into_iter().collect(),
        preview,
        ..Default::default()
    }
}

fn image(output: GenerationOutput) -> Image {
    match output {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.pop().unwrap()
        }
        other => panic!("expected image output, got {other:?}"),
    }
}

fn save_strip(dir: &Path, route: &str, frames: &[PreviewFrame]) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let frame_w = frames[0].image.width as usize;
    let frame_h = frames[0].image.height as usize;
    let mut strip = vec![0u8; frame_w * frames.len() * frame_h * 3];
    for (frame_index, frame) in frames.iter().enumerate() {
        assert_eq!(
            (frame.image.width as usize, frame.image.height as usize),
            (frame_w, frame_h)
        );
        for y in 0..frame_h {
            let src = y * frame_w * 3;
            let dst = (y * frame_w * frames.len() + frame_index * frame_w) * 3;
            strip[dst..dst + frame_w * 3]
                .copy_from_slice(&frame.image.pixels[src..src + frame_w * 3]);
        }
    }
    let path = dir.join(format!("{route}_strip.png"));
    image::save_buffer(
        &path,
        &strip,
        (frame_w * frames.len()) as u32,
        frame_h as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
    path
}

fn save_final(dir: &Path, route: &str, output: &Image) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{route}_final.png"));
    image::save_buffer(
        &path,
        &output.pixels,
        output.width,
        output.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    path
}

#[test]
#[ignore = "needs real Z-Image weights, exact control overlays for control routes, and Metal"]
fn registered_route_emits_exact_step_cadence_without_changing_output() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    let route = std::env::var("ZIMAGE_PREVIEW_ROUTE").expect("set ZIMAGE_PREVIEW_ROUTE");
    let snapshot = if route.starts_with("z_image_turbo") {
        required("ZIMAGE_TURBO_SNAPSHOT")
    } else {
        required("ZIMAGE_BASE_SNAPSHOT")
    };
    let control = match route.as_str() {
        "z_image_turbo_control" => Some(required("ZIMAGE_CONTROL_WEIGHTS")),
        "z_image_control" => Some(required("ZIMAGE_BASE_CONTROL_WEIGHTS")),
        "z_image_turbo" | "z_image" => None,
        _ => panic!("unknown ZIMAGE_PREVIEW_ROUTE {route}"),
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()));
    if let Some(path) = &control {
        spec = spec.with_control(WeightsSource::File(path.clone()));
    }
    let generator = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load(&route, &spec)
        .unwrap_or_else(|error| panic!("load {route}: {error}"));

    let inert = image(
        generator
            .generate(&request(&route, PreviewSink::default()), &mut |_| {})
            .unwrap(),
    );
    let captured = Arc::new(Mutex::new(Vec::<PreviewFrame>::new()));
    let captured_sink = Arc::clone(&captured);
    let active = image(
        generator
            .generate(
                &request(
                    &route,
                    PreviewSink::new(move |frame| captured_sink.lock().unwrap().push(frame)),
                ),
                &mut |_| {},
            )
            .unwrap(),
    );
    assert_eq!(
        inert, active,
        "{route}: active preview changed final RGB8 bytes"
    );

    let frames = captured.lock().unwrap();
    assert_eq!(
        frames.len(),
        STEPS as usize,
        "{route}: one frame per actual step"
    );
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, STEPS));
    }
    assert!(
        frames.windows(2).any(|pair| pair[0].image != pair[1].image),
        "{route}: preview frames are not evolving"
    );
    let dir = required("ZIMAGE_PREVIEW_ARTIFACT_DIR");
    let strip = save_strip(&dir, &route, &frames);
    let final_image = save_final(&dir, &route, &active);
    println!(
        "sc-16631 route={route} snapshot={} control={} steps={} frames=1..{}/{} exact_active_rgb8_identity=true strip={} final={}",
        snapshot.display(),
        control
            .as_ref()
            .map_or_else(|| "none".into(), |path| path.display().to_string()),
        STEPS,
        frames.len(),
        STEPS,
        strip.display(),
        final_image.display()
    );
}
