//! Real-weight active/inert preview acceptance for every public SD3.5 generator and route.
//!
//! Run exactly one provider/route pair per process so large Metal workloads remain serialized:
//!
//! ```sh
//! SC16634_ID=sd3_5_large SC16634_ROUTE=txt2img SC16634_SNAPSHOT=/path/to/snapshot \
//! SC16634_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-sd3 --test integration preview_real_weights:: -- --ignored --nocapture
//! ```
//!
//! Accepted evidence pins official SD3.5-Large revision
//! `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f`, Large-Turbo Q4 revision
//! `e9166f4632ec64f74d560be3ac778d346f89a364`, and Medium Q4 revision
//! `5413e962bb326db248be2026a93b147c323392b6`. All three VAE files are byte-identical:
//! 167,666,902 bytes, SHA-256
//! `8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    PreviewFrame, PreviewSink, WeightsSource,
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: u32 = 4;
const IMG2IMG_STRENGTH: f32 = 0.5;

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

fn reference() -> Image {
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 3) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y * WIDTH + x) * 3) as usize;
            pixels[offset] = (x * 255 / (WIDTH - 1)) as u8;
            pixels[offset + 1] = (y * 255 / (HEIGHT - 1)) as u8;
            pixels[offset + 2] = if (x / 32 + y / 32) % 2 == 0 { 192 } else { 32 };
        }
    }
    Image {
        width: WIDTH,
        height: HEIGHT,
        pixels,
    }
}

fn request(id: &str, route: &str, preview: PreviewSink) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red ceramic teapot on a blue table, soft window light".into(),
        guidance: (id != "sd3_5_large_turbo").then_some(3.5),
        negative_prompt: (id != "sd3_5_large_turbo").then_some("blurry, distorted".into()),
        width: WIDTH,
        height: HEIGHT,
        steps: Some(STEPS),
        seed: Some(16634),
        conditioning: (route == "img2img")
            .then(|| Conditioning::Reference {
                image: reference(),
                strength: Some(IMG2IMG_STRENGTH),
            })
            .into_iter()
            .collect(),
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

fn save_strip(dir: &Path, stem: &str, frames: &[PreviewFrame], final_image: &Image) {
    std::fs::create_dir_all(dir).unwrap();
    let frame_w = frames[0].image.width as usize;
    let frame_h = frames[0].image.height as usize;
    let mut strip = vec![0u8; frame_w * frames.len() * frame_h * 3];
    for (frame_index, frame) in frames.iter().enumerate() {
        for y in 0..frame_h {
            let src = y * frame_w * 3;
            let dst = (y * frame_w * frames.len() + frame_index * frame_w) * 3;
            strip[dst..dst + frame_w * 3]
                .copy_from_slice(&frame.image.pixels[src..src + frame_w * 3]);
        }
    }
    image::save_buffer(
        dir.join(format!("{stem}_strip.png")),
        &strip,
        (frame_w * frames.len()) as u32,
        frame_h as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
    image::save_buffer(
        dir.join(format!("{stem}_final.png")),
        &final_image.pixels,
        final_image.width,
        final_image.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs pinned real SD3.5 weights and a Metal-capable macOS host"]
fn public_route_emits_exact_outer_step_cadence_without_changing_output() {
    let id = required("SC16634_ID");
    let route = required("SC16634_ROUTE");
    assert!(
        ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"].contains(&id.as_str()),
        "unknown SD3 id {id}"
    );
    assert!(["txt2img", "img2img"].contains(&route.as_str()));
    let generator = mlx_gen_sd3::provider_registry()
        .unwrap()
        .load(
            &id,
            &LoadSpec::new(WeightsSource::Dir(PathBuf::from(required(
                "SC16634_SNAPSHOT",
            ))))
            .with_offload_policy(OffloadPolicy::Sequential),
        )
        .unwrap_or_else(|error| panic!("load {id}: {error}"));
    let inert = image(
        generator
            .generate(&request(&id, &route, PreviewSink::default()), &mut |_| {})
            .unwrap(),
    );
    let frames = Arc::new(Mutex::new(Vec::<PreviewFrame>::new()));
    let captured = Arc::clone(&frames);
    let active = image(
        generator
            .generate(
                &request(
                    &id,
                    &route,
                    PreviewSink::new(move |frame| captured.lock().unwrap().push(frame)),
                ),
                &mut |_| {},
            )
            .unwrap(),
    );
    assert_eq!(active, inert, "active preview changed final RGB8 output");
    let frames = frames.lock().unwrap();
    let expected = if route == "img2img" {
        STEPS - mlx_gen::img2img::init_time_step(STEPS as usize, Some(IMG2IMG_STRENGTH)) as u32
    } else {
        STEPS
    };
    assert_eq!(frames.len(), expected as usize);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, expected));
        assert_eq!(
            (frame.image.width, frame.image.height),
            (WIDTH / 8, HEIGHT / 8)
        );
    }
    assert!(
        frames
            .windows(2)
            .any(|pair| pair[0].image.pixels != pair[1].image.pixels),
        "preview frames must evolve"
    );
    if let Ok(path) = std::env::var("SC16634_ARTIFACT_DIR") {
        save_strip(
            &PathBuf::from(path),
            &format!("{id}_{route}"),
            &frames,
            &active,
        );
    }
}
