//! Real-weight active/inert preview acceptance for both public SANA generators and routes.
//!
//! Run exactly one provider/route pair per process so large Metal workloads remain serialized:
//!
//! ```sh
//! SC16635_ID=sana_1600m SC16635_ROUTE=txt2img SC16635_SNAPSHOT=/path/to/tier \
//! SC16635_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-sana --test integration preview_real_weights:: -- --ignored --nocapture
//! ```
//!
//! The acceptance tiers are pinned to `SceneWorks/Sana_1600M_1024px_mlx` revision
//! `ba22f36ba3d1feb78c9a1055a808ad68eda8adf8` and
//! `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` revision
//! `0b0d18484cac2fb515e76d25a09a5911ae4ab58e`. Their resolved DC-AE files are each
//! 1,249,044,836 bytes but have different SHA-256 digests (`15a4b09e…d9d87f` Base,
//! `dfd991d1…4454bb` Sprint), so this test deliberately exercises both provider-owned fits.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    PreviewFrame, PreviewSink, WeightsSource,
};
use mlx_gen_sana::pipeline::SPATIAL_SCALE;

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
        guidance: Some(4.5),
        negative_prompt: (id == "sana_1600m").then_some("blurry, distorted".into()),
        width: WIDTH,
        height: HEIGHT,
        steps: Some(STEPS),
        seed: Some(16635),
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
#[ignore = "needs pinned real SANA weights and a Metal-capable macOS host"]
fn public_route_emits_exact_outer_step_cadence_without_changing_output() {
    let id = required("SC16635_ID");
    let route = required("SC16635_ROUTE");
    assert!(["sana_1600m", "sana_sprint_1600m"].contains(&id.as_str()));
    assert!(["txt2img", "img2img"].contains(&route.as_str()));
    let generator = mlx_gen_sana::provider_registry()
        .unwrap()
        .load(
            &id,
            &LoadSpec::new(WeightsSource::Dir(PathBuf::from(required(
                "SC16635_SNAPSHOT",
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
            (WIDTH / SPATIAL_SCALE, HEIGHT / SPATIAL_SCALE)
        );
    }
    assert!(
        frames
            .windows(2)
            .any(|pair| pair[0].image.pixels != pair[1].image.pixels),
        "preview frames must evolve"
    );
    if let Ok(path) = std::env::var("SC16635_ARTIFACT_DIR") {
        save_strip(
            &PathBuf::from(path),
            &format!("{id}_{route}"),
            &frames,
            &active,
        );
    }
}
