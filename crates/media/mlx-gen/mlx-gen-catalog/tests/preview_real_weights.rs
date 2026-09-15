//! sc-16632 on-device acceptance for every registered FLUX.1-family preview route.
//!
//! Run exactly one route per process so model loads and task-owned downloads remain serialized:
//!
//! ```sh
//! SC16632_ROUTE=flux1_schnell SC16632_SNAPSHOT=/path/to/flux1-schnell/q4 ...
//! SC16632_ROUTE=flux1_dev SC16632_SNAPSHOT=/path/to/flux1-dev/q4 ...
//! SC16632_ROUTE=flux1_dev_control SC16632_SNAPSHOT=/path/to/flux1-dev/q4 \
//!   SC16632_CONTROL=/path/to/control.safetensors ...
//! SC16632_ROUTE=chroma1_hd SC16632_SNAPSHOT=/path/to/chroma-hd/q4 ...
//! SC16632_ROUTE=chroma1_base SC16632_SNAPSHOT=/path/to/chroma-base/q4 ...
//! SC16632_ROUTE=chroma1_flash SC16632_SNAPSHOT=/path/to/chroma-flash/q4 ...
//! SC16632_ROUTE=pulid_flux SC16632_SNAPSHOT=/path/to/flux1-dev/q4 \
//!   SC16632_ID_ENCODER=/path/to/pulid.safetensors SC16632_EVA=/path/to/eva.safetensors \
//!   SC16632_FACE_DIR=/path/to/face-dir SC16632_REFERENCE=/path/to/face.png ...
//! ```
//!
//! The accepted run pinned FLUX.1-dev Q4 to `SceneWorks/flux1-dev-mlx` revision
//! `323fd12d79f78ad444e882e8d8e871914584f2b9`, official FLUX.1-schnell to revision
//! `741f7c3ce8b383c54771c7003378a50191e9efe9`, and the Shakker Union-Pro 2.0 control overlay to
//! revision `5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5` (4,281,779,224 bytes, SHA-256
//! `9d03f63f36206bab2f36aed5cfedc8693c2881397534e9d5f9ae9a0a41362517`). Chroma revisions and
//! their exact official-FLUX.1 VAE lineage are recorded in `mlx_gen_flux::preview`.
//!
//! PuLID used `guozinan/PuLID` revision `492b1451255dc9d9bc3c857259690b5f8b998d4a`, the converted
//! EVA/face components at revisions `78ef91f977eae16d66fb191caf003154b7a0a0b8` and
//! `bca0cacf8e5e04529bb2b326a521361b02be84fd`, plus upstream PuLID's `lifeifei.jpg` at commit
//! `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa` (SHA-256
//! `c7b26e78b94ccd8c30a40efca3c52c8a04573188b41eb4b3d1fc517ec8577b35`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, IdentityWeights, Image,
    LoadSpec, PreviewFrame, PreviewSink, WeightsSource,
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

fn reference() -> Image {
    let decoded = image::open(required("SC16632_REFERENCE"))
        .expect("decode reference face")
        .to_rgb8();
    Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    }
}

fn request(route: &str, preview: PreviewSink) -> GenerationRequest {
    let conditioning = match route {
        "flux1_dev_control" => vec![Conditioning::Control {
            image: synthetic_control(),
            kind: ControlKind::Canny,
            scale: Some(0.7),
        }],
        "pulid_flux" => vec![Conditioning::Reference {
            image: reference(),
            strength: Some(0.8),
        }],
        _ => vec![],
    };
    let true_cfg = (route == "pulid_flux"
        && std::env::var("SC16632_PULID_TRUE_CFG").as_deref() == Ok("1"))
    .then_some(2.0);
    GenerationRequest {
        prompt: if route == "pulid_flux" {
            "a portrait photo of a person, headshot, looking at the camera".into()
        } else {
            "a red ceramic teapot on a blue table, soft window light".into()
        },
        negative_prompt: true_cfg.map(|_| "blurry, distorted".into()),
        guidance: matches!(route, "flux1_dev" | "flux1_dev_control" | "pulid_flux").then_some(1.0),
        true_cfg: if route.starts_with("chroma1_") {
            Some(1.0)
        } else {
            true_cfg
        },
        width: WIDTH,
        height: HEIGHT,
        steps: Some(STEPS),
        seed: Some(16632),
        conditioning,
        preview,
        ..Default::default()
    }
}

fn load_spec(route: &str) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(required("SC16632_SNAPSHOT")));
    if route == "flux1_dev_control" {
        spec = spec.with_control(WeightsSource::File(required("SC16632_CONTROL")));
    }
    if route == "pulid_flux" {
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File(required("SC16632_ID_ENCODER"))),
            eva: Some(WeightsSource::File(required("SC16632_EVA"))),
            face_dir: Some(WeightsSource::Dir(required("SC16632_FACE_DIR"))),
        });
    }
    spec
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

fn save_strip(dir: &Path, stem: &str, frames: &[PreviewFrame]) {
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
    image::save_buffer(
        dir.join(format!("{stem}_strip.png")),
        &strip,
        (frame_w * frames.len()) as u32,
        frame_h as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs pinned real provider weights, representative inputs, and Metal"]
fn registered_route_emits_exact_step_cadence_without_changing_output() {
    let route = std::env::var("SC16632_ROUTE").expect("set SC16632_ROUTE");
    assert!(
        [
            "flux1_schnell",
            "flux1_dev",
            "flux1_dev_control",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "pulid_flux",
        ]
        .contains(&route.as_str()),
        "unknown sc-16632 route {route}"
    );
    let generator = mlx_gen_catalog::provider_registry()
        .unwrap()
        .load(&route, &load_spec(&route))
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
    assert_eq!(active, inert, "active preview changed final RGB8 output");
    let frames = captured.lock().unwrap();
    assert_eq!(
        frames.len(),
        STEPS as usize,
        "one frame per real outer solver step"
    );
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.current, frame.total), (index as u32 + 1, STEPS));
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

    if let Ok(path) = std::env::var("SC16632_ARTIFACT_DIR") {
        let dir = PathBuf::from(path);
        let stem = if route == "pulid_flux" {
            format!(
                "{route}_{}",
                if std::env::var("SC16632_PULID_TRUE_CFG").as_deref() == Ok("1") {
                    "true_cfg"
                } else {
                    "fake_cfg"
                }
            )
        } else {
            route.clone()
        };
        save_strip(&dir, &stem, &frames);
        std::fs::create_dir_all(&dir).unwrap();
        image::save_buffer(
            dir.join(format!("{stem}_final.png")),
            &active.pixels,
            active.width,
            active.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
}
