//! Prove FLUX.2-dev img2img: load an init PNG, refine it with a new prompt at a given strength via
//! `Conditioning::Reference`. Run from the workspace root:
//!   FLUX2_DEV_SNAPSHOT=~/Models/aether/flux2-dev-q8 \
//!   FLUX2_IMG2IMG_INIT=/tmp/flux2_dev_q8_proof.png \
//!     cargo run --release --example flux2_dev_img2img -p mlx-gen-flux2
use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen_flux2::load_dev;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Decode a PNG/JPEG file into a gen-core RGB `Image`.
fn load_image(path: &str) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open init {path}: {e}"))
        .to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

fn save(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone()).expect("buf");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or(
        "FLUX2_DEV_SNAPSHOT",
        "/Users/zakkeown/Models/aether/flux2-dev-q8",
    );
    let init_path = env_or("FLUX2_IMG2IMG_INIT", "/tmp/flux2_dev_q8_proof.png");
    let prompt = env_or(
        "FLUX2_IMG2IMG_PROMPT",
        "the same weathered man, now smiling warmly, bright sunny day, clear blue sky, calm sea",
    );
    let strength: f32 = env_or("FLUX2_IMG2IMG_STRENGTH", "0.62")
        .parse()
        .expect("strength");
    let steps: u32 = env_or("FLUX2_IMG2IMG_STEPS", "14").parse().expect("steps");
    let out_path = env_or("FLUX2_IMG2IMG_OUT", "/tmp/flux2_img2img_proof.png");

    let init = load_image(&init_path);
    eprintln!(
        "[img2img] init {}x{} from {init_path}; strength={strength} steps={steps}",
        init.width, init.height
    );

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)))
        .with_offload_policy(OffloadPolicy::Sequential);
    let gen = load_dev(&spec).expect("load flux2_dev");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(7),
        steps: Some(steps),
        conditioning: vec![Conditioning::Reference {
            image: init,
            strength: Some(strength),
        }],
        ..Default::default()
    };
    gen.validate(&req).expect("validate img2img request");

    eprintln!("[img2img] '{prompt}'");
    let t = Instant::now();
    let out = gen.generate(&req, &mut |_| eprint!(".")).expect("generate");
    eprintln!("\n[img2img] done in {:.0}s", t.elapsed().as_secs_f32());

    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().expect("one image"),
        _ => panic!("non-image"),
    };
    let (mn, mx) = (
        *img.pixels.iter().min().unwrap(),
        *img.pixels.iter().max().unwrap(),
    );
    assert!(mx > mn, "degenerate output");
    save(&img, &out_path);
    eprintln!("[img2img] wrote {out_path} ({}x{})", img.width, img.height);
}
