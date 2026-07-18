//! Prove Ideogram-4 inpaint: take a source image + a drawn mask (white = regenerate) and regenerate
//! just that region from a prompt (Reference source + Mask). Run from the workspace root:
//!   IDEOGRAM_Q8=~/Models/aether/ideogram-4-q8 SRC=/tmp/flux2_dev_q8_proof.png \
//!     cargo run --release --example ideogram_inpaint -p mlx-gen-ideogram
use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};
use mlx_gen::media::Image;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn load_img(path: &str) -> Image {
    let rgb = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

/// A mask whose TOP `frac` of rows is white (regenerate — the sky/background) and the rest black.
fn top_mask(w: u32, h: u32, frac: f32) -> Image {
    let cut = (h as f32 * frac) as u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..cut {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = 255;
            pixels[i + 1] = 255;
            pixels[i + 2] = 255;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn save(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone()).expect("buf");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or("IDEOGRAM_Q8", "/Users/zakkeown/Models/aether/ideogram-4-q8");
    let src_path = env_or("SRC", "/tmp/flux2_dev_q8_proof.png");
    let prompt = env_or(
        "PROMPT",
        "a dramatic sunset sky with vivid orange and purple clouds",
    );
    let steps: u32 = env_or("STEPS", "20").parse().expect("steps");

    let source = load_img(&src_path);
    let mask = top_mask(source.width, source.height, 0.40);
    save(&mask, "/tmp/ideogram_mask.png");
    eprintln!(
        "[inpaint] source {}x{} from {src_path}; mask=top 40%; steps={steps}",
        source.width, source.height
    );

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)))
        .with_offload_policy(OffloadPolicy::Sequential);
    let gen = mlx_gen_ideogram::provider_registry()
        .expect("registry")
        .load("ideogram_4", &spec)
        .expect("load ideogram_4");

    let (w, h) = (source.width, source.height);
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(3),
        steps: Some(steps),
        conditioning: vec![
            Conditioning::Reference {
                image: source,
                strength: None,
            },
            Conditioning::Mask { image: mask },
        ],
        ..Default::default()
    };
    gen.validate(&req).expect("validate inpaint");

    eprintln!("[inpaint] '{prompt}'");
    let t = Instant::now();
    let out = gen.generate(&req, &mut |_| eprint!(".")).expect("generate");
    eprintln!("\n[inpaint] done in {:.0}s", t.elapsed().as_secs_f32());
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().expect("one image"),
        _ => panic!("non-image"),
    };
    save(&img, "/tmp/ideogram_inpaint_proof.png");
    eprintln!(
        "[inpaint] wrote /tmp/ideogram_inpaint_proof.png ({}x{})",
        img.width, img.height
    );
}
