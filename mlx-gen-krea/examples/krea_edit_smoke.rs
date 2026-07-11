//! Real-weight smoke for the Krea 2 dual-conditioned image-edit path (epic 10871 P2.3, sc-10881).
//!
//! Loads the cached Krea 2 Raw diffusers snapshot + the community `krea2-identity-edit` LoRA, runs a
//! dual-conditioned edit (in-context VAE tokens + Qwen3-VL grounding) on a source image, and writes the
//! result. This is a MANUAL on-Metal validation (a 12.9B model), NOT a CI test. Paths default to the
//! local HF cache; override via env (`KREA_SNAPSHOT`, `KREA_EDIT_LORA`, `KREA_EDIT_SOURCE`,
//! `KREA_EDIT_INSTRUCTION`, `KREA_EDIT_OUT`, `KREA_EDIT_STEPS`, `KREA_EDIT_GUIDANCE`).
//!
//! Run: `cargo run --release --example krea_edit_smoke -p mlx-gen-krea`

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_krea::pipeline::{KreaPipeline, TurboOptions};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn load_rgb(path: &str) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open source image {path}: {e}"))
        .to_rgb8();
    let (width, height) = img.dimensions();
    Image {
        width,
        height,
        pixels: img.into_raw(),
    }
}

fn save_png(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
            .expect("output image buffer");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or(
        "KREA_SNAPSHOT",
        "/Users/michael/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494",
    );
    let lora = env_or(
        "KREA_EDIT_LORA",
        "/Users/michael/.cache/huggingface/hub/models--conradlocke--krea2-identity-edit/snapshots/8f3856364fcee7db52116f72558fce0c233eaac4/krea2_identity_edit_v1_1_r128.safetensors",
    );
    let source = env_or(
        "KREA_EDIT_SOURCE",
        "/Users/michael/.cache/huggingface/hub/models--conradlocke--krea2-identity-edit/snapshots/8f3856364fcee7db52116f72558fce0c233eaac4/showcase/release_1.png",
    );
    let instruction = env_or(
        "KREA_EDIT_INSTRUCTION",
        "change the background to a snowy mountain landscape",
    );
    let out_path = env_or("KREA_EDIT_OUT", "/tmp/krea_edit_out.png");
    let steps: usize = env_or("KREA_EDIT_STEPS", "16").parse().expect("steps");
    let guidance: f32 = env_or("KREA_EDIT_GUIDANCE", "3.0")
        .parse()
        .expect("guidance");

    eprintln!("[smoke] loading snapshot {snapshot}");
    let mut pipe = KreaPipeline::from_snapshot(&snapshot).expect("load pipeline");
    eprintln!("[smoke] applying edit LoRA {lora}");
    pipe.apply_adapters(&[AdapterSpec::new(
        PathBuf::from(&lora),
        1.0,
        AdapterKind::Lora,
    )])
    .expect("apply lora");

    let src = load_rgb(&source);
    eprintln!(
        "[smoke] source {}x{} → edit '{instruction}' ({steps} steps, g={guidance})",
        src.width, src.height
    );
    let opts = TurboOptions {
        width: 1024,
        height: 1024,
        steps,
        seed: 42,
        sampler: None,
        scheduler: None,
    };
    let out = pipe
        .generate_edit(&instruction, "", guidance, &src, &opts)
        .expect("generate edit");

    // Basic sanity: a non-degenerate (non-constant) image.
    let mn = *out.pixels.iter().min().unwrap();
    let mx = *out.pixels.iter().max().unwrap();
    let mean: f64 = out.pixels.iter().map(|&p| p as f64).sum::<f64>() / out.pixels.len() as f64;
    eprintln!(
        "[smoke] output {}x{} px range [{mn},{mx}] mean {mean:.1}",
        out.width, out.height
    );
    assert!(mx > mn, "degenerate (constant) output image");
    save_png(&out, &out_path);
    eprintln!("[smoke] wrote {out_path}");
}
