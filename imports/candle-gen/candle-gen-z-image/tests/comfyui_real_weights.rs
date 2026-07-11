//! Real-weight GPU validation of the in-place ComfyUI Z-Image load (epic 10451 Phase 2, sc-10668).
//!
//! Renders a Z-Image image straight from a user's ComfyUI single-file components — the bf16 DiT, the
//! Qwen3 text encoder, and the BFL/ldm VAE — read in place via
//! [`candle_gen_z_image::load_from_comfyui_components`]. This exercises the whole seam end-to-end on a
//! real GPU: the DiT fused-qkv split + renames, the VAE ldm→diffusers remap (up-block reversal +
//! 1×1-conv→Linear squeeze), and the verbatim Qwen3 encoder. A coherent (non-degenerate) render proves
//! the key/shape remaps are correct — a wrong remap decodes to noise, not an error.
//!
//! Ignored (needs the real files + a resident tokenizer snapshot + a CUDA GPU). Run:
//! ```text
//! cargo test -p candle-gen-z-image --release --features cuda --test comfyui_real_weights -- --ignored --nocapture
//! ```
//! Override paths via `COMFYUI_ROOT` (the ComfyUI `models/` dir) and `ZIMAGE_TOKENIZER_DIR` (a diffusers
//! Z-Image snapshot with `tokenizer/tokenizer.json`).

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{GenerationOutput, GenerationRequest, Image, Progress};

/// A non-degenerate render: the byte buffer is the declared size, not uniform, and has real spatial
/// variance (a mis-remapped DiT/VAE decodes to a flat or noise field).
fn assert_coherent(image: &Image, tag: &str) {
    let pixels = &image.pixels;
    assert_eq!(
        pixels.len() as u32,
        image.width * image.height * 3,
        "{tag}: buffer size mismatch"
    );
    assert!(
        pixels.iter().any(|&p| p != pixels[0]),
        "{tag}: uniform image — the remap produced a flat field"
    );
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / pixels.len() as f64;
    let var = pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;
    assert!(
        var.sqrt() > 10.0,
        "{tag}: pixel std {:.1} too low — degenerate render",
        var.sqrt()
    );
}

#[test]
#[ignore = "needs the ComfyUI Z-Image files + a resident tokenizer snapshot + a CUDA GPU; run --release --features cuda --ignored"]
fn comfyui_zimage_renders_in_place() {
    candle_gen_z_image::force_link();

    let root = PathBuf::from(
        std::env::var("COMFYUI_ROOT")
            .unwrap_or_else(|_| r"C:\Users\Michael\ComfyUI-Shared\models".to_owned()),
    );
    let tokenizer_dir = PathBuf::from(std::env::var("ZIMAGE_TOKENIZER_DIR").unwrap_or_else(|_| {
        r"D:\.cache\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\f332072aa78be7aecdf3ee76d5c247082da564a6".to_owned()
    }));

    let transformer = root.join("unet").join("z_image_turbo_bf16.safetensors");
    let text_encoder = root.join("text_encoders").join("qwen_3_4b.safetensors");
    let vae = root.join("vae").join("ae.safetensors");
    for (label, path) in [
        ("transformer", &transformer),
        ("text_encoder", &text_encoder),
        ("vae", &vae),
        ("tokenizer_dir", &tokenizer_dir),
    ] {
        assert!(path.exists(), "{label} missing: {}", path.display());
    }

    let load = Instant::now();
    let gen = candle_gen_z_image::load_from_comfyui_components(
        &transformer,
        &text_encoder,
        &vae,
        &tokenizer_dir,
    )
    .expect("load_from_comfyui_components");
    eprintln!("[comfyui] load handle {:.2}s", load.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };

    let render = Instant::now();
    let mut on_progress = |_p: Progress| {};
    let out = gen
        .generate(&req, &mut on_progress)
        .expect("comfyui z-image generate");
    eprintln!(
        "[comfyui] first render (incl. lazy component load) {:.1}s",
        render.elapsed().as_secs_f32()
    );

    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => panic!("expected images, got video"),
    };
    assert_eq!(images.len(), 1, "expected 1 image");
    assert_coherent(&images[0], "comfyui");

    if let Some(buf) =
        image::RgbImage::from_raw(images[0].width, images[0].height, images[0].pixels.clone())
    {
        let out_path = std::env::temp_dir().join("z_image_comfyui_inplace.png");
        let _ = buf.save(&out_path);
        eprintln!("[comfyui] wrote {}", out_path.display());
    }
}
