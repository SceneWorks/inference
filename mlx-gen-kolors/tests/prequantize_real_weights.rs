//! sc-9946 (epic 8506): maintainer's on-device proof that a **pre-quantized packed** Kolors tier built
//! by [`mlx_gen_kolors::convert::prequantize_turnkey`] loads directly via the packed-detect loaders
//! (the ChatGLM3 [`mlx_gen_kolors::chatglm3`] packed-detect + the SDXL packed-aware U-Net loader) and
//! renders a coherent image — no dense fp16 transient, no in-app `.quantize`. This render is the
//! completeness gate for the packed path: a missed quantized site loads u32 codes as dense floats → a
//! degenerate (flat) render, which the pixel-range assertion catches.
//!
//! Kolors packs **two** components (the ChatGLM3-6B text encoder + the SDXL U-Net); the VAE stays
//! dense (never quantized). The tokenizer is baked (`KOLORS_TOKENIZER_JSON` → the derived
//! `SceneWorks/kolors-chatglm3-tokenizer` fast tokenizer.json; ChatGLM3 ships slow-only upstream).
//!
//! `#[ignore]`d — needs a real `Kwai-Kolors/Kolors-diffusers` snapshot. Run per tier:
//!   KOLORS_SNAPSHOT=<snap> KOLORS_BITS=4 KOLORS_TOKENIZER_JSON=<tok.json> \
//!     cargo test -p mlx-gen-kolors --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: KOLORS_SNAPSHOT (source snapshot dir; else the newest HF-cache snapshot), KOLORS_OUT
//! (tier output dir), KOLORS_BITS (4 default / 8), KOLORS_TOKENIZER_JSON (derived fast tokenizer.json;
//! optional if the source `tokenizer/` already has one), KOLORS_KEEP (retain the built tier).

mod common;

use mlx_gen_kolors::Kolors;
use mlx_rs::Dtype;
use std::path::{Path, PathBuf};

/// The derived fast tokenizer.json override, if the source `tokenizer/` lacks one (raw HF snapshots
/// ship ChatGLM3 slow-only). `None` ⇒ rely on the source already having `tokenizer/tokenizer.json`.
fn tokenizer_json() -> Option<PathBuf> {
    std::env::var("KOLORS_TOKENIZER_JSON")
        .ok()
        .map(PathBuf::from)
}

/// Build the packed tier + report each packed component's on-disk size — the hostable-tier producer
/// for the epic-8506 rollout. Run per tier (Q4/Q8):
///   KOLORS_SNAPSHOT=<snap> KOLORS_OUT=<staging/q4> KOLORS_BITS=4 KOLORS_TOKENIZER_JSON=<tok.json> \
///     cargo test -p mlx-gen-kolors --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set KOLORS_SNAPSHOT/OUT/BITS/TOKENIZER_JSON"]
fn build_tier_only() {
    let src = common::snapshot();
    let out = PathBuf::from(std::env::var("KOLORS_OUT").expect("KOLORS_OUT (tier output dir)"));
    let bits: i32 = std::env::var("KOLORS_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    assert!(
        bits == 4 || bits == 8,
        "KOLORS_BITS must be 4 or 8 (the dense bf16 tier is a verbatim mirror of the source — copy \
         the snapshot dir directly, deref'ing symlinks, and overlay the derived tokenizer.json)"
    );
    let tj = tokenizer_json();
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    mlx_gen_kolors::convert::prequantize_turnkey(&src, &out, bits, tj.as_deref())
        .expect("prequantize_turnkey succeeds");
    for (comp, stem) in [
        ("unet", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
    ] {
        let f = out.join(comp).join(format!("{stem}.safetensors"));
        let sz = std::fs::metadata(&f)
            .unwrap_or_else(|_| panic!("missing {comp}/{stem}.safetensors"))
            .len();
        println!("  {comp}/{stem}.safetensors = {:.3} GB", sz as f64 / 1e9);
    }
    assert!(
        out.join("tokenizer/tokenizer.json").is_file(),
        "tier tokenizer/tokenizer.json must be baked"
    );
    assert!(out.join("vae").is_dir(), "dense VAE must be mirrored");
    println!("✓ built {}", out.display());
}

#[test]
#[ignore = "needs a real Kolors snapshot; builds a packed tier + renders (set KOLORS_SNAPSHOT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = common::snapshot_opt() else {
        eprintln!("skip: no Kolors snapshot (set KOLORS_SNAPSHOT or populate the HF cache)");
        return;
    };
    let bits: i32 = std::env::var("KOLORS_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    assert!(bits == 4 || bits == 8, "KOLORS_BITS must be 4 or 8");
    let out = std::env::var("KOLORS_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("kolors-tier-q{bits}")));

    println!(
        "building Q{bits} turnkey: {} -> {}",
        src.display(),
        out.display()
    );
    let tj = tokenizer_json();
    mlx_gen_kolors::convert::prequantize_turnkey(&src, &out, bits, tj.as_deref())
        .expect("prequantize_turnkey succeeds");
    for (comp, stem) in [
        ("unet", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
    ] {
        let f = out.join(comp).join(format!("{stem}.safetensors"));
        assert!(f.is_file(), "missing packed {comp}/{stem}.safetensors");
    }
    assert!(
        out.join("tokenizer/tokenizer.json").is_file(),
        "baked tokenizer"
    );

    // Load DIRECTLY from the tier dir — `Kolors::load` (not `load_quantized`) packed-detects the
    // ChatGLM3 encoder + the SDXL U-Net and loads the dense VAE, with no in-app `.quantize`.
    let model = Kolors::load(Path::new(&out), Dtype::Bfloat16).expect("packed kolors loads");

    // 512² / few-step — packed-load-path proof, not a quality bench.
    let img = model
        .generate(
            "a red fox sitting in a snowy forest, photorealistic, sharp focus",
            "blurry, lowres, deformed",
            8,
            5.0,
            42,
            512,
            512,
        )
        .expect("packed generate succeeds");
    assert_eq!((img.width, img.height), (512, 512), "image size");

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    println!("✓ packed Q{bits} kolors: 512x512; px min={min} max={max} mean={mean:.1}");
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat (a missed packed site loads codes as \
         dense floats)"
    );

    if std::env::var("KOLORS_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set KOLORS_KEEP to retain)", out.display());
    }
}
