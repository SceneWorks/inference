//! sc-5989/sc-5990 — produce a pre-quantized **Q4 turnkey** from the dense bf16 snapshot, then load
//! it **packed** and generate. Validates the packed loader (no dense bf16 transient, ~¼ the on-disk
//! size) and produces the artifact the publish (sc-5990) uploads.
//!
//! `#[ignore]` — needs the bf16 snapshot (~53 GB) and writes the Q4 snapshot (~14 GB). Run:
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert IDEOGRAM4_Q4=~/.cache/ideogram4-mlx-q4 \
//!     cargo test -p mlx-gen-ideogram --test prequantize -- --ignored --nocapture

mod common;

use std::path::{Path, PathBuf};

use common::CAPTION_JSON;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_ideogram::{convert, MODEL_ID};
use mlx_rs::memory::{get_peak_memory, reset_peak_memory};

fn env_dir(key: &str, default_rel: &str) -> PathBuf {
    std::env::var(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").expect("HOME")).join(default_rel))
}

/// Recursive on-disk size of a directory, in GB.
fn dir_size_gb(p: &Path) -> f64 {
    fn bytes(p: &Path) -> u64 {
        let mut t = 0;
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    t += bytes(&path);
                } else if let Ok(m) = e.metadata() {
                    t += m.len();
                }
            }
        }
        t
    }
    bytes(p) as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "needs the bf16 snapshot (~53 GB) + writes the packed snapshot"]
fn prequantize_loads_and_generates() {
    let src = env_dir("IDEOGRAM4_MLX", ".cache/ideogram4-mlx-convert");
    // `IDEOGRAM4_BITS` (4 = Q4 default, 8 = Q8); dst defaults to `~/.cache/ideogram4-mlx-q{bits}`.
    let bits: i32 = std::env::var("IDEOGRAM4_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let dst = env_dir("IDEOGRAM4_QDIR", &format!(".cache/ideogram4-mlx-q{bits}"));

    // 1. Convert bf16 → packed Q{bits} (idempotent: skip if the packed transformer is already there).
    if !dst.join("transformer/model.safetensors").exists() {
        println!(
            "converting {} → packed Q{bits} at {} …",
            src.display(),
            dst.display()
        );
        convert::prequantize_turnkey(&src, &dst, bits).expect("prequantize_turnkey");
    }
    println!(
        "Q{bits} turnkey: {:.2} GB on disk (vs ~50 GB bf16 source) at {}",
        dir_size_gb(&dst),
        dst.display()
    );

    // 2. Load the Q4 snapshot PACKED — `quantize=None`, so the lin loaders auto-detect the packed
    //    weights and build quantized linears directly (no dense bf16 ever materialized).
    reset_peak_memory();
    let spec = LoadSpec::new(WeightsSource::Dir(dst.clone()));
    let g = mlx_gen_ideogram::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load packed Q4 snapshot");
    assert_eq!(g.descriptor().id, "ideogram_4");

    // 3. Generate + measure the peak (should be ~Q4 runtime levels, never the 53 GB dense source).
    let envn = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let res = envn("IDEOGRAM4_SMOKE_RES", 1024);
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 40);
    let req = GenerationRequest {
        prompt: CAPTION_JSON.into(),
        width: res,
        height: res,
        steps: Some(steps),
        guidance: Some(7.0),
        seed: Some(0),
        ..Default::default()
    };
    let out = g
        .generate(&req, &mut |_| {})
        .expect("generate from packed snapshot");
    let peak = get_peak_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("packed Q{bits} generate @{res}²/{steps}step: peak {peak:.2} GB");

    let imgs = match out {
        GenerationOutput::Images(v) => v,
        other => panic!("expected Images, got {other:?}"),
    };
    let im = &imgs[0];
    assert_eq!((im.width, im.height), (res, res));
    let (mn, mx) = (
        *im.pixels.iter().min().unwrap(),
        *im.pixels.iter().max().unwrap(),
    );
    assert!(
        mx > mn,
        "degenerate image — packed Q{bits} load broke the forward"
    );

    let out_path = std::env::temp_dir().join(format!("ideogram4_q{bits}_turnkey.png"));
    image::RgbImage::from_raw(res, res, im.pixels.clone())
        .unwrap()
        .save(&out_path)
        .unwrap();
    println!("wrote {}", out_path.display());
}
