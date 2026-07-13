//! sc-3059: SDXL IP-Adapter decoupled cross-attention engine checks (real weights).
//!
//! `#[ignore]`d — needs the SDXL base snapshot + `h94/IP-Adapter` (`ip-adapter-plus_sdxl_vit-h`).
//! Run: cargo test -p mlx-gen-sdxl --release --test ip_adapter_decoupled -- --ignored --nocapture
//!
//! Validates the injection primitive + the 70-layer walk-order remap WITHOUT a deep golden:
//!   1. Installing the real K/V pairs must succeed AND every cross-attn `forward_with_ip` reshape
//!      must line up — a wrong walk order maps a 640-d projection onto a 1280-d layer and panics.
//!   2. `forward_with_ip(scale = 0)` == plain `forward` **byte-for-byte** (the IP term is `0·o_ip`),
//!      proving the base path is untouched.
//!   3. `forward_with_ip(scale > 0)` != plain `forward` — the IP branch is actually wired in.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_sdxl as _; // force-link the provider so `inventory` registers "sdxl"
use mlx_gen_sdxl::{load_ip_kv_pairs, load_unet_dtype, text_time_ids};
use mlx_rs::{Array, Dtype};

fn sdxl_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn ip_weights() -> Weights {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--h94--IP-Adapter/snapshots");
    let dir = std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for h94/IP-Adapter")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    let mut w =
        Weights::from_file(dir.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors")).unwrap();
    w.cast_all(Dtype::Float16).unwrap();
    w
}

fn randn(shape: &[i32], seed: u64) -> Array {
    mlx_rs::random::seed(seed).unwrap();
    mlx_rs::random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

#[test]
#[ignore = "needs the SDXL base snapshot + h94/IP-Adapter weights"]
fn ip_decoupled_attn_remap_and_scale_zero() {
    let mut unet = load_unet_dtype(&sdxl_snapshot(), Dtype::Float16).unwrap();
    let pairs = load_ip_kv_pairs(&ip_weights()).unwrap();
    assert_eq!(pairs.len(), 70, "SDXL IP-Adapter has 70 cross-attn layers");
    // Panics here if the walk order maps a projection onto a mismatched-dim layer.
    unet.install_ip_adapter(pairs).unwrap();

    // CFG-batched dummy inputs (B=2).
    let latents = randn(&[2, 64, 64, 4], 1);
    let cond = randn(&[2, 77, 2048], 2);
    let pooled = randn(&[2, 1280], 3);
    let time_ids = text_time_ids(2);
    let ip_tokens = randn(&[2, 16, 2048], 4);
    let t = 500.0;

    let eps_plain = unet
        .forward(&latents, t, &cond, &pooled, &time_ids)
        .unwrap();
    let eps_ip0 = unet
        .forward_with_ip(&latents, t, &cond, &pooled, &time_ids, (&ip_tokens, 0.0))
        .unwrap();
    let eps_ip = unet
        .forward_with_ip(&latents, t, &cond, &pooled, &time_ids, (&ip_tokens, 0.6))
        .unwrap();

    let d0 = max_abs(&eps_plain, &eps_ip0);
    let di = max_abs(&eps_plain, &eps_ip);
    println!("[ip-adapter] forward vs forward_with_ip(scale=0): max|Δ|={d0:.3e}");
    println!("[ip-adapter] forward vs forward_with_ip(scale=0.6): max|Δ|={di:.3e}");
    assert_eq!(d0, 0.0, "scale=0 must be byte-identical to plain forward");
    assert!(
        di > 1e-3,
        "scale>0 must change the prediction (IP branch wired)"
    );
}

fn h94_snapshot() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--h94--IP-Adapter/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for h94/IP-Adapter")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn gradient(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// e2e wiring: in IP mode a `Reference` at `ip_adapter_scale = 0` (strength 0) reproduces plain
/// txt2img byte-for-byte (the IP branch is zeroed and draws no RNG), while `scale > 0` changes the
/// image. Proves load → token pipeline → denoise_ip → CFG zeros-uncond → scale knob.
#[test]
#[ignore = "needs the SDXL base snapshot + h94/IP-Adapter weights"]
fn ip_generate_scale_zero_equals_txt2img() {
    let model = mlx_gen_sdxl::load(
        &LoadSpec::new(WeightsSource::Dir(sdxl_snapshot()))
            .with_ip_adapter(WeightsSource::Dir(h94_snapshot())),
    )
    .unwrap();
    let refimg = gradient(512, 512);

    let req = |scale: Option<f32>| {
        let conditioning = match scale {
            Some(s) => vec![Conditioning::Reference {
                image: refimg.clone(),
                strength: Some(s),
            }],
            None => vec![],
        };
        GenerationRequest {
            prompt: "a portrait".to_string(),
            width: 512,
            height: 512,
            seed: Some(5),
            steps: Some(6),
            conditioning,
            ..Default::default()
        }
    };
    let run = |r: &GenerationRequest| match model.generate(r, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!(),
    };

    let plain = run(&req(None));
    let ip0 = run(&req(Some(0.0)));
    let ip = run(&req(Some(0.6)));

    let diff0 = plain
        .pixels
        .iter()
        .zip(&ip0.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let diffi = plain
        .pixels
        .iter()
        .zip(&ip.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!("[ip-adapter] txt2img vs IP(scale=0): {diff0} px bytes differ");
    println!("[ip-adapter] txt2img vs IP(scale=0.6): {diffi} px bytes differ");
    assert_eq!(diff0, 0, "IP scale=0 must equal plain txt2img");
    assert!(diffi > 0, "IP scale=0.6 must change the image");
}
