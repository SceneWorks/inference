//! Krea 2 pose-ControlNet **provider** smoke (sc-8464, epic 8459) — drives the packaged
//! [`Krea2Control`](candle_gen_krea::Krea2Control) exactly as the worker `KreaControl` route does:
//! load the Turbo snapshot + a trained control-branch overlay once, then render one pose-conditioned
//! image from a skeleton PNG.
//!
//! This is the deployable-path sibling of `krea-control-infer` (which stays the low-level byte-identity
//! diagnostic): it validates the public provider API end-to-end against the sc-8460 spike checkpoint
//! before the worker lane wires it. Reproduce the spike's pose-lock:
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-provider --features cuda --release -- \
//!   --snapshot <krea-2-turbo snapshot dir> --ckpt control_step5000.safetensors \
//!   --pose pose.png --prompt "a person dancing" --scale 0.6 --seed 42 --out out.png
//! ```
//!
//! Flags: `--snapshot <dir>` `--ckpt <safetensors>` (required) `--pose <png>` (required)
//! `--prompt <str>` `--scale F` (default 0.6) `--seed N` `--steps N` (default 8) `--size N`
//! (square, default 1024) `--out <png>`.

use std::path::PathBuf;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, OffloadPolicy, PreviewSink, Progress, Quant};
use candle_gen_krea::{
    Krea2Control, Krea2ControlPaths, Krea2ControlRequest, DEFAULT_CONTROL_SCALE,
};

struct Args {
    snapshot: PathBuf,
    ckpt: PathBuf,
    pose: PathBuf,
    prompt: String,
    scale: f32,
    seed: u64,
    steps: usize,
    size: u32,
    out: PathBuf,
    /// Quantize the control-branch overlay for the small-card load (sc-11743): `q4` / `q8` keep it
    /// packed in VRAM (dequant-on-forward), `bf16` (default) is the full-precision branch.
    branch_tier: Option<Quant>,
}

/// `q4` / `q8` → the packed branch load; `bf16` → dense. Any other value panics (example CLI).
fn parse_branch_tier(v: &str) -> Option<Quant> {
    match v {
        "q4" | "Q4" => Some(Quant::Q4),
        "q8" | "Q8" => Some(Quant::Q8),
        "bf16" | "none" => None,
        other => panic!("--branch-quant must be q4|q8|bf16 (got {other})"),
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        snapshot: PathBuf::from("D:/models/Krea-2-Turbo"),
        ckpt: PathBuf::new(),
        pose: PathBuf::new(),
        prompt: "a person standing in a colorful room".into(),
        scale: DEFAULT_CONTROL_SCALE,
        seed: 42,
        steps: 8,
        size: 1024,
        out: PathBuf::from("krea_control_provider.png"),
        branch_tier: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {key}"))
                .clone()
        };
        match key {
            "--snapshot" => a.snapshot = val().into(),
            "--ckpt" => a.ckpt = val().into(),
            "--pose" => a.pose = val().into(),
            "--prompt" => a.prompt = val(),
            "--scale" => a.scale = val().parse().expect("--scale"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--steps" => a.steps = val().parse().expect("--steps"),
            "--size" => a.size = val().parse().expect("--size"),
            "--out" => a.out = val().into(),
            "--branch-quant" => a.branch_tier = parse_branch_tier(&val()),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    assert!(!a.ckpt.as_os_str().is_empty(), "--ckpt is required");
    assert!(!a.pose.as_os_str().is_empty(), "--pose is required");
    a
}

/// Load a skeleton PNG into a gen_core `Image` (HWC RGB u8) at the render size — the provider requires
/// the control image already at `size`×`size` (the worker driver renders it there; the lib carries no
/// codec). The spike poses are square-canonical, so a direct resize matches the train-time letterbox.
fn load_pose(path: &PathBuf, size: u32) -> Result<Image, Box<dyn std::error::Error>> {
    let rgb = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&rgb, size, size, image::imageops::FilterType::Lanczos3);
    Ok(Image {
        width: size,
        height: size,
        pixels: resized.into_raw(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args();
    let pose = load_pose(&a.pose, a.size)?;

    // `KREA_CHUNK_ATTN=1` engages sc-6217-style query-row attention chunking on the base stack + branch
    // (sc-11745) — the fit-ladder's activation-peak rung; default false = the unchunked full-speed
    // forward. Set at load, so it flips before the sampler loop (the measurement harness A/Bs this).
    let chunk_attention = std::env::var("KREA_CHUNK_ATTN")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let model = Krea2Control::load(&Krea2ControlPaths {
        root: a.snapshot,
        convrot_dit: std::env::var("KREA_CONTROL_CONVROT_DIT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from),
        native_dit: None,
        control: a.ckpt,
        adapters: Vec::new(),
        branch_tier: a.branch_tier,
        chunk_attention,
        // Legacy compatibility only; set `Krea2ControlRequest::stage_residency` per request.
        offload_policy: OffloadPolicy::Resident,
    })?;
    eprintln!(
        "loaded Krea2Control (branch_tier {:?}, chunk_attention {chunk_attention}); rendering {}x{} @ scale {}",
        a.branch_tier, a.size, a.size, a.scale
    );

    // `KREA_TILE_VAE=1` forces the seam-free tiled VAE decode below the im2col threshold (sc-11744) —
    // the fit-ladder's cheapest VRAM rung; default false = the monolithic full-speed decode.
    let tile_vae_decode = std::env::var("KREA_TILE_VAE")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    // `KREA_PREVIEW_DIR=<dir>` attaches a live per-step preview sink (epic 16948, sc-16950) and writes
    // each latent-resolution frame as a PNG — the control route's real-weight preview check, since it
    // is a bespoke by-name provider rather than a registered generator a test can drive. Unset leaves
    // the inert default, which is byte-identical to a render with no preview at all.
    let preview = match std::env::var("KREA_PREVIEW_DIR").ok().map(PathBuf::from) {
        Some(dir) => {
            std::fs::create_dir_all(&dir)?;
            eprintln!("preview frames → {}", dir.display());
            PreviewSink::new(move |frame| {
                let path = dir.join(format!("preview_{:03}.png", frame.current));
                let saved = image::RgbImage::from_raw(
                    frame.image.width,
                    frame.image.height,
                    frame.image.pixels,
                )
                .map(|buf| buf.save(&path));
                eprintln!(
                    "  preview {}/{} {}x{} → {}",
                    frame.current,
                    frame.total,
                    frame.image.width,
                    frame.image.height,
                    if matches!(saved, Some(Ok(()))) {
                        path.display().to_string()
                    } else {
                        "(write failed)".to_owned()
                    }
                );
            })
        }
        None => PreviewSink::default(),
    };
    let req = Krea2ControlRequest {
        prompt: a.prompt,
        width: a.size,
        height: a.size,
        steps: a.steps,
        control_scale: a.scale,
        text_style_gain: None,
        seed: a.seed,
        tile_vae_decode,
        stage_residency: false,
        cancel: CancelFlag::new(),
        preview,
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            eprintln!("step {current}/{total}");
        }
    };
    let out = model.generate(&req, &pose, &mut on_progress)?;

    let buf = image::RgbImage::from_raw(out.width, out.height, out.pixels)
        .ok_or("bad output image buffer")?;
    buf.save(&a.out)?;
    eprintln!("wrote {} ({}x{})", a.out.display(), out.width, out.height);
    Ok(())
}
