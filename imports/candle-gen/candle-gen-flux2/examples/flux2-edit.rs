//! FLUX.2-klein **edit** smoke / validation driver (sc-5487, epic 5480) — exercises the bespoke
//! [`candle_gen_flux2::Flux2Edit`] reference-edit provider against a local FLUX.2-klein snapshot on a
//! real GPU, and writes the outputs for an eyeball check. The human-validation behind the candle
//! FLUX.2 edit lane.
//!
//! ```text
//! cargo run --release --example flux2-edit --features cuda -- \
//!   --snapshot "C:\Users\…\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<hash>" \
//!   [--reference portrait.png] \
//!   --prompt "make the person wear a bright red wizard hat" --steps 4 --seed 42 --out edit.png
//! ```
//!
//! With no `--reference`, the harness first txt2img-generates a base portrait (so it is fully
//! self-contained) and edits that. It also: (a) runs a same-seed/same-prompt txt2img baseline (no
//! reference) and prints the mean-abs pixel diff — a decisive ablation that the reference is actually
//! conditioning the output; (b) checks the pre-cancel and mid-denoise cancel contract.

use std::path::PathBuf;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use candle_gen_flux2::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn save(img: &Image, path: &PathBuf) -> Result<()> {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("invalid RGB buffer dimensions")?;
    buf.save(path)?;
    Ok(())
}

/// Mean absolute per-channel pixel difference between two same-size RGB8 images.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "size mismatch");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.pixels.len() as f64
}

/// txt2img via the inventory-registered klein generator (the no-reference baseline + the optional
/// self-generated reference).
fn txt2img(snapshot: &str, prompt: &str, w: u32, h: u32, steps: u32, seed: u64) -> Result<Image> {
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snapshot)));
    let gen = gen_core::registry::load("flux2_klein_9b", &spec)?;
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut noop = |_: Progress| {};
    match gen.generate(&req, &mut noop)? {
        GenerationOutput::Images(mut imgs) => imgs.pop().ok_or_else(|| "no image".into()),
        GenerationOutput::Video { .. } => Err("expected images".into()),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("FLUX2_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set FLUX2_SNAPSHOT) at a FLUX.2-klein snapshot")?;
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "make the person wear a bright red wizard hat".into());
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("flux2_edit.png"));

    candle_gen_flux2::force_link();

    // The reference image: a user PNG, else a self-generated base portrait (so the harness is
    // standalone). Resized to the render size by the provider.
    let reference = match arg(&args, "--reference") {
        Some(path) => {
            let img = image::open(&path)?.to_rgb8();
            println!("[edit] reference={path} ({}x{})", img.width(), img.height());
            Image {
                width: img.width(),
                height: img.height(),
                pixels: img.into_raw(),
            }
        }
        None => {
            let base_prompt =
                "a photorealistic studio portrait of a young woman with long red hair, \
                               neutral background, soft lighting";
            println!("[edit] no --reference; txt2img-generating a base portrait first");
            let base = txt2img(&snapshot, base_prompt, width, height, steps, 1)?;
            let ref_path = PathBuf::from(format!("{}_reference.png", out.display()));
            save(&base, &ref_path)?;
            println!("[edit] wrote {}", ref_path.display());
            base
        }
    };

    println!(
        "[edit] {width}x{height} steps={steps} guidance={guidance} seed={seed}\n[edit] prompt={prompt:?}"
    );

    // Ablation baseline FIRST, while no edit model is resident: same prompt+seed txt2img with NO
    // reference. Two F32 9B models do not fit in VRAM at once, so this txt2img generator loads + drops
    // before the edit model loads.
    let noref = txt2img(&snapshot, &prompt, width, height, steps, seed)?;
    save(
        &noref,
        &PathBuf::from(format!("{}_noref.png", out.display())),
    )?;

    let model = Flux2Edit::load(&Flux2EditPaths {
        root: PathBuf::from(&snapshot),
    })?;

    // 1) The edit, conditioned on the reference.
    let req = Flux2EditRequest {
        prompt: prompt.clone(),
        negative: String::new(),
        width,
        height,
        steps: steps as usize,
        guidance,
        seed,
        cancel: CancelFlag::new(),
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            print!("\r[edit] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    };
    let t0 = std::time::Instant::now();
    let edited = model.generate(&req, std::slice::from_ref(&reference), &mut on_progress)?;
    println!("\n[edit] edit done in {:.1}s", t0.elapsed().as_secs_f32());
    save(&edited, &out)?;
    println!("[edit] wrote {}", out.display());

    // 2) Ablation: the edit (reference-conditioned) vs the same prompt+seed txt2img (no reference)
    // computed above → the diff must be decisive (the reference is actually conditioning the output).
    let diff = mean_abs_diff(&edited, &noref);
    println!("[edit] ablation: mean|edit − noref| = {diff:.2} (decisive when >> 0; the reference conditions the output)");

    // 3) Cancel contract: pre-cancel and mid-denoise cancel both return Canceled.
    let pre = Flux2EditRequest {
        cancel: CancelFlag::new(),
        ..req.clone()
    };
    pre.cancel.cancel();
    let mut noop = |_: Progress| {};
    let pre_ok = matches!(
        model.generate(&pre, std::slice::from_ref(&reference), &mut noop),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[edit] pre-cancel → Canceled: {}",
        if pre_ok { "PASS" } else { "FAIL" }
    );

    let mid = Flux2EditRequest {
        cancel: CancelFlag::new(),
        ..req.clone()
    };
    let mid_flag = mid.cancel.clone();
    let mut cancel_after_1 = |p: Progress| {
        if let Progress::Step { current, .. } = p {
            if current >= 1 {
                mid_flag.cancel();
            }
        }
    };
    let mid_ok = matches!(
        model.generate(&mid, std::slice::from_ref(&reference), &mut cancel_after_1),
        Err(candle_gen::CandleError::Canceled)
    );
    println!(
        "[edit] mid-cancel → Canceled: {}",
        if mid_ok { "PASS" } else { "FAIL" }
    );

    Ok(())
}
