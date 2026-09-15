//! sc-15799 (epic 8459 → epic 15448): real-weight proof that the pose-control branch's tier is
//! **tier integrity, not a memory lever** — the numeric gate that replaces the sc-11748 A/B.
//!
//! ## What changed, and why the old test could not survive it
//!
//! sc-11748 shipped a *budget gate*: on a q4 base the branch stayed **bf16** on any Mac with headroom
//! and packed to q4 only when the projected footprint would not fit. The old test proved exactly that —
//! it rendered twice, once at the real device budget (branch bf16) and once under a lowered MLX memory
//! limit (branch q4), and asserted the two stayed coherent and that the packed arm's resident high-water
//! was materially lower.
//!
//! Under sc-15799 the bf16 arm **no longer exists in production**: the branch's tier is
//! `control_branch_quant_bits(base_bits)` — a q4 base floors its branch at **q8** (the declared, measured
//! exception: a q4 control residual measures "pose-locked; non-pose details drift"), a q8 base carries a
//! q8 branch, and only a dense base carries a dense branch. So the old A/B would have been testing a
//! configuration the loader can no longer produce.
//!
//! The two claims this asserts instead are the ones that are now load-bearing:
//!
//! 1. **Budget independence.** The same q4-base load under the real budget and under a starved budget
//!    must produce a **byte-identical** render and the same resident high-water. This is the direct
//!    falsifier: restore the sc-11748 gate and the starved arm packs to q4 instead of q8, so the pixels
//!    and the peak both move and this fails loudly.
//! 2. **The estimator never under-predicts the tier it actually loads.** The measured resident
//!    high-water must stay within `mlx_gen_krea::memory`'s q8-branch prediction. That module's
//!    over-predict convention (an under-shoot is an OOM; an over-shoot only adapts slightly sooner) is
//!    only a claim until a real render checks it, and the branch tier the estimator prices changed in
//!    this story.
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache), same sources as
//! `control_memory_calibration_real_weights.rs`. Base: `SceneWorks/krea-2-turbo-mlx` bf16 dir, env
//! `KREA_CONTROL_DIR` (quantized to q4 at load via `with_quant`, the sc-11727 packed-base path). Overlay:
//! `SceneWorks/krea2-pose-controlnet-beta/control_step5000.safetensors`, env `KREA_CONTROL_OVERLAY`.
//!
//! Run:
//! ```text
//! cargo test -p mlx-gen-krea --release --test integration control_branch_tier_real_weights:: -- --ignored --nocapture
//! ```

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_gen_krea::memory::{control_denoise_peak_ex_text_gib, qwen_vae_decode_peak_ex_text_gib};
use mlx_gen_krea::Krea2Config;
use mlx_rs::memory::{get_peak_memory, reset_peak_memory, set_memory_limit};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The trained S0 overlay's copied block count (`Krea2ControlBranch::num_blocks` for the shipped
/// recipe) — needed to price the same branch the loader builds.
const BRANCH_BLOCKS: usize = 7;

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = std::path::PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"))
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn base_dir() -> PathBuf {
    std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-turbo-mlx").join("bf16"))
}

fn overlay() -> PathBuf {
    std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--SceneWorks--krea2-pose-controlnet-beta")
                .join("control_step5000.safetensors")
        })
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A deterministic RGB pose stand-in (content is irrelevant — both renders hold the pose fixed; the
/// comparison is budget-vs-budget, not pose fidelity vs the skeleton).
fn fixed_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(((x + y) * 127 / (w + h).max(1)) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn request(size: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_BQ_STEPS", 8)),
        conditioning: vec![Conditioning::Control {
            image: fixed_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    }
}

/// A q4-base control spec under `Resident` (so the branch tier is fixed at `load()`).
fn spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(base_dir()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_quant(Quant::Q4)
}

/// One render + its resident high-water (GiB). `starve` wraps only the LOAD in a lowered MLX memory
/// limit (`KREA_BQ_STARVE_GIB`, default 12 GiB) so `safe_budget_gib()` drops far below the worst-case
/// (2048²) control peak; the real limit is restored before `generate`. Under sc-11748 that flipped the
/// branch to q4; under sc-15799 it must change nothing at all.
fn render(size: u32, starve: bool) -> (Image, f64) {
    let registry =
        mlx_gen_krea::provider_registry().expect("build explicit Krea provider registry");
    let model = if starve {
        let starve_bytes = env_u32("KREA_BQ_STARVE_GIB", 12) as usize * 1024 * 1024 * 1024;
        let prev = set_memory_limit(starve_bytes);
        let m = registry.load("krea_2_turbo_control", &spec());
        set_memory_limit(prev); // restore so the render has memory
        m
    } else {
        registry.load("krea_2_turbo_control", &spec())
    }
    .unwrap_or_else(|e| panic!("load krea_2_turbo_control (starve={starve}): {e}"));

    reset_peak_memory();
    let out = model
        .generate(&request(size), &mut |_: Progress| {})
        .unwrap_or_else(|e| panic!("generate (starve={starve}): {e}"));
    let peak = get_peak_memory() as f64 / GIB;

    let GenerationOutput::Images(mut imgs) = out else {
        panic!("expected images");
    };
    (imgs.swap_remove(0), peak)
}

#[test]
#[ignore = "needs real Krea base + pose overlay (see module header); run with --ignored on a Metal Mac"]
fn branch_tier_is_budget_independent_and_within_its_estimate() {
    let size = env_u32("KREA_BQ_SIZE", 768);

    let (img_free, peak_free) = render(size, false);
    let (img_starved, peak_starved) = render(size, true);

    // Optional visual artifact for a human pose-lock check.
    if let Ok(dir) = std::env::var("KREA_BQ_OUT") {
        for (img, name) in [
            (&img_free, "branch_q8_free"),
            (&img_starved, "branch_q8_starved"),
        ] {
            let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                .expect("RGB buffer");
            let path = format!("{dir}/{name}_{size}.png");
            buf.save(&path)
                .unwrap_or_else(|e| panic!("save {path}: {e}"));
            println!("wrote {path}");
        }
    }

    println!(
        "sc-15799 branch tier @ {size}²: resident peak at the real budget = {peak_free:.2} GiB, \
         under a {} GiB starved load = {peak_starved:.2} GiB",
        env_u32("KREA_BQ_STARVE_GIB", 12),
    );

    // CLAIM 1 — budget independence. The branch tier is a function of the BASE tier alone, so starving
    // the load budget must not change one byte. Restoring the sc-11748 gate packs the starved arm to q4
    // and both of these fail.
    assert_eq!(
        (img_free.width, img_free.height),
        (img_starved.width, img_starved.height),
        "the starved load must not change the output geometry"
    );
    assert!(
        img_free.pixels == img_starved.pixels,
        "the branch tier must not depend on the device budget: starving the load changed the render, \
         which means something re-picked the branch's precision from free memory"
    );
    assert!(
        (peak_free - peak_starved).abs() <= 0.5,
        "same configuration, so the resident high-water must match within allocator noise: \
         {peak_free:.2} vs {peak_starved:.2} GiB"
    );

    // CLAIM 2 — the estimator prices the tier that actually loaded, and over-predicts (never under).
    // An under-prediction is an OOM; this is the module's stated convention, checked against a real run.
    let cfg = Krea2Config::turbo();
    let predicted = control_denoise_peak_ex_text_gib(
        &cfg,
        BRANCH_BLOCKS,
        Some(Quant::Q4),
        Some(Quant::Q8),
        size,
        size,
    )
    .max(qwen_vae_decode_peak_ex_text_gib(
        &cfg,
        BRANCH_BLOCKS,
        Some(Quant::Q4),
        Some(Quant::Q8),
        size,
        size,
    ));
    println!("q8-branch prediction (ex-text) = {predicted:.2} GiB");
    assert!(
        peak_free <= predicted,
        "the q8-branch estimate must not UNDER-predict the real resident peak (measured \
         {peak_free:.2} GiB > predicted {predicted:.2} GiB) — an under-shoot is an OOM"
    );
}
