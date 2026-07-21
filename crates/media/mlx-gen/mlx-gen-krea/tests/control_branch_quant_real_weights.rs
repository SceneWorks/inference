//! sc-11748 (epic 8459): real-weight **A/B of the quantized pose-control branch** — the story's numeric
//! gate. Renders the SAME pose + seed twice on a `krea_2_turbo_control` model with a **q4 base** — render
//! A with the branch bf16 (headroom → the sc-11748 budget gate keeps it dense), render B with the branch
//! packed to q4 (the gate forced on by starving the device budget during load) — then asserts that (1) the
//! two renders stay coherent (a PSNR floor that catches a structural break; q4 holds the pose-lock with
//! the candle-#480 mild haze, ~23 dB measured, pose visually identical), and (2) the q4-branch render's
//! resident high-water is materially lower (proof the branch actually packed, ~4.4 GiB bf16→q4 measured).
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache), same sources as
//! `control_memory_calibration_real_weights.rs`. Base: `SceneWorks/krea-2-turbo-mlx` bf16 dir, env
//! `KREA_CONTROL_DIR` (quantized to q4 at load via `with_quant`, the sc-11727 packed-base path). Overlay:
//! `SceneWorks/krea2-pose-controlnet-beta/control_step5000.safetensors`, env `KREA_CONTROL_OVERLAY`.
//!
//! Run:
//! ```text
//! cargo test -p mlx-gen-krea --release --test control_branch_quant_real_weights -- --ignored --nocapture
//! ```
//!
//! ## How the two branch tiers are forced without a production toggle
//!
//! The branch tier is chosen at LOAD by `should_quantize_control_branch(safe_budget_gib(), …)`. Both
//! renders load a q4 base under `Resident` (so the branch — and thus the gate — is decided at `load()`).
//! Render A loads at the real device budget, so the gate returns `false` (a big Mac has headroom) and the
//! branch stays bf16. Render B wraps the explicit Krea registry load in a lowered MLX memory limit
//! (`KREA_BQ_STARVE_GIB`,
//! default 12 GiB), dropping `safe_budget_gib()` below the worst-case (2048²) control peak so the gate
//! returns `true` and packs the branch to q4; the real limit is RESTORED before `generate` so the render
//! itself has memory. This drives the real production gate end to end (no test-only branch-quant hook), on
//! real weights.

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{get_peak_memory, reset_peak_memory, set_memory_limit};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

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

/// A deterministic RGB pose stand-in (content is irrelevant — the A/B holds the pose fixed across both
/// renders; the comparison is quantized-branch vs bf16-branch, not pose fidelity vs the skeleton).
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

/// A q4-base control spec under `Resident` (so the branch tier is decided at `load()`).
fn spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(base_dir()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_quant(Quant::Q4)
}

/// One render + its resident high-water (GiB). `starve` wraps only the LOAD in a lowered MLX memory limit
/// (`KREA_BQ_STARVE_GIB`, default 12 GiB) so `safe_budget_gib()` drops below the worst-case (2048²) control
/// peak and the sc-11748 gate packs the branch to q4; the real limit is restored before `generate`. A
/// moderate limit (vs. 1 byte) forces the gate without thrashing the weight load under constant eviction.
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

/// PSNR (dB) between two same-size RGB images — the coherence proxy. A structural break (broken pose /
/// confetti) tanks PSNR (well below ~15 dB); the candle-#480 mild q4 haze / trajectory drift leaves it
/// moderate (~23 dB measured) with the pose held.
fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "A/B renders must be the same size"
    );
    let mse = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / (a.pixels.len() as f64);
    if mse == 0.0 {
        return f64::INFINITY;
    }
    20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
}

#[test]
#[ignore = "needs real Krea base + pose overlay (see module header); run with --ignored on a Metal Mac"]
fn q4_branch_holds_pose_lock_and_cuts_resident_memory() {
    let size = env_u32("KREA_BQ_SIZE", 768);

    let (img_bf16, peak_bf16) = render(size, false);
    let (img_q4, peak_q4) = render(size, true);

    // Optional visual artifact for a human pose-lock check (PSNR is only a coarse coherence proxy).
    if let Ok(dir) = std::env::var("KREA_BQ_OUT") {
        for (img, name) in [(&img_bf16, "branch_bf16"), (&img_q4, "branch_q4")] {
            let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                .expect("RGB buffer");
            let path = format!("{dir}/{name}_{size}.png");
            buf.save(&path)
                .unwrap_or_else(|e| panic!("save {path}: {e}"));
            println!("wrote {path}");
        }
    }

    let db = psnr(&img_bf16, &img_q4);
    let saved = peak_bf16 - peak_q4;
    println!(
        "sc-11748 branch A/B @ {size}²: PSNR(bf16-branch vs q4-branch) = {db:.2} dB; \
         resident peak bf16 = {peak_bf16:.2} GiB, q4 = {peak_q4:.2} GiB, saved = {saved:.2} GiB"
    );

    // Coherence guard — a STRUCTURAL-break floor, NOT a fidelity metric. Measured on real weights
    // (512², this Mac, sc-11748): PSNR ≈ 23.3 dB with the pose HELD — same stance/hands/head-tilt/framing,
    // the q4 numeric perturbation only drifts outfit texture + fine detail (the candle-#480 "mild haze"
    // regime; visually confirmed via `KREA_BQ_OUT`). PSNR here penalizes that acceptable trajectory drift,
    // so it is only a proxy: an actual break (broken pose / confetti) would score well below ~15 dB. `18`
    // sits under the measured coherent value with margin while still catching a genuine break; the real
    // pose-lock gate is the visual A/B (`KREA_BQ_OUT`), backed by candle #480's GPU proof.
    assert!(
        db >= env_u32("KREA_BQ_MIN_PSNR", 18) as f64,
        "q4 branch appears structurally broken vs bf16 (PSNR {db:.2} dB) — expected the mild-haze regime \
         (~23 dB, pose held), not a break. Inspect the renders with KREA_BQ_OUT set."
    );

    // Proof the branch actually packed: the q4-branch resident high-water must be materially lower.
    // Measured saving ≈ 4.4 GiB (matching the story's ~4.6 GiB target on the N=7 bf16→q4 overlay); require
    // ≥ 2 GiB to absorb activation/allocator noise while still failing loudly if the gate did NOT pack it.
    assert!(
        saved >= 2.0,
        "q4-branch resident peak should drop ≥ 2 GiB vs bf16 branch (saved {saved:.2} GiB) — did the \
         budget gate actually pack the branch?"
    );
}
