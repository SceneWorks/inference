//! sc-11030 (epic 10834 Phase 1 fan-out): the `Sequential` component-residency A/B on real Lens
//! weights.
//!
//! `#[ignore]`d — needs the real `microsoft/Lens-Turbo` snapshot (env `LENS_DIR`, else the HF cache).
//! Run:
//!   cargo test -p mlx-gen-lens --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Two claims (same as the SDXL/Z-Image/Qwen A/B): (1) `Sequential` peaks LOWER than `Resident`
//! because the gpt-oss MoE text encoder is dropped (+ `clear_cache()`) before the DiT + the denoise
//! activations materialize, and (2) the output is BYTE-IDENTICAL. **Measured Q8 768²/4-step: Resident
//! 34.5 → Sequential 22.3 GiB (−12.2, 35%)** — 22.3 GiB fits a 32 GB Mac, 34.5 does not. A repeat-job
//! check confirms nothing stays resident across jobs.
//!
//! Two things make this work and are worth remembering (sc-11030):
//! - The gpt-oss encoder is the LARGEST component, so the saving is NOT the encoder's size — it is the
//!   denoise-phase DiT + activations that `Resident` stacks ON TOP of the resident encoder. Sequential
//!   drops the encoder first, so the peak stays at the (lower) encode phase.
//! - The encoder loader must **consume** its `Weights` (free each layer's source as built), else the
//!   source(13 GB)+built LOAD spike is the peak and staging saves nothing. That fix is in
//!   `LensTextEncoder::with_selected_layers`.
//! - **Q8/Q4 are the tiers that matter** (a 32 GB Mac can't hold bf16 either way). `LENS_SEQ_BF16`
//!   does NOT stage-win — its ~63 GB encoder is nearly the whole footprint.
//!
//! The probe overrides `guidance > 1.0` with a **non-empty** negative prompt, so both the joint-CFG
//! (B=2) DiT forward AND the second `encode_prompt` (the negative branch) run — the stringent case
//! for the "encode once, then drop the encoder" byte-identity. Default is Q8; set `LENS_SEQ_Q4=1` for
//! Q4, `LENS_SEQ_BF16=1` for the (non-winning) dense case, `LENS_SEQ_STEPS`/`LENS_SEQ_SIZE`/
//! `LENS_SEQ_GUIDANCE` to tune.

// `provider_registry().load("lens_turbo", …)`).

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const MODEL_ID: &str = "lens_turbo";
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> PathBuf {
    let p = std::env::var("LENS_DIR").unwrap_or_else(|_| panic!("set LENS_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // A fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here (Resident
    // vs Sequential, not a golden). A guidance > 1.0 + a non-empty negative forces the joint-CFG
    // forward AND the negative-branch encode, the stringent case for the encode-once residency.
    let size = env_u32("LENS_SEQ_SIZE", 1024);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("LENS_SEQ_STEPS", 8)),
        guidance: Some(env_f32("LENS_SEQ_GUIDANCE", 4.0)),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    // `LENS_SEQ_BF16` runs the dense path (no load-time quant) — the epic's headline −13.1 GB case
    // where the whole bf16 encoder is dropped. Otherwise quantize the encoder MoE + DiT (Q8 default,
    // `LENS_SEQ_Q4` for Q4).
    if std::env::var("LENS_SEQ_BF16").is_ok() {
        spec
    } else if std::env::var("LENS_SEQ_Q4").is_ok() {
        spec.with_quant(Quant::Q4)
    } else {
        spec.with_quant(Quant::Q8)
    }
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen_lens::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load lens_turbo");
    reset_peak_memory();
    let out = model.generate(req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "expected a single image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    (pixels, peak)
}

#[test]
#[ignore = "needs a real microsoft/Lens-Turbo snapshot (LENS_DIR or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Lens-Turbo {}x{} @ {} steps, guidance {}{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        req.guidance.unwrap(),
        if std::env::var("LENS_SEQ_Q4").is_ok() { " (Q4)" } else { " (Q8)" },
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical)",
        pixels_resident.len()
    );
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real microsoft/Lens-Turbo snapshot (LENS_DIR or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Lens-Turbo Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    let slop = peak1 / 10;
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}
