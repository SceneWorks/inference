//! sc-10839 (epic 10834 Phase 1): the `Sequential` component-residency A/B on real SDXL weights.
//!
//! `#[ignore]`d — needs the real SDXL snapshot (`SDXL_SNAPSHOT`, else the HF cache). Run:
//!   cargo test -p mlx-gen-sdxl --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Proves the epic's two claims for the image lane:
//!
//! 1. **Bounded peak** — the same generation under [`OffloadPolicy::Sequential`] peaks LOWER than
//!    under `Resident`, because the CLIP text encoders are dropped (+ `clear_cache()`) before the
//!    U-Net materializes, so peak is `max(encoders, U-Net+VAE)` instead of the sum.
//! 2. **Zero parity cost** — the `Sequential` render is BYTE-IDENTICAL to `Resident` (same encode,
//!    denoise, decode; only the load/drop schedule differs), so it is safe to select from the
//!    fit-gate with no output change.
//!
//! A third check drives TWO back-to-back `Sequential` jobs and confirms the second peaks the same as
//! the first — i.e. nothing large stayed resident across jobs (the property that lets a small Mac
//! keep running the model, not just survive the first request).
//!
//! Optional: set `SDXL_SEQ_Q8=1` to run the epic table's `illustrious q8`-class case (Q8 weights),
//! and `SDXL_SEQ_STEPS` / `SDXL_SEQ_SIZE` to tune the (quality-irrelevant) probe generation.

mod common;

// Force-link the provider crate so its `inventory` generator registration runs — this test only
// calls `mlx_gen::load("sdxl", …)` and would otherwise dead-strip `mlx_gen_sdxl` (cf. sibling tests).
use mlx_gen_sdxl as _;

use common::snapshot;
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // A tiny, deterministic probe — image quality is irrelevant, we compare Resident vs Sequential
    // to each other (not to a golden), so a few steps at a modest size is enough and keeps the test
    // fast. A fixed seed makes the byte-identity assertion meaningful.
    let size = env_u32("SDXL_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, lowres".into()),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("SDXL_SEQ_STEPS", 6)),
        guidance: Some(7.0),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if std::env::var("SDXL_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

/// Render `req` under `policy`, measuring the peak memory of the generate call alone (reset right
/// before `generate`, after the model is constructed). The model is dropped + the cache cleared
/// before returning so the NEXT measurement starts from a clean allocator.
fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen::load("sdxl", &spec).expect("load sdxl");
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
#[ignore = "needs the real SDXL snapshot (SDXL_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();

    // Run A — the resident default (every component held for the whole job).
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    // Run B — sequential residency (encoders dropped before the U-Net loads).
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "SDXL {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("SDXL_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    // (2) Byte-identical output — the Sequential schedule must not perturb a single pixel (same
    // encode/denoise/decode; the drop + clear_cache + the extra materialization barrier draw no RNG).
    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical to \
         Resident — the schedule only changes what is co-resident, never the math)",
        pixels_resident.len()
    );

    // (1) Bounded peak — dropping the text encoders before the U-Net loads must lower the peak.
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder drop did not \
         reduce peak (is the eval-before-drop materialization barrier in place?)",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot (SDXL_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    // The property that actually lets a small Mac RUN (not just survive once): a second Sequential
    // job must peak the same as the first, i.e. nothing large stayed resident across jobs. If the
    // U-Net were cached warm, job 2 would load the encoders ON TOP of it and peak at the resident
    // sum — defeating the fit-gate's promise.
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "SDXL Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    // Allow a small slop for allocator variation; the point is job 2 is NOT ~a text-encoder larger.
    let slop = peak1 / 10; // 10%
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident \
         across jobs, so peak is creeping toward the resident sum",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}
