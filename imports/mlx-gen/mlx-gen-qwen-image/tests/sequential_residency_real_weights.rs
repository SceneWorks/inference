//! sc-11000 (epic 10834 Phase 1 fan-out): the `Sequential` component-residency A/B on real
//! Qwen-Image weights.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot (env `QWEN_IMAGE_SNAPSHOT`, else the HF
//! cache). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Two claims (same as the SDXL/Z-Image A/B): (1) `Sequential` peaks LOWER than `Resident` because
//! the Qwen2.5-VL text encoder is dropped (+ `clear_cache()`) before the DiT materializes, and
//! (2) the output is BYTE-IDENTICAL. Qwen-Image's ~15 GB encoder is comparable to the ~20 GB DiT, so
//! this is the biggest image-lane saving (36→20 GB, fits a 32 GB Mac). A repeat-job check confirms
//! nothing stays resident across jobs. Set `QWEN_SEQ_Q8=1` for the Q8 case, `QWEN_SEQ_STEPS`/
//! `QWEN_SEQ_SIZE` to tune.

// Force-link the provider crate so its `inventory` generator registration runs (this test only
// calls `mlx_gen::load("qwen_image", …)`).
use mlx_gen_qwen_image as _;

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // A fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here
    // (Resident vs Sequential, not a golden). Qwen-Image is true-CFG — the default (unset) sampler is
    // the production flow-match path with a negative branch, exercising two encode_prompt calls.
    let size = env_u32("QWEN_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("QWEN_SEQ_STEPS", 8)),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if std::env::var("QWEN_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen::load("qwen_image", &spec).expect("load qwen_image");
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
#[ignore = "needs a real Qwen/Qwen-Image snapshot (QWEN_IMAGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Qwen-Image {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("QWEN_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
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
#[ignore = "needs a real Qwen/Qwen-Image snapshot (QWEN_IMAGE_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Qwen-Image Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
