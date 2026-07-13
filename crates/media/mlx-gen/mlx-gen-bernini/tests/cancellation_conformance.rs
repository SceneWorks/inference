//! Real-weight typed-cancellation conformance gate for `bernini_renderer` (story 6883, task 5).
//!
//! Binds the repo-wide typed-cancellation contract to the Bernini renderer: drives the registered
//! `Generator` through the reusable testkit check (`gen_core_testkit::check_cancellation_with`),
//! which trips `req.cancel` at the first emitted `Progress::Step` and asserts `generate()` returns
//! the typed `Err(Error::Canceled)` within ≤2 further steps. Uses the text-only t2i (`t2v_apg`) path
//! — the minimal valid request, mirroring `render_real.rs::t2i_real_weight_smoke`.
//!
//! `#[ignore]` because it assembles + loads the ~56 GB Bernini renderer snapshot (the crate's e2e
//! gating convention).

use std::path::PathBuf;

use mlx_gen::{GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_bernini::pipeline::MODEL_ID;
use mlx_gen_wan::convert::assemble_bernini_renderer_snapshot;

fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

/// Assemble the converted renderer snapshot once (reused across reruns), returning its dir
/// (mirrors `render_real.rs::ensure_snapshot`).
fn ensure_snapshot() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let snapshot = home.join(".cache/mlx-gen-models/bernini_renderer_mlx_bf16");
    if !snapshot.join("high_noise_model.safetensors").is_file() {
        let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
            .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
        let base = home.join(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16");
        assert!(
            base.join("high_noise_model.safetensors").is_file(),
            "converted base Wan2.2-T2V-A14B snapshot required at {}",
            base.display()
        );
        assemble_bernini_renderer_snapshot(&snapshot, &pkg, &base, None, true).expect("assemble");
    }
    snapshot
}

#[test]
#[ignore = "real weights: assembles + loads the ~56 GB Bernini renderer snapshot, runs a denoise"]
fn bernini_renderer_honors_typed_cancellation() {
    let gen = mlx_gen_bernini::provider_registry()
        .unwrap()
        .load(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(ensure_snapshot())),
        )
        .expect("load bernini_renderer");

    // Minimal valid t2i (text-only) request: 1 frame, 256² (within 16..=1280), `t2v_apg` guidance,
    // with step headroom so a honoring provider visibly stops before completion.
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(6),
        seed: Some(0),
        video_mode: Some("t2v_apg".into()),
        ..Default::default()
    };

    gen_core_testkit::check_cancellation_with(gen.as_ref(), &req)
        .expect("bernini_renderer must honor the typed-cancellation contract");
}
