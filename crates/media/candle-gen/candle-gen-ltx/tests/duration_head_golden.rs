//! `DurationHead` golden parity vs the v1.2.0 reference (sc-18774) — the candle sibling of
//! `mlx-gen-ltx/tests/duration_head_golden.rs`. See that file's docs for the golden's provenance
//! (`tools/dump_ltx_duration_head_golden.py`, run once against the real weights and copied into
//! both crates' `tests/fixtures/`).
//!
//! `#[ignore]`d: needs the real (~4 MB) `model_patches/ltx-2.5-duration-head-bf16.safetensors`.
//!
//! Run:
//! ```text
//! LTX25_DURATION_HEAD_FILE=/path/to/ltx-2.5-duration-head-bf16.safetensors \
//!   cargo test -p candle-gen-ltx --test duration_head_golden -- --ignored --nocapture
//! ```

use candle_gen::candle_core::{DType, Device};
use candle_gen::weights::Weights;
use candle_gen_ltx::duration_head::DurationHead;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_duration_head_golden.safetensors"
);

/// The three modality combinations the golden records, in the order its `seconds_*` keys are named.
pub const SHARED_FIXTURE_DURATION_HEAD_CASES: [&str; 3] = ["video_only", "audio_only", "both"];
/// Relative-error ceiling the golden's predicted seconds are held to on both lanes.
pub const SHARED_FIXTURE_DURATION_HEAD_REL_TOLERANCE: f32 = 5e-3;

fn duration_head_file() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LTX25_DURATION_HEAD_FILE") {
        return p.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home).join(
        "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_5/model_patches/\
         ltx-2.5-duration-head-bf16.safetensors",
    )
}

fn rel_err(got: f32, want: f32) -> f32 {
    (got - want).abs() / want.abs().max(1e-6)
}

#[test]
#[ignore = "needs the real ltx-2.5-duration-head-bf16.safetensors (~4 MB)"]
fn duration_head_matches_reference_all_modalities() {
    let device = Device::Cpu;
    let path = duration_head_file();
    let w = Weights::from_file(&path, &device, DType::F32)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
    let head = DurationHead::from_weights(&w, &device).expect("build DurationHead");

    let golden_device = Device::Cpu;
    let g = Weights::from_file(
        &std::path::PathBuf::from(GOLDEN),
        &golden_device,
        DType::F32,
    )
    .expect("golden");
    let video = g.require("video_tokens").unwrap();
    let audio = g.require("audio_tokens").unwrap();

    let [video_only, audio_only, both] = SHARED_FIXTURE_DURATION_HEAD_CASES;
    for (name, v, a) in [
        (video_only, Some(&video), None),
        (audio_only, None, Some(&audio)),
        (both, Some(&video), Some(&audio)),
    ] {
        let want_t = g.require(&format!("seconds_{name}")).unwrap();
        let want: f32 = want_t
            .reshape(())
            .unwrap()
            .to_scalar()
            .expect("golden scalar");
        let got = head.predict_seconds(v, a).expect("predict_seconds");
        let err = rel_err(got, want);
        eprintln!("{name}: got={got:.6} want={want:.6} rel_err={err:.3e}");
        assert!(
            err < SHARED_FIXTURE_DURATION_HEAD_REL_TOLERANCE,
            "{name}: rel_err {err:.3e} too high (got {got}, want {want})"
        );
    }
}
