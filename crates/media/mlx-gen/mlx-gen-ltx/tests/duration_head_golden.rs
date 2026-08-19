//! `DurationHead` golden parity vs the v1.2.0 reference (sc-18774).
//!
//! `#[ignore]`d: needs the real (~4 MB) `model_patches/ltx-2.5-duration-head-bf16.safetensors`. The
//! committed golden (`tests/fixtures/ltx_duration_head_golden.safetensors`, from
//! `tools/dump_ltx_duration_head_golden.py`) holds the reference f32 input/output for all three
//! modality combinations (video-only, audio-only, both); this test loads the SAME real weights and
//! checks the Rust `DurationHead` reproduces the predicted seconds.
//!
//! Run:
//! ```text
//! LTX25_DURATION_HEAD_FILE=/path/to/ltx-2.5-duration-head-bf16.safetensors \
//!   cargo test -p mlx-gen-ltx --test duration_head_golden -- --ignored --nocapture
//! ```

use mlx_gen::weights::Weights;
use mlx_gen_ltx::DurationHead;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_duration_head_golden.safetensors"
);

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

/// Relative error, matching the tolerance style used elsewhere in this crate (e.g.
/// `connector_parity.rs`'s `peak_rel`). Both sides do an identical bf16->f32 upcast then f32
/// arithmetic; the tolerance covers PyTorch's fused `nn.MultiheadAttention` kernel vs this port's
/// manual matmul/softmax/matmul taking a (numerically equivalent, not bit-identical) reduction
/// order.
fn rel_err(got: f32, want: f32) -> f32 {
    (got - want).abs() / want.abs().max(1e-6)
}

#[test]
#[ignore = "needs the real ltx-2.5-duration-head-bf16.safetensors (~4 MB)"]
fn duration_head_matches_reference_all_modalities() {
    let path = duration_head_file();
    let w = Weights::from_file(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
    let head = DurationHead::from_weights(&w).expect("build DurationHead");

    let g = Weights::from_file(GOLDEN).expect("golden");
    let video = g.require("video_tokens").unwrap();
    let audio = g.require("audio_tokens").unwrap();

    let cases: [(&str, Option<&mlx_rs::Array>, Option<&mlx_rs::Array>); 3] = [
        ("video_only", Some(video), None),
        ("audio_only", None, Some(audio)),
        ("both", Some(video), Some(audio)),
    ];
    for (name, v, a) in cases {
        let want = g.require(&format!("seconds_{name}")).unwrap().item::<f32>();
        let got = head.predict_seconds(v, a).expect("predict_seconds");
        let err = rel_err(got, want);
        eprintln!("{name}: got={got:.6} want={want:.6} rel_err={err:.3e}");
        assert!(
            err < 5e-3,
            "{name}: rel_err {err:.3e} too high (got {got}, want {want})"
        );
    }
}
