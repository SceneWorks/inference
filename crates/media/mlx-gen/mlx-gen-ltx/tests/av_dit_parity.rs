//! S2 AudioVideo-DiT (velocity) parity vs the reference joint `LTXModel(video, audio)` (sc-2684).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB). The committed
//! goldens (`tests/fixtures/ltx_av_dit_golden{,_bf16}.safetensors`, from
//! `tools/dump_ltx_av_dit_golden.py`) hold the reference video + audio velocities over synthetic
//! joint inputs; this test loads the SAME Q8 weights into the Rust `AvDiT` and checks BOTH velocities
//! reproduce — the bf16 path bit-exact, the f32 path within the tight cross-stack bounds below (the
//! distilled sampler is chaos-sensitive, so the dual-stream forward stays as tight as attainable).
//!
//! **The goldens MUST match the Rust build's MLX — now 0.32.0** (re-dumped sc-12896 on the non-NAX
//! from-source env; see `dit_parity.rs` for the full 0.32.0 cross-stack contract rationale).
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test av_dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::transformer::{AvDiT, Precision};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_dit_golden.safetensors"
);
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_dit_golden_bf16.safetensors"
);

/// sc-12896: cross-stack f32 bounds at matched MLX 0.32.0 (see `dit_parity.rs` for the mechanism).
/// Measured (2026-07-18, non-NAX dt15.0, fresh goldens): video peak_rel 3.553e-5 / mean_rel
/// 3.248e-5; audio peak_rel 1.525e-6 / mean_rel 9.679e-7. Bounds ~4-6× the measurements; the bf16
/// path stays asserted exact (0.0).
const AV_F32_XSTACK_VIDEO_PEAK_REL: f32 = 2.0e-4;
const AV_F32_XSTACK_VIDEO_MEAN_REL: f32 = 1.5e-4;
const AV_F32_XSTACK_AUDIO_PEAK_REL: f32 = 1.0e-5;
const AV_F32_XSTACK_AUDIO_MEAN_REL: f32 = 1.0e-5;

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), want).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(want).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

fn run(bf16: bool, golden: &str) {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    // Quant geometry (bits/group) rides on `split_model.json` (sc-2686).
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let prec = if bf16 {
        Precision::quant_bf16(split.bits, split.group)
    } else {
        Precision::quant_f32(split.bits, split.group)
    };
    let w =
        Weights::from_file(dir.join("transformer.safetensors")).expect("transformer.safetensors");
    let dit = AvDiT::from_weights(&w, &cfg, prec).expect("build AvDiT");
    let g = Weights::from_file(golden).expect("golden (run tools/dump_ltx_av_dit_golden.py)");

    let (v_vel, a_vel) = dit
        .forward(
            g.require("video_latent").unwrap(),
            g.require("video_timestep").unwrap(),
            g.require("video_context").unwrap(),
            None,
            g.require("video_positions").unwrap(),
            g.require("audio_latent").unwrap(),
            g.require("audio_timestep").unwrap(),
            g.require("audio_context").unwrap(),
            None,
            g.require("audio_positions").unwrap(),
            None,
        )
        .expect("av dit forward");

    let want_v = g.require("video_velocity").unwrap();
    let want_a = g.require("audio_velocity").unwrap();
    assert_eq!(v_vel.shape(), want_v.shape(), "video velocity shape");
    assert_eq!(a_vel.shape(), want_a.shape(), "audio velocity shape");
    let (pvr, mvr) = (peak_rel(&v_vel, want_v), mean_rel(&v_vel, want_v));
    let (par, mar) = (peak_rel(&a_vel, want_a), mean_rel(&a_vel, want_a));
    eprintln!("av dit ({prec:?}): video peak_rel {pvr:.3e} mean_rel {mvr:.3e} | audio peak_rel {par:.3e} mean_rel {mar:.3e}");

    // sc-7141: the per-stage RoPE epoch fast path keys all FOUR stream memos (video/audio × self/cross).
    // Re-run with `Some(epoch)` on the SAME inputs and assert both velocities are byte-identical to the
    // content path — transitively gating the epoch path against the reference golden on real weights.
    let (v_epoch, a_epoch) = dit
        .forward(
            g.require("video_latent").unwrap(),
            g.require("video_timestep").unwrap(),
            g.require("video_context").unwrap(),
            None,
            g.require("video_positions").unwrap(),
            g.require("audio_latent").unwrap(),
            g.require("audio_timestep").unwrap(),
            g.require("audio_context").unwrap(),
            None,
            g.require("audio_positions").unwrap(),
            Some(dit.next_rope_epoch()),
        )
        .expect("av dit forward (epoch path)");
    mlx_rs::transforms::eval([&v_vel, &a_vel, &v_epoch, &a_epoch]).unwrap();
    assert_eq!(
        f32(&v_epoch).as_slice::<f32>(),
        f32(&v_vel).as_slice::<f32>(),
        "sc-7141: epoch-path video velocity must be byte-identical to the content path"
    );
    assert_eq!(
        f32(&a_epoch).as_slice::<f32>(),
        f32(&a_vel).as_slice::<f32>(),
        "sc-7141: epoch-path audio velocity must be byte-identical to the content path"
    );

    // bf16 stays bit-exact; f32 uses the sc-12896 cross-stack bounds (see dit_parity.rs).
    if bf16 {
        assert!(
            pvr == 0.0 && mvr == 0.0,
            "bf16 video velocity not bit-exact (peak {pvr:.3e} mean {mvr:.3e})"
        );
        assert!(
            par == 0.0 && mar == 0.0,
            "bf16 audio velocity not bit-exact (peak {par:.3e} mean {mar:.3e})"
        );
    } else {
        assert!(
            pvr <= AV_F32_XSTACK_VIDEO_PEAK_REL && mvr <= AV_F32_XSTACK_VIDEO_MEAN_REL,
            "f32 video velocity exceeds the 0.32.0 cross-stack bounds (peak {pvr:.3e} mean {mvr:.3e}, sc-12896)"
        );
        assert!(
            par <= AV_F32_XSTACK_AUDIO_PEAK_REL && mar <= AV_F32_XSTACK_AUDIO_MEAN_REL,
            "f32 audio velocity exceeds the 0.32.0 cross-stack bounds (peak {par:.3e} mean {mar:.3e}, sc-12896)"
        );
    }
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn av_dit_velocity_matches_reference() {
    run(false, GOLDEN);
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn av_dit_velocity_matches_reference_bf16() {
    run(true, GOLDEN_BF16);
}
