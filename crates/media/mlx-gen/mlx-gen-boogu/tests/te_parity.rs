//! E2 — real-weight parity for the Boogu Qwen3-VL-8B condition encoder against the reference
//! `last_hidden_state` captured by `tools/golden_dump.py`.
//!
//! `#[ignore]` — needs the Base snapshot (`mllm/`) and the golden file. Run:
//!   BOOGU_BASE_DIR=<snapshot> BOOGU_GOLDEN=<...>/boogu_golden.safetensors \
//!     CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test integration te_parity:: -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::testing::{cosine_f32_device, require_env_path};
use mlx_gen::weights::Weights;
use mlx_gen_boogu::{load_text_encoder, load_transformer};

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_golden.safetensors")
        })
}

#[test]
#[ignore = "needs real weights + golden (tools/golden_dump.py)"]
fn te_matches_reference_last_hidden() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/golden_dump.py");
    let te = load_text_encoder(require_env_path(
        "BOOGU_BASE_DIR",
        "the Boogu snapshot root",
    ))
    .expect("load Qwen3-VL condition encoder");
    let out = te
        .last_hidden(
            g.require("tok_input_ids").unwrap(),
            g.require("tok_attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("instruction_hidden_states").unwrap();
    assert_eq!(out.shape(), want.shape(), "last_hidden_state shape");

    let c = cosine_f32_device(&out, want);
    println!("Boogu Qwen3-VL TE parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "TE parity cosine {c} too low — structural mismatch"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED te_parity::te_matches_reference_last_hidden");
}

/// E3 — real-weight parity for the DiT velocity. Feeds the golden's `dit_in_latent_chw`, `timestep`,
/// raw `instruction_hidden_states`, and `instruction_attention_mask` through the full mixed-stream
/// forward and compares to the reference `dit_out_velocity_chw`. Run:
///   BOOGU_BASE_DIR=<snapshot> BOOGU_GOLDEN=<...>/boogu_golden.safetensors \
///     CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
///     cargo test -p mlx-gen-boogu --test integration te_parity::dit_matches -- --ignored --nocapture
#[test]
#[ignore = "needs real weights + golden (tools/golden_dump.py)"]
fn dit_matches_reference_velocity() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/golden_dump.py");
    let dit = load_transformer(require_env_path(
        "BOOGU_BASE_DIR",
        "the Boogu snapshot root",
    ))
    .expect("load Boogu DiT");

    // The golden saves the latent / velocity already batched as [1, C, H, W].
    let out = dit
        .forward(
            g.require("dit_in_latent_chw").unwrap(),
            g.require("timestep").unwrap(),
            g.require("instruction_hidden_states").unwrap(),
            g.require("instruction_attention_mask").unwrap(),
        )
        .unwrap();

    let want = g.require("dit_out_velocity_chw").unwrap();
    assert_eq!(out.shape(), want.shape(), "velocity shape");

    let c = cosine_f32_device(&out, want);
    println!("Boogu DiT velocity parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "DiT velocity parity cosine {c} too low — structural mismatch"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED te_parity::dit_matches_reference_velocity");
}
