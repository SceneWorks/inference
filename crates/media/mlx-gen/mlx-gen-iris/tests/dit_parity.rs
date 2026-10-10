//! Masked backbone + pixel-head forward vs upstream `IrisDiT.forward` on the miniature checkpoint
//! (every switch of the release: hybrid trunk, GQA, attention gate, sandwich norm, shared-bias
//! adaLN, isotropic RoPE on a NON-square 3×2 patch grid, text RoPE + position table, `lap_blocks2`
//! adapter with unequal masks across the batch, post-modulation PiT head).
//!
//! Tolerances:
//! * FP32 (`Precision::Fp32` compute, MLX CPU stream) vs upstream's FP32 CPU path: only summation
//!   order differs — 1e-4 of peak (measured ~2e-6 adapter, ~3e-5 velocity).
//! * bf16 compute (the release's autocast policy) on the GPU vs the FP32 oracle: bf16 matmul and
//!   attention rounding — 5e-2 of peak.

use mlx_gen::gen_core::iris::IrisConfig;
use mlx_gen_iris::{load_backbone, TextBatch};
use mlx_rs::Dtype;

use crate::common::{assert_close, fixture, host_i32, on_cpu, tiny_backbone};

fn masks(flat: Vec<i32>, rows: usize) -> Vec<Vec<i32>> {
    flat.chunks(flat.len() / rows)
        .map(<[i32]>::to_vec)
        .collect()
}

#[test]
fn backbone_and_pixel_head_match_upstream_in_fp32() {
    on_cpu(|| {
        let config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
        let dit = load_backbone(&tiny_backbone(), &config, Dtype::Float32).unwrap();
        let golden = fixture("iris_dit_golden.safetensors");
        let mask = masks(host_i32(golden.require("y_mask").unwrap()), 2);
        let text = TextBatch {
            states: golden.require("y").unwrap(),
            mask: &mask,
        };
        let adapter = dit.text_adapter(&text).unwrap();
        assert_close(
            "layerwise text adapter",
            &adapter,
            golden.require("adapter_out").unwrap(),
            1e-4,
        );
        let out = dit
            .forward(
                golden.require("x").unwrap(),
                golden.require("t").unwrap(),
                &text,
            )
            .unwrap();
        assert_close("velocity", &out, golden.require("out").unwrap(), 1e-4);
    });
}

#[test]
fn bf16_compute_stays_within_bf16_distance_of_fp32() {
    let config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
    let dit = load_backbone(&tiny_backbone(), &config, Dtype::Bfloat16).unwrap();
    let golden = fixture("iris_dit_golden.safetensors");
    let mask = masks(host_i32(golden.require("y_mask").unwrap()), 2);
    let text = TextBatch {
        states: golden.require("y").unwrap(),
        mask: &mask,
    };
    let out = dit
        .forward(
            golden.require("x").unwrap(),
            golden.require("t").unwrap(),
            &text,
        )
        .unwrap();
    assert_eq!(out.dtype(), Dtype::Float32);
    assert_close(
        "velocity (bf16 compute)",
        &out,
        golden.require("out").unwrap(),
        5e-2,
    );
}

#[test]
fn a_config_that_disagrees_with_the_checkpoint_is_a_load_error() {
    let mut config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
    config.model.depth = 2; // drops block 2 → its keys are left over
    let err = load_backbone(&tiny_backbone(), &config, Dtype::Float32)
        .err()
        .expect("leftover keys must fail");
    assert!(err.to_string().contains("blocks.2"), "{err}");
}
