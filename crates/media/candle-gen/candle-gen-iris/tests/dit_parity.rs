//! Masked backbone + pixel-head forward vs upstream `IrisDiT.forward` on the miniature checkpoint
//! (every switch of the release: hybrid trunk, GQA, attention gate, sandwich norm, shared-bias
//! adaLN, isotropic RoPE on a NON-square 3×2 patch grid, text RoPE + position table, `lap_blocks2`
//! adapter with unequal masks across the batch, post-modulation PiT head) — the same fixture the
//! MLX twin's `dit_parity` reads.
//!
//! Tolerances:
//! * FP32 (`Precision::Fp32` compute, Candle CPU) vs upstream's FP32 CPU path: only summation
//!   order differs — 1e-4 of peak (the MLX twin's bound).
//! * bf16 compute (the release's autocast policy) vs the FP32 oracle: bf16 matmul and attention
//!   rounding — 5e-2 of peak (the MLX twin's bound). Runs on the build's device, so the CUDA lane
//!   holds the real bf16 kernels to the same fixture.

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::IrisConfig;
use candle_gen_iris::{load_backbone, TextBatch};

use crate::common::{assert_close, cpu, device, fixture, host_i32, tiny_backbone};

fn masks(flat: Vec<i32>, rows: usize) -> Vec<Vec<i32>> {
    flat.chunks(flat.len() / rows)
        .map(<[i32]>::to_vec)
        .collect()
}

#[test]
fn backbone_and_pixel_head_match_upstream_in_fp32() {
    let config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
    let dit = load_backbone(&tiny_backbone(), &config, DType::F32, &cpu()).unwrap();
    let golden = fixture("iris_dit_golden.safetensors");
    let mask = masks(host_i32(golden.require("y_mask")), 2);
    let text = TextBatch {
        states: golden.require("y"),
        mask: &mask,
    };
    let adapter = dit.text_adapter(&text).unwrap();
    assert_close(
        "layerwise text adapter",
        &adapter,
        golden.require("adapter_out"),
        1e-4,
    );
    let out = dit
        .forward(golden.require("x"), golden.require("t"), &text)
        .unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_close("velocity", &out, golden.require("out"), 1e-4);
}

#[test]
fn bf16_compute_stays_within_bf16_distance_of_fp32() {
    let config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
    let device = device();
    let dit = load_backbone(&tiny_backbone(), &config, DType::BF16, &device).unwrap();
    let golden = fixture("iris_dit_golden.safetensors");
    let on = |key: &str| golden.require(key).to_device(&device).unwrap();
    let mask = masks(host_i32(golden.require("y_mask")), 2);
    let states = on("y");
    let text = TextBatch {
        states: &states,
        mask: &mask,
    };
    let out = dit.forward(&on("x"), &on("t"), &text).unwrap();
    eprintln!("bf16 velocity device: {:?}", out.device().location());
    assert_eq!(out.dtype(), DType::F32);
    assert_close("velocity (bf16 compute)", &out, golden.require("out"), 5e-2);
}

#[test]
fn a_config_that_disagrees_with_the_checkpoint_is_a_load_error() {
    let mut config = IrisConfig::from_dir(&tiny_backbone()).unwrap();
    config.model.depth = 2; // drops block 2 → its keys are left over
    let err = load_backbone(&tiny_backbone(), &config, DType::F32, &cpu())
        .err()
        .expect("leftover keys must fail");
    assert!(err.to_string().contains("blocks.2"), "{err}");
}
