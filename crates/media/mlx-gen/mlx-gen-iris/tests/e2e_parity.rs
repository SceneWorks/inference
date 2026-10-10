//! End to end vs upstream `iris3b.sampling.generate` on the miniature snapshot: tokenize the prompt
//! and the empty negative through the template → Qwen3-VL (bf16, as released) → FP32 backbone →
//! 6-step FlowDPM-Solver++ with CFG 3 → clamp, from the same injected noise.
//!
//! Tolerance: the backbone and solver run FP32 on the MLX CPU stream (true f32, like the oracle);
//! the text tower is bf16 on both sides and MLX vs torch-CPU differ by 1–2 bf16 ulps there (see
//! `text_parity`), which flows through six CFG-3-amplified network evaluations. Measured max |Δ|
//! 1.5e-2 (mean 1.4e-3) on the [−1, 1] image; bound 2e-2. The FP32 backbone and solver alone are
//! held to 1e-4 / 1e-5 by `dit_parity` / `solver_parity`.

use mlx_gen::gen_core::iris::GenerationParams;
use mlx_gen::{CancelFlag, GenerationRequest};
use mlx_gen_iris::{denoise, encode, load_backbone, IrisTextEncoder};
use mlx_rs::Dtype;

use crate::common::{assert_close, fixture, on_cpu, tiny_backbone, tiny_config, tiny_text_encoder};

#[test]
fn generate_matches_upstream_from_injected_noise() {
    on_cpu(run);
}

fn run() {
    let config = tiny_config();
    let te = IrisTextEncoder::load(&tiny_text_encoder(), &config.text_encoder).unwrap();
    let dit = load_backbone(&tiny_backbone(), &config, Dtype::Float32).unwrap();
    let golden = fixture("iris_e2e_golden.safetensors");
    let req = GenerationRequest {
        prompt: "a red fox in the snow".into(),
        width: 12,
        height: 8,
        steps: Some(6),
        guidance: Some(3.0),
        seed: Some(0),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&req, 0);
    let conditioning = encode(&te, &req.prompt, &params).unwrap();
    assert!(
        conditioning.uncond.is_some(),
        "CFG 3 encodes the empty-negative null"
    );
    let mut steps = Vec::new();
    let image = denoise(
        &dit,
        &config.flow,
        &conditioning,
        golden.require("noise").unwrap(),
        &params,
        &CancelFlag::new(),
        |i| steps.push(i),
    )
    .unwrap();
    assert_eq!(steps, [1, 2, 3, 4, 5, 6]);
    assert_close("image", &image, golden.require("image").unwrap(), 2e-2);
}
