//! End to end vs upstream `iris3b.sampling.generate` on the miniature snapshot: tokenize the prompt
//! and the empty negative through the template → Qwen3-VL (bf16, as released) → FP32 backbone →
//! FlowDPM-Solver++ with CFG → clamp, from the same injected noise. The MLX twin's `e2e_parity`
//! fixture and bound.
//!
//! Tolerance: the backbone and solver run FP32 on Candle CPU (true f32, like the oracle); the text
//! tower is bf16 on both sides and differs by bf16 ulps there (see `text_parity`), which flows
//! through every CFG-amplified network evaluation. Bound 6e-2 on the [−1, 1] image (upstream's own
//! bf16-vs-fp32 tower moves this image by max |Δ| 4.0e-2). The FP32
//! backbone and solver alone are held to 1e-4 / 1e-5 by `dit_parity` / `solver_parity`.

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::GenerationParams;
use candle_gen::gen_core::{CancelFlag, GenerationRequest};
use candle_gen_iris::{denoise, encode, load_backbone, IrisTextEncoder};

use crate::common::{assert_close, cpu, fixture, tiny_backbone, tiny_config, tiny_text_encoder};

#[test]
fn generate_matches_upstream_from_injected_noise() {
    let config = tiny_config();
    let te = IrisTextEncoder::load(&tiny_text_encoder(), &config.text_encoder, &cpu()).unwrap();
    let dit = load_backbone(&tiny_backbone(), &config, DType::F32, &cpu()).unwrap();
    let golden = fixture("iris_e2e_golden.safetensors");
    let req = GenerationRequest {
        prompt: golden.meta("prompt").to_owned(),
        width: golden.meta("width").parse().unwrap(),
        height: golden.meta("height").parse().unwrap(),
        steps: Some(golden.meta("steps").parse().unwrap()),
        guidance: Some(golden.meta("cfg_scale").parse().unwrap()),
        seed: Some(0),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&req, 0);
    let conditioning = encode(&te, &req.prompt, &params).unwrap();
    assert!(
        conditioning.uncond.is_some(),
        "CFG > 1 encodes the empty-negative null"
    );
    let mut steps = Vec::new();
    let image = denoise(
        &dit,
        &config.flow,
        &conditioning,
        golden.require("noise"),
        &params,
        &CancelFlag::new(),
        |i| steps.push(i),
    )
    .unwrap();
    assert_eq!(steps, (1..=params.steps).collect::<Vec<_>>());
    assert_close("image", &image, golden.require("image"), 6e-2);
}
