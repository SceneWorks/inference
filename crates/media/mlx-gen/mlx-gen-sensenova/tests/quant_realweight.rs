//! sc-3193: real-weight (35GB) Q4/Q8 vs bf16 T2I parity — `#[ignore]`, run locally.
//!
//! Loads `sensenova_u1_8b` at bf16, Q8, and Q4 via the registry and generates the same T2I prompt
//! at a fixed seed through each. Quantization is a quality knob, not a bit-parity target, so the
//! gate is directional: Q8 ≈ bf16 (near-lossless) and Q4 coherent. Validates the backbone quant
//! seam end to end on real weights + the footprint win (Q4 loads where bf16's ~35 GB is heavy).
//!
//! Run: `cargo test -p mlx-gen-sensenova --test integration quant_realweight:: -- --ignored --nocapture`

use mlx_gen::testing::{cosine_f64_sums, require_env_path, Denom};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant, WeightsSource,
};

fn render(spec: &LoadSpec) -> Image {
    let model = mlx_gen_sensenova::provider_registry()
        .unwrap()
        .load("sensenova_u1_8b", spec)
        .expect("load");
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        count: 1,
        steps: Some(8),
        guidance: Some(2.0),
        seed: Some(42),
        ..Default::default()
    };
    let mut noop = |_: Progress| {};
    match model.generate(&req, &mut noop).expect("generate") {
        GenerationOutput::Images(mut imgs) => imgs.pop().unwrap(),
        _ => panic!("expected Images"),
    }
}

#[test]
#[ignore = "needs the local 35GB checkpoint; run with --ignored"]
fn quant_vs_bf16_t2i() {
    let snap = require_env_path("SENSENOVA_SNAPSHOT", "the SenseNova-U1-8B-MoT snapshot dir");
    if !snap.exists() {
        eprintln!("skipping: snapshot missing at {}", snap.display());
        crate::model_inference_device::record_model_inference_prerequisite_missing(
            "required work unavailable in quant_vs_bf16_t2i; see preceding skip diagnostic",
        );
        return;
    }
    let src = WeightsSource::Dir(snap);
    let bf16 = render(&LoadSpec::new(src.clone()));
    let q8 = render(&LoadSpec::new(src.clone()).with_quant(Quant::Q8));
    let q4 = render(&LoadSpec::new(src).with_quant(Quant::Q4));

    let c8 = cosine_f64_sums(&bf16.pixels, &q8.pixels, Denom::PlusEps(1e-12));
    let c4 = cosine_f64_sums(&bf16.pixels, &q4.pixels, Denom::PlusEps(1e-12));
    println!("Q8 vs bf16 cosine={c8:.5}   Q4 vs bf16 cosine={c4:.5}");
    // Q8 is near-lossless (the fidelity footprint choice). Q4 is the aggressive footprint mode:
    // 4-bit weight error, amplified by CFG (the precision sensitivity from sc-3189), shifts the
    // image to recognizably-the-same-content but not pixel-close — a quality/size tradeoff, coherent
    // not bit-faithful. (Measured ~0.998 / ~0.84 at cfg=2, 8-step, 256².)
    assert!(c8 > 0.97, "Q8 should be near-lossless vs bf16, got {c8:.5}");
    assert!(c4 > 0.8, "Q4 should stay coherent vs bf16, got {c4:.5}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED quant_realweight::quant_vs_bf16_t2i");
}
