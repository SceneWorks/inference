//! Physical q4/q8/bf16 acceptance for sc-14053.
//!
//! This is intentionally ignored in ordinary CI: it loads the complete checkpoint and must run on
//! the CUDA lane with enough host RAM/VRAM. It drives the registered production loader, so removing
//! the quant from `LoadSpec` or leaving the DiT dense makes the tier checks fail at the actual seam.

use candle_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use candle_gen_mage::BASE_REGISTRATION;

fn snapshot() -> std::path::PathBuf {
    std::env::var_os("CANDLE_MAGE_SNAPSHOT")
        .map(Into::into)
        .expect("set CANDLE_MAGE_SNAPSHOT to a complete Mage-Flow-Base snapshot")
}

fn generate(quant: Option<Quant>, prompt: &str) -> Vec<u8> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = quant;
    let generator = (BASE_REGISTRATION.load)(&spec).expect("registered Mage load");
    let output = generator
        .generate(
            &GenerationRequest {
                prompt: prompt.into(),
                width: 512,
                height: 512,
                steps: Some(4),
                guidance: Some(1.0),
                seed: Some(14053),
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("Mage generation");
    match output {
        GenerationOutput::Images(images) => images.into_iter().next().expect("one image").pixels,
        _ => panic!("expected image output"),
    }
}

fn mae(left: &[u8], right: &[u8]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a.abs_diff(b)))
        .sum::<f64>()
        / left.len() as f64
}

#[test]
#[ignore = "requires CANDLE_MAGE_SNAPSHOT plus a physical CUDA runner"]
fn all_three_registered_tiers_render_and_preserve_prompt_discrimination() {
    let dense = generate(None, "a red fox beneath a pine tree");
    let q8 = generate(Some(Quant::Q8), "a red fox beneath a pine tree");
    let q4 = generate(Some(Quant::Q4), "a red fox beneath a pine tree");
    let q4_mutated = generate(Some(Quant::Q4), "a blue sailboat on a stormy sea");

    for (name, image) in [("bf16", &dense), ("q8", &q8), ("q4", &q4)] {
        assert_eq!(image.len(), 512 * 512 * 3, "{name}: image geometry");
        let (min, max) = image.iter().fold((u8::MAX, u8::MIN), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
        assert!(
            max.saturating_sub(min) >= 96,
            "{name}: collapsed dynamic range {min}..{max}"
        );
    }

    let q8_error = mae(&q8, &dense);
    let q4_error = mae(&q4, &dense);
    let prompt_delta = mae(&q4, &q4_mutated);
    assert!(
        q8_error < q4_error,
        "Q8 must remain closer to BF16: q8={q8_error:.3}, q4={q4_error:.3}"
    );
    assert!(
        prompt_delta >= 5.0,
        "Q4 ignored the prompt mutation: MAE {prompt_delta:.3}"
    );
}
