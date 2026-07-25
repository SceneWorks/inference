//! Real-checkpoint acceptance for sc-14045.
//!
//! `MAGE_BASE_SNAPSHOT` must point to `microsoft/Mage-Flow-Base`, never the RL or Turbo checkpoint.

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_mage::model::{MAX_COUNT, REGISTRATION_BASE};
use mlx_gen_mage::{GsKey, MageFlowPipeline};

fn base_snapshot() -> String {
    std::env::var("MAGE_BASE_SNAPSHOT")
        .expect("set MAGE_BASE_SNAPSHOT to the full microsoft/Mage-Flow-Base snapshot")
}

#[test]
#[ignore = "needs full Base, RL, and Turbo snapshot artifacts"]
fn base_identity_accepts_base_and_rejects_other_generation_checkpoints() {
    let base = LoadSpec::new(WeightsSource::Dir(base_snapshot().into()));
    (REGISTRATION_BASE.load)(&base).unwrap();

    for (label, variable) in [("RL", "MAGE_RL_SNAPSHOT"), ("Turbo", "MAGE_TURBO_SNAPSHOT")] {
        let substituted = std::env::var(variable)
            .unwrap_or_else(|_| panic!("set {variable} to the full Mage-Flow {label} snapshot"));
        let err = (REGISTRATION_BASE.load)(&LoadSpec::new(WeightsSource::Dir(substituted.into())))
            .err()
            .unwrap_or_else(|| panic!("{label} weights must not load under the Base registration"));
        assert!(
            err.to_string().contains("checkpoint fingerprint mismatch"),
            "{label} substitution failed for the wrong reason: {err}"
        );
    }
}

#[test]
#[ignore = "needs full MAGE_BASE_SNAPSHOT weights and an authorized Metal device"]
fn registered_base_renders_1024_at_exact_thirty_step_cfg_five_defaults() {
    let generator =
        (REGISTRATION_BASE.load)(&LoadSpec::new(WeightsSource::Dir(base_snapshot().into())))
            .unwrap();
    let descriptor = generator.descriptor();
    assert_eq!(descriptor.id, "mage_flow_base");
    assert_eq!(descriptor.capabilities.max_count, MAX_COUNT);
    assert!(descriptor.capabilities.supports_guidance);
    assert!(descriptor.capabilities.supports_negative_prompt);

    let request = GenerationRequest {
        prompt: "a glass greenhouse glowing at dusk in a snowy botanical garden".into(),
        negative_prompt: Some("blurry, distorted, low detail".into()),
        width: 1024,
        height: 1024,
        seed: Some(14045),
        // Omitted deliberately: Base must select the published steps=30 and cfg=5 defaults.
        steps: None,
        guidance: None,
        ..Default::default()
    };
    let mut steps = Vec::new();
    let GenerationOutput::Images(images) = generator
        .generate(&request, &mut |progress| {
            if let Progress::Step { current, total } = progress {
                steps.push((current, total));
            }
        })
        .unwrap()
    else {
        panic!("Base returned a non-image output");
    };
    assert_eq!(
        steps,
        (1..=30).map(|current| (current, 30)).collect::<Vec<_>>()
    );
    assert_eq!(images.len(), 1);
    let image = &images[0];
    assert_eq!((image.width, image.height), (1024, 1024));
    assert_eq!(image.pixels.len(), 1024 * 1024 * 3);
    let (min, max) = image
        .pixels
        .iter()
        .fold((u8::MAX, u8::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(
        max.saturating_sub(min) >= 32,
        "Base render has collapsed dynamic range: {min}..={max}"
    );
    let repeated_rows = image
        .pixels
        .chunks_exact(1024 * 3)
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|rows| rows[0] == rows[1])
        .count();
    println!(
        "Base render dynamic range {min}..={max}; repeated adjacent rows {repeated_rows}/1023"
    );
    assert!(
        repeated_rows < 102,
        "Base render has {repeated_rows} repeated adjacent rows"
    );
}

#[test]
#[ignore = "needs full MAGE_BASE_SNAPSHOT weights and an authorized Metal device"]
fn cfg_five_uses_the_negative_prompt() {
    let pipeline = MageFlowPipeline::load(base_snapshot()).unwrap();
    let key = GsKey::from_u64(14045);
    let run = |negative: &str| {
        pipeline
            .generate(
                "a blue ceramic bowl on a wooden table",
                negative,
                512,
                512,
                1,
                5.0,
                14045,
                &key,
                false,
            )
            .unwrap()
    };
    let first = run("photograph, sharp focus");
    let second = run("oil painting, soft focus");
    mlx_rs::transforms::eval([&first, &second]).unwrap();
    assert_ne!(
        first.as_slice::<u8>(),
        second.as_slice::<u8>(),
        "cfg=5 output ignored a changed negative prompt"
    );
}
