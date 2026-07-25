//! Real-checkpoint acceptance for sc-14044.
//!
//! `MAGE_TURBO_SNAPSHOT` must point to `microsoft/Mage-Flow-Turbo`, never the RL checkpoint.

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_mage::model::{MAX_COUNT, REGISTRATION_TURBO};
use mlx_gen_mage::{GsKey, MageFlowPipeline};

fn turbo_snapshot() -> String {
    std::env::var("MAGE_TURBO_SNAPSHOT")
        .expect("set MAGE_TURBO_SNAPSHOT to the full microsoft/Mage-Flow-Turbo snapshot")
}

#[test]
#[ignore = "needs full MAGE_TURBO_SNAPSHOT and MAGE_RL_SNAPSHOT artifacts"]
fn turbo_identity_accepts_turbo_and_rejects_rl_substitution() {
    let turbo = LoadSpec::new(WeightsSource::Dir(turbo_snapshot().into()));
    (REGISTRATION_TURBO.load)(&turbo).unwrap();

    let rl = std::env::var("MAGE_RL_SNAPSHOT")
        .expect("set MAGE_RL_SNAPSHOT to the full microsoft/Mage-Flow RL snapshot");
    let err = (REGISTRATION_TURBO.load)(&LoadSpec::new(WeightsSource::Dir(rl.into())))
        .err()
        .expect("RL weights must not load under the Turbo registration");
    assert!(
        err.to_string().contains("checkpoint fingerprint mismatch"),
        "RL substitution failed for the wrong reason: {err}"
    );
}

#[test]
#[ignore = "needs full MAGE_TURBO_SNAPSHOT weights and an authorized Metal device"]
fn registered_turbo_renders_1024_at_exact_four_step_defaults() {
    let generator =
        (REGISTRATION_TURBO.load)(&LoadSpec::new(WeightsSource::Dir(turbo_snapshot().into())))
            .unwrap();
    let descriptor = generator.descriptor();
    assert_eq!(descriptor.id, "mage_flow_turbo");
    assert_eq!(descriptor.capabilities.max_count, MAX_COUNT);
    assert!(!descriptor.capabilities.supports_guidance);
    assert!(!descriptor.capabilities.supports_negative_prompt);

    let request = GenerationRequest {
        prompt: "a small red fox sleeping beneath a snow-covered pine tree".into(),
        width: 1024,
        height: 1024,
        seed: Some(14044),
        // Omitted deliberately: the registered Turbo variant must select steps=4 and cfg=1.
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
        panic!("Turbo returned a non-image output");
    };
    assert_eq!(steps, vec![(1, 4), (2, 4), (3, 4), (4, 4)]);
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
        "Turbo render has collapsed dynamic range: {min}..={max}"
    );
    let repeated_rows = image
        .pixels
        .chunks_exact(1024 * 3)
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|rows| rows[0] == rows[1])
        .count();
    assert!(
        repeated_rows < 102,
        "Turbo render has {repeated_rows} repeated adjacent rows"
    );
}

#[test]
#[ignore = "needs full MAGE_TURBO_SNAPSHOT weights and an authorized Metal device"]
fn cfg_one_never_encodes_or_uses_the_negative_prompt() {
    let pipeline = MageFlowPipeline::load(turbo_snapshot()).unwrap();
    let key = GsKey::from_u64(14044);
    let run = |negative: &str| {
        pipeline
            .generate(
                "a blue ceramic bowl on a wooden table",
                negative,
                512,
                512,
                1,
                1.0,
                14044,
                &key,
                false,
            )
            .unwrap()
    };
    let first = run("this text must never enter the encoder");
    let second = run("a completely different ignored negative prompt");
    mlx_rs::transforms::eval([&first, &second]).unwrap();
    assert_eq!(
        first.as_slice::<u8>(),
        second.as_slice::<u8>(),
        "cfg=1 changed when only the negative prompt changed"
    );
}
