//! **Weights-free** named-component load-gate conformance (epic 13657, sc-13666) for both shipping
//! MMAudio providers. Proves each provider converts a missing OR unrecognized
//! [`LoadSpec::components`] entry into a **load-time** error (via `gen_core::require_component` /
//! `gen_core::reject_unknown_components`) rather than a mid-render surprise — driven against the real
//! `load` closures through the testkit's [`check_component_load_gate`]. No weights are read: `load` is
//! lazy, so this runs in the ordinary CPU lane (not `#[ignore]`d).

use candle_audio_mmaudio::candle_audio::gen_core::{LoadSpec, WeightsSource};

/// A base spec that stages every required component with a placeholder path (never read — the gate
/// only exercises the load-time validators). `weights` is an ignored placeholder for mmaudio.
fn staged(required: &[&str]) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(
        std::env::temp_dir().join("mmaudio-unused-base"),
    ));
    for id in required {
        spec = spec.with_component(
            *id,
            WeightsSource::File(std::path::PathBuf::from(format!("/nonexistent/{id}.bin"))),
        );
    }
    spec
}

#[test]
fn small_16k_gates_missing_and_unknown_components_at_load() {
    let required = candle_audio_mmaudio::generator::descriptor().required_components;
    gen_core_testkit::check_component_load_gate(
        candle_audio_mmaudio::generator::load,
        &staged(required),
        required,
    )
    .expect("mmaudio_small_16k gates every required + unknown component at load");
}

#[test]
fn large_44k_gates_missing_and_unknown_components_at_load() {
    let required = candle_audio_mmaudio::generator_44k::descriptor().required_components;
    gen_core_testkit::check_component_load_gate(
        candle_audio_mmaudio::generator_44k::load,
        &staged(required),
        required,
    )
    .expect("mmaudio_large_44k gates every required + unknown component at load");
}
