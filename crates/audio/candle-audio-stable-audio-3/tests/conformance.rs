//! Real-weight gen-core conformance and request-local RNG isolation for both registered variants.

use std::path::PathBuf;
use std::sync::Arc;

use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::{load_variant, StableAudio3SmallGenerator, Variant};

struct Case {
    variant: Variant,
    env: &'static str,
    prompt: &'static str,
}

/// The two registered post-trained checkpoints. The music prompt is the one sc-14543 shipped and is
/// left untouched; the SFX prompt is a real shipped `demo_cond` entry from its own
/// `model_config.json`.
const CASES: &[Case] = &[
    Case {
        variant: Variant::SmallMusic,
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        prompt: "tight electronic percussion and warm analog bass",
    },
    Case {
        variant: Variant::SmallSfx,
        env: "SA3_SMALL_SFX_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    },
];

fn snapshot(env: &str) -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    ))
}

fn request(prompt: &str, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        seed: Some(seed),
        steps: Some(1),
        sampler: Some("pingpong".into()),
        audio: Some(AudioParams {
            target_duration: Some(0.25),
            sample_rate: Some(44_100),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn samples(generator: &dyn Generator, prompt: &str, seed: u64) -> Vec<f32> {
    match generator
        .generate(&request(prompt, seed), &mut |_| {})
        .unwrap()
    {
        GenerationOutput::Audio(track) => track.samples,
        other => panic!("expected audio, got {other:?}"),
    }
}

fn run_conformance(case: &Case) {
    let spec = LoadSpec::new(snapshot(case.env));
    let profile = gen_core_testkit::AudioProfile {
        prompt: case.prompt.into(),
        steps: 1,
        seed: 42,
        cancel_steps: 3,
        audio: AudioParams {
            target_duration: Some(0.25),
            sample_rate: Some(44_100),
            ..Default::default()
        },
    };
    gen_core_testkit::audio_conformance(
        || {
            candle_audio_stable_audio_3::provider_registry()
                .unwrap()
                .load(case.variant.model_id(), &spec)
                .unwrap()
        },
        &profile,
    );
}

fn run_rng_isolation(case: &Case) {
    let generator: Arc<StableAudio3SmallGenerator> =
        Arc::new(load_variant(case.variant, &LoadSpec::new(snapshot(case.env))).unwrap());
    assert_eq!(generator.descriptor().id, case.variant.model_id());

    // Load once before spawning so this test isolates synthesis concurrency from lazy-load locking.
    let baseline = samples(generator.as_ref(), case.prompt, 42);
    let alternate = samples(generator.as_ref(), case.prompt, 43);
    assert_ne!(baseline, alternate, "alternate seed must change PCM");
    assert_eq!(
        samples(generator.as_ref(), case.prompt, 42),
        baseline,
        "same seed must be byte-identical across sequential requests"
    );

    let first = Arc::clone(&generator);
    let second = Arc::clone(&generator);
    let prompt = case.prompt;
    let a = std::thread::spawn(move || samples(first.as_ref(), prompt, 42));
    let b = std::thread::spawn(move || samples(second.as_ref(), prompt, 42));
    assert_eq!(a.join().unwrap(), baseline);
    assert_eq!(b.join().unwrap(), baseline);
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-music snapshot"]
fn registered_provider_passes_full_audio_conformance() {
    run_conformance(&CASES[0]);
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-music snapshot"]
fn concurrent_requests_are_deterministic_and_do_not_share_rng_state() {
    run_rng_isolation(&CASES[0]);
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-sfx snapshot"]
fn registered_sfx_provider_passes_full_audio_conformance() {
    run_conformance(&CASES[1]);
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-sfx snapshot"]
fn concurrent_sfx_requests_are_deterministic_and_do_not_share_rng_state() {
    run_rng_isolation(&CASES[1]);
}
