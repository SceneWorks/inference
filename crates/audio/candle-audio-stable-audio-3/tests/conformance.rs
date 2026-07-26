//! Real-weight gen-core conformance and request-local RNG isolation for every registered variant.

use std::path::PathBuf;
use std::sync::Arc;

use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::{load_variant, StableAudio3Generator, Variant};

struct Case {
    variant: Variant,
    env: &'static str,
    prompt: &'static str,
}

/// The three registered post-trained checkpoints. The music prompt is the one sc-14543 shipped and
/// is left untouched; the SFX and medium prompts are real shipped `demo_cond` entries from their own
/// `model_config.json` files.
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
    Case {
        variant: Variant::Medium,
        env: "SA3_MEDIUM_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
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
    let generator: Arc<StableAudio3Generator> =
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

#[test]
#[ignore = "requires the pinned 10.4 GB medium snapshot"]
fn registered_medium_provider_passes_full_audio_conformance() {
    run_conformance(&CASES[2]);
}

#[test]
#[ignore = "requires the pinned 10.4 GB medium snapshot"]
fn concurrent_medium_requests_are_deterministic_and_do_not_share_rng_state() {
    run_rng_isolation(&CASES[2]);
}

/// The advertised cap must be the one the adapted geometry can actually serve, and it must be a
/// per-variant number rather than the crate-global `120.0` that sc-14545 replaced.
///
/// This runs without weights: it is a descriptor assertion, and it discriminates because a
/// regression to a shared constant makes medium's advertised cap 120 s while `sample_size` still
/// says `16,777,216`, and a regression that hands medium the smalls' geometry makes the requested
/// frame count unreachable.
#[test]
fn each_variant_advertises_a_cap_its_own_geometry_can_serve() {
    for (variant, expected_cap, sample_size) in [
        (Variant::SmallMusic, 120.0f32, 5_292_032usize),
        (Variant::SmallSfx, 120.0, 5_292_032),
        (Variant::Medium, 380.0, 16_777_216),
    ] {
        let descriptor = variant.descriptor();
        assert_eq!(
            descriptor.capabilities.max_audio_duration_secs,
            Some(expected_cap),
            "{} advertised cap",
            variant.model_id()
        );
        assert_eq!(variant.shape().sample_size, sample_size);
        let frames = (expected_cap as f64 * 44_100.0).floor() as usize;
        assert!(
            frames <= sample_size,
            "{}: advertised cap {expected_cap}s needs {frames} frames but sample_size is \
             {sample_size}",
            variant.model_id()
        );
        // One second past the cap must not fit, otherwise the cap is understated by a whole second.
        let over = ((expected_cap as f64 + 1.0) * 44_100.0).floor() as usize;
        assert!(
            over > sample_size,
            "{}: {expected_cap}s is not the tightest advertised cap; {} s also fits",
            variant.model_id(),
            expected_cap + 1.0
        );
    }
    assert_ne!(
        Variant::Medium
            .descriptor()
            .capabilities
            .max_audio_duration_secs,
        Variant::SmallMusic
            .descriptor()
            .capabilities
            .max_audio_duration_secs,
        "the cap must be variant-bound, not the crate-global constant it replaced"
    );
}
