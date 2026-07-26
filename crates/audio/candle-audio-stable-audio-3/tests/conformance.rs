//! Real-weight gen-core conformance and request-local RNG isolation.

use std::path::PathBuf;
use std::sync::Arc;

use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::StableAudio3SmallMusicGenerator;

fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var("SA3_SMALL_MUSIC_SNAPSHOT")
            .expect("set SA3_SMALL_MUSIC_SNAPSHOT to the pinned small-music snapshot"),
    ))
}

fn request(seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "tight electronic percussion and warm analog bass".into(),
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

fn samples(generator: &dyn Generator, seed: u64) -> Vec<f32> {
    match generator.generate(&request(seed), &mut |_| {}).unwrap() {
        GenerationOutput::Audio(track) => track.samples,
        other => panic!("expected audio, got {other:?}"),
    }
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-music snapshot"]
fn registered_provider_passes_full_audio_conformance() {
    let spec = LoadSpec::new(snapshot());
    let profile = gen_core_testkit::AudioProfile {
        prompt: "tight electronic percussion and warm analog bass".into(),
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
                .load(candle_audio_stable_audio_3::MODEL_ID, &spec)
                .unwrap()
        },
        &profile,
    );
}

#[test]
#[ignore = "requires the pinned 3.45 GB small-music snapshot"]
fn concurrent_requests_are_deterministic_and_do_not_share_rng_state() {
    let generator: Arc<StableAudio3SmallMusicGenerator> =
        Arc::new(candle_audio_stable_audio_3::load_generator(&LoadSpec::new(snapshot())).unwrap());

    // Load once before spawning so this test isolates synthesis concurrency from lazy-load locking.
    let baseline = samples(generator.as_ref(), 42);
    let alternate = samples(generator.as_ref(), 43);
    assert_ne!(baseline, alternate, "alternate seed must change PCM");

    let first = Arc::clone(&generator);
    let second = Arc::clone(&generator);
    let a = std::thread::spawn(move || samples(first.as_ref(), 42));
    let b = std::thread::spawn(move || samples(second.as_ref(), 42));
    assert_eq!(a.join().unwrap(), baseline);
    assert_eq!(b.join().unwrap(), baseline);
}
