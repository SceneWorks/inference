//! Connected-provider real-weight gates for `stable_audio_3_small_music`.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio;
use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var("SA3_SMALL_MUSIC_SNAPSHOT")
            .expect("set SA3_SMALL_MUSIC_SNAPSHOT to the pinned small-music snapshot"),
    ))
}

fn request(duration: f32, steps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "warm cinematic post-rock with bowed strings and restrained drums".into(),
        seed: Some(seed),
        steps: Some(steps),
        sampler: Some("pingpong".into()),
        audio: Some(AudioParams {
            target_duration: Some(duration),
            sample_rate: Some(44_100),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_MUSIC_SNAPSHOT"]
fn connected_short_generation_is_stereo_finite_and_exact_length() {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(
            candle_audio_stable_audio_3::MODEL_ID,
            &LoadSpec::new(snapshot()),
        )
        .expect("strict registered small-music load");
    assert_eq!(
        generator.descriptor().id,
        candle_audio_stable_audio_3::MODEL_ID
    );
    let duration = std::env::var("SA3_TEST_DURATION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.25f32);
    let steps = std::env::var("SA3_TEST_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1u32);
    let mut seen_steps = Vec::new();
    let mut decoding = 0usize;
    let output = generator
        .generate(
            &request(duration, steps, 42),
            &mut |progress| match progress {
                Progress::Step { current, total } => seen_steps.push((current, total)),
                Progress::Decoding => decoding += 1,
                Progress::Loading(_) => {}
            },
        )
        .expect("connected generation");
    assert_eq!(
        seen_steps,
        (1..=steps)
            .map(|current| (current, steps))
            .collect::<Vec<_>>()
    );
    assert_eq!(decoding, 1);
    let track = match output {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(track.sample_rate, 44_100);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty());
    let expected_frames = (duration as f64 * 44_100.0) as usize;
    assert_eq!(track.samples.len(), expected_frames * 2);
    assert!(track.samples.iter().all(|sample| sample.is_finite()));
    assert!(track
        .samples
        .iter()
        .all(|sample| (-1.0..=1.0).contains(sample)));
    let rms = (track
        .samples
        .iter()
        .map(|sample| sample * sample)
        .sum::<f32>()
        / track.samples.len() as f32)
        .sqrt();
    assert!(rms > 1e-6, "decoded output is silent: rms={rms}");
    let channel_delta = track
        .samples
        .chunks_exact(2)
        .map(|frame| (frame[0] - frame[1]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        channel_delta > 1e-6,
        "decoded stereo channels are duplicated mono"
    );

    if let Some(path) = std::env::var_os("SA3_SMALL_MUSIC_WAV_OUT") {
        candle_audio::wav::write_wav_pcm16(&PathBuf::from(path), &track).expect("write WAV");
    }
}
