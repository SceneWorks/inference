//! Connected-provider real-weight gates for both registered Stable Audio 3 small variants.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio;
use candle_audio_stable_audio_3::gen_core::{
    AudioParams, AudioTrack, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use candle_audio_stable_audio_3::Variant;

struct Case {
    variant: Variant,
    env: &'static str,
    /// A real shipped `demo_cond` prompt from this snapshot's own `model_config.json`.
    prompt: &'static str,
    wav_out: &'static str,
}

const CASES: &[Case] = &[
    Case {
        variant: Variant::SmallMusic,
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        prompt: "warm cinematic post-rock with bowed strings and restrained drums",
        wav_out: "SA3_SMALL_MUSIC_WAV_OUT",
    },
    Case {
        variant: Variant::SmallSfx,
        env: "SA3_SMALL_SFX_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
        wav_out: "SA3_SMALL_SFX_WAV_OUT",
    },
];

fn snapshot(env: &str) -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    ))
}

fn request(prompt: &str, duration: f32, steps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
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

fn env_f32(name: &str, fallback: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn rms(samples: &[f32]) -> f64 {
    (samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

/// Lag-1 autocorrelation of one channel.
///
/// White noise sits at ~0. Real 44.1 kHz audio is heavily oversampled relative to its bandwidth and
/// sits well above 0, so this discriminates "the decoder emitted a noise field" from "the decoder
/// emitted audio".
fn lag_one_autocorrelation(channel: &[f32]) -> f64 {
    let mean = channel.iter().map(|s| *s as f64).sum::<f64>() / channel.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 0..channel.len() {
        let centered = channel[index] as f64 - mean;
        denominator += centered * centered;
        if index + 1 < channel.len() {
            numerator += centered * (channel[index + 1] as f64 - mean);
        }
    }
    numerator / denominator.max(f64::MIN_POSITIVE)
}

/// Coefficient of variation of the intervals between zero crossings.
///
/// A pure tone crosses zero at a fixed period, so this collapses to ~0. Anything with real spectral
/// content spreads it out.
fn zero_crossing_interval_spread(channel: &[f32]) -> f64 {
    let mut crossings = Vec::new();
    for index in 1..channel.len() {
        if (channel[index - 1] < 0.0) != (channel[index] < 0.0) {
            crossings.push(index);
        }
    }
    if crossings.len() < 8 {
        // Too few crossings to characterize: treat as "not a pure tone" only if it is not silent.
        return f64::INFINITY;
    }
    let intervals = crossings
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as f64)
        .collect::<Vec<_>>();
    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let variance =
        intervals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
    variance.sqrt() / mean.max(f64::MIN_POSITIVE)
}

/// Minimum side/mid RMS ratio each checkpoint must clear.
///
/// Both variants decode through the byte-identical embedded SAME-S, so this measures the latents
/// the DiT produced, not the decode path. What either gate must catch is a decode path that emits
/// one channel twice, which is exactly 0 — this is a degeneracy detector, not a stereo-width
/// quality bar.
///
/// The two floors are separate because the checkpoints genuinely differ, and both are set from
/// measurement rather than a shared guess. Measured on Metal at 10 s / 8 steps:
///
/// - music, on its shipped `demo_cond` prompt: 0.308;
/// - SFX, across all four shipped `demo_cond` prompts at seeds 42 and 7: 5.3e-4, 7.1e-4, 7.4e-4,
///   8.0e-4, 1.1e-3, 1.9e-3, 0.655 — i.e. the SFX checkpoint usually renders a near-centred image
///   and occasionally a wide one, spanning three orders of magnitude.
///
/// The SFX floor therefore sits an order of magnitude under the smallest observed value rather than
/// near it; tightening it toward the 5.3e-4 sample would make the gate flaky without catching
/// anything the 0 case does not already trigger.
const fn minimum_side_ratio(variant: Variant) -> f64 {
    match variant {
        Variant::SmallMusic => 1e-2,
        Variant::SmallSfx => 1e-4,
    }
}

/// Every shape/quality gate a registered SA3 small variant must satisfy on real weights.
fn assert_real_audio(variant: Variant, track: &AudioTrack, duration: f32) {
    let label = variant.model_id();
    assert_eq!(track.sample_rate, 44_100, "{label} sample rate");
    assert_eq!(track.channels, 2, "{label} channel count");
    assert!(track.stems.is_empty(), "{label} stems");

    let expected_frames = (duration as f64 * 44_100.0).floor() as usize;
    assert_eq!(
        track.samples.len(),
        expected_frames * 2,
        "{label}: expected exactly floor({duration} * 44100) = {expected_frames} stereo frames"
    );

    assert!(
        track.samples.iter().all(|sample| sample.is_finite()),
        "{label}: non-finite PCM"
    );
    assert!(
        track
            .samples
            .iter()
            .all(|sample| (-1.0..=1.0).contains(sample)),
        "{label}: PCM outside [-1, 1]"
    );

    let overall_rms = rms(&track.samples);
    let peak = track.samples.iter().fold(0.0f32, |max, s| max.max(s.abs()));
    eprintln!("{label}: frames={expected_frames} rms={overall_rms:.9} peak={peak:.9}");
    assert!(
        overall_rms > 1e-4,
        "{label}: output is silent (rms={overall_rms})"
    );
    assert!(peak > 1e-3, "{label}: output has no peak (peak={peak})");

    let left = track
        .samples
        .chunks_exact(2)
        .map(|frame| frame[0])
        .collect::<Vec<_>>();
    let right = track
        .samples
        .chunks_exact(2)
        .map(|frame| frame[1])
        .collect::<Vec<_>>();
    let side = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l - r) * 0.5)
        .collect::<Vec<_>>();
    let side_ratio = rms(&side) / overall_rms.max(f64::MIN_POSITIVE);
    let floor = minimum_side_ratio(variant);
    eprintln!("{label}: side/mid rms ratio={side_ratio:.9} (floor {floor:e})");
    assert!(
        side_ratio > floor,
        "{label}: stereo channels are duplicated mono (side ratio {side_ratio}, floor {floor:e})"
    );

    for (channel_name, channel) in [("left", &left), ("right", &right)] {
        let autocorrelation = lag_one_autocorrelation(channel);
        let spread = zero_crossing_interval_spread(channel);
        eprintln!("{label}/{channel_name}: lag1={autocorrelation:.6} zc_spread={spread:.6}");
        assert!(
            autocorrelation > 0.2,
            "{label}/{channel_name}: lag-1 autocorrelation {autocorrelation} is white-noise-like"
        );
        assert!(
            spread > 0.05,
            "{label}/{channel_name}: zero-crossing intervals are constant — a pure tone, not audio"
        );
    }
}

fn run_case(case: &Case) {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(case.variant.model_id(), &LoadSpec::new(snapshot(case.env)))
        .expect("strict registered variant-bound load");
    assert_eq!(generator.descriptor().id, case.variant.model_id());

    let duration = env_f32("SA3_TEST_DURATION", 0.25);
    let steps = env_u32("SA3_TEST_STEPS", 1);
    // Operator knobs for characterizing a checkpoint's behaviour across its own prompt space; the
    // committed defaults are a real shipped `demo_cond` prompt and a fixed seed.
    let prompt = std::env::var("SA3_TEST_PROMPT").unwrap_or_else(|_| case.prompt.to_owned());
    let seed = std::env::var("SA3_TEST_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(42u64);
    let mut seen_steps = Vec::new();
    let mut decoding = 0usize;
    let output = generator
        .generate(
            &request(&prompt, duration, steps, seed),
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
    assert_real_audio(case.variant, &track, duration);

    if let Some(path) = std::env::var_os(case.wav_out) {
        candle_audio::wav::write_wav_pcm16(&PathBuf::from(path), &track).expect("write WAV");
    }
}

#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_MUSIC_SNAPSHOT"]
fn connected_short_generation_is_stereo_finite_and_exact_length() {
    run_case(&CASES[0]);
}

#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_SFX_SNAPSHOT"]
fn connected_sfx_generation_is_stereo_finite_and_exact_length() {
    run_case(&CASES[1]);
}
