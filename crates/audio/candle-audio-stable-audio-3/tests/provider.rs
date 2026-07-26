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
///
/// Fails closed. Fewer than eight crossings in a take means the channel is DC, near-DC, or a tone
/// below ~16 Hz at the 0.25 s committed default — none of which is audio, and all of which would
/// otherwise clear both this gate and the lag-1 autocorrelation gate by returning "infinitely
/// spread". Returning 0 makes that input fail the caller's `spread > 0.05` assertion instead.
fn zero_crossing_interval_spread(channel: &[f32]) -> f64 {
    let mut crossings = Vec::new();
    for index in 1..channel.len() {
        if (channel[index - 1] < 0.0) != (channel[index] < 0.0) {
            crossings.push(index);
        }
    }
    if crossings.len() < 8 {
        return 0.0;
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

/// Analysis window for the per-window stereo measurement, in frames (~23 ms at 44.1 kHz).
///
/// Small enough that the 0.25 s committed default still yields ten windows.
const STEREO_WINDOW_FRAMES: usize = 1024;

/// Side/mid RMS ratio, measured globally and as a median over windows.
struct StereoWidth {
    /// `rms(side) / rms(mid)` over the whole take.
    global: f64,
    /// Median of `rms(side) / rms(mid)` over the non-silent windows.
    median_window: f64,
    windows: usize,
}

/// Measure how distinct the two channels are, both overall and window by window.
///
/// The global ratio alone is a single number over the whole take: a decode path that duplicates one
/// channel and differs in a handful of samples can clear a small floor on the strength of those
/// samples alone. The median over windows cannot be moved by a localized artefact — for a
/// duplicated channel it is exactly 0 no matter how loud the artefact is — so the two together
/// separate "near-mono but genuinely two channels" from "one channel emitted twice".
fn stereo_width(left: &[f32], right: &[f32]) -> StereoWidth {
    let mid = left
        .iter()
        .zip(right)
        .map(|(l, r)| (l + r) * 0.5)
        .collect::<Vec<_>>();
    let side = left
        .iter()
        .zip(right)
        .map(|(l, r)| (l - r) * 0.5)
        .collect::<Vec<_>>();
    let global_mid = rms(&mid);
    let global = rms(&side) / global_mid.max(f64::MIN_POSITIVE);

    // Silent windows carry no stereo information; including them would let a long tail of digital
    // silence dominate the median in either direction.
    let mut ratios = mid
        .chunks(STEREO_WINDOW_FRAMES)
        .zip(side.chunks(STEREO_WINDOW_FRAMES))
        .filter(|(mid, _)| rms(mid) > 0.1 * global_mid)
        .map(|(mid, side)| rms(side) / rms(mid).max(f64::MIN_POSITIVE))
        .collect::<Vec<_>>();
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
    let median_window = if ratios.is_empty() {
        0.0
    } else {
        ratios[ratios.len() / 2]
    };
    StereoWidth {
        global,
        median_window,
        windows: ratios.len(),
    }
}

/// Minimum side/mid RMS ratio each checkpoint must clear, globally and per-window-median.
///
/// Both variants decode through the byte-identical embedded SAME-S, so this measures the latents
/// the DiT produced, not the decode path. What it must catch is a decode path that emits one
/// channel twice — exactly 0 — or one that emits a channel plus numerical dust, which is orders of
/// magnitude below anything either checkpoint actually renders.
///
/// **This is not a stereo-width quality bar.** The SFX checkpoint genuinely renders a near-centred
/// image on most prompts, so a floor set near "audibly wide" would be a flaky gate on honest
/// output, not a correctness gate.
///
/// The floors are per variant and set from the sweep in
/// [`sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds`], which is committed and runs
/// on real weights so the calibration is re-checked rather than asserted in prose.
const fn minimum_side_ratio(variant: Variant) -> f64 {
    match variant {
        Variant::SmallMusic => 1e-2,
        Variant::SmallSfx => SFX_SIDE_RATIO_FLOOR,
    }
}

/// Set from the 25-sample sweep (five prompts x five seeds, Metal, 10 s / 8 steps) in
/// [`sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds`]:
///
/// - global side/mid ratio: min 4.91e-4, 24 of 25 samples in 4.9e-4 … 1.9e-3, one at 0.868;
/// - median-window side/mid ratio: min 4.06e-4, same shape.
///
/// The floor sits 2.03x below the smallest of the 50 measured numbers. It is deliberately *not*
/// pushed closer: the low end of the distribution is a real property of this checkpoint, and a
/// floor inside its natural spread would flake on honest output. It is deliberately not pushed
/// lower either — the previous 1e-4 was three orders below the typical value, which made the gate
/// equivalent to "the side signal is not exactly zero". The sweep asserts both directions.
const SFX_SIDE_RATIO_FLOOR: f64 = 2e-4;

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
    let width = stereo_width(&left, &right);
    let floor = minimum_side_ratio(variant);
    eprintln!(
        "{label}: side/mid global={:.9} median_window={:.9} over {} windows (floor {floor:e})",
        width.global, width.median_window, width.windows
    );
    assert!(
        width.windows > 0,
        "{label}: no non-silent analysis window — the stereo measurement is undefined"
    );
    assert!(
        width.global > floor,
        "{label}: stereo channels are duplicated mono (global side ratio {}, floor {floor:e})",
        width.global
    );
    assert!(
        width.median_window > floor,
        "{label}: the median window is duplicated mono (median side ratio {}, floor {floor:e}) — \
         the global ratio is carried by a localized artefact, not a real two-channel image",
        width.median_window
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

/// Cheap deterministic pseudo-noise, so the discrimination tests below need no dependency.
fn pseudo_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) as f64 / (1u64 << 31) as f64 - 1.0) as f32 * 0.25
        })
        .collect()
}

/// A gate that cannot fail is not a gate.
///
/// Every heuristic in this file is asserted against the exact degeneracy it is named for, with a
/// passing control alongside it. These run without weights, so a regression in the analysis itself
/// surfaces in the ordinary test lane instead of only on a real-weight runner — and the real-weight
/// gates above are only meaningful because these hold.
#[test]
fn the_quality_gates_reject_the_degeneracies_they_are_named_for() {
    let sfx_floor = minimum_side_ratio(Variant::SmallSfx);
    let signal = pseudo_noise(44_100, 0xA53F);

    // Duplicated mono: exactly what the side-ratio gate exists to catch.
    let duplicated = stereo_width(&signal, &signal);
    assert_eq!(duplicated.global, 0.0);
    assert_eq!(duplicated.median_window, 0.0);

    // A channel plus numerical dust — the "near-mono" case the old 1e-4 floor let through.
    let dusted = signal
        .iter()
        .enumerate()
        .map(|(index, sample)| sample + if index % 2 == 0 { 1e-5 } else { -1e-5 })
        .collect::<Vec<_>>();
    let dust = stereo_width(&signal, &dusted);
    assert!(
        dust.global < sfx_floor && dust.median_window < sfx_floor,
        "dusted mono must fail the SFX floor: {} / {}",
        dust.global,
        dust.median_window
    );

    // Duplicated mono with one loud localized burst: the global ratio clears the floor on the
    // strength of that burst alone, and only the per-window median catches it. This is the case
    // that justifies carrying a second measurement.
    let mut bursty = signal.clone();
    for sample in bursty.iter_mut().take(2_048) {
        *sample += 0.5;
    }
    let burst = stereo_width(&signal, &bursty);
    assert!(
        burst.global > sfx_floor,
        "the localized-burst control must clear the global floor, else it proves nothing: {}",
        burst.global
    );
    assert!(
        burst.median_window < sfx_floor,
        "the per-window median must reject a localized artefact standing in for a stereo image: {}",
        burst.median_window
    );

    // The passing control: two genuinely independent channels.
    let distinct = stereo_width(&signal, &pseudo_noise(44_100, 0x1234));
    assert!(distinct.global > sfx_floor && distinct.median_window > sfx_floor);

    // Zero-crossing spread: a pure tone, a DC offset, and a sub-audio-rate tone must all fail.
    let tone = |hz: f64, len: usize| {
        (0..len)
            .map(|index| (index as f64 * hz * std::f64::consts::TAU / 44_100.0).sin() as f32)
            .collect::<Vec<f32>>()
    };
    assert!(zero_crossing_interval_spread(&tone(440.0, 11_025)) <= 0.05);
    assert_eq!(zero_crossing_interval_spread(&vec![0.5f32; 11_025]), 0.0);
    // 10 Hz over the committed 0.25 s default yields five crossings. Before the fail-closed fix
    // this returned infinity and passed, while also clearing the lag-1 autocorrelation gate.
    assert_eq!(zero_crossing_interval_spread(&tone(10.0, 11_025)), 0.0);
    assert!(lag_one_autocorrelation(&tone(10.0, 11_025)) > 0.2);
    assert!(zero_crossing_interval_spread(&signal) > 0.05);

    // Lag-1 autocorrelation: white noise fails, a smooth signal passes.
    assert!(lag_one_autocorrelation(&signal) < 0.2);
    assert!(lag_one_autocorrelation(&tone(440.0, 11_025)) > 0.2);
}

/// Every shipped SFX `demo_cond` prompt, plus the neutral prompt the divergence gate uses.
const SFX_SWEEP_PROMPTS: &[&str] = &[
    "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    "Dog barking next to a waterfall",
    "Sparkling fantasy energy swirl, mystical shimmer, rising magical burst",
    "Running footsteps on pavement, fast pace, urban street environment, energetic motion sound",
    "a short bright transient followed by a decaying tail",
];

const SFX_SWEEP_SEEDS: &[u64] = &[42, 7, 14_544, 2_026, 31_337];

/// The calibration behind [`SFX_SIDE_RATIO_FLOOR`], executed rather than asserted in prose.
///
/// The SFX checkpoint's stereo image spans three orders of magnitude across its own prompt space,
/// so the floor cannot be read off one run and cannot be set to a value that would read as
/// "audibly wide" — that would gate honest output. This drives the registered provider over every
/// shipped `demo_cond` prompt and the neutral divergence prompt at five seeds, and pins the floor
/// to the measured minimum:
///
/// - every sample must clear the floor, and
/// - the floor must sit within one order of magnitude *below* the measured minimum. A floor three
///   orders under the data is not a gate on anything except an exactly-zero side signal, which was
///   the review finding this test answers.
///
/// Both directions matter. If a future change narrows the checkpoint's image, the first assertion
/// fails here — on a 25-sample sweep — rather than flaking in the single-run gate. If the image
/// widens enough that the floor is no longer meaningful, the second fails and the floor is
/// recalibrated deliberately.
///
/// Measured on Metal at 10 s / 8 steps: `min_global=4.90523e-4`, `min_median_window=4.05910e-4`,
/// against the committed `2e-4` floor — a 2.03x margin.
#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_SFX_SNAPSHOT"]
fn sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds() {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(
            Variant::SmallSfx.model_id(),
            &LoadSpec::new(snapshot("SA3_SMALL_SFX_SNAPSHOT")),
        )
        .expect("strict registered variant-bound load");
    let duration = env_f32("SA3_TEST_DURATION", 10.0);
    let steps = env_u32("SA3_TEST_STEPS", 8);

    let mut minimum_global = f64::INFINITY;
    let mut minimum_median = f64::INFINITY;
    for prompt in SFX_SWEEP_PROMPTS {
        for &seed in SFX_SWEEP_SEEDS {
            let track = match generator
                .generate(&request(prompt, duration, steps, seed), &mut |_| {})
                .expect("connected generation")
            {
                GenerationOutput::Audio(track) => track,
                other => panic!("expected audio, got {other:?}"),
            };
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
            let width = stereo_width(&left, &right);
            eprintln!(
                "sweep seed={seed:<6} global={:.9} median_window={:.9} windows={:<4} prompt={prompt}",
                width.global, width.median_window, width.windows
            );
            minimum_global = minimum_global.min(width.global);
            minimum_median = minimum_median.min(width.median_window);
        }
    }

    let floor = SFX_SIDE_RATIO_FLOOR;
    let measured = minimum_global.min(minimum_median);
    eprintln!(
        "sfx side-ratio sweep over {} prompts x {} seeds at {duration}s/{steps} steps: \
         min_global={minimum_global:.9} min_median_window={minimum_median:.9} \
         floor={floor:e} margin={:.2}x",
        SFX_SWEEP_PROMPTS.len(),
        SFX_SWEEP_SEEDS.len(),
        measured / floor
    );
    assert!(
        measured > floor,
        "the SFX side-ratio floor {floor:e} is above the measured minimum {measured} — the \
         per-run gate would flake on honest output"
    );
    assert!(
        measured / floor <= 10.0,
        "the SFX side-ratio floor {floor:e} sits {:.1}x below the measured minimum {measured}; a \
         floor that far under the data only catches an exactly-zero side signal",
        measured / floor
    );
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
