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
    /// The prompt this variant's per-run gate renders.
    ///
    /// The SFX case is a real shipped `demo_cond` prompt from that snapshot's own
    /// `model_config.json`. The music case is the prompt `sc-14543` registered with — not a
    /// `demo_cond` entry — kept verbatim so this story does not silently move the music gate's
    /// operating point. Both appear in their variant's calibration sweep below, so the floors
    /// these renders enforce are measured on exactly the prompt they enforce against.
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
    Case {
        variant: Variant::Medium,
        env: "SA3_MEDIUM_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
        wav_out: "SA3_MEDIUM_WAV_OUT",
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
/// The floors are per variant and set from the per-variant sweeps below, which are committed, run
/// on real weights, and — critically — run **at the same duration, step count, and backend the
/// floor is enforced at**. A floor measured at one configuration and enforced at another is not a
/// calibrated gate, it is a guess with a decimal point.
const fn minimum_side_ratio(variant: Variant) -> f64 {
    match variant {
        Variant::SmallMusic => MUSIC_SIDE_RATIO_FLOOR,
        Variant::SmallSfx => SFX_SIDE_RATIO_FLOOR,
        Variant::Medium => MEDIUM_SIDE_RATIO_FLOOR,
    }
}

/// The one configuration every side-ratio number in this file is measured at, and the one the CI
/// render steps enforce at. Kept as constants so the sweep's default and the workflow's render
/// cannot drift apart silently.
const CALIBRATION_DURATION_SECS: f32 = 30.0;
const CALIBRATION_STEPS: u32 = 8;

/// Set from the 25-sample SFX sweep (five prompts x five seeds) in
/// [`sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds`], run at the enforced
/// 30 s / 8 steps on **both** backends that enforce it:
///
/// | backend | min global | min median-window |
/// |---|---:|---:|
/// | Metal | 5.71451e-4 | 4.78244e-4 |
/// | CUDA | 5.71427e-4 | 4.78424e-4 |
///
/// The two backends agree to four significant figures, which is itself worth knowing: the floor is
/// not absorbing a cross-backend discrepancy, there isn't one. Union minimum 4.78244e-4 (Metal), so
/// the floor sits 2.39x below the smallest of the 100 measured numbers.
/// The distribution is bimodal: 20 of 25 samples per backend land in 5.7e-4 … 1.4e-3 and the
/// "Sparkling fantasy energy swirl" prompt lands near 1.0 at every seed.
///
/// The floor is deliberately *not* pushed closer — the low end is a real property of this
/// checkpoint, and a floor inside its natural spread would flake on honest output. It is
/// deliberately not pushed lower either: the 1e-4 it replaced sat three orders below the typical
/// value, which made the gate equivalent to "the side signal is not exactly zero". That is not a
/// claim in prose — [`the_quality_gates_reject_the_degeneracies_they_are_named_for`] carries a
/// control that 1e-4 admits and this floor rejects, so the tightening has gate strength behind it.
const SFX_SIDE_RATIO_FLOOR: f64 = 2e-4;

/// The floor [`SFX_SIDE_RATIO_FLOOR`] replaced, kept so the discrimination between the two is a
/// committed, executed assertion rather than a claim in a commit message.
const PREVIOUS_SFX_SIDE_RATIO_FLOOR: f64 = 1e-4;

/// Set from the 25-sample music sweep in
/// [`music_stereo_width_floor_is_calibrated_across_prompts_and_seeds`], at the same enforced
/// 30 s / 8 steps on both backends:
///
/// | backend | min global | min median-window |
/// |---|---:|---:|
/// | Metal | 1.64826e-1 | 2.00091e-1 |
/// | CUDA | 1.64825e-1 | 2.00093e-1 |
///
/// The music checkpoint renders a genuinely wide image on every prompt in its own demo set — the
/// union minimum (1.64825e-1, CUDA) is 16.48x above this floor. It is left at the shipped `1e-2`
/// rather than
/// raised into that gap on purpose: this is a *duplicated-mono* gate, not a stereo-width quality
/// bar, and the sweep's upper-bound assertion is relaxed for this variant accordingly. What the
/// sweep adds is the measurement that was missing — the per-window median assertion is now
/// enforced only at a configuration where it has been measured.
const MUSIC_SIDE_RATIO_FLOOR: f64 = 1e-2;

/// Set from the 25-sample medium sweep in
/// [`medium_stereo_width_floor_is_calibrated_across_prompts_and_seeds`], run at the enforced
/// 30 s / 8 steps on Metal:
///
/// | backend | min global | min median-window |
/// |---|---:|---:|
/// | Metal | 1.20543e-4 | 1.02879e-4 |
///
/// Medium is the only SA3 checkpoint registered for **both** domains, and its sweep says so: the
/// three music prompts and the footsteps prompt all land between 2.65e-1 and 1.03, while
/// "Dog barking next to a waterfall" collapses to 1.2e-4 at two of five seeds and recovers to
/// 2.2e-1 / 3.4e-1 at two others. A floor set anywhere inside the music distribution would gate
/// honest sparse-SFX output on this id, which is precisely the failure a generalist checkpoint
/// invites. The floor is therefore taken from the union minimum with the same ~2x margin the SFX
/// specialist uses: 1.02879e-4 / 5e-5 = 2.06x.
///
/// Like the other two, this is a **duplicated-mono** gate, not a stereo-width bar. What it must
/// catch is a decode path that emits one channel twice — exactly 0 — which matters more here than
/// for the smalls because medium decodes through SAME-L rather than SAME-S.
const MEDIUM_SIDE_RATIO_FLOOR: f64 = 5e-5;

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
/// surfaces on every PR instead of only on a real-weight runner — and the real-weight gates above
/// are only meaningful because these hold.
///
/// The lane is named, because a gate nothing executes is not a gate: the `Candle CPU packages
/// (Linux)` job's `Test Stable Audio 3 weight-free quality gates` step runs this target. The
/// step exists specifically because the step above it is `--lib`, which skips integration targets
/// like this one; the `--lib --tests` sweep is on the manual `windows-cuda` job, and the
/// real-weight lanes pass `-- --ignored`, so neither reaches this test.
#[test]
fn the_quality_gates_reject_the_degeneracies_they_are_named_for() {
    let sfx_floor = minimum_side_ratio(Variant::SmallSfx);
    let signal = pseudo_noise(44_100, 0xA53F);

    // Duplicated mono: exactly what the side-ratio gate exists to catch.
    let duplicated = stereo_width(&signal, &signal);
    assert_eq!(duplicated.global, 0.0);
    assert_eq!(duplicated.median_window, 0.0);

    // A channel plus numerical dust: side/mid ~3.46e-5, roughly -89 dB. This is the degenerate end
    // of near-mono and both the committed floor and the 1e-4 it replaced reject it — it is a
    // sanity control on the measurement, not evidence for the tightening. The control that carries
    // the tightening is `near_mono` below.
    let dusted = signal
        .iter()
        .enumerate()
        .map(|(index, sample)| sample + if index % 2 == 0 { 1e-5 } else { -1e-5 })
        .collect::<Vec<_>>();
    let dust = stereo_width(&signal, &dusted);
    assert!(
        dust.global < PREVIOUS_SFX_SIDE_RATIO_FLOOR && dust.median_window < sfx_floor,
        "dusted mono must fail both the committed floor and the one it replaced: {} / {}",
        dust.global,
        dust.median_window
    );

    // The control that discriminates the committed floor from the 1e-4 it replaced.
    //
    // One channel duplicated with an alternating differential ~77 dB below the programme. That is
    // not a stereo image — it is one channel plus dither-scale noise — and yet 1e-4 admits it.
    // Without a control in the 1e-4 … 2e-4 band the tightening would halve the margin and buy no
    // gate strength at all, which is precisely the review finding this answers.
    let near_mono = signal
        .iter()
        .enumerate()
        .map(|(index, sample)| sample + if index % 2 == 0 { 4e-5 } else { -4e-5 })
        .collect::<Vec<_>>();
    let near = stereo_width(&signal, &near_mono);
    eprintln!(
        "near-mono control: global={:.9} median_window={:.9} (previous floor {:e}, committed floor {:e})",
        near.global, near.median_window, PREVIOUS_SFX_SIDE_RATIO_FLOOR, sfx_floor
    );
    assert!(
        near.global > PREVIOUS_SFX_SIDE_RATIO_FLOOR
            && near.median_window > PREVIOUS_SFX_SIDE_RATIO_FLOOR,
        "the near-mono control must PASS the {PREVIOUS_SFX_SIDE_RATIO_FLOOR:e} floor it \
         discriminates against, otherwise it proves nothing about the tightening: {} / {}",
        near.global,
        near.median_window
    );
    assert!(
        near.global < sfx_floor && near.median_window < sfx_floor,
        "the committed floor must REJECT a channel duplicated with a ~77 dB-down differential, \
         which is the whole reason it was tightened from {PREVIOUS_SFX_SIDE_RATIO_FLOOR:e}: {} / {}",
        near.global,
        near.median_window
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

/// Every shipped music `demo_cond` prompt, plus the prompt the music per-run gate renders.
///
/// The last entry is `CASES[0].prompt` verbatim. Without it the sweep would calibrate a floor the
/// per-run gate never operates at, which is the exact defect this sweep exists to prevent.
const MUSIC_SWEEP_PROMPTS: &[&str] = &[
    "A beautiful piano arpeggio grows into a grand cinematic climax",
    "Elegant and sophisticated Latin jazz piece with a Cuban base",
    "Amen break 174 BPM",
    "lofi house loop",
    "warm cinematic post-rock with bowed strings and restrained drums",
];

/// The medium sweep deliberately spans **both** domains: two shipped music `demo_cond` prompts from
/// medium's own `model_config.json`, two shipped SFX `demo_cond` prompts from `small-sfx`, and the
/// prompt medium's per-run gate renders. Medium is the only SA3 checkpoint tagged for both domains,
/// so a floor calibrated on music alone would not cover what this id is registered to serve.
const MEDIUM_SWEEP_PROMPTS: &[&str] = &[
    "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
    "A tropical house track with upbeat melodies, a driving bassline, and cheery vibes",
    "Dog barking next to a waterfall",
    "Running footsteps on pavement, fast pace, urban street environment, energetic motion sound",
    "warm cinematic post-rock with bowed strings and restrained drums",
];

/// Includes 42, the seed every per-run gate renders at.
const SWEEP_SEEDS: &[u64] = &[42, 7, 14_544, 2_026, 31_337];

/// Drive one registered variant over its own prompt space and return the measured side-ratio
/// minima, then assert the committed floor is a calibrated gate rather than a guess.
///
/// - Every sample must clear the floor. If a future change narrows the checkpoint's image this
///   fails here, on a 25-sample sweep, rather than flaking in the single-run gate.
/// - The floor must not sit further than `maximum_margin` below the measured minimum. A floor
///   three orders under the data is not a gate on anything except an exactly-zero side signal.
///
/// The default duration and step count are [`CALIBRATION_DURATION_SECS`] and [`CALIBRATION_STEPS`]
/// — the configuration the CI render steps enforce the floor at — so running this test with no
/// environment overrides calibrates exactly what is enforced.
fn calibrate_side_ratio_floor(variant: Variant, env: &str, prompts: &[&str], maximum_margin: f64) {
    let label = variant.model_id();
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(variant.model_id(), &LoadSpec::new(snapshot(env)))
        .expect("strict registered variant-bound load");
    let duration = env_f32("SA3_TEST_DURATION", CALIBRATION_DURATION_SECS);
    let steps = env_u32("SA3_TEST_STEPS", CALIBRATION_STEPS);

    let mut minimum_global = f64::INFINITY;
    let mut minimum_median = f64::INFINITY;
    for prompt in prompts {
        for &seed in SWEEP_SEEDS {
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
                "sweep {label} seed={seed:<6} global={:.9} median_window={:.9} windows={:<4} prompt={prompt}",
                width.global, width.median_window, width.windows
            );
            minimum_global = minimum_global.min(width.global);
            minimum_median = minimum_median.min(width.median_window);
        }
    }

    let floor = minimum_side_ratio(variant);
    let measured = minimum_global.min(minimum_median);
    eprintln!(
        "{label} side-ratio sweep over {} prompts x {} seeds at {duration}s/{steps} steps: \
         min_global={minimum_global:.9} min_median_window={minimum_median:.9} \
         floor={floor:e} margin={:.2}x (max {maximum_margin:.0}x)",
        prompts.len(),
        SWEEP_SEEDS.len(),
        measured / floor
    );
    assert!(
        measured > floor,
        "the {label} side-ratio floor {floor:e} is above the measured minimum {measured} — the \
         per-run gate would flake on honest output"
    );
    assert!(
        measured / floor <= maximum_margin,
        "the {label} side-ratio floor {floor:e} sits {:.1}x below the measured minimum {measured}, \
         past the {maximum_margin:.0}x this variant allows; a floor that far under the data only \
         catches an exactly-zero side signal",
        measured / floor
    );
}

/// The calibration behind [`SFX_SIDE_RATIO_FLOOR`], executed rather than asserted in prose.
///
/// The SFX checkpoint's stereo image spans three orders of magnitude across its own prompt space,
/// so the floor cannot be read off one run and cannot be set to a value that would read as
/// "audibly wide" — that would gate honest output.
#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_SFX_SNAPSHOT"]
fn sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds() {
    calibrate_side_ratio_floor(
        Variant::SmallSfx,
        "SA3_SMALL_SFX_SNAPSHOT",
        SFX_SWEEP_PROMPTS,
        10.0,
    );
}

/// The calibration behind [`MUSIC_SIDE_RATIO_FLOOR`], and the measurement the per-window median
/// assertion was previously missing on this variant entirely.
///
/// The margin bound is relaxed to 50x here, unlike the SFX sweep's 10x. That is not a weaker
/// standard applied to hide a number — the music checkpoint's image is genuinely two to three
/// orders wider than the SFX checkpoint's, and `1e-2` is the shipped duplicated-mono floor rather
/// than a width bar. Raising the floor into the measured distribution would convert a correctness
/// gate into a quality gate on honest output.
///
/// The drift worth catching — the music image collapsing toward the SFX one — is caught by the
/// `measured > floor` assertion, not by this margin bound. At the SFX scale (~4.8e-4) the ratio
/// against the shipped `1e-2` floor is ~0.05, which passes 50x comfortably; what fails is the
/// floor itself. The margin bound guards the opposite direction: a floor left far under the data.
#[test]
#[ignore = "real 3.45 GB weights; set SA3_SMALL_MUSIC_SNAPSHOT"]
fn music_stereo_width_floor_is_calibrated_across_prompts_and_seeds() {
    calibrate_side_ratio_floor(
        Variant::SmallMusic,
        "SA3_SMALL_MUSIC_SNAPSHOT",
        MUSIC_SWEEP_PROMPTS,
        50.0,
    );
}

/// Long-form medium renders, measured rather than asserted.
///
/// This is the variant's entire reason to exist: the smalls stop at 120 s, medium advertises 380 s,
/// and the failure mode that matters is the one that only appears once the decode plan spans tens of
/// SAME-L chunks. The durations are operator-selectable so one binary can produce the whole table,
/// and the gate is the same `assert_real_audio` every other real-weight render must satisfy — exact
/// `floor(seconds * 44100)` framing, finite and clamped PCM, a genuine two-channel image, non-silent
/// output, and no white-noise or pure-tone degeneracy.
///
/// Set `SA3_MEDIUM_LONG_SECONDS` to a comma-separated list (default `120,300`). Wall-clock per
/// render is printed; peak resident memory is captured by the caller (`/usr/bin/time -l` on macOS,
/// the CUDA job's allocator probe on Windows), because a process-wide peak is not something a test
/// can attribute to one render.
#[test]
#[ignore = "real 10.4 GB weights; set SA3_MEDIUM_SNAPSHOT"]
fn medium_long_form_renders_are_exact_and_timed() {
    let case = &CASES[2];
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(case.variant.model_id(), &LoadSpec::new(snapshot(case.env)))
        .expect("strict registered variant-bound load");
    let steps = env_u32("SA3_TEST_STEPS", CALIBRATION_STEPS);
    let seed = 42u64;
    let durations = std::env::var("SA3_MEDIUM_LONG_SECONDS")
        .unwrap_or_else(|_| "120,300".to_owned())
        .split(',')
        .map(|value| value.trim().parse::<f32>().expect("duration list"))
        .collect::<Vec<_>>();
    assert!(
        durations.iter().copied().fold(0.0f32, f32::max) >= 300.0,
        "the >=300 s measurement is the point of this test; got {durations:?}"
    );
    for duration in durations {
        let started = std::time::Instant::now();
        let track = match generator
            .generate(&request(case.prompt, duration, steps, seed), &mut |_| {})
            .expect("connected long-form generation")
        {
            GenerationOutput::Audio(track) => track,
            other => panic!("expected audio, got {other:?}"),
        };
        let elapsed = started.elapsed();
        eprintln!(
            "medium long-form: seconds={duration} steps={steps} wall_clock_s={:.3} \
             realtime_factor={:.3}x",
            elapsed.as_secs_f64(),
            duration as f64 / elapsed.as_secs_f64()
        );
        assert_real_audio(case.variant, &track, duration);
        if let Some(path) = std::env::var_os("SA3_MEDIUM_LONG_WAV_DIR") {
            let path = PathBuf::from(path).join(format!("sa3-medium-{duration}s.wav"));
            candle_audio::wav::write_wav_pcm16(&path, &track).expect("write WAV");
            eprintln!("medium long-form WAV: {}", path.display());
        }
    }
}

/// The calibration behind [`MEDIUM_SIDE_RATIO_FLOOR`], over medium's own two-domain prompt space.
#[test]
#[ignore = "real 10.4 GB weights; set SA3_MEDIUM_SNAPSHOT"]
fn medium_stereo_width_floor_is_calibrated_across_prompts_and_seeds() {
    calibrate_side_ratio_floor(
        Variant::Medium,
        "SA3_MEDIUM_SNAPSHOT",
        MEDIUM_SWEEP_PROMPTS,
        10.0,
    );
}

#[test]
#[ignore = "real 10.4 GB weights; set SA3_MEDIUM_SNAPSHOT"]
fn connected_medium_generation_is_stereo_finite_and_exact_length() {
    run_case(&CASES[2]);
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
