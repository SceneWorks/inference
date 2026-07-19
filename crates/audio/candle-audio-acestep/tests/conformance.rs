//! Real-weight conformance for the candle ACE-Step 1.5 music provider (sc-12842): real pinned
//! weights → registry load-by-id → generate → a real, music-shaped stereo WAV.
//!
//! Both tests are `#[ignore]`d and snapshot-gated like every other family's real-weight tests:
//! set `ACESTEP_SNAPSHOT` to an `ACE-Step/acestep-v15-xl-turbo-diffusers` snapshot dir, or leave
//! it unset to resolve the pinned snapshot through the audio lane's F-029 hub path (downloads
//! several GB into the ordinary HF cache on first run).
//!
//! ```text
//! cargo test --locked -p candle-audio-acestep --test conformance -- --ignored --nocapture
//! ```
//!
//! `acestep_music_wav_conformance` writes the synthesized WAV (`ACESTEP_WAV_OUT` overrides the
//! path) so a human can listen to the evidence.

use std::path::PathBuf;

use candle_audio_acestep::candle_audio;
use candle_audio_acestep::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

/// Resolve the snapshot: `ACESTEP_SNAPSHOT` env (a snapshot dir) or the pinned hub path.
fn snapshot() -> WeightsSource {
    match std::env::var("ACESTEP_SNAPSHOT") {
        Ok(dir) => WeightsSource::Dir(PathBuf::from(dir)),
        Err(_) => candle_audio_acestep::resolve_pinned_snapshot()
            .expect("resolve the pinned ACE-Step/acestep-v15-xl-turbo-diffusers snapshot"),
    }
}

/// The backend-neutral gen-core conformance suite (validate honesty, progress contract, typed
/// mid-run and pre-generate cancellation, seed determinism) against the real provider, resolved
/// through the explicit registry.
#[test]
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT or network); run with --ignored"]
fn acestep_conformance() {
    let spec = LoadSpec::new(snapshot());
    let profile = gen_core_testkit::Profile {
        prompt: "gentle ambient piano with soft pads".to_owned(),
        width: 256,
        height: 256,
        steps: 2,
        seed: 42,
        cancel_steps: 6,
    };
    gen_core_testkit::conformance(
        || {
            candle_audio_acestep::provider_registry()
                .unwrap()
                .load(candle_audio_acestep::MODEL_ID, &spec)
                .expect("acestep_v15_turbo loads through the explicit registry")
        },
        &profile,
    );
}

/// The real-WAV DoD: fixed prompt + lyrics + seed + duration → non-empty, 48 kHz, stereo, finite,
/// NON-SILENT, NON-DEGENERATE **music** of the requested duration, written as a playable WAV.
///
/// The anti-degeneracy assertions and what each catches (all computed on the mono downmix):
/// - **interior RMS floor** — all-silence output (a dead decoder / all-zero latents);
/// - **frame-RMS coefficient of variation** — energy that never moves: constant tones, steady
///   unmodulated noise beds, DC drones (music has phrasing, dynamics, note onsets);
/// - **peak-bin energy cap** — a single pure tone dominating the long-term spectrum;
/// - **octave-band spread floor** — narrowband output (music occupies several octaves);
/// - **spectral-flatness ceiling** — white-noise-only output (flatness → 1 for white noise);
/// - **rhythmic periodicity** — a beat: the frame-energy envelope's autocorrelation must show a
///   periodic peak in a musical tempo band (catches arrhythmic drones/noise);
/// - **byte-identical re-synthesis** — the seed law.
#[test]
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT or network); run with --ignored"]
fn acestep_music_wav_conformance() {
    let spec = LoadSpec::new(snapshot());
    let registry = candle_audio_acestep::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_acestep::MODEL_ID, &spec)
        .expect("acestep_v15_turbo loads through the explicit registry");
    assert_eq!(generator.descriptor().id, "acestep_v15_turbo");

    const TARGET_SECS: f32 = 12.0;
    const STEPS: u32 = 8;
    let req = GenerationRequest {
        prompt: "upbeat electronic dance track, driving synth bass, crisp hi-hats".into(),
        seed: Some(42),
        steps: Some(STEPS),
        audio: Some(AudioParams {
            target_duration: Some(TARGET_SECS),
            sample_rate: Some(48_000),
            language: Some("en".into()),
            bpm: Some(128.0),
            lyrics: Some("[verse]\nlights up on the floor tonight\n[chorus]\nwe move until the morning light".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut steps = 0u32;
    let mut decoding = 0u32;
    let out = generator
        .generate(&req, &mut |p| match p {
            Progress::Step { .. } => steps += 1,
            Progress::Decoding => decoding += 1,
            Progress::Loading(_) => {}
        })
        .expect("generate");
    assert_eq!(steps, STEPS);
    assert_eq!(decoding, 1);

    let track = match out {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected GenerationOutput::Audio, got {other:?}"),
    };
    assert_eq!(track.sample_rate, 48_000);
    assert_eq!(track.channels, 2, "ACE-Step is stereo");
    assert!(
        track.stems.is_empty(),
        "text-to-music emits a mix, not stems"
    );
    assert!(!track.samples.is_empty());
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );

    // Duration honored (interleaved stereo → frames = samples / channels).
    let frames = track.samples.len() / track.channels as usize;
    let secs = frames as f32 / track.sample_rate as f32;
    assert!(
        (secs - TARGET_SECS).abs() <= 0.1,
        "duration {secs:.3}s not within 100 ms of the requested {TARGET_SECS}s"
    );

    // Mono downmix for analysis.
    let mono: Vec<f32> = track
        .samples
        .chunks(track.channels as usize)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    let n = mono.len();
    let interior = &mono[n / 20..n - n / 20];

    // NON-SILENT.
    let rms = (interior.iter().map(|s| s * s).sum::<f32>() / interior.len() as f32).sqrt();
    assert!(rms > 0.01, "interior RMS {rms:.5} — silence is a failure");

    // Energy-over-time variation (25 ms frames).
    let frame_len = 1200; // 25 ms @ 48 kHz
    let frame_rms: Vec<f32> = mono
        .chunks(frame_len)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let mean_frame = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;
    let var_frame = frame_rms
        .iter()
        .map(|r| (r - mean_frame).powi(2))
        .sum::<f32>()
        / frame_rms.len() as f32;
    let cv = var_frame.sqrt() / mean_frame.max(1e-9);
    assert!(
        cv > 0.15,
        "frame-RMS coefficient of variation {cv:.3} — constant energy is not music (catches \
         tones, steady noise beds, DC drones)"
    );

    // Long-term power spectrum via the shared radix-2 STFT.
    let n_fft = 2048;
    let window = candle_audio::dsp::hann_window(n_fft);
    let spec = candle_audio::dsp::stft(interior, n_fft, n_fft / 2, &window).expect("stft");
    let mag = spec.magnitude();
    let mut bin_energy = vec![0f64; spec.n_bins];
    for (bin, e) in bin_energy.iter_mut().enumerate() {
        *e = mag[bin * spec.n_frames..(bin + 1) * spec.n_frames]
            .iter()
            .map(|m| (*m as f64) * (*m as f64))
            .sum();
    }
    let total: f64 = bin_energy.iter().sum();
    assert!(total > 0.0, "spectrum has no energy");

    // PEAK-BIN CAP.
    let peak = bin_energy.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        peak / total < 0.5,
        "peak spectral bin carries {:.1}% of total energy — a pure tone is not music",
        100.0 * peak / total
    );

    // OCTAVE-BAND SPREAD: ≥ 4 of 7 octave bands each carry ≥ 0.5% of the total energy.
    let hz_per_bin = 48_000.0 / n_fft as f64;
    let bands = [
        (0.0, 375.0),
        (375.0, 750.0),
        (750.0, 1_500.0),
        (1_500.0, 3_000.0),
        (3_000.0, 6_000.0),
        (6_000.0, 12_000.0),
        (12_000.0, 24_000.0),
    ];
    let mut occupied = 0;
    for (lo, hi) in bands {
        let e: f64 = bin_energy
            .iter()
            .enumerate()
            .filter(|(b, _)| {
                let hz = *b as f64 * hz_per_bin;
                hz >= lo && hz < hi
            })
            .map(|(_, e)| *e)
            .sum();
        if e / total >= 0.005 {
            occupied += 1;
        }
    }
    assert!(
        occupied >= 4,
        "only {occupied}/7 octave bands carry ≥0.5% energy — narrowband is not music"
    );

    // SPECTRAL-FLATNESS CEILING (white-noise catch).
    let eps = 1e-12;
    let powers: Vec<f64> = bin_energy[1..]
        .iter()
        .map(|e| e / spec.n_frames as f64 + eps)
        .collect();
    let log_mean = powers.iter().map(|p| p.ln()).sum::<f64>() / powers.len() as f64;
    let arith_mean = powers.iter().sum::<f64>() / powers.len() as f64;
    let flatness = log_mean.exp() / arith_mean;
    assert!(
        flatness < 0.5,
        "spectral flatness {flatness:.3} — white noise is not rendered music"
    );

    // RHYTHMIC PERIODICITY: autocorrelate the (mean-removed) frame-energy envelope and require a
    // periodic peak in a musical tempo band (40–200 bpm at the 25 ms frame hop).
    let env: Vec<f32> = frame_rms.iter().map(|r| r - mean_frame).collect();
    let frames_per_sec = 48_000.0 / frame_len as f32;
    let lag_lo = (frames_per_sec * 60.0 / 200.0).round() as usize; // 200 bpm
    let lag_hi = (frames_per_sec * 60.0 / 40.0).round() as usize; // 40 bpm
    let energy0: f32 = env.iter().map(|x| x * x).sum::<f32>().max(1e-9);
    let mut best = 0.0f32;
    for lag in lag_lo..=lag_hi.min(env.len().saturating_sub(1)) {
        let ac: f32 = env
            .iter()
            .zip(env.iter().skip(lag))
            .map(|(a, b)| a * b)
            .sum();
        best = best.max(ac / energy0);
    }
    assert!(
        best > 0.1,
        "frame-energy autocorrelation peak {best:.3} in the 40–200 bpm band is too weak — no beat \
         (catches arrhythmic drones / noise)"
    );

    // Write the playable evidence.
    let out_path = std::env::var("ACESTEP_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-music-sc12842.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write WAV");
    println!(
        "acestep_music_wav_conformance: wrote {} ({secs:.2}s @ {} Hz stereo, RMS {rms:.4}, frame CV \
         {cv:.3}, peak-bin {:.1}%, bands ≥0.5%: {occupied}/7, flatness {flatness:.3}, beat-ac {best:.3})",
        out_path.display(),
        track.sample_rate,
        100.0 * peak / total,
    );

    // Determinism at the WAV layer.
    let out2 = generator
        .generate(&req, &mut |_| {})
        .expect("second generate");
    let track2 = match out2 {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(
        track.samples, track2.samples,
        "seeded synthesis must be deterministic"
    );
}
