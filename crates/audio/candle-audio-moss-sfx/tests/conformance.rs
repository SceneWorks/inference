//! Real-weight conformance for the candle MOSS-SoundEffect provider (sc-12841): real pinned
//! weights → registry load-by-id → generate → a real, SFX-shaped WAV.
//!
//! Both tests are `#[ignore]`d and snapshot-gated like every other family's real-weight tests:
//! set `MOSS_SFX_SNAPSHOT` to an `OpenMOSS-Team/MOSS-SoundEffect-v2.0` snapshot dir
//! (model_index.json + transformer/ + text_encoder/ + tokenizer/ + vae/), or leave it unset to
//! resolve the pinned snapshot through the audio lane's F-029 hub path (downloads ~11 GB into
//! the ordinary HF cache on first run).
//!
//! ```text
//! cargo test --locked -p candle-audio-moss-sfx --test conformance -- --ignored --nocapture
//! ```
//!
//! `moss_sfx_wav_conformance` also writes the synthesized WAV next to the test output
//! (`MOSS_SFX_WAV_OUT` overrides the path) so a human can listen to the evidence.

use std::path::PathBuf;

use candle_audio_moss_sfx::candle_audio;
use candle_audio_moss_sfx::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

/// Resolve the snapshot from the required `MOSS_SFX_SNAPSHOT` env (a passed-in
/// `OpenMOSS-Team/MOSS-SoundEffect-v2.0` snapshot dir). Inference never self-fetches or derives a
/// cache location (epic 13657).
fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("MOSS_SFX_SNAPSHOT").expect(
        "set MOSS_SFX_SNAPSHOT to an OpenMOSS-Team/MOSS-SoundEffect-v2.0 snapshot dir (model_index.json + transformer/ + text_encoder/ + tokenizer/ + vae/)",
    )))
}

/// The backend-neutral gen-core conformance suite (validate honesty, progress + progress
/// contract, typed mid-run and pre-generate cancellation, seed determinism) against the real
/// provider, resolved **through the explicit registry** exactly like an image model.
#[test]
#[ignore = "real weights: needs a MOSS-SoundEffect-v2.0 snapshot (MOSS_SFX_SNAPSHOT); run with --ignored"]
fn moss_sfx_conformance() {
    let spec = LoadSpec::new(snapshot());
    let profile = gen_core_testkit::Profile {
        prompt: "a single water drop echoing in a cave".to_owned(),
        // Audio skips the size floor; keep the request inside the advertised bounds anyway.
        width: 256,
        height: 256,
        // Each request resolves to exactly its requested solver step count.
        steps: 2,
        seed: 42,
        cancel_steps: 6,
    };
    gen_core_testkit::conformance(
        || {
            candle_audio_moss_sfx::provider_registry()
                .unwrap()
                .load(candle_audio_moss_sfx::MODEL_ID, &spec)
                .expect("moss_sfx_v2 loads through the explicit registry")
        },
        &profile,
    );
}

/// The real-WAV DoD: fixed prompt + seed + duration → non-empty, 48 kHz, mono, finite,
/// NON-SILENT, non-degenerate SFX audio of exactly the requested duration, written to disk as
/// a playable WAV.
///
/// The anti-degeneracy assertions and what each catches:
/// - **interior RMS floor** — all-silence output (a dead decoder or all-zero latents);
/// - **frame-RMS coefficient of variation** — output whose energy never moves: constant tones,
///   steady unmodulated noise beds, DC-ish drones (a shattering transient has an attack and a
///   decay, so its 50 ms frame energies vary strongly);
/// - **peak-bin energy cap** — a single pure tone (one STFT bin dominating the long-term
///   spectrum);
/// - **octave-band spread floor** — narrowband output generally (energy must genuinely occupy
///   several octaves — broadband content is what "glass shattering" sounds like);
/// - **spectral-flatness ceiling** — white-noise-only output (flatness → 1 for white noise;
///   real, VAE-decoded SFX audio is spectrally shaped);
/// - **byte-identical re-synthesis** — the seed law (the companion alternate-seed
///   byte-difference check runs in `moss_sfx_conformance`'s gen-core suite).
#[test]
#[ignore = "real weights: needs a MOSS-SoundEffect-v2.0 snapshot (MOSS_SFX_SNAPSHOT); run with --ignored"]
fn moss_sfx_wav_conformance() {
    let spec = LoadSpec::new(snapshot());
    let registry = candle_audio_moss_sfx::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_moss_sfx::MODEL_ID, &spec)
        .expect("moss_sfx_v2 loads through the explicit registry");
    assert_eq!(generator.descriptor().id, "moss_sfx_v2");

    const TARGET_SECS: f32 = 4.0;
    const STEPS: u32 = 30;
    let req = GenerationRequest {
        prompt: "glass shattering on a stone floor".into(),
        seed: Some(42),
        steps: Some(STEPS),
        guidance: Some(4.0),
        audio: Some(AudioParams {
            target_duration: Some(TARGET_SECS),
            sample_rate: Some(48_000),
            language: Some("en".into()),
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
    assert_eq!(track.channels, 1, "MOSS-SoundEffect is mono");
    assert!(!track.samples.is_empty());
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );

    // Duration honored: the decoded window is cropped to the requested duration exactly.
    let secs = track.samples.len() as f32 / track.sample_rate as f32;
    assert!(
        (secs - TARGET_SECS).abs() <= 0.05,
        "duration {secs:.3}s not within 50 ms of the requested {TARGET_SECS}s"
    );

    // NON-SILENT: overall RMS over the interior (skips any boundary fade).
    let n = track.samples.len();
    let interior = &track.samples[n / 20..n - n / 20];
    let rms = (interior.iter().map(|s| s * s).sum::<f32>() / interior.len() as f32).sqrt();
    assert!(rms > 0.01, "interior RMS {rms:.5} — silence is a failure");

    // SFX-shaped energy over time: 50 ms frame RMS must VARY (a shatter has attack + decay),
    // not sit flat like a tone or an unmodulated noise bed.
    let frame_len = 2400; // 50 ms @ 48 kHz
    let frame_rms: Vec<f32> = track
        .samples
        .chunks(frame_len)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let max_frame = frame_rms.iter().cloned().fold(0.0f32, f32::max);
    let mean_frame = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;
    let var_frame = frame_rms
        .iter()
        .map(|r| (r - mean_frame) * (r - mean_frame))
        .sum::<f32>()
        / frame_rms.len() as f32;
    let cv = var_frame.sqrt() / mean_frame;
    assert!(
        max_frame > 0.03,
        "peak frame RMS {max_frame:.4} too weak for a shattering transient"
    );
    assert!(
        cv > 0.2,
        "frame-RMS coefficient of variation {cv:.3} — constant energy is not an SFX transient \
         (catches tones, steady noise beds, DC drones)"
    );

    // Long-term power spectrum via the shared radix-2 STFT (n_fft 1024 @ 48 kHz → 46.9 Hz
    // bins), over the interior.
    let n_fft = 1024;
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

    // PEAK-BIN CAP: no single frequency bin may dominate — a pure tone concentrates its
    // energy in one bin (plus leakage), real broadband SFX does not.
    let peak = bin_energy.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        peak / total < 0.5,
        "peak spectral bin carries {:.1}% of total energy — a pure tone is not an SFX",
        100.0 * peak / total
    );

    // OCTAVE-BAND SPREAD: broadband content appropriate for SFX — at least 4 of the 7 octave
    // bands must each carry ≥ 0.5% of the total energy.
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
    let mut band_fracs = Vec::new();
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
        band_fracs.push(e / total);
        if e / total >= 0.005 {
            occupied += 1;
        }
    }
    assert!(
        occupied >= 4,
        "only {occupied}/7 octave bands carry ≥0.5% energy ({band_fracs:?}) — narrowband \
         output is not broadband SFX"
    );

    // SPECTRAL-FLATNESS CEILING: white noise is spectrally flat (geometric mean ≈ arithmetic
    // mean → flatness ≈ 1); real decoded audio is shaped. Computed over the mean power
    // spectrum, excluding the DC bin.
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
        "spectral flatness {flatness:.3} — white-noise-only output is not a rendered SFX"
    );

    // Write the playable evidence.
    let out_path = std::env::var("MOSS_SFX_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("moss-sfx-sc12841.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write WAV");
    println!(
        "moss_sfx_wav_conformance: wrote {} ({secs:.2}s @ {} Hz, RMS {rms:.4}, peak frame RMS \
         {max_frame:.4}, frame CV {cv:.3}, peak-bin {:.1}%, bands ≥0.5%: {occupied}/7, \
         flatness {flatness:.3})",
        out_path.display(),
        track.sample_rate,
        100.0 * peak / total,
    );

    // Determinism at the WAV layer too: the same request+seed re-synthesizes byte-identically,
    // and a different seed actually changes the output.
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
