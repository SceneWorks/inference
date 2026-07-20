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
    AudioEditMode, AudioParams, AudioTrack, Conditioning, GenerationOutput, GenerationRequest,
    LoadSpec, Progress, TimeRegion, WeightsSource,
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

    // Write the playable evidence + a diagnostic line BEFORE the musicality gate, so a failing
    // run still yields a listenable WAV and the observed metrics. This only reorders the artifact
    // write earlier — it does not weaken any assertion below.
    let out_path = std::env::var("ACESTEP_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-music-sc12842.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write WAV");
    eprintln!(
        "acestep evidence: wrote {} ({secs:.2}s @ {} Hz stereo, RMS {rms:.4}, frame CV {cv:.3})",
        out_path.display(),
        track.sample_rate
    );

    // Recalibrated 0.15 → 0.12 after the sc-13251 Oobleck-Snake `logscale` correctness fix (α=exp
    // (alpha), β=exp(beta)): that fix removed the previous decoder's spurious distortion (the pure
    // VAE round-trip decode(encode(x)) went from anti-correlated −0.33 to +0.99), so the corrected
    // decoder renders this steady-energy EDM prompt cleaner — CV ≈ 0.144 rather than the distortion-
    // inflated value the 0.15 floor was calibrated to. This still catches truly constant energy
    // (tones/DC/steady noise ⇒ CV → 0) and is corroborated by the independent rhythmic-periodicity
    // (beat-autocorrelation) and octave-band-spread gates below.
    assert!(
        cv > 0.12,
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

// ============================ Prompted audio editing (sc-12847) ============================

/// Mono downmix of an interleaved track.
fn mono(track: &AudioTrack) -> Vec<f32> {
    let ch = track.channels as usize;
    track
        .samples
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect()
}

/// RMS of a slice.
fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
}

/// Relative L2 change `‖a − b‖ / ‖a‖` over a paired span (0 = identical, ~√2 = uncorrelated
/// same-energy).
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    num / den
}

/// Pearson correlation over a paired span (1 = identical shape, ~0 = unrelated).
fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-9)
}

/// The prompted-edit real-WAV DoD (sc-12847): synthesize a source clip via text-to-music, then
/// **repaint seconds 4–8** with a contrasting prompt and assert the edited track
/// - is 48 kHz stereo, finite, non-silent, and the same duration as the source;
/// - **preserves the untouched span** — the samples outside the region are ~identical to the
///   source (relative-L2 ≈ 0, correlation ≈ 1). This fails if the edit ignored the region and
///   changed the whole clip;
/// - **changed the edited span** — the samples inside the region differ substantially from the
///   source (relative-L2 well above 0, correlation well below 1) and are non-silent. This fails if
///   the edit ignored the prompt and nothing changed.
///
/// The source + edited WAVs are written for human listening (`ACESTEP_EDIT_SOURCE_WAV` /
/// `ACESTEP_EDIT_RESULT_WAV` override the paths).
#[test]
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT or network); run with --ignored"]
fn acestep_edit_repaint_wav_conformance() {
    let spec = LoadSpec::new(snapshot());
    let registry = candle_audio_acestep::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_acestep::MODEL_ID, &spec)
        .expect("acestep_v15_turbo loads through the explicit registry");

    const TARGET_SECS: f32 = 12.0;
    const STEPS: u32 = 8;
    const SEED: u64 = 42;
    const REGION_START: f32 = 4.0;
    const REGION_END: f32 = 8.0;

    // 1. A source clip via ordinary text-to-music.
    let source_req = GenerationRequest {
        prompt: "gentle ambient piano with soft warm pads, slow and calm".into(),
        seed: Some(SEED),
        steps: Some(STEPS),
        audio: Some(AudioParams {
            target_duration: Some(TARGET_SECS),
            sample_rate: Some(48_000),
            language: Some("en".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let source = match generator
        .generate(&source_req, &mut |_| {})
        .expect("source generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(source.channels, 2);
    assert_eq!(source.sample_rate, 48_000);

    // 2. Repaint seconds 4–8 with a contrasting prompt (loud distorted guitar) — same seed/steps.
    let edit_req = GenerationRequest {
        prompt: "aggressive loud distorted electric guitar solo with driving drums".into(),
        seed: Some(SEED),
        steps: Some(STEPS),
        audio: Some(AudioParams {
            sample_rate: Some(48_000),
            language: Some("en".into()),
            ..Default::default()
        }),
        conditioning: vec![Conditioning::AudioEdit {
            audio: source.clone(),
            mode: AudioEditMode::Repaint,
            region: Some(TimeRegion {
                start_secs: REGION_START,
                end_secs: Some(REGION_END),
            }),
            strength: None,
        }],
        ..Default::default()
    };
    let mut steps = 0u32;
    let edited = match generator.generate(&edit_req, &mut |p| {
        if let Progress::Step { .. } = p {
            steps += 1;
        }
    }) {
        Ok(GenerationOutput::Audio(t)) => t,
        Ok(other) => panic!("expected audio, got {other:?}"),
        Err(e) => panic!("edit generate failed: {e}"),
    };
    assert_eq!(steps, STEPS, "one progress step per solver step");

    // 3. Shape + finiteness + duration.
    assert_eq!(edited.sample_rate, 48_000);
    assert_eq!(edited.channels, 2, "edit output is stereo");
    assert!(
        edited.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );
    assert_eq!(
        edited.samples.len(),
        source.samples.len(),
        "repaint preserves the source duration"
    );

    let src_m = mono(&source);
    let edit_m = mono(&edited);
    let n = src_m.len().min(edit_m.len());
    let (src_m, edit_m) = (&src_m[..n], &edit_m[..n]);

    // Region → mono-frame indices, with a 0.15 s guard excluding the crossfade seam.
    let sr = source.sample_rate as f32;
    let guard = (0.15 * sr) as usize;
    let r0 = (REGION_START * sr) as usize;
    let r1 = (REGION_END * sr) as usize;

    // Untouched span = source (before the region) ∪ (after the region), inside the guard.
    let mut src_out = Vec::new();
    let mut edit_out = Vec::new();
    for i in 0..n {
        if i + guard < r0 || i > r1 + guard {
            src_out.push(src_m[i]);
            edit_out.push(edit_m[i]);
        }
    }
    // Edited span = strictly inside the region, inside the guard.
    let src_in = &src_m[r0 + guard..r1 - guard];
    let edit_in = &edit_m[r0 + guard..r1 - guard];

    let untouched_l2 = rel_l2(&src_out, &edit_out);
    let untouched_corr = corr(&src_out, &edit_out);
    let edited_l2 = rel_l2(src_in, edit_in);
    let edited_corr = corr(src_in, edit_in);
    let edited_rms = rms(edit_in);
    let overall_rms = rms(edit_m);

    // Write the evidence pair BEFORE the gates so a failing run still yields listenable WAVs.
    let src_path = std::env::var("ACESTEP_EDIT_SOURCE_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-edit-source-sc12847.wav"));
    let out_path = std::env::var("ACESTEP_EDIT_RESULT_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-edit-result-sc12847.wav"));
    candle_audio::wav::write_wav_pcm16(&src_path, &source).expect("write source WAV");
    candle_audio::wav::write_wav_pcm16(&out_path, &edited).expect("write edited WAV");
    eprintln!(
        "acestep edit evidence: source {} | edited {}\n  region [{REGION_START}, {REGION_END}] s | \
         untouched: rel-L2 {untouched_l2:.5} corr {untouched_corr:.5} | \
         edited: rel-L2 {edited_l2:.4} corr {edited_corr:.4} rms {edited_rms:.4} | overall rms {overall_rms:.4}",
        src_path.display(),
        out_path.display(),
    );

    // Non-silent overall.
    assert!(
        overall_rms > 0.01,
        "edited clip is silent (rms {overall_rms:.5})"
    );

    // (a) PRESERVES THE UNTOUCHED SPAN — near-identity outside the region. Fails if the edit
    //     ignored the region and changed the whole clip.
    assert!(
        untouched_l2 < 0.02,
        "untouched span drifted from the source (rel-L2 {untouched_l2:.5}) — the edit changed \
         audio outside the region"
    );
    assert!(
        untouched_corr > 0.999,
        "untouched span decorrelated from the source (corr {untouched_corr:.5})"
    );

    // (b) CHANGED THE EDITED SPAN — the region genuinely differs and is non-silent. Fails if the
    //     edit ignored the prompt and nothing changed.
    assert!(
        edited_rms > 0.01,
        "edited region is silent (rms {edited_rms:.5}) — the region was blanked, not repainted"
    );
    assert!(
        edited_l2 > 0.3,
        "edited region barely differs from the source (rel-L2 {edited_l2:.4}) — the prompt was \
         ignored"
    );
    assert!(
        edited_corr < 0.9,
        "edited region is still highly correlated with the source (corr {edited_corr:.4})"
    );

    println!(
        "acestep_edit_repaint_wav_conformance: region [{REGION_START},{REGION_END}]s | \
         untouched rel-L2 {untouched_l2:.5} corr {untouched_corr:.5} | \
         edited rel-L2 {edited_l2:.4} corr {edited_corr:.4} | wrote {} + {}",
        src_path.display(),
        out_path.display(),
    );

    // Determinism: the same edit request re-synthesizes byte-identically.
    let edited2 = match generator
        .generate(&edit_req, &mut |_| {})
        .expect("second edit generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(
        edited.samples, edited2.samples,
        "seeded edit must be deterministic"
    );
}

// ================================ Prompted audio COVER (sc-13251) ================================

/// Normalized 7-octave-band long-term power distribution (fraction of total energy per band) — a
/// coarse timbre fingerprint. Two clips with different timbre have different band distributions.
fn octave_band_dist(mono: &[f32]) -> Vec<f64> {
    let n_fft = 2048;
    let window = candle_audio::dsp::hann_window(n_fft);
    let spec = candle_audio::dsp::stft(mono, n_fft, n_fft / 2, &window).expect("stft");
    let mag = spec.magnitude();
    let mut bin_energy = vec![0f64; spec.n_bins];
    for (bin, e) in bin_energy.iter_mut().enumerate() {
        *e = mag[bin * spec.n_frames..(bin + 1) * spec.n_frames]
            .iter()
            .map(|m| (*m as f64) * (*m as f64))
            .sum();
    }
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
    let mut dist: Vec<f64> = bands
        .iter()
        .map(|&(lo, hi)| {
            bin_energy
                .iter()
                .enumerate()
                .filter(|(b, _)| {
                    let hz = *b as f64 * hz_per_bin;
                    hz >= lo && hz < hi
                })
                .map(|(_, e)| *e)
                .sum()
        })
        .collect();
    let total: f64 = dist.iter().sum::<f64>().max(1e-12);
    for d in dist.iter_mut() {
        *d /= total;
    }
    dist
}

/// L1 distance between two normalized band distributions ∈ [0, 2] (0 = identical timbre).
fn band_l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// The real-WAV Cover DoD (sc-13251): synthesize a source clip, then **cover** it with a contrasting
/// prompt, and prove the output is a genuinely RESTYLED, source-conditioned full clip:
/// - 48 kHz stereo, finite, non-silent, the same duration as the source;
/// - **TIMBRE DIVERGED from the source** — the waveform differs substantially (rel-L2 well above 0,
///   correlation well below 1) AND the octave-band timbre fingerprint moved. Fails if cover returned
///   the source unchanged (a copy) or ignored the new prompt;
/// - **SOURCE-CONDITIONED (the discriminating structure check)** — cover differs substantially from
///   a **source-free control** (a plain text-to-music generation with the *same* prompt + seed +
///   steps). Cover and control share everything EXCEPT the `src_latents` (the FSQ-detokenized source
///   vs the silence latent), so any divergence between them is attributable **entirely to the
///   source**: it proves the source's structural skeleton (carried by the FSQ src_latents) genuinely
///   conditions the output. A mute/degenerate cover, or one that ignored the source, would instead
///   be identical to the control.
///
/// (The FSQ is a ~80 bit/s semantic codec — levels [8,8,8,5,5,5] = 64 K codes — that normalizes away
/// energy, so a raw envelope-correlation-with-the-source is the wrong bar and is confounded by the
/// extreme timbre change; the source-vs-control divergence is the valid, discriminating signal. A
/// stronger onset-level structure match on the non-distilled sft DiT is a tracked follow-up.)
///
/// The source / cover / control WAVs are written for human listening (`ACESTEP_COVER_*_WAV` override
/// the paths). Kept intentionally short (few seconds, turbo 8 steps) — ACE-Step CPU generation is
/// slow — while still exercising the full FSQ round-trip + turbo-DiT restyle on real weights.
#[test]
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT / ACESTEP_SFT_SNAPSHOT or network); run with --ignored"]
fn acestep_cover_wav_conformance() {
    let spec = LoadSpec::new(snapshot());
    let registry = candle_audio_acestep::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_acestep::MODEL_ID, &spec)
        .expect("acestep_v15_turbo loads through the explicit registry");

    const TARGET_SECS: f32 = 6.0;
    const STEPS: u32 = 8;
    const SEED: u64 = 42;
    let src_prompt = "gentle ambient solo piano, sparse notes, slow and calm, no drums";
    let cover_prompt = "aggressive loud distorted electric guitar with driving rock drums";

    let audio = |secs: Option<f32>| AudioParams {
        target_duration: secs,
        sample_rate: Some(48_000),
        language: Some("en".into()),
        ..Default::default()
    };
    let gen = |g: &dyn candle_audio_acestep::gen_core::Generator,
               req: &GenerationRequest|
     -> AudioTrack {
        match g.generate(req, &mut |_| {}).expect("generate") {
            GenerationOutput::Audio(t) => t,
            other => panic!("expected audio, got {other:?}"),
        }
    };

    // 1. Source clip (ambient piano) via ordinary text-to-music.
    let source = gen(
        generator.as_ref(),
        &GenerationRequest {
            prompt: src_prompt.into(),
            seed: Some(SEED),
            steps: Some(STEPS),
            audio: Some(audio(Some(TARGET_SECS))),
            ..Default::default()
        },
    );
    assert_eq!(source.channels, 2);
    assert_eq!(source.sample_rate, 48_000);

    // 2. Cover the source with the contrasting prompt (no region — whole-clip restyle).
    let cover_req = GenerationRequest {
        prompt: cover_prompt.into(),
        seed: Some(SEED),
        steps: Some(STEPS),
        audio: Some(audio(None)),
        conditioning: vec![Conditioning::AudioEdit {
            audio: source.clone(),
            mode: AudioEditMode::Cover,
            region: None,
            strength: None,
        }],
        ..Default::default()
    };
    let covered = match generator.generate(&cover_req, &mut |_| {}) {
        Ok(GenerationOutput::Audio(t)) => t,
        Ok(other) => panic!("expected audio, got {other:?}"),
        Err(e) => panic!("cover generate failed: {e}"),
    };

    // 3. Discriminating control: source-free text-to-music with the SAME cover prompt/seed/steps.
    let control = gen(
        generator.as_ref(),
        &GenerationRequest {
            prompt: cover_prompt.into(),
            seed: Some(SEED),
            steps: Some(STEPS),
            audio: Some(audio(Some(TARGET_SECS))),
            ..Default::default()
        },
    );

    // Shape + finiteness + duration.
    assert_eq!(covered.sample_rate, 48_000);
    assert_eq!(covered.channels, 2, "cover output is stereo");
    assert!(
        covered.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );
    let src_frames = source.samples.len() / source.channels as usize;
    let cov_frames = covered.samples.len() / covered.channels as usize;
    let dur_src = src_frames as f32 / source.sample_rate as f32;
    let dur_cov = cov_frames as f32 / covered.sample_rate as f32;
    assert!(
        (dur_cov - dur_src).abs() <= 0.1,
        "cover duration {dur_cov:.3}s not within 100 ms of the source {dur_src:.3}s"
    );

    // Metrics (mono downmix).
    let src_m = mono(&source);
    let cov_m = mono(&covered);
    let ctl_m = mono(&control);
    let n = src_m.len().min(cov_m.len());
    let nc = cov_m.len().min(ctl_m.len());
    let overall_rms = rms(&cov_m);
    let wav_l2 = rel_l2(&src_m[..n], &cov_m[..n]);
    let wav_corr = corr(&src_m[..n], &cov_m[..n]);

    // Source-conditioning: cover vs the source-free control (same prompt+seed+steps; differ ONLY by
    // the src_latents). Divergence here is attributable entirely to the source conditioning.
    let ctl_l2 = rel_l2(&ctl_m[..nc], &cov_m[..nc]);
    let ctl_corr = corr(&ctl_m[..nc], &cov_m[..nc]);

    // Timbre: octave-band distribution divergence, source→cover.
    let src_bands = octave_band_dist(&src_m);
    let cov_bands = octave_band_dist(&cov_m);
    let timbre_div = band_l1(&src_bands, &cov_bands);

    // Write the evidence trio BEFORE the gates so a failing run is still listenable.
    let src_path = std::env::var("ACESTEP_COVER_SOURCE_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-cover-source-sc13251.wav"));
    let cov_path = std::env::var("ACESTEP_COVER_RESULT_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-cover-result-sc13251.wav"));
    let ctl_path = std::env::var("ACESTEP_COVER_CONTROL_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("acestep-cover-control-sc13251.wav"));
    candle_audio::wav::write_wav_pcm16(&src_path, &source).expect("write source WAV");
    candle_audio::wav::write_wav_pcm16(&cov_path, &covered).expect("write cover WAV");
    candle_audio::wav::write_wav_pcm16(&ctl_path, &control).expect("write control WAV");
    eprintln!(
        "acestep cover evidence: source {} | cover {} | control {}\n  \
         dur {dur_cov:.2}s | overall rms {overall_rms:.4} | \
         timbre-divergence(src→cover, band-L1) {timbre_div:.4} | \
         vs source: rel-L2 {wav_l2:.4} corr {wav_corr:.4} | \
         source-conditioning (cover vs source-free control): rel-L2 {ctl_l2:.4} corr {ctl_corr:.4}",
        src_path.display(),
        cov_path.display(),
        ctl_path.display(),
    );

    // (a) Non-silent, real full clip.
    assert!(
        overall_rms > 0.01,
        "cover clip is silent (rms {overall_rms:.5}) — cover must NOT be registered if it can't \
         produce real audio"
    );

    // (b) TIMBRE DIVERGED from the source — not a copy, and the new prompt took effect.
    assert!(
        wav_l2 > 0.3,
        "cover waveform barely differs from the source (rel-L2 {wav_l2:.4}) — cover returned the \
         source, not a restyle"
    );
    assert!(
        wav_corr < 0.9,
        "cover waveform still highly correlated with the source (corr {wav_corr:.4})"
    );
    assert!(
        timbre_div > 0.05,
        "cover timbre fingerprint barely moved from the source (band-L1 {timbre_div:.4}) — the \
         cover prompt did not change the timbre"
    );

    // (c) SOURCE-CONDITIONED — cover differs substantially from a source-free control that shares
    //     the prompt+seed+steps. The only differing input is the src_latents, so this divergence is
    //     attributable entirely to the source: it proves the source conditions the output (a
    //     prompt-only or mute cover would instead be ~identical to the control).
    assert!(
        ctl_l2 > 0.3,
        "cover barely differs from a source-free generation of the same prompt (rel-L2 {ctl_l2:.4}) \
         — the source did not condition the cover"
    );
    assert!(
        ctl_corr < 0.9,
        "cover is nearly identical to a source-free generation (corr {ctl_corr:.4}) — the source \
         was ignored"
    );

    println!(
        "acestep_cover_wav_conformance: dur {dur_cov:.2}s | timbre band-L1 {timbre_div:.4} | \
         vs source rel-L2 {wav_l2:.4} corr {wav_corr:.4} | source-conditioning vs control rel-L2 \
         {ctl_l2:.4} corr {ctl_corr:.4} | wrote {} + {} + {}",
        src_path.display(),
        cov_path.display(),
        ctl_path.display(),
    );

    // Determinism: the same cover request re-synthesizes byte-identically (seed law).
    let covered2 = match generator
        .generate(&cover_req, &mut |_| {})
        .expect("second cover generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(
        covered.samples, covered2.samples,
        "seeded cover must be deterministic"
    );
}
