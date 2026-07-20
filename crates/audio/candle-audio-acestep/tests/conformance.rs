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

/// 12-bin **chroma** (pitch-class) profile: fold the long-term STFT magnitude spectrum onto the 12
/// pitch classes (C, C#, … B), summed over time, then L2-normalize. A pitch and its harmonics map
/// largely onto the same pitch classes regardless of the instrument voicing them, so chroma captures
/// the melodic/harmonic content (the *what notes / what key*) while being largely invariant to timbre
/// (the *what instrument*). That is exactly the content a musical **cover** preserves while it changes
/// the timbre — so it is the right feature to gate Cover's SEMANTIC structure-preservation (sc-13251,
/// Option A: "preserves musical structure" = melodic/genre preservation, not onset/beat ordering).
///
/// `n_fft = 8192` (≈ 5.86 Hz/bin at 48 kHz) resolves adjacent semitones down to the low-mid register;
/// only the ~C2–C8 musical band is folded in (sub-bass rumble and very-high harmonics muddy the
/// pitch-class estimate). Energy (magnitude²) is accumulated so louder partials dominate the profile.
fn chroma(mono: &[f32]) -> [f64; 12] {
    let n_fft = 8192;
    let window = candle_audio::dsp::hann_window(n_fft);
    let spec = candle_audio::dsp::stft(mono, n_fft, n_fft / 4, &window).expect("stft");
    let mag = spec.magnitude();
    let hz_per_bin = 48_000.0 / n_fft as f64;
    let (f_lo, f_hi) = (60.0, 5_000.0);
    let mut pc = [0f64; 12];
    for bin in 1..spec.n_bins {
        let hz = bin as f64 * hz_per_bin;
        if hz < f_lo || hz > f_hi {
            continue;
        }
        // MIDI note number → pitch class: 69 + 12·log2(f/440), rounded, mod 12.
        let midi = 69.0 + 12.0 * (hz / 440.0).log2();
        let class = (midi.round() as i64).rem_euclid(12) as usize;
        let e: f64 = mag[bin * spec.n_frames..(bin + 1) * spec.n_frames]
            .iter()
            .map(|m| (*m as f64) * (*m as f64))
            .sum();
        pc[class] += e;
    }
    let norm = pc.iter().map(|c| c * c).sum::<f64>().sqrt().max(1e-12);
    for c in pc.iter_mut() {
        *c /= norm;
    }
    pc
}

/// Pearson correlation of two 12-bin chroma profiles ∈ [−1, 1] — the primary SEMANTIC similarity.
/// Mean-centering removes the "every clip has some energy in every pitch class" baseline that biases
/// a raw cosine high, so it discriminates the *shape* of the pitch-class distribution (the key/tonal
/// centre) far more sharply than cosine: two clips in the same key correlate near 1, unrelated tonal
/// content correlates near 0 (or negative).
fn chroma_corr(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let ma = a.iter().sum::<f64>() / 12.0;
    let mb = b.iter().sum::<f64>() / 12.0;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-12)
}

/// The real-WAV Cover DoD (sc-13251, Option A): synthesize a source clip, then **cover** it with a
/// contrasting prompt, and prove the output is a genuinely RESTYLED clip that PRESERVES the source's
/// musical/genre content while the timbre changes. "Preserves musical structure" is read per Michael's
/// decision as SEMANTIC/melodic/genre preservation — how an ACE-Step cover actually behaves. The FSQ
/// conditioning is a ~80 bit/s semantic codec at 5 Hz: it carries the source's genre/melodic/tonal
/// CHARACTER, but NOT its onset/beat ordering (sc-13251 proved that unachievable) and NOT its absolute
/// key (a pitch-shifted source's cover is no less chroma-similar to the original than its own cover —
/// the cover re-anchors the key). The gate has two halves, both of which must hold:
///
/// - **CONTENT PRESERVED (positive, discriminating, NON-MASKING SEMANTIC gate)** — the feature is the
///   timbre-invariant **chroma** (pitch-class) profile: a pitch and its harmonics map onto the same
///   pitch classes regardless of instrument, so chroma captures the melodic/tonal content the cover
///   keeps while being blind to the timbre it changes. With two chroma-distinct sources A and B and
///   their covers under ONE shared `cover(prompt, seed, steps)`, the gate asserts, PER DIRECTION and
///   independently (never on a mean that could hide a weak leg): each source's own cover PRESERVES its
///   chroma (matched ↑ over a floor) AND is source-SPECIFIC (matched − mismatched, i.e. source ↔ its
///   OWN cover minus source ↔ the OTHER source's cover, over a floor). Cover runs on the non-distilled
///   **sft DiT** (see [`AceStepPipeline::cover`]); the distilled 8-step turbo DiT could not clear this
///   per-direction for the weaker leg (sc-13251). Per-direction is load-bearing — the mean alone would
///   let one leg sit at the noise floor while the other carries it, letting a source-agnostic cover of
///   the weak source slip through; a fully source-agnostic / broken-conditioning cover (e.g. the
///   known-bad VAE-encode run) makes the two covers ≈ identical ⇒ every margin ≈ 0, failing the gate.
///   (Chroma is a two-source **distribution/character** comparison, NOT an absolute-key one — a
///   pitch-shifted control gives ≈ 0 because the cover re-anchors key. LAION CLAP was also evaluated
///   but discriminated far less consistently across source pairs, so chroma is the gate. The FSQ codec
///   preserves DISTINCTIVE content; a generic bright clip is compressed to generic codes and loses its
///   per-source specificity — so the sources are chosen distinctive + tonal, e.g. the sitar.)
/// - **TIMBRE CHANGED (restyle proven)** — the octave-band timbre fingerprint moved from the source
///   (band-L1) AND the waveform diverged (rel-L2 up, correlation down). Fails if cover returned the
///   source unchanged or ignored the new prompt.
///
/// Plus: 48 kHz stereo, finite, non-silent, same duration as the source, deterministic (seed law).
///
/// The four WAVs are written for human listening (`ACESTEP_COVER_*_WAV` override the paths). Sources
/// are generated on the fast turbo DiT; only the two covers run on the slower non-distilled sft DiT.
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
    // The cover request's noise seed — SHARED by both covers so the only difference between them is
    // the source that conditions them.
    const COVER_SEED: u64 = 42;
    // Two chroma-DISTINCT sources (different prompts AND seeds) whose content the FSQ codec KEEPS, so
    // the non-distilled sft cover DiT preserves each per-direction. The FSQ (~80 bit/s at 5 Hz)
    // preserves a source's melodic content only when that content is DISTINCTIVE (a generic bright
    // clip is compressed to generic codes and its per-source specificity is lost — measured across
    // sources, sc-13251); and the two-source margin needs the sources chroma-DISTINCT. A = a solo
    // sitar raga: distinctive TIMBRE (so its content survives the codec — matched ≈ 0.79) AND tonal
    // on a specific scale (so its chroma is distinct from B — the rare combination that clears BOTH
    // legs). B = dark dissonant industrial electronic (distinctive too — matched ≈ 0.67). The
    // distilled 8-step turbo DiT could not clear the per-direction floor for the weaker leg on ANY
    // pair; the sft DiT does (verified sitar/industrial AND steel-drum/industrial, sc-13251).
    const SRC_A_SEED: u64 = 42;
    const SRC_B_SEED: u64 = 7;
    let src_a_prompt =
        "a hypnotic solo sitar raga with a resonant drone, distinctive twanging strings, meditative Indian classical";
    let src_b_prompt =
        "dark aggressive industrial electronic, heavy distorted bass, ominous grinding machine drones";
    // A brass cover shared by both covers — spectrally distinct from both sources (octave-band timbre
    // moves hard). The shared new timbre cancels in the matched-vs-mismatched comparison, leaving each
    // source's carried-over tonal character.
    let cover_prompt = "a brass ensemble of trumpets and trombones";

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
    let text2music = |prompt: &str, seed: u64| GenerationRequest {
        prompt: prompt.into(),
        seed: Some(seed),
        steps: Some(STEPS),
        audio: Some(audio(Some(TARGET_SECS))),
        ..Default::default()
    };
    // Same cover request for both sources: prompt + seed + steps identical, so the ONLY difference
    // between the two covers is the source's content.
    let cover_of = |src: &AudioTrack| GenerationRequest {
        prompt: cover_prompt.into(),
        seed: Some(COVER_SEED),
        steps: Some(STEPS),
        audio: Some(audio(None)),
        conditioning: vec![Conditioning::AudioEdit {
            audio: src.clone(),
            mode: AudioEditMode::Cover,
            region: None,
            strength: None,
        }],
        ..Default::default()
    };

    // 1–2. Two contrasting source clips via ordinary text-to-music.
    let source_a = gen(generator.as_ref(), &text2music(src_a_prompt, SRC_A_SEED));
    let source_b = gen(generator.as_ref(), &text2music(src_b_prompt, SRC_B_SEED));
    assert_eq!(source_a.channels, 2);
    assert_eq!(source_a.sample_rate, 48_000);

    // 3. Cover source A (the matched cover under test).
    let cover_req_a = cover_of(&source_a);
    let cover_a = match generator.generate(&cover_req_a, &mut |_| {}) {
        Ok(GenerationOutput::Audio(t)) => t,
        Ok(other) => panic!("expected audio, got {other:?}"),
        Err(e) => panic!("cover generate failed: {e}"),
    };
    // 4. Cover source B — the mismatched control: SAME cover prompt+seed+steps, contrasting source.
    let cover_b = gen(generator.as_ref(), &cover_of(&source_b));

    // Shape + finiteness + duration (on the matched cover).
    assert_eq!(cover_a.sample_rate, 48_000);
    assert_eq!(cover_a.channels, 2, "cover output is stereo");
    assert!(
        cover_a.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );
    let src_frames = source_a.samples.len() / source_a.channels as usize;
    let cov_frames = cover_a.samples.len() / cover_a.channels as usize;
    let dur_src = src_frames as f32 / source_a.sample_rate as f32;
    let dur_cov = cov_frames as f32 / cover_a.sample_rate as f32;
    assert!(
        (dur_cov - dur_src).abs() <= 0.1,
        "cover duration {dur_cov:.3}s not within 100 ms of the source {dur_src:.3}s"
    );

    // SEMANTIC feature: chroma (pitch-class) profiles of the two sources and two covers.
    let (sa, sb) = (mono(&source_a), mono(&source_b));
    let (ca, cb) = (mono(&cover_a), mono(&cover_b));
    let (chr_sa, chr_sb) = (chroma(&sa), chroma(&sb));
    let (chr_ca, chr_cb) = (chroma(&ca), chroma(&cb));
    //   matched    = source ↔ its OWN cover;   mismatched = source ↔ the OTHER source's cover.
    let matched_a = chroma_corr(&chr_sa, &chr_ca);
    let mismatched_a = chroma_corr(&chr_sa, &chr_cb);
    let matched_b = chroma_corr(&chr_sb, &chr_cb);
    let mismatched_b = chroma_corr(&chr_sb, &chr_ca);
    // PER-DIRECTION margins — EACH source's own cover must independently beat the mismatched control.
    // Gating on the mean alone lets one direction sit at the noise floor while the other compensates,
    // which would let a source-agnostic cover of the weak source slip through; the per-direction floor
    // closes that.
    let margin_a = matched_a - mismatched_a;
    let margin_b = matched_b - mismatched_b;
    let chroma_matched = 0.5 * (matched_a + matched_b);
    let chroma_mismatched = 0.5 * (mismatched_a + mismatched_b);
    let chroma_margin = chroma_matched - chroma_mismatched;
    // Precondition: the two sources are tonally distinct (else the discrimination is vacuous).
    let src_ab_chroma = chroma_corr(&chr_sa, &chr_sb);

    // TIMBRE divergence + waveform divergence, source A → its cover.
    let timbre_div = band_l1(&octave_band_dist(&sa), &octave_band_dist(&ca));
    let n = sa.len().min(ca.len());
    let wav_l2 = rel_l2(&sa[..n], &ca[..n]);
    let wav_corr = corr(&sa[..n], &ca[..n]);
    let rms_a = rms(&ca);
    let rms_b = rms(&cb);

    // Write the evidence BEFORE the gates so a failing run is still listenable + fully reported.
    let path = |var: &str, default: &str| {
        std::env::var(var)
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join(default))
    };
    let src_a_path = path(
        "ACESTEP_COVER_SOURCE_WAV",
        "acestep-cover-source-a-sc13251.wav",
    );
    let src_b_path = path(
        "ACESTEP_COVER_SOURCE_B_WAV",
        "acestep-cover-source-b-sc13251.wav",
    );
    let cov_a_path = path(
        "ACESTEP_COVER_RESULT_WAV",
        "acestep-cover-result-a-sc13251.wav",
    );
    let cov_b_path = path(
        "ACESTEP_COVER_CONTROL_WAV",
        "acestep-cover-result-b-sc13251.wav",
    );
    candle_audio::wav::write_wav_pcm16(&src_a_path, &source_a).expect("write source A WAV");
    candle_audio::wav::write_wav_pcm16(&src_b_path, &source_b).expect("write source B WAV");
    candle_audio::wav::write_wav_pcm16(&cov_a_path, &cover_a).expect("write cover A WAV");
    candle_audio::wav::write_wav_pcm16(&cov_b_path, &cover_b).expect("write cover B WAV");
    eprintln!(
        "acestep cover evidence (sc-13251, Option A chroma semantic gate):\n  \
         source A {} | source B {} | cover A {} | cover B (mismatched) {}\n  \
         dur {dur_cov:.2}s | cover-A rms {rms_a:.4} cover-B rms {rms_b:.4} | \
         timbre band-L1(srcA→coverA) {timbre_div:.4} | vs source A: rel-L2 {wav_l2:.4} corr {wav_corr:.4}\n  \
         chroma-corr A: matched {matched_a:.4} mismatched {mismatched_a:.4} = margin {margin_a:+.4} | \
         chroma-corr B: matched {matched_b:.4} mismatched {mismatched_b:.4} = margin {margin_b:+.4}\n  \
         matched-mean {chroma_matched:.4} - mismatched-mean {chroma_mismatched:.4} = MARGIN \
         {chroma_margin:+.4} | srcA↔srcB chroma {src_ab_chroma:.4}",
        src_a_path.display(),
        src_b_path.display(),
        cov_a_path.display(),
        cov_b_path.display(),
    );

    // (a) Non-silent, real full clips (both covers).
    assert!(
        rms_a > 0.01 && rms_b > 0.01,
        "a cover clip is silent (A rms {rms_a:.5}, B rms {rms_b:.5}) — cover must NOT be registered \
         if it can't produce real audio"
    );

    // (b) TIMBRE CHANGED from the source — not a copy, and the new prompt took effect.
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

    // Precondition for a MEANINGFUL discrimination: the two sources are tonally distinct (different
    // genre/instrumentation ⇒ different chroma profiles). If they were near-identical the mismatched
    // control could not discriminate. (These are contrasting prompts + seeds, so this holds.)
    assert!(
        src_ab_chroma < 0.85,
        "sources A and B are too tonally similar (chroma-corr {src_ab_chroma:.4}) — the mismatched \
         control cannot discriminate; pick more contrasting source prompts/seeds"
    );

    // (c) CONTENT PRESERVED — the positive, discriminating, NON-MASKING SEMANTIC gate on the
    //     non-distilled sft cover DiT. Each cover's chroma is closer to ITS OWN source than to the
    //     other source's cover, proving the cover carries THIS source's tonal/melodic character (not
    //     just the shared prompt). A source-agnostic or broken-conditioning cover makes the two
    //     covers ≈ identical ⇒ every margin ≈ 0. FOUR conjuncts, all required, chosen so NO single
    //     source's result can be masked by the other's:
    //       - EACH per-direction MATCHED chroma-corr (source ↔ its OWN cover) clears a floor — so each
    //         source's content is GENUINELY preserved (a high margin is not enough on its own: a cover
    //         that barely resembles its source can still out-score an even-more-unrelated other cover);
    //       - EACH per-direction MARGIN (matched − mismatched) clears a floor — source-SPECIFIC
    //         preservation, in both directions independently;
    //       - the aggregate margin clears its own (larger) floor.
    //     The distilled 8-step turbo DiT could NOT clear these per-direction (its weak leg sat at the
    //     noise floor, sc-13251); the sft DiT does — for sources whose content the FSQ codec keeps
    //     (distinctive + tonal, e.g. the sitar), verified over ≥2 chroma-distinct pairs. Floors sit
    //     below the measured values with headroom (numbers in the story comment / PR).
    const CHROMA_PER_DIR_MATCHED_FLOOR: f64 = 0.40;
    const CHROMA_PER_DIR_FLOOR: f64 = 0.03;
    const CHROMA_MARGIN: f64 = 0.05;
    assert!(
        matched_a > CHROMA_PER_DIR_MATCHED_FLOOR,
        "source A's own cover does not preserve source A's tonal content (matched {matched_a:.4} ≤ \
         floor {CHROMA_PER_DIR_MATCHED_FLOOR}) — content not preserved for this source"
    );
    assert!(
        matched_b > CHROMA_PER_DIR_MATCHED_FLOOR,
        "source B's own cover does not preserve source B's tonal content (matched {matched_b:.4} ≤ \
         floor {CHROMA_PER_DIR_MATCHED_FLOOR}) — content not preserved for this source"
    );
    assert!(
        margin_a > CHROMA_PER_DIR_FLOOR,
        "source A's cover is not conditioned on source A: matched {matched_a:.4} does not beat \
         mismatched {mismatched_a:.4} by > {CHROMA_PER_DIR_FLOOR} (margin {margin_a:+.4}) — this \
         direction is at the noise floor and would let a source-agnostic cover of A slip through"
    );
    assert!(
        margin_b > CHROMA_PER_DIR_FLOOR,
        "source B's cover is not conditioned on source B: matched {matched_b:.4} does not beat \
         mismatched {mismatched_b:.4} by > {CHROMA_PER_DIR_FLOOR} (margin {margin_b:+.4}) — this \
         direction is at the noise floor and would let a source-agnostic cover of B slip through"
    );
    assert!(
        chroma_margin > CHROMA_MARGIN,
        "cover is not conditioned on THIS source's content: chroma matched-mean {chroma_matched:.4} \
         does not beat mismatched-mean {chroma_mismatched:.4} by > {CHROMA_MARGIN} (margin \
         {chroma_margin:+.4}) — the cover is generic, not source-preserving"
    );

    println!(
        "acestep_cover_wav_conformance (sft DiT): dur {dur_cov:.2}s | timbre band-L1 {timbre_div:.4} | \
         per-direction matched A {matched_a:.4} B {matched_b:.4} (floor {CHROMA_PER_DIR_MATCHED_FLOOR}) | \
         per-direction margins A {margin_a:+.4} B {margin_b:+.4} (floor {CHROMA_PER_DIR_FLOOR}) | \
         aggregate matched {chroma_matched:.4} > mismatched {chroma_mismatched:.4} \
         (margin {chroma_margin:+.4}, min-margin {CHROMA_MARGIN}, srcA↔srcB {src_ab_chroma:.4}) | \
         wrote {} + {} + {} + {}",
        src_a_path.display(),
        src_b_path.display(),
        cov_a_path.display(),
        cov_b_path.display(),
    );

    // Determinism: the same cover request re-synthesizes byte-identically (seed law).
    let cover_a2 = match generator
        .generate(&cover_req_a, &mut |_| {})
        .expect("second cover generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    assert_eq!(
        cover_a.samples, cover_a2.samples,
        "seeded cover must be deterministic"
    );
}
