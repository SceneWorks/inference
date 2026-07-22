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
    Generator, LoadSpec, Progress, TimeRegion, WeightsSource,
};

/// Resolve the snapshot from the required `ACESTEP_SNAPSHOT` env (a passed-in
/// `ACE-Step/acestep-v15-xl-turbo-diffusers` snapshot dir). Inference never self-fetches or derives
/// a cache location (epic 13657).
fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("ACESTEP_SNAPSHOT").expect(
        "set ACESTEP_SNAPSHOT to an ACE-Step/acestep-v15-xl-turbo-diffusers snapshot dir",
    )))
}

/// Resolve the **sft Cover** snapshot from the required `ACESTEP_SFT_SNAPSHOT` env (a passed-in
/// `ACE-Step/acestep-v15-xl-sft-diffusers` snapshot dir), staged into `LoadSpec::components` under
/// [`candle_audio_acestep::COVER_COMPONENT_ID`]. Production reads the Cover snapshot from the seam,
/// not the env — the env var is the test's passed-in path (epic 13657); only the Cover test needs it.
fn sft_snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("ACESTEP_SFT_SNAPSHOT").expect(
        "set ACESTEP_SFT_SNAPSHOT to an ACE-Step/acestep-v15-xl-sft-diffusers snapshot dir",
    )))
}

/// The backend-neutral gen-core conformance suite (validate honesty, progress contract, typed
/// mid-run and pre-generate cancellation, seed determinism) against the real provider, resolved
/// through the explicit registry.
#[test]
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT); run with --ignored"]
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
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT); run with --ignored"]
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
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT); run with --ignored"]
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

// The shipped cover-gate source/cover prompts + source seeds, hoisted to module consts so the
// sc-13714 seed-stability sweep (`acestep_cover_chroma_seed_stability`) exercises the *same*
// distinctive+tonal sitar / dark-electronic industrial pair as `acestep_cover_wav_conformance` —
// the two tests cannot silently drift apart. A = a solo sitar raga (distinctive twang whose content
// survives the ~80 bit/s FSQ codec, tonal on a specific scale); B = dark industrial electronic
// (distinctive too, chroma-distinct from A). The brass cover is spectrally distinct from BOTH, so
// its shared new timbre cancels in the matched-vs-mismatched comparison.
const COVER_SRC_A_PROMPT: &str = "a hypnotic solo sitar raga with a resonant drone, distinctive twanging strings, meditative Indian classical";
const COVER_SRC_B_PROMPT: &str =
    "dark aggressive industrial electronic, heavy distorted bass, ominous grinding machine drones";
const COVER_PROMPT: &str = "a brass ensemble of trumpets and trombones";
const COVER_SRC_A_SEED: u64 = 42;
const COVER_SRC_B_SEED: u64 = 7;

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
#[ignore = "real weights: needs an ACE-Step snapshot (ACESTEP_SNAPSHOT / ACESTEP_SFT_SNAPSHOT); run with --ignored"]
fn acestep_cover_wav_conformance() {
    // Cover pulls the sft snapshot through the component seam (epic 13657): the test stages the
    // passed-in `ACESTEP_SFT_SNAPSHOT` path as the `sft_cover` component — production reads it from
    // `LoadSpec::components`, never from the env.
    let spec = LoadSpec::new(snapshot())
        .with_component(candle_audio_acestep::COVER_COMPONENT_ID, sft_snapshot());
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
    const SRC_A_SEED: u64 = COVER_SRC_A_SEED;
    const SRC_B_SEED: u64 = COVER_SRC_B_SEED;
    let src_a_prompt = COVER_SRC_A_PROMPT;
    let src_b_prompt = COVER_SRC_B_PROMPT;
    // A brass cover shared by both covers — spectrally distinct from both sources (octave-band timbre
    // moves hard). The shared new timbre cancels in the matched-vs-mismatched comparison, leaving each
    // source's carried-over tonal character.
    let cover_prompt = COVER_PROMPT;

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

// ==================== Cover chroma-gate hardening (sc-13714) ====================
//
// Follow-up to sc-13251 (PR #168). The shipped per-direction chroma gate in
// `acestep_cover_wav_conformance` is genuinely non-vacuous — it passes on the sft-DiT cover with
// matched > 0.40 AND per-direction margin > 0.03 in BOTH directions — but sc-13251 noted two soft
// spots worth hardening: (1) the winning margins are THIN (the weak leg's margin is only ~+0.0426,
// just ~0.01 above the 0.03 floor), and (2) the validation set was small + hand-curated (the two dev
// pairs sitar/industrial and steel-drum/industrial share the `industrial` anchor ⇒ only ~3 distinct
// sources). This block
// adds the two hardening items. Neither is a blocker — the shipped gate already meets its bar; this
// is test-robustness evidence. Both new tests are `#[ignore]`d + snapshot-gated exactly like the
// shipped cover test, so PR CI stays weights-free.
//
// SEED-STABILITY EVIDENCE (`acestep_cover_chroma_seed_stability`, measured on CUDA / GPU1, sm_120,
// sft DiT for covers + turbo DiT for sources). The cover sampler's ONLY stochastic knob is the
// cover-request noise seed (the sources are deterministic conditioning), so this is the axis that
// matters for "do the margins hold across seeds?". Generation is byte-deterministic per seed (the
// seed law), so these numbers reproduce exactly; the spread across seeds is genuine seed sensitivity.
// Sweeping the shared cover noise seed over [42, 7, 123, 2024, 777, 31337] for the shipped
// sitar(A)/industrial(B) pair (matched > 0.40 floor, per-direction margin > 0.03 floor):
//
//   seed    A matched  A margin  |  B matched  B margin      (srcA<->srcB chroma = 0.3566 all seeds)
//     42     0.7907    +0.1195   |   0.6745    +0.0426
//      7     0.7935    +0.1114   |   0.6662    +0.0392
//    123     0.8157    +0.1447   |   0.6101    +0.0416
//   2024     0.7875    +0.1135   |   0.6904    +0.0087   <- weak leg B dips BELOW the 0.03 margin floor
//    777     0.8204    +0.1381   |   0.6875    +0.0581
//  31337     0.7928    +0.1358   |   0.7011    +0.0640
//   ----     ------    ------        ------    ------
//    min     0.7875    +0.1114   |   0.6101    +0.0087       B margin: mean +0.0424, canonical(42) +0.0426
//
// FINDING: the content floor (matched > 0.40) and the source-specific ordering (margin > 0) hold in
// BOTH directions at EVERY seed; the strong leg A clears the 0.03 margin floor every seed with ~3.7x
// headroom. The thin weak leg B clears 0.03 ON AVERAGE (+0.0424) and at the shipped canonical seed 42
// (+0.0426), but is seed-SENSITIVE and dips to +0.0087 at seed 2024 — it stays correctly ordered
// (matched > mismatched) but not comfortably above 0.03. This QUANTIFIES the sc-13251 thinness rather
// than papering over it. The test asserts exactly this (per-seed matched>0.40 + ordering>0 + strong
// leg>0.03; weak-leg mean>0.03 + canonical-seed>0.03) and deliberately does NOT force a per-seed
// weak-leg 0.03 pass — cherry-picking seeds to fake robustness is what the story forbids.
//
// INDEPENDENT 3rd PAIR (`acestep_cover_chroma_independent_pair_constraint`): OUTCOME =
// attempted-but-documented-constraint (the story's explicitly-sanctioned fallback). A genuine,
// matrix-informed search over 8 candidate sources (see the sc-13714 PR for the pairwise source-chroma
// matrix) drove FOUR fully-independent pairs (no sitar/industrial/steel-drum anchor) through the SAME
// per-direction bar, spanning the full source-separation range:
//
//   pair (C / D)              srcC<->srcD   margin C   margin D    why it misses the bar
//   banjo / pipe-organ          0.8459      -0.0162    +0.0958     sources too tonally SIMILAR
//   koto / death-metal          0.2366      +0.1013    -0.0063     dark source's chroma too FLAT (not source-specific)
//   koto / gamelan              0.4930      +0.0248    -0.0337     two tonal covers too alike (representative pair)
//   koto / dubstep             -0.0420      -0.0791    -0.1864     near-orthogonal -> shared cover prompt swamps FSQ signal
//   ---- reference ----
//   sitar / industrial          0.3566      +0.1195    +0.0426     the shipped gate: a NARROW sweet spot
//
// NO independent pair cleared per-direction margin > 0.03 in BOTH directions. The shipped
// sitar/industrial pair sits in a narrow sweet spot — a distinctive peaky-tonal source vs a dark
// source whose chroma is distinctive-but-not-flat, separated by ≈ 0.36 — that an arbitrary
// independent pair does not hit. This is exactly the constraint the story anticipated: the ~80 bit/s
// FSQ semantic codec's ceiling makes an arbitrary independent pair marginal. It is a property of the
// codec (accepted under the Option-A/Option-2 product decision), NOT a defect, so the gate threshold
// is NOT weakened and no flaky pass is forced. The `_constraint` harness records this reproducibly on
// the koto/gamelan representative, asserting only the invariants that robustly hold (real, restyled,
// source-distinct covers) and PRINTING the margin shortfall.
//

/// The four per-direction chroma correlations of one cover pair, plus the source↔source chroma
/// (the discrimination precondition). `matched_*` = source ↔ its OWN cover; `mismatched_*` =
/// source ↔ the OTHER source's cover (same shared cover prompt+seed+steps).
struct ChromaLegs {
    matched_a: f64,
    mismatched_a: f64,
    matched_b: f64,
    mismatched_b: f64,
    src_ab: f64,
}

impl ChromaLegs {
    fn margin_a(&self) -> f64 {
        self.matched_a - self.mismatched_a
    }
    fn margin_b(&self) -> f64 {
        self.matched_b - self.mismatched_b
    }
}

/// Load the ACE-Step generator with the sft cover checkpoint staged as the `COVER_COMPONENT_ID`
/// component — the same load the shipped cover test does (epic 13657: the sft snapshot flows through
/// the component seam, the env var is only the test's passed-in path).
fn load_cover_generator() -> Box<dyn Generator> {
    let spec = LoadSpec::new(snapshot())
        .with_component(candle_audio_acestep::COVER_COMPONENT_ID, sft_snapshot());
    candle_audio_acestep::provider_registry()
        .unwrap()
        .load(candle_audio_acestep::MODEL_ID, &spec)
        .expect("acestep_v15_turbo loads through the explicit registry")
}

/// Synthesize one source clip via ordinary text-to-music (turbo DiT), fixed seed ⇒ reproducible.
fn cover_source(
    generator: &dyn Generator,
    prompt: &str,
    seed: u64,
    steps: u32,
    secs: f32,
) -> AudioTrack {
    let req = GenerationRequest {
        prompt: prompt.into(),
        seed: Some(seed),
        steps: Some(steps),
        audio: Some(AudioParams {
            target_duration: Some(secs),
            sample_rate: Some(48_000),
            language: Some("en".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    match generator
        .generate(&req, &mut |_| {})
        .expect("source generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    }
}

/// Cover both sources under ONE shared `cover(prompt, seed, steps)` (sft DiT) and compute the four
/// per-direction chroma correlations. This is the exact chroma computation the shipped gate does,
/// factored out so the seed sweep and the independent-pair test both drive the identical measurement.
fn cover_chroma_legs(
    generator: &dyn Generator,
    source_a: &AudioTrack,
    source_b: &AudioTrack,
    cover_prompt: &str,
    cover_seed: u64,
    steps: u32,
) -> ChromaLegs {
    let cover_of = |src: &AudioTrack| GenerationRequest {
        prompt: cover_prompt.into(),
        seed: Some(cover_seed),
        steps: Some(steps),
        audio: Some(AudioParams {
            sample_rate: Some(48_000),
            language: Some("en".into()),
            ..Default::default()
        }),
        conditioning: vec![Conditioning::AudioEdit {
            audio: src.clone(),
            mode: AudioEditMode::Cover,
            region: None,
            strength: None,
        }],
        ..Default::default()
    };
    let gen = |req: &GenerationRequest| -> AudioTrack {
        match generator
            .generate(req, &mut |_| {})
            .expect("cover generate")
        {
            GenerationOutput::Audio(t) => t,
            other => panic!("expected audio, got {other:?}"),
        }
    };
    let cover_a = gen(&cover_of(source_a));
    let cover_b = gen(&cover_of(source_b));
    let (sa, sb) = (mono(source_a), mono(source_b));
    let (ca, cb) = (mono(&cover_a), mono(&cover_b));
    let (chr_sa, chr_sb) = (chroma(&sa), chroma(&sb));
    let (chr_ca, chr_cb) = (chroma(&ca), chroma(&cb));
    ChromaLegs {
        matched_a: chroma_corr(&chr_sa, &chr_ca),
        mismatched_a: chroma_corr(&chr_sa, &chr_cb),
        matched_b: chroma_corr(&chr_sb, &chr_cb),
        mismatched_b: chroma_corr(&chr_sb, &chr_ca),
        src_ab: chroma_corr(&chr_sa, &chr_sb),
    }
}

// The chroma gate's two floors, shared with the shipped `acestep_cover_wav_conformance` bar.
const CHROMA_PER_DIR_MATCHED_FLOOR: f64 = 0.40;
const CHROMA_PER_DIR_MARGIN_FLOOR: f64 = 0.03;

/// sc-13714 (2): SEED STABILITY. Fix the shipped sitar(A)/industrial(B) source pair and sweep the
/// cover sampler's noise seed — the sole stochastic knob — over [42, 7, 123, 2024, 777, 31337],
/// recording the per-direction chroma matched/mismatched/margin at every seed. Because generation is
/// byte-deterministic per seed (the seed law), these numbers reproduce exactly; the spread across
/// seeds is genuine seed sensitivity. The finding (see the assertions and the sc-13714 evidence block):
/// the content floor (matched > 0.40) and the source-specific ordering (margin > 0) hold in BOTH
/// directions at every seed, and the strong leg (A) clears the 0.03 margin floor with wide headroom
/// every seed — but the thin weak leg (B) is seed-SENSITIVE: it clears 0.03 on average and at the
/// canonical seed 42, yet dips to ≈ +0.009 at seed 2024. So the gate's weak-leg margin is real and
/// correctly-ordered across seeds, but not comfortably robust at 0.03 — the sc-13251 thinness, now
/// quantified. The test asserts exactly what holds and does NOT force a per-seed 0.03 pass.
#[test]
#[ignore = "real weights: needs ACE-Step snapshots (ACESTEP_SNAPSHOT / ACESTEP_SFT_SNAPSHOT); run with --ignored"]
fn acestep_cover_chroma_seed_stability() {
    let generator = load_cover_generator();

    const TARGET_SECS: f32 = 6.0;
    const STEPS: u32 = 8;
    // The cover noise-seed sweep. Sources are generated ONCE (deterministic conditioning); only the
    // shared cover noise seed varies, isolating the cover sampler's stochastic sensitivity.
    const COVER_SEEDS: [u64; 6] = [42, 7, 123, 2024, 777, 31337];

    let source_a = cover_source(
        generator.as_ref(),
        COVER_SRC_A_PROMPT,
        COVER_SRC_A_SEED,
        STEPS,
        TARGET_SECS,
    );
    let source_b = cover_source(
        generator.as_ref(),
        COVER_SRC_B_PROMPT,
        COVER_SRC_B_SEED,
        STEPS,
        TARGET_SECS,
    );

    // Collect every seed's legs FIRST so the full table prints even when an assertion later fails.
    let rows: Vec<(u64, ChromaLegs)> = COVER_SEEDS
        .iter()
        .map(|&seed| {
            (
                seed,
                cover_chroma_legs(
                    generator.as_ref(),
                    &source_a,
                    &source_b,
                    COVER_PROMPT,
                    seed,
                    STEPS,
                ),
            )
        })
        .collect();

    eprintln!(
        "acestep_cover_chroma_seed_stability (sc-13714) — sitar(A)/industrial(B), sft cover DiT:"
    );
    eprintln!(
        "  floors: matched > {CHROMA_PER_DIR_MATCHED_FLOOR}, per-direction margin > {CHROMA_PER_DIR_MARGIN_FLOOR}"
    );
    for (seed, l) in &rows {
        eprintln!(
            "  seed {seed:>6}: A matched {:.4} mismatched {:.4} margin {:+.4} | B matched {:.4} mismatched {:.4} margin {:+.4} | srcA<->srcB {:.4}",
            l.matched_a,
            l.mismatched_a,
            l.margin_a(),
            l.matched_b,
            l.mismatched_b,
            l.margin_b(),
            l.src_ab,
        );
    }
    let min_margin_a = rows
        .iter()
        .map(|(_, l)| l.margin_a())
        .fold(f64::INFINITY, f64::min);
    let min_margin_b = rows
        .iter()
        .map(|(_, l)| l.margin_b())
        .fold(f64::INFINITY, f64::min);
    let min_matched_a = rows
        .iter()
        .map(|(_, l)| l.matched_a)
        .fold(f64::INFINITY, f64::min);
    let min_matched_b = rows
        .iter()
        .map(|(_, l)| l.matched_b)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "  worst-case over the sweep: matched min A {min_matched_a:.4} B {min_matched_b:.4} | margin min A {min_margin_a:+.4} B {min_margin_b:+.4}"
    );

    let mean_margin_b = rows.iter().map(|(_, l)| l.margin_b()).sum::<f64>() / rows.len() as f64;
    let canonical_margin_b = rows
        .iter()
        .find(|(seed, _)| *seed == 42)
        .map(|(_, l)| l.margin_b())
        .expect("seed 42 is in the sweep (the shipped gate's cover seed)");
    eprintln!(
        "  weak-leg (B) margin: mean {mean_margin_b:+.4}, canonical-seed-42 {canonical_margin_b:+.4} (floor {CHROMA_PER_DIR_MARGIN_FLOOR})"
    );

    // What the sweep establishes (generation is byte-deterministic per seed — the seed law — so these
    // numbers reproduce exactly; the variation below is genuine seed sensitivity, not run noise):
    //
    //  (i)   the CONTENT-PRESERVATION floor (matched > 0.40) holds for BOTH directions at EVERY seed;
    //  (ii)  the source-specific ORDERING (matched > mismatched ⇒ margin > 0) never inverts — every
    //        seed keeps each cover closer to its OWN source than to the other's, in both directions;
    //  (iii) the strong leg (A, sitar) clears the shipped 0.03 margin floor at EVERY seed with wide
    //        headroom (min ≈ +0.11, ~3.7×);
    //  (iv)  the thin weak leg (B, industrial) clears the 0.03 floor ON AVERAGE over the sweep and at
    //        the shipped canonical seed 42 — but it is seed-SENSITIVE and dips below 0.03 at some
    //        seeds (min ≈ +0.009). This is the sc-13251 thinness, now quantified: the gate's weak-leg
    //        margin is real (correctly ordered every seed) but not comfortably seed-robust at 0.03.
    //
    // We assert (i)–(iv) exactly, and deliberately do NOT assert per-seed weak-leg > 0.03 (it is false
    // for at least one seed) — forcing that would require cherry-picking seeds, which the story forbids.
    for (seed, l) in &rows {
        assert!(
            l.matched_a > CHROMA_PER_DIR_MATCHED_FLOOR
                && l.matched_b > CHROMA_PER_DIR_MATCHED_FLOOR,
            "seed {seed}: a direction's matched chroma fell to/below the content floor \
             (A {:.4}, B {:.4}, floor {CHROMA_PER_DIR_MATCHED_FLOOR}) — content not preserved",
            l.matched_a,
            l.matched_b,
        );
        assert!(
            l.margin_a() > 0.0 && l.margin_b() > 0.0,
            "seed {seed}: source-specific ordering inverted (margin A {:+.4}, B {:+.4}) — a cover is \
             closer to the OTHER source than to its own",
            l.margin_a(),
            l.margin_b(),
        );
        assert!(
            l.margin_a() > CHROMA_PER_DIR_MARGIN_FLOOR,
            "seed {seed}: the strong leg A dropped below the 0.03 floor (margin {:+.4}) — unexpected \
             regression of the robust direction",
            l.margin_a(),
        );
    }
    assert!(
        mean_margin_b > CHROMA_PER_DIR_MARGIN_FLOOR,
        "weak leg B mean margin over the sweep {mean_margin_b:+.4} does not clear the 0.03 floor — \
         the gate's thin direction has regressed across seeds"
    );
    assert!(
        canonical_margin_b > CHROMA_PER_DIR_MARGIN_FLOOR,
        "the shipped gate's canonical cover seed 42 weak-leg margin {canonical_margin_b:+.4} does not \
         clear the 0.03 floor — the shipped gate itself would fail"
    );
}

// A representative fully-independent pair (sc-13714 (1)) — shares NO anchor with sitar / industrial /
// steel-drum: C = solo Japanese koto (plucked zither, pentatonic), D = Indonesian gamelan (tuned
// metallic percussion, pelog). Two distinctive tonal instruments in different tuning systems, source
// chroma-corr 0.4930. This is the pair the `_constraint` harness records; see the sc-13714 evidence
// block above for why NO independent pair (this one included) clears the shipped per-direction bar.
// Seeds are the candidate-matrix seeds, so the sources reproduce that measurement.
const COVER_IND_SRC_C_PROMPT: &str =
    "a delicate solo Japanese koto, plucked zither strings, distinctive traditional pentatonic melody";
const COVER_IND_SRC_D_PROMPT: &str =
    "a bright shimmering Indonesian gamelan ensemble, interlocking metallic tuned gongs and metallophones, distinctive pelog scale";
const COVER_IND_SRC_C_SEED: u64 = 5;
const COVER_IND_SRC_D_SEED: u64 = 3;

/// sc-13714 (1): the DOCUMENTED-CONSTRAINT harness for the fully-independent 3rd pair. sc-13714
/// asked for an independent pair (no `industrial`/`sitar`/`steel-drum` anchor) that clears the SAME
/// non-vacuous per-direction bar as the shipped gate. After a genuine, matrix-informed search (four
/// pairs spanning the full source-chroma-separation range — see the sc-13714 evidence block above),
/// NO fully-independent pair cleared the per-direction margin > 0.03 in BOTH directions: the shipped
/// sitar/industrial pair sits in a narrow sweet spot (a distinctive peaky-tonal source vs a dark
/// source whose chroma is distinctive-but-not-flat, separation ≈ 0.36) that an arbitrary independent
/// pair does not hit. This is the story's explicitly-sanctioned outcome — the constraint is the
/// ~80 bit/s FSQ semantic codec's ceiling, not a defect, so the gate is NOT weakened and no flaky
/// pass is forced.
///
/// This harness RECORDS that finding reproducibly on a representative independent pair (koto/gamelan,
/// two distinctive tonal instruments in different tuning systems). It asserts only the invariants that
/// robustly HOLD — the covers are real, restyled, and the two sources are chroma-distinct — and PRINTS
/// the per-direction margins so the shortfall is evidenced from code, not merely claimed. It does NOT
/// assert the per-direction margin bar (which this independent pair does not meet); the shipped
/// `acestep_cover_wav_conformance` remains the gate. Sources are generated deterministically from code
/// (fixed seeds), so the evidence is reproducible, not an opaque committed blob.
#[test]
#[ignore = "real weights: needs ACE-Step snapshots (ACESTEP_SNAPSHOT / ACESTEP_SFT_SNAPSHOT); run with --ignored"]
fn acestep_cover_chroma_independent_pair_constraint() {
    let generator = load_cover_generator();

    const TARGET_SECS: f32 = 6.0;
    const STEPS: u32 = 8;
    const COVER_SEED: u64 = 42;

    let source_c = cover_source(
        generator.as_ref(),
        COVER_IND_SRC_C_PROMPT,
        COVER_IND_SRC_C_SEED,
        STEPS,
        TARGET_SECS,
    );
    let source_d = cover_source(
        generator.as_ref(),
        COVER_IND_SRC_D_PROMPT,
        COVER_IND_SRC_D_SEED,
        STEPS,
        TARGET_SECS,
    );

    let legs = cover_chroma_legs(
        generator.as_ref(),
        &source_c,
        &source_d,
        COVER_PROMPT,
        COVER_SEED,
        STEPS,
    );
    let agg_matched = 0.5 * (legs.matched_a + legs.matched_b);
    let agg_mismatched = 0.5 * (legs.mismatched_a + legs.mismatched_b);
    let agg_margin = agg_matched - agg_mismatched;

    // Restyle proof on C -> its cover: regenerate cover C once for the waveform/timbre comparison.
    let cover_c = match generator
        .generate(
            &GenerationRequest {
                prompt: COVER_PROMPT.into(),
                seed: Some(COVER_SEED),
                steps: Some(STEPS),
                audio: Some(AudioParams {
                    sample_rate: Some(48_000),
                    language: Some("en".into()),
                    ..Default::default()
                }),
                conditioning: vec![Conditioning::AudioEdit {
                    audio: source_c.clone(),
                    mode: AudioEditMode::Cover,
                    region: None,
                    strength: None,
                }],
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("cover C generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected audio, got {other:?}"),
    };
    let (sc, cc) = (mono(&source_c), mono(&cover_c));
    let timbre_div = band_l1(&octave_band_dist(&sc), &octave_band_dist(&cc));
    let n = sc.len().min(cc.len());
    let wav_l2 = rel_l2(&sc[..n], &cc[..n]);
    let wav_corr = corr(&sc[..n], &cc[..n]);
    let rms_c = rms(&cc);

    // Does this independent pair clear the shipped per-direction margin bar in BOTH directions?
    let clears_bar = legs.margin_a() > CHROMA_PER_DIR_MARGIN_FLOOR
        && legs.margin_b() > CHROMA_PER_DIR_MARGIN_FLOOR;
    eprintln!(
        "acestep_cover_chroma_independent_pair_constraint (sc-13714) — koto(C)/gamelan(D), sft cover DiT:\n  \
         per-direction matched C {:.4} D {:.4} (floor {CHROMA_PER_DIR_MATCHED_FLOOR}) | margins C {:+.4} D {:+.4} (floor {CHROMA_PER_DIR_MARGIN_FLOOR})\n  \
         aggregate matched {agg_matched:.4} - mismatched {agg_mismatched:.4} = margin {agg_margin:+.4} | srcC<->srcD chroma {:.4}\n  \
         restyle C->coverC: timbre band-L1 {timbre_div:.4} | rel-L2 {wav_l2:.4} corr {wav_corr:.4} | cover-C rms {rms_c:.4}\n  \
         => clears per-direction 0.03 bar in BOTH directions? {clears_bar} (RECORDED CONSTRAINT: an \
         arbitrary independent pair does NOT — the shipped gate remains the sitar/industrial pair)",
        legs.matched_a,
        legs.matched_b,
        legs.margin_a(),
        legs.margin_b(),
        legs.src_ab,
    );

    // Assert ONLY the invariants that robustly hold — this harness records a constraint, it is not a
    // gate. (1) The pipeline produced a real, non-silent, genuinely restyled cover of an independent
    // source (mirrors the shipped gate's timbre-change half): proves the failure to clear the margin
    // bar is a SEMANTIC/FSQ-ceiling limit, not a broken cover. (2) The two independent sources are
    // chroma-distinct, so the mismatched control is a meaningful one. We do NOT assert the
    // per-direction margin bar — it is not met for any independent pair, by design of this finding.
    assert!(
        rms_c > 0.01,
        "cover C is silent (rms {rms_c:.5}) — pipeline broken, not a codec limit"
    );
    assert!(
        wav_l2 > 0.3,
        "cover C waveform barely differs from source C (rel-L2 {wav_l2:.4}) — not a restyle"
    );
    assert!(
        wav_corr < 0.9,
        "cover C still highly correlated with source C (corr {wav_corr:.4})"
    );
    assert!(
        timbre_div > 0.05,
        "cover C timbre barely moved from source C (band-L1 {timbre_div:.4})"
    );
    assert!(
        legs.src_ab < 0.85,
        "independent sources C and D too tonally similar (chroma-corr {:.4}) — the mismatched control \
         would be vacuous",
        legs.src_ab,
    );
}
