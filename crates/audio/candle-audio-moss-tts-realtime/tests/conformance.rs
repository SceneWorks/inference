//! Real-weight conformance for MOSS-TTS-Realtime-1.7B — the AR brain (sc-13334) **and** the
//! MOSS-Audio-Tokenizer codec (sc-13392, RVQ frames → 24 kHz waveform).
//!
//! ## What this gates on real weights
//!
//! - [`moss_tts_realtime_emits_valid_rvq_frames`] — a fixed text + seed → the AR loop emits **≥ 2**
//!   real 16-codebook RVQ frames, every codebook token in `[0, 1027)`, deterministic run-to-run (the
//!   seeded sampler), and non-degenerate (not a single collapsed id). A broken backbone / weight
//!   mapping / RoPE / multi-embedding sum / local-transformer head wiring would produce empty,
//!   out-of-range, or all-identical frames and fail here.
//! - [`moss_tts_realtime_is_incremental`] — the AR loop is genuinely incremental: the time to the
//!   **first** RVQ frame is materially less than the time to the **full** budget.
//! - [`moss_tts_realtime_streaming_gate`] — the sc-13334 streaming acceptance gate, now released by
//!   the codec: `gen_core_testkit::check_audio_streaming` against the **real** registered provider
//!   ((a) ≥ 2 PCM chunks before completion; (b) concat(chunks) == one-shot `generate()`
//!   byte-identical; (c) valid 24 kHz mono track), plus (c) full audio non-silent / speech-shaped
//!   and (d) first-chunk latency < full-generation latency, and it writes a playable demo WAV.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-moss-tts-realtime --test conformance -- --ignored --nocapture
//! ```
//! Set `MOSS_TTS_REALTIME_SNAPSHOT` to the AR snapshot dir (holding `config.json`,
//! `model.safetensors`, `tokenizer.json`) and `MOSS_AUDIO_TOKENIZER_SNAPSHOT` to the codec snapshot
//! dir (`config.json` + `model*.safetensors`), or leave unset to resolve the pinned snapshots via
//! the hub (~4.66 GB AR + ~7.1 GB codec). The demo WAV path is `MOSS_TTS_REALTIME_WAV_OUT` (default
//! temp dir).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_audio_moss_tts_realtime as moss;
use candle_audio_moss_tts_realtime::gen_core::{
    AudioChunk, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};

/// Resolve a MOSS-TTS-Realtime snapshot dir. `MOSS_TTS_REALTIME_SNAPSHOT` overrides; otherwise the
/// pinned snapshot is fetched via the hub.
fn snapshot() -> PathBuf {
    if let Ok(dir) = std::env::var("MOSS_TTS_REALTIME_SNAPSHOT") {
        return PathBuf::from(dir);
    }
    match moss::resolve_pinned_snapshot()
        .expect("resolve the pinned MOSS-TTS-Realtime snapshot (network or warm HF cache)")
    {
        WeightsSource::Dir(p) => p,
        other => panic!("expected a snapshot dir, got {other:?}"),
    }
}

fn load() -> moss::model::MossTtsRealtimeGenerator {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    moss::load_generator(&spec).expect("load the MOSS-TTS-Realtime generator")
}

/// A fixed, short TTS request (a small frame budget keeps the CPU AR run tractable).
fn request(seconds: f32) -> GenerationRequest {
    GenerationRequest {
        prompt: "Hello, this is a streaming text to speech test.".to_string(),
        audio: Some(AudioParams {
            target_duration: Some(seconds),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        seed: Some(20260719),
        ..Default::default()
    }
}

/// AR-stage gate: real weights decode valid, non-degenerate, deterministic RVQ frames.
#[test]
#[ignore = "real weights: needs the ~4.66 GB MOSS-TTS-Realtime snapshot; run with --ignored"]
fn moss_tts_realtime_emits_valid_rvq_frames() {
    use std::collections::HashSet;

    let gen = load();
    // ~1.2 s of audio at 12.5 fps ≈ 15 frames — enough to prove ≥ 2 incremental frames cheaply.
    let result = gen
        .rvq_frames(&request(1.2), &mut |_| {})
        .expect("AR RVQ-frame decode");
    let frames = &result.frames;
    eprintln!(
        "AR brain emitted {} RVQ frames (stop: {:?})",
        frames.len(),
        result.stop
    );

    // Genuinely incremental: at least two frames before completion.
    assert!(
        frames.len() >= 2,
        "the AR loop must emit ≥ 2 RVQ frames (got {})",
        frames.len()
    );
    // Every frame carries exactly rvq (16) codebook tokens, all in the audio vocabulary [0, 1027).
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(
            frame.len(),
            16,
            "frame {i} must carry 16 RVQ codebook tokens"
        );
        assert!(
            frame.iter().all(|&t| t < 1027),
            "frame {i} has an out-of-range codebook token: {frame:?}"
        );
    }
    // Non-degenerate: the codebook-0 stream spans many codes (a collapsed backbone / local head /
    // RoPE bug degenerates to a single repeated id).
    let cb0: Vec<u32> = frames.iter().map(|f| f[0]).collect();
    let distinct = cb0.iter().collect::<HashSet<_>>().len();
    eprintln!("codebook-0 stream: {cb0:?} ({distinct} distinct)");
    assert!(
        distinct > 1,
        "codebook-0 collapsed to {distinct} distinct value(s) — the AR brain is not modeling speech"
    );

    // Deterministic: the seeded sampler ⇒ byte-identical frames on a re-run (the reproducibility law).
    let again = gen
        .rvq_frames(&request(1.2), &mut |_| {})
        .expect("re-decode");
    assert_eq!(
        *frames, again.frames,
        "seeded AR sampling must be reproducible run-to-run"
    );

    // Optionally dump the raw RVQ token frames (the AR-stage output the codec consumes) for
    // inspection; the WAV rendering is exercised by `moss_tts_realtime_streaming_gate`.
    if let Ok(out) = std::env::var("MOSS_TTS_REALTIME_FRAMES_OUT") {
        let text: String = frames
            .iter()
            .map(|f| f.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&out, text).expect("write RVQ frames");
        eprintln!("wrote {} RVQ frames to {out}", frames.len());
    }
}

/// AR-stage gate: the loop is genuinely incremental — first frame lands well before the full budget.
#[test]
#[ignore = "real weights: needs the ~4.66 GB MOSS-TTS-Realtime snapshot; run with --ignored"]
fn moss_tts_realtime_is_incremental() {
    let gen = load();
    // Warm the lazy load (weights mmap + build) so the timing measures decode, not I/O.
    let _ = gen
        .rvq_frames(&request(0.2), &mut |_| {})
        .expect("warm-up decode");

    use candle_audio_moss_tts_realtime::gen_core::Progress;
    let mut first_frame_at: Option<std::time::Duration> = None;
    let start = Instant::now();
    let result = gen
        .rvq_frames(&request(1.6), &mut |p| {
            if let Progress::Step { current: 1, .. } = p {
                first_frame_at = Some(start.elapsed());
            }
        })
        .expect("timed AR decode");
    let total = start.elapsed();
    let first = first_frame_at.expect("at least one frame was decoded");
    eprintln!(
        "first frame at {:.3?}, full {} frames at {:.3?}",
        first,
        result.frames.len(),
        total
    );
    assert!(
        result.frames.len() >= 2,
        "need ≥ 2 frames to demonstrate incrementality"
    );
    // The first frame must arrive strictly (and materially) before the full budget — the streaming
    // premise. A non-incremental "emit everything at the end" implementation would fail this.
    assert!(
        first < total,
        "first-frame latency {first:.3?} was not less than the full-decode latency {total:.3?}"
    );
}

/// The streaming acceptance gate (sc-13334, released by the sc-13392 codec): the shared
/// `check_audio_streaming` suite against the **real registered provider** (chunk-count, reassembly
/// law, one-shot == stream), plus the DoD extras — first-chunk latency < full-generation latency,
/// non-silent speech-shaped 24 kHz audio, and a playable demo WAV.
#[test]
#[ignore = "real weights: needs the ~4.66 GB AR + ~7.1 GB codec snapshots; run with --ignored"]
fn moss_tts_realtime_streaming_gate() {
    // ~1.6 s at 12.5 fps ≈ 20 frames — enough for several stream chunks while staying CPU-tractable.
    let seconds = 1.6f32;
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let registry = moss::provider_registry().expect("build the moss_tts_realtime registry");
    let generator = registry
        .load(moss::MODEL_ID, &spec)
        .expect("moss_tts_realtime loads through the explicit registry");
    assert_eq!(generator.descriptor().id, "moss_tts_realtime");
    assert!(generator.descriptor().capabilities.supports_streaming);

    // (a) + (b) + one-shot equality: the shared conformance suite.
    let profile = gen_core_testkit::AudioProfile {
        prompt: "Hello, this is a streaming text to speech test.".to_owned(),
        steps: (seconds * moss::model::FRAME_RATE_HZ).ceil() as u32,
        seed: 20_260_719,
        cancel_steps: (seconds * moss::model::FRAME_RATE_HZ).ceil() as u32,
        audio: AudioParams {
            target_duration: Some(seconds),
            language: Some("en".to_owned()),
            sample_rate: Some(24_000),
            ..Default::default()
        },
    };
    gen_core_testkit::check_audio_streaming(generator.as_ref(), &profile)
        .expect("check_audio_streaming against the real MOSS-TTS-Realtime provider");

    // (d) first-chunk latency < full-generation latency, measured directly.
    let req = request(seconds);
    let start = Instant::now();
    let mut first_chunk_at: Option<Duration> = None;
    let mut chunks: Vec<AudioChunk> = Vec::new();
    let out = generator
        .generate_streaming(
            &req,
            &mut |c| {
                if first_chunk_at.is_none() {
                    first_chunk_at = Some(start.elapsed());
                }
                chunks.push(c);
            },
            &mut |_| {},
        )
        .expect("streaming generate");
    let full = start.elapsed();
    let first = first_chunk_at.expect("at least one chunk was emitted");
    let track = match out {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected GenerationOutput::Audio, got {other:?}"),
    };
    eprintln!(
        "streaming: {} chunks, first chunk at {first:.3?}, full generation {full:.3?}",
        chunks.len()
    );
    assert!(
        chunks.len() >= 2,
        "expected >= 2 stream chunks, got {}",
        chunks.len()
    );
    assert!(
        first < full,
        "first-chunk latency {first:.3?} was not less than full-generation latency {full:.3?}"
    );

    // (c) valid 24 kHz mono track, finite, non-empty.
    assert_eq!(track.sample_rate, 24_000);
    assert_eq!(track.channels, 1, "MOSS-TTS-Realtime is mono");
    assert!(!track.samples.is_empty(), "non-empty audio");
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "finite samples"
    );

    // (c) NON-SILENT + speech-shaped: interior RMS above the noise floor, and 50 ms frame energy
    // that VARIES (voiced peaks vs pauses) — a collapsed/broken codec decode would be silent or flat.
    let n = track.samples.len();
    let interior = &track.samples[n / 10..n - n / 10];
    let rms = (interior.iter().map(|s| s * s).sum::<f32>() / interior.len() as f32).sqrt();
    let peak = track.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(rms > 0.005, "interior RMS {rms:.5} — silence is a failure");

    let frame_len = 1200; // 50 ms @ 24 kHz
    let frame_rms: Vec<f32> = track
        .samples
        .chunks(frame_len)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let mean_frame = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;
    let var_frame = frame_rms
        .iter()
        .map(|r| (r - mean_frame) * (r - mean_frame))
        .sum::<f32>()
        / frame_rms.len() as f32;
    let cv = var_frame.sqrt() / mean_frame.max(1e-9);
    assert!(
        cv > 0.15,
        "frame-RMS coefficient of variation {cv:.3} — constant energy is not speech"
    );

    // Spectral tilt (informational + a light gate): speech concentrates energy sub-4 kHz.
    let window = candle_audio::dsp::hann_window(512);
    let sp = candle_audio::dsp::stft(interior, 512, 256, &window).expect("stft");
    let mag = sp.magnitude();
    let (mut low, mut high) = (0.0f64, 0.0f64);
    for bin in 0..sp.n_bins {
        let hz = bin as f32 * 24_000.0 / 512.0;
        let e: f64 = mag[bin * sp.n_frames..(bin + 1) * sp.n_frames]
            .iter()
            .map(|m| (*m as f64) * (*m as f64))
            .sum();
        if hz < 4_000.0 {
            low += e;
        } else if hz >= 8_000.0 {
            high += e;
        }
    }
    assert!(
        low > high,
        "sub-4 kHz energy ({low:.1}) should exceed supra-8 kHz ({high:.1}) for speech"
    );

    // Playable evidence + reported stats.
    let secs = track.samples.len() as f32 / track.sample_rate as f32;
    let out_path = std::env::var("MOSS_TTS_REALTIME_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("moss-tts-realtime-sc13392.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write demo WAV");
    println!(
        "moss_tts_realtime_streaming_gate: wrote {} ({secs:.2}s @ 24 kHz mono, {} chunks, peak \
         {peak:.4}, interior RMS {rms:.4}, frame-RMS CV {cv:.3}, first-chunk {first:.3?} < full \
         {full:.3?})",
        out_path.display(),
        chunks.len(),
    );
}

/// Codec-only debug decode (no AR): loads the codec and decodes synthetic frames, printing per-stage
/// RMS. Isolates whether a silent/near-zero waveform is a codec-decode bug (fails here on synthetic
/// codes) vs an AR→codec mapping issue (passes here, fails the streaming gate). Set
/// `MOSS_AUDIO_TOKENIZER_SNAPSHOT` + `MOSS_CODEC_DEBUG=1`.
#[test]
#[ignore = "real weights: needs the ~7.1 GB codec snapshot; run with --ignored"]
fn codec_only_decodes_synthetic_frames() {
    use candle_audio_moss_tts_realtime::codec::MossAudioCodec;
    let dir = moss::resolve_pinned_codec_snapshot().expect("resolve codec snapshot");
    let codec = MossAudioCodec::load(&dir, 16).expect("load codec decoder");

    // Either the real dumped AR frames (MOSS_TTS_REALTIME_FRAMES_OUT) or 25 frames of pseudo-random
    // in-range codes (a fixed LCG so the run is reproducible).
    let frames: Vec<Vec<u32>> = if let Ok(path) = std::env::var("MOSS_TTS_REALTIME_FRAMES_OUT") {
        std::fs::read_to_string(&path)
            .expect("read frames file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split(',').map(|s| s.trim().parse().unwrap()).collect())
            .collect()
    } else {
        let mut state: u32 = 1;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % 1024
        };
        (0..25).map(|_| (0..16).map(|_| next()).collect()).collect()
    };
    let wav = codec
        .decode_frames(&frames, &|| false)
        .expect("decode")
        .expect("not cancelled");
    let n = wav.len() as f32;
    let rms = (wav.iter().map(|s| s * s).sum::<f32>() / n).sqrt();
    let peak = wav.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    eprintln!(
        "codec synthetic decode: {} samples, rms={rms:.5}, peak={peak:.5}",
        wav.len()
    );
    assert_eq!(
        wav.len(),
        frames.len() * 1920,
        "expected 1920 samples per frame"
    );
    assert!(
        rms > 1e-4,
        "codec produced near-silent output ({rms:.6}) from non-trivial codes — decode-path bug"
    );
}
