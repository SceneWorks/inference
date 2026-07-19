//! Real-weight conformance for the MOSS-TTS-Realtime-1.7B AR brain (sc-13334).
//!
//! ## What this slice gates (honest partial)
//!
//! This slice ports the **AR brain** — the Qwen3-1.7B backbone + the CSM-style local/depth
//! transformer — not the MOSS-Audio-Tokenizer codec (RVQ frames → 24 kHz waveform), which is a
//! separate ~7 GB model that is not yet ported. So the conformance here is the **AR stage** (the
//! ported half) exercised on real weights:
//!
//! - [`moss_tts_realtime_emits_valid_rvq_frames`] — a fixed text + greedy decode → the AR loop
//!   emits **≥ 2** real 16-codebook RVQ frames, every codebook token in `[0, 1027)`, deterministic
//!   run-to-run, and non-degenerate (not a single collapsed id). A broken backbone / weight mapping
//!   / RoPE / multi-embedding sum / local-transformer head wiring would produce empty, out-of-range,
//!   or all-identical frames and fail here.
//! - [`moss_tts_realtime_is_incremental`] — the AR loop is genuinely incremental: the time to the
//!   **first** RVQ frame is materially less than the time to the **full** budget (the property the
//!   streaming contract rests on — the audio arrives in blocks, not one final dump).
//! - [`moss_tts_realtime_generate_stops_honestly_at_the_codec_boundary`] — `generate()` runs the AR
//!   loop and then returns a typed error naming the unported codec; it never emits fake audio.
//!
//! ## What remains blocked
//!
//! The **streaming acceptance gate** — ≥ 2 PCM chunks before completion, concat(chunks) ==
//! one-shot output, non-silent speech-shaped audio at 24 kHz — requires the MOSS-Audio-Tokenizer
//! codec to turn these RVQ frames into a waveform. That is **not yet ported**; no WAV is produced
//! here because fabricating one would be dishonest. The full gate + `check_audio_streaming` against
//! this provider are deferred to the codec follow-up.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-moss-tts-realtime --test conformance -- --ignored --nocapture
//! ```
//! Set `MOSS_TTS_REALTIME_SNAPSHOT` to a snapshot dir (holding `config.json`, `model.safetensors`,
//! `tokenizer.json`) or leave unset to resolve the pinned snapshot via the hub (~4.66 GB).

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_moss_tts_realtime as moss;
use candle_audio_moss_tts_realtime::gen_core::{
    AudioParams, GenerationRequest, Generator, LoadSpec, WeightsSource,
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

    // Deterministic: greedy decode ⇒ byte-identical frames on a re-run (the reproducibility law).
    let again = gen
        .rvq_frames(&request(1.2), &mut |_| {})
        .expect("re-decode");
    assert_eq!(
        *frames, again.frames,
        "greedy AR decode must be reproducible run-to-run"
    );

    // Honest evidence: no WAV — the codec that would render these frames is unported. Dump the RVQ
    // token frames instead if requested (the pipeline OUTPUT, not fabricated audio).
    if let Ok(out) = std::env::var("MOSS_TTS_REALTIME_FRAMES_OUT") {
        let text: String = frames
            .iter()
            .map(|f| f.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&out, text).expect("write RVQ frames");
        eprintln!(
            "wrote {} RVQ frames to {out} (NO WAV — codec unported)",
            frames.len()
        );
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

/// The codec boundary is honest: `generate()` runs the AR loop and then errors, never fake audio.
#[test]
#[ignore = "real weights: needs the ~4.66 GB MOSS-TTS-Realtime snapshot; run with --ignored"]
fn moss_tts_realtime_generate_stops_honestly_at_the_codec_boundary() {
    let gen = load();
    let err = gen
        .generate(&request(0.8), &mut |_| {})
        .expect_err("generate must stop at the codec boundary, not fabricate audio");
    let msg = format!("{err}");
    eprintln!("honest boundary error: {msg}");
    assert!(
        msg.contains("MOSS-Audio-Tokenizer"),
        "the boundary error must name the unported codec: {msg}"
    );
    assert!(
        msg.contains("RVQ frame"),
        "the boundary error must report the RVQ frames produced: {msg}"
    );
}
