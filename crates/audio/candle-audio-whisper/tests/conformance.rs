//! Real-weight conformance for the candle Whisper transcriber (sc-12850) — the epic's DoD gate:
//! real pinned weights → registry load-by-id → transcribe → a real transcript.
//!
//! The DoD test is a **Kokoro TTS → Whisper ASR round-trip**: the merged Kokoro provider
//! (`kokoro_82m`, sc-12836) synthesizes a WAV from KNOWN text, Whisper transcribes it, and the
//! transcript is asserted to match the known text within a small character-error-rate (CER)
//! threshold. This catches exactly the failures that matter — an empty/garbage transcript, or a
//! transcriber that ignores the audio and emits boilerplate, both blow past the CER bound. It also
//! asserts the emitted segment timestamps are monotonic.
//!
//! Both tests are `#[ignore]`d and snapshot-gated like every other family's real-weight tests:
//! - `WHISPER_SNAPSHOT` → an `openai/whisper-base` snapshot dir (config.json + tokenizer.json +
//!   model.safetensors), or unset to resolve the pinned snapshot through the audio lane's F-029 hub
//!   path (downloads ~150 MB into the ordinary HF cache on first run);
//! - `KOKORO_SNAPSHOT` → a `hexgrad/Kokoro-82M` snapshot dir, or unset to resolve the pinned one.
//!
//! ```text
//! cargo test --locked -p candle-audio-whisper --test conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use candle_audio_whisper::gen_core::{
    LoadSpec, Progress, TimestampGranularity, TranscribeOptions, TranscribeRequest, WeightsSource,
};

/// The known script the round-trip drives through Kokoro → Whisper.
const KNOWN_TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// Resolve the Whisper snapshot from the required `WHISPER_SNAPSHOT` env (a passed-in
/// `openai/whisper-base` snapshot dir). Inference never self-fetches or derives a cache location
/// (epic 13657).
fn whisper_snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("WHISPER_SNAPSHOT").expect(
        "set WHISPER_SNAPSHOT to an openai/whisper-base snapshot dir (config.json + tokenizer.json + model.safetensors)",
    )))
}

/// Resolve the Kokoro snapshot from the required `KOKORO_SNAPSHOT` env (a passed-in
/// `hexgrad/Kokoro-82M` snapshot dir).
fn kokoro_snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var("KOKORO_SNAPSHOT")
            .expect("set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir"),
    ))
}

/// Synthesize `text` to an AudioTrack with the merged Kokoro provider (`kokoro_82m`).
fn synthesize_with_kokoro(text: &str) -> candle_audio_whisper::gen_core::AudioTrack {
    use candle_audio_kokoro::gen_core::{
        AudioParams, GenerationOutput, GenerationRequest, LoadSpec as KLoadSpec,
    };
    let spec = KLoadSpec::new(kokoro_snapshot());
    let registry = candle_audio_kokoro::provider_registry().unwrap();
    let generator = registry
        .load(candle_audio_kokoro::MODEL_ID, &spec)
        .expect("kokoro_82m loads through the explicit registry");
    let req = GenerationRequest {
        prompt: text.into(),
        seed: Some(42),
        audio: Some(AudioParams {
            voice: Some("af_heart".into()),
            language: Some("en".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    match generator
        .generate(&req, &mut |_| {})
        .expect("kokoro generate")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected GenerationOutput::Audio, got {other:?}"),
    }
}

/// Normalize transcript/reference text for CER: lowercase, strip punctuation, collapse whitespace.
fn normalize(s: &str) -> String {
    let cleaned: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Character error rate = Levenshtein(reference, hypothesis) / reference.len().
fn character_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    // Standard DP Levenshtein over the two rows.
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut curr = vec![0usize; h.len() + 1];
    for (i, &rc) in r.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &hc) in h.iter().enumerate() {
            let cost = if rc == hc { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[h.len()] as f32 / r.len() as f32
}

/// The real-transcript DoD: a Kokoro-synthesized clip of KNOWN text transcribes back to that text
/// within a small CER, with monotonic segment timestamps. A stub/garbage transcriber (empty or
/// audio-ignoring output) fails the CER bound; a broken timestamp parse fails the monotonicity
/// check.
#[test]
#[ignore = "real weights: needs openai/whisper-base + hexgrad/Kokoro-82M snapshots (from the required env snapshots); run with --ignored"]
fn whisper_transcribes_kokoro_roundtrip_within_cer() {
    // 1) Kokoro TTS: KNOWN text → a real speech WAV (24 kHz mono).
    let track = synthesize_with_kokoro(KNOWN_TEXT);
    assert!(!track.samples.is_empty(), "kokoro produced no audio");

    // 2) Whisper ASR: transcribe the clip through the explicit registry, load-by-id.
    let spec = LoadSpec::new(whisper_snapshot());
    let transcriber = candle_audio_whisper::provider_registry()
        .unwrap()
        .load_transcriber(candle_audio_whisper::MODEL_ID, &spec)
        .expect("whisper_base loads through the explicit registry");
    assert_eq!(transcriber.descriptor().id, "whisper_base");

    let req = TranscribeRequest {
        audio: track,
        options: TranscribeOptions {
            language: Some("en".into()),
            timestamps: TimestampGranularity::Segment,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut steps = 0u32;
    let mut decoding = 0u32;
    let out = transcriber
        .transcribe(&req, &mut |p| match p {
            Progress::Step { .. } => steps += 1,
            Progress::Decoding => decoding += 1,
            Progress::Loading(_) => {}
        })
        .expect("transcribe");
    assert!(steps >= 1 && decoding == 1, "progress contract");

    // 3) The transcript matches KNOWN text within a small CER.
    let hyp = normalize(&out.text);
    let reference = normalize(KNOWN_TEXT);
    let cer = character_error_rate(&reference, &hyp);
    println!("whisper roundtrip: known={reference:?} transcript={hyp:?} CER={cer:.3}");
    assert!(
        !hyp.trim().is_empty(),
        "empty transcript — the transcriber produced nothing"
    );
    assert!(
        cer <= 0.15,
        "CER {cer:.3} exceeds 0.15 — transcript {hyp:?} does not match known text {reference:?} \
         (an empty/garbage transcript or an audio-ignoring transcriber fails here)"
    );

    // 4) Segment timestamps (when emitted) are monotonic non-decreasing and non-negative.
    let mut last_end = 0.0f32;
    for seg in &out.segments {
        assert!(
            seg.start >= 0.0 && seg.end >= seg.start,
            "segment span invalid: {seg:?}"
        );
        assert!(
            seg.start + 1e-3 >= last_end - 1e-3,
            "segment starts before the previous ended (non-monotonic): {seg:?}"
        );
        last_end = seg.end;
    }
    println!(
        "whisper roundtrip: {} segment(s), detected language {:?}, {} token(s)",
        out.segments.len(),
        out.language,
        out.generated_tokens.unwrap_or(0),
    );
}

/// Determinism: greedy (temperature 0) transcription of the same clip is byte-identical text.
#[test]
#[ignore = "real weights: needs openai/whisper-base + hexgrad/Kokoro-82M snapshots (from the required env snapshots); run with --ignored"]
fn whisper_greedy_transcription_is_deterministic() {
    let track = synthesize_with_kokoro(KNOWN_TEXT);
    let spec = LoadSpec::new(whisper_snapshot());
    let transcriber = candle_audio_whisper::provider_registry()
        .unwrap()
        .load_transcriber(candle_audio_whisper::MODEL_ID, &spec)
        .expect("whisper_base loads through the explicit registry");
    let req = TranscribeRequest {
        audio: track,
        options: TranscribeOptions {
            language: Some("en".into()),
            timestamps: TimestampGranularity::None,
            ..Default::default()
        },
        ..Default::default()
    };
    let a = transcriber.transcribe(&req, &mut |_| {}).expect("first");
    let b = transcriber.transcribe(&req, &mut |_| {}).expect("second");
    assert_eq!(a.text, b.text, "greedy transcription must be deterministic");
}
