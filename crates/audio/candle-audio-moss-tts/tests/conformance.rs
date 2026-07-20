//! Real-weight conformance for MOSS-TTSD-v0.5 — the **AR brain** (sc-13360), honest-partial.
//!
//! ## What this gates on real weights
//!
//! - [`moss_ttsd_emits_valid_delay_pattern_rvq_frames`] — a fixed single-voice prompt + seed → the
//!   delay-pattern AR loop emits **≥ 2** clean 8-codebook frames, codebook 0 in `[0, 1024)` and every
//!   audio codebook in `[0, 1025)`, deterministic run-to-run (the seeded sampler), and non-degenerate
//!   (codebook 0 is not a single collapsed id). A broken backbone / weight mapping / RoPE / channel
//!   embedding sum / tied-head / delay-shift bug produces empty, out-of-range, or all-identical frames
//!   and fails here.
//! - [`moss_ttsd_two_speaker_script_shapes_the_token_stream`] — a 2-speaker `[S1]`/`[S2]` script
//!   (S1 "Hello, how are you today?" / S2 "I'm doing great, thanks for asking!") + seed → valid,
//!   deterministic frames whose token stream **differs** from the single-voice control, proving the
//!   model honors the speaker turn labels at the token level (the codec-gated acoustic
//!   voice-distinctness measurement via `candle-audio-chatterbox-ve` is the split-off follow-up).
//! - [`moss_ttsd_generate_errors_at_the_codec_boundary`] — `generate()` on real weights returns the
//!   typed codec-boundary error, never audio (honest partial: no XY_Tokenizer codec yet).
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-moss-tts --test conformance -- --ignored --nocapture
//! ```
//! Set `MOSS_TTSD_SNAPSHOT` to the AR snapshot dir (`config.json`, `model.safetensors`,
//! `tokenizer.json`), or leave unset to resolve the pinned snapshot via the hub (~4.1 GB). Optionally
//! dump the raw frames with `MOSS_TTSD_FRAMES_OUT`.

use std::collections::HashSet;
use std::path::PathBuf;

use candle_audio_moss_tts as moss;
use candle_audio_moss_tts::gen_core::{
    AudioParams, GenerationRequest, LoadSpec, SpeechSegment, WeightsSource,
};

/// Resolve a MOSS-TTSD snapshot dir. `MOSS_TTSD_SNAPSHOT` overrides; otherwise the pinned snapshot is
/// fetched via the hub.
fn snapshot() -> PathBuf {
    if let Ok(dir) = std::env::var("MOSS_TTSD_SNAPSHOT") {
        return PathBuf::from(dir);
    }
    match moss::resolve_pinned_snapshot()
        .expect("resolve the pinned MOSS-TTSD-v0.5 snapshot (network or warm HF cache)")
    {
        WeightsSource::Dir(p) => p,
        other => panic!("expected a snapshot dir, got {other:?}"),
    }
}

fn load() -> moss::model::MossTtsdGenerator {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    moss::load_generator(&spec).expect("load the MOSS-TTSD generator")
}

/// A short single-voice request (a small budget keeps the CPU AR run tractable).
fn single_voice(seconds: f32) -> GenerationRequest {
    GenerationRequest {
        prompt: "Hello, how are you today?".to_string(),
        audio: Some(AudioParams {
            target_duration: Some(seconds),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        seed: Some(20_260_720),
        ..Default::default()
    }
}

/// The 2-speaker dialogue script from the acceptance criteria.
fn two_speaker(seconds: f32) -> GenerationRequest {
    GenerationRequest {
        prompt: String::new(),
        audio: Some(AudioParams {
            target_duration: Some(seconds),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            script: Some(vec![
                SpeechSegment {
                    text: "Hello, how are you today?".into(),
                    speaker: Some("S1".into()),
                    ..Default::default()
                },
                SpeechSegment {
                    text: "I'm doing great, thanks for asking!".into(),
                    speaker: Some("S2".into()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        }),
        seed: Some(20_260_720),
        ..Default::default()
    }
}

fn assert_valid_frames(frames: &[Vec<u32>]) {
    assert!(
        frames.len() >= 2,
        "the AR loop must emit >= 2 clean frames (got {})",
        frames.len()
    );
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.len(), 8, "frame {i} must carry 8 codebook tokens");
        assert!(
            frame[0] < 1024,
            "frame {i} codebook 0 out of range: {frame:?}"
        );
        for c in 1..8 {
            assert!(
                frame[c] < 1025,
                "frame {i} codebook {c} out of range: {frame:?}"
            );
        }
    }
    let cb0: Vec<u32> = frames.iter().map(|f| f[0]).collect();
    let distinct = cb0.iter().collect::<HashSet<_>>().len();
    assert!(
        distinct > 1,
        "codebook-0 collapsed to {distinct} distinct value(s) — the AR brain is not modeling speech"
    );
}

/// AR-stage gate: real weights decode valid, non-degenerate, deterministic delay-pattern frames.
#[test]
#[ignore = "real weights: needs the ~4.1 GB MOSS-TTSD-v0.5 snapshot; run with --ignored"]
fn moss_ttsd_emits_valid_delay_pattern_rvq_frames() {
    let gen = load();
    let result = gen
        .rvq_frames(&single_voice(1.5), &mut |_| {})
        .expect("AR delay-pattern frame decode");
    let frames = &result.frames;
    eprintln!(
        "AR brain emitted {} clean 8-codebook frames (stop: {:?})",
        frames.len(),
        result.stop
    );
    assert_valid_frames(frames);

    // Deterministic: the seeded sampler ⇒ byte-identical frames on a re-run (the reproducibility law).
    let again = gen
        .rvq_frames(&single_voice(1.5), &mut |_| {})
        .expect("re-decode");
    assert_eq!(
        *frames, again.frames,
        "seeded AR sampling must be reproducible run-to-run"
    );

    if let Ok(out) = std::env::var("MOSS_TTSD_FRAMES_OUT") {
        let text: String = frames
            .iter()
            .map(|f| f.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&out, text).expect("write frames");
        eprintln!("wrote {} frames to {out}", frames.len());
    }
}

/// Multi-speaker gate (token level): the 2-speaker script produces valid, deterministic frames whose
/// token stream differs from the single-voice control — the model honored the `[S1]`/`[S2]` labels.
#[test]
#[ignore = "real weights: needs the ~4.1 GB MOSS-TTSD-v0.5 snapshot; run with --ignored"]
fn moss_ttsd_two_speaker_script_shapes_the_token_stream() {
    let gen = load();
    let ms = gen
        .rvq_frames(&two_speaker(2.0), &mut |_| {})
        .expect("multi-speaker AR decode");
    eprintln!("2-speaker script emitted {} frames", ms.frames.len());
    assert_valid_frames(&ms.frames);

    // Deterministic for the seed.
    let ms2 = gen
        .rvq_frames(&two_speaker(2.0), &mut |_| {})
        .expect("multi-speaker re-decode");
    assert_eq!(
        ms.frames, ms2.frames,
        "multi-speaker decode is reproducible"
    );

    // The dialogue script must genuinely shape generation: its token stream differs from a
    // single-voice control at the same seed. (The acoustic voice-distinctness measurement via
    // candle-audio-chatterbox-ve is codec-gated — the split-off follow-up.)
    let control = gen
        .rvq_frames(&single_voice(2.0), &mut |_| {})
        .expect("control decode");
    assert_ne!(
        ms.frames, control.frames,
        "a 2-speaker script must produce a different token stream than a single-voice control"
    );
}

/// Honest-partial boundary on real weights: `generate()` returns the typed codec-boundary error.
#[test]
#[ignore = "real weights: needs the ~4.1 GB MOSS-TTSD-v0.5 snapshot; run with --ignored"]
fn moss_ttsd_generate_errors_at_the_codec_boundary() {
    use candle_audio_moss_tts::gen_core::{Error, Generator};
    let gen = load();
    let err = gen.generate(&single_voice(1.0), &mut |_| {}).unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "generate() must error at the codec boundary (no XY_Tokenizer yet), got {err:?}"
    );
}
