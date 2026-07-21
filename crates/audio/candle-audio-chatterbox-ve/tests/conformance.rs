//! Real-weight conformance for the candle Chatterbox voice embedder (sc-12844) — the sc-12838
//! release gate: real pinned `ve.safetensors` → registry load-by-id → embed reference clips →
//! a real 256-d speaker vector that is unit-norm, finite, and **discriminative** (same-speaker
//! clips cosine-closer than a different-speaker control).
//!
//! `#[ignore]`d and snapshot-gated like every other family's real-weight tests. The distinct
//! reference voices come from Kokoro (the sanctioned "Kokoro-generated reference audio" path):
//! set `KOKORO_SNAPSHOT` to a `hexgrad/Kokoro-82M` snapshot dir (or leave unset to resolve the
//! pinned hub snapshot), and `CHATTERBOX_VE_SNAPSHOT` to a `ResembleAI/chatterbox` snapshot dir
//! holding `ve.safetensors` (or leave unset to resolve the pinned single file via the hub).
//!
//! ```text
//! cargo test --locked -p candle-audio-chatterbox-ve --test conformance -- --ignored --nocapture
//! ```
//!
//! `chatterbox_ve_wav_conformance` also writes the reference clip next to the test output
//! (`CHATTERBOX_VE_WAV_OUT` overrides the path) so a human can listen to the evidence.

use std::path::PathBuf;

use candle_audio_chatterbox_ve as ve;
use candle_audio_chatterbox_ve::gen_core::{
    AudioParams, AudioTrack, GenerationOutput, GenerationRequest, LoadSpec, VoiceEmbedder,
    WeightsSource,
};

/// Resolve the `ve.safetensors` weights from the required `CHATTERBOX_VE_SNAPSHOT` env path (a
/// snapshot dir holding `ve.safetensors`, or the file itself) — the "passed-in path" the provider
/// consumes. Inference never self-fetches or derives a cache location (epic 13657).
fn ve_weights() -> WeightsSource {
    let p = PathBuf::from(std::env::var("CHATTERBOX_VE_SNAPSHOT").expect(
        "set CHATTERBOX_VE_SNAPSHOT to a ResembleAI/chatterbox snapshot dir holding ve.safetensors (or the file itself)",
    ));
    let file = if p.is_dir() {
        p.join(ve::WEIGHTS_FILE)
    } else {
        p
    };
    WeightsSource::File(file)
}

/// The embedder, resolved **through the explicit registry** by id (exactly like a media model).
fn load_embedder() -> Box<dyn VoiceEmbedder> {
    let spec = LoadSpec::new(ve_weights());
    ve::provider_registry()
        .unwrap()
        .load_voice_embedder(ve::MODEL_ID, &spec)
        .expect("chatterbox_ve loads through the explicit registry")
}

/// Synthesize a reference clip with Kokoro (24 kHz mono).
fn kokoro_clip(text: &str, voice: &str) -> AudioTrack {
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(
        std::env::var("KOKORO_SNAPSHOT")
            .expect("set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir"),
    )));
    let gen = candle_audio_kokoro::load(&spec).expect("load kokoro");
    let req = GenerationRequest {
        prompt: text.to_string(),
        audio: Some(AudioParams {
            voice: Some(voice.to_string()),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        ..Default::default()
    };
    match gen.generate(&req, &mut |_| {}).expect("kokoro generate") {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    }
}

fn assert_valid_embedding(e: &[f32]) {
    assert_eq!(e.len(), ve::descriptor().embedding_dim);
    assert!(e.iter().all(|v| v.is_finite()), "embedding must be finite");
    let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embedding must be L2-normalized, got norm {norm}"
    );
    assert!(
        e.iter().any(|&v| v != 0.0),
        "embedding must not be all-zero"
    );
}

/// The sc-12838 release gate: the real `ve` weights turn distinct reference voices into
/// discriminative speaker vectors — same-speaker clips are cosine-closer than a different-speaker
/// control by a clear margin. A vector that ignored the reference audio would fail this.
#[test]
#[ignore = "real weights: needs ve.safetensors + a Kokoro snapshot (CHATTERBOX_VE_SNAPSHOT/KOKORO_SNAPSHOT or network); run with --ignored"]
fn chatterbox_ve_discriminates_speakers() {
    let embedder = load_embedder();

    let a1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let a2 = kokoro_clip(
        "She sells seashells by the seashore on a bright summer morning.",
        "af_heart",
    );
    let b1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "am_michael",
    );

    let ea1 = embedder.embed(&a1).expect("embed a1");
    let ea2 = embedder.embed(&a2).expect("embed a2");
    let eb1 = embedder.embed(&b1).expect("embed b1");
    for e in [&ea1, &ea2, &eb1] {
        assert_valid_embedding(e);
    }

    let same = ve::cosine_similarity(&ea1, &ea2);
    let cross = 0.5 * (ve::cosine_similarity(&ea1, &eb1) + ve::cosine_similarity(&ea2, &eb1));
    eprintln!(
        "same-speaker cos = {same:.4}, diff-speaker cos = {cross:.4}, margin = {:.4}",
        same - cross
    );
    // A working GE2E speaker encoder puts same-speaker clips near 1.0 and a different speaker
    // clearly lower; the margin is the assertion that fails if the clone ignored the reference.
    assert!(
        same > 0.75,
        "same-speaker clips must be strongly similar (got {same:.4})"
    );
    assert!(
        same > cross + 0.1,
        "same-speaker must be clearly closer than different-speaker (same={same:.4}, cross={cross:.4})"
    );
}

/// The real-WAV DoD companion: the reference clip the embedder consumed is non-silent, 24 kHz
/// mono, and of the expected order-of-magnitude duration — written out for a human to listen to.
#[test]
#[ignore = "real weights: needs ve.safetensors + a Kokoro snapshot; run with --ignored"]
fn chatterbox_ve_wav_conformance() {
    let embedder = load_embedder();
    let reference = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    assert_eq!(reference.sample_rate, 24_000);
    assert_eq!(reference.channels, 1);
    let secs = reference.samples.len() as f32 / 24_000.0;
    assert!(
        (1.0..30.0).contains(&secs),
        "reference duration {secs:.2}s out of range"
    );
    let peak = reference
        .samples
        .iter()
        .fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        peak > 0.05,
        "reference clip is effectively silent (peak {peak})"
    );

    // The embedder produces a valid vector from that exact clip.
    let e = embedder.embed(&reference).expect("embed reference");
    assert_valid_embedding(&e);

    let out = std::env::var("CHATTERBOX_VE_WAV_OUT")
        .unwrap_or_else(|_| "chatterbox-ve-reference.wav".to_string());
    candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out), &reference)
        .expect("write reference wav");
    eprintln!("wrote reference clip: {out} ({secs:.2}s, peak {peak:.3})");
}
