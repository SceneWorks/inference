//! Real-weight conformance for the candle OpenVoice V2 voice converter (sc-13223) — the sc-12839
//! release gate: real pinned `converter/checkpoint.pth` → registry load-by-id → convert a source
//! speech clip into a DIFFERENT target voice → a converted WAV that is non-silent, finite, at the
//! model's native rate, **duration-preserving**, and whose timbre has measurably shifted toward the
//! target speaker.
//!
//! `#[ignore]`d and snapshot-gated like every other family's real-weight tests. The distinct source
//! and target voices come from Kokoro (the sanctioned "Kokoro-generated reference audio" path), and
//! the timbre shift is MEASURED with the merged Chatterbox voice embedder (`chatterbox_ve`) — a
//! speaker-identity cosine, the same discriminative embedder sc-12844 shipped. Set
//! `OPENVOICE_V2_SNAPSHOT` to a `myshell-ai/OpenVoiceV2` `converter/` dir (holding `config.json` and
//! `checkpoint.pth`), or leave unset to resolve the pinned hub snapshot; likewise `KOKORO_SNAPSHOT`
//! and `CHATTERBOX_VE_SNAPSHOT`.
//!
//! ```text
//! cargo test --locked -p candle-audio-openvoice --test conformance -- --ignored --nocapture
//! ```
//!
//! ## What the timbre assertion catches
//!
//! The converted clip's speaker embedding must be cosine-**closer to the target** than to the
//! source, AND closer to the target than the source itself was. A converter that ignored the target
//! reference (output == source timbre) would leave the converted embedding sitting on top of the
//! source's — failing both inequalities. That is the assertion that fails if `target_reference` is
//! not actually consumed.

use std::path::PathBuf;

use candle_audio_chatterbox_ve as ve;
use candle_audio_openvoice as ov;
use candle_audio_openvoice::gen_core::{
    AudioParams, AudioTrack, AudioTransformRequest, GenerationOutput, GenerationRequest, LoadSpec,
    VoiceEmbedder, WeightsSource,
};

/// Resolve the OpenVoice V2 converter snapshot dir: `OPENVOICE_V2_SNAPSHOT` (either the repo
/// snapshot root — a `converter/` subdir is picked up automatically — or the `converter/` dir
/// itself) or the pinned hub snapshot.
fn openvoice_snapshot() -> WeightsSource {
    match std::env::var("OPENVOICE_V2_SNAPSHOT") {
        Ok(dir) => {
            let root = PathBuf::from(dir);
            let converter = root.join("converter");
            let loadable = if converter.join("checkpoint.pth").is_file() {
                converter
            } else {
                root
            };
            WeightsSource::Dir(loadable)
        }
        Err(_) => ov::resolve_pinned_snapshot()
            .expect("resolve the pinned myshell-ai/OpenVoiceV2 converter snapshot (network or warm HF cache)"),
    }
}

/// The converter, resolved **through the explicit registry** by id (exactly like a media model).
fn load_converter() -> Box<dyn candle_audio_openvoice::gen_core::AudioTransform> {
    let spec = LoadSpec::new(openvoice_snapshot());
    ov::provider_registry()
        .unwrap()
        .load_audio_transform(ov::MODEL_ID, &spec)
        .expect("openvoice_v2 loads through the explicit registry")
}

/// Synthesize a clip with Kokoro (24 kHz mono).
fn kokoro_clip(text: &str, voice: &str) -> AudioTrack {
    let spec = LoadSpec::new(match std::env::var("KOKORO_SNAPSHOT") {
        Ok(dir) => WeightsSource::Dir(PathBuf::from(dir)),
        Err(_) => candle_audio_kokoro::resolve_pinned_snapshot()
            .expect("resolve the pinned hexgrad/Kokoro-82M snapshot (network or warm HF cache)"),
    });
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

/// Load the Chatterbox voice embedder that measures the timbre shift.
fn load_embedder() -> Box<dyn VoiceEmbedder> {
    let weights = match std::env::var("CHATTERBOX_VE_SNAPSHOT") {
        Ok(dir) => WeightsSource::File(PathBuf::from(dir).join(ve::WEIGHTS_FILE)),
        Err(_) => ve::resolve_pinned_file().expect(
            "resolve the pinned ResembleAI/chatterbox ve.safetensors (network or warm HF cache)",
        ),
    };
    ve::provider_registry()
        .unwrap()
        .load_voice_embedder(ve::MODEL_ID, &LoadSpec::new(weights))
        .expect("chatterbox_ve loads")
}

fn duration_secs(t: &AudioTrack) -> f32 {
    t.samples.len() as f32 / t.sample_rate as f32
}

/// The sc-12839 release gate: the real converter shifts a source clip's timbre toward a target
/// voice while preserving its content/duration.
#[test]
#[ignore = "real weights: needs OpenVoiceV2 converter + Kokoro + chatterbox_ve snapshots (or network); run with --ignored"]
fn openvoice_v2_converts_toward_the_target_voice() {
    let converter = load_converter();

    // Source: af_heart (female) speaking a sentence. Target reference: am_michael (male) — a
    // DIFFERENT voice, saying something else (tone color is text-independent).
    let source = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let target_reference = kokoro_clip(
        "She sells seashells by the seashore on a bright summer morning today.",
        "am_michael",
    );

    let req = AudioTransformRequest {
        audio: source.clone(),
        target_reference: Some(target_reference.clone()),
        seed: Some(1234),
        ..Default::default()
    };
    let out = converter.apply(&req, &mut |_| {}).expect("convert");
    assert_eq!(out.len(), 1, "VoiceConversion produces exactly one track");
    let converted = &out[0];

    // (a) non-silent, finite, expected rate + mono.
    assert_eq!(
        converted.sample_rate,
        ov::OUTPUT_SAMPLE_RATE,
        "native 22.05 kHz"
    );
    assert_eq!(converted.channels, 1);
    assert!(
        converted.samples.iter().all(|s| s.is_finite()),
        "converted samples must be finite"
    );
    let peak = converted
        .samples
        .iter()
        .fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        peak > 0.05,
        "converted clip is effectively silent (peak {peak})"
    );

    // (b) duration preserved (within one hop-frame worth of tolerance; rates differ so compare
    // seconds).
    let src_secs = duration_secs(&source);
    let conv_secs = duration_secs(converted);
    assert!(
        (conv_secs - src_secs).abs() < 0.1,
        "duration must be preserved: source {src_secs:.3}s vs converted {conv_secs:.3}s"
    );

    // (c) timbre shifted toward the target — MEASURED with the chatterbox_ve speaker embedder.
    let embedder = load_embedder();
    let e_src = embedder.embed(&source).expect("embed source");
    let e_tgt = embedder.embed(&target_reference).expect("embed target");
    let e_conv = embedder.embed(converted).expect("embed converted");

    let cos_conv_tgt = ve::cosine_similarity(&e_conv, &e_tgt);
    let cos_conv_src = ve::cosine_similarity(&e_conv, &e_src);
    let cos_src_tgt = ve::cosine_similarity(&e_src, &e_tgt);
    eprintln!(
        "source={:.2}s converted={:.2}s | cos(conv,tgt)={cos_conv_tgt:.4} cos(conv,src)={cos_conv_src:.4} cos(src,tgt)={cos_src_tgt:.4} | shift-toward-target={:.4}",
        src_secs,
        conv_secs,
        cos_conv_tgt - cos_src_tgt
    );

    // The converted clip is closer to the TARGET speaker than to the SOURCE speaker: an output that
    // ignored target_reference (== source timbre) would have cos(conv,src) ≈ 1 ≫ cos(conv,tgt).
    assert!(
        cos_conv_tgt > cos_conv_src,
        "converted must be closer to target than to source (conv,tgt={cos_conv_tgt:.4} conv,src={cos_conv_src:.4})"
    );
    // ...and the conversion moved TOWARD the target relative to where the source sat.
    assert!(
        cos_conv_tgt > cos_src_tgt,
        "conversion must move toward the target (conv,tgt={cos_conv_tgt:.4} src,tgt={cos_src_tgt:.4})"
    );

    // WAV evidence for a human to listen to.
    let out_path = std::env::var("OPENVOICE_WAV_OUT")
        .unwrap_or_else(|_| "openvoice-convert-sc13223.wav".to_string());
    candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out_path), converted)
        .expect("write converted wav");
    eprintln!("wrote converted clip: {out_path} ({conv_secs:.2}s, peak {peak:.3})");
}

/// Determinism: same request + seed ⇒ byte-identical converted samples.
#[test]
#[ignore = "real weights: needs OpenVoiceV2 converter + Kokoro snapshots (or network); run with --ignored"]
fn openvoice_v2_is_deterministic() {
    let converter = load_converter();
    let source = kokoro_clip("A short deterministic check of the converter.", "af_heart");
    let target_reference = kokoro_clip(
        "Any target voice reference clip will do here.",
        "am_michael",
    );
    let req = AudioTransformRequest {
        audio: source,
        target_reference: Some(target_reference),
        seed: Some(7),
        ..Default::default()
    };
    let a = converter.apply(&req, &mut |_| {}).expect("convert a");
    let b = converter.apply(&req, &mut |_| {}).expect("convert b");
    assert_eq!(
        a[0].samples, b[0].samples,
        "same request + seed ⇒ identical samples"
    );
}
