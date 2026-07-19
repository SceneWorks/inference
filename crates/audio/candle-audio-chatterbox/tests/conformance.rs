//! Real-weight conformance for the Chatterbox clone-TTS generator (sc-13222).
//!
//! ## What this slice gates (honest partial)
//!
//! This slice ports the **T3 speech-token LM** (real `t3_cfg.safetensors`), not yet the S3Gen
//! token→waveform stack (see `candle_audio_chatterbox::s3gen`). So the conformance here is the
//! **T3 stage** — the ported half — exercised on real weights:
//!
//! - [`chatterbox_t3_produces_valid_speech_tokens`] — a text prompt + a real `chatterbox_ve`
//!   voice embedding → the T3 LM decodes a non-empty, in-range, non-degenerate speech-token
//!   sequence and terminates on the stop token. This is a genuine gate: a broken backbone / weight
//!   mapping / RoPE / sampling would produce an empty, all-one-value, or never-terminating
//!   sequence and fail here.
//! - [`chatterbox_t3_responds_to_the_reference_voice`] — two *different* reference voices produce
//!   *different* token sequences under the same text + seed. The assertion fails if T3 ignores the
//!   speaker conditioning (a generic/default voice) — the same "must not ignore the reference"
//!   property the sc-12838 gate demands, applied at the T3 stage.
//!
//! ## What remains blocked
//!
//! The **full sc-12838 clone gate** — a cloned-voice WAV whose `chatterbox_ve` embedding is closer
//! to the reference than to a different-voice control — requires the S3Gen stack (s3tokenizer FSQ +
//! CAMPPlus + flow-matching decoder + HiFTNet vocoder) to turn these tokens into a waveform. That
//! is **not yet ported**; [`chatterbox_generate_stops_honestly_at_the_s3gen_boundary`] asserts that
//! `generate()` runs T3 and then returns a typed error naming the gap — it must never emit fake
//! audio. The full WAV gate is deferred to the S3Gen follow-up stories.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-chatterbox --test conformance -- --ignored --nocapture
//! ```
//! Set `CHATTERBOX_SNAPSHOT` to a `ResembleAI/chatterbox` snapshot dir (holding the
//! `t3_cfg.safetensors` and `tokenizer.json` files) or leave unset to resolve the pinned files via
//! the hub. `KOKORO_SNAPSHOT` and `CHATTERBOX_VE_SNAPSHOT` supply the reference clips and the voice
//! embedder as elsewhere.

use std::path::PathBuf;

use candle_audio_chatterbox as cb;
use candle_audio_chatterbox::gen_core::{
    AudioParams, AudioTrack, Conditioning, GenerationOutput, GenerationRequest, LoadSpec,
    WeightsSource,
};

/// Resolve a Chatterbox snapshot dir holding at least `t3_cfg.safetensors` + `tokenizer.json`.
/// `CHATTERBOX_SNAPSHOT` overrides; otherwise the pinned T3 + tokenizer files are fetched via the
/// hub (the S3Gen checkpoint is intentionally NOT required — this slice only loads T3).
fn chatterbox_snapshot() -> PathBuf {
    if let Ok(dir) = std::env::var("CHATTERBOX_SNAPSHOT") {
        return PathBuf::from(dir);
    }
    // Fetch just the T3 files (avoids the ~1 GB S3Gen download the full resolver would pull).
    use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
    let dir = pinned_snapshot_dir(cb::HUB_REPO, cb::HUB_REVISION, cb::T3_WEIGHTS_FILE)
        .expect("resolve the pinned chatterbox t3_cfg.safetensors (network or warm HF cache)");
    hf_get_pinned(cb::HUB_REPO, cb::HUB_REVISION, cb::TOKENIZER_FILE)
        .expect("resolve the pinned chatterbox tokenizer.json");
    match dir {
        WeightsSource::Dir(p) => p,
        other => panic!("expected a snapshot dir, got {other:?}"),
    }
}

fn load_generator() -> cb::ChatterboxGenerator {
    let spec = LoadSpec::new(WeightsSource::Dir(chatterbox_snapshot()));
    cb::load_generator(&spec).expect("load the chatterbox generator")
}

/// A reference clip synthesized with Kokoro (24 kHz mono) — the sanctioned reference-audio path.
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
        // Seed the reference synthesis so the derived voice embedding — and thus the whole T3
        // decode — is reproducible run-to-run (the gen-core seed law); an un-seeded reference
        // would drift the conditioning and the token sequence.
        seed: Some(20260719),
        ..Default::default()
    };
    match gen.generate(&req, &mut |_| {}).expect("kokoro generate") {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    }
}

/// The 256-d `chatterbox_ve` embedding of a reference clip.
fn voice_embedding(clip: &AudioTrack) -> Vec<f32> {
    let spec = LoadSpec::new(match std::env::var("CHATTERBOX_VE_SNAPSHOT") {
        Ok(dir) => {
            WeightsSource::File(PathBuf::from(dir).join(candle_audio_chatterbox_ve::WEIGHTS_FILE))
        }
        Err(_) => candle_audio_chatterbox_ve::resolve_pinned_file()
            .expect("resolve the pinned ve.safetensors (network or warm HF cache)"),
    });
    let embedder = candle_audio_chatterbox_ve::load(&spec).expect("load chatterbox_ve");
    embedder.embed(clip).expect("embed reference")
}

fn request(prompt: &str, embedding: Vec<f32>, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_string(),
        audio: Some(AudioParams {
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        conditioning: vec![Conditioning::VoiceEmbedding {
            embedding,
            strength: None,
        }],
        seed: Some(seed),
        ..Default::default()
    }
}

/// T3 stage gate: real weights decode a valid, non-degenerate speech-token sequence.
#[test]
#[ignore = "real weights: needs t3_cfg.safetensors + ve + a Kokoro snapshot; run with --ignored"]
fn chatterbox_t3_produces_valid_speech_tokens() {
    let gen = load_generator();
    let clip = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let emb = voice_embedding(&clip);
    let req = request(
        "Hello, this is a test of the cloned voice synthesizer.",
        emb,
        42,
    );

    let (raw, real) = gen
        .speech_tokens(&req, &mut |_| {})
        .expect("T3 speech-token decode");
    eprintln!(
        "T3 decoded {} raw speech tokens ({} after dropping specials)",
        raw.len(),
        real.len()
    );
    // Non-empty and terminated by the stop-speech token.
    assert!(!raw.is_empty(), "T3 produced no speech tokens");
    assert!(
        raw.len() < cb::T3Config::LLAMA_520M.max_speech_tokens,
        "T3 never emitted a stop token (ran to the cap) — decode is broken"
    );
    assert_eq!(
        *raw.last().unwrap(),
        cb::T3Config::LLAMA_520M.stop_speech_token,
        "the last token must be the stop-speech token"
    );
    // The real (stripped) tokens are in the valid S3 codebook and not all identical (a collapsed
    // backbone / RoPE / sampling bug degenerates to a single repeated id).
    assert!(
        !real.is_empty(),
        "no real speech tokens after stripping specials"
    );
    assert!(
        real.iter()
            .all(|&t| (t as usize) < cb::config::SPEECH_VOCAB_SIZE),
        "stripped tokens must all be valid S3 codes"
    );
    let distinct = real.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 3,
        "speech tokens collapsed to {distinct} distinct value(s) — the LM is not modeling speech"
    );

    // Honest WAV evidence: this slice cannot render the CLONE waveform (S3Gen unported), so the
    // artifact written is the REFERENCE clip the T3 conditioning consumed — the pipeline INPUT, not
    // a clone output. Never a fabricated clone.
    if let Ok(out) = std::env::var("CHATTERBOX_WAV_OUT") {
        candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out), &clip)
            .expect("write reference clip");
        let secs = clip.samples.len() as f32 / clip.sample_rate as f32;
        eprintln!(
            "wrote REFERENCE input clip (NOT a clone — S3Gen unported): {out} ({secs:.2}s, {} tokens \
             decoded by T3)",
            real.len()
        );
    }
}

/// T3 stage gate: the LM RESPONDS to the reference voice (different voices → different tokens).
#[test]
#[ignore = "real weights: needs t3_cfg.safetensors + ve + a Kokoro snapshot; run with --ignored"]
fn chatterbox_t3_responds_to_the_reference_voice() {
    let gen = load_generator();
    let text = "She sells seashells by the seashore.";
    let seed = 7;

    let voice_a = voice_embedding(&kokoro_clip(text, "af_heart"));
    let voice_b = voice_embedding(&kokoro_clip(text, "am_michael"));

    let (_, real_a) = gen
        .speech_tokens(&request(text, voice_a, seed), &mut |_| {})
        .expect("decode voice A");
    let (_, real_b) = gen
        .speech_tokens(&request(text, voice_b, seed), &mut |_| {})
        .expect("decode voice B");

    eprintln!(
        "voice A: {} tokens, voice B: {} tokens",
        real_a.len(),
        real_b.len()
    );
    // Same text + seed but different speaker conditioning must NOT produce identical token
    // sequences — that is exactly the "clone ignored the reference" failure the gate must catch.
    assert_ne!(
        real_a, real_b,
        "different reference voices produced identical speech tokens — T3 ignored the speaker \
         conditioning (the clone would be voice-agnostic)"
    );
}

/// The S3Gen boundary is honest: `generate()` runs T3 and then errors, never emitting fake audio.
#[test]
#[ignore = "real weights: needs t3_cfg.safetensors + ve + a Kokoro snapshot; run with --ignored"]
fn chatterbox_generate_stops_honestly_at_the_s3gen_boundary() {
    use candle_audio_chatterbox::gen_core::Generator;
    let gen = load_generator();
    let emb = voice_embedding(&kokoro_clip("Testing the boundary.", "af_heart"));
    let req = request("The vocoder is not yet ported.", emb, 1);

    let err = gen
        .generate(&req, &mut |_| {})
        .expect_err("generate must stop at the S3Gen boundary, not fabricate audio");
    let msg = format!("{err}");
    eprintln!("honest boundary error: {msg}");
    assert!(
        msg.contains("S3Gen"),
        "the boundary error must name S3Gen: {msg}"
    );
    assert!(
        msg.contains("speech tokens"),
        "the boundary error must report the T3 tokens produced: {msg}"
    );
}

// ---------------------------------------------------------------------------------------------
// s3tokenizer (sc-13235): the Whisper-v2 FSMN encoder + FSQ head → 25 Hz speech tokens.
// ---------------------------------------------------------------------------------------------

/// Resolve a snapshot dir holding `s3gen.safetensors` (the s3tokenizer weights). `CHATTERBOX_SNAPSHOT`
/// overrides (and must contain the S3Gen checkpoint); otherwise the pinned `s3gen.safetensors`
/// (~1 GB) is fetched via the hub.
fn s3gen_snapshot() -> PathBuf {
    if let Ok(dir) = std::env::var("CHATTERBOX_SNAPSHOT") {
        let p = PathBuf::from(dir);
        assert!(
            p.join(cb::s3gen::S3GEN_WEIGHTS_FILE).is_file(),
            "CHATTERBOX_SNAPSHOT {} must contain {} for the s3tokenizer gate",
            p.display(),
            cb::s3gen::S3GEN_WEIGHTS_FILE
        );
        return p;
    }
    use candle_audio::hub::pinned_snapshot_dir;
    match pinned_snapshot_dir(
        cb::HUB_REPO,
        cb::HUB_REVISION,
        cb::s3gen::S3GEN_WEIGHTS_FILE,
    )
    .expect("resolve the pinned chatterbox s3gen.safetensors (network or warm HF cache)")
    {
        WeightsSource::Dir(p) => p,
        other => panic!("expected a snapshot dir, got {other:?}"),
    }
}

/// The DoD gate: the ported s3tokenizer tokenizes a real Kokoro reference clip into a plausible
/// 25 Hz speech-token sequence in `[0, 6560]`, deterministically, and responds to the content.
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn s3tokenizer_encodes_a_reference_at_25hz() {
    use std::collections::HashSet;

    let tok = cb::S3Tokenizer::from_snapshot(&s3gen_snapshot())
        .expect("load the s3tokenizer from s3gen.safetensors");

    let clip = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let dur = clip.samples.len() as f32 / clip.sample_rate as f32;

    let codes = tok
        .encode(&clip.samples, clip.sample_rate)
        .expect("tokenize the reference clip");
    let n = codes.len();
    let min = *codes.iter().min().unwrap();
    let max = *codes.iter().max().unwrap();
    let distinct = codes.iter().collect::<HashSet<_>>().len();
    let rate = n as f32 / dur;
    eprintln!(
        "s3tokenizer: {n} tokens over {dur:.2}s = {rate:.2} Hz; range [{min}, {max}], \
         {distinct} distinct"
    );

    // Non-empty and every token a valid FSQ code (`3^8 = 6561` codebook → ids in [0, 6560]).
    assert!(!codes.is_empty(), "s3tokenizer produced no tokens");
    assert!(
        min >= 0 && max <= 6560,
        "tokens must be valid S3 FSQ codes in [0, 6560], got [{min}, {max}]"
    );
    // ~25 tokens/second (the defining property of the model; generous tolerance for the
    // resample + conv-boundary framing).
    assert!(
        (rate - 25.0).abs() < 2.5,
        "token rate {rate:.2} Hz is not ≈ 25 Hz"
    );
    // Count matches the clip duration to within framing slack.
    let expected = (dur * 25.0).round() as i64;
    assert!(
        (n as i64 - expected).abs() <= (0.1 * expected as f32) as i64 + 2,
        "token count {n} not ≈ {expected} for a {dur:.2}s clip"
    );
    // Non-degenerate: real speech spans many codes (a collapsed encoder/RoPE/FSMN/FSQ bug would
    // emit one repeated id or a tiny alphabet).
    assert!(
        distinct > 10,
        "tokens collapsed to {distinct} distinct value(s) — the encoder is not modeling speech"
    );

    // Deterministic: same clip ⇒ byte-identical tokens (the reproducibility law).
    let codes2 = tok
        .encode(&clip.samples, clip.sample_rate)
        .expect("re-tokenize");
    assert_eq!(codes, codes2, "s3tokenizer must be deterministic");

    // Responds to content: a different clip ⇒ a different token sequence (a content-agnostic
    // encoder would fail here).
    let other = kokoro_clip("She sells seashells by the seashore.", "am_michael");
    let codes_other = tok
        .encode(&other.samples, other.sample_rate)
        .expect("tokenize a different clip");
    assert_ne!(
        codes, codes_other,
        "different references produced identical tokens — the encoder ignores content"
    );
}

/// Provider integration: a `ReferenceAudio` request now derives T3's `cond_prompt_speech_tokens`
/// from the clip via the s3tokenizer (empty in sc-13222). The prompt is capped at the T3
/// `speech_cond_prompt_len` (150) and every token is a valid S3 code.
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn chatterbox_reference_audio_fills_the_t3_prompt_tokens() {
    let spec = LoadSpec::new(WeightsSource::Dir(s3gen_snapshot()));
    let gen = cb::load_generator(&spec).expect("load the chatterbox generator");

    // A ≥ 6 s reference so the prompt fills to the full speech_cond_prompt_len.
    let clip = kokoro_clip(
        "The quick brown fox jumps over the lazy dog, then trots back across the meadow at dawn \
         while the birds begin their morning chorus over the quiet valley below.",
        "af_heart",
    );
    let dur = clip.samples.len() as f32 / clip.sample_rate as f32;
    let prompt = gen
        .reference_speech_tokens(&clip)
        .expect("derive prompt tokens from the reference clip");
    eprintln!(
        "T3 conditioning prompt: {} tokens from a {dur:.2}s reference (cap 150)",
        prompt.len()
    );
    assert!(
        !prompt.is_empty(),
        "reference conditioning yielded no prompt tokens"
    );
    assert!(
        prompt.len() <= cb::config::SPEECH_COND_PROMPT_LEN,
        "prompt must be capped at speech_cond_prompt_len ({})",
        cb::config::SPEECH_COND_PROMPT_LEN
    );
    assert!(
        prompt
            .iter()
            .all(|&t| (t as usize) < cb::config::SPEECH_VOCAB_SIZE),
        "every prompt token must be a valid S3 code (< 6561)"
    );
    if dur >= 6.0 {
        assert_eq!(
            prompt.len(),
            cb::config::SPEECH_COND_PROMPT_LEN,
            "a ≥6s reference should fill the full 150-token prompt"
        );
    }
}
