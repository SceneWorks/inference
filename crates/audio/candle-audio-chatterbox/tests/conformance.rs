//! Real-weight conformance for the Chatterbox clone-TTS generator (sc-13222 → sc-13239).
//!
//! ## What this gates (end-to-end)
//!
//! The whole clone pipeline is ported: T3 speech-token LM → S3Gen token→waveform → PerTh watermark.
//! The **T3 stage** is gated on real weights on its own here:
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
//! ## The full clone gate (sc-13239)
//!
//! The **full sc-12838 clone gate** — a cloned-voice WAV whose `chatterbox_ve` embedding is closer
//! to the reference than to a different-voice control — is
//! [`chatterbox_clones_a_reference_voice_end_to_end`]: it runs the whole registry path
//! (`generate()`) end-to-end through the assembled S3Gen stack (s3tokenizer FSQ + CAMPPlus +
//! flow-matching decoder + HiFTNet vocoder + PerTh watermark) and asserts voice similarity,
//! non-silence, the 24 kHz rate, a token-proportional duration, and the provenance watermark. A
//! VoiceEmbedding-only request still stops honestly, since S3Gen's reference is not recoverable from
//! the ve vector ([`chatterbox_generate_requires_reference_audio_for_a_full_clone`]).
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-chatterbox --test conformance -- --ignored --nocapture
//! ```
//! Set `CHATTERBOX_SNAPSHOT` to a `ResembleAI/chatterbox` snapshot dir (holding the
//! `t3_cfg.safetensors` and `tokenizer.json` files) or leave unset to resolve the pinned files via
//! the hub. `KOKORO_SNAPSHOT` supplies the reference clips as elsewhere.
//!
//! ## Passed-in component snapshots (sc-13660)
//!
//! The provider no longer self-fetches its `perth` and `voice_embedding` co-requisites — they are
//! staged as `LoadSpec` components from explicit local paths. These `#[ignore]`d real-weight tests
//! read those paths from env vars (a dir holding the file, or the file itself), each falling back to
//! the pinned-SHA hub fetch when unset (mirroring the base-snapshot helpers above), and stage them
//! via [`staged_spec`]:
//!
//! - `CHATTERBOX_VE_SNAPSHOT` → the `voice_embedding` component (`ve.safetensors`).
//! - `CHATTERBOX_PERTH_SNAPSHOT` → the `perth` component (`perth_implicit.safetensors`).
//!
//! `real-weights.yml` / the SceneWorks smoke harness set these to pre-materialized dirs.
//! [`chatterbox_gates_required_components_at_load`] is a **weights-free** (non-`#[ignore]`d) proof
//! that a missing/unknown component fails at LOAD.

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

/// The `voice_embedding` component: the `ve.safetensors` file from `CHATTERBOX_VE_SNAPSHOT` (a dir
/// holding it, or the file itself). Falls back to the pinned-SHA hub fetch when unset — never the
/// `resolve_pinned_*` production helper (sc-13660).
fn ve_weights_file() -> PathBuf {
    match std::env::var("CHATTERBOX_VE_SNAPSHOT") {
        Ok(p) => {
            let p = PathBuf::from(p);
            if p.is_dir() {
                p.join(candle_audio_chatterbox_ve::WEIGHTS_FILE)
            } else {
                p
            }
        }
        Err(_) => candle_audio::hub::hf_get_pinned(
            candle_audio_chatterbox_ve::HUB_REPO,
            candle_audio_chatterbox_ve::HUB_REVISION,
            candle_audio_chatterbox_ve::WEIGHTS_FILE,
        )
        .expect("fetch the pinned ve.safetensors (network or warm HF cache)"),
    }
}

/// The `perth` component: the `perth_implicit.safetensors` file from `CHATTERBOX_PERTH_SNAPSHOT` (a
/// dir holding it, or the file itself). Falls back to the pinned-SHA hub fetch of the
/// `SceneWorks/perth-implicit` pin when unset — the provider itself no longer self-fetches (sc-13660);
/// the test stages the passed-in path as the `perth` component.
fn perth_weights_file() -> PathBuf {
    match std::env::var("CHATTERBOX_PERTH_SNAPSHOT") {
        Ok(p) => {
            let p = PathBuf::from(p);
            if p.is_dir() {
                p.join(cb::PERTH_WEIGHTS_FILE)
            } else {
                p
            }
        }
        Err(_) => candle_audio::hub::hf_get_pinned(
            cb::PERTH_HUB_REPO,
            cb::PERTH_HUB_REVISION,
            cb::PERTH_WEIGHTS_FILE,
        )
        .expect("fetch the pinned perth_implicit.safetensors (network or warm HF cache)"),
    }
}

/// A load spec over a chatterbox snapshot `dir` with BOTH required components staged from their
/// env-pointed local paths — the "passed-in components" the provider consumes now that `perth` and
/// `voice_embedding` are no longer self-fetched (sc-13660).
fn staged_spec(dir: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(dir))
        .with_component(
            cb::COMPONENT_PERTH,
            WeightsSource::File(perth_weights_file()),
        )
        .with_component(
            cb::COMPONENT_VOICE_EMBEDDING,
            WeightsSource::File(ve_weights_file()),
        )
}

/// **Weights-free load-gate conformance (sc-13660 / sc-13658).** The provider declares
/// `required_components = ["perth", "voice_embedding"]`, so a MISSING required component — or an
/// unrecognized component key — must be a LOAD-time error, not a mid-render fetch or a first-`generate`
/// failure. The gen-core testkit drives the real `cb::load` with placeholder component paths (never
/// read), so this needs no weights and no network and runs as an ordinary (non-`#[ignore]`d) test.
#[test]
fn chatterbox_gates_required_components_at_load() {
    let base = LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()))
        .with_component(
            cb::COMPONENT_PERTH,
            WeightsSource::File(PathBuf::from("unused-perth.safetensors")),
        )
        .with_component(
            cb::COMPONENT_VOICE_EMBEDDING,
            WeightsSource::File(PathBuf::from("unused-ve.safetensors")),
        );
    gen_core_testkit::check_component_load_gate(cb::load, &base, cb::REQUIRED_COMPONENTS)
        .expect("chatterbox must gate its required components at load");
}

fn load_generator() -> cb::ChatterboxGenerator {
    let spec = staged_spec(chatterbox_snapshot());
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

/// The 256-d `chatterbox_ve` embedding of a reference clip, from the same env-pointed `ve.safetensors`
/// the `voice_embedding` component uses (sc-13660: no `resolve_pinned_*`).
fn voice_embedding(clip: &AudioTrack) -> Vec<f32> {
    let spec = LoadSpec::new(WeightsSource::File(ve_weights_file()));
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

    // This T3-stage test writes the REFERENCE clip the conditioning consumed — the pipeline INPUT,
    // not a clone output (the full clone WAV is rendered + gated by
    // `chatterbox_clones_a_reference_voice_end_to_end`). Never a fabricated clone.
    if let Ok(out) = std::env::var("CHATTERBOX_T3_REF_WAV_OUT") {
        candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out), &clip)
            .expect("write reference clip");
        let secs = clip.samples.len() as f32 / clip.sample_rate as f32;
        eprintln!(
            "wrote REFERENCE input clip (the T3 conditioning input, not a clone): {out} \
             ({secs:.2}s, {} tokens decoded by T3)",
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

/// VoiceEmbedding-only conditioning is an honest boundary: `generate()` runs T3 but cannot render a
/// full clone WAV (S3Gen needs the reference clip's mel / prompt tokens / x-vector, none recoverable
/// from the 256-d ve vector), so it returns a typed error naming `ReferenceAudio` rather than
/// fabricating a voice-agnostic clone. The full clone WAV is gated by
/// [`chatterbox_clones_a_reference_voice_end_to_end`] (ReferenceAudio conditioning).
#[test]
#[ignore = "real weights: needs t3_cfg.safetensors + ve + a Kokoro snapshot; run with --ignored"]
fn chatterbox_generate_requires_reference_audio_for_a_full_clone() {
    use candle_audio_chatterbox::gen_core::Generator;
    let gen = load_generator();
    let emb = voice_embedding(&kokoro_clip("Testing the boundary.", "af_heart"));
    let req = request(
        "A bare voice vector cannot supply the S3Gen reference.",
        emb,
        1,
    );

    let err = gen
        .generate(&req, &mut |_| {})
        .expect_err("VoiceEmbedding-only generate must not fabricate a voice-agnostic clone");
    let msg = format!("{err}");
    eprintln!("honest boundary error: {msg}");
    assert!(
        msg.contains("ReferenceAudio"),
        "the boundary error must name the required ReferenceAudio conditioning: {msg}"
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

/// A long (>30 s) reference built by concatenating several distinct Kokoro sentences, so the
/// s3tokenizer's >30 s sliding-window path is genuinely exercised (Chatterbox's own references are
/// ≤10 s and never reach it). Returns the concatenated 24 kHz samples + rate.
fn long_kokoro_samples() -> (Vec<f32>, u32) {
    // Nine varied ~4 s sentences (same voice) → ~35 s, comfortably past the 30 s window.
    let sentences = [
        "The quick brown fox jumps over the lazy dog near the river bank at first light.",
        "She sells seashells by the seashore while the morning tide rolls gently back in.",
        "A journey of a thousand miles begins beneath the weary traveler's very first footstep.",
        "The five boxing wizards jump quickly as the autumn wind scatters the fallen oak leaves.",
        "How razorback jumping frogs can level six piqued gymnasts standing near the old stone mill.",
        "Pack my box with five dozen liquor jugs before the long voyage across the open sea.",
        "The early morning fog lifted slowly over the quiet valley and the sleeping town below it.",
        "Bright vixens jump while dozy fowl quack loudly as the farmer opens the heavy wooden gate.",
        "Sphinx of black quartz, judge my vow, said the weary knight to the silent evening sky.",
    ];
    let mut samples = Vec::new();
    let mut sr = 24_000u32;
    for text in sentences {
        let clip = kokoro_clip(text, "af_heart");
        sr = clip.sample_rate;
        samples.extend_from_slice(&clip.samples);
    }
    (samples, sr)
}

/// The sc-13380 DoD gate: the ported long-audio sliding-window segmentation + `merge_tokenized_
/// segments` tokenizes a **real >30 s** clip into one continuous 25 Hz stream. It asserts (a) the
/// token rate is 25 Hz across the *whole* duration (i.e. across the window seams — a naive per-window
/// concat would over-count the 4 s overlaps and inflate the rate), (b) every token is a valid FSQ
/// code in `[0, 6560]`, and (c) CONTINUITY: over the non-boundary interior the windowed stream
/// matches a single-pass tokenization of the same head audio, so the overlap is stitched without
/// duplication or gaps. A broken window plan, overlap-dedup, or merge would fail (a) (wrong rate) or
/// (c) (interior divergence). Reports the actual duration, token count, rate, and continuity fraction.
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn s3tokenizer_windows_audio_longer_than_30s() {
    use std::collections::HashSet;

    let tok = cb::S3Tokenizer::from_snapshot(&s3gen_snapshot())
        .expect("load the s3tokenizer from s3gen.safetensors");

    let (samples, sr) = long_kokoro_samples();
    let dur = samples.len() as f32 / sr as f32;
    assert!(
        dur > 30.5,
        "the long clip must exceed the 30 s window to exercise the sliding-window path; got {dur:.2}s"
    );

    let codes = tok.encode(&samples, sr).expect("tokenize the >30 s clip");
    let n = codes.len();
    let min = *codes.iter().min().unwrap();
    let max = *codes.iter().max().unwrap();
    let distinct = codes.iter().collect::<HashSet<_>>().len();
    let rate = n as f32 / dur;
    eprintln!(
        "s3tokenizer >30s (windowed): {n} tokens over {dur:.2}s = {rate:.2} Hz; \
         range [{min}, {max}], {distinct} distinct"
    );

    // (a) 25 Hz over the FULL duration, across the window seams (the merge de-duplicated the
    // overlaps; a naive concat of the per-window streams would run well above 25 Hz).
    assert!(
        (rate - 25.0).abs() < 1.5,
        "token rate {rate:.2} Hz is not ≈ 25 Hz over the full {dur:.2}s — the overlap was not \
         stitched correctly"
    );
    // (b) every token a valid FSQ code, and the stream is non-degenerate + deterministic.
    assert!(
        min >= 0 && max <= 6560,
        "tokens must be valid S3 FSQ codes in [0, 6560], got [{min}, {max}]"
    );
    assert!(
        distinct > 10,
        "tokens collapsed to {distinct} distinct value(s) — the encoder is not modeling speech"
    );
    let codes2 = tok.encode(&samples, sr).expect("re-tokenize");
    assert_eq!(
        codes, codes2,
        "the windowed tokenization must be deterministic"
    );

    // (c) CONTINUITY vs single-pass. The first ~29.5 s (≤30 s → single-pass) is the same leading
    // audio the first window sees; the merge keeps that window's head (its first 700 tokens = the
    // leading 28 s), so the windowed stream must agree with a single-pass tokenization over that
    // non-boundary interior — proof the overlap is stitched without duplication or gaps.
    let head_secs = 29.5f32;
    let head_len = ((head_secs * sr as f32) as usize).min(samples.len());
    let head = &samples[..head_len];
    let single = tok.encode(head, sr).expect("single-pass tokenize the head");
    // Compare the interior below the first-window seam (700) and below the single-pass clip's own
    // reflect-padded tail (its last few tokens see a boundary the full clip does not).
    let k = single.len().min(codes.len()).min(700).saturating_sub(8);
    assert!(
        k > 400,
        "continuity window too small ({k}) — the head is not long enough"
    );
    let agree = (0..k).filter(|&i| codes[i] == single[i]).count();
    let frac = agree as f32 / k as f32;
    let first_mismatch = (0..k).find(|&i| codes[i] != single[i]);
    eprintln!(
        "continuity: windowed vs single-pass agree on {agree}/{k} = {frac:.4} of the leading \
         interior tokens (first mismatch at {first_mismatch:?})"
    );
    assert!(
        frac > 0.9,
        "the windowed result diverges from single-pass over the non-boundary interior \
         ({frac:.4}) — the window/merge is not continuous"
    );
}

/// Provider integration: a `ReferenceAudio` request now derives T3's `cond_prompt_speech_tokens`
/// from the clip via the s3tokenizer (empty in sc-13222). The prompt is capped at the T3
/// `speech_cond_prompt_len` (150) and every token is a valid S3 code.
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn chatterbox_reference_audio_fills_the_t3_prompt_tokens() {
    let spec = staged_spec(s3gen_snapshot());
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

// ---------------------------------------------------------------------------------------------
// CAMPPlus speaker encoder (sc-13236): the D-TDNN x-vector network → 192-d speaker vector.
// ---------------------------------------------------------------------------------------------

/// The DoD gate: the ported CAMPPlus derives DISCRIMINATIVE 192-d x-vectors from real
/// `speaker_encoder.*` weights on distinct Kokoro voices — same-voice cosine materially exceeds
/// cross-voice cosine — and is deterministic. This is the sc-13236 clone-conditioning gate (a
/// broken fbank / TDNN / CAM / stats-pool / weight mapping degenerates to non-discriminative or
/// NaN vectors and fails here).
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn campplus_derives_discriminative_x_vectors() {
    use candle_audio_chatterbox::campplus::{cosine_similarity, SPK_EMBED_DIM, XVECTOR_DIM};

    let enc = cb::Campplus::from_snapshot(&s3gen_snapshot())
        .expect("load CAMPPlus from s3gen.safetensors");

    // Two clips of the SAME voice (different text) + one clip of a DIFFERENT voice.
    let a1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let a2 = kokoro_clip("She sells seashells by the seashore at dawn.", "af_heart");
    let b1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "am_michael",
    );

    let xa1 = enc.embed(&a1.samples, a1.sample_rate).expect("x-vector a1");
    let xa2 = enc.embed(&a2.samples, a2.sample_rate).expect("x-vector a2");
    let xb1 = enc.embed(&b1.samples, b1.sample_rate).expect("x-vector b1");

    assert_eq!(xa1.len(), XVECTOR_DIM, "x-vector must be 192-d");
    assert!(
        xa1.iter().chain(&xa2).chain(&xb1).all(|v| v.is_finite()),
        "x-vectors must be finite (no NaN/inf from the trunk)"
    );

    let same = cosine_similarity(&xa1, &xa2); // same voice, different text
    let cross_ab = cosine_similarity(&xa1, &xb1); // different voice
    let cross_a2b = cosine_similarity(&xa2, &xb1);
    eprintln!(
        "CAMPPlus x-vector cosine — same-voice(af_heart) = {same:.4}; \
         cross-voice(af_heart vs am_michael) = {cross_ab:.4} / {cross_a2b:.4}"
    );

    // Discrimination: the same-voice pair is materially closer than the cross-voice pairs.
    assert!(
        same > cross_ab + 0.05 && same > cross_a2b + 0.05,
        "x-vectors are not discriminative: same {same:.4} vs cross {cross_ab:.4}/{cross_a2b:.4}"
    );

    // Deterministic: same clip ⇒ byte-identical x-vector (the reproducibility law).
    let xa1_again = enc.embed(&a1.samples, a1.sample_rate).expect("re-embed a1");
    assert_eq!(xa1, xa1_again, "CAMPPlus must be deterministic");

    // The flow-ready 80-d speaker embedding (L2-norm + spk_embed_affine_layer 192→80) is well-formed
    // and still voice-discriminative.
    let fa1 = enc
        .spk_embed_flow(&a1.samples, a1.sample_rate)
        .expect("flow spk-embed a1");
    let fa2 = enc
        .spk_embed_flow(&a2.samples, a2.sample_rate)
        .expect("flow spk-embed a2");
    let fb1 = enc
        .spk_embed_flow(&b1.samples, b1.sample_rate)
        .expect("flow spk-embed b1");
    assert_eq!(
        fa1.len(),
        SPK_EMBED_DIM,
        "flow speaker embedding must be 80-d"
    );
    assert!(fa1.iter().all(|v| v.is_finite()));
    let same80 = cosine_similarity(&fa1, &fa2);
    let cross80 = cosine_similarity(&fa1, &fb1);
    eprintln!("flow 80-d spk-embed cosine — same {same80:.4} vs cross {cross80:.4}");
    assert!(
        same80 > cross80,
        "flow speaker embedding lost discrimination: same {same80:.4} vs cross {cross80:.4}"
    );
}

// ---------------------------------------------------------------------------------------------
// S3Gen flow (sc-13237): the CosyVoice flow-matching token→mel decoder.
// ---------------------------------------------------------------------------------------------

/// The sc-13237 DoD gate: the ported flow (UpsampleConformerEncoder + encoder_proj +
/// CausalConditionalCFM + ConditionalDecoder) renders a **sane 80-bin mel** from real `flow.*`
/// weights, with real conditioning derived end-to-end — the s3tokenizer's speech tokens, the
/// CAMPPlus 80-d flow speaker embedding, and the new 24 kHz prompt-mel front-end — of a Kokoro
/// reference clip. A broken encoder / rel-pos attention / CFM schedule / U-Net estimator / weight
/// mapping degenerates to a NaN, constant, or wrongly-shaped mel and fails here. It also asserts
/// the flow is deterministic under a fixed seed (the reproducibility law).
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn flow_synthesizes_a_sane_mel_from_speech_tokens() {
    let dir = s3gen_snapshot();
    let flow = cb::Flow::from_snapshot(&dir).expect("load the flow from s3gen.safetensors");
    let tok = cb::S3Tokenizer::from_snapshot(&dir).expect("load the s3tokenizer");
    let spk = cb::Campplus::from_snapshot(&dir).expect("load CAMPPlus");

    // Reference voice (prompt) and a target clip whose tokens we render in that voice.
    let reference = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let target = kokoro_clip("She sells seashells by the seashore at dawn.", "af_heart");

    // Real conditioning, derived exactly as the reference pipeline does.
    let prompt_tokens: Vec<u32> = tok
        .encode(&reference.samples, reference.sample_rate)
        .expect("tokenize the reference")
        .into_iter()
        .map(|c| c as u32)
        .collect();
    let prompt_mel = flow
        .mel_extractor()
        .mel(&reference.samples, reference.sample_rate, flow.device())
        .expect("prompt mel");
    let spk_embed = spk
        .spk_embed_flow(&reference.samples, reference.sample_rate)
        .expect("flow speaker embedding");
    let speech_tokens: Vec<u32> = tok
        .encode(&target.samples, target.sample_rate)
        .expect("tokenize the target")
        .into_iter()
        .map(|c| c as u32)
        .collect();

    let (pm_frames, _) = prompt_mel.dims2().expect("prompt mel dims");
    eprintln!(
        "flow inputs: {} prompt tokens, {} prompt-mel frames, {} speech tokens, spk {}-d",
        prompt_tokens.len(),
        pm_frames,
        speech_tokens.len(),
        spk_embed.len()
    );

    let mel = flow
        .inference(
            &speech_tokens,
            &prompt_tokens,
            &prompt_mel,
            &spk_embed,
            20260719,
        )
        .expect("flow token→mel inference");

    let (bins, frames) = mel.dims2().expect("mel must be [80, T]");
    let vals: Vec<f32> = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "flow mel: shape [{bins}, {frames}]; min {min:.4} max {max:.4} mean {mean:.4} std {std:.4}"
    );

    // Shape: 80 bins, ≈ 2 × speech tokens (token_mel_ratio = 2), within framing/alignment slack.
    assert_eq!(bins, 80, "mel must have 80 bins");
    let expected = 2 * speech_tokens.len();
    assert!(
        (frames as i64 - expected as i64).abs() <= 2,
        "mel frames {frames} not ≈ 2×{} = {expected}",
        speech_tokens.len()
    );
    // Sane: finite, non-degenerate (a collapsed decoder emits a constant or NaN mel), and in a
    // plausible log-mel range (the reference mel is log-compressed, roughly [-12, 3]).
    assert!(vals.iter().all(|v| v.is_finite()), "mel must be finite");
    assert!(
        std > 0.1,
        "mel is degenerate (std {std:.4}) — the decoder is not modeling speech"
    );
    assert!(
        (-30.0..=10.0).contains(&min) && (-30.0..=10.0).contains(&max),
        "mel values out of a plausible log-mel range: [{min:.4}, {max:.4}]"
    );

    // Deterministic under a fixed seed (the reproducibility law).
    let mel2 = flow
        .inference(
            &speech_tokens,
            &prompt_tokens,
            &prompt_mel,
            &spk_embed,
            20260719,
        )
        .expect("re-run the flow");
    let vals2: Vec<f32> = mel2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        vals, vals2,
        "the flow must be deterministic for a fixed seed"
    );
}

// ---------------------------------------------------------------------------------------------
// S3Gen HiFTNet vocoder (sc-13238): the NSF/iSTFT mel→waveform generator.
// ---------------------------------------------------------------------------------------------

/// The sc-13238 DoD gate: the ported HiFTNet vocoder (`ConvRNNF0Predictor` + `SourceModuleHnNSF`
/// NSF source + weight-normed upsample trunk + iSTFT head) turns a **real** 80-bin 24 kHz log-mel
/// (from [`cb::Mel24Extractor`] on a Kokoro reference clip) into a **non-silent, finite** 24 kHz
/// waveform of exactly `480 · n_mel_frames` samples, deterministically. A broken F0 predictor /
/// NSF source / weight-norm reconstruction / upsample math / iSTFT head degenerates to a NaN,
/// silent, or wrongly-sized waveform and fails here. (Vocoding a mel24 mel of real speech is the
/// vocoder's job in isolation — the full token→clone WAV is sc-13239's integration.)
#[test]
#[ignore = "real weights: needs s3gen.safetensors + a Kokoro snapshot; run with --ignored"]
fn hift_vocodes_a_real_mel_to_nonsilent_waveform() {
    let dir = s3gen_snapshot();
    let hift = cb::HiftGenerator::from_snapshot(&dir).expect("load the HiFTNet vocoder from s3gen");

    // A real 24 kHz reference clip → its real 80-bin log-mel (the vocoder's input distribution).
    let clip = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let mel_ext = cb::Mel24Extractor::default();
    let mel_nt = mel_ext
        .mel(&clip.samples, clip.sample_rate, hift.device())
        .expect("24 kHz log-mel of the reference"); // [n_frames, 80]
    let (n_frames, bins) = mel_nt.dims2().expect("mel dims");
    assert_eq!(bins, 80, "mel must have 80 bins");
    let mel = mel_nt.transpose(0, 1).unwrap().contiguous().unwrap(); // [80, n_frames]

    let seed = 20260719u64;
    let wav = hift.decode(&mel, seed).expect("vocode the mel");
    let samples: Vec<f32> = wav.to_vec1::<f32>().expect("waveform to host");

    let n = samples.len();
    let expected = 480 * n_frames;
    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    let sample_rate = cb::config::S3GEN_SR; // the vocoder always emits 24 kHz
    eprintln!(
        "hift waveform: {n} samples (= 480 × {n_frames} mel frames) @ {sample_rate} Hz; \
         peak {peak:.4}, rms {rms:.4}"
    );

    // Length is exactly 480 samples per mel frame → 24 kHz from the 50 Hz mel.
    assert_eq!(
        n, expected,
        "waveform must be 480 · n_frames = {expected} samples"
    );
    // Finite (no NaN/inf from the trunk, source, or iSTFT).
    assert!(
        samples.iter().all(|v| v.is_finite()),
        "waveform must be finite"
    );
    // Non-silent: a real speech mel must vocode to audible energy well above a noise floor.
    assert!(rms > 1e-3, "waveform is (near-)silent: rms {rms:.6}");
    assert!(
        peak > 1e-2,
        "waveform peak {peak:.6} is below an audible floor"
    );
    // Within the audio_limit clamp.
    assert!(
        peak <= 0.99 + 1e-6,
        "waveform exceeds the audio_limit clamp: {peak:.4}"
    );

    // Deterministic under a fixed seed (the reproducibility law).
    let wav2 = hift.decode(&mel, seed).expect("re-vocode");
    assert_eq!(
        samples,
        wav2.to_vec1::<f32>().unwrap(),
        "the vocoder must be deterministic for a fixed seed"
    );
}

// ---------------------------------------------------------------------------------------------
// PerTh implicit watermarker (sc-13240): the provenance watermark Chatterbox always applies.
// ---------------------------------------------------------------------------------------------

/// A deterministic, speech-like signal at `sample_rate`: a wandering pitch with a dozen harmonics
/// filling the ≈0–1.5 kHz watermark subband, amplitude modulation, and a little broadband dither —
/// rich enough for the spectral watermark to embed into, without a TTS dependency.
fn perth_test_signal(sample_rate: u32, secs: f32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = (sr * secs) as usize;
    let tau = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let pitch = 120.0 + 40.0 * (tau * 3.0 * t).sin();
            let mut s = 0.0f32;
            for h in 1..=12 {
                s += (1.0 / h as f32) * (tau * pitch * h as f32 * t).sin();
            }
            let env = 0.6 + 0.4 * (tau * 4.0 * t).sin();
            let dither = ((i.wrapping_mul(2_654_435_761) & 0xffff) as f32 / 65_535.0) - 0.5;
            0.3 * s * env + 0.02 * dither
        })
        .collect()
}

/// The sc-13240 DoD gate: the natively-ported PerTh implicit watermarker (real weights) embeds a
/// **recoverable** watermark that is **imperceptible**, exercised on Chatterbox's exact 24 kHz output
/// rate (embed resamples 24→32 kHz, watermarks, and resamples back). A watermarked signal is detected
/// with high confidence and is clearly separated from the un-watermarked signal; the watermarked
/// signal stays perceptually close to the original (high SNR). A broken encoder/decoder weight
/// mapping, magnitude normalization, magmask, multi-scale resample, or softmax-attention combine
/// collapses the detection separation and fails here. This is the watermarker in isolation; wiring it
/// into the clone Generator's `generate()` output is sc-13239.
#[test]
#[ignore = "real weights: perth_implicit.safetensors from CHATTERBOX_PERTH_SNAPSHOT (or the SceneWorks/perth-implicit hub pin); run with --ignored"]
fn perth_watermark_roundtrips_and_is_imperceptible() {
    let wm = cb::PerthWatermarker::from_safetensors(&perth_weights_file())
        .expect("load the converted PerTh weights");

    // (1) Native 32 kHz — the watermark's true imperceptibility, isolated from any resampling. Here
    //     the SNR is a faithful measure of the encoder residual alone (no resample roundtrip).
    let x32 = perth_test_signal(cb::PERTH_SR, 2.0);
    let clean32 = wm
        .get_watermark(&x32, cb::PERTH_SR)
        .expect("detect clean @32k");
    let marked32 = wm.embed(&x32, cb::PERTH_SR).expect("embed @32k");
    let conf32 = wm
        .get_watermark(&marked32, cb::PERTH_SR)
        .expect("detect watermarked @32k");
    // Compare over the interior (the outer STFT frames have partial window coverage).
    let g = 2048.min(x32.len() / 4);
    let n32 = x32.len().min(marked32.len());
    let snr32 = cb::snr_db(&x32[g..n32 - g], &marked32[g..n32 - g]);
    eprintln!(
        "perth @ 32000 Hz (native): clean={clean32:.4}  watermarked={conf32:.4}  SNR={snr32:.2} dB"
    );

    // (2) Chatterbox's exact 24 kHz output path — embed resamples 24→32 kHz, watermarks, and
    //     resamples back. Detection must survive the resample roundtrip (the real deployment case).
    let x24 = perth_test_signal(24_000, 2.0);
    let clean24 = wm.get_watermark(&x24, 24_000).expect("detect clean @24k");
    let marked24 = wm.embed(&x24, 24_000).expect("embed @24k");
    let conf24 = wm
        .get_watermark(&marked24, 24_000)
        .expect("detect watermarked @24k");
    let snr24 = cb::snr_db(&x24, &marked24);
    eprintln!(
        "perth @ 24000 Hz (Chatterbox path): clean={clean24:.4}  watermarked={conf24:.4}  \
         SNR={snr24:.2} dB  ({} samples)",
        marked24.len()
    );

    // Recoverable + clearly separated, at BOTH rates (a broken port collapses the separation).
    assert!(conf32 > 0.5, "watermark not recovered @32k: {conf32:.4}");
    assert!(conf24 > 0.5, "watermark not recovered @24k: {conf24:.4}");
    assert!(
        conf32 - clean32 > 0.25,
        "insufficient separation @32k: clean {clean32:.4} vs watermarked {conf32:.4}"
    );
    assert!(
        conf24 - clean24 > 0.25,
        "insufficient separation @24k: clean {clean24:.4} vs watermarked {conf24:.4}"
    );
    // Imperceptible ("implicit"): PerTh's transparency is psychoacoustic (the watermark is confined
    // to the masked low-frequency subband under an energy mask and trained with a psychoacoustic
    // loss), so its raw SNR is modest by design — the reference's own test only requires SNR > 0.
    // These floors (well above that bar, ~3–4 dB below the deterministic measured values ≈16/15 dB)
    // catch a port that grossly corrupts the audio. The native-rate figure isolates the watermark
    // from resample artifacts; the 24 kHz figure additionally carries the linear-resample roundtrip.
    assert!(
        snr32 > 12.0,
        "watermark not imperceptible @32k: SNR {snr32:.2} dB"
    );
    assert!(
        snr24 > 12.0,
        "24 kHz watermarked signal too degraded: SNR {snr24:.2} dB"
    );
}

// ---------------------------------------------------------------------------------------------
// The sc-12838 clone-WAV DoD gate (sc-13239): the epic-releasing end-to-end test.
// ---------------------------------------------------------------------------------------------

/// A `ReferenceAudio`-conditioned request: the reference clip drives BOTH T3 (via the ve vector the
/// provider derives internally) and S3Gen (prompt mel + prompt tokens + CAMPPlus x-vector).
fn reference_request(prompt: &str, reference: AudioTrack, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_string(),
        audio: Some(AudioParams {
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        conditioning: vec![Conditioning::ReferenceAudio {
            audio: reference,
            strength: None,
        }],
        seed: Some(seed),
        ..Default::default()
    }
}

/// **The sc-12838 clone gate — the make-or-break test this epic releases.** The FULL registry path
/// (`generate()` on the loaded provider) renders a real 24 kHz cloned-voice WAV from a reference clip
/// and text, and it must be voice-similar to the reference: the `chatterbox_ve` embedding of the
/// output is closer to the reference voice than to a DIFFERENT control voice, by a material margin.
/// The assertion FAILS if the clone ignores the reference (a voice-agnostic pipeline). It also gates
/// non-silence, finiteness, the 24 kHz rate, a token-proportional duration, speech-shaped energy, and
/// the always-applied PerTh provenance watermark.
///
/// Needs the FULL `CHATTERBOX_SNAPSHOT` (`s3gen.safetensors`), a Kokoro snapshot (reference/control
/// voices), and the `voice_embedding` + `perth` components staged via [`staged_spec`]
/// (`CHATTERBOX_VE_SNAPSHOT` / `CHATTERBOX_PERTH_SNAPSHOT`, each hub-fallback when unset).
#[test]
#[ignore = "real weights: needs the full chatterbox snapshot (s3gen.safetensors) + Kokoro + the ve/perth components (CHATTERBOX_VE_SNAPSHOT/CHATTERBOX_PERTH_SNAPSHOT, hub-fallback); run with --ignored"]
fn chatterbox_clones_a_reference_voice_end_to_end() {
    use candle_audio_chatterbox::campplus::cosine_similarity;

    let spec = staged_spec(s3gen_snapshot());
    let gen = cb::load(&spec).expect("load the chatterbox generator");

    // The reference voice (af_heart) and a DIFFERENT control voice (am_michael) — distinct Kokoro
    // voices so voice similarity is a genuine, falsifiable claim.
    let reference = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank at first light.",
        "af_heart",
    );
    let control = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank at first light.",
        "am_michael",
    );

    let text = "Hello there. This sentence is spoken in a cloned voice by the synthesizer.";
    let seed = 20260719u64;

    // The full provider path: reference clip + text → cloned WAV.
    let out = match gen
        .generate(
            &reference_request(text, reference.clone(), seed),
            &mut |_| {},
        )
        .expect("generate the cloned WAV")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    };

    let n = out.samples.len();
    let secs = n as f32 / out.sample_rate as f32;
    let peak = out.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rms = (out.samples.iter().map(|v| v * v).sum::<f32>() / n.max(1) as f32).sqrt();

    // Token-proportional expected duration: mel_len2 ≈ 2·speech_tokens, hift emits 480 samples/frame
    // → duration ≈ speech_tokens/25 s (the 25 Hz token rate).
    let (_, real_tokens) = gen_speech_tokens(&spec, text, reference.clone(), seed);
    let expected_secs = real_tokens.len() as f32 / 25.0;

    eprintln!(
        "clone WAV: {n} samples @ {} Hz = {secs:.2}s (T3 {} tokens → expected ≈ {expected_secs:.2}s); \
         peak {peak:.4}, rms {rms:.4}",
        out.sample_rate,
        real_tokens.len(),
    );

    // Shape / sanity.
    assert_eq!(out.sample_rate, 24_000, "clone must be 24 kHz");
    assert_eq!(out.channels, 1, "clone must be mono");
    assert!(
        out.samples.iter().all(|v| v.is_finite()),
        "clone must be finite"
    );
    assert!(rms > 1e-3, "clone is (near-)silent: rms {rms:.6}");
    assert!(peak > 1e-2, "clone peak {peak:.6} below an audible floor");
    assert!(
        peak <= 0.99 + 1e-6,
        "clone exceeds the audio_limit clamp: {peak:.4}"
    );
    assert!(
        (secs - expected_secs).abs() <= 0.25 * expected_secs + 0.2,
        "clone duration {secs:.2}s not ≈ token-proportional {expected_secs:.2}s"
    );
    // Speech-shaped: not a DC offset / constant tone — the waveform crosses zero many times.
    let zero_crossings = out
        .samples
        .windows(2)
        .filter(|w| (w[0] <= 0.0) != (w[1] <= 0.0))
        .count();
    let zcr = zero_crossings as f32 / secs;
    eprintln!("clone zero-crossing rate: {zcr:.1} Hz");
    assert!(
        (200.0..6000.0).contains(&zcr),
        "zero-crossing rate {zcr:.1} Hz is not speech-shaped"
    );

    // VOICE SIMILARITY — the crux. cos(ve(out), ve(reference)) must beat cos(ve(out), ve(control)).
    let ve_out = voice_embedding(&out);
    let ve_ref = voice_embedding(&reference);
    let ve_ctrl = voice_embedding(&control);
    let cos_ref = cosine_similarity(&ve_out, &ve_ref);
    let cos_ctrl = cosine_similarity(&ve_out, &ve_ctrl);
    let margin = cos_ref - cos_ctrl;
    eprintln!(
        "voice similarity: cos(out, reference af_heart) = {cos_ref:.4}; \
         cos(out, control am_michael) = {cos_ctrl:.4}; margin = {margin:.4}"
    );
    // Floor hardened 0.02 → 0.15 (sc-13443): comfortably below the observed +0.318 margin, well
    // above a voice-agnostic ≈0, so it catches partial-regression drift without going flaky.
    assert!(
        margin > 0.15,
        "clone is not voice-similar to the reference: cos_ref {cos_ref:.4} vs cos_ctrl \
         {cos_ctrl:.4} (margin {margin:.4}) — the clone IGNORED the reference voice"
    );

    // Provenance watermark: always applied, detected with high confidence.
    let wm = cb::PerthWatermarker::from_safetensors(&perth_weights_file())
        .expect("load PerTh watermarker");
    let conf = wm
        .get_watermark(&out.samples, out.sample_rate)
        .expect("detect the watermark");
    eprintln!("PerTh watermark confidence on the clone: {conf:.4}");
    assert!(
        conf > 0.5,
        "the clone is not watermarked (confidence {conf:.4}) — generate() must always watermark"
    );

    // Demo WAV to the scratchpad (or CHATTERBOX_WAV_OUT).
    let out_path = std::env::var("CHATTERBOX_WAV_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("chatterbox_clone_demo.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &out).expect("write the demo clone WAV");
    eprintln!("wrote cloned-voice demo WAV: {}", out_path.display());
}

/// The T3 speech tokens for a `ReferenceAudio` request (a fresh generator so the DoD test's `dyn`
/// generator stays untouched), used to compute the token-proportional expected duration.
fn gen_speech_tokens(
    spec: &LoadSpec,
    text: &str,
    reference: AudioTrack,
    seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    let gen = cb::load_generator(spec).expect("load the chatterbox generator");
    gen.speech_tokens(&reference_request(text, reference, seed), &mut |_| {})
        .expect("T3 speech tokens")
}
