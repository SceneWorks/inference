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
//! - [`moss_ttsd_renders_multi_speaker_audio`] — the **full acoustic multi-speaker DoD** (sc-13518):
//!   the 2-speaker script + seed → ONE `AudioTrack` that (a) is non-silent, finite, 24 kHz, plausible
//!   duration; (b) renders the two speakers in **different voices**, measured via
//!   `candle-audio-chatterbox-ve` — the cross-segment speaker-embedding cosine is materially below
//!   the same-speaker self-similarity; (c) a single-voice control works; (d) is byte-identical on a
//!   re-synth for the seed.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-moss-tts --test conformance -- --ignored --nocapture
//! ```
//! Set `MOSS_TTSD_SNAPSHOT` to the AR snapshot dir (`config.json`, `model.safetensors`,
//! `tokenizer.json`), or leave unset to resolve the pinned snapshot via the hub (~4.1 GB). The
//! XY_Tokenizer codec (~2.1 GB) resolves via the hub or `MOSS_XY_TOKENIZER_SNAPSHOT`; the
//! `chatterbox_ve` weights via the hub or `CHATTERBOX_VE_SNAPSHOT`. Optionally dump the raw frames
//! with `MOSS_TTSD_FRAMES_OUT` and a demo WAV with `MOSS_TTSD_WAV_OUT`.

use std::collections::HashSet;
use std::path::PathBuf;

use candle_audio_moss_tts as moss;
use candle_audio_moss_tts::gen_core::{
    AudioParams, AudioTrack, GenerationRequest, Generator, LoadSpec, SpeechSegment, WeightsSource,
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

/// The `chatterbox_ve` 256-d speaker embedding of a waveform slice (any sample rate; the encoder
/// resamples internally). Built once per call — the DoD only embeds a handful of segments.
fn ve_embed(samples: &[f32]) -> Vec<f32> {
    let spec = LoadSpec::new(match std::env::var("CHATTERBOX_VE_SNAPSHOT") {
        Ok(dir) => {
            WeightsSource::File(PathBuf::from(dir).join(candle_audio_chatterbox_ve::WEIGHTS_FILE))
        }
        Err(_) => candle_audio_chatterbox_ve::resolve_pinned_file()
            .expect("resolve the pinned chatterbox_ve weights"),
    });
    let embedder = candle_audio_chatterbox_ve::load(&spec).expect("load chatterbox_ve");
    let track = AudioTrack {
        samples: samples.to_vec(),
        sample_rate: moss::codec::SAMPLE_RATE,
        channels: 1,
        stems: Vec::new(),
    };
    embedder.embed(&track).expect("embed segment")
}

fn wav_stats(samples: &[f32]) -> (f32, f32) {
    let n = samples.len().max(1) as f32;
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / n).sqrt();
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    (peak, rms)
}

/// The full acoustic multi-speaker DoD (sc-13518): a 2-speaker script renders one real 24 kHz track
/// whose two speakers are acoustically distinct (measured via `chatterbox_ve`), with a working
/// single-voice control and byte-identical re-synth for the seed.
#[test]
#[ignore = "real weights: needs MOSS-TTSD-v0.5 (~4.1 GB) + XY_Tokenizer (~2.1 GB) + chatterbox_ve"]
fn moss_ttsd_renders_multi_speaker_audio() {
    let gen = load();

    // (a) One dialogue track: non-silent, finite, 24 kHz, mono, plausible duration.
    let secs = 6.0f32;
    let out = gen
        .generate(&two_speaker(secs), &mut |_| {})
        .expect("multi-speaker synthesis");
    let track = match out {
        candle_audio_moss_tts::gen_core::GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio output, got {other:?}"),
    };
    assert_eq!(track.sample_rate, 24_000, "24 kHz output");
    assert_eq!(track.channels, 1, "mono");
    assert!(!track.samples.is_empty(), "non-empty track");
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "all samples finite"
    );
    let (peak, rms) = wav_stats(&track.samples);
    let dur = track.samples.len() as f32 / 24_000.0;
    eprintln!(
        "2-speaker track: {} samples ({dur:.2}s), peak={peak:.4}, rms={rms:.5}",
        track.samples.len()
    );
    assert!(rms > 1e-3, "track must be non-silent (rms={rms})");
    assert!(peak <= 1.5, "track must not clip absurdly (peak={peak})");
    // The delay-pattern loop stops at EOS, so the rendered clip is <= the budget but a real,
    // multi-second utterance — sanity-bound it well away from empty.
    assert!(
        dur > 0.5 && dur <= secs + 1.0,
        "duration {dur}s implausible for a {secs}s budget"
    );

    // (b) Voice distinctness: split the dialogue into its first / second half (S1 then S2), embed
    // each with chatterbox_ve, and require the cross-speaker cosine to sit materially BELOW the
    // same-speaker self-similarity (each half split again into quarters).
    let half = track.samples.len() / 2;
    let (s1, s2) = track.samples.split_at(half);
    let q1a = &s1[..s1.len() / 2];
    let q1b = &s1[s1.len() / 2..];
    let q2a = &s2[..s2.len() / 2];
    let q2b = &s2[s2.len() / 2..];

    let e_s1 = ve_embed(s1);
    let e_s2 = ve_embed(s2);
    let cross = candle_audio_chatterbox_ve::cosine_similarity(&e_s1, &e_s2);

    // Same-speaker self-similarity within each half (two quarters of the same voice).
    let self_s1 = candle_audio_chatterbox_ve::cosine_similarity(&ve_embed(q1a), &ve_embed(q1b));
    let self_s2 = candle_audio_chatterbox_ve::cosine_similarity(&ve_embed(q2a), &ve_embed(q2b));
    let self_sim = 0.5 * (self_s1 + self_s2);
    eprintln!(
        "voice distinctness — cross-speaker cosine = {cross:.4}; self-similarity S1 = {self_s1:.4}, \
         S2 = {self_s2:.4} (mean {self_sim:.4})"
    );
    assert!(
        cross < self_sim - 0.05,
        "the two speakers must be acoustically distinct: cross-speaker cosine {cross:.4} is not \
         materially below same-speaker self-similarity {self_sim:.4}"
    );

    // (c) Single-voice control (no script) renders a valid non-silent track.
    let ctrl = gen
        .generate(&single_voice(3.0), &mut |_| {})
        .expect("single-voice control synthesis");
    let ctrl = match ctrl {
        candle_audio_moss_tts::gen_core::GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio output, got {other:?}"),
    };
    let (_, ctrl_rms) = wav_stats(&ctrl.samples);
    assert!(
        !ctrl.samples.is_empty() && ctrl_rms > 1e-3,
        "single-voice control must render non-silent audio (rms={ctrl_rms})"
    );

    // (d) Deterministic: the same request + seed re-synthesizes byte-identical audio.
    let again = gen
        .generate(&two_speaker(secs), &mut |_| {})
        .expect("re-synthesis");
    let again = match again {
        candle_audio_moss_tts::gen_core::GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio output, got {other:?}"),
    };
    assert_eq!(
        track.samples, again.samples,
        "seeded synthesis must be byte-identical run-to-run"
    );

    if let Ok(out) = std::env::var("MOSS_TTSD_WAV_OUT") {
        candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out), &track)
            .expect("write demo WAV");
        eprintln!("wrote demo WAV to {out}");
    }
}

/// Task 12906 (non-English): MOSS-TTSD reads text in-band (no external G2P), advertising ~20
/// languages. A Chinese prompt synthesizes real, non-silent 24 kHz audio whose waveform genuinely
/// differs from the English control at the same seed — the model responds to the non-English script.
#[test]
#[ignore = "real weights: needs MOSS-TTSD-v0.5 (~4.1 GB) + XY_Tokenizer (~2.1 GB)"]
fn moss_ttsd_renders_non_english() {
    let gen = load();
    let zh = GenerationRequest {
        prompt: "你好，今天天气怎么样？我很好，谢谢你的关心。".to_string(),
        audio: Some(AudioParams {
            target_duration: Some(5.0),
            language: Some("zh".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        seed: Some(20_260_720),
        ..Default::default()
    };
    let out = gen.generate(&zh, &mut |_| {}).expect("zh synthesis");
    let track = match out {
        candle_audio_moss_tts::gen_core::GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio output, got {other:?}"),
    };
    let (peak, rms) = wav_stats(&track.samples);
    let dur = track.samples.len() as f32 / 24_000.0;
    eprintln!(
        "zh track: {} samples ({dur:.2}s), peak={peak:.4}, rms={rms:.5}",
        track.samples.len()
    );
    assert_eq!(track.sample_rate, 24_000);
    assert!(
        !track.samples.is_empty() && track.samples.iter().all(|s| s.is_finite()) && rms > 1e-3,
        "the Chinese prompt must render non-silent, finite audio (rms={rms})"
    );

    // The non-English script genuinely drives generation: its waveform differs from the English
    // single-voice control at the same seed.
    let en = match gen
        .generate(&single_voice(5.0), &mut |_| {})
        .expect("en control")
    {
        candle_audio_moss_tts::gen_core::GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio output, got {other:?}"),
    };
    assert_ne!(
        track.samples, en.samples,
        "a Chinese prompt must produce different audio than an English control"
    );

    if let Ok(out) = std::env::var("MOSS_TTSD_ZH_WAV_OUT") {
        candle_audio::wav::write_wav_pcm16(std::path::Path::new(&out), &track)
            .expect("write zh demo WAV");
        eprintln!("wrote zh demo WAV to {out}");
    }
}

/// Codec-parity harness: decode a fixed set of 8-codebook frames (`MOSS_TTSD_CODES_IN`, one frame
/// per line as 8 comma-separated code ids) through the candle XY_Tokenizer codec **only** and write
/// the raw little-endian `f32` mono waveform to `MOSS_TTSD_PARITY_OUT`. The Python reference decodes
/// the same frames with the upstream `XY_Tokenizer`; a companion script compares the two waveforms
/// (cosine / max-abs-diff). Not an assertion — an evidence generator.
#[test]
#[ignore = "codec parity: needs the ~2.1 GB XY_Tokenizer checkpoint + a MOSS_TTSD_CODES_IN file"]
fn codec_decode_frames_from_file() {
    let codes_path = std::env::var("MOSS_TTSD_CODES_IN").expect("set MOSS_TTSD_CODES_IN");
    let out_path = std::env::var("MOSS_TTSD_PARITY_OUT").expect("set MOSS_TTSD_PARITY_OUT");
    let text = std::fs::read_to_string(&codes_path).expect("read codes file");
    let frames: Vec<Vec<u32>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|s| s.trim().parse::<u32>().expect("code id"))
                .collect()
        })
        .collect();
    eprintln!("decoding {} frames through the candle codec", frames.len());
    let ckpt = moss::resolve_pinned_codec_checkpoint().expect("resolve XY_Tokenizer checkpoint");
    let codec = moss::codec::XyTokenizerCodec::load(&ckpt).expect("load codec");
    let wav = codec
        .decode_frames(&frames, &|| false)
        .expect("decode")
        .expect("not canceled");
    let mut bytes = Vec::with_capacity(wav.len() * 4);
    for s in &wav {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes).expect("write parity waveform");
    let (peak, rms) = wav_stats(&wav);
    eprintln!(
        "candle codec: {} samples, peak={peak:.4}, rms={rms:.5} -> {out_path}",
        wav.len()
    );
}
