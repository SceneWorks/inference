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
//! - [`moss_tts_realtime_asr_roundtrip_fidelity`] — the sc-13433 **text-fidelity** gate: a curated
//!   fixed prompt set is synthesized at the shipped sampling default and transcribed back with
//!   `whisper_base`; each transcript must match its prompt within a character-error-rate bound (and
//!   the mean CER within a tighter one). This is the ASR round-trip regression gate for
//!   prompt-following — a model that regressed to silence / an unrelated utterance (the pre-sc-13433
//!   spurious early-EOS failure) blows past the CER bound. It also asserts the metric discriminates
//!   (an unrelated reference does *not* pass the same bound).
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-moss-tts-realtime --test conformance -- --ignored --nocapture
//! ```
//! Set `MOSS_TTS_REALTIME_SNAPSHOT` to the AR snapshot dir (~4.66 GB, holding `config.json`,
//! `model.safetensors`, `tokenizer.json`) — **required**, a passed-in path: inference never
//! self-fetches or derives a cache location (epic 13657). The MOSS-Audio-Tokenizer codec (~7.1 GB) is
//! likewise a passed-in component (sc-13662): it is **required** from `MOSS_AUDIO_TOKENIZER_SNAPSHOT`
//! (the codec snapshot dir, `config.json` + `model*.safetensors`) — the provider never self-fetches
//! it, so this must point at a materialized snapshot. The demo WAV path is `MOSS_TTS_REALTIME_WAV_OUT`
//! (default temp dir). The fidelity gate additionally uses `whisper_base` — **required** from
//! `WHISPER_SNAPSHOT` (the ~150 MB snapshot dir), also a passed-in path, never a hub fetch.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_audio_moss_tts_realtime as moss;
use candle_audio_moss_tts_realtime::gen_core::{
    AudioChunk, AudioParams, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    WeightsSource,
};

/// Resolve a MOSS-TTS-Realtime snapshot dir from the required `MOSS_TTS_REALTIME_SNAPSHOT` env (a
/// passed-in AR snapshot dir). Inference never self-fetches or derives a cache location (epic 13657).
fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("MOSS_TTS_REALTIME_SNAPSHOT").expect(
        "set MOSS_TTS_REALTIME_SNAPSHOT to a MOSS-TTS-Realtime AR snapshot dir (config.json + model.safetensors + tokenizer)",
    ))
}

/// The MOSS-Audio-Tokenizer codec snapshot directory, staged as the passed-in `codec` component
/// (sc-13662, epic 13657). Resolved from `MOSS_AUDIO_TOKENIZER_SNAPSHOT`. Required: the provider no
/// longer self-fetches the codec, so the real-weight harness must point at a materialized snapshot.
fn codec_dir() -> PathBuf {
    PathBuf::from(std::env::var("MOSS_AUDIO_TOKENIZER_SNAPSHOT").expect(
        "set MOSS_AUDIO_TOKENIZER_SNAPSHOT to the MOSS-Audio-Tokenizer codec snapshot dir (the codec \
         is now a passed-in component, sc-13662)",
    ))
}

/// The codec staged as a `codec` component source (a snapshot directory).
fn codec_component() -> WeightsSource {
    WeightsSource::Dir(codec_dir())
}

/// A `LoadSpec` for the AR snapshot with the required `codec` component staged (sc-13662).
fn spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_component(moss::CODEC_COMPONENT_ID, codec_component())
}

fn load() -> moss::model::MossTtsRealtimeGenerator {
    moss::load_generator(&spec()).expect("load the MOSS-TTS-Realtime generator")
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
    let spec = spec();
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
    let dir = codec_dir();
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

// ---------------------------------------------------------------------------------------------
// sc-13433 — ASR round-trip text-fidelity gate (prompt → MOSS-TTS-Realtime → whisper_base → CER).
// ---------------------------------------------------------------------------------------------

/// Curated fixed prompt set the shipped sampling default renders faithfully (measured on real
/// weights). The last three are the **sc-13570 regression guard**: under the old chat-completion
/// prompt (a fabricated `<|im_start|>user…` turn + `text_pad`-only generation) they collapsed to
/// silence / babble / an unrelated word ("bye"); with the reference delay-pattern conditioning
/// restored ([`moss::decode::build_prompt_frames`]) they render the full sentence at CER ≈ 0.00 with
/// **no** minimum-length floor. `The weather…` was the sc-13433 floor's showcase; it now renders
/// faithfully from conditioning alone (CER ≈ 0.00, floor off). A future regression to silence / an
/// unrelated utterance — from either a conditioning or a codec break — drives every prompt's CER
/// past the bound.
const FIDELITY_PROMPTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "The train arrives at nine in the morning.",
    "The weather is very nice this afternoon.",
    "Please remember to buy milk and bread today.",
    // sc-13570 — previously silent/babble under the old conditioning, now CER ≈ 0.00.
    "Hello, this is a streaming text to speech test.",
    "I would like a cup of coffee please.",
    "Thank you very much for your help.",
    // sc-13570 — a > DELAY_TOKENS_LEN (18-token) prompt: exercises the delay-pattern *streaming*
    // path (text tokens fed one-per-frame during the AR loop), which the short prompts above do not.
    "Welcome to the world of streaming text to speech, where every sentence flows naturally and clearly.",
];

/// An utterance no FIDELITY_PROMPT transcribes to — used to prove the CER bound discriminates
/// (a faithful transcript must NOT match this within the same bound).
const UNRELATED_DECOY: &str = "the stock market fell sharply on tuesday afternoon";

/// Per-prompt CER ceiling. Measured faithful transcripts sit at 0.00–0.14 (whisper's `nine`→`9` and
/// `please`→`the peas` account for the non-zero ones); silence / unrelated-utterance regressions
/// measure ≥ 0.72. 0.35 sits in that wide gap with margin on both sides.
const MAX_PROMPT_CER: f32 = 0.35;
/// Mean-CER ceiling across the set (tighter than the per-prompt bound — the set as a whole must be
/// faithful, not merely each prompt individually under the loose per-prompt cap).
const MAX_MEAN_CER: f32 = 0.20;

/// Normalize transcript/reference text for CER: lowercase, strip punctuation, collapse whitespace.
fn normalize(s: &str) -> String {
    let cleaned: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Character error rate = Levenshtein(reference, hypothesis) / reference.len() (the same metric the
/// `candle-audio-whisper` Kokoro round-trip uses).
fn character_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut curr = vec![0usize; h.len() + 1];
    for (i, &rc) in r.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &hc) in h.iter().enumerate() {
            let cost = usize::from(rc != hc);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[h.len()] as f32 / r.len() as f32
}

/// A fidelity request: a generous target duration so audio-EOS terminates the sentence naturally
/// (short budgets truncate the utterance — a measurement artifact, not a fidelity failure).
fn fidelity_request(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_string(),
        audio: Some(AudioParams {
            target_duration: Some(8.0),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        seed: Some(20260719),
        ..Default::default()
    }
}

/// The text-fidelity regression gate (sc-13433, extended by sc-13570). Synthesizes the curated
/// prompt set at the **shipped sampling default** (reference temperature 0.8, reference delay-pattern
/// conditioning, no min-length floor — no env overrides), transcribes each clip with `whisper_base`,
/// and asserts prompt-following within a CER bound. Guards against the pre-sc-13433 failure mode (a
/// full sentence collapsing to a sub-second spurious-EOS fragment / silence), the sc-13570 failure
/// mode (prompt-specific silence/babble from the old chat-completion conditioning — the last three
/// prompts), and any future regression to unrelated speech.
#[test]
#[ignore = "real weights: needs the MOSS-TTS-Realtime AR + codec + whisper_base snapshots; run with --ignored --nocapture"]
fn moss_tts_realtime_asr_roundtrip_fidelity() {
    use candle_audio_whisper::gen_core::{
        AudioTrack as WAudioTrack, LoadSpec as WLoadSpec, TimestampGranularity, TranscribeOptions,
        TranscribeRequest, TranscribeTask, WeightsSource as WWeightsSource,
    };

    // The registered generator at the shipped sampling default (`generate` drives the shared
    // synthesis path). `load()` resolves the pinned AR snapshot (or MOSS_TTS_REALTIME_SNAPSHOT).
    let generator = load();

    // whisper_base transcriber (pinned ~150 MB snapshot or WHISPER_SNAPSHOT).
    let wspec = WLoadSpec::new(WWeightsSource::Dir(PathBuf::from(
        std::env::var("WHISPER_SNAPSHOT")
            .expect("set WHISPER_SNAPSHOT to an openai/whisper-base snapshot dir"),
    )));
    let transcriber = candle_audio_whisper::provider_registry()
        .expect("whisper registry")
        .load_transcriber(candle_audio_whisper::MODEL_ID, &wspec)
        .expect("whisper_base loads through the explicit registry");

    let mut cers: Vec<f32> = Vec::new();
    let mut first_transcript = String::new();
    for prompt in FIDELITY_PROMPTS {
        let track = match generator
            .generate(&fidelity_request(prompt), &mut |_| {})
            .expect("moss_tts_realtime generate")
        {
            GenerationOutput::Audio(t) => t,
            other => panic!("expected GenerationOutput::Audio, got {other:?}"),
        };
        assert!(!track.samples.is_empty(), "empty audio for {prompt:?}");

        let treq = TranscribeRequest {
            audio: WAudioTrack {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                ..Default::default()
            },
            options: TranscribeOptions {
                language: Some("en".into()),
                task: TranscribeTask::Transcribe,
                timestamps: TimestampGranularity::None,
            },
            ..Default::default()
        };
        let out = transcriber
            .transcribe(&treq, &mut |_| {})
            .expect("whisper transcribe");
        let hyp = normalize(&out.text);
        let refn = normalize(prompt);
        let cer = character_error_rate(&refn, &hyp);
        let secs = track.samples.len() as f32 / track.sample_rate as f32;
        println!("fidelity: prompt={refn:?} transcript={hyp:?} CER={cer:.3} ({secs:.2}s audio)");
        assert!(
            !hyp.trim().is_empty(),
            "empty transcript for {prompt:?} — the model produced nothing intelligible"
        );
        assert!(
            cer <= MAX_PROMPT_CER,
            "CER {cer:.3} > {MAX_PROMPT_CER} for {prompt:?}: transcript {hyp:?} does not follow the \
             prompt (a spurious early-EOS fragment / silence / unrelated utterance fails here)"
        );
        if first_transcript.is_empty() {
            first_transcript = hyp;
        }
        cers.push(cer);
    }

    let mean = cers.iter().sum::<f32>() / cers.len() as f32;
    println!(
        "fidelity: mean CER {mean:.3} over {} prompts (per-prompt cap {MAX_PROMPT_CER}, mean cap {MAX_MEAN_CER})",
        cers.len()
    );
    assert!(
        mean <= MAX_MEAN_CER,
        "mean CER {mean:.3} > {MAX_MEAN_CER} — the prompt set as a whole is not being followed"
    );

    // Discrimination: the same (passing) transcript must NOT match an unrelated reference within the
    // bound — proof the CER threshold distinguishes right-words from wrong-words, so a model that
    // regressed to an unrelated utterance could not slip through.
    let decoy_cer = character_error_rate(&normalize(UNRELATED_DECOY), &first_transcript);
    assert!(
        decoy_cer > MAX_PROMPT_CER,
        "discrimination failed: a faithful transcript {first_transcript:?} scored CER {decoy_cer:.3} \
         against the unrelated decoy — the bound {MAX_PROMPT_CER} is too loose to be meaningful"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-14148 — the MOSS-Audio-Tokenizer ENCODER (waveform → RVQ codes), the analysis direction that
// voice cloning (sc-14149) needs. Two gates: a self-contained encode→decode round-trip, and — when
// the reference outputs are provisioned — a strong codebook-0 cross-check against the reference
// PyTorch `codec.encode` on a byte-identical clip.
// ---------------------------------------------------------------------------------------------

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let (mut da, mut db) = (0.0f32, 0.0f32);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn read_f32le(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read clip.f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read a codes CSV (`frames[T][nq]`, one comma-separated row per frame).
fn read_codes_csv(path: &str) -> Vec<Vec<u32>> {
    std::fs::read_to_string(path)
        .expect("read codes csv")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().parse().unwrap()).collect())
        .collect()
}

#[test]
#[ignore = "real weights: needs the ~7.1 GB MOSS-Audio-Tokenizer codec snapshot; run with --ignored --nocapture"]
fn moss_audio_codec_encode_roundtrip_and_reference() {
    use candle_audio_moss_tts_realtime::codec::MossAudioCodec;
    let codec = MossAudioCodec::load(&codec_dir(), 16).expect("load codec");

    // Strong cross-check (when provisioned): the port's encode codebook-0 must match the reference
    // PyTorch `codec.encode` on a byte-identical clip. `MOSS_CODEC_CLIP` = raw f32-LE mono samples at
    // 24 kHz; `MOSS_CODEC_REF_CODES` = the reference codes CSV (`frames[T][16]`).
    if let (Ok(clip_p), Ok(ref_p)) = (
        std::env::var("MOSS_CODEC_CLIP"),
        std::env::var("MOSS_CODEC_REF_CODES"),
    ) {
        let clip = read_f32le(&clip_p);
        let port = codec.encode(&clip, 24_000).expect("encode reference clip");
        let refc = read_codes_csv(&ref_p);
        let n = port.len().min(refc.len());
        assert!(n > 0, "no frames to compare");
        let (mut cb0, mut allm, mut tot) = (0usize, 0usize, 0usize);
        for f in 0..n {
            if port[f][0] == refc[f][0] {
                cb0 += 1;
            }
            for q in 0..16 {
                tot += 1;
                if port[f].get(q) == refc[f].get(q) {
                    allm += 1;
                }
            }
        }
        let cb0_rate = cb0 as f32 / n as f32;
        let all_rate = allm as f32 / tot as f32;
        println!(
            "codec ref cross-check: port {} vs ref {} frames (cmp {n}); cb0 agree {cb0_rate:.3}, \
             all-cb agree {all_rate:.3}",
            port.len(),
            refc.len()
        );
        assert_eq!(
            port.len(),
            refc.len(),
            "frame count must match the reference"
        );
        // The port matches the reference codec.encode exactly (measured 1.000 across all 16
        // codebooks); the bounds sit just under that to allow only cross-platform argmax tie noise, so
        // a real regression on codebook-0 OR any higher quantizer fails here.
        assert!(
            cb0_rate >= 0.99,
            "port encode codebook-0 must match the reference encoder (agree {cb0_rate:.3})"
        );
        assert!(
            all_rate >= 0.98,
            "port encode must match the reference across all 16 codebooks (agree {all_rate:.3})"
        );
    } else {
        println!(
            "codec ref cross-check SKIPPED (set MOSS_CODEC_CLIP + MOSS_CODEC_REF_CODES to enable)"
        );
    }

    // Self-contained round-trip: a real codec waveform (decode a fixed pseudo-random 40-frame pattern)
    // → encode → decode must reconstruct it — the encoder emits decodable, faithful codes.
    let mut state: u32 = 1;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) % 1024
    };
    let frames0: Vec<Vec<u32>> = (0..40).map(|_| (0..16).map(|_| next()).collect()).collect();
    let w0 = codec
        .decode_frames(&frames0, &|| false)
        .unwrap()
        .expect("decode w0");
    let codes1 = codec.encode(&w0, 24_000).expect("encode w0");
    assert_eq!(
        codes1.len(),
        frames0.len(),
        "encode preserves the frame count (1 frame per {} samples)",
        moss::codec::DOWNSAMPLE_RATE,
    );
    assert!(
        codes1.iter().all(|f| f.len() == 16),
        "16 codebook codes per frame"
    );
    assert!(
        codes1.iter().flatten().all(|&c| c < 1024),
        "codes in the codebook range [0, 1024)"
    );
    let w1 = codec
        .decode_frames(&codes1, &|| false)
        .unwrap()
        .expect("decode w1");
    let corr = pearson(&w0, &w1);
    let rms1 = (w1.iter().map(|s| s * s).sum::<f32>() / w1.len().max(1) as f32).sqrt();
    println!(
        "codec round-trip: {} frames, corr(w0,w1)={corr:.3}, rms_out={rms1:.4}",
        codes1.len()
    );
    assert!(
        rms1 > 1e-3,
        "encode→decode round-trip is silent (rms {rms1:.5})"
    );
    assert!(
        corr > 0.3,
        "encode→decode must reconstruct the codec waveform (corr {corr:.3})"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-14181 — chunked/streaming encode for long reference clips. The first analysis stage runs at
// ~100 fps, so a single-shot encode materializes a `[1, H, T, T]` attention that is quadratic in the
// clip length (a 60 s clip → T ≈ 6000 → multi-GB per layer). The streaming path bounds that to
// `[1, H, chunk, chunk + context]` per layer. This gate asserts the two paths emit **identical
// codes** on a real ≥ 30 s clip, and that the streaming path's peak RSS sits well below single-shot's.
// ---------------------------------------------------------------------------------------------

/// Current resident-set size of this process, in bytes, via `ps -o rss=` (KiB on both macOS and
/// Linux). Unlike `getrusage`'s `ru_maxrss` — a monotonic high-water mark the 7 GB codec load already
/// pins far above any encode transient — this reads the *instantaneous* RSS, so a sampler can catch
/// the transient attention spike. `0` if the probe fails (the memory assertion then no-ops).
fn current_rss_bytes() -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .unwrap_or(0)
}

/// Run `f` while a background thread samples [`current_rss_bytes`] every few ms, and return
/// `(result, peak_rss_during_f)`. Used to measure each encode path's transient memory spike.
fn with_peak_rss<T>(f: impl FnOnce() -> T) -> (T, u64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let (s, p) = (Arc::clone(&stop), Arc::clone(&peak));
    let sampler = std::thread::spawn(move || {
        while !s.load(Ordering::Relaxed) {
            p.fetch_max(current_rss_bytes(), Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let out = f();
    peak.fetch_max(current_rss_bytes(), Ordering::Relaxed);
    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("rss sampler thread");
    (out, peak.load(Ordering::Relaxed))
}

/// Total number of differing codes between two `frames[T][nq]` grids (plus any length gap), for the
/// single-shot-vs-chunked equality gate.
fn count_code_mismatches(a: &[Vec<u32>], b: &[Vec<u32>]) -> usize {
    let mut diff = 0usize;
    for (fa, fb) in a.iter().zip(b.iter()) {
        let n = fa.len().max(fb.len());
        for q in 0..n {
            if fa.get(q) != fb.get(q) {
                diff += 1;
            }
        }
    }
    diff + a.len().abs_diff(b.len()) * 16
}

/// The sc-14181 DoD gate: on a real ≥ 30 s clip the chunked/streaming encode is **byte-identical** to
/// the single-shot encode (code for code, at two chunk durations), and its peak memory is bounded well
/// below the single-shot quadratic attention. Deterministic on the CPU default build (no metal/cuda
/// feature) — see the `codec::tests::chunked_stage_matches_single_shot` unit gate for the stage-level
/// equivalence proof this end-to-end test complements.
#[test]
#[ignore = "real weights: needs the ~7.1 GB MOSS-Audio-Tokenizer codec snapshot; run with --ignored --nocapture"]
fn moss_audio_codec_chunked_encode_matches_single_shot() {
    use candle_audio_moss_tts_realtime::codec::MossAudioCodec;
    let codec = MossAudioCodec::load(&codec_dir(), 16).expect("load codec");

    // A realistic ~32 s codec-manifold clip: decode a fixed pseudo-random 400-frame RVQ pattern
    // (12.5 fps → 400 frames = 32.0 s @ 24 kHz). Decoding is cheap (the decoder runs at 12.5 fps); the
    // encoder is where T explodes (its first stage runs at 100 fps → T ≈ 3200 for this clip).
    let n_frames = 400usize;
    let mut state: u32 = 0x0012_3455;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) % 1024
    };
    let pattern: Vec<Vec<u32>> = (0..n_frames)
        .map(|_| (0..16).map(|_| next()).collect())
        .collect();
    let clip = codec
        .decode_frames(&pattern, &|| false)
        .expect("decode long clip")
        .expect("not cancelled");
    let secs = clip.len() as f32 / moss::codec::SAMPLE_RATE as f32;
    assert!(
        secs >= 30.0,
        "clip must be ≥ 30 s to exercise the bound (got {secs:.1}s)"
    );

    // Warm the lazy encoder half (mmap + build) on a short slice so the memory probe below measures
    // the analysis attention, not the one-time weight fault-in.
    let warm_len = 48_000.min(clip.len());
    let _ = codec
        .encode_chunked(&clip[..warm_len], moss::codec::SAMPLE_RATE, 10.0)
        .expect("warm encoder half");

    // Sample the instantaneous RSS spike each path adds above its immediately-preceding resting RSS,
    // so the transient attention allocation is isolated from the (huge, already-resident) model.
    let rest_chunk = current_rss_bytes();
    let (chunked_small, peak_chunk) = with_peak_rss(|| {
        codec
            .encode_chunked(&clip, moss::codec::SAMPLE_RATE, 1.5)
            .expect("chunked encode (1.5 s window)")
    });
    let rest_single = current_rss_bytes();
    let (single, peak_single) = with_peak_rss(|| {
        codec
            .encode_single_shot(&clip, moss::codec::SAMPLE_RATE)
            .expect("single-shot encode")
    });
    let chunked_big = codec
        .encode_chunked(&clip, moss::codec::SAMPLE_RATE, 10.0)
        .expect("chunked encode (10 s window)");

    // (1) Identity — the DoD's primary bar: chunked matches single-shot exactly, at both windows.
    assert_eq!(
        single.len(),
        chunked_small.len(),
        "frame count: single-shot vs chunked(1.5 s)"
    );
    assert_eq!(
        single.len(),
        chunked_big.len(),
        "frame count: single-shot vs chunked(10 s)"
    );
    let miss_small = count_code_mismatches(&single, &chunked_small);
    let miss_big = count_code_mismatches(&single, &chunked_big);
    let total_codes = single.len() * 16;
    println!(
        "sc-14181 chunked encode: {} frames ({secs:.1}s, {total_codes} codes); mismatches vs \
         single-shot — chunked(1.5s)={miss_small}, chunked(10s)={miss_big}",
        single.len()
    );
    assert_eq!(
        miss_small, 0,
        "chunked(1.5 s) must reproduce single-shot codes exactly ({miss_small}/{total_codes} differ)"
    );
    assert_eq!(
        miss_big, 0,
        "chunked(10 s) must reproduce single-shot codes exactly ({miss_big}/{total_codes} differ)"
    );

    // (2) Memory bound: single-shot's quadratic first-stage attention spikes materially above the
    // bounded streaming path. For this clip the first stage is ~[1,20,3200,3200] f32 ≈ 780 MB/layer
    // single-shot vs the chunked ~[1,20,~150,~1150] ≈ 55 MB — a several-hundred-MB gap.
    let chunk_spike = peak_chunk.saturating_sub(rest_chunk);
    let single_spike = peak_single.saturating_sub(rest_single);
    println!(
        "sc-14181 transient RSS spike above resting: chunked(1.5s) +{:.0} MB, single-shot +{:.0} MB",
        chunk_spike as f64 / 1e6,
        single_spike as f64 / 1e6,
    );
    if peak_chunk > 0 && peak_single > 0 {
        assert!(
            single_spike > chunk_spike + 200_000_000,
            "single-shot's transient RSS spike (+{} MB) should exceed the chunked path's (+{} MB) by \
             >200 MB — the streaming path is not bounding the first-stage attention",
            single_spike / 1_000_000,
            chunk_spike / 1_000_000,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// sc-14149 — voice cloning: generate the same text with the default voice and with a reference
// clip; both must be intelligible (ASR CER) and the cloned output must DIFFER from the default
// (the reference timbre conditioning takes effect). `MOSS_VOICECLONE_REF` = a 24 kHz f32-LE mono
// reference clip. The speaker-identity (x-vector similarity) gate lands with the CAMPPlus harness.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "real weights: MOSS-TTS-Realtime AR + codec + whisper_base; run with --ignored --nocapture"]
fn moss_tts_realtime_voice_clone() {
    use candle_audio_whisper::gen_core::{
        AudioTrack as WAudioTrack, LoadSpec as WLoadSpec, TimestampGranularity, TranscribeOptions,
        TranscribeRequest, TranscribeTask, WeightsSource as WWeightsSource,
    };
    use moss::gen_core::{AudioTrack, Conditioning};

    let generator = load();
    let wspec = WLoadSpec::new(WWeightsSource::Dir(PathBuf::from(
        std::env::var("WHISPER_SNAPSHOT")
            .expect("set WHISPER_SNAPSHOT to an openai/whisper-base dir"),
    )));
    let transcriber = candle_audio_whisper::provider_registry()
        .expect("whisper registry")
        .load_transcriber(candle_audio_whisper::MODEL_ID, &wspec)
        .expect("whisper_base loads");
    let ref_clip = read_f32le(
        &std::env::var("MOSS_VOICECLONE_REF")
            .expect("set MOSS_VOICECLONE_REF to a 24 kHz f32-LE mono reference clip"),
    );

    let text = "The quick brown fox jumps over the lazy dog.";
    let req = |conditioning: Vec<Conditioning>| GenerationRequest {
        prompt: text.to_string(),
        audio: Some(AudioParams {
            target_duration: Some(6.0),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        seed: Some(20_260_719),
        conditioning,
        ..Default::default()
    };
    let synth = |conditioning| match generator
        .generate(&req(conditioning), &mut |_| {})
        .expect("generate")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    let transcribe = |track: &AudioTrack| {
        let treq = TranscribeRequest {
            audio: WAudioTrack {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                ..Default::default()
            },
            options: TranscribeOptions {
                language: Some("en".into()),
                task: TranscribeTask::Transcribe,
                timestamps: TimestampGranularity::None,
            },
            ..Default::default()
        };
        normalize(
            &transcriber
                .transcribe(&treq, &mut |_| {})
                .expect("transcribe")
                .text,
        )
    };

    let default = synth(vec![]);
    let clone = synth(vec![Conditioning::ReferenceAudio {
        audio: AudioTrack {
            samples: ref_clip.clone(),
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        },
        strength: None,
    }]);

    let (hyp_d, hyp_c) = (transcribe(&default), transcribe(&clone));
    let (cer_d, cer_c) = (
        character_error_rate(&normalize(text), &hyp_d),
        character_error_rate(&normalize(text), &hyp_c),
    );
    let corr = pearson(&clone.samples, &default.samples);
    println!(
        "voice-clone: default CER {cer_d:.3} ({:.2}s) {hyp_d:?}; clone CER {cer_c:.3} ({:.2}s) \
         {hyp_c:?}; clone-vs-default corr {corr:.3}",
        default.samples.len() as f32 / 24_000.0,
        clone.samples.len() as f32 / 24_000.0,
    );
    // Both intelligible — cloning must not break speech.
    assert!(
        cer_d <= MAX_PROMPT_CER,
        "default voice not intelligible (CER {cer_d:.3})"
    );
    assert!(
        cer_c <= MAX_PROMPT_CER,
        "cloned voice not intelligible (CER {cer_c:.3})"
    );
    // The reference timbre conditioning must change the output (a no-op would give corr ≈ 1.0).
    assert!(
        corr < 0.9,
        "cloned output must differ from the default voice — the reference conditioning took no \
         effect (corr {corr:.3})"
    );

    // Speaker identity (the sc-14149 DoD, **required** — the assertion that proves the clone carries
    // the reference speaker, so it is not skippable): the cloned output's CAMPPlus x-vector must
    // resemble the reference more than the default voice does. `CHATTERBOX_SNAPSHOT` = the Chatterbox
    // snapshot dir (its S3Gen checkpoint holds the CAMPPlus speaker encoder).
    let cb = std::env::var("CHATTERBOX_SNAPSHOT")
        .expect("set CHATTERBOX_SNAPSHOT to a Chatterbox snapshot dir (CAMPPlus speaker encoder)");
    let campplus = candle_audio_chatterbox::Campplus::from_snapshot(std::path::Path::new(&cb))
        .expect("load CAMPPlus speaker encoder from the Chatterbox snapshot");
    let embed = |s: &[f32]| campplus.embed(s, 24_000).expect("x-vector embed");
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-9)
    };
    let (e_ref, e_clone, e_def) = (
        embed(&ref_clip),
        embed(&clone.samples),
        embed(&default.samples),
    );
    let (sim_clone, sim_def) = (cos(&e_clone, &e_ref), cos(&e_def, &e_ref));
    println!(
        "voice-clone speaker sim (CAMPPlus x-vector cosine): clone↔ref {sim_clone:.3}, \
         default↔ref {sim_def:.3}"
    );
    assert!(
        sim_clone > sim_def + 0.05,
        "cloned output must resemble the reference speaker MORE than the default voice \
         (clone↔ref {sim_clone:.3} vs default↔ref {sim_def:.3})"
    );
}
