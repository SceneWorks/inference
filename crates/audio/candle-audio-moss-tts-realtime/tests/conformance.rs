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
//! - [`moss_tts_realtime_is_incremental`] — the AR loop's progress reporting is one-per-frame:
//!   `Progress::Step { current }` arrives once per returned frame, starting at 1 and advancing by
//!   one. sc-19556 replaced the former "time to the first frame < time to the full budget" bound,
//!   which was degenerate as well as clock-bound; see the note at the assertion. This does NOT
//!   distinguish an incremental decode from a buffer-everything one.
//! - [`moss_tts_realtime_streaming_gate`] — the sc-13334 streaming acceptance gate, now released by
//!   the codec: `gen_core_testkit::check_audio_streaming` against the **real** registered provider
//!   ((a) ≥ 2 PCM chunks before completion; (b) concat(chunks) == one-shot `generate()`
//!   byte-identical; (c) valid 24 kHz mono track), plus (c) full audio non-silent / speech-shaped
//!   and (d) the chunk stream is a faithful partition of the returned track — non-empty first
//!   chunk, strictly smaller than the track, and the chunks reassemble to it exactly. It also
//!   writes a playable demo WAV. sc-19556 replaced the former first-chunk-latency bound here too.
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
//! cargo test --locked -p candle-audio-moss-tts-realtime --test conformance -- \
//!     --ignored --nocapture --test-threads=1
//! ```
//! `--test-threads=1` is **required**, not tidiness: this binary installs a process-wide
//! [`TrackingAlloc`] global allocator (sc-17263) and
//! [`moss_audio_codec_chunked_encode_matches_single_shot`] measures heap high-water marks through
//! it, so a test running concurrently in the same process would land in its measurement window.
//! Set `MOSS_TTS_REALTIME_SNAPSHOT` to the AR snapshot dir (~4.66 GB, holding `config.json`,
//! `model.safetensors`, `tokenizer.json`) — **required**, a passed-in path: inference never
//! self-fetches or derives a cache location (epic 13657). The MOSS-Audio-Tokenizer codec (~7.1 GB) is
//! likewise a passed-in component (sc-13662): it is **required** from `MOSS_AUDIO_TOKENIZER_SNAPSHOT`
//! (the codec snapshot dir, `config.json` + `model*.safetensors`) — the provider never self-fetches
//! it, so this must point at a materialized snapshot. The demo WAV path is `MOSS_TTS_REALTIME_WAV_OUT`
//! (default temp dir). The fidelity gate additionally uses `whisper_base` — **required** from
//! `WHISPER_SNAPSHOT` (the ~150 MB snapshot dir), also a passed-in path, never a hub fetch.

use std::path::PathBuf;

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

/// Require the emitted chunks to be the returned track's exact ordered sample partition.
#[track_caller]
fn assert_chunks_reassemble_exact<'a>(chunks: impl IntoIterator<Item = &'a [f32]>, track: &[f32]) {
    let reassembled: Vec<f32> = chunks.into_iter().flatten().copied().collect();
    assert_eq!(
        reassembled.len(),
        track.len(),
        "the chunks must reassemble to the returned track's exact sample count"
    );
    if let Some(index) = reassembled
        .iter()
        .zip(track)
        .position(|(chunk, track)| chunk != track)
    {
        panic!(
            "the chunks must reassemble sample-for-sample to exactly the returned track: sample \
             {index} differs (chunk {}, track {})",
            reassembled[index], track[index]
        );
    }
}

#[test]
fn exact_chunk_reassembly_rejects_equal_length_sample_corruption() {
    let first = [0.25, -0.5];
    let second = [0.75, 1.0];
    let track = [0.25, -0.5, 0.75, 1.0];
    assert_chunks_reassemble_exact([first.as_slice(), second.as_slice()], &track);

    let corrupted_first = [-0.25, -0.5];
    assert!(
        std::panic::catch_unwind(|| {
            assert_chunks_reassemble_exact([corrupted_first.as_slice(), second.as_slice()], &track)
        })
        .is_err(),
        "equal-length chunks with changed sample content must be rejected"
    );
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
    // sc-19556: this used to compare `first_frame_at` against the total elapsed decode. That
    // comparison was DEGENERATE as well as clock-bound: `total` is sampled after `rvq_frames`
    // returns, and the callback necessarily fires before then, so `first < total` held for ANY
    // implementation — including the "emit everything at the end" one the assertion named as the
    // case it existed to catch.
    //
    // What replaces it is a PROGRESS-CADENCE check, read from the callback stream with no clock:
    // `Step { current }` must arrive once per returned frame, starting at 1 and advancing by one.
    // That catches a decode whose reporting has come loose from its production — a wrong count, a
    // skipped or repeated index, an out-of-order stream, or no reporting at all.
    //
    // SCOPE, stated plainly because the assertion being replaced overclaimed at exactly this point:
    // this does NOT catch the buffer-everything implementation. A decode that computed every frame
    // first and then fired N callbacks 1..N satisfies the count, first-index and consecutiveness
    // checks below. Separating that case from a real incremental decode needs evidence that the
    // callback fires BEFORE the remaining frames exist, which is either a clock — just removed here
    // for being degenerate — or a seam this generator does not expose. So the cadence claim is the
    // one the stream can actually support, and it is the only one made.
    let mut steps: Vec<u32> = Vec::new();
    let result = gen
        .rvq_frames(&request(1.6), &mut |p| {
            if let Progress::Step { current, .. } = p {
                steps.push(current);
            }
        })
        .expect("AR decode");
    eprintln!(
        "progress steps {:?} for {} frames",
        steps,
        result.frames.len()
    );
    assert!(
        result.frames.len() >= 2,
        "need ≥ 2 frames to demonstrate incrementality"
    );
    // One report per frame, strictly increasing, starting at the first frame: the AR loop announced
    // each frame as it produced it.
    assert_eq!(
        steps.len(),
        result.frames.len(),
        "the AR loop must report progress once per frame, got {} reports for {} frames",
        steps.len(),
        result.frames.len()
    );
    assert_eq!(
        steps.first().copied(),
        Some(1),
        "the first progress report must be frame 1, got {steps:?}"
    );
    assert!(
        steps.windows(2).all(|w| w[1] == w[0] + 1),
        "progress must advance one frame at a time and in order, got {steps:?}"
    );
}

/// The streaming acceptance gate (sc-13334, released by the sc-13392 codec): the shared
/// `check_audio_streaming` suite against the **real registered provider** (chunk-count, reassembly
/// law, one-shot == stream), plus the DoD extras — the chunk stream is a faithful partition of the
/// returned track, non-silent speech-shaped 24 kHz audio, and a playable demo WAV. The first-chunk
/// latency bound this used to carry was removed as degenerate in sc-19556; see the note inline.
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

    // (d) the chunk stream is a faithful partition of the returned track, checked directly.
    let req = request(seconds);
    let mut chunks: Vec<AudioChunk> = Vec::new();
    let out = generator
        .generate_streaming(&req, &mut |c| chunks.push(c), &mut |_| {})
        .expect("streaming generate");
    let track = match out {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected GenerationOutput::Audio, got {other:?}"),
    };
    eprintln!(
        "streaming: {} chunks, first chunk {} samples, full track {} samples",
        chunks.len(),
        chunks.first().map(|c| c.samples.len()).unwrap_or(0),
        track.samples.len()
    );
    assert!(
        chunks.len() >= 2,
        "expected >= 2 stream chunks, got {}",
        chunks.len()
    );
    // sc-19556: `first < full` (first-chunk latency vs full-generation latency) was replaced. Like
    // its AR-stage twin above it was degenerate — `full` is sampled after `generate_streaming`
    // returns, so any implementation that emits a chunk at all satisfies it, including the
    // buffer-everything one it named.
    //
    // What replaces it is a CHUNKING-AND-REASSEMBLY check, with no clock: the first chunk carries
    // audio, it is strictly smaller than the finished track, and the chunks sum to exactly the
    // track that was returned. The reassembly law is genuinely new coverage — nothing here
    // previously tied the emitted chunks to the returned track at all, so a stream that dropped,
    // duplicated or rescaled samples relative to the track passed.
    //
    // SCOPE, stated plainly for the same reason as its AR-stage twin above: this does NOT catch the
    // buffer-everything implementation either. One that generated the whole track and then sliced
    // it into >= 2 chunks satisfies non-empty-first, first < track, and exact reassembly. What is
    // gated is that the chunk stream is a faithful partition of the track, not that it was produced
    // before the track existed — the latter needs a clock or a seam this generator does not expose.
    let first_chunk = chunks.first().expect("at least one chunk was emitted");
    assert!(
        !first_chunk.samples.is_empty(),
        "the first chunk must carry audio, not an empty priming chunk"
    );
    assert!(
        first_chunk.samples.len() < track.samples.len(),
        "the first chunk carries {} of the track's {} samples — a first chunk holding the whole \
         track is a buffered generate wearing a streaming signature",
        first_chunk.samples.len(),
        track.samples.len()
    );
    // The reassembly law (`check_audio_streaming` gates it too) is what makes the size comparison
    // above mean "a prefix" rather than "a differently-shaped buffer".
    assert_chunks_reassemble_exact(
        chunks.iter().map(|chunk| chunk.samples.as_slice()),
        &track.samples,
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
         {peak:.4}, interior RMS {rms:.4}, frame-RMS CV {cv:.3}, first chunk {} of {} samples)",
        out_path.display(),
        chunks.len(),
        chunks.first().map(|c| c.samples.len()).unwrap_or(0),
        track.samples.len(),
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
        .decode_frames(
            &frames,
            moss::codec::decode_partition_schedule(frames.len()),
            &|| false,
        )
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
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read f32-LE clip {path}: {e}"));
    // A trailing partial sample means the file is truncated, not that the last sample is optional;
    // `chunks_exact` would drop it and hand back a quietly shorter waveform.
    assert!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "f32-LE clip {path} is {} bytes — empty or not a whole number of f32 samples",
        bytes.len()
    );
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

/// Resolve a committed fixture under `tests/fixtures/`, with `env` as an override.
///
/// Two fixture sets ride on this, both for the same reason — an env-only path meant an unset
/// variable silently removed real-weight coverage that nothing else replaced:
///
/// - **Encode reference parity (sc-17270)** — `MOSS_CODEC_CLIP` / `MOSS_CODEC_REF_CODES`. The
///   cross-check used to sit behind `if let Ok(..)`, so an unset variable dropped it entirely while
///   libtest still reported the test as passing. Regenerate with
///   `scripts/reference/moss_audio_codec_reference.py`.
/// - **Voice cloning (sc-17264)** — `MOSS_VOICECLONE_REF`. These two tests *did* fail loudly on an
///   unset variable, so the coverage was lost a rung earlier: both were left out of the real-weight
///   lane altogether because the clip existed nowhere. Regenerate with
///   `cargo run -p candle-audio-moss-tts-realtime --example voiceclone_ref_clip`.
///
/// The variables survive as an override for pointing the same gate at different audio; what they
/// can no longer do is decide whether the gate runs at all.
///
/// An empty or whitespace-only value counts as unset. An unconfigured `${{ vars.X }}` expands to
/// the empty string in a workflow, so treating it as an override would turn a mis-wired lane into
/// a confusing read failure instead of simply using the fixture that ships with the test.
fn fixture_path(env: &str, file: &str) -> String {
    match std::env::var(env) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(file)
            .to_string_lossy()
            .into_owned(),
    }
}

#[test]
#[ignore = "real weights: needs the ~7.1 GB MOSS-Audio-Tokenizer codec snapshot; run with --ignored --nocapture"]
fn moss_audio_codec_encode_roundtrip_and_reference() {
    use candle_audio_moss_tts_realtime::codec::MossAudioCodec;
    let codec = MossAudioCodec::load(&codec_dir(), 16).expect("load codec");

    // Strong cross-check: the port's encode must match the reference PyTorch `codec.encode` on a
    // byte-identical clip. `MOSS_CODEC_CLIP` = raw f32-LE mono samples at 24 kHz;
    // `MOSS_CODEC_REF_CODES` = the reference codes CSV (`frames[T][16]`). Both default to the
    // committed fixture (sc-17270), so this arm ALWAYS runs — it is the half of this test that
    // actually gates the port against the reference encoder, and it used to be skippable by simply
    // not setting two variables that nothing in the repository set.
    // Both or neither: half an override pairs a custom clip against the committed codes, which
    // fails with a frame-count or agreement message that says nothing about the real mistake.
    assert_eq!(
        std::env::var("MOSS_CODEC_CLIP").is_ok(),
        std::env::var("MOSS_CODEC_REF_CODES").is_ok(),
        "set MOSS_CODEC_CLIP and MOSS_CODEC_REF_CODES together or not at all — the codes are only \
         valid for the clip they were generated from"
    );
    let clip_p = fixture_path("MOSS_CODEC_CLIP", "moss_codec_ref_clip.f32");
    let ref_p = fixture_path("MOSS_CODEC_REF_CODES", "moss_codec_ref_codes.csv");
    let clip = read_f32le(&clip_p);
    let port = codec.encode(&clip, 24_000).expect("encode reference clip");
    let refc = read_codes_csv(&ref_p);
    let n = port.len().min(refc.len());
    assert!(n > 0, "no frames to compare");
    assert_eq!(
        port.len(),
        refc.len(),
        "frame count must match the reference"
    );
    // Per-quantizer, not just pooled. A pooled rate over 16 codebooks hides its own worst case: at
    // the 0.98 bound below, one deep quantizer may disagree on 32 of 100 frames and still pass,
    // because the other fifteen are perfect. Count each codebook separately so a regression
    // confined to one of them cannot hide in the average.
    let mut agree_per_q = [0usize; 16];
    for f in 0..n {
        assert_eq!(
            port[f].len(),
            16,
            "port frame {f} has {} codebooks, expected 16",
            port[f].len()
        );
        assert_eq!(
            refc[f].len(),
            16,
            "reference frame {f} has {} codebooks, expected 16",
            refc[f].len()
        );
        for q in 0..16 {
            if port[f][q] == refc[f][q] {
                agree_per_q[q] += 1;
            }
        }
    }
    let cb0_rate = agree_per_q[0] as f32 / n as f32;
    let all_rate = agree_per_q.iter().sum::<usize>() as f32 / (n * 16) as f32;
    let (worst_q, worst_agree) = agree_per_q
        .iter()
        .enumerate()
        .min_by_key(|(_, agree)| **agree)
        .map(|(q, agree)| (q, *agree))
        .expect("16 quantizers");
    let worst_rate = worst_agree as f32 / n as f32;
    println!(
        "codec ref cross-check: port {} vs ref {} frames (cmp {n}); cb0 agree {cb0_rate:.3}, \
         all-cb agree {all_rate:.3}, worst codebook {worst_q} agree {worst_rate:.3}",
        port.len(),
        refc.len()
    );
    // The port matches the reference codec.encode exactly — measured 1.000 on every one of the 16
    // codebooks against the committed fixture. The bounds sit just under that to allow only
    // cross-platform argmax tie noise, so a real regression on codebook-0, on the pooled rate, or
    // on any single higher quantizer fails here.
    assert!(
        cb0_rate >= 0.99,
        "port encode codebook-0 must match the reference encoder (agree {cb0_rate:.3})"
    );
    assert!(
        all_rate >= 0.98,
        "port encode must match the reference across all 16 codebooks (agree {all_rate:.3})"
    );
    assert!(
        worst_rate >= 0.95,
        "every codebook must match the reference; codebook {worst_q} agrees only {worst_rate:.3}"
    );

    // Self-contained round-trip: a real codec waveform (decode a fixed pseudo-random 40-frame pattern)
    // → encode → decode must reconstruct it — the encoder emits decodable, faithful codes.
    let mut state: u32 = 1;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) % 1024
    };
    let frames0: Vec<Vec<u32>> = (0..40).map(|_| (0..16).map(|_| next()).collect()).collect();
    let w0 = codec
        .decode_frames(
            &frames0,
            moss::codec::decode_partition_schedule(frames0.len()),
            &|| false,
        )
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
        .decode_frames(
            &codes1,
            moss::codec::decode_partition_schedule(codes1.len()),
            &|| false,
        )
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
// codes** on a real ≥ 30 s clip, and that the streaming path's peak heap sits well below
// single-shot's — measured at the allocator, not by sampling RSS on a timer (sc-17263).
// ---------------------------------------------------------------------------------------------

/// One `AtomicU64` on its own cache line. Every allocation in the process touches both counters
/// below, from whatever rayon worker candle's gemm is using; packed adjacently they would share a
/// line and turn each allocation into a false-sharing round trip. 128 B is Apple Silicon's line
/// size (and a safe over-estimate of x86's 64 B).
#[repr(align(128))]
struct PaddedCounter(std::sync::atomic::AtomicU64);

/// Live heap bytes currently handed out by [`TrackingAlloc`], and a resettable high-water mark of
/// that same quantity. `PEAK` is only ever raised by the allocator and re-armed by
/// [`with_peak_alloc`]; both are plain byte counts, meaningful only relative to the base captured
/// at the start of a measurement window.
static LIVE_BYTES: PaddedCounter = PaddedCounter(std::sync::atomic::AtomicU64::new(0));
static PEAK_BYTES: PaddedCounter = PaddedCounter(std::sync::atomic::AtomicU64::new(0));

/// A `System`-delegating global allocator that tracks in-flight heap bytes and their high-water
/// mark (sc-17263).
///
/// This replaces the previous probe — `ps -o rss=` sampled on a 5 ms timer — which measured a
/// *transient* with a wall-clock sampler: the faster the box, the shorter the quadratic-attention
/// burst, the fewer samples landed inside it, and the further the measured peak fell below the true
/// one. That made the gate's verdict a function of machine speed (it read ~291 MB of a ~1.6 GB
/// spike on an M5 Max and failed a bound tuned elsewhere). Counting at the allocator is
/// event-driven, so it observes every byte regardless of how briefly it is held.
///
/// **Precision.** Exact for `alloc`/`alloc_zeroed`/`dealloc`, in the sense that every byte the
/// caller *requested* is counted — not the size class the allocator actually reserved, so the true
/// footprint is a little larger than the number reported. `realloc` is counted as a delta, so a
/// *relocating* grow (malloc-new + copy + free-old) undercounts the window in which both blocks are
/// live. Neither approximation is load-bearing here: the quantity under test is one multi-hundred-MB
/// `Tensor` allocation, and the bounds below clear the error by orders of magnitude.
///
/// **Cost to the rest of the binary.** Installing this is process-wide, so the other real-weight
/// tests here pay two relaxed atomic RMWs per allocation. A/B'd on the heaviest of them
/// (`moss_tts_realtime_asr_roundtrip_fidelity`, a whisper round-trip): 517 s / 311 s installed vs
/// 327 s / 361 s not installed — no measurable penalty, and the installed arm was not the slower
/// one. Run-to-run spread on a loaded box (~60%) is far wider than any effect here, so read that as
/// ruling out a large regression, not as a precise overhead figure.
///
/// It measures the right thing here because candle's CPU tensor storage is a `Vec<T>`
/// (`cpu_backend/mod.rs`), so the first stage's `[1, H, T, T]` attention is a single
/// global-allocator request. Note the codec's ~7.1 GB of weights are **not** excluded by being
/// mmapped: `VarBuilder::from_mmaped_safetensors` copies each tensor onto the Rust heap on the way
/// in (`convert_slice` → `Tensor::from_slice` → `to_cpu_storage` → `data.to_vec()`), so they sit in
/// `LIVE_BYTES` too. What isolates the transient is [`with_peak_alloc`] subtracting the resting
/// total — which is why the warm-up call in the test below is load-bearing, not hygiene.
struct TrackingAlloc;

impl TrackingAlloc {
    /// Add `n` bytes to the live count and raise the high-water mark to match.
    fn record_alloc(n: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        if n == 0 {
            return;
        }
        let live = LIVE_BYTES.0.fetch_add(n as u64, Relaxed) + n as u64;
        PEAK_BYTES.0.fetch_max(live, Relaxed);
    }
}

// SAFETY: every arm delegates to `System` (itself a valid `GlobalAlloc`) with the caller's original
// pointer/layout contract untouched, and only adds relaxed atomic bookkeeping around it. The
// counters are advisory-only — no allocation decision reads them — so a torn or reordered count can
// never affect memory safety.
unsafe impl std::alloc::GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = std::alloc::System.alloc(layout);
        if !p.is_null() {
            Self::record_alloc(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = std::alloc::System.alloc_zeroed(layout);
        if !p.is_null() {
            Self::record_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        LIVE_BYTES
            .0
            .fetch_sub(layout.size() as u64, std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let p = std::alloc::System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            // Only the delta moves; a grow can set a new peak, a shrink never can.
            match new_size.checked_sub(layout.size()) {
                Some(grew) => Self::record_alloc(grew),
                None => {
                    LIVE_BYTES.0.fetch_sub(
                        (layout.size() - new_size) as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
        }
        p
    }
}

/// Installed for this **test binary only** — `tests/conformance.rs` is its own crate root, so the
/// shipping library and its consumers keep the default allocator.
#[global_allocator]
static ALLOC: TrackingAlloc = TrackingAlloc;

/// Run `f` and return `(result, peak_heap_bytes_above_the_pre-call_live_total)` — the transient this
/// call added, with the already-resident heap (weights included) subtracted out.
///
/// Measured process-wide, so a *concurrently running* test in the same binary would add noise: the
/// real-weight harness selects one test at a time (`--exact`) and the module doc requires
/// `--test-threads=1`. The two encode paths are measured back-to-back on the same thread, and the
/// ordering here rests on that same-thread program order — not on the `Relaxed` counter updates,
/// which deliberately claim no cross-thread happens-before. Allocations made by rayon workers
/// candle has already fenced into the call are counted; anything genuinely concurrent is noise the
/// bounds below are sized to absorb.
fn with_peak_alloc<T>(f: impl FnOnce() -> T) -> (T, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let base = LIVE_BYTES.0.load(Relaxed);
    PEAK_BYTES.0.store(base, Relaxed);
    let out = f();
    let peak = PEAK_BYTES.0.load(Relaxed);
    (out, peak.saturating_sub(base))
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
#[ignore = "real weights: needs the ~7.1 GB MOSS-Audio-Tokenizer codec snapshot; run with --ignored --nocapture --test-threads=1 (the heap probe is process-wide)"]
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
        .decode_frames(
            &pattern,
            moss::codec::decode_partition_schedule(pattern.len()),
            &|| false,
        )
        .expect("decode long clip")
        .expect("not cancelled");
    let secs = clip.len() as f32 / moss::codec::SAMPLE_RATE as f32;
    assert!(
        secs >= 30.0,
        "clip must be ≥ 30 s to exercise the bound (got {secs:.1}s)"
    );

    // Warm the lazy encoder half on a short slice. This is **load-bearing for the measurement**, not
    // hygiene: `from_mmaped_safetensors` copies every weight onto the Rust heap, so building the
    // encoder half allocates GBs through the probe. Doing it here folds that into the resting total
    // `with_peak_alloc` subtracts, leaving the windows below measuring analysis attention alone.
    let warm_len = 48_000.min(clip.len());
    let _ = codec
        .encode_chunked(&clip[..warm_len], moss::codec::SAMPLE_RATE, 10.0)
        .expect("warm encoder half");

    // Measure the heap high-water mark each path adds above the already-resident total, so the
    // transient attention allocation is isolated from the (huge, heap-resident) model. The half-clip
    // chunked window is measured too: it is what turns "peak memory is independent of clip length"
    // from an unbacked claim in a comment into something this test actually observes.
    let (_, chunk_spike_half) = with_peak_alloc(|| {
        codec
            .encode_chunked(&clip[..clip.len() / 2], moss::codec::SAMPLE_RATE, 1.5)
            .expect("chunked encode (1.5 s window, half clip)")
    });
    let (chunked_small, chunk_spike) = with_peak_alloc(|| {
        codec
            .encode_chunked(&clip, moss::codec::SAMPLE_RATE, 1.5)
            .expect("chunked encode (1.5 s window)")
    });
    let (single, single_spike) = with_peak_alloc(|| {
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
    // single-shot — scores plus the softmax over them, so ~1.6 GB in flight — vs the chunked
    // ~[1,20,~150,~1150]. Measured: +1669 MB vs +68 MB, a ~24x gap, byte-identical run to run.
    //
    // Four bounds. (a) alone is not enough: it goes green whenever the two paths differ by 200 MB,
    // including the case this gate most needs to catch — *both* arms expensive (chunked 1.0 GB,
    // single-shot 1.3 GB) because the streaming window stopped bounding anything. So:
    //   (a) the gap — sc-14181's original >200 MB bar, unchanged;
    //   (b) the ratio — scale-free, so proportional drift on another machine does not erode it;
    //   (c) an absolute ceiling on the chunked arm — catches "both expensive", and stays meaningful
    //       even if the single-shot arm ever stops being quadratic;
    //   (d) growth across clip length — the streaming claim itself, measured rather than asserted.
    //
    // Heap-counted, so this arm needs the CPU device: on `metal`/`cuda` the stage tensors are device
    // buffers the global allocator never sees, and both paths would read as ~nothing. Keyed on the
    // device the codec actually loaded onto (`candle_audio::default_device`, exactly what
    // `MossAudioCodec::load` calls) rather than on this crate's feature flags, which are only a
    // proxy for it.
    //
    // This REFUSES rather than skips. A skip would pass while measuring nothing, and the weekly
    // lane's `test result: ok. 1 passed` grep cannot see a branch that was not taken — the precise
    // false green sc-17270 removed from this file and now gates against in
    // `scripts/tests/test_moss_audio_codec_reference.py`. Failing on a build that cannot run this
    // gate costs one clear error message; skipping costs the coverage, silently.
    assert!(
        moss::candle_audio::default_device()
            .expect("resolve the codec's device")
            .is_cpu(),
        "this gate measures a HEAP high-water mark, but the codec loaded onto a non-CPU device \
         whose stage tensors are device buffers the allocator never sees. Run it on the default \
         (CPU) build — `-p candle-audio-moss-tts-realtime` with no metal/cuda feature.",
    );
    println!(
        "sc-14181 transient heap high-water above resting: chunked(1.5s) +{:.0} MB, single-shot \
         +{:.0} MB (ratio {:.1}x); chunked on the half clip +{:.0} MB",
        chunk_spike as f64 / 1e6,
        single_spike as f64 / 1e6,
        single_spike as f64 / chunk_spike.max(1) as f64,
        chunk_spike_half as f64 / 1e6,
    );
    assert!(
        single_spike > chunk_spike + 200_000_000,
        "single-shot's transient heap spike (+{} MB) should exceed the chunked path's (+{} MB) by \
         >200 MB — the streaming path is not bounding the first-stage attention",
        single_spike / 1_000_000,
        chunk_spike / 1_000_000,
    );
    assert!(
        single_spike >= chunk_spike.saturating_mul(8),
        "single-shot's transient heap spike (+{} MB) should be at least 8x the chunked path's \
         (+{} MB) — the streaming window is not bounding attention the way `forward_chunked` claims",
        single_spike / 1_000_000,
        chunk_spike / 1_000_000,
    );
    assert!(
        chunk_spike < 250_000_000,
        "the chunked path's transient heap spike (+{} MB) on this {secs:.1}s clip must stay under \
         250 MB — a bounded sliding window cannot need that much, so the streaming path is \
         materializing something proportional to the clip",
        chunk_spike / 1_000_000,
    );
    // Doubling the clip must not double the chunked transient: the window is fixed, so only the
    // linear side-buffers (input PCM, per-stage activations) may grow. A quadratic — or even
    // linear-dominated — chunked path fails here while (a)-(c) could still pass.
    //
    // Measured 56 MB → 68 MB across the 16 s → 32 s doubling, i.e. 1.21x. A linear-dominated path
    // would read ~2.0x and a quadratic one ~4x, so the 1.5x bar sits between what is healthy and
    // the cheapest regression it must catch.
    assert!(
        chunk_spike < chunk_spike_half.saturating_mul(3) / 2,
        "chunked spike grew from +{} MB on {:.1}s to +{} MB on {secs:.1}s — doubling the clip must \
         not scale the streaming path's peak memory; the window is supposed to bound it",
        chunk_spike_half / 1_000_000,
        secs / 2.0,
        chunk_spike / 1_000_000,
    );
}

// ---------------------------------------------------------------------------------------------
// sc-14149 — voice cloning: generate the same text with the default voice and with a reference
// clip; both must be intelligible (ASR CER) and the cloned output must DIFFER from the default
// (the reference timbre conditioning takes effect). The reference speaker is a committed fixture
// (sc-17264) rendered by Kokoro — `MOSS_VOICECLONE_REF` overrides it with another 24 kHz f32-LE
// mono clip, but no longer decides whether this test can run at all.
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
    let ref_clip = read_f32le(&fixture_path(
        "MOSS_VOICECLONE_REF",
        "moss_voiceclone_ref_clip.f32",
    ));

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

// ---------------------------------------------------------------------------------------------
// sc-14151 — multi-turn conversational continuation, the model's headline capability: turn N's speech
// conditioned on the prior turns, through BOTH the stateless history-in-request path (A,
// `generate` + `Conditioning::ConversationHistory`) and the stateful warm-KV session (B,
// `open_conversation` + `step`). The DoD gate on real weights.
// ---------------------------------------------------------------------------------------------

/// CAMPPlus x-vector cosine, loaded from the Chatterbox snapshot (its S3Gen checkpoint holds the
/// speaker encoder) — the same speaker-similarity metric the voice-clone gate uses.
fn campplus_cos(cb_snapshot: &str) -> impl Fn(&[f32], &[f32]) -> f32 {
    let campplus =
        candle_audio_chatterbox::Campplus::from_snapshot(std::path::Path::new(cb_snapshot))
            .expect("load CAMPPlus speaker encoder from the Chatterbox snapshot");
    move |a: &[f32], b: &[f32]| {
        let ea = campplus.embed(a, 24_000).expect("x-vector embed a");
        let eb = campplus.embed(b, 24_000).expect("x-vector embed b");
        let dot: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
        let na = ea.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = eb.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-9)
    }
}

/// The DoD gate for multi-turn conversational continuation (sc-14151). A short scripted conversation
/// renders end-to-end through BOTH shapes and must satisfy:
///
/// - **A≡B byte-identical** — the whole conversation rendered in one stateless `generate` (path A)
///   equals the concatenation of the per-turn stateful `session.step`s (path B), sample-for-sample —
///   the session is a warm-cache optimization of the batch render, not a different computation.
/// - **each turn intelligible** — every synthesized reply transcribes back to its text within the CER
///   bound (a later turn that collapsed to silence/babble fails here).
/// - **conditioned on the prior turns** — the *same* target turn at the *same* seed, rendered after
///   *different* prior context, produces *different* audio (a provider that ignores the conversation
///   history renders it identically); and the assistant turns are the **same speaker** across the
///   conversation (CAMPPlus x-vector cosine above a bound — voice/prosody continuity).
/// - **deterministic** — the same conversation + seed re-renders byte-identical.
#[test]
#[ignore = "real weights: MOSS-TTS-Realtime AR + codec + whisper_base + Chatterbox CAMPPlus; run with --ignored --nocapture"]
fn moss_tts_realtime_multi_turn_conversation() {
    use candle_audio_whisper::gen_core::{
        AudioTrack as WAudioTrack, LoadSpec as WLoadSpec, TimestampGranularity, TranscribeOptions,
        TranscribeRequest, TranscribeTask, WeightsSource as WWeightsSource,
    };
    use moss::gen_core::{AudioTrack, Conditioning, ConversationRole, ConversationTurn};

    let generator = load();
    let wspec = WLoadSpec::new(WWeightsSource::Dir(PathBuf::from(
        std::env::var("WHISPER_SNAPSHOT")
            .expect("set WHISPER_SNAPSHOT to an openai/whisper-base dir"),
    )));
    let transcriber = candle_audio_whisper::provider_registry()
        .expect("whisper registry")
        .load_transcriber(candle_audio_whisper::MODEL_ID, &wspec)
        .expect("whisper_base loads");
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

    let asst = |t: &str| ConversationTurn {
        role: ConversationRole::Assistant,
        text: t.to_string(),
        audio: None,
    };
    // A conversation-level request (seed + audio params); the text rides in the turns.
    let conv_audio = || {
        Some(AudioParams {
            target_duration: Some(8.0),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        })
    };
    let conv_req = |turns: Vec<ConversationTurn>| GenerationRequest {
        prompt: String::new(),
        audio: conv_audio(),
        seed: Some(20_260_719),
        conditioning: vec![Conditioning::ConversationHistory { turns }],
        ..Default::default()
    };
    let open_req = || GenerationRequest {
        prompt: String::new(),
        audio: conv_audio(),
        seed: Some(20_260_719),
        ..Default::default()
    };

    let t0 = "The weather is very nice this afternoon.";
    let t1 = "The train arrives at nine in the morning.";

    // --- Path A: render the whole conversation in one generate (assistant replies concatenated). ---
    let track_a = match generator
        .generate(&conv_req(vec![asst(t0), asst(t1)]), &mut |_| {})
        .expect("path A: generate the conversation")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };

    // --- Path B: a stateful session, one step per turn. ---
    let mut session = generator
        .open_conversation(&open_req())
        .expect("path B: open the conversational session");
    let b0 = match_audio(
        session.step(&asst(t0), &mut |_| {}, &mut |_| {}),
        "path B: step turn 0",
    );
    let b1 = match_audio(
        session.step(&asst(t1), &mut |_| {}, &mut |_| {}),
        "path B: step turn 1",
    );
    session.finish().expect("path B: finish session");
    let b_concat: Vec<f32> = b0.samples.iter().chain(&b1.samples).copied().collect();

    println!(
        "multi-turn: path A {} samples ({:.2}s); path B turns {} + {} = {} samples",
        track_a.samples.len(),
        track_a.samples.len() as f32 / 24_000.0,
        b0.samples.len(),
        b1.samples.len(),
        b_concat.len(),
    );

    // === The A≡B equivalence law: byte-identical. ===
    assert_eq!(
        track_a.samples, b_concat,
        "A≡B: the stateless batch render must equal the concatenated stateful session steps, \
         sample-for-sample (the session is a warm-cache optimization of generate, not a different \
         computation)"
    );

    // === Each turn intelligible (per-turn ASR CER, using the session's per-turn tracks). ===
    for (text, track) in [(t0, &b0), (t1, &b1)] {
        assert!(!track.samples.is_empty(), "empty audio for turn {text:?}");
        let cer = character_error_rate(&normalize(text), &transcribe(track));
        println!(
            "multi-turn: turn {text:?} CER {cer:.3} ({:.2}s)",
            track.samples.len() as f32 / 24_000.0
        );
        assert!(
            cer <= MAX_PROMPT_CER,
            "turn {text:?} is not intelligible (CER {cer:.3}) — a later turn that collapsed to \
             silence/babble fails here"
        );
    }

    // === Conditioned on the prior turns: the SAME target turn at the SAME seed (ordinal 1 in both),
    // rendered after DIFFERENT prior context, must differ. Same-seed isolates the effect to the
    // conversation history — a provider that ignores it renders the two byte-identical (corr 1.0).
    // NOTE: two different prior turns also shift the target block's absolute cache offset, so this
    // gate confirms the target *depends on* the conversation but does not by itself isolate
    // attention-over-prior-content from position shift; the load-bearing proof that a warm prefill
    // genuinely attends over the prior turns' KV is the weightless
    // `backbone::tests::warm_cache_second_prefill_matches_full_recompute` mask-equivalence test. ===
    let render_target_after = |ctx: &str| -> AudioTrack {
        let mut s = generator
            .open_conversation(&open_req())
            .expect("open session for the discriminator");
        match_audio(
            s.step(&asst(ctx), &mut |_| {}, &mut |_| {}),
            "discriminator: context turn",
        );
        match_audio(
            s.step(&asst(t1), &mut |_| {}, &mut |_| {}),
            "discriminator: target turn",
        )
    };
    let after_weather = render_target_after("Let me tell you about the local weather forecast.");
    let after_schedule = render_target_after("Here is today's train and bus schedule.");
    let ctx_corr = pearson(&after_weather.samples, &after_schedule.samples);
    println!("multi-turn: same-turn-different-context corr {ctx_corr:.3}");
    assert!(
        ctx_corr < 0.99,
        "the target turn must depend on the prior turn — the same turn at the same seed after two \
         different contexts produced near-identical audio (corr {ctx_corr:.3}), so the provider \
         appears to ignore the conversation history"
    );

    // === Determinism: the same conversation + seed re-renders byte-identical. ===
    let track_a2 = match generator
        .generate(&conv_req(vec![asst(t0), asst(t1)]), &mut |_| {})
        .expect("path A re-render")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    assert_eq!(
        track_a.samples, track_a2.samples,
        "the same conversation + seed must re-render byte-identical (per-conversation determinism)"
    );

    // === Cross-turn speaker continuity (CAMPPlus x-vector): the assistant is the same speaker across
    // turns. Required — the DoD's voice/prosody-continuity proof. ===
    let cb = std::env::var("CHATTERBOX_SNAPSHOT")
        .expect("set CHATTERBOX_SNAPSHOT to a Chatterbox snapshot dir (CAMPPlus speaker encoder)");
    let cos = campplus_cos(&cb);
    let cross_turn_sim = cos(&b0.samples, &b1.samples);
    // A conversational voice-agent's turns are the same synthesized speaker; distinct speakers score
    // well below this in the CAMPPlus space (the voice-clone gate's default↔ref sat ~0.79 for a
    // *matched* speaker, and cross-speaker pairs sit far lower).
    println!("multi-turn: cross-turn speaker similarity (CAMPPlus) {cross_turn_sim:.3}");
    assert!(
        cross_turn_sim > 0.5,
        "the assistant's turns must be the same speaker across the conversation (cross-turn CAMPPlus \
         cosine {cross_turn_sim:.3}) — a later turn drifting to a different voice fails here"
    );
}

/// Unwrap a `session.step` result to its `AudioTrack`, panicking with `ctx` on error.
fn match_audio(
    r: candle_audio_moss_tts_realtime::gen_core::Result<
        candle_audio_moss_tts_realtime::gen_core::AudioTrack,
    >,
    ctx: &str,
) -> candle_audio_moss_tts_realtime::gen_core::AudioTrack {
    r.unwrap_or_else(|e| panic!("{ctx}: {e}"))
}

/// Load `whisper_base` and return a transcribe closure (shared by the multi-turn real-weight tests).
fn whisper_transcriber() -> impl Fn(&candle_audio_moss_tts_realtime::gen_core::AudioTrack) -> String
{
    use candle_audio_whisper::gen_core::{
        AudioTrack as WAudioTrack, LoadSpec as WLoadSpec, TimestampGranularity, TranscribeOptions,
        TranscribeRequest, TranscribeTask, WeightsSource as WWeightsSource,
    };
    let wspec = WLoadSpec::new(WWeightsSource::Dir(PathBuf::from(
        std::env::var("WHISPER_SNAPSHOT")
            .expect("set WHISPER_SNAPSHOT to an openai/whisper-base dir"),
    )));
    let transcriber = candle_audio_whisper::provider_registry()
        .expect("whisper registry")
        .load_transcriber(candle_audio_whisper::MODEL_ID, &wspec)
        .expect("whisper_base loads");
    move |track: &candle_audio_moss_tts_realtime::gen_core::AudioTrack| {
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
    }
}

/// The audio sub-block shared by the multi-turn conversation requests (8 s/turn, en, 24 kHz).
fn conv_audio() -> Option<AudioParams> {
    Some(AudioParams {
        target_duration: Some(8.0),
        language: Some("en".to_string()),
        sample_rate: Some(24_000),
        ..Default::default()
    })
}

/// sc-14151 (M1) — the **user↔assistant** conversation shape (the reference voice-agent case): an
/// assistant turn is synthesized conditioned on a preceding **user** turn (the reference
/// `make_user_prompt` block — the user's own speech delay-aligned on the audio channels). This
/// exercises `conversation::user_body_frames` end-to-end on real weights (the assistant-chaining DoD
/// test does not feed a user turn). The user speech is bootstrapped from the model itself, so the test
/// is self-contained (no external clip). Asserts: A≡B for the round; the assistant reply is
/// intelligible; and the reply is conditioned on the user turn (a different user turn changes it).
#[test]
#[ignore = "real weights: MOSS-TTS-Realtime AR + codec + whisper_base; run with --ignored --nocapture"]
fn moss_tts_realtime_multi_turn_user_context() {
    use moss::gen_core::{AudioTrack, Conditioning, ConversationRole, ConversationTurn};

    let generator = load();
    let transcribe = whisper_transcriber();

    // Bootstrap real "user" speech from the model (self-contained — the user turn needs real audio).
    let user_speech = |text: &str| -> AudioTrack {
        match generator
            .generate(&fidelity_request(text), &mut |_| {})
            .expect("bootstrap user audio")
        {
            GenerationOutput::Audio(t) => t,
            other => panic!("expected Audio, got {other:?}"),
        }
    };
    let user_turn = |text: &str, audio: AudioTrack| ConversationTurn {
        role: ConversationRole::User,
        text: text.to_string(),
        audio: Some(audio),
    };
    let asst = |text: &str| ConversationTurn {
        role: ConversationRole::Assistant,
        text: text.to_string(),
        audio: None,
    };
    let conv_req = |turns: Vec<ConversationTurn>| GenerationRequest {
        prompt: String::new(),
        audio: conv_audio(),
        seed: Some(20_260_719),
        conditioning: vec![Conditioning::ConversationHistory { turns }],
        ..Default::default()
    };
    let open_req = || GenerationRequest {
        prompt: String::new(),
        audio: conv_audio(),
        seed: Some(20_260_719),
        ..Default::default()
    };

    let uc = "What time does the museum open on weekends?";
    let reply = "The museum opens at ten on Saturdays and Sundays.";
    let u_audio = user_speech(uc);

    // Path A: [user, assistant] → the assistant reply is the only synthesized turn, so path A's output
    // equals that single reply.
    let a_reply = match generator
        .generate(
            &conv_req(vec![user_turn(uc, u_audio.clone()), asst(reply)]),
            &mut |_| {},
        )
        .expect("path A: user→assistant round")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    // Path B: fold the user turn into the session, then synthesize the reply.
    let mut session = generator
        .open_conversation(&open_req())
        .expect("open session");
    let _user_echo = match_audio(
        session.step(&user_turn(uc, u_audio.clone()), &mut |_| {}, &mut |_| {}),
        "path B: user turn",
    );
    let b_reply = match_audio(
        session.step(&asst(reply), &mut |_| {}, &mut |_| {}),
        "path B: assistant reply",
    );
    assert_eq!(
        a_reply.samples, b_reply.samples,
        "A≡B for a user→assistant round (the batch render equals the session's reply)"
    );

    let cer = character_error_rate(&normalize(reply), &transcribe(&b_reply));
    println!(
        "multi-turn user-context: reply CER {cer:.3} ({:.2}s)",
        b_reply.samples.len() as f32 / 24_000.0
    );
    assert!(
        cer <= MAX_PROMPT_CER,
        "the assistant reply after a user turn is not intelligible (CER {cer:.3}) — the \
         make_user_prompt user-body conditioning is broken"
    );

    // Conditioned on the user turn: the SAME reply after a DIFFERENT user turn (its own bootstrapped
    // speech) must differ — the reply depends on what the user said (their text AND audio).
    let uc2 = "Where can I find a good vegetarian restaurant nearby?";
    let u2_audio = user_speech(uc2);
    let a_reply2 = match generator
        .generate(
            &conv_req(vec![user_turn(uc2, u2_audio), asst(reply)]),
            &mut |_| {},
        )
        .expect("path A: different user turn")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    let corr = pearson(&a_reply.samples, &a_reply2.samples);
    println!("multi-turn user-context: same-reply-different-user corr {corr:.3}");
    assert!(
        corr < 0.99,
        "the assistant reply must depend on the user turn — the same reply after two different user \
         turns produced near-identical audio (corr {corr:.3}), so the user turn is ignored"
    );
}

/// sc-14151 (M2) — voice cloning **composed with a multi-turn conversation**: the cloned timbre is
/// prefilled once and must be held constant across every turn. Asserts: A≡B still holds when a
/// `ReferenceAudio` clip conditions the whole conversation; both turns are intelligible; and — the
/// DoD's "held constant across the whole conversation" — the cloned output of **every** turn carries
/// the reference speaker more than the default (no-clone) voice does (so the clone threads into the
/// later turn, not only turn 0), and the turns are mutually the same speaker.
#[test]
#[ignore = "real weights: MOSS-TTS-Realtime AR + codec + whisper_base + Chatterbox CAMPPlus; run with --ignored --nocapture"]
fn moss_tts_realtime_multi_turn_voice_clone() {
    use moss::gen_core::{AudioTrack, Conditioning, ConversationRole, ConversationTurn};

    let generator = load();
    let transcribe = whisper_transcriber();
    let ref_clip = read_f32le(&fixture_path(
        "MOSS_VOICECLONE_REF",
        "moss_voiceclone_ref_clip.f32",
    ));
    let ref_track = AudioTrack {
        samples: ref_clip.clone(),
        sample_rate: 24_000,
        channels: 1,
        ..Default::default()
    };
    let clone_cond = Conditioning::ReferenceAudio {
        audio: ref_track,
        strength: None,
    };

    let asst = |text: &str| ConversationTurn {
        role: ConversationRole::Assistant,
        text: text.to_string(),
        audio: None,
    };
    let t0 = "The weather is very nice this afternoon.";
    let t1 = "The train arrives at nine in the morning.";
    // Extra conditioning (the voice clone) rides alongside the ConversationHistory.
    let req = |extra: Vec<Conditioning>| {
        let mut conditioning = vec![Conditioning::ConversationHistory {
            turns: vec![asst(t0), asst(t1)],
        }];
        conditioning.extend(extra);
        GenerationRequest {
            prompt: String::new(),
            audio: conv_audio(),
            seed: Some(20_260_719),
            conditioning,
            ..Default::default()
        }
    };
    let open_req = |extra: Vec<Conditioning>| GenerationRequest {
        prompt: String::new(),
        audio: conv_audio(),
        seed: Some(20_260_719),
        conditioning: extra,
        ..Default::default()
    };
    let step2 = |extra: Vec<Conditioning>| -> (AudioTrack, AudioTrack) {
        let mut s = generator
            .open_conversation(&open_req(extra))
            .expect("open session");
        let a = match_audio(s.step(&asst(t0), &mut |_| {}, &mut |_| {}), "turn 0");
        let b = match_audio(s.step(&asst(t1), &mut |_| {}, &mut |_| {}), "turn 1");
        (a, b)
    };

    // Path A cloned conversation vs path B cloned session — A≡B must still hold WITH the clone.
    let a_clone = match generator
        .generate(&req(vec![clone_cond.clone()]), &mut |_| {})
        .expect("path A: cloned conversation")
    {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    let (c0, c1) = step2(vec![clone_cond.clone()]);
    let b_concat: Vec<f32> = c0.samples.iter().chain(&c1.samples).copied().collect();
    assert_eq!(
        a_clone.samples, b_concat,
        "A≡B must compose with voice cloning (cloned batch render == cloned session steps)"
    );

    // Both cloned turns intelligible.
    for (text, track) in [(t0, &c0), (t1, &c1)] {
        let cer = character_error_rate(&normalize(text), &transcribe(track));
        println!("multi-turn voice-clone: cloned turn {text:?} CER {cer:.3}");
        assert!(
            cer <= MAX_PROMPT_CER,
            "cloned turn {text:?} not intelligible (CER {cer:.3})"
        );
    }

    // Default (no clone) per-turn, for the speaker-identity comparison.
    let (d0, d1) = step2(vec![]);

    let cb = std::env::var("CHATTERBOX_SNAPSHOT")
        .expect("set CHATTERBOX_SNAPSHOT to a Chatterbox snapshot dir (CAMPPlus speaker encoder)");
    let cos = campplus_cos(&cb);
    // The clone must carry the reference speaker in EVERY turn — the DoD's "held constant across the
    // whole conversation". Each cloned turn resembles the reference MORE than the default voice does.
    let (sim_c0, sim_d0) = (cos(&c0.samples, &ref_clip), cos(&d0.samples, &ref_clip));
    let (sim_c1, sim_d1) = (cos(&c1.samples, &ref_clip), cos(&d1.samples, &ref_clip));
    let sim_cross = cos(&c0.samples, &c1.samples);
    println!(
        "multi-turn voice-clone speaker sim (CAMPPlus): turn0 clone↔ref {sim_c0:.3} vs default↔ref \
         {sim_d0:.3}; turn1 clone↔ref {sim_c1:.3} vs default↔ref {sim_d1:.3}; cross-turn clone \
         {sim_cross:.3}"
    );
    assert!(
        sim_c0 > sim_d0 + 0.05,
        "turn 0 must carry the reference speaker (clone↔ref {sim_c0:.3} vs default↔ref {sim_d0:.3})"
    );
    assert!(
        sim_c1 > sim_d1 + 0.05,
        "turn 1 must ALSO carry the reference speaker — the cloned voice must be held constant across \
         the whole conversation, not just the first turn (clone↔ref {sim_c1:.3} vs default↔ref \
         {sim_d1:.3})"
    );
    assert!(
        sim_cross > 0.5,
        "the cloned turns must be the same speaker as each other (cross-turn {sim_cross:.3})"
    );
}
