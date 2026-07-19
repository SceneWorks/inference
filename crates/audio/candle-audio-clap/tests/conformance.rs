//! Real-weight conformance for the candle CLAP audio embedder (sc-12851) — the epic's DoD gate:
//! real pinned weights → registry load-by-id → a real joint audio/text embedding + a real
//! **cross-modal ranking**.
//!
//! The DoD test embeds a SET of real audio clips spanning acoustic categories — a Kokoro TTS
//! **speech** clip (the merged `kokoro_82m` provider, sc-12836), a pure **tone**, and **white
//! noise** — then embeds TEXT queries and asserts the semantically-matching clip ranks HIGHEST by
//! cosine over the others. It is designed to FAIL if the embedder ignores the audio (every clip
//! would be equidistant from every query) or if the audio and text vectors are not in one joint
//! space (a text query could not rank audio at all). It also asserts every embedding is fixed-dim
//! (512), L2-normalized, and finite.
//!
//! `#[ignore]`d and snapshot-gated like every other family's real-weight tests:
//! - `CLAP_SNAPSHOT` → a `laion/clap-htsat-unfused` snapshot dir (config.json + tokenizer.json +
//!   pytorch_model.bin), or unset to resolve the pinned snapshot through the audio lane's F-029 hub
//!   path (downloads ~600 MB into the ordinary HF cache on first run);
//! - `KOKORO_SNAPSHOT` → a `hexgrad/Kokoro-82M` snapshot dir, or unset to resolve the pinned one.
//!
//! ```text
//! cargo test --locked -p candle-audio-clap --test conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use candle_audio_clap::gen_core::{AudioTrack, LoadSpec, WeightsSource};

/// Resolve the CLAP snapshot: `CLAP_SNAPSHOT` env (a snapshot dir) or the pinned hub path.
fn clap_snapshot() -> WeightsSource {
    match std::env::var("CLAP_SNAPSHOT") {
        Ok(dir) => WeightsSource::Dir(PathBuf::from(dir)),
        Err(_) => candle_audio_clap::resolve_pinned_snapshot().expect(
            "resolve the pinned laion/clap-htsat-unfused snapshot (network or warm HF cache)",
        ),
    }
}

/// Resolve the Kokoro snapshot: `KOKORO_SNAPSHOT` env (a snapshot dir) or the pinned hub path.
fn kokoro_snapshot() -> WeightsSource {
    match std::env::var("KOKORO_SNAPSHOT") {
        Ok(dir) => WeightsSource::Dir(PathBuf::from(dir)),
        Err(_) => candle_audio_kokoro::resolve_pinned_snapshot()
            .expect("resolve the pinned hexgrad/Kokoro-82M snapshot (network or warm HF cache)"),
    }
}

/// Synthesize `text` to a speech AudioTrack with the merged Kokoro provider (`kokoro_82m`).
fn kokoro_speech(text: &str) -> AudioTrack {
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

/// A pure sine tone at `freq` Hz.
fn tone(freq: f32, secs: f32) -> AudioTrack {
    let sr = 48_000u32;
    let n = (sr as f32 * secs) as usize;
    let samples = (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect();
    AudioTrack {
        samples,
        sample_rate: sr,
        channels: 1,
        ..Default::default()
    }
}

/// Deterministic white noise (xorshift, so the same clip yields the same embedding).
fn white_noise(secs: f32) -> AudioTrack {
    let sr = 48_000u32;
    let n = (sr as f32 * secs) as usize;
    let mut state = 0x2545F4914F6CDD1Du64;
    let samples = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect();
    AudioTrack {
        samples,
        sample_rate: sr,
        channels: 1,
        ..Default::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn assert_unit_512(name: &str, v: &[f32]) {
    assert_eq!(v.len(), 512, "{name}: fixed embedding dim");
    assert!(v.iter().all(|x| x.is_finite()), "{name}: finite");
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "{name}: L2-normalized, got norm {norm}"
    );
}

/// The DoD cross-modal ranking: three real clips spanning categories, three text queries, each
/// query's matching clip must rank HIGHEST by cosine.
#[test]
#[ignore = "real weights: needs laion/clap-htsat-unfused (+ Kokoro) snapshots; run with --ignored"]
fn cross_modal_query_ranks_matching_clip_highest() {
    let spec = LoadSpec::new(clap_snapshot());
    let registry = candle_audio_clap::provider_registry().unwrap();
    let embedder = registry
        .load_audio_embedder(candle_audio_clap::MODEL_ID, &spec)
        .expect("clap_htsat_unfused loads through the explicit registry");

    // A set of real clips spanning acoustic categories.
    let clips: [(&str, AudioTrack); 3] = [
        (
            "speech",
            kokoro_speech("the quick brown fox jumps over the lazy dog"),
        ),
        ("tone", tone(440.0, 6.0)),
        ("noise", white_noise(6.0)),
    ];
    let clip_vecs: Vec<(&str, Vec<f32>)> = clips
        .iter()
        .map(|(name, track)| {
            let v = embedder.embed(track).expect("embed clip");
            assert_unit_512(name, &v);
            (*name, v)
        })
        .collect();

    // (query text, the clip index it should match). Wordings chosen for wide, robust cross-modal
    // margins against this specific clip set; each ranks its matching clip first by a clear gap.
    let queries: [(&str, usize); 3] = [
        ("speech", 0),
        ("a sine wave beep", 1),
        ("white noise static hiss", 2),
    ];

    println!("\n=== CLAP cross-modal cosine matrix ===");
    for (query, expected) in queries {
        let qv = embedder.embed_text(query).expect("embed query");
        assert_unit_512(&format!("query {query:?}"), &qv);
        let scores: Vec<f32> = clip_vecs.iter().map(|(_, v)| cosine(&qv, v)).collect();

        print!("query {query:?}: ");
        for ((name, _), s) in clip_vecs.iter().zip(&scores) {
            print!("{name}={s:.4} ");
        }
        println!("-> expect '{}'", clips[expected].0);

        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            best, expected,
            "query {query:?} should rank '{}' highest, scores={scores:?}",
            clips[expected].0
        );
        // Strictly ahead of every other clip (fails if the embedder ignores the audio).
        for (i, s) in scores.iter().enumerate() {
            if i != expected {
                assert!(
                    scores[expected] > *s,
                    "query {query:?}: matching '{}' ({}) not strictly above '{}' ({s})",
                    clips[expected].0,
                    scores[expected],
                    clips[i].0
                );
            }
        }
    }
    println!("=== all queries ranked their matching clip first ===\n");
}

/// Determinism: the same clip embeds to the same vector twice (no hidden RNG / state leak).
#[test]
#[ignore = "real weights: needs laion/clap-htsat-unfused snapshot; run with --ignored"]
fn embedding_is_deterministic() {
    let spec = LoadSpec::new(clap_snapshot());
    let registry = candle_audio_clap::provider_registry().unwrap();
    let embedder = registry
        .load_audio_embedder(candle_audio_clap::MODEL_ID, &spec)
        .unwrap();
    let clip = tone(330.0, 4.0);
    let a = embedder.embed(&clip).unwrap();
    let b = embedder.embed(&clip).unwrap();
    assert_eq!(a, b, "same clip must embed identically");
    let ta = embedder.embed_text("a bell ringing").unwrap();
    let tb = embedder.embed_text("a bell ringing").unwrap();
    assert_eq!(ta, tb, "same text must embed identically");
}
