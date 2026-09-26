//! The review artifact on upstream's committed token oracles (no model needed): assessment,
//! persistence, integrity, replay, and the refusals.

use super::*;
use crate::events::parse_tokens_txt;
use crate::pipeline::{DEFAULT_LOOKAHEAD_SECONDS, DEFAULT_OVERLAP_SECONDS, MAX_OUTPUT_SEQ_LEN};
use crate::tokenizer::FULL_TASK_PROMPTS;

const REAL_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_full/tokens.txt");
const SILENCE_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/silence/tokens.txt");
const SYNTH_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/synth_full/tokens.txt");

fn tokenizer() -> Tokenizer {
    Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap()
}

fn identity() -> ClosureIdentity {
    ClosureIdentity {
        sheetsage2: [
            "m-a-p/SheetSage2".into(),
            "eab522a8168e8b8b8c4856bf8609cd86198f01fe".into(),
            "b235f68091a5f5b644000f2b5acb57d1e70432aca2b34ab1b9cf27236e1f4274".into(),
            "a986e63f5d831ecb823c11d19cfb371d763f25ae2e614ec9acd714c0b8bb87fd".into(),
        ],
        mert: [
            "m-a-p/MERT-v2-FullSong".into(),
            "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42".into(),
            "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d".into(),
            "f2e194895f58be3ddba327255db129ff0e3bee550cc0ecf08e4d22d79ce3bca3".into(),
        ],
        ported_code_revision: crate::PORTED_CODE_REVISION.into(),
        tokenizer_fingerprint: "5ba3325af0344c7f".into(),
        device: "cpu".into(),
    }
}

fn transcription(tokens: &str, duration: f64) -> Transcription {
    let tokenizer = tokenizer();
    let windows: Vec<Vec<u32>> = parse_tokens_txt(tokens)
        .unwrap()
        .into_iter()
        .map(|r| r.tokens)
        .collect();
    let stitched = Stitcher::new(
        &tokenizer,
        &FULL_TASK_PROMPTS,
        duration,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_LOOKAHEAD_SECONDS,
        MAX_OUTPUT_SEQ_LEN,
    )
    .unwrap()
    .replay(&windows)
    .unwrap();
    let samples = (duration * 24_000.0).round() as usize;
    let source = SourceIdentity {
        sha256: "0".repeat(64),
        samples,
        sample_rate: 24_000,
        name: Some("fixture".into()),
        original_sha256: None,
        conversion: "none".into(),
    };
    Transcription::build(
        source,
        TranscriptionSettings::default(),
        identity(),
        stitched,
        None,
    )
    .unwrap()
}

fn codes(review: &Review) -> Vec<&str> {
    review.warnings.iter().map(|w| w.code.as_str()).collect()
}

/// The real recording's known problems are surfaced, not fixed: the collapsed harmony and the high
/// vocal register are warnings; both cover modes stay available after review; the full and
/// melody-only scores are upstream's byte for byte.
///
/// Mutations that must fail: drop the harmony-collapse check, or build the melody-only score with
/// `melody_only = false`.
#[test]
fn real_recording_is_reviewable_with_its_problems_visible() {
    let t = transcription(REAL_TOKENS, 60.0);
    let codes = codes(&t.review);
    assert!(codes.contains(&"harmony_collapsed"), "{codes:?}");
    assert!(codes.contains(&"octave_high_register"), "{codes:?}");
    assert!(!codes.contains(&"empty_melody"));
    assert_eq!(t.review.melody_cover, Readiness::Ready);
    assert_eq!(t.review.full_cover, Readiness::Ready);
    assert_eq!(t.review.voices[0].notes, 76);
    assert_eq!(t.review.voices[1].notes, 0);
    assert_eq!(t.review.bars, 27);
    assert_eq!(
        t.full.abc.as_deref(),
        Some(include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/real_full/score.abc"
        ))
    );
    assert_eq!(
        t.melody_abc.as_deref(),
        Some(include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/real_melody/score.abc"
        ))
    );
}

/// Digital silence: upstream returns a rest-only score silently. Here both cover modes are refused
/// with the reason, the warning is recorded, and the partial artifacts (rest-only score, beats,
/// key) are still produced for review.
///
/// Mutation that must fail: treat `melody_notes == 0` as ready (the upstream behaviour).
#[test]
fn empty_melody_is_refused_but_kept_reviewable() {
    let t = transcription(SILENCE_TOKENS, 12.0);
    assert!(codes(&t.review).contains(&"empty_melody"));
    assert!(matches!(t.review.melody_cover, Readiness::Refused(ref r) if r.contains("no melody")));
    assert!(matches!(t.review.full_cover, Readiness::Refused(_)));
    assert!(
        t.full.abc.is_some(),
        "the rest-only score is kept for review"
    );
    assert!(t.full.text("beat.lab").is_some());
}

#[test]
fn synthetic_melody_voiced_as_ins_is_flagged() {
    let t = transcription(SYNTH_TOKENS, 39.4);
    assert!(codes(&t.review).contains(&"melody_only_in_ins"));
    assert!(!codes(&t.review).contains(&"harmony_collapsed"));
}

/// Persist → open (integrity) → replay (byte-identical re-derivation) → provenance round trip.
///
/// Mutations that must fail: skip the digest check in `open`, or replay without comparing bytes.
#[test]
fn persisted_artifact_replays_byte_for_byte_and_detects_tampering() {
    let t = transcription(REAL_TOKENS, 60.0);
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("review");
    let manifest_sha = t.save(&dir, &tokenizer()).unwrap();
    let artifact = ReviewArtifact::open(&dir).unwrap();
    assert_eq!(artifact.manifest_sha256(), manifest_sha);
    assert_eq!(
        artifact.settings().unwrap(),
        TranscriptionSettings::default()
    );
    assert_eq!(artifact.closure().unwrap(), identity());
    assert_eq!(artifact.source().unwrap().samples, 1_440_000);
    assert_eq!(artifact.readiness(true), Readiness::Ready);
    let report = artifact.replay().unwrap();
    assert_eq!(report.windows, 1);
    assert!(report.artifacts_matched >= 25, "{report:?}");
    assert_eq!(
        artifact.read("score.abc").unwrap(),
        t.full.abc.clone().unwrap().into_bytes()
    );
    assert!(artifact.read(MELODY_SCORE).is_ok());
    assert!(artifact.read("../escape").is_err());

    // A fresh directory is required.
    assert!(t.save(&dir, &tokenizer()).is_err());

    // Tampering with a file is caught on open.
    let score = dir.join("score.abc");
    let original = std::fs::read(&score).unwrap();
    std::fs::write(&score, b"X:1\n").unwrap();
    assert!(ReviewArtifact::open(&dir).is_err());
    std::fs::write(&score, &original).unwrap();

    // A self-consistent forgery (tokens changed AND digest updated) is caught by replay.
    let tokens_path = dir.join(TOKENS_JSON);
    let mut windows: Vec<Value> =
        serde_json::from_slice(&std::fs::read(&tokens_path).unwrap()).unwrap();
    let list = windows[0]["tokens"].as_array_mut().unwrap();
    // Raise the first melody pitch token by a semitone.
    let at = list
        .iter()
        .position(|t| (31_398..31_526).contains(&t.as_u64().unwrap()))
        .unwrap();
    let pitch = list[at].as_u64().unwrap();
    list[at] = json!(pitch + 1);
    let forged = serde_json::to_vec_pretty(&windows).unwrap();
    std::fs::write(&tokens_path, &forged).unwrap();
    let manifest_path = dir.join(MANIFEST);
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["artifacts"][TOKENS_JSON] = json!(sha256_hex(&forged));
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let reopened = ReviewArtifact::open(&dir).unwrap();
    let err = reopened.replay().unwrap_err();
    assert!(err.to_string().contains("differs"), "{err}");
}
