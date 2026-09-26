//! The review artifact on upstream's committed token oracles (no model needed): assessment,
//! persistence, integrity, replay, and the refusals.

use super::*;
use crate::events::parse_tokens_txt;
use crate::pipeline::{DEFAULT_LOOKAHEAD_SECONDS, DEFAULT_OVERLAP_SECONDS, MAX_OUTPUT_SEQ_LEN};
use crate::tokenizer::FULL_TASK_PROMPTS;

pub(crate) const REAL_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_full/tokens.txt");
pub(crate) const SILENCE_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/silence/tokens.txt");
pub(crate) const SYNTH_TOKENS: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/synth_full/tokens.txt");

pub(crate) fn tokenizer() -> Tokenizer {
    Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap()
}

pub(crate) fn identity() -> ClosureIdentity {
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

pub(crate) fn transcription(tokens: &str, duration: f64) -> Transcription {
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
    // A stand-in model input of the recorded length (the oracles carry tokens, not audio): digital
    // silence, so the octave evidence is computed but finds nothing.
    let audio = vec![0.0f32; (duration * 24_000.0).round() as usize];
    let source = SourceIdentity {
        sha256: samples_sha256(&audio),
        samples: audio.len(),
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
        audio,
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

    // A self-consistent forgery (tokens changed AND their digest updated in the manifest) is
    // caught by the replay inside `open`. The digest is replaced textually so nothing else in the
    // manifest changes.
    let tokens_path = dir.join(TOKENS_JSON);
    let original_tokens = std::fs::read(&tokens_path).unwrap();
    let mut windows: Vec<Value> = serde_json::from_slice(&original_tokens).unwrap();
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
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let edited = manifest.replace(&sha256_hex(&original_tokens), &sha256_hex(&forged));
    assert_ne!(edited, manifest);
    std::fs::write(&manifest_path, edited).unwrap();
    let err = ReviewArtifact::open(&dir).unwrap_err();
    assert!(err.to_string().contains("differs"), "{err}");

    // `read` re-hashes: a file replaced after `open` is refused.
    std::fs::write(&tokens_path, &original_tokens).unwrap();
    std::fs::write(&manifest_path, manifest).unwrap();
    let artifact = ReviewArtifact::open(&dir).unwrap();
    std::fs::write(&score, b"X:1\n").unwrap();
    assert!(artifact.read("score.abc").is_err());
}

/// Every manifest field that is not a file digest is verified too: a hand-edited cover readiness,
/// a removed warning, an ABC error, a note count, a source digest or a closure digest is refused
/// by `open` — so no `ReviewArtifact`, and no cover plan, can be made from it.
///
/// Mutations that must fail: compare only the artifact digests in `replay` (the pre-fix
/// behaviour), or skip `check_pinned_closure`.
#[test]
fn hand_edited_manifest_fields_are_refused() {
    let t = transcription(SILENCE_TOKENS, 12.0);
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("review");
    t.save(&dir, &tokenizer()).unwrap();
    let manifest_path = dir.join(MANIFEST);
    let original = std::fs::read(&manifest_path).unwrap();
    let parsed: Value = serde_json::from_slice(&original).unwrap();
    // Control: rewriting the manifest unchanged keeps it valid, so each refusal below is caused by
    // its edit, not by re-serialization.
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&parsed).unwrap()).unwrap();
    assert!(ReviewArtifact::open(&dir).is_ok());
    type Edit = Box<dyn Fn(&mut Value)>;
    let edits: Vec<(&str, Edit)> = vec![
        (
            "review.cover.melody forced ready",
            Box::new(|m| m["review"]["cover"]["melody"] = json!({"ready": true})),
        ),
        (
            "review.warnings emptied",
            Box::new(|m| m["review"]["warnings"] = json!([])),
        ),
        (
            "abc_error.full invented",
            Box::new(|m| m["abc_error"]["full"] = json!("forged")),
        ),
        (
            "melody_notes inflated",
            Box::new(|m| m["melody_notes"] = json!(12)),
        ),
        (
            "source.sha256 replaced",
            Box::new(|m| m["source"]["sha256"] = json!("1".repeat(64))),
        ),
        (
            "closure weights digest replaced",
            Box::new(|m| m["closure"]["sheetsage2"]["weights_sha256"] = json!("2".repeat(64))),
        ),
        (
            "closure device invented",
            Box::new(|m| m["closure"]["device"] = json!("tpu")),
        ),
    ];
    for (label, edit) in &edits {
        let mut forged = parsed.clone();
        edit(&mut forged);
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&forged).unwrap()).unwrap();
        let err = ReviewArtifact::open(&dir).unwrap_err();
        assert!(matches!(err, Error::Replay(_)), "{label}: {err}");
    }
    std::fs::write(&manifest_path, &original).unwrap();
    assert!(ReviewArtifact::open(&dir).is_ok());
}
