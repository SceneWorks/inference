use std::cell::Cell;

use candle_audio::candle_core::Device;
use candle_audio_yue2::CotMode;

use super::*;
use crate::provider::tests::{tiny_files, tiny_identity};
use crate::review::tests::{tokenizer, transcription, REAL_TOKENS, SILENCE_TOKENS};

const STYLE: &str = "English, warm female vocal, gentle acoustic folk, fingerpicked guitar, 80 BPM";
const LYRICS: &str = "[Verse]\nO say can you see by the dawn's early light\nWhat so proudly we \
                      hailed at the twilight's last gleaming\n";

fn saved(tokens: &str, duration: f64) -> (tempfile::TempDir, ReviewArtifact) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("transcription");
    transcription(tokens, duration)
        .save(&dir, &tokenizer())
        .unwrap();
    let artifact = ReviewArtifact::open(&dir).unwrap();
    (root, artifact)
}

/// The melody-only cover plans from the transcription's melody-only score (no chord symbols) with
/// `cot=melody`; the full cover plans from the full score with `cot=full`. Provenance names the
/// transcription, the score and the request.
#[test]
fn a_reviewed_transcription_plans_melody_and_full_covers() {
    let (_root, artifact) = saved(REAL_TOKENS, 60.0);
    let melody = plan_cover(
        &artifact,
        &CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS)),
    )
    .unwrap();
    assert_eq!(melody.prepared.request.cot(), CotMode::Melody);
    let melody_score = String::from_utf8(artifact.read(MELODY_SCORE).unwrap()).unwrap();
    assert_eq!(melody.prepared.request.abc(), Some(melody_score.as_str()));
    assert_eq!(melody.prepared.report.chords_removed, 0);
    assert_eq!(
        melody.provenance["transcription"]["manifest_sha256"],
        artifact.manifest_sha256()
    );
    assert_eq!(
        melody.provenance["score"]["record"]["source"],
        "transcribed"
    );
    assert!(melody.provenance["transcription"]["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w["code"] == "harmony_collapsed"));

    let full = plan_cover(
        &artifact,
        &CoverOptions::new(CoverMode::Full, STYLE, CoverLyrics::source(LYRICS)),
    )
    .unwrap();
    assert_eq!(full.prepared.request.cot(), CotMode::Full);
    let full_score = String::from_utf8(artifact.read("score.abc").unwrap()).unwrap();
    assert_eq!(full.prepared.request.abc(), Some(full_score.as_str()));

    // A reviewer's edited full score is stripped for a melody cover, and the melody change is
    // recorded.
    let edited = full_score.replacen("\"C\"g2e2|", "\"C\"a2e2|", 1);
    assert_ne!(edited, full_score);
    let reviewed = plan_cover(
        &artifact,
        &CoverOptions {
            reviewed_score: Some(edited),
            ..CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS))
        },
    )
    .unwrap();
    assert!(reviewed.prepared.report.chords_removed > 0);
    assert_eq!(reviewed.provenance["score"]["record"]["source"], "reviewed");
    let first = reviewed.provenance["score"]["record"]["melody_differences_from_transcription"][0]
        .as_str()
        .unwrap();
    assert!(
        first.starts_with("Vocal: sounding notes differ starting at note 1"),
        "{first}"
    );
}

/// The reviewer's forgery: an empty transcription whose manifest is edited to claim a ready melody
/// cover, no warnings and a different source. It cannot be opened, so no cover can be planned from
/// it; and a verified artifact whose files change after opening is refused by `plan_cover` itself.
///
/// Mutations that must fail: drop the manifest comparison in `replay`, or the `replay()` call in
/// `plan_cover`.
#[test]
fn a_forged_or_altered_artifact_cannot_be_planned() {
    let (root, artifact) = saved(SILENCE_TOKENS, 12.0);
    let dir = root.path().join("transcription");
    let manifest_path = dir.join(crate::review::MANIFEST);
    let original = std::fs::read(&manifest_path).unwrap();
    let mut forged: Value = serde_json::from_slice(&original).unwrap();
    forged["review"]["cover"]["melody"] = json!({"ready": true});
    forged["review"]["warnings"] = json!([]);
    forged["source"]["sha256"] = json!("1".repeat(64));
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&forged).unwrap()).unwrap();
    assert!(ReviewArtifact::open(&dir).is_err());
    std::fs::write(&manifest_path, &original).unwrap();

    // Opened while intact, altered afterwards: plan_cover replays and refuses.
    let score = dir.join(MELODY_SCORE);
    std::fs::write(&score, b"X:1\n").unwrap();
    let err = plan_cover(
        &artifact,
        &CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS)),
    )
    .unwrap_err();
    assert!(matches!(err, Error::Replay(_)), "{err}");
}

/// A transcription with no melody cannot be covered as transcribed (upstream would hand YuE2 a
/// rest-only score); a reviewer-supplied score replaces it explicitly.
///
/// Mutation that must fail: ignore the recorded readiness in `plan_cover`.
#[test]
fn an_empty_transcription_is_refused_unless_a_reviewed_score_replaces_it() {
    let (_root, artifact) = saved(SILENCE_TOKENS, 12.0);
    let err = plan_cover(
        &artifact,
        &CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS)),
    )
    .unwrap_err();
    assert!(err.to_string().contains("no melody"), "{err}");
    let (_root2, real) = saved(REAL_TOKENS, 60.0);
    let score = String::from_utf8(real.read(MELODY_SCORE).unwrap()).unwrap();
    let plan = plan_cover(
        &artifact,
        &CoverOptions {
            reviewed_score: Some(score),
            ..CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS))
        },
    )
    .unwrap();
    assert_eq!(plan.provenance["score"]["record"]["source"], "reviewed");
}

struct StandIn<'a> {
    live_at_load: usize,
    generated: &'a Cell<bool>,
}

impl CoverGenerator for StandIn<'_> {
    fn generate(
        &self,
        request: &SongRequest,
        run_dir: &Path,
        _: &dyn Fn() -> bool,
    ) -> Result<Value, Error> {
        assert_eq!(self.live_at_load, 0);
        std::fs::create_dir_all(run_dir).unwrap();
        self.generated.set(true);
        Ok(json!({"status": "complete", "identity": "stand-in", "cot": request.cot().as_str()}))
    }
}

/// The transcriber is unloaded — observably — before the generator is even loaded, and a
/// SheetSage2 model still alive anywhere in the process blocks generation.
///
/// Mutations that must fail: load the generator before unloading, or skip the `live_models`
/// check.
#[test]
fn generation_starts_only_after_the_transcription_model_is_gone() {
    let _serial = crate::test_lock();
    let (_root, artifact) = saved(REAL_TOKENS, 60.0);
    let plan = plan_cover(
        &artifact,
        &CoverOptions::new(CoverMode::Melody, STYLE, CoverLyrics::source(LYRICS)),
    )
    .unwrap();
    let out = tempfile::tempdir().unwrap();
    let generated = Cell::new(false);
    let transcriber = Transcriber::from_files(tiny_files(), tiny_identity(), &Device::Cpu).unwrap();
    assert_eq!(live_models(), 1);
    let outcome = run_cover(
        Some(transcriber),
        &plan,
        || {
            Ok(StandIn {
                live_at_load: live_models(),
                generated: &generated,
            })
        },
        &out.path().join("cover"),
        &|| false,
    )
    .unwrap();
    assert!(generated.get());
    let receipt = outcome.unload.unwrap();
    assert!(receipt.released);
    assert_eq!(receipt.live_models_after, 0);
    let written: Value =
        serde_json::from_slice(&std::fs::read(&outcome.provenance_path).unwrap()).unwrap();
    assert_eq!(
        written["transcription_unloaded_before_generation"]
            ["live_sheetsage2_models_at_generator_load"],
        0
    );
    assert_eq!(written["request"]["cot"], "melody");
    assert_eq!(written["run"]["identity"], "stand-in");

    // A model that was not handed in (still loaded elsewhere) blocks the load.
    let straggler = Transcriber::from_files(tiny_files(), tiny_identity(), &Device::Cpu).unwrap();
    let loaded = Cell::new(false);
    let err = run_cover(
        None,
        &plan,
        || {
            loaded.set(true);
            Ok(StandIn {
                live_at_load: live_models(),
                generated: &generated,
            })
        },
        &out.path().join("cover2"),
        &|| false,
    )
    .unwrap_err();
    assert!(err.to_string().contains("still loaded"), "{err}");
    assert!(!loaded.get(), "the generator must not be loaded");
    straggler.unload();
}
