//! Plan save / restore parity with upstream `SymbolicPlan` (plans saved by the pinned upstream
//! under `tests/fixtures/protocol/plans/`), and the integrity refusals.

use std::path::Path;

use serde_json::{json, Value};

use super::*;
use crate::protocol::{SamplingOverrides, ABC_END, ABC_START, MUSIC_START};
use crate::test_fixtures::{dir as fixture_dir, ids, json as fixture, synthetic};

const PLANS: [&str; 4] = [
    "full_generated",
    "melody_generated_truncated",
    "melody_external",
    "off",
];

fn record(name: &str) -> Value {
    fixture("plans.json")["plans"][name].clone()
}

fn upstream_dir(name: &str) -> std::path::PathBuf {
    fixture_dir().join("plans").join(name)
}

/// A private, writable copy of an upstream-saved plan.
fn copy(name: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for entry in fs::read_dir(upstream_dir(name)).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    tmp
}

/// Recompute `plan_manifest.json` over the files it lists (what someone editing a plan in place
/// would do to get past the hash check).
fn rehash(dir: &Path) {
    let path = dir.join(PLAN_MANIFEST);
    let manifest: Map<String, Value> = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let rehashed: Map<String, Value> = manifest
        .keys()
        .map(|name| {
            let digest = hex(&Sha256::digest(fs::read(dir.join(name)).unwrap()));
            (name.clone(), Value::String(digest))
        })
        .collect();
    fs::write(&path, serde_json::to_vec(&rehashed).unwrap()).unwrap();
}

fn edit_plan_json(dir: &Path, edit: impl FnOnce(&mut Map<String, Value>)) {
    let path = dir.join(PLAN_JSON);
    let mut data: Map<String, Value> = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut data);
    fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
}

fn restore(dir: &Path) -> Result<SymbolicPlan, PlanError> {
    SymbolicPlan::restore(dir, synthetic())
}

/// Every plan upstream saved restores to its exact recorded token IDs, prefix, text and flags.
#[test]
fn upstream_saved_plans_restore_exactly() {
    for name in PLANS {
        let r = record(name);
        let plan = restore(&upstream_dir(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(plan.abc_ids(), ids(&r["abc_ids"]), "{name}");
        assert_eq!(plan.prefix(), ids(&r["prefix"]), "{name}");
        assert_eq!(plan.abc(), r["abc"].as_str(), "{name}");
        assert_eq!(
            plan.truncated(),
            r["truncated"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(Value::Object(plan.timing().clone()), r["timing"], "{name}");
        assert_eq!(
            plan.request().to_json()["seed"],
            r["request"]["seed"],
            "{name}"
        );
        let kind = match r["kind"].as_str().unwrap() {
            "off" => PlanKind::Off,
            "external" => PlanKind::External,
            _ => PlanKind::Generated,
        };
        assert_eq!(plan.kind(), kind, "{name}");
    }
    // The truncated plan's text really does end mid-character, so its IDs are not recoverable
    // from the text: restore must read them from the arrays.
    let truncated = restore(&upstream_dir("melody_generated_truncated")).unwrap();
    assert!(truncated.abc().unwrap().ends_with('\u{fffd}'));
    assert_ne!(
        synthetic().encode(truncated.abc().unwrap()).unwrap(),
        truncated.abc_ids()
    );
}

/// Planning natively (upstream `pipeline.plan` branches) builds the same plans upstream saved.
#[test]
fn native_planning_rebuilds_the_upstream_plans() {
    let tok = synthetic();
    for name in PLANS {
        let upstream = restore(&upstream_dir(name)).unwrap();
        let native = match SymbolicPlan::prepare(upstream.request().clone(), tok).unwrap() {
            PlanStep::Ready(plan) => plan,
            PlanStep::GenerateAbc(planning) => {
                assert_eq!(*planning.prefix().last().unwrap(), ABC_START);
                planning
                    .finish(
                        tok,
                        upstream.abc_ids().to_vec(),
                        upstream.timing().clone(),
                        upstream.truncated(),
                    )
                    .unwrap()
            }
        };
        assert_eq!(native, upstream, "{name}");
        assert_eq!(native.identity(), upstream.identity(), "{name}");
    }
}

/// A native save is upstream's layout: the token arrays and score are byte-identical to the
/// files upstream wrote, and the save restores to the same plan and identity.
#[test]
fn native_save_matches_upstream_layout_and_round_trips() {
    for name in PLANS {
        let plan = restore(&upstream_dir(name)).unwrap();
        let out = tempfile::tempdir().unwrap();
        let identity = plan.save(out.path()).unwrap();
        for file in [ABC_TOKENS_NPY, PREFIX_NPY, SCORE_ABC] {
            let upstream = fs::read(upstream_dir(name).join(file)).ok();
            let native = fs::read(out.path().join(file)).ok();
            assert_eq!(native, upstream, "{name}/{file}");
        }
        let restored = SymbolicPlan::restore_expecting(out.path(), synthetic(), &identity).unwrap();
        assert_eq!(restored, plan, "{name}");
    }
}

#[test]
fn save_never_overwrites_a_saved_plan() {
    let plan = restore(&upstream_dir("full_generated")).unwrap();
    let out = tempfile::tempdir().unwrap();
    plan.save(out.path()).unwrap();
    assert!(matches!(
        plan.save(out.path()),
        Err(PlanError::AlreadySaved { .. })
    ));
    // A stray score from something else is refused too, so a cot=off save cannot leave it behind.
    let off = restore(&upstream_dir("off")).unwrap();
    let other = tempfile::tempdir().unwrap();
    fs::write(other.path().join(SCORE_ABC), "X:1").unwrap();
    assert!(matches!(
        off.save(other.path()),
        Err(PlanError::AlreadySaved { .. })
    ));
}

/// Editing the score in place is caught by the manifest hash.
#[test]
fn edited_score_is_refused() {
    let dir = copy("full_generated");
    fs::write(dir.path().join(SCORE_ABC), "X:1\nK:C\nC D E F|\n").unwrap();
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(err, PlanError::Changed { ref file } if file == SCORE_ABC),
        "{err}"
    );
}

/// Recomputing the manifest after editing the score does not help: the score must equal the
/// recorded ABC text.
#[test]
fn edited_score_with_recomputed_manifest_is_refused() {
    let dir = copy("full_generated");
    fs::write(dir.path().join(SCORE_ABC), "X:1\nK:C\nC D E F|\n").unwrap();
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(matches!(err, PlanError::AbcTextMismatch(_)), "{err}");
}

/// Rewriting the score and `plan.json`'s text (manifest recomputed) while keeping the sampled IDs
/// is the masquerade upstream's `load` lets through: the text no longer decodes from the IDs.
#[test]
fn edited_text_over_unchanged_ids_is_refused() {
    let dir = copy("full_generated");
    let edited = "X:1\nK:C\nC D E F|\n";
    fs::write(dir.path().join(SCORE_ABC), edited).unwrap();
    edit_plan_json(dir.path(), |d| {
        d.insert("abc".into(), Value::from(edited));
    });
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(err, PlanError::Inconsistent(ref m) if m.contains("decoding")),
        "{err}"
    );
}

/// An external plan whose request ABC was edited (everything textual rewritten, IDs kept) is
/// refused: the IDs are not the encoded external ABC.
#[test]
fn edited_external_abc_over_unchanged_ids_is_refused() {
    let dir = copy("melody_external");
    let edited = "X:1\nK:G\nG A B c|\n";
    fs::write(dir.path().join(SCORE_ABC), edited).unwrap();
    edit_plan_json(dir.path(), |d| {
        d.insert("abc".into(), Value::from(edited));
        d["request"]["abc"] = Value::from(edited);
    });
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(err, PlanError::Inconsistent(ref m) if m.contains("encoded")),
        "{err}"
    );
}

/// Changing the IDs but not the prefix (arrays and manifest updated) is refused: the prefix is
/// re-derived from the request and exact IDs.
#[test]
fn ids_that_disagree_with_the_prefix_are_refused() {
    let dir = copy("full_generated");
    let plan = restore(&upstream_dir("full_generated")).unwrap();
    let mut abc_ids = plan.abc_ids().to_vec();
    abc_ids.pop();
    let abc = synthetic().decode(&abc_ids);
    fs::write(dir.path().join(SCORE_ABC), &abc).unwrap();
    fs::write(
        dir.path().join(ABC_TOKENS_NPY),
        npy_int32(&abc_ids).unwrap(),
    )
    .unwrap();
    edit_plan_json(dir.path(), |d| {
        d.insert("abc".into(), Value::from(abc));
        d.insert("abc_ids".into(), Value::from(abc_ids));
    });
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(err, PlanError::Inconsistent(ref m) if m.contains("prefix")),
        "{err}"
    );
}

/// Token arrays must equal `plan.json`'s lists.
#[test]
fn token_array_mismatch_is_refused() {
    for (file, field) in [(ABC_TOKENS_NPY, "abc_ids"), (PREFIX_NPY, "prefix")] {
        let dir = copy("full_generated");
        let plan = restore(&upstream_dir("full_generated")).unwrap();
        let mut values = if field == "prefix" {
            plan.prefix().to_vec()
        } else {
            plan.abc_ids().to_vec()
        };
        values[1] += 1;
        fs::write(dir.path().join(file), npy_int32(&values).unwrap()).unwrap();
        rehash(dir.path());
        let err = restore(dir.path()).unwrap_err();
        assert!(
            matches!(err, PlanError::TokenArrayMismatch(f) if f == file),
            "{file}: {err}"
        );
    }
}

#[test]
fn manifest_structure_is_enforced() {
    let dir = copy("full_generated");
    let path = dir.path().join(PLAN_MANIFEST);
    let original = fs::read(&path).unwrap();
    let mut manifest: Map<String, Value> = serde_json::from_slice(&original).unwrap();
    manifest.remove(PREFIX_NPY);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        restore(dir.path()),
        Err(PlanError::InvalidArtifact(_))
    ));

    let mut manifest: Map<String, Value> = serde_json::from_slice(&original).unwrap();
    fs::write(dir.path().join("audio.flac"), b"x").unwrap();
    manifest.insert("audio.flac".into(), Value::from(hex(&Sha256::digest(b"x"))));
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(matches!(
        restore(dir.path()),
        Err(PlanError::InvalidArtifact(_))
    ));

    fs::write(&path, &original).unwrap();
    restore(dir.path()).unwrap();
}

#[cfg(unix)]
#[test]
fn symlinked_artifacts_are_refused() {
    let dir = copy("full_generated");
    let real = dir.path().join("score.real");
    fs::rename(dir.path().join(SCORE_ABC), &real).unwrap();
    std::os::unix::fs::symlink(&real, dir.path().join(SCORE_ABC)).unwrap();
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(err, PlanError::InvalidArtifact(ref m) if m.contains("symlink")),
        "{err}"
    );
}

/// A cot=off plan cannot acquire a score.
#[test]
fn score_on_a_plan_without_abc_is_refused() {
    let dir = copy("off");
    fs::write(dir.path().join(SCORE_ABC), "X:1\n").unwrap();
    let path = dir.path().join(PLAN_MANIFEST);
    let mut manifest: Map<String, Value> =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest.insert(SCORE_ABC.into(), Value::Null);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(matches!(err, PlanError::AbcTextMismatch(_)), "{err}");
}

/// The saved request is re-validated.
#[test]
fn invalid_saved_request_is_refused() {
    let dir = copy("off");
    edit_plan_json(dir.path(), |d| {
        d["request"]["seed"] = json!(-1);
    });
    rehash(dir.path());
    let err = restore(dir.path()).unwrap_err();
    assert!(
        matches!(
            err,
            PlanError::Protocol(ProtocolError::Invalid { field: "seed", .. })
        ),
        "{err}"
    );
}

/// No secret-free check can stop a directory whose files were all rewritten consistently — that
/// is a new plan. A caller holding the identity it saved refuses it.
#[test]
fn a_consistent_rewrite_is_a_different_plan() {
    let tok = synthetic();
    let original = restore(&upstream_dir("full_generated")).unwrap();
    let saved = tempfile::tempdir().unwrap();
    let identity = original.save(saved.path()).unwrap();

    let forged_ids = tok.encode("X:1\nK:C\nC D E F|\n").unwrap();
    let forged = match SymbolicPlan::prepare(original.request().clone(), tok).unwrap() {
        PlanStep::GenerateAbc(p) => p.finish(tok, forged_ids, Map::new(), false).unwrap(),
        PlanStep::Ready(_) => unreachable!("a full-cot request without ABC is sampled"),
    };
    let rewritten = tempfile::tempdir().unwrap();
    forged.save(rewritten.path()).unwrap();
    for name in ALLOWED.iter().chain([&PLAN_MANIFEST]) {
        let _ = fs::remove_file(saved.path().join(name));
        if let Ok(bytes) = fs::read(rewritten.path().join(name)) {
            fs::write(saved.path().join(name), bytes).unwrap();
        }
    }
    let restored = SymbolicPlan::restore(saved.path(), tok).unwrap();
    assert_ne!(restored.identity(), identity);
    let err = SymbolicPlan::restore_expecting(saved.path(), tok, &identity).unwrap_err();
    assert!(matches!(err, PlanError::IdentityMismatch { .. }), "{err}");
}

/// The documented edit path: an edited copy of a plan's ABC becomes a new external-ABC request,
/// planned fresh from the edited text — a different request and a different plan, never the
/// saved plan.
#[test]
fn an_edited_abc_is_a_new_request() {
    let tok = synthetic();
    let saved = restore(&upstream_dir("full_generated")).unwrap();
    let edited_text = saved.abc().unwrap().replace("E2", "F2");
    assert_ne!(edited_text, saved.abc().unwrap());
    let request = saved.request().with_abc(edited_text.clone()).unwrap();
    assert_ne!(&request, saved.request());
    let PlanStep::Ready(plan) = SymbolicPlan::prepare(request, tok).unwrap() else {
        panic!("external ABC needs no sampling");
    };
    assert_eq!(plan.kind(), PlanKind::External);
    assert_eq!(plan.abc(), Some(edited_text.as_str()));
    assert_eq!(plan.abc_ids(), tok.encode(&edited_text).unwrap());
    assert_ne!(plan.identity(), saved.identity());
    assert_eq!(
        &plan.prefix()[plan.prefix().len() - 2..],
        &[ABC_END, MUSIC_START]
    );
}

/// Semantic conditioning re-derives the prefix and refuses a plan whose prefix disagrees with its
/// request and exact IDs (upstream `generate_semantic`), builds the negative exactly when the
/// guidance is not 1, and checks the budget.
#[test]
fn semantic_conditioning() {
    let tok = synthetic();
    let semantic = Sampling::semantic_default();
    let external = restore(&upstream_dir("melody_external")).unwrap();
    assert_eq!(external.request().guidance(), 1.5);
    let c = external.semantic_conditioning(tok, &semantic).unwrap();
    assert_eq!(c.positive, external.prefix());
    let negative =
        crate::protocol::negative_prefix(external.request(), tok, Some(external.abc_ids()))
            .unwrap();
    assert_eq!(c.negative, Some(negative));
    assert!(!c.legacy_off);

    let generated = restore(&upstream_dir("full_generated")).unwrap();
    assert_eq!(
        generated
            .semantic_conditioning(tok, &semantic)
            .unwrap()
            .negative,
        None
    );

    let off = restore(&upstream_dir("off")).unwrap();
    let c = off.semantic_conditioning(tok, &semantic).unwrap();
    assert!(c.legacy_off && c.negative.is_some() && c.cfg_scale == 1.01);

    let huge = semantic
        .with_overrides(&SamplingOverrides {
            max_tokens: Some(24_576),
            ..Default::default()
        })
        .unwrap();
    assert!(matches!(
        generated.semantic_conditioning(tok, &huge),
        Err(PlanError::Protocol(
            ProtocolError::BudgetExceedsContext { .. }
        ))
    ));

    let mut tampered = generated.clone();
    tampered.prefix.push(1);
    assert!(matches!(
        tampered.semantic_conditioning(tok, &semantic),
        Err(PlanError::Inconsistent(_))
    ));
}

/// The identity covers every model-facing part: changing any one of them changes it.
#[test]
fn identity_covers_the_model_facing_content() {
    let plan = restore(&upstream_dir("full_generated")).unwrap();
    let base = plan.identity();
    let mut ids = plan.clone();
    ids.abc_ids[0] += 1;
    let mut prefix = plan.clone();
    prefix.prefix[1] += 1;
    let mut text = plan.clone();
    text.abc = Some("X".into());
    let mut truncated = plan.clone();
    truncated.truncated = true;
    // Each request variant changes exactly one field, through validation.
    let with_request = |edit: &dyn Fn(&mut crate::protocol::SongRequestSpec)| {
        let mut spec = plan.request.spec();
        edit(&mut spec);
        let mut changed = plan.clone();
        changed.request = SongRequest::new(spec).unwrap();
        assert_ne!(changed.request, plan.request);
        changed
    };
    let requests = [
        ("request.seed", with_request(&|s| s.seed = 1)),
        (
            "request.cfg_scale",
            with_request(&|s| s.cfg_scale = Some(1.5)),
        ),
        (
            "request.abc",
            with_request(&|s| s.abc = Some("X:1\nK:C\nC|".into())),
        ),
        ("request.style", with_request(&|s| s.style.push('!'))),
        ("request.lyrics", with_request(&|s| s.lyrics.push('!'))),
        ("request.cot", with_request(&|s| s.cot = CotMode::Melody)),
        ("request.id", with_request(&|s| s.id = "other_song".into())),
    ];
    for (what, changed) in [
        ("abc_ids", ids),
        ("prefix", prefix),
        ("abc", text),
        ("truncated", truncated),
    ]
    .into_iter()
    .chain(requests)
    {
        assert_ne!(changed.identity(), base, "{what}");
    }
    let mut timing = plan.clone();
    timing.timing.insert("seconds".into(), Value::from(99.0));
    assert_eq!(
        timing.identity(),
        base,
        "timing is a measurement, not identity"
    );
}

#[test]
fn npy_reader_accepts_every_integer_layout_upstream_does() {
    fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}\n");
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.bytes());
        out.extend(data);
        out
    }
    let f = "prefix.npy";
    assert_eq!(
        read_npy_ints(f, &npy("<i4", "(2,)", &[1, 0, 0, 0, 255, 255, 255, 255])).unwrap(),
        [1, -1]
    );
    assert_eq!(
        read_npy_ints(f, &npy(">i2", "(1,)", &[1, 2])).unwrap(),
        [258]
    );
    assert_eq!(
        read_npy_ints(f, &npy("<u8", "(1,)", &[7, 0, 0, 0, 0, 0, 0, 0])).unwrap(),
        [7]
    );
    assert_eq!(
        read_npy_ints(f, &npy("|u1", "(3,)", &[1, 2, 255])).unwrap(),
        [1, 2, 255]
    );
    assert_eq!(
        read_npy_ints(f, &npy("<i8", "(0,)", &[])).unwrap(),
        Vec::<i128>::new()
    );
    for (descr, shape, data) in [
        ("<f4", "(1,)", &[0u8, 0, 0, 0][..]),
        ("<i4", "(1, 1)", &[0, 0, 0, 0]),
        ("<i4", "(2,)", &[0, 0, 0, 0]),
        ("<i4", "(1,)", &[0, 0, 0, 0, 9]),
        ("|i4", "(1,)", &[0, 0, 0, 0]),
    ] {
        assert!(
            read_npy_ints(f, &npy(descr, shape, data)).is_err(),
            "{descr} {shape}"
        );
    }
    assert!(read_npy_ints(f, b"not numpy").is_err());
}
