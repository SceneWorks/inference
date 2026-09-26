//! End-to-end engine tests on the synthetic models (sc-22994): the tiny YuE2 MoT with NAR heads,
//! the synthetic tokenizer ranks (padded so any sampled id decodes) and the fixture VAEs. Every
//! stage runs the production code; only the weights are synthetic.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use serde_json::Value;

use super::*;
use crate::engine::{EngineHooks, EngineObserver, EngineOptions, SongSettings, Stage, StageEvent};
use crate::protocol::{
    CotMode, GenerationConfig, Sampling, SamplingOverrides, SongRequest, SongRequestSpec,
};

/// Small token budgets and two midpoint steps keep a whole song to seconds on the CPU.
pub(crate) fn settings(decoder: VaeVariant) -> SongSettings {
    let o = |min: i64, max: i64| SamplingOverrides {
        min_tokens: Some(min),
        max_tokens: Some(max),
        ..Default::default()
    };
    let abc = Sampling::abc_default().with_overrides(&o(2, 6)).unwrap();
    let semantic = Sampling::semantic_default()
        .with_overrides(&o(12, 16))
        .unwrap();
    SongSettings {
        generation: GenerationConfig::new(abc, semantic, 2).unwrap(),
        decoder,
    }
}

fn request(cot: CotMode, seed: u64) -> SongRequest {
    let mut spec = SongRequestSpec::new("warm piano pop", "[Verse]\nla la la\n");
    spec.cot = cot;
    spec.seed = seed;
    SongRequest::new(spec).unwrap()
}

fn engine() -> Yue2Engine {
    Yue2Engine::synthetic(EngineOptions::default())
}

/// Records every stage event and, optionally, trips the shared cancel flag when a stage starts.
struct Recorder {
    events: Vec<(Stage, StageEvent)>,
    cancel_at: Option<Stage>,
    flag: Rc<Cell<bool>>,
}

impl Recorder {
    fn count(&self, stage: Stage, event: StageEvent) -> usize {
        self.events.iter().filter(|&&e| e == (stage, event)).count()
    }
}

impl EngineObserver for Recorder {
    fn on_stage(&mut self, stage: Stage, event: StageEvent) {
        self.events.push((stage, event));
        if event == StageEvent::Started && self.cancel_at == Some(stage) {
            self.flag.set(true);
        }
    }
}

/// Run `f` with hooks over a recorder that cancels when `cancel_at` starts.
fn with_hooks<T>(
    cancel_at: Option<Stage>,
    f: impl FnOnce(&mut EngineHooks<'_>) -> T,
) -> (T, Recorder) {
    let flag = Rc::new(Cell::new(false));
    let mut recorder = Recorder {
        events: Vec::new(),
        cancel_at,
        flag: Rc::clone(&flag),
    };
    let cancelled = move || flag.get();
    let out = {
        let mut hooks = EngineHooks {
            cancelled: &cancelled,
            observer: &mut recorder,
        };
        f(&mut hooks)
    };
    (out, recorder)
}

fn run(
    engine: &Yue2Engine,
    input: &SongInput,
    s: &SongSettings,
    output: &RunOutput,
) -> (Result<RunOutcome, RunError>, Recorder) {
    with_hooks(None, |h| engine.generate_to(input, s, output, h))
}

/// Every file under `dir` with its bytes (to prove a directory was not touched).
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(
                    p.strip_prefix(dir).unwrap().to_path_buf(),
                    fs::read(&p).unwrap(),
                );
            }
        }
    }
    out
}

/// Flip the low bit of the byte `back` bytes before the end.
fn flip_byte(path: &Path, back: usize) {
    let mut bytes = fs::read(path).unwrap();
    let at = bytes.len() - back;
    bytes[at] ^= 0x01;
    fs::write(path, bytes).unwrap();
}

fn flip_last_byte(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(path, bytes).unwrap();
}

fn latent_sha(result: &Value) -> String {
    result["latent"]["sha256"].as_str().unwrap().to_string()
}

#[test]
fn a_full_request_publishes_every_artifact() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("song");
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Full, 7));
    let (outcome, rec) = run(&engine, &input, &s, &RunOutput::fresh(&dir));
    let outcome = outcome.unwrap();
    for stage in Stage::ALL {
        assert_eq!(rec.count(stage, StageEvent::Started), 1, "{stage:?}");
        assert_eq!(rec.count(stage, StageEvent::Finished), 1, "{stage:?}");
    }
    assert!(
        !partial_dir(&dir).exists(),
        "the working directory is renamed away"
    );
    let result = verify_run(&dir, Some(&engine.input_identity(&input, &s).unwrap())).unwrap();
    assert_eq!(result, outcome.result);
    assert_eq!(result["kind"], "song");
    assert_eq!(result["status"], "complete");
    assert_eq!(result["sample_rate"], 48_000);
    assert_eq!(result["channels"], 2);
    assert!(result["truncated"]["abc"].is_boolean());
    assert!(result["truncated"]["semantic"].is_boolean());
    assert_eq!(result["weights"]["engine"], "yue2");
    assert_eq!(result["decoder"]["decoder_release"], "standard");
    assert_eq!(
        result["license"]["intended_use"],
        "noncommercial_experimentation"
    );
    assert!(!result["license"]["attributions"]
        .as_array()
        .unwrap()
        .is_empty());
    for key in [
        "abc",
        "semantic",
        "nar_seconds",
        "vae_seconds",
        "e2e_seconds",
    ] {
        assert!(result["timing"].get(key).is_some(), "timing.{key}");
    }
    for stage in ["plan", "semantic", "synthesis", "decode"] {
        assert_eq!(result["stages"][stage]["reused"], false, "{stage}");
        assert_eq!(
            result["stages"][stage]["identity"].as_str().unwrap().len(),
            64
        );
    }
    let artifacts = result["artifacts"].as_object().unwrap();
    for name in [
        PLAN_JSON,
        ABC_TOKENS_NPY,
        PREFIX_NPY,
        PLAN_MANIFEST,
        SCORE_ABC,
        SEMANTIC_NPY,
        LATENT_FILE,
        IDENTITY_FILE,
        AUDIO_WAV,
        REQUEST_JSON,
        CONFIG_JSON,
        "stages/plan.json",
        "stages/semantic.json",
        "stages/synthesis.json",
    ] {
        assert!(artifacts.contains_key(name), "{name} is recorded");
    }
    // verify_run re-hashed every recorded file above; the audio file is the audio returned.
    let (samples, rate, channels) = read_wav_f32(&fs::read(dir.join(AUDIO_WAV)).unwrap()).unwrap();
    assert_eq!((rate, channels), (48_000, 2));
    assert_eq!(samples, outcome.samples);
    assert!(samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    assert!(samples.iter().any(|&v| v != 0.0), "non-silent");
    let frames = result["latent"]["shape"][0].as_u64().unwrap() as usize;
    assert_eq!(samples.len(), 2 * (1920 * frames - 64));
    // The exact plan restores with the recorded identity, and the latents with theirs.
    let plan = SymbolicPlan::restore(&dir, engine.tokenizer()).unwrap();
    assert_eq!(plan.identity().to_string(), result["plan_identity"]);
    assert!(plan.abc().is_some(), "cot = full plans a score");
    let latents = AcousticLatents::load(&dir).unwrap();
    assert_eq!(latents.identity().sha256, latent_sha(&result));
    let config = read_json(&dir.join(CONFIG_JSON)).unwrap();
    assert_eq!(config["model_dtype"], "float32");
    assert_eq!(config["generation"]["ode_steps"], 2);
    assert_eq!(config["decoder_release"], "standard");
}

#[test]
fn plan_only_then_restored_stages_equal_the_one_shot_run() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let req = request(CotMode::Melody, 11);

    let one_shot = with_hooks(None, |h| engine.generate(&req, &s, h))
        .0
        .unwrap();

    // Plan only, published transactionally.
    let plan_dir = tmp.path().join("plan");
    let (_, plan_id, _) = with_hooks(None, |h| {
        engine.plan_to(&req, &s.generation, &RunOutput::fresh(&plan_dir), h)
    })
    .0
    .unwrap();
    let recorded = verify_run(&plan_dir, None).unwrap();
    assert_eq!(recorded["kind"], "plan");
    assert_eq!(recorded["plan_identity"], plan_id.to_string());

    // Restore it, then run each stage on its own.
    let restored =
        SymbolicPlan::restore_expecting(&plan_dir, engine.tokenizer(), &plan_id).unwrap();
    // Exact plan, exact token ids (timings are measurements and differ).
    assert_eq!(restored.identity(), one_shot.semantic.plan.identity());
    assert_eq!(restored.abc_ids(), one_shot.semantic.plan.abc_ids());
    assert_eq!(restored.prefix(), one_shot.semantic.plan.prefix());
    let (staged, _) = with_hooks(None, |h| {
        let semantic = engine.generate_semantic(&restored, &s.generation, h)?;
        let synthesis = engine.synthesize(&semantic, &s.generation, h)?;
        let audio = engine.decode(&synthesis.latents, s.decoder, h)?;
        Ok::<_, gen_core::Error>((semantic, synthesis, audio))
    });
    let (semantic, synthesis, audio) = staged.unwrap();
    assert_eq!(semantic.codes, one_shot.semantic.codes);
    assert_eq!(synthesis.latents.identity(), one_shot.latents.identity());
    assert_eq!(audio.samples(), one_shot.audio.samples());

    // And as an artifact-backed run from the restored plan.
    let dir = tmp.path().join("from-plan");
    let (outcome, rec) = run(
        &engine,
        &SongInput::Plan(restored),
        &s,
        &RunOutput::fresh(&dir),
    );
    let outcome = outcome.unwrap();
    assert_eq!(
        latent_sha(&outcome.result),
        one_shot.latents.identity().sha256
    );
    assert_eq!(outcome.samples, one_shot.audio.samples());
    assert_eq!(rec.count(Stage::Semantic, StageEvent::Started), 1);
}

#[test]
fn a_decoder_switch_decodes_the_same_verified_latent_without_touching_the_source() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("standard");
    let input = SongInput::Request(request(CotMode::Off, 3));
    run(
        &engine,
        &input,
        &settings(VaeVariant::Standard),
        &RunOutput::fresh(&source),
    )
    .0
    .unwrap();
    let before = snapshot(&source);
    let out = tmp.path().join("legacy");
    let (outcome, rec) = with_hooks(None, |h| {
        engine.decode_cached(
            &source,
            VaeVariant::Legacy,
            Some(&RunOutput::fresh(&out)),
            h,
        )
    });
    let outcome = outcome.unwrap();
    assert_eq!(snapshot(&source), before, "the source run is never written");
    for stage in [Stage::Plan, Stage::Semantic, Stage::Synthesis] {
        assert_eq!(rec.count(stage, StageEvent::Started), 0, "{stage:?} rerun");
        assert_eq!(rec.count(stage, StageEvent::Reused), 1, "{stage:?} reused");
    }
    let source_result = verify_run(&source, None).unwrap();
    let result = verify_run(&out, None).unwrap();
    assert_eq!(result["kind"], "cached_decode");
    assert_eq!(
        result["latent"], source_result["latent"],
        "one latent identity"
    );
    assert_eq!(result["decoder"]["decoder_release"], "legacy");
    assert_ne!(result["decoder"], source_result["decoder"]);
    assert_eq!(result["source_identity"], source_result["identity"]);
    assert_eq!(
        fs::read(out.join(LATENT_FILE)).unwrap(),
        fs::read(source.join(LATENT_FILE)).unwrap()
    );
    let src_audio = read_wav_f32(&fs::read(source.join(AUDIO_WAV)).unwrap())
        .unwrap()
        .0;
    assert_eq!(src_audio.len(), outcome.samples.len());
    assert_ne!(src_audio, outcome.samples, "a different decoder ran");
    let config = read_json(&out.join(CONFIG_JSON)).unwrap();
    assert_eq!(config["decoder_release"], "legacy");
    assert_eq!(
        config["cached_decode"]["source_identity"],
        source_result["identity"]
    );
    let source_record = read_json(&out.join(SOURCE_GENERATION_JSON)).unwrap();
    assert_eq!(
        source_record["source_latent_sha256"],
        file_digest(&source.join(LATENT_FILE)).unwrap().0
    );

    // A cached decode never writes into its source.
    let (r, _) = with_hooks(None, |h| {
        engine.decode_cached(
            &source,
            VaeVariant::Legacy,
            Some(&RunOutput::resume(&source)),
            h,
        )
    });
    assert!(matches!(r, Err(RunError::Invalid(_))), "{r:?}");

    // A tampered source latent is refused before anything is decoded.
    let tampered = tmp.path().join("tampered");
    for (name, bytes) in snapshot(&source) {
        let path = tampered.join(&name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    flip_last_byte(&tampered.join(LATENT_FILE));
    let (err, rec) = with_hooks(None, |h| {
        engine.decode_cached(&tampered, VaeVariant::Legacy, None, h)
    });
    assert!(matches!(err, Err(RunError::Corrupt { .. })), "{err:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);

    // So is a source whose record no longer verifies anywhere — here only its audio changed, which
    // the decode itself never reads: the source must be a verified complete run.
    let unverified = tmp.path().join("unverified");
    for (name, bytes) in snapshot(&source) {
        let path = unverified.join(&name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    flip_byte(&unverified.join(AUDIO_WAV), 4);
    let (err, rec) = with_hooks(None, |h| {
        engine.decode_cached(&unverified, VaeVariant::Legacy, None, h)
    });
    assert!(matches!(err, Err(RunError::Corrupt { .. })), "{err:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
}

#[test]
fn resume_reuses_only_the_matching_completed_stages() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Full, 21));

    let reference = tmp.path().join("reference");
    let expected = run(&engine, &input, &s, &RunOutput::fresh(&reference))
        .0
        .unwrap();

    // Interrupt during synthesis: plan and semantic are checkpointed, nothing is published.
    let dir = tmp.path().join("song");
    let (cancelled, _) = with_hooks(Some(Stage::Synthesis), |h| {
        engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
    });
    assert!(matches!(
        cancelled,
        Err(RunError::Engine(gen_core::Error::Canceled))
    ));
    assert!(!dir.exists());
    assert!(partial_dir(&dir).join("stages/semantic.json").is_file());
    assert!(!partial_dir(&dir).join("stages/synthesis.json").exists());

    // A fresh run refuses the interrupted directory; a resume continues it.
    assert!(matches!(
        run(&engine, &input, &s, &RunOutput::fresh(&dir)).0,
        Err(RunError::Interrupted(_))
    ));
    let (resumed, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    let resumed = resumed.unwrap();
    assert_eq!(rec.count(Stage::Plan, StageEvent::Reused), 1);
    assert_eq!(rec.count(Stage::Plan, StageEvent::Started), 0);
    assert_eq!(rec.count(Stage::Semantic, StageEvent::Reused), 1);
    assert_eq!(rec.count(Stage::Semantic, StageEvent::Started), 0);
    assert_eq!(rec.count(Stage::Synthesis, StageEvent::Started), 1);
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 1);
    assert_eq!(resumed.result["stages"]["semantic"]["reused"], true);
    assert_eq!(resumed.result["stages"]["synthesis"]["reused"], false);
    assert_eq!(latent_sha(&resumed.result), latent_sha(&expected.result));
    assert_eq!(resumed.samples, expected.samples);
    assert_eq!(resumed.result["identity"], expected.result["identity"]);

    // Resuming the complete run recomputes nothing and returns the published audio.
    let (again, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    let again = again.unwrap();
    for stage in Stage::ALL {
        assert_eq!(rec.count(stage, StageEvent::Started), 0, "{stage:?}");
        assert_eq!(rec.count(stage, StageEvent::Reused), 1, "{stage:?}");
    }
    assert_eq!(again.samples, expected.samples);
    // A fresh run never overwrites it.
    assert!(matches!(
        run(&engine, &input, &s, &RunOutput::fresh(&dir)).0,
        Err(RunError::Exists(_))
    ));
}

#[test]
fn resume_rejects_mismatched_and_corrupt_work_without_overwriting_it() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Off, 5));

    // A complete run resumed with a different request.
    let done = tmp.path().join("done");
    run(&engine, &input, &s, &RunOutput::fresh(&done))
        .0
        .unwrap();
    let before = snapshot(&done);
    let other = SongInput::Request(request(CotMode::Off, 6));
    let err = run(&engine, &other, &s, &RunOutput::resume(&done)).0;
    assert!(
        matches!(err, Err(RunError::IdentityMismatch { what: "run", .. })),
        "{err:?}"
    );
    assert_eq!(snapshot(&done), before);
    // A complete run with a corrupted audio file is refused on resume and by verification.
    let mut wav = fs::read(done.join(AUDIO_WAV)).unwrap();
    let mid = wav.len() / 2;
    wav[mid] ^= 0x40;
    fs::write(done.join(AUDIO_WAV), &wav).unwrap();
    assert!(matches!(
        verify_run(&done, None),
        Err(RunError::Corrupt { .. })
    ));
    assert!(matches!(
        run(&engine, &input, &s, &RunOutput::resume(&done)).0,
        Err(RunError::Corrupt { .. })
    ));

    // An interrupted run (plan + semantic checkpointed).
    let make_partial = |name: &str| {
        let dir = tmp.path().join(name);
        let (r, _) = with_hooks(Some(Stage::Synthesis), |h| {
            engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
        });
        assert!(r.is_err());
        dir
    };

    // Different semantic sampling: the semantic checkpoint's identity does not match.
    let dir = make_partial("mismatch");
    let before = snapshot(&partial_dir(&dir));
    let changed = SongSettings {
        generation: GenerationConfig::new(
            *s.generation.abc(),
            s.generation
                .semantic()
                .with_overrides(&SamplingOverrides {
                    temperature: Some(0.5),
                    ..Default::default()
                })
                .unwrap(),
            2,
        )
        .unwrap(),
        ..s.clone()
    };
    let (err, rec) = run(&engine, &input, &changed, &RunOutput::resume(&dir));
    assert!(
        matches!(
            err,
            Err(RunError::IdentityMismatch {
                what: "semantic",
                ..
            })
        ),
        "{err:?}"
    );
    assert_eq!(rec.count(Stage::Semantic, StageEvent::Started), 0);
    assert_eq!(
        snapshot(&partial_dir(&dir)),
        before,
        "never recomputed over"
    );
    assert!(!dir.exists());

    // A corrupted checkpoint artifact: the low byte of the last int32 code, so the file still
    // parses as valid codes — only the recorded digest can tell.
    let dir = make_partial("corrupt");
    flip_byte(&partial_dir(&dir).join(SEMANTIC_NPY), 4);
    let (err, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    assert!(matches!(err, Err(RunError::Corrupt { .. })), "{err:?}");
    assert_eq!(rec.count(Stage::Semantic, StageEvent::Started), 0);
    assert!(!dir.exists());

    // A different request over an interrupted run: its plan checkpoint belongs to another request.
    let dir = make_partial("other-request");
    let other = SongInput::Request(request(CotMode::Off, 99));
    let (err, rec) = run(&engine, &other, &s, &RunOutput::resume(&dir));
    assert!(
        matches!(err, Err(RunError::IdentityMismatch { what: "plan", .. })),
        "{err:?}"
    );
    assert_eq!(rec.count(Stage::Plan, StageEvent::Started), 0);

    // A plan checkpoint rewritten consistently (plan manifest and record digests too) is still
    // refused: the restored plan's identity is not the recorded one.
    let dir = make_partial("rewritten");
    let work = partial_dir(&dir);
    let plan_json = work.join(PLAN_JSON);
    let mut plan: Value = read_json(&plan_json).unwrap();
    plan["timing"] = serde_json::json!({"edited": true});
    plan["truncated"] = Value::Bool(true);
    write_json(&plan_json, &plan).unwrap();
    let manifest_path = work.join(PLAN_MANIFEST);
    let mut manifest = read_json(&manifest_path).unwrap();
    manifest[PLAN_JSON] = Value::String(file_digest(&plan_json).unwrap().0);
    write_json(&manifest_path, &manifest).unwrap();
    let record_path = work.join("stages/plan.json");
    let mut record = read_json(&record_path).unwrap();
    for name in [PLAN_JSON, PLAN_MANIFEST] {
        let (sha, bytes) = file_digest(&work.join(name)).unwrap();
        record["artifacts"][name] = serde_json::json!({"sha256": sha, "bytes": bytes});
    }
    write_json(&record_path, &record).unwrap();
    let err = run(&engine, &input, &s, &RunOutput::resume(&dir)).0;
    assert!(
        matches!(err, Err(RunError::IdentityMismatch { what: "plan", .. })),
        "{err:?}"
    );
}

#[test]
fn a_tampered_synthesis_checkpoint_is_rejected() {
    // Latents rewritten together with their sidecar (so `AcousticLatents::load` accepts them) and
    // the checkpoint's digests are refused: the checkpoint recorded another latent identity.
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Off, 13));
    let dir = tmp.path().join("song");
    let (r, _) = with_hooks(Some(Stage::Decode), |h| {
        engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
    });
    assert!(r.is_err());
    let work = partial_dir(&dir);
    assert!(work.join("stages/synthesis.json").is_file());
    let latents = AcousticLatents::load(&work).unwrap();
    let mut values = latents.values().to_vec();
    values[0] += 0.5;
    let forged =
        AcousticLatents::new(values, latents.frames(), latents.identity().source.clone()).unwrap();
    fs::remove_file(work.join(LATENT_FILE)).unwrap();
    fs::remove_file(work.join(IDENTITY_FILE)).unwrap();
    forged.save(&work).unwrap();
    // …and the checkpoint's artifact digests updated to the forged bytes, so only the latent
    // identity the checkpoint recorded can tell.
    let record_path = work.join("stages/synthesis.json");
    let mut record = read_json(&record_path).unwrap();
    for name in [LATENT_FILE, IDENTITY_FILE] {
        let (sha, bytes) = file_digest(&work.join(name)).unwrap();
        record["artifacts"][name] = serde_json::json!({"sha256": sha, "bytes": bytes});
    }
    write_json(&record_path, &record).unwrap();
    let (err, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    assert!(matches!(err, Err(RunError::Corrupt { .. })), "{err:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
    assert!(!dir.exists());
}

#[test]
fn cancellation_at_any_stage_publishes_nothing_complete() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Full, 9));
    for stage in Stage::ALL {
        let dir = tmp.path().join(stage.name());
        let (r, rec) = with_hooks(Some(stage), |h| {
            engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
        });
        assert!(
            matches!(r, Err(RunError::Engine(gen_core::Error::Canceled))),
            "{stage:?}: {r:?}"
        );
        assert_eq!(rec.count(stage, StageEvent::Finished), 0, "{stage:?}");
        assert!(!dir.exists(), "{stage:?}: nothing at the run directory");
        assert!(verify_run(&dir, None).is_err());
        let work = partial_dir(&dir);
        assert!(!work.join(RESULT_JSON).exists(), "{stage:?}");
        assert!(
            verify_run(&work, None).is_err(),
            "a .partial is never complete"
        );
        assert!(
            !work.join(format!("stages/{}.json", stage.name())).exists(),
            "{stage:?}: the cancelled stage is not checkpointed"
        );
    }
    // Plan-only cancellation publishes no plan.
    let plan_dir = tmp.path().join("plan-only");
    let (r, _) = with_hooks(Some(Stage::Plan), |h| {
        engine.plan_to(
            input.request(),
            &s.generation,
            &RunOutput::fresh(&plan_dir),
            h,
        )
    });
    assert!(r.is_err());
    assert!(!plan_dir.exists());
    // A cancelled cached decode publishes nothing either.
    let source = tmp.path().join("source");
    run(&engine, &input, &s, &RunOutput::fresh(&source))
        .0
        .unwrap();
    let out = tmp.path().join("decoded");
    let (r, _) = with_hooks(Some(Stage::Decode), |h| {
        engine.decode_cached(
            &source,
            VaeVariant::Legacy,
            Some(&RunOutput::fresh(&out)),
            h,
        )
    });
    assert!(matches!(
        r,
        Err(RunError::Engine(gen_core::Error::Canceled))
    ));
    assert!(!out.exists());
}

#[test]
fn a_failing_stage_leaves_no_published_run() {
    // A semantic budget that cannot fit the context is refused by the semantic stage — after the
    // plan was checkpointed — and nothing is published.
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let mut s = settings(VaeVariant::Standard);
    s.generation = GenerationConfig::new(
        *s.generation.abc(),
        Sampling::semantic_default()
            .with_overrides(&SamplingOverrides {
                max_tokens: Some(crate::protocol::CONTEXT as i64),
                ..Default::default()
            })
            .unwrap(),
        2,
    )
    .unwrap();
    let dir = tmp.path().join("song");
    let input = SongInput::Request(request(CotMode::Off, 1));
    let (r, rec) = run(&engine, &input, &s, &RunOutput::fresh(&dir));
    assert!(matches!(r, Err(RunError::Engine(_))), "{r:?}");
    assert_eq!(rec.count(Stage::Plan, StageEvent::Finished), 1);
    assert!(!dir.exists());
    assert!(!partial_dir(&dir).join(RESULT_JSON).exists());
}

#[test]
fn wav_round_trips_and_rejects_other_formats() {
    let samples = vec![0.25f32, -0.5, 1.0, -1.0, 0.0, 0.125];
    let bytes = wav_f32_bytes(&samples, 48_000, 2);
    assert_eq!(read_wav_f32(&bytes).unwrap(), (samples.clone(), 48_000, 2));
    let mut pcm = bytes.clone();
    pcm[20] = 1; // PCM format tag
    assert!(read_wav_f32(&pcm).is_err());
    assert!(read_wav_f32(&bytes[..bytes.len() - 3]).is_err());
}

// ---------------------------------------------------------------------------------------------
// Review fix pass (sc-22994): one test per integrity check.
// ---------------------------------------------------------------------------------------------

/// Copy the run directory `from` (recursively) to `to`.
fn copy_run(from: &Path, to: &Path) {
    for (name, bytes) in snapshot(from) {
        let path = to.join(&name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

/// Rewrite `names`' digests in `dir/result.json` to the bytes now on disk (what someone replacing
/// artifacts consistently would do to get past `verify_run`).
fn rehash_result(dir: &Path, names: &[&str]) {
    let path = dir.join(RESULT_JSON);
    let mut result = read_json(&path).unwrap();
    for name in names {
        let (sha, bytes) = file_digest(&dir.join(name)).unwrap();
        result["artifacts"][*name] = serde_json::json!({"sha256": sha, "bytes": bytes});
    }
    write_json(&path, &result).unwrap();
}

fn published_source(engine: &Yue2Engine, root: &Path) -> PathBuf {
    let source = root.join("source");
    run(
        engine,
        &SongInput::Request(request(CotMode::Off, 31)),
        &settings(VaeVariant::Standard),
        &RunOutput::fresh(&source),
    )
    .0
    .unwrap();
    source
}

#[test]
fn a_cached_decode_never_writes_into_beside_or_above_its_source() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let source = published_source(&engine, tmp.path());
    let before = snapshot(&source);
    let link = tmp.path().join("link");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&source, &link).unwrap();
    let mut outputs = vec![
        RunOutput::fresh(source.join("nested")),
        RunOutput::fresh(source.join("a/b")),
        RunOutput::fresh(source.join("x/../nested")),
        RunOutput::fresh(tmp.path().join("elsewhere/../source/inner")),
        RunOutput::resume(&source),
        // An ancestor of the source.
        RunOutput::fresh(tmp.path()),
    ];
    if cfg!(unix) {
        outputs.push(RunOutput::fresh(link.join("inner")));
        outputs.push(RunOutput::fresh(&link));
    }
    for output in outputs {
        let (r, rec) = with_hooks(None, |h| {
            engine.decode_cached(&source, VaeVariant::Legacy, Some(&output), h)
        });
        assert!(matches!(r, Err(RunError::Invalid(_))), "{output:?}: {r:?}");
        assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
        assert_eq!(
            snapshot(&source),
            before,
            "{output:?}: the source is untouched"
        );
    }
    // A sibling is fine.
    let (r, _) = with_hooks(None, |h| {
        engine.decode_cached(
            &source,
            VaeVariant::Legacy,
            Some(&RunOutput::fresh(tmp.path().join("sibling"))),
            h,
        )
    });
    r.unwrap();
    assert_eq!(snapshot(&source), before);
}

#[test]
fn every_stage_identity_binds_every_input_it_depends_on() {
    let engine = engine();
    let s = settings(VaeVariant::Standard);
    let req = request(CotMode::Off, 1);
    let plan = with_hooks(None, |h| engine.plan(&req, &s.generation, h))
        .0
        .unwrap();
    let semantic = SemanticResult {
        plan: plan.clone(),
        codes: vec![3, 1, 4, 1, 5],
        truncated: false,
        timing: Default::default(),
    };
    let base = engine.identity_keys();
    assert_eq!(base.device, "cpu");
    assert_eq!(base.runtime, crate::engine::SOURCE_DIGEST);
    assert_eq!(base.runtime.len(), 64);
    // The engine's stage identities are exactly these pure functions.
    assert_eq!(
        engine
            .plan_stage_identity(&req, s.generation.abc())
            .unwrap(),
        base.plan(&req, s.generation.abc())
    );
    assert_eq!(
        engine
            .synthesis_stage_identity(&semantic, &s.generation)
            .unwrap(),
        base.synthesis(&semantic, &s.generation)
    );

    let decode = |k: &crate::engine::IdentityKeys| {
        k.cached_decode(
            "source",
            &serde_json::json!("latent"),
            &serde_json::json!("vae"),
            &serde_json::json!("tiles"),
            &serde_json::json!("mot"),
        )
    };
    let ids = |k: &crate::engine::IdentityKeys| {
        [
            k.plan(&req, s.generation.abc()),
            k.semantic(&plan, s.generation.semantic()),
            k.synthesis(&semantic, &s.generation),
            k.nar(&semantic, &s.generation),
            decode(k),
        ]
    };
    let b = ids(&base);
    // (key change, which of [plan, semantic, synthesis, nar, cached decode] must change)
    let cases: [(&str, crate::engine::IdentityKeys, [bool; 5]); 7] = [
        (
            "tier",
            crate::engine::IdentityKeys {
                tier: "q8",
                ..base.clone()
            },
            [true, true, true, false, false],
        ),
        (
            "ar",
            crate::engine::IdentityKeys {
                ar: "fp8",
                ..base.clone()
            },
            [true, true, false, false, false],
        ),
        (
            "weights",
            crate::engine::IdentityKeys {
                weights_sha256: "other-weights".into(),
                ..base.clone()
            },
            [true, true, true, true, false],
        ),
        (
            "tokenizer",
            crate::engine::IdentityKeys {
                tokenizer: serde_json::json!({"component": "other"}),
                ..base.clone()
            },
            [true, true, false, false, false],
        ),
        (
            "dtype",
            crate::engine::IdentityKeys {
                dtype: candle_audio::candle_core::DType::BF16,
                ..base.clone()
            },
            [true, true, true, true, false],
        ),
        (
            "device",
            crate::engine::IdentityKeys {
                device: "metal",
                ..base.clone()
            },
            [true, true, true, false, true],
        ),
        (
            "runtime",
            crate::engine::IdentityKeys {
                runtime: "another-build",
                ..base.clone()
            },
            [true, true, true, false, true],
        ),
    ];
    for (name, keys, changes) in cases {
        let got = ids(&keys);
        for (i, stage) in ["plan", "semantic", "synthesis", "nar", "cached_decode"]
            .iter()
            .enumerate()
        {
            assert_eq!(got[i] != b[i], changes[i], "{name} → {stage}");
        }
    }

    // The stage inputs themselves.
    let other = |edit: fn(&mut SamplingOverrides)| {
        let mut o = SamplingOverrides::default();
        edit(&mut o);
        o
    };
    let abc2 = s
        .generation
        .abc()
        .with_overrides(&other(|o| o.top_k = Some(7)))
        .unwrap();
    assert_ne!(base.plan(&req, &abc2), b[0], "ABC sampling → plan");
    let sem2 = s
        .generation
        .semantic()
        .with_overrides(&other(|o| o.temperature = Some(0.3)))
        .unwrap();
    assert_ne!(
        base.semantic(&plan, &sem2),
        b[1],
        "semantic sampling → semantic"
    );
    let steps3 = GenerationConfig::new(*s.generation.abc(), *s.generation.semantic(), 3).unwrap();
    assert_ne!(
        base.synthesis(&semantic, &steps3),
        b[2],
        "steps → synthesis"
    );
    assert_ne!(base.nar(&semantic, &steps3), b[3], "steps → latent source");
    let req2 = request(CotMode::Off, 2);
    assert_ne!(base.plan(&req2, s.generation.abc()), b[0], "seed → plan");
    // Same prefix and codes, another seed: only the song's noise differs.
    let plan2 = with_hooks(None, |h| engine.plan(&req2, &s.generation, h))
        .0
        .unwrap();
    assert_eq!(plan2.prefix(), plan.prefix());
    let semantic2 = SemanticResult {
        plan: plan2,
        ..semantic.clone()
    };
    assert_ne!(
        base.nar(&semantic2, &s.generation),
        b[3],
        "seed → noise → latent source"
    );
    assert_ne!(
        base.synthesis(&semantic2, &s.generation),
        b[2],
        "seed → synthesis"
    );
}

#[test]
fn a_cached_decode_refuses_a_source_generated_by_another_model() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let source = published_source(&engine, tmp.path());
    let other = tmp.path().join("other-model");
    copy_run(&source, &other);
    let path = other.join(RESULT_JSON);
    let mut result = read_json(&path).unwrap();
    result["weights"]["mot"] = serde_json::json!({"component": "another_mot"});
    write_json(&path, &result).unwrap();
    verify_run(&other, None).expect("result.json is not itself a recorded artifact");
    let (r, rec) = with_hooks(None, |h| {
        engine.decode_cached(&other, VaeVariant::Standard, None, h)
    });
    assert!(matches!(r, Err(RunError::Invalid(_))), "{r:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
}

#[test]
fn an_unpublished_result_in_an_interrupted_run_is_refused_and_left_alone() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Off, 41));
    let dir = tmp.path().join("song");
    let (r, _) = with_hooks(Some(Stage::Synthesis), |h| {
        engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
    });
    assert!(r.is_err());
    let work = partial_dir(&dir);
    write_json(
        &work.join(RESULT_JSON),
        &serde_json::json!({"status": "complete", "schema": RUN_SCHEMA}),
    )
    .unwrap();
    let before = snapshot(&work);
    let (r, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    assert!(matches!(r, Err(RunError::Corrupt { .. })), "{r:?}");
    assert_eq!(
        rec.count(Stage::Plan, StageEvent::Reused),
        0,
        "nothing reused"
    );
    assert_eq!(
        snapshot(&work),
        before,
        "the interrupted run is left as it was"
    );
    assert!(!dir.exists());
}

#[test]
fn a_cached_decode_refuses_latents_the_source_did_not_record() {
    // Forged latents with a consistent sidecar and consistent result.json digests pass
    // `verify_run` and `AcousticLatents::load`; only the latent identity the source recorded tells.
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let source = published_source(&engine, tmp.path());
    let forged_run = tmp.path().join("forged");
    copy_run(&source, &forged_run);
    let latents = AcousticLatents::load(&forged_run).unwrap();
    let mut values = latents.values().to_vec();
    values[0] += 0.25;
    let forged =
        AcousticLatents::new(values, latents.frames(), latents.identity().source.clone()).unwrap();
    fs::remove_file(forged_run.join(LATENT_FILE)).unwrap();
    fs::remove_file(forged_run.join(IDENTITY_FILE)).unwrap();
    forged.save(&forged_run).unwrap();
    rehash_result(&forged_run, &[LATENT_FILE, IDENTITY_FILE]);
    verify_run(&forged_run, None).unwrap();
    AcousticLatents::load(&forged_run).unwrap();
    let (r, rec) = with_hooks(None, |h| {
        engine.decode_cached(&forged_run, VaeVariant::Standard, None, h)
    });
    assert!(matches!(r, Err(RunError::Corrupt { .. })), "{r:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
}

#[test]
fn checkpointed_latents_from_another_source_are_refused() {
    // The same values re-attributed to another source, with the checkpoint's digests and its
    // recorded latent identity both updated: only the "produced by this synthesis" check tells.
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Off, 43));
    let dir = tmp.path().join("song");
    let (r, _) = with_hooks(Some(Stage::Decode), |h| {
        engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
    });
    assert!(r.is_err());
    let work = partial_dir(&dir);
    let latents = AcousticLatents::load(&work).unwrap();
    let imported = AcousticLatents::new(
        latents.values().to_vec(),
        latents.frames(),
        LatentSource::Imported {
            file_sha256: "0".repeat(64),
        },
    )
    .unwrap();
    fs::remove_file(work.join(LATENT_FILE)).unwrap();
    fs::remove_file(work.join(IDENTITY_FILE)).unwrap();
    imported.save(&work).unwrap();
    let record_path = work.join("stages/synthesis.json");
    let mut record = read_json(&record_path).unwrap();
    for name in [LATENT_FILE, IDENTITY_FILE] {
        let (sha, bytes) = file_digest(&work.join(name)).unwrap();
        record["artifacts"][name] = serde_json::json!({"sha256": sha, "bytes": bytes});
    }
    record["data"]["latent"] = imported.identity().to_json();
    write_json(&record_path, &record).unwrap();
    let (r, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    assert!(matches!(r, Err(RunError::Corrupt { .. })), "{r:?}");
    assert_eq!(rec.count(Stage::Decode, StageEvent::Started), 0);
    assert!(!dir.exists());
}

#[test]
fn a_working_directory_is_claimed_by_one_run_at_a_time() {
    let engine = engine();
    let tmp = tempfile::tempdir().unwrap();
    let s = settings(VaeVariant::Standard);
    let input = SongInput::Request(request(CotMode::Off, 47));
    let dir = tmp.path().join("song");
    let (r, _) = with_hooks(Some(Stage::Synthesis), |h| {
        engine.generate_to(&input, &s, &RunOutput::fresh(&dir), h)
    });
    assert!(r.is_err());
    let work = partial_dir(&dir);
    assert!(
        !work.join(LOCK_FILE).exists(),
        "a failed run releases its claim"
    );

    // Two concurrent claims: the second is refused while the first is held.
    let first = Claim::take(&work).unwrap();
    assert!(matches!(Claim::take(&work), Err(RunError::Locked(_))));
    let before = snapshot(&work);
    let (r, rec) = run(&engine, &input, &s, &RunOutput::resume(&dir));
    assert!(matches!(r, Err(RunError::Locked(_))), "{r:?}");
    assert!(rec.events.is_empty(), "a refused run does nothing");
    assert_eq!(snapshot(&work), before);
    drop(first);
    assert!(!work.join(LOCK_FILE).exists());

    // Released: the resume proceeds, and the published run carries no lock.
    run(&engine, &input, &s, &RunOutput::resume(&dir))
        .0
        .unwrap();
    assert!(!dir.join(LOCK_FILE).exists());
    assert!(!work.exists());
    verify_run(&dir, None).unwrap();
}

#[test]
fn a_missing_decoder_is_refused_before_any_stage_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let standard = tmp.path().join("YuE2-Vae");
    fs::create_dir(&standard).unwrap();
    let dirs = crate::snapshot::SnapshotDirs::new().with(
        crate::inventory::ComponentId::VaeStandard
            .component()
            .repo
            .id,
        &standard,
    );
    let engine = engine().with_checked_decoders(dirs);
    let input = SongInput::Request(request(CotMode::Off, 53));
    let dir = tmp.path().join("song");
    let (r, rec) = run(
        &engine,
        &input,
        &settings(VaeVariant::Legacy),
        &RunOutput::fresh(&dir),
    );
    let err = r.unwrap_err();
    assert!(err.to_string().contains("offline cache miss"), "{err}");
    assert!(rec.events.is_empty(), "no stage started");
    assert!(!dir.exists() && !partial_dir(&dir).exists());
    let (r, rec) = with_hooks(None, |h| {
        engine.generate(input.request(), &settings(VaeVariant::Legacy), h)
    });
    assert!(r.is_err());
    assert!(rec.events.is_empty());
    engine
        .check_decoder_available(VaeVariant::Standard)
        .unwrap();
}

#[test]
fn the_returned_record_is_the_published_record() {
    // `1.2533222419999999` (a timing CI recorded) does not round-trip bit for bit through serde_json's
    // default float parser; the outcome must carry what a reader of result.json gets.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("run");
    let Opened::Work(work) = open_output(&RunOutput::fresh(&dir)).unwrap() else {
        panic!("a fresh directory opens for work")
    };
    fs::write(work.path(AUDIO_WAV), b"x").unwrap();
    let mut result = serde_json::Map::new();
    result.insert(
        "e2e_seconds".into(),
        serde_json::json!(1.253_322_241_999_999_9_f64),
    );
    let (published_dir, returned) = work.publish(result).unwrap();
    assert_eq!(published_dir, dir);
    assert_eq!(returned, read_json(&dir.join(RESULT_JSON)).unwrap());
}
