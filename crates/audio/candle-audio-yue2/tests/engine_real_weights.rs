//! **Real-weight end-to-end runs of the native YuE2 engine** (sc-22994).
//!
//! `#[ignore]`d in ordinary runs (CI has no weights); under `--ignored` a missing `YUE2_HF_HUB`
//! panics rather than silently passing. `YUE2_HF_HUB` is a hub directory holding the pinned
//! `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae` and `m-a-p/YuE2-Vae-legacy` revisions
//! (`models--m-a-p--YuE2-3B/snapshots/<revision>/…`, see `scripts/reference/yue2/README.md`).
//!
//! * [`registered_loader_generates_a_song_with_every_artifact`] — the worker's `LoadSpec` shape
//!   (weights = the YuE2-3B snapshot, `vae` / `vae_legacy` = the decoder snapshots) through the
//!   provider registry's `load`, one ~8 s song (`cot = full`, planned score) published with every
//!   artifact, resumed without regenerating, then re-decoded with the legacy decoder from its
//!   cached latents. No Python process is involved.
//! * [`a_saved_closure_serves_plan_only_and_restored_plan_runs`] — the closure saved to one
//!   directory and loaded back through the same `LoadSpec` gate, then plan-only → restore →
//!   a run from the exact restored plan through the public entry points.
//!
//! CPU, F32 (the BF16 checkpoint upcast; accelerator tiers are sc-22995). Run one test per process
//! under an external RSS guard; each loads both MoT paths in F32 (~16 GB) plus one decode tile:
//!
//! ```text
//! YUE2_HF_HUB=<hub dir> cargo test --release -p candle-audio-yue2 \
//!   --test engine_real_weights -- --ignored --nocapture --exact --test-threads 1 <name>
//! ```
//!
//! Measured 2026-09-26 (Apple M-series CPU, release) — see the story's PR for the run record.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_audio_yue2::closure::save_closure;
use candle_audio_yue2::gen_core::{
    AudioArtifacts, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress,
    SongDecoder, SongParams, SongPlanning, TokenSampling, WeightsSource,
};
use candle_audio_yue2::inventory::{self, VaeVariant};
use candle_audio_yue2::protocol::{GenerationConfig, SamplingOverrides, SongRequestSpec};
use candle_audio_yue2::provider::{load_generator, VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID};
use candle_audio_yue2::run::{read_wav_f32, verify_run, RunOutput, SongInput, AUDIO_WAV};
use candle_audio_yue2::{EngineHooks, SnapshotDirs, SongSettings, SymbolicPlan, PROVIDER_ID};

fn hub() -> PathBuf {
    PathBuf::from(std::env::var_os("YUE2_HF_HUB").unwrap_or_else(|| {
        panic!(
            "real-weight test run without YUE2_HF_HUB (a hub directory holding the pinned repos)"
        )
    }))
}

fn snapshot(repo: &inventory::UpstreamRepo) -> PathBuf {
    let dir = hub()
        .join(format!("models--{}", repo.id.replace('/', "--")))
        .join("snapshots")
        .join(repo.revision);
    assert!(dir.is_dir(), "{} is not staged", dir.display());
    dir
}

/// The worker's `LoadSpec` shape.
fn worker_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(snapshot(&inventory::YUE2_3B_REPO)))
        .with_component(
            VAE_COMPONENT_ID,
            WeightsSource::Dir(snapshot(&inventory::YUE2_VAE_REPO)),
        )
        .with_component(
            VAE_LEGACY_COMPONENT_ID,
            WeightsSource::Dir(snapshot(&inventory::YUE2_VAE_LEGACY_REPO)),
        )
}

const STYLE: &str = "English, warm piano, acoustic pop, female vocal, gentle, 90 bpm";
const LYRICS: &str = "[Verse]\nMorning light across the floor\nOpen up the kitchen door\n";

fn budget(min: u32, max: u32) -> TokenSampling {
    TokenSampling {
        min_tokens: Some(min),
        max_tokens: Some(max),
        ..Default::default()
    }
}

fn audio_of(out: GenerationOutput) -> Vec<f32> {
    match out {
        GenerationOutput::Audio(track) => {
            assert_eq!((track.sample_rate, track.channels), (48_000, 2));
            track.samples
        }
        other => panic!("expected audio, got {other:?}"),
    }
}

/// Finite, inside full scale, not silent; returns (seconds, rms).
fn check_playable(samples: &[f32]) -> (f64, f64) {
    assert!(samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    let rms =
        (samples.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    assert!(rms > 1e-4, "the audio is silent (rms {rms})");
    (samples.len() as f64 / 2.0 / 48_000.0, rms)
}

fn list(dir: &Path) -> Vec<String> {
    let result = verify_run(dir, None).expect("a verified run");
    result["artifacts"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

#[test]
#[ignore = "needs the pinned YuE2 weights (YUE2_HF_HUB); CPU, ~16 GB+ RSS"]
fn registered_loader_generates_a_song_with_every_artifact() {
    let out = tempfile::tempdir().unwrap();
    let dir = out.path().join("song");
    let t = Instant::now();
    let registry = candle_audio_yue2::provider_registry().unwrap();
    let generator = registry.load(PROVIDER_ID, &worker_spec()).unwrap();
    eprintln!(
        "load (resolve + verify + load): {:.1} s",
        t.elapsed().as_secs_f64()
    );

    // ~8 s: 200 codec frames at 25 frames/s; a 48-token planned score.
    let req = GenerationRequest {
        prompt: STYLE.into(),
        seed: Some(831_001),
        audio: Some(AudioParams {
            lyrics: Some(LYRICS.into()),
            song: Some(SongParams {
                planning: Some(SongPlanning::Full),
                score_sampling: Some(budget(8, 48)),
                semantic_sampling: Some(budget(200, 200)),
                ..Default::default()
            }),
            artifacts: Some(AudioArtifacts {
                dir: dir.clone(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let t = Instant::now();
    let samples = audio_of(
        generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    if current == total || current % 16 == 0 {
                        eprintln!("  acoustic step {current}/{total}");
                    }
                }
            })
            .unwrap(),
    );
    let (seconds, rms) = check_playable(&samples);
    eprintln!(
        "generated {seconds:.2} s of 48 kHz stereo (rms {rms:.4}) in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let result = verify_run(&dir, None).unwrap();
    eprintln!("truncated: {}", result["truncated"]);
    eprintln!("timing: {}", result["timing"]);
    eprintln!("artifacts: {:?}", list(&dir));
    assert!((6.0..=10.0).contains(&seconds), "{seconds} s");
    assert_eq!(result["weights"]["model_dtype"], "float32");
    assert_eq!(
        result["weights"]["mot"]["revision"],
        inventory::YUE2_3B_REPO.revision
    );
    assert_eq!(result["decoder"]["decoder_release"], "standard");
    assert_eq!(
        result["license"]["intended_use"],
        "noncommercial_experimentation"
    );
    let wav = read_wav_f32(&std::fs::read(dir.join(AUDIO_WAV)).unwrap()).unwrap();
    assert_eq!(wav.0, samples);
    if let Ok(keep) = std::env::var("YUE2_KEEP_WAV") {
        std::fs::copy(dir.join(AUDIO_WAV), &keep).unwrap();
        eprintln!("wrote {keep}");
    }

    // Resume: verified and returned, nothing regenerated.
    let mut resume = req.clone();
    resume
        .audio
        .as_mut()
        .unwrap()
        .artifacts
        .as_mut()
        .unwrap()
        .resume = true;
    let t = Instant::now();
    let mut steps = 0;
    let again = audio_of(
        generator
            .generate(&resume, &mut |p| {
                if matches!(p, Progress::Step { .. }) {
                    steps += 1;
                }
            })
            .unwrap(),
    );
    assert_eq!(steps, 0);
    assert_eq!(again, samples);
    eprintln!(
        "resume of the complete run: {:.2} s",
        t.elapsed().as_secs_f64()
    );

    // The legacy decoder over the same cached latents, into a new run.
    let legacy_dir = out.path().join("legacy");
    let decode = GenerationRequest {
        audio: Some(AudioParams {
            song: Some(SongParams {
                cached_latents: Some(dir.clone()),
                decoder: Some(SongDecoder::Legacy),
                ..Default::default()
            }),
            artifacts: Some(AudioArtifacts {
                dir: legacy_dir.clone(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let t = Instant::now();
    let legacy = audio_of(generator.generate(&decode, &mut |_| {}).unwrap());
    check_playable(&legacy);
    let legacy_result = verify_run(&legacy_dir, None).unwrap();
    assert_eq!(legacy_result["latent"], result["latent"]);
    assert_eq!(legacy_result["decoder"]["decoder_release"], "legacy");
    eprintln!(
        "legacy re-decode of the cached latents: {:.1} s",
        t.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "needs the pinned YuE2 weights (YUE2_HF_HUB); CPU, ~16 GB+ RSS; copies the closure"]
fn a_saved_closure_serves_plan_only_and_restored_plan_runs() {
    let out = tempfile::tempdir().unwrap();
    let dirs = inventory::REPOS
        .iter()
        .filter(|r| r.id.starts_with("m-a-p/YuE2"))
        .fold(SnapshotDirs::new(), |d, r| d.with(r.id, snapshot(r)));
    let closure = out.path().join("closure");
    let t = Instant::now();
    let metadata = save_closure(
        &dirs,
        &[VaeVariant::Standard, VaeVariant::Legacy],
        &GenerationConfig::default(),
        &closure,
    )
    .unwrap();
    eprintln!(
        "saved closure in {:.1} s: {metadata}",
        t.elapsed().as_secs_f64()
    );
    for licence in [
        "YuE2-3B/LICENSE",
        "YuE2-3B/THIRD_PARTY_NOTICES.md",
        "YuE2-Vae/LICENSE",
    ] {
        assert!(
            closure.join(licence).is_file(),
            "{licence} travels with the copy"
        );
    }

    let t = Instant::now();
    let generator = load_generator(&LoadSpec::new(WeightsSource::Dir(closure.clone()))).unwrap();
    eprintln!(
        "load from the saved closure: {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let engine = generator.engine();
    let mut spec = SongRequestSpec::new(STYLE, LYRICS);
    spec.seed = 7;
    let request = candle_audio_yue2::SongRequest::new(spec).unwrap();
    let o = |min: i64, max: i64| SamplingOverrides {
        min_tokens: Some(min),
        max_tokens: Some(max),
        ..Default::default()
    };
    let config = engine.generation_config();
    let settings = SongSettings {
        generation: GenerationConfig::new(
            config.abc().with_overrides(&o(8, 32)).unwrap(),
            config.semantic().with_overrides(&o(50, 50)).unwrap(),
            8,
        )
        .unwrap(),
        decoder: VaeVariant::Standard,
    };
    let never = || false;
    let mut hooks = EngineHooks {
        cancelled: &never,
        observer: &mut (),
    };
    let plan_dir = out.path().join("plan");
    let t = Instant::now();
    let (_, plan_id, _) = engine
        .plan_to(
            &request,
            settings.generation.abc(),
            &RunOutput::fresh(&plan_dir),
            &mut hooks,
        )
        .unwrap();
    eprintln!("plan only: {:.1} s", t.elapsed().as_secs_f64());
    let restored =
        SymbolicPlan::restore_expecting(&plan_dir, engine.tokenizer(), &plan_id).unwrap();
    eprintln!("restored score: {:?}", restored.abc());
    let run_dir = out.path().join("from-plan");
    let t = Instant::now();
    let outcome = engine
        .generate_to(
            &SongInput::Plan(restored.clone()),
            &settings,
            &RunOutput::fresh(&run_dir),
            &mut hooks,
        )
        .unwrap();
    let (seconds, rms) = check_playable(&outcome.samples);
    eprintln!(
        "from the restored plan: {seconds:.2} s (rms {rms:.4}) in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    assert_eq!(outcome.result["plan_identity"], plan_id.to_string());
    let saved = SymbolicPlan::restore(&run_dir, engine.tokenizer()).unwrap();
    assert_eq!(saved.abc_ids(), restored.abc_ids(), "exact token ids kept");
}
