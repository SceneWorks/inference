//! The registered provider: the `LoadSpec` gate, request mapping and refusals, the offline
//! cache-miss path, and generations that go registry → `load` → `generate` /
//! `generate_with_report` → the pipeline → published artifacts (the synthetic engine stands in for
//! the verified snapshot load only, behind the same precision gate).

use std::path::Path;

use candle_audio::gen_core::{
    AudioArtifacts, AudioParams, CancelFlag, GenerationMemory, GenerationOutput, GenerationReport,
    GenerationRequest, GraphEvalCadence, LoadSpec, MemoryPhase, OffloadPolicy, Progress, Quant,
    SavedPlan, SongCover, SongCoverMode, SongDecoder, SongParams, SongPlanning, TokenSampling,
    WeightsSource,
};
use serde_json::Value;

use super::*;
use crate::run::{verify_run, AUDIO_WAV, CONFIG_JSON, REQUEST_JSON, RESULT_JSON};

/// A SheetSage2 transcription of a public-domain recording (the committed sc-23003 oracle, also
/// used by the cover module's tests): a full score with chord symbols, and its melody-only form.
const FULL: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_full/score.abc");
const MELODY: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_melody/score.abc");
const COVER_LYRICS: &str = "[Verse]\nO say can you see by the dawn's early light\nWhat so proudly \
                            we hailed at the twilight's last gleaming\n";
const COVER_STYLE: &str = "English, warm female vocal, gentle acoustic folk, 80 BPM";

/// A `LoadSpec` in the worker's shape: `weights` = the YuE2-3B snapshot, `vae` / `vae_legacy` =
/// the decoder snapshots staged. The directories are created (empty): only the existence check of
/// the early decoder probe reads them; the synthetic engine stands in for their contents.
fn spec_with(root: &Path, standard: bool, legacy: bool) -> LoadSpec {
    std::fs::create_dir_all(root.join("YuE2-3B")).unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.join("YuE2-3B")));
    for (on, id, dir) in [
        (standard, VAE_COMPONENT_ID, "YuE2-Vae"),
        (legacy, VAE_LEGACY_COMPONENT_ID, "YuE2-Vae-legacy"),
    ] {
        if on {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            spec = spec.with_component(id, WeightsSource::Dir(root.join(dir)));
        }
    }
    spec
}

fn spec(root: &Path, legacy: bool) -> LoadSpec {
    spec_with(root, true, legacy)
}

/// Load `yue2` through the explicit registry with the synthetic engine.
fn load_synthetic(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    SYNTHETIC_ENGINE.with(|s| s.set(true));
    let registry = crate::provider_registry().unwrap();
    let out = registry.load(PROVIDER_ID, spec);
    SYNTHETIC_ENGINE.with(|s| s.set(false));
    out
}

fn small(min: u32, max: u32) -> TokenSampling {
    TokenSampling {
        min_tokens: Some(min),
        max_tokens: Some(max),
        ..Default::default()
    }
}

fn song_request(dir: Option<&Path>) -> GenerationRequest {
    GenerationRequest {
        prompt: "warm piano pop".into(),
        seed: Some(17),
        steps: Some(2),
        audio: Some(AudioParams {
            lyrics: Some("[Verse]\nla la la\n".into()),
            song: Some(SongParams {
                planning: Some(SongPlanning::Full),
                score_sampling: Some(small(2, 6)),
                semantic_sampling: Some(small(12, 16)),
                ..Default::default()
            }),
            artifacts: dir.map(|d| AudioArtifacts {
                dir: d.to_path_buf(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn song_of(req: &mut GenerationRequest) -> &mut SongParams {
    req.audio.as_mut().unwrap().song.as_mut().unwrap()
}

fn audio_of(out: GenerationOutput) -> gen_core::AudioTrack {
    match out {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    }
}

/// `generate_with_report` through the trait object, with the progress it streamed.
fn reported(
    generator: &dyn Generator,
    req: &GenerationRequest,
) -> (GenerationReport, Vec<Progress>) {
    let mut progress = Vec::new();
    let report = generator
        .generate_with_report(req, &mut |p| progress.push(p))
        .unwrap();
    (report, progress)
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn codes(report: &GenerationReport) -> Vec<&str> {
    report.warnings.iter().map(|w| w.code.as_str()).collect()
}

#[test]
fn the_registration_is_a_distinct_conforming_yue2_provider() {
    let registry = crate::provider_registry().unwrap();
    let ids: Vec<&str> = registry.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(ids, ["yue2"]);
    assert!(
        !ids.iter().any(|id| id.starts_with("yue_")),
        "never YuE1's ids"
    );
    assert_eq!(
        registry.descriptor_conformance_errors(),
        Vec::<String>::new()
    );
    let d = descriptor();
    assert_eq!(d.family, "yue2");
    let c = &d.capabilities;
    assert_eq!(c.audio_sample_rates, [48_000]);
    assert!(c.supports_symbolic_song && c.supports_audio_artifacts);
    assert!(c.supports_song_plan_only && c.supports_song_cover);
    assert!(c.supports_sequential_offload, "LoadSpec.offload_policy");
    assert_eq!(
        c.request_memory_strategies,
        [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention
        ]
    );
    // Either decoder alone is a complete install (at least one is required at load).
    assert!(d.required_components.is_empty());
    assert_eq!(
        DECODER_COMPONENTS,
        [VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID]
    );
    for row in crate::COMPONENT_LICENSES {
        assert!(
            row.is_well_formed(gen_core::LICENSE_FAMILIES),
            "{}",
            row.component
        );
    }
    let rows: Vec<&str> = PROVIDER_COMPONENT_LICENSES
        .iter()
        .map(|r| r.component)
        .collect();
    assert_eq!(
        rows, PROVIDER_COMPONENTS[0].components,
        "exactly the loaded components have published rows"
    );
}

#[test]
fn a_registered_load_reaches_the_pipeline_and_publishes_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), true)).unwrap();
    let dir = tmp.path().join("runs/song");
    let req = song_request(Some(&dir));
    generator.validate(&req).unwrap();
    let mut progress = Vec::new();
    let track = audio_of(generator.generate(&req, &mut |p| progress.push(p)).unwrap());
    assert_eq!((track.sample_rate, track.channels), (48_000, 2));
    assert!(track.stems.is_empty(), "no V1 stem promise");
    assert!(track.samples.iter().all(|v| v.is_finite()));
    assert!(progress.contains(&Progress::Decoding));
    assert!(progress.contains(&Progress::Step {
        current: 2,
        total: 2
    }));
    let result = verify_run(&dir, None).unwrap();
    assert_eq!(result["weights"]["engine"], "yue2");
    assert!(result["timing"]["semantic"]["prefix_tokens"].is_u64());
    let config = read_json(&dir.join(CONFIG_JSON));
    assert_eq!(
        config["generation"]["ode_steps"], 2,
        "request steps → ODE steps"
    );
    assert_eq!(config["generation"]["semantic"]["max_tokens"], 16);
    let wav = crate::run::read_wav_f32(&std::fs::read(dir.join(AUDIO_WAV)).unwrap()).unwrap();
    assert_eq!(
        wav.0, track.samples,
        "the returned audio is the published audio"
    );

    // The same request resumed returns the published run without generating.
    let mut resumed = req.clone();
    resumed
        .audio
        .as_mut()
        .unwrap()
        .artifacts
        .as_mut()
        .unwrap()
        .resume = true;
    let mut steps = 0;
    let again = audio_of(
        generator
            .generate(&resumed, &mut |p| {
                if matches!(p, Progress::Step { .. }) {
                    steps += 1
                }
            })
            .unwrap(),
    );
    assert_eq!(again.samples, track.samples);
    assert_eq!(steps, 0, "nothing was synthesized again");

    // The legacy decoder over the run's cached latents, into a new run.
    let legacy_dir = tmp.path().join("runs/legacy");
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
    let (report, _) = reported(generator.as_ref(), &decode);
    let legacy = audio_of(report.output.unwrap());
    assert_eq!(legacy.samples.len(), track.samples.len());
    let legacy_result = verify_run(&legacy_dir, None).unwrap();
    assert_eq!(legacy_result["latent"], result["latent"]);
    assert_eq!(legacy_result["decoder"]["decoder_release"], "legacy");
    let record = report.artifacts.unwrap();
    assert_eq!(
        (record.dir, record.kind.as_str(), &record.identity),
        (
            legacy_dir,
            "cached_decode",
            &legacy_result["identity"].as_str().unwrap().to_string()
        )
    );

    // An exact saved plan (the run directory holds one) re-rendered in memory.
    let from_plan = GenerationRequest {
        steps: Some(2),
        audio: Some(AudioParams {
            song: Some(SongParams {
                plan: Some(SavedPlan {
                    dir: dir.clone(),
                    identity: result["plan_identity"].as_str().map(str::to_string),
                }),
                semantic_sampling: Some(small(12, 16)),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let replay = audio_of(generator.generate(&from_plan, &mut |_| {}).unwrap());
    assert_eq!(replay.samples, track.samples, "the same plan and seed");
    let mut wrong = from_plan.clone();
    song_of(&mut wrong).plan.as_mut().unwrap().identity = Some("0".repeat(64));
    assert!(generator.generate(&wrong, &mut |_| {}).is_err());
}

/// A restored plan fixes its request: a `prompt` or `audio.lyrics` that disagrees with it is an
/// edited plan — a new request — and is refused through the registered generator; the matching
/// ones pass.
///
/// Mutations that must fail: drop the prompt comparison (`req.prompt != r.style()` → `false`), or
/// drop the lyrics comparison.
#[test]
fn a_restored_plan_refuses_a_different_prompt_or_lyrics() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let plan_dir = tmp.path().join("plan");
    let mut plan_req = song_request(Some(&plan_dir));
    song_of(&mut plan_req).plan_only = true;
    let (report, _) = reported(generator.as_ref(), &plan_req);
    let plan_identity = verify_run(&plan_dir, None).unwrap()["plan_identity"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(report.output.is_none());
    let from_plan = |prompt: &str, lyrics: Option<&str>| GenerationRequest {
        prompt: prompt.into(),
        steps: Some(2),
        audio: Some(AudioParams {
            lyrics: lyrics.map(str::to_string),
            song: Some(SongParams {
                plan: Some(SavedPlan {
                    dir: plan_dir.clone(),
                    identity: Some(plan_identity.clone()),
                }),
                semantic_sampling: Some(small(12, 16)),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    for (what, req) in [
        ("prompt", from_plan("cold synth wave", None)),
        ("lyrics", from_plan("", Some("[Verse]\nsomething else\n"))),
    ] {
        let err = generator.generate(&req, &mut |_| {}).unwrap_err();
        assert!(
            err.to_string().contains("disagree with the restored plan"),
            "{what}: {err}"
        );
    }
    let same = from_plan("warm piano pop", Some("[Verse]\nla la la\n"));
    audio_of(generator.generate(&same, &mut |_| {}).unwrap());
}

#[test]
fn cancellation_through_the_generator_publishes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let dir = tmp.path().join("song");
    let mut req = song_request(Some(&dir));
    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    // Trip the flag at the first progress step (the first sampled ABC token).
    let r = generator.generate(&req, &mut |p| {
        if matches!(p, Progress::Step { .. }) {
            cancel.cancel();
        }
    });
    assert!(matches!(r, Err(gen_core::Error::Canceled)), "{r:?}");
    assert!(!dir.join(RESULT_JSON).exists());
    assert!(!dir.exists());
}

#[test]
fn the_offline_load_fails_explicitly_on_a_cache_miss() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = crate::provider_registry().unwrap();
    let missing = LoadSpec::new(WeightsSource::Dir(tmp.path().join("YuE2-3B"))).with_component(
        VAE_COMPONENT_ID,
        WeightsSource::Dir(tmp.path().join("YuE2-Vae")),
    );
    let err = registry
        .load(PROVIDER_ID, &missing)
        .err()
        .expect("nothing is staged");
    let text = err.to_string();
    assert!(text.contains("offline cache miss"), "{text}");
    assert!(text.contains("never downloads"), "{text}");
}

/// The `LoadSpec` gate, including the tier assertion on the registered path: the synthetic model
/// holds the released `bf16` weights, so `quantize = Q8 / Q4` is refused exactly as it is against a
/// staged `bf16` snapshot.
///
/// Mutation that must fail: build the loader's precision with `tier: None` (the asserted tier
/// never reaches the engine, and Q8 / Q4 load).
#[test]
fn the_load_gate_refuses_what_it_cannot_honour() {
    let tmp = tempfile::tempdir().unwrap();
    let ok = spec(tmp.path(), true);
    assert!(load_synthetic(&ok).is_ok());
    let no_vae = spec_with(tmp.path(), false, false);
    let err = load_synthetic(&no_vae).err().unwrap().to_string();
    assert!(err.contains("at least one decoder"), "{err}");
    let unknown = ok
        .clone()
        .with_component("stage2", WeightsSource::Dir(tmp.path().into()));
    assert!(matches!(
        load_synthetic(&unknown),
        Err(gen_core::Error::Unsupported(_))
    ));
    let mut quant = ok.clone();
    for (q, name) in [(Quant::Q8, "q8"), (Quant::Q4, "q4")] {
        quant.quantize = Some(q);
        let err = load_synthetic(&quant).err().unwrap();
        assert!(
            matches!(&err, gen_core::Error::Unsupported(m)
                if m.contains(&format!("the {name} tier was requested")) && m.contains("bf16")),
            "{name}: {err}"
        );
    }
    quant.quantize = Some(Quant::Nvfp4);
    assert!(matches!(
        load_synthetic(&quant),
        Err(gen_core::Error::Unsupported(_))
    ));
    let mut deferred = ok.clone();
    deferred.load_shape = gen_core::LoadShape::DeferredMaterialization;
    assert!(matches!(
        load_synthetic(&deferred),
        Err(gen_core::Error::Unsupported(m)) if m.contains("load_shape")
    ));
    let file = LoadSpec::new(WeightsSource::File(tmp.path().join("model.safetensors")))
        .with_component(VAE_COMPONENT_ID, WeightsSource::Dir(tmp.path().into()));
    assert!(load_synthetic(&file).is_err());
}

/// The advertised tiers are exactly the accepted ones: every `Quant` the descriptor lists passes
/// the `LoadSpec` gate and reaches the tier assertion (refused only because the synthetic model
/// holds `bf16` weights — a derived tier snapshot loads, see `engine_real_weights`), and every
/// other `Quant` is refused at the gate as unadvertised.
///
/// Mutations that must fail: advertise `&[]` (the pre-review descriptor), or drop the
/// advertised-set gate while advertising only `Q8`.
#[test]
fn the_advertised_quants_are_exactly_the_accepted_ones() {
    // Exhaustive: a new `Quant` fails to compile here until it is classified.
    let every = [Quant::Q4, Quant::Q8, Quant::Nvfp4];
    for q in every {
        match q {
            Quant::Q4 | Quant::Q8 | Quant::Nvfp4 => {}
        }
    }
    let advertised = descriptor().capabilities.supported_quants;
    assert_eq!(advertised, [Quant::Q4, Quant::Q8]);
    let tmp = tempfile::tempdir().unwrap();
    let mut spec = spec(tmp.path(), false);
    for q in every {
        spec.quantize = Some(q);
        let err = load_synthetic(&spec).err().unwrap().to_string();
        let accepted = err.contains("tier was requested but");
        let refused = err.contains("is not an advertised YuE2 tier");
        assert!(accepted != refused, "{q:?}: {err}");
        assert_eq!(accepted, advertised.contains(&q), "{q:?}: {err}");
    }
}

#[test]
fn unread_or_conflicting_request_fields_are_refused() {
    let base = GenerationConfig::default();
    let with_audio = |edit: fn(&mut AudioParams)| {
        let mut req = song_request(None);
        edit(req.audio.as_mut().unwrap());
        req
    };
    for (name, req) in [
        ("bpm", with_audio(|a| a.bpm = Some(120.0))),
        ("voice", with_audio(|a| a.voice = Some("x".into()))),
        (
            "target_duration",
            with_audio(|a| a.target_duration = Some(30.0)),
        ),
        (
            "repetition_penalty",
            with_audio(|a| a.repetition_penalty = Some(1.1)),
        ),
        ("segments", with_audio(|a| a.segments = Some(2))),
    ] {
        assert!(
            matches!(
                map_request(&req, &base),
                Err(gen_core::Error::Unsupported(_))
            ),
            "{name}"
        );
    }
    let no_lyrics = with_audio(|a| a.lyrics = None);
    assert!(map_request(&no_lyrics, &base).is_err());
    let off_with_score = with_audio(|a| {
        let song = a.song.as_mut().unwrap();
        song.planning = Some(SongPlanning::Off);
        song.score = Some(MELODY.into());
    });
    assert!(
        map_request(&off_with_score, &base).is_err(),
        "off takes no score"
    );
    let bad_sampling = with_audio(|a| {
        a.song.as_mut().unwrap().semantic_sampling = Some(TokenSampling {
            top_p: Some(1.5),
            ..Default::default()
        })
    });
    assert!(map_request(&bad_sampling, &base).is_err());
    let cached_with_seed = GenerationRequest {
        seed: Some(1),
        audio: Some(AudioParams {
            song: Some(SongParams {
                cached_latents: Some("run".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(map_request(&cached_with_seed, &base).is_err());
    let plan_with = |edit: fn(&mut SongParams)| {
        let mut song = SongParams {
            plan: Some(SavedPlan::default()),
            ..Default::default()
        };
        edit(&mut song);
        GenerationRequest {
            audio: Some(AudioParams {
                song: Some(song),
                artifacts: Some(AudioArtifacts::default()),
                ..Default::default()
            }),
            ..Default::default()
        }
    };
    for (name, req) in [
        (
            "score_sampling",
            plan_with(|s| s.score_sampling = Some(small(1, 2))),
        ),
        ("plan_only", plan_with(|s| s.plan_only = true)),
    ] {
        let err = map_request(&req, &base).unwrap_err().to_string();
        assert!(
            err.contains("cannot be combined with audio.song.plan"),
            "{name}: {err}"
        );
    }
    let cover_with_planning = with_audio(|a| {
        let song = a.song.as_mut().unwrap();
        song.cover = Some(SongCover {
            mode: SongCoverMode::Melody,
            score: FULL.into(),
            keep: None,
            translated_from: None,
        });
    });
    let err = map_request(&cover_with_planning, &base)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("audio.song.planning cannot be combined with audio.song.cover"),
        "{err}"
    );

    // The mapping itself: guidance keeps its decimal, steps are the ODE steps.
    let mut req = song_request(None);
    req.guidance = Some(1.01);
    let mapped = map_request(&req, &base).unwrap();
    let Job::Generate(r) = mapped.job else {
        panic!("a plain request generates")
    };
    assert_eq!(r.guidance(), 1.01);
    assert_eq!(r.seed(), 17);
    assert_eq!(r.style(), "warm piano pop");
    assert_eq!(mapped.settings.generation.ode_steps(), 2);
    assert_eq!(mapped.settings.decoder, VaeVariant::Standard);
    assert_eq!(mapped.settings.generation.semantic().max_tokens(), 16);
    assert_eq!(
        mapped.settings.generation.abc().temperature(),
        base.abc().temperature(),
        "unset fields keep the defaults"
    );
}

/// A supplied score must be in the native dialect: an out-of-dialect ABC is refused through the
/// registered generator with the parser's own reason; a dialect score passes.
///
/// Mutation that must fail: drop the `cover::abc::parse` call in `map_request`.
#[test]
fn a_supplied_score_outside_the_native_dialect_is_refused_visibly() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let mut req = song_request(None);
    song_of(&mut req).planning = Some(SongPlanning::Melody);
    // Upstream's protocol accepts this; the native dialect does not (no two-voice layout).
    song_of(&mut req).score =
        Some("X:1\nL:1/8\nM:4/4\nK:C\nV:Vocal\n[V:Vocal] C2 D2 E2 F2 |\n".into());
    let err = generator.validate(&req).unwrap_err();
    assert!(
        matches!(&err, gen_core::Error::Msg(m)
            if m.contains("not in the native two-voice ABC dialect")),
        "{err}"
    );
    song_of(&mut req).score = Some(MELODY.into());
    generator.validate(&req).unwrap();
}

/// `plan_only` publishes the exact plan as a run of kind `plan` ([`Yue2Engine::plan_to`]) and
/// renders nothing: the result is the record, reachable through `generate_with_report`; plain
/// `generate` refuses it before any compute; it needs `audio.artifacts`.
///
/// Mutations that must fail: map `plan_only` to `false` (a whole song renders), or let `generate`
/// run it.
#[test]
fn plan_only_publishes_a_plan_record_and_renders_no_audio() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let dir = tmp.path().join("plan");
    let mut req = song_request(Some(&dir));
    song_of(&mut req).plan_only = true;

    let mut progress = 0;
    let err = generator
        .generate(&req, &mut |_| progress += 1)
        .unwrap_err();
    assert!(
        matches!(&err, gen_core::Error::Unsupported(m) if m.contains("generate_with_report")),
        "{err}"
    );
    assert_eq!(progress, 0);
    assert!(!dir.exists() && !crate::run::partial_dir(&dir).exists());

    let (report, progress) = reported(generator.as_ref(), &req);
    assert!(report.output.is_none(), "a plan renders no audio");
    let result = verify_run(&dir, None).unwrap();
    let record = report.artifacts.unwrap();
    assert_eq!(record.kind, "plan");
    assert_eq!(record.dir, dir);
    assert_eq!(record.identity, result["identity"].as_str().unwrap());
    assert!(dir.join(crate::plan::PLAN_JSON).is_file());
    assert!(!dir.join(AUDIO_WAV).exists());
    assert!(
        progress
            .iter()
            .all(|p| matches!(p, Progress::Step { total: 6, .. })),
        "only the ABC phase ran: {progress:?}"
    );
    assert!(!progress.is_empty());

    // Without a record to publish into, it is refused by the shared floor.
    let mut orphan = req.clone();
    orphan.audio.as_mut().unwrap().artifacts = None;
    assert!(matches!(
        generator.validate(&orphan),
        Err(gen_core::Error::Msg(m)) if m.contains("requires audio.artifacts")
    ));
}

fn memory_render(
    generator: &dyn Generator,
    dir: &Path,
    memory: Option<GenerationMemory>,
) -> (Value, Value, Vec<f32>) {
    let mut req = song_request(Some(dir));
    req.memory = memory;
    let (report, _) = reported(generator, &req);
    let samples = audio_of(report.output.unwrap()).samples;
    (
        read_json(&dir.join(CONFIG_JSON)),
        verify_run(dir, None).unwrap(),
        samples,
    )
}

/// Every honoured memory control reaches the engine for the request that sets it — observed in the
/// published effective configuration — and changes no result (the same latents, the same audio).
///
/// Mutations that must fail (one at a time): `offload_ar: false` in `memory_options`; the query
/// tile left at the engine's; the decode left at the engine's.
#[test]
fn memory_controls_reach_the_engine_per_request_and_change_no_result() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let elements: u32 = 1 << 30;
    let synthetic = crate::nar::synthetic::model(1.0);
    let cfg = synthetic.lm().config();
    assert!(
        elements as usize >= cfg.num_attention_heads * cfg.max_position_embeddings,
        "the chosen chunk holds a row at the synthetic model's full context"
    );
    let (base_config, base_result, base_audio) =
        memory_render(generator.as_ref(), &tmp.path().join("base"), None);
    assert_eq!(base_config["offload_ar"], false);
    assert_eq!(base_config["query_tile"], "upstream");
    assert_eq!(
        base_config["vae_core_frames"],
        DecodeOptions::production().core_frames().unwrap()
    );
    let memory = GenerationMemory {
        stage_residency: true,
        chunk_attention: true,
        attention_chunk_size: Some(elements),
        tile_vae_decode: true,
        decode_tile_edge: Some(3),
        ..Default::default()
    };
    let (config, result, audio) = memory_render(
        generator.as_ref(),
        &tmp.path().join("bounded"),
        Some(memory),
    );
    assert_eq!(config["offload_ar"], true, "stage_residency → offload_ar");
    assert_eq!(
        config["query_tile"],
        serde_json::json!({"score_elements": elements}),
        "attention_chunk_size → the acoustic query tile"
    );
    assert_eq!(
        config["vae_core_frames"], 3,
        "decode_tile_edge → the decode core"
    );
    assert_eq!(config["vae_decode"], "halo_crop");
    // Result-invariant: the same latents bit for bit, and the same audio.
    assert_eq!(result["latent"], base_result["latent"]);
    assert_eq!(result["plan_identity"], base_result["plan_identity"]);
    let worst = audio
        .iter()
        .zip(&base_audio)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(audio.len(), base_audio.len());
    assert!(worst <= 1e-6, "tiling changed the audio by {worst}");
}

/// `LoadSpec.offload_policy = Sequential` is the engine's own AR offload; an explicit memory block
/// overrides it for its request (`stage_residency: false` keeps the weights resident).
///
/// Mutation that must fail: `offload_ar: false` in `engine_options`.
#[test]
fn the_offload_policy_is_the_engines_default_and_a_request_can_override_it() {
    let tmp = tempfile::tempdir().unwrap();
    let sequential = spec(tmp.path(), false).with_offload_policy(OffloadPolicy::Sequential);
    let generator = load_synthetic(&sequential).unwrap();
    let (config, ..) = memory_render(generator.as_ref(), &tmp.path().join("a"), None);
    assert_eq!(config["offload_ar"], true);
    let (config, ..) = memory_render(
        generator.as_ref(),
        &tmp.path().join("b"),
        Some(GenerationMemory::default()),
    );
    assert_eq!(config["offload_ar"], false);
}

/// Every memory control YuE2 does not honour — and every honoured one with an unusable value — is
/// refused before any compute, never ignored.
#[test]
fn unsupported_or_invalid_memory_controls_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let with = |memory: GenerationMemory| {
        let mut req = song_request(None);
        req.memory = Some(memory);
        generator.validate(&req)
    };
    let unsupported = [
        GenerationMemory {
            stream_transformer_blocks: true,
            ..Default::default()
        },
        GenerationMemory {
            transformer_window_size: Some(2),
            ..Default::default()
        },
        GenerationMemory {
            tile_vae_decode: true,
            decode_overlap: Some(8),
            ..Default::default()
        },
        GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
            ..Default::default()
        },
        {
            let mut m = GenerationMemory::default();
            m.authorize_calibration_fault(MemoryPhase::Decode);
            m
        },
    ];
    for memory in unsupported {
        assert!(
            matches!(with(memory), Err(gen_core::Error::Unsupported(_))),
            "{memory:?}"
        );
    }
    let invalid = [
        (
            GenerationMemory {
                attention_chunk_size: Some(1 << 30),
                ..Default::default()
            },
            "read only with memory.chunk_attention",
        ),
        (
            GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(1),
                ..Default::default()
            },
            "cannot hold one query row",
        ),
        (
            GenerationMemory {
                decode_tile_edge: Some(3),
                ..Default::default()
            },
            "read only with memory.tile_vae_decode",
        ),
        (
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(0),
                ..Default::default()
            },
            "1..=1024",
        ),
        (
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(1025),
                ..Default::default()
            },
            "1..=1024",
        ),
    ];
    for (memory, why) in invalid {
        let err = with(memory).unwrap_err();
        assert!(
            matches!(&err, gen_core::Error::Msg(m) if m.contains(why)),
            "{memory:?}: {err}"
        );
    }
    with(GenerationMemory::default()).unwrap();
}

fn cover_request(dir: Option<&Path>, mode: SongCoverMode, score: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: COVER_STYLE.into(),
        seed: Some(5),
        steps: Some(2),
        audio: Some(AudioParams {
            lyrics: Some(COVER_LYRICS.into()),
            song: Some(SongParams {
                cover: Some(SongCover {
                    mode,
                    score: score.into(),
                    keep: None,
                    translated_from: None,
                }),
                semantic_sampling: Some(small(12, 16)),
                ..Default::default()
            }),
            artifacts: dir.map(|d| AudioArtifacts {
                dir: d.to_path_buf(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A cover through the registered generator: the reviewed score is prepared by
/// [`crate::cover::prepare_cover`] (a melody cover plans from the chord-stripped score with
/// `cot = melody`), its warnings are reported, and its report is published inside the run under
/// the run's digests; a refusal publishes nothing.
///
/// Mutations that must fail: publish no `cover.json` attachment; map a melody cover to
/// `CoverMode::Full`; drop the cover warnings from the report.
#[test]
fn a_cover_plans_from_the_reviewed_score_and_publishes_its_report() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let dir = tmp.path().join("cover");
    let req = cover_request(Some(&dir), SongCoverMode::Melody, FULL);
    let expected = prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        FULL,
        COVER_STYLE,
        CoverLyrics::source(COVER_LYRICS),
    ))
    .unwrap()
    .report;
    assert!(
        !expected.warnings.is_empty(),
        "the fixture exercises warnings"
    );
    let (report, _) = reported(generator.as_ref(), &req);
    assert!(matches!(report.output, Some(GenerationOutput::Audio(_))));
    let want: Vec<&str> = expected.warnings.iter().map(|w| w.code).collect();
    assert_eq!(
        codes(&report)[..want.len()],
        want[..],
        "cover warnings come first"
    );

    let result = verify_run(&dir, None).unwrap();
    assert!(
        result["artifacts"][COVER_JSON]["sha256"].is_string(),
        "cover.json is under the run's digests"
    );
    let cover = read_json(&dir.join(COVER_JSON));
    assert_eq!(cover["schema"], COVER_SCHEMA);
    assert_eq!(cover["mode"], "melody");
    assert_eq!(cover["report"], expected.to_json());
    let chords = crate::cover::abc::parse(FULL).unwrap().chord_count();
    assert!(chords > 0);
    assert_eq!(cover["report"]["chords_removed"], chords);
    let planned = std::fs::read_to_string(dir.join(crate::plan::SCORE_ABC)).unwrap();
    assert_eq!(
        crate::cover::abc::parse(&planned).unwrap().chord_count(),
        0,
        "a melody cover plans from the chord-stripped score"
    );
    assert_eq!(read_json(&dir.join(REQUEST_JSON))["cot"], "melody");

    // The same cover resumes (its record is bound into the run identity) …
    let mut resumed = req.clone();
    resumed
        .audio
        .as_mut()
        .unwrap()
        .artifacts
        .as_mut()
        .unwrap()
        .resume = true;
    reported(generator.as_ref(), &resumed);
    // … while a cover whose record differs (the same request, now declared a translation) is not
    // this run: the attachment is bound into the run identity …
    let mut translated = resumed.clone();
    song_of(&mut translated)
        .cover
        .as_mut()
        .unwrap()
        .translated_from = Some(COVER_LYRICS.into());
    let err = generator
        .generate_with_report(&translated, &mut |_| {})
        .unwrap_err();
    assert!(err.to_string().contains("identity"), "{err}");
    // … and a changed cover.json no longer verifies.
    std::fs::write(dir.join(COVER_JSON), b"{}\n").unwrap();
    assert!(verify_run(&dir, None).is_err());

    // A full-score cover needs harmony: refused, nothing published.
    let bare = tmp.path().join("bare");
    let err = generator
        .generate_with_report(
            &cover_request(Some(&bare), SongCoverMode::Full, MELODY),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(err.to_string().contains("no chord symbols"), "{err}");
    assert!(!bare.exists() && !crate::run::partial_dir(&bare).exists());
    // An out-of-dialect score is refused with the parser's reason.
    let err = generator
        .validate(&cover_request(
            None,
            SongCoverMode::Melody,
            "X:1\nK:C\nC|\n",
        ))
        .unwrap_err();
    assert!(err.to_string().contains("cover score"), "{err}");
}

/// A truncated phase is visible in the result, with or without artifacts: `abc_truncated` /
/// `semantic_truncated`, and only for the phase that truncated (`cot = off` plans no ABC).
///
/// Mutations that must fail: drop the truncation warnings of the in-memory path, or of the
/// published path.
#[test]
fn a_truncated_phase_is_reported_in_the_result() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    // `min_tokens == max_tokens`: the end id is masked for the whole budget, so both truncate.
    let request = |planning: SongPlanning, dir: Option<&Path>| {
        let mut req = song_request(dir);
        let song = song_of(&mut req);
        song.planning = Some(planning);
        song.score_sampling = (planning != SongPlanning::Off).then(|| small(2, 2));
        song.semantic_sampling = Some(small(4, 4));
        req
    };
    let (report, _) = reported(generator.as_ref(), &request(SongPlanning::Off, None));
    assert_eq!(codes(&report), ["semantic_truncated"]);
    let (report, _) = reported(generator.as_ref(), &request(SongPlanning::Full, None));
    assert_eq!(codes(&report), TRUNCATION_CODES);
    let dir = tmp.path().join("run");
    let (report, _) = reported(generator.as_ref(), &request(SongPlanning::Full, Some(&dir)));
    assert_eq!(codes(&report), TRUNCATION_CODES);
    let result = verify_run(&dir, None).unwrap();
    assert_eq!(
        result["truncated"],
        serde_json::json!({"abc": true, "semantic": true})
    );
}

/// Progress covers the long AR phases: one `Step` per sampled token against the phase's
/// `max_tokens` — the ABC stage, then the semantic stage, each counting from 1 — before the
/// acoustic steps and `Decoding`.
///
/// Mutation that must fail: emit nothing from `ProgressBridge::on_token`.
#[test]
fn progress_counts_each_ar_token_against_its_phase_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let (_, progress) = reported(generator.as_ref(), &song_request(None));
    let steps: Vec<(u32, u32)> = progress
        .iter()
        .filter_map(|p| match p {
            Progress::Step { current, total } => Some((*current, *total)),
            _ => None,
        })
        .collect();
    let phase = |total: u32| -> Vec<u32> {
        steps
            .iter()
            .filter(|(_, t)| *t == total)
            .map(|(c, _)| *c)
            .collect()
    };
    let (abc, semantic) = (phase(6), phase(16));
    assert!((2..=6).contains(&abc.len()), "ABC tokens: {abc:?}");
    assert!(
        (12..=16).contains(&semantic.len()),
        "semantic tokens: {semantic:?}"
    );
    assert_eq!(abc, (1..=abc.len() as u32).collect::<Vec<_>>());
    assert_eq!(semantic, (1..=semantic.len() as u32).collect::<Vec<_>>());
    let first_semantic = steps.iter().position(|s| *s == (1, 16)).unwrap();
    assert!(steps[..first_semantic].iter().all(|(_, t)| *t == 6));
    assert!(
        steps[first_semantic + semantic.len()..]
            .iter()
            .all(|(_, t)| *t != 6 && *t != 16),
        "the acoustic steps follow"
    );
    assert_eq!(progress.last(), Some(&Progress::Decoding));
}

/// An install with only the legacy decoder loads; it renders with `decoder: Legacy`, and a
/// request for the unstaged standard decoder is refused before any compute.
///
/// Mutation that must fail: require the standard decoder at load again.
#[test]
fn a_legacy_only_install_loads_and_decodes_with_the_legacy_decoder() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec_with(tmp.path(), false, true)).unwrap();
    let mut req = song_request(None);
    let err = generator.validate(&req).unwrap_err();
    assert!(err.to_string().contains("offline cache miss"), "{err}");
    song_of(&mut req).decoder = Some(SongDecoder::Legacy);
    let track = audio_of(generator.generate(&req, &mut |_| {}).unwrap());
    assert!(!track.samples.is_empty());
}

#[test]
fn an_unprovisioned_decoder_is_refused_before_any_compute() {
    let tmp = tempfile::tempdir().unwrap();
    // Only the standard decoder is provisioned.
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let dir = tmp.path().join("song");
    let mut req = song_request(Some(&dir));
    song_of(&mut req).decoder = Some(SongDecoder::Legacy);
    let err = generator.validate(&req).unwrap_err();
    assert!(err.to_string().contains("offline cache miss"), "{err}");
    let mut progress = 0;
    let err = generator
        .generate(&req, &mut |_| progress += 1)
        .unwrap_err();
    assert!(err.to_string().contains("offline cache miss"), "{err}");
    assert_eq!(progress, 0, "nothing ran");
    assert!(!dir.exists() && !crate::run::partial_dir(&dir).exists());
    // The provisioned decoder passes the same check.
    let mut standard = req.clone();
    song_of(&mut standard).decoder = Some(SongDecoder::Standard);
    generator.validate(&standard).unwrap();
}
