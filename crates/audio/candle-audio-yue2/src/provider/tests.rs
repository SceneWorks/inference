//! The registered provider: the `LoadSpec` gate, request mapping and refusals, the offline
//! cache-miss path, and generations that go registry → `load` → `generate` → the pipeline →
//! published artifacts (the synthetic engine stands in for the verified snapshot load only).

use std::path::Path;

use candle_audio::gen_core::{
    AudioArtifacts, AudioParams, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    Progress, Quant, SavedPlan, SongDecoder, SongParams, SongPlanning, TokenSampling,
    WeightsSource,
};

use super::*;
use crate::run::{verify_run, AUDIO_WAV, RESULT_JSON};

/// A `LoadSpec` in the worker's shape: `weights` = the YuE2-3B snapshot, `vae` (+ optional
/// `vae_legacy`) = the decoder snapshots.
fn spec(root: &Path, legacy: bool) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.join("YuE2-3B")))
        .with_component(VAE_COMPONENT_ID, WeightsSource::Dir(root.join("YuE2-Vae")));
    if legacy {
        spec = spec.with_component(
            VAE_LEGACY_COMPONENT_ID,
            WeightsSource::Dir(root.join("YuE2-Vae-legacy")),
        );
    }
    spec
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

fn audio_of(out: GenerationOutput) -> gen_core::AudioTrack {
    match out {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    }
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
    assert_eq!(d.capabilities.audio_sample_rates, [48_000]);
    assert!(d.capabilities.supports_symbolic_song && d.capabilities.supports_audio_artifacts);
    assert_eq!(d.required_components, [VAE_COMPONENT_ID]);
    for row in crate::COMPONENT_LICENSES {
        assert!(
            row.is_well_formed(gen_core::LICENSE_FAMILIES),
            "{}",
            row.component
        );
    }
    for component in PROVIDER_COMPONENTS[0].components {
        assert!(
            crate::COMPONENT_LICENSES
                .iter()
                .any(|r| r.component == *component),
            "{component} has a licence row"
        );
    }
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
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
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
    let legacy = audio_of(generator.generate(&decode, &mut |_| {}).unwrap());
    assert_eq!(legacy.samples.len(), track.samples.len());
    let legacy_result = verify_run(&legacy_dir, None).unwrap();
    assert_eq!(legacy_result["latent"], result["latent"]);
    assert_eq!(legacy_result["decoder"]["decoder_release"], "legacy");

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
    wrong
        .audio
        .as_mut()
        .unwrap()
        .song
        .as_mut()
        .unwrap()
        .plan
        .as_mut()
        .unwrap()
        .identity = Some("0".repeat(64));
    assert!(generator.generate(&wrong, &mut |_| {}).is_err());
    let mut other_lyrics = from_plan.clone();
    other_lyrics.audio.as_mut().unwrap().lyrics = Some("[Verse]\nsomething else\n".into());
    assert!(
        generator.generate(&other_lyrics, &mut |_| {}).is_err(),
        "an edited plan is a new request"
    );
}

#[test]
fn cancellation_through_the_generator_publishes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let generator = load_synthetic(&spec(tmp.path(), false)).unwrap();
    let dir = tmp.path().join("song");
    let mut req = song_request(Some(&dir));
    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    // Trip the flag at the first acoustic step.
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
    let err = registry
        .load(PROVIDER_ID, &spec(tmp.path(), false))
        .err()
        .expect("nothing is staged");
    let text = err.to_string();
    assert!(text.contains("offline cache miss"), "{text}");
    assert!(text.contains("never downloads"), "{text}");
}

#[test]
fn the_load_gate_refuses_what_it_cannot_honour() {
    let tmp = tempfile::tempdir().unwrap();
    let ok = spec(tmp.path(), true);
    assert!(load_synthetic(&ok).is_ok());
    let no_vae = LoadSpec::new(WeightsSource::Dir(tmp.path().join("YuE2-3B")));
    assert!(
        load_synthetic(&no_vae).is_err(),
        "the standard decoder is required"
    );
    let unknown = ok
        .clone()
        .with_component("stage2", WeightsSource::Dir(tmp.path().into()));
    assert!(matches!(
        load_synthetic(&unknown),
        Err(gen_core::Error::Unsupported(_))
    ));
    let mut quant = ok.clone();
    quant.quantize = Some(Quant::Q8);
    assert!(load_synthetic(&quant).is_err());
    let file = LoadSpec::new(WeightsSource::File(tmp.path().join("model.safetensors")))
        .with_component(VAE_COMPONENT_ID, WeightsSource::Dir(tmp.path().into()));
    assert!(load_synthetic(&file).is_err());
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
        song.score = Some("X:1\nK:C\nC|".into());
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
    let plan_with_score_sampling = GenerationRequest {
        audio: Some(AudioParams {
            song: Some(SongParams {
                plan: Some(SavedPlan::default()),
                score_sampling: Some(small(1, 2)),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(map_request(&plan_with_score_sampling, &base).is_err());

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
