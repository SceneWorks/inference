//! Targeted YuE2 stage-precision proof on the production accelerator and pinned real weights.
//! The ignored test requires the offline `YUE2_HF_HUB`, the independently pinned upstream
//! `YUE2_VAE_REFERENCE_DIR` fixture, and an output path in `YUE2_PRECISION_RECEIPT`. It never
//! downloads weights. Run it alone, once per backend, with an external device/host-memory sampler.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_yue2::decode::{decode_latents, DecodeMode, DecodeOptions};
use candle_audio_yue2::gen_core::{
    AudioArtifacts, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, Progress,
    SongDecoder, SongParams, SongPlanning, TokenSampling, WeightsSource, Yue2ComputePolicy,
};
use candle_audio_yue2::inventory::{self, ComponentId};
use candle_audio_yue2::latent::{AcousticLatents, LatentSource};
use candle_audio_yue2::provider::{VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID};
use candle_audio_yue2::run::{verify_run, CONFIG_JSON, SOURCE_GENERATION_JSON};
use candle_audio_yue2::snapshot::resolve_component;
use candle_audio_yue2::vae::{VaeParts, Yue2Vae};
use candle_audio_yue2::{SnapshotDirs, PROVIDER_ID};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const BF16_MIN_SNR_DB: f64 = 20.0;
const BF16_MAX_TILE_ERROR: f32 = 1.0 / 64.0;
const F32_MAX_ERROR: f32 = 2e-4;
const F32_MIN_SNR_DB: f64 = 90.0;

fn timed<T>(stage: &str, work: impl FnOnce() -> T) -> T {
    let now_ms = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    };
    let started = Instant::now();
    println!(
        "YUE2_PRECISION_STAGE {}",
        json!({"stage":stage,"event":"start","unixMs":now_ms()})
    );
    let out = work();
    println!(
        "YUE2_PRECISION_STAGE {}",
        json!({"stage":stage,"event":"end","unixMs":now_ms(),"elapsedMs":started.elapsed().as_millis()})
    );
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hub() -> SnapshotDirs {
    let dir = PathBuf::from(std::env::var_os("YUE2_HF_HUB").expect("YUE2_HF_HUB required"));
    inventory::REPOS
        .iter()
        .fold(SnapshotDirs::new(), |dirs, repo| {
            dirs.with(
                repo.id,
                dir.join(format!("models--{}", repo.id.replace('/', "--")))
                    .join("snapshots")
                    .join(repo.revision),
            )
        })
}

fn reference() -> (Value, std::collections::HashMap<String, Tensor>, String) {
    let meta: Value =
        serde_json::from_str(include_str!("fixtures/vae_real_reference.json")).unwrap();
    let dir = PathBuf::from(
        std::env::var_os("YUE2_VAE_REFERENCE_DIR").expect("YUE2_VAE_REFERENCE_DIR required"),
    );
    let bytes = std::fs::read(dir.join(meta["reference_file"].as_str().unwrap())).unwrap();
    let digest = hex(&Sha256::digest(&bytes));
    assert_eq!(
        digest, meta["reference_sha256"],
        "unverified reference fixture"
    );
    let tensors =
        candle_audio::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu).unwrap();
    (meta, tensors, digest)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn clamped_interleaved(raw: &Tensor) -> Vec<f32> {
    flat(
        &raw.clamp(-1f32, 1f32)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap(),
    )
}

fn seam_error(a: &[f32], b: &[f32], core: usize, frames: usize) -> f32 {
    let mut worst = 0.0f32;
    for seam in (core..frames).step_by(core).map(|f| f * 1920) {
        for sample in seam.saturating_sub(128)..(seam + 128).min(a.len() / 2) {
            for channel in 0..2 {
                let at = sample * 2 + channel;
                worst = worst.max((a[at] - b[at]).abs());
            }
        }
    }
    worst
}

fn metrics(got: &[f32], want: &[f32]) -> Value {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()));
    let (mut max, mut signal, mut noise) = (0.0f32, 0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        max = max.max((a - b).abs());
        signal += (b as f64).powi(2);
        noise += ((a - b) as f64).powi(2);
    }
    json!({"maxAbs": max, "snrDb": 10.0 * (signal / noise.max(1e-30)).log10()})
}

fn assert_reference(got: &[f32], want: &[f32], strict_f32: bool) -> Value {
    let m = metrics(got, want);
    if strict_f32 {
        assert!(m["maxAbs"].as_f64().unwrap() < F32_MAX_ERROR as f64, "{m}");
        assert!(m["snrDb"].as_f64().unwrap() > F32_MIN_SNR_DB, "{m}");
    } else {
        assert!(m["snrDb"].as_f64().unwrap() >= BF16_MIN_SNR_DB, "{m}");
    }
    m
}

fn latents(r: &std::collections::HashMap<String, Tensor>, name: &str) -> AcousticLatents {
    AcousticLatents::from_tensor(
        &r[name],
        LatentSource::Synthesis {
            stage_identity: format!("precision_reference:{name}"),
        },
    )
    .unwrap()
}

fn decode(vae: &Yue2Vae, latents: &AcousticLatents, options: DecodeOptions) -> Vec<f32> {
    decode_latents(vae, latents, &options, &|| false, &mut |_, _| {})
        .unwrap()
        .samples()
        .to_vec()
}

fn worker_spec(hub: &SnapshotDirs, policy: Yue2ComputePolicy) -> LoadSpec {
    let model = inventory::YUE2_3B_REPO;
    let standard = inventory::YUE2_VAE_REPO;
    let legacy = inventory::YUE2_VAE_LEGACY_REPO;
    LoadSpec::new(WeightsSource::Dir(hub.snapshot_dir(&model).unwrap()))
        .with_component(
            VAE_COMPONENT_ID,
            WeightsSource::Dir(hub.snapshot_dir(&standard).unwrap()),
        )
        .with_component(
            VAE_LEGACY_COMPONENT_ID,
            WeightsSource::Dir(hub.snapshot_dir(&legacy).unwrap()),
        )
        .with_yue2_compute_policy(policy)
}

fn small(min: u32, max: u32) -> TokenSampling {
    TokenSampling {
        min_tokens: Some(min),
        max_tokens: Some(max),
        ..Default::default()
    }
}

fn generate(hub: &SnapshotDirs, policy: Yue2ComputePolicy, root: &Path) -> Value {
    let registry = candle_audio_yue2::provider_registry().unwrap();
    let generator = timed(&format!("{policy:?}:registered_load"), || {
        registry
            .load(PROVIDER_ID, &worker_spec(hub, policy))
            .unwrap()
    });
    let song = root.join(format!("{policy:?}-song"));
    let request = GenerationRequest {
        prompt: "English, warm piano, acoustic pop, female vocal, gentle, 90 bpm".into(),
        seed: Some(831_001),
        audio: Some(AudioParams {
            lyrics: Some(
                "[Verse]\nMorning light across the floor\nOpen up the kitchen door\n".into(),
            ),
            song: Some(SongParams {
                planning: Some(SongPlanning::Full),
                score_sampling: Some(small(8, 48)),
                semantic_sampling: Some(small(200, 200)),
                ..Default::default()
            }),
            artifacts: Some(AudioArtifacts {
                dir: song.clone(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let output = timed(&format!("{policy:?}:registered_generation"), || {
        let mut last_total = 0;
        let mut phase = 0;
        generator.generate(&request, &mut |event| {
            let unix_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
            match event {
                Progress::Step { total, .. } if total != last_total => {
                    if last_total != 0 {
                        println!("YUE2_PRECISION_STAGE {}", json!({"stage":format!("{policy:?}:progress_phase_{phase}"),"event":"last_progress","unixMs":unix_ms,"total":last_total}));
                    }
                    phase += 1;
                    last_total = total;
                    println!("YUE2_PRECISION_STAGE {}", json!({"stage":format!("{policy:?}:progress_phase_{phase}"),"event":"first_progress","unixMs":unix_ms,"total":total}));
                }
                Progress::Decoding => {
                    println!("YUE2_PRECISION_STAGE {}", json!({"stage":format!("{policy:?}:progress_phase_{phase}"),"event":"last_progress","unixMs":unix_ms,"total":last_total}));
                    println!("YUE2_PRECISION_STAGE {}", json!({"stage":format!("{policy:?}:registered_vae_decode"),"event":"start","unixMs":unix_ms}));
                }
                _ => {}
            }
        }).unwrap()
    });
    let GenerationOutput::Audio(track) = output else {
        panic!("expected audio")
    };
    assert!(track
        .samples
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 1.0));
    let result = verify_run(&song, None).unwrap();
    let config: Value =
        serde_json::from_slice(&std::fs::read(song.join(CONFIG_JSON)).unwrap()).unwrap();
    let wanted = match policy {
        Yue2ComputePolicy::Legacy => None,
        Yue2ComputePolicy::Auto => Some("auto"),
        Yue2ComputePolicy::Bf16 => Some("bf16"),
        Yue2ComputePolicy::Fp32 => Some("fp32"),
    };
    assert_eq!(config.get("compute_policy").and_then(Value::as_str), wanted);
    let expected = if policy == Yue2ComputePolicy::Bf16 {
        "bfloat16"
    } else {
        "float32"
    };
    assert_eq!(config["vae_dtype"], expected);
    let legacy_dir = root.join(format!("{policy:?}-legacy"));
    let request = GenerationRequest {
        audio: Some(AudioParams {
            song: Some(SongParams {
                cached_latents: Some(song.clone()),
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
    let legacy = timed(&format!("{policy:?}:cached_legacy_decode"), || {
        generator.generate(&request, &mut |_| {}).unwrap()
    });
    let GenerationOutput::Audio(track) = legacy else {
        panic!("expected legacy audio")
    };
    assert!(track
        .samples
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 1.0));
    let old = verify_run(&legacy_dir, None).unwrap();
    assert_eq!(result["latent"], old["latent"]);
    json!({"runIdentity": result["identity"], "legacyCacheIdentity": old["identity"], "stages": result["stages"], "weights": result["weights"], "audioSha256": result["artifacts"]["audio.wav"]["sha256"], "legacyAudioSha256": old["artifacts"]["audio.wav"]["sha256"], "config": config})
}

/// A single bounded target per backend: three policies, each with real AR/NAR generation,
/// standard+legacy decode, and both VAE encoders on independently pinned inputs.
#[test]
#[ignore = "requires pinned YuE2 snapshots, upstream VAE fixture, and CUDA/Metal ownership"]
fn explicit_stage_precision_real_weights() {
    let device = candle_audio::default_device().unwrap();
    assert!(
        !device.is_cpu(),
        "BF16 VAE forward requires a CUDA or Metal backend"
    );
    let backend = if matches!(device, Device::Cuda(_)) {
        "cuda"
    } else {
        "metal"
    };
    let out = PathBuf::from(
        std::env::var_os("YUE2_PRECISION_RECEIPT").expect("YUE2_PRECISION_RECEIPT required"),
    );
    assert!(!out.exists(), "refuse to replace a precision receipt");
    let hub = hub();
    let (meta, r, reference_sha256) = reference();
    let work = PathBuf::from(
        std::env::var_os("YUE2_PRECISION_WORK_DIR")
            .expect("YUE2_PRECISION_WORK_DIR required for retained listening outputs"),
    );
    assert!(
        work.is_absolute(),
        "listening output must have an absolute path"
    );
    std::fs::create_dir(&work).expect("refuse an existing or unavailable listening directory");
    println!("YUE2_PRECISION_LISTENING_DIR {}", work.display());
    let mut cases = Vec::new();
    for policy in [
        Yue2ComputePolicy::Bf16,
        Yue2ComputePolicy::Auto,
        Yue2ComputePolicy::Fp32,
    ] {
        let strict_f32 = policy != Yue2ComputePolicy::Bf16;
        let dtype = if strict_f32 { DType::F32 } else { DType::BF16 };
        let generation = generate(&hub, policy, &work);
        let mut decoders = Vec::new();
        for (id, key) in [
            (ComponentId::VaeStandard, "standard"),
            (ComponentId::VaeLegacy, "legacy"),
        ] {
            let verified = resolve_component(id, &hub).unwrap();
            let vae = timed(&format!("{policy:?}:{key}:vae_load"), || {
                Yue2Vae::load_with_dtype(&verified, VaeParts::Full, &device, dtype).unwrap()
            });
            assert_eq!(vae.dtype(), dtype);
            assert_eq!(
                vae.identity().weights_sha256,
                meta["decoders"][key]["weights_sha256"]
            );
            let mut decode_cases = Vec::new();
            for (name, frames_key, core_key, latent_key, reference_key) in [
                (
                    "long",
                    "long_frames",
                    "long_core_frames",
                    "long_latent",
                    format!("{key}.long_full_raw"),
                ),
                (
                    "production",
                    "prod_frames",
                    "prod_core_frames",
                    "prod_latent",
                    format!("{key}.prod_pipeline"),
                ),
            ] {
                let source = latents(&r, latent_key);
                let frames = meta[frames_key].as_u64().unwrap() as usize;
                let core = meta[core_key].as_u64().unwrap() as usize;
                assert_eq!(source.frames(), frames);
                assert!(frames > 2 * core, "{name} must have multiple tile seams");
                let full = timed(&format!("{policy:?}:{key}:{name}:full_decode"), || {
                    decode(&vae, &source, DecodeOptions::reference_full())
                });
                let tiled = timed(&format!("{policy:?}:{key}:{name}:tiled_decode"), || {
                    decode(
                        &vae,
                        &source,
                        DecodeOptions {
                            mode: DecodeMode::Tiled { core_frames: core },
                            halo_frames: 16,
                        },
                    )
                });
                let reference = if name == "long" {
                    clamped_interleaved(&r[&reference_key])
                } else {
                    flat(&r[&reference_key])
                };
                let full_reference = assert_reference(&full, &reference, strict_f32);
                let tiled_reference = assert_reference(&tiled, &reference, strict_f32);
                let tile_error = metrics(&tiled, &full);
                let seam_max = seam_error(&tiled, &full, core, frames);
                let bound = if strict_f32 {
                    1e-4
                } else {
                    BF16_MAX_TILE_ERROR as f64
                };
                assert!(
                    tile_error["maxAbs"].as_f64().unwrap() <= bound,
                    "{name} full/tiled: {tile_error}"
                );
                assert!(seam_max as f64 <= bound, "{name} seam: {seam_max}");
                decode_cases.push(json!({"name":name,"frames":frames,"coreFrames":core,"latentSha256":source.identity().sha256,"fullReference":full_reference,"tiledReference":tiled_reference,"tileError":tile_error,"seamMaxAbs":seam_max}));
            }
            let posterior = timed(&format!("{policy:?}:{key}:encoder"), || {
                vae.encode(&r["clip"]).unwrap()
            });
            assert_eq!(posterior.mean.dtype(), dtype);
            assert_eq!(posterior.scale.dtype(), dtype);
            assert_eq!(posterior.stdev.dtype(), dtype);
            let sampled = posterior
                .sample(&posterior.mean.zeros_like().unwrap())
                .unwrap();
            assert_eq!(sampled.dtype(), dtype);
            assert!(flat(&sampled).iter().all(|v| v.is_finite()));
            let mean = assert_reference(
                &flat(&posterior.mean),
                &flat(&r[&format!("{key}.encode_mean")]),
                strict_f32,
            );
            let scale = assert_reference(
                &flat(&posterior.scale),
                &flat(&r[&format!("{key}.encode_scale")]),
                strict_f32,
            );
            decoders.push(json!({"variant": key, "weightsSha256": vae.identity().weights_sha256, "parameterDtype": format!("{dtype:?}"), "activationDtype": format!("{dtype:?}"), "decodeCases": decode_cases, "encoderMean": mean, "encoderScale": scale}));
        }
        cases.push(json!({"requestedPolicy": format!("{policy:?}"), "effectiveDtypes": {"ar": generation["config"]["model_dtype"], "nar": generation["config"]["model_dtype"], "vaeDecoder": generation["config"]["vae_dtype"], "vaeEncoder": if strict_f32 { "float32" } else { "bfloat16" }}, "generation": generation, "decoders": decoders}));
    }
    for left in 0..cases.len() {
        for right in left + 1..cases.len() {
            assert_ne!(
                cases[left]["generation"]["runIdentity"], cases[right]["generation"]["runIdentity"],
                "compute policy must bind the published run identity"
            );
            assert_ne!(
                cases[left]["generation"]["legacyCacheIdentity"],
                cases[right]["generation"]["legacyCacheIdentity"],
                "compute policy must bind cached legacy decode identity"
            );
        }
    }
    let legacy_source = generate(&hub, Yue2ComputePolicy::Legacy, &work);
    let source = work.join("Legacy-song");
    let target = work.join("Bf16-from-Legacy");
    let registry = candle_audio_yue2::provider_registry().unwrap();
    let bf16 = timed("Bf16:cross_policy_load", || {
        registry
            .load(PROVIDER_ID, &worker_spec(&hub, Yue2ComputePolicy::Bf16))
            .unwrap()
    });
    let request = GenerationRequest {
        audio: Some(AudioParams {
            song: Some(SongParams {
                cached_latents: Some(source),
                decoder: Some(SongDecoder::Legacy),
                ..Default::default()
            }),
            artifacts: Some(AudioArtifacts {
                dir: target.clone(),
                resume: false,
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    timed("Bf16:cross_policy_cached_decode", || {
        bf16.generate(&request, &mut |_| {}).unwrap()
    });
    let cross_result = verify_run(&target, None).unwrap();
    let cross_config: Value =
        serde_json::from_slice(&std::fs::read(target.join(CONFIG_JSON)).unwrap()).unwrap();
    let source_generation: Value =
        serde_json::from_slice(&std::fs::read(target.join(SOURCE_GENERATION_JSON)).unwrap())
            .unwrap();
    assert_eq!(cross_config["compute_policy"], "bf16");
    assert_eq!(cross_config["vae_dtype"], "bfloat16");
    assert!(source_generation["config"].get("compute_policy").is_none());
    assert_eq!(source_generation["identity"], legacy_source["runIdentity"]);
    assert_ne!(
        cross_result["identity"],
        legacy_source["legacyCacheIdentity"]
    );
    let receipt = json!({"schemaVersion": 1, "backend": backend, "referenceSha256": reference_sha256, "listeningDir": work, "cases": cases, "legacyToBf16CachedDecode": {"sourceIdentity": legacy_source["runIdentity"], "targetIdentity": cross_result["identity"], "targetConfig": cross_config, "sourceGeneration": source_generation}});
    std::fs::write(&out, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    println!("YUE2_PRECISION_RECEIPT {}", out.display());
}
