//! Device fixtures replace only weight-heavy conditioner/decoder sources. The actual Generator,
//! native packed DiT, learned predictor, scheduler, latent assembly and outcome path execute.
use crate::dit::{MiniMaxH3Dit, MiniMaxH3DitConfig};
use mlx_gen::gen_core::{GenerationOutput, GenerationRequest, Generator, LoadShape, VedaRequest};
use mlx_gen::{weights::Weights, Result};
use mlx_rs::{Array, Dtype};

#[derive(Clone, Default)]
pub(crate) struct Fixture {
    pub predictor: bool,
    pub fail_self_test: bool,
}
fn config() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 128,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
        ..Default::default()
    }
}
/// Header-only source scaffold for the actual Generator's selected-partition probe.
/// Values come from the SAME config object the in-memory DiT uses, never stock defaults or
/// a second geometry literal. Keep the TempDir guard alive across the whole generation test.
fn source_root(cfg: &MiniMaxH3DitConfig) -> tempfile::TempDir {
    let root = tempfile::Builder::new()
        .prefix("h3-veda-source-")
        .tempdir()
        .unwrap();
    let mut header = serde_json::Map::new();
    macro_rules! config_fields {
        ($($field:ident),+ $(,)?) => {
            $(header.insert(stringify!($field).to_owned(), serde_json::json!(cfg.$field));)+
        };
    }
    config_fields!(
        num_attention_heads,
        attention_head_dim,
        hidden_size,
        num_layers,
        num_refiner_layers,
        ffn_dim,
        in_channels,
        audio_in_channels,
        patch_size,
        text_dim,
        freq_dim,
        time_embed_hidden_dim,
        time_embed_dim,
        rope_freq_dim,
        rope_theta,
        norm_eps,
        qk_norm_eps,
        final_norm_eps
    );
    let encoded = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    for partition in [
        crate::model::BASE_DIT_PARTITION,
        crate::model::REFERENCE_DIT_PARTITION,
    ] {
        let directory = root.path().join(partition);
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("config.json"), &encoded).unwrap();
    }
    root
}

#[test]
fn generator_fixture_headers_match_the_in_memory_config_and_refuse_missing_headers() {
    use crate::model::{MiniMaxH3Task, BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION};
    // The actual model test must consume this scaffold too; a correct isolated helper with
    // the old hardcoded generator root would recreate the very integration failure being fixed.
    let source = include_str!("veda_fixture.rs");
    let (_, native) = source
        .split_once(concat!(
            "fn native_generator_",
            "learned_sparse_off_and_failure_paths()"
        ))
        .unwrap();
    let native = native.split("\n#[test]").next().unwrap();
    let compact: String = native.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(compact.contains("letroot=source_root(&config());"));
    assert!(compact
        .contains("generator_with(partition.to_str().unwrap(),LoadShape::EagerMaterialization"));
    assert!(!native.contains("\"/fixture/transformer\""));
    let expected = config();
    let root = source_root(&expected);
    let other = source_root(&expected);
    assert_ne!(
        root.path(),
        other.path(),
        "concurrent fixtures must never share a source root"
    );
    let base = root.path().join(BASE_DIT_PARTITION);
    let generator = crate::model::tests::generator_with(
        base.to_str().unwrap(),
        LoadShape::EagerMaterialization,
    );
    for task in [
        MiniMaxH3Task::T2va,
        MiniMaxH3Task::Fl2va,
        MiniMaxH3Task::Ref2va,
    ] {
        assert_eq!(
            generator.task_dit_config(task).unwrap(),
            expected,
            "the selected header must describe the exact in-memory DiT"
        );
    }
    for partition in [BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION] {
        let names: Vec<_> = std::fs::read_dir(root.path().join(partition))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            names,
            vec![std::ffi::OsString::from("config.json")],
            "this CPU fixture must not contain or map tensor payloads"
        );
    }
    std::fs::remove_file(base.join("config.json")).unwrap();
    let error = generator
        .task_dit_config(MiniMaxH3Task::T2va)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(base.join("config.json").to_str().unwrap()),
        "{error}"
    );

    assert!(
        error.contains("no task config")
            && error.contains(BASE_DIT_PARTITION)
            && error.contains("reimport")
            && error.contains("select a partition"),
        "{error}"
    );
    assert_eq!(
        generator.task_dit_config(MiniMaxH3Task::Ref2va).unwrap(),
        expected,
        "a missing base header must not change the selected reference source"
    );
}

pub(crate) fn dit() -> Result<MiniMaxH3Dit> {
    let mut w = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dit_block.safetensors"
    ))?;
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    // Widen only attention fixture weights to the production D128 kernel shape. The committed
    // fixture's nonzero channels remain, with zero projection padding and unit norm gains.
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.contains(".attn."))
        .map(str::to_owned)
        .collect();
    for key in keys {
        let value = w.require(&key)?.clone();
        let widened = if key.ends_with("norm_q.weight") || key.ends_with("norm_k.weight") {
            mlx_rs::ops::concatenate_axis(&[&value, &mlx_rs::ops::ones::<f32>(&[104])?], 0)?
        } else if key.ends_with("to_q.weight")
            || key.ends_with("to_k.weight")
            || key.ends_with("to_v.weight")
        {
            mlx_rs::ops::concatenate_axis(
                &[
                    &value.reshape(&[4, 24, 64])?,
                    &mlx_rs::ops::zeros::<f32>(&[4, 104, 64])?,
                ],
                1,
            )?
            .reshape(&[512, 64])?
        } else if key.ends_with("to_out.0.weight") {
            mlx_rs::ops::concatenate_axis(
                &[
                    &value.reshape(&[64, 4, 24])?,
                    &mlx_rs::ops::zeros::<f32>(&[64, 4, 104])?,
                ],
                2,
            )?
            .reshape(&[64, 512])?
        } else {
            value
        };
        w.insert(key, widened);
    }
    MiniMaxH3Dit::from_weights(&mut w, &config(), Dtype::Float32)
}
pub(crate) fn context() -> Array {
    Array::from_slice(
        &(0..6 * 40)
            .map(|i| (i % 13) as f32 / 13.0 - 0.5)
            .collect::<Vec<_>>(),
        &[1, 6, 40],
    )
}
pub(crate) fn decode_video(
    req: &GenerationRequest,
    latents: &Array,
) -> Result<Vec<mlx_gen::media::Image>> {
    use sha2::{Digest, Sha256};
    let flat = latents.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    mlx_rs::transforms::eval([&flat])?;
    let mut hash = Sha256::new();
    for x in flat.as_slice::<f32>() {
        hash.update(x.to_le_bytes());
    }
    let digest = hash.finalize();
    Ok(vec![
        mlx_gen::media::Image {
            width: 1,
            height: 1,
            pixels: digest[..3].to_vec()
        };
        req.frames.unwrap_or(124) as usize
    ])
}
pub(crate) fn decode_audio(latents: &Array) -> Result<mlx_gen::media::AudioTrack> {
    let flat = latents.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    mlx_rs::transforms::eval([&flat])?;
    Ok(mlx_gen::media::AudioTrack {
        samples: (0..330_666)
            .map(|i| flat.as_slice::<f32>()[i % 32])
            .collect(),
        sample_rate: 32000,
        channels: 2,
        ..Default::default()
    })
}
/// Tiny genuine FP8/scaled/BF16-loaded predictor, with nonzero learned weights.
pub(crate) fn predictor() -> crate::veda::contract::Bundle {
    use serde_json::json;
    let heads = 4usize;
    let dim = 128usize;
    let layers = 2usize;
    let mut metadata = serde_json::Map::new();
    for (k, v) in [
        ("format", "miowtion-veda-predictor-v1"),
        ("num_layers", "2"),
        ("num_heads", "4"),
        ("head_dim", "128"),
        ("keep_ratio", "0.1"),
        ("dtype", "float8_e4m3fn"),
    ] {
        metadata.insert(k.into(), json!(v));
    }
    metadata.insert("plans".into(),json!(json!({"canonical":{"geometry":"16x9_t37","grid":[37,24,42],"shapes":["8x4x4","2x8x8"],"head_shape":[[0,0,1,1],[1,1,0,0]]}}).to_string()));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!(metadata));
    let mut bytes = Vec::new();
    for layer in 0..layers {
        for name in ["proj_q", "proj_k"] {
            let key = format!("layers.{layer}.{name}");
            let start = bytes.len();
            bytes.extend((0..heads * 3 * dim * dim).map(|i| {
                if (i + layer) % 5 == 0 {
                    0x9b
                } else {
                    0x1a
                }
            }));
            header.insert(key.clone(),json!({"dtype":"F8_E4M3","shape":[heads,3*dim,dim],"data_offsets":[start,bytes.len()]}));
            let start = bytes.len();
            for h in 0..heads {
                bytes.extend((0.03 + h as f32 * 0.005).to_le_bytes());
            }
            header.insert(
                format!("{key}.__scale"),
                json!({"dtype":"F32","shape":[heads],"data_offsets":[start,bytes.len()]}),
            );
        }
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(bytes);
    let meta = mlx_gen::gen_core::weightsmeta::CheckpointMeta::from_bytes(file).unwrap();
    crate::veda::contract::Bundle::from_meta(&meta, layers, heads, dim).unwrap()
}
#[test]
#[ignore = "requires the exclusive local Metal token"]
fn native_generator_learned_sparse_off_and_failure_paths() {
    let root = source_root(&config());
    let partition = root.path().join(crate::model::BASE_DIT_PARTITION);
    let mut generator = crate::model::tests::generator_with(
        partition.to_str().unwrap(),
        LoadShape::EagerMaterialization,
    );
    generator.fixture = Some(Fixture {
        predictor: true,
        ..Default::default()
    });
    let mut request = GenerationRequest {
        prompt: "bounded walking path".into(),
        width: 1344,
        height: 768,
        frames: Some(124),
        seed: Some(25769),
        steps: Some(1),
        ..Default::default()
    };
    let run = |g: &crate::model::MiniMaxH3, r: &GenerationRequest| {
        let out = g.generate(r, &mut |_| {}).unwrap();
        let GenerationOutput::Video { frames, fps, audio } = out else {
            panic!("video output")
        };
        assert_eq!(frames.len(), 124);
        assert_eq!(fps, 24);
        assert!(audio.is_some());
        frames
    };
    // A configured but missing optional component must not be touched by omitted/off settings.
    let missing_predictor = root.path().join("missing-veda-predictor.safetensors");
    assert!(!missing_predictor.exists());
    generator.veda_component = Some(missing_predictor);
    let dense = run(&generator, &request);
    let off_report = generator.veda_execution_report().unwrap();
    assert!(!off_report.requested);
    assert_eq!(off_report.predictor_load_attempts, 0);
    assert_eq!(off_report.workspace_bytes, 0);
    request.veda = Some(VedaRequest::default());
    assert_eq!(run(&generator, &request), dense);
    assert_eq!(
        generator
            .veda_execution_report()
            .unwrap()
            .predictor_load_attempts,
        0
    );
    request.veda = Some(VedaRequest {
        enabled: true,
        ..Default::default()
    });
    let sparse = run(&generator, &request);
    let report = generator.veda_execution_report().unwrap();
    assert_eq!(report.sparse_calls, 2);
    assert!(report.predictor_score_calls > 0);
    assert!(report.selected_token_pairs < report.dense_token_pairs);
    assert!(report.exact_plan);
    let outcome = report.outcome_json();
    assert_eq!(outcome["effective"], "sparse");
    assert_eq!(outcome["planMatch"], "exact");
    assert_eq!(outcome["executionGrid"], outcome["planGrid"]);
    assert!(outcome["totalAttentionFraction"].as_f64().unwrap() < 1.0);
    assert!(report.fallback_reason.is_none());
    assert_eq!(
        report.kernel.as_deref(),
        Some(crate::veda::kernel_id().unwrap())
    );
    assert_ne!(
        sparse, dense,
        "learned sparse path must affect output assembly"
    );
    generator.fixture.as_mut().unwrap().predictor = false;
    assert_eq!(run(&generator, &request), dense);
    let missing = generator.veda_execution_report().unwrap();
    assert_eq!(missing.sparse_calls, 0);
    assert_eq!(missing.dense_calls, 2);
    assert_eq!(missing.outcome_json()["effective"], "dense");
    assert_eq!(missing.outcome_json()["totalAttentionFraction"], 1.0);
    assert!(missing.outcome_json()["kernel"].is_null());
    assert!(missing
        .fallback_reason
        .unwrap()
        .contains("managed component"));
    generator.fixture.as_mut().unwrap().predictor = true;
    generator.fixture.as_mut().unwrap().fail_self_test = true;
    assert_eq!(run(&generator, &request), dense);
    let failed = generator.veda_execution_report().unwrap();
    assert_eq!(failed.sparse_calls, 0);
    assert_eq!(failed.dense_calls, 2);
    assert_eq!(failed.outcome_json()["effective"], "dense");
    assert_eq!(failed.outcome_json()["totalAttentionFraction"], 1.0);
    assert!(failed.outcome_json()["kernel"].is_null());
    assert!(failed
        .fallback_reason
        .unwrap()
        .contains("self-test forced failure"));
    // No sparse call ran under the chosen plan, so none is reported as executed.
    assert!(failed.plan.is_none() && failed.plan_grid.is_none());
    request.veda = None;
    assert_eq!(run(&generator, &request), dense);
    assert!(generator
        .veda_execution_report()
        .unwrap()
        .fallback_reason
        .is_none());
    let later = generator.veda_execution_report().unwrap().outcome_json();
    assert_eq!(later["requested"], false);
    assert!(later["kernel"].is_null());
    assert!(later["plan"].is_null());
    assert!(later["fallbackReason"].is_null());
    println!("S7 observed generator outcome: {}", outcome);
    println!("S1 native Generator: canonical1344x768/124f, 2 fixture DiT blocks, learned score calls={}, selected/dense={}/{}, dense/off identity, missing/self-test fallbacks verified",report.predictor_score_calls,report.selected_token_pairs,report.dense_token_pairs);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED veda_fixture::native_generator_learned_sparse_off_and_failure_paths");
}
#[test]
#[ignore = "requires the exclusive local Metal token"]
fn selected_kernel_actual_device_smoke() {
    crate::veda::device_self_test().unwrap();
}

#[test]
fn predictor_fixture_strict_loader_and_bf16_storage() {
    let p = predictor();
    assert_eq!((p.layers, p.heads, p.dim), (2, 4, 128));
    assert!(p.q.iter().flatten().any(|&v| v != 0));
    let proj = p.projection(0, &[0], true);
    assert!(proj.iter().all(|x| x.to_bits() & 0xffff == 0));
    assert_eq!(p.choose([37, 24, 42]).name, "16x9_t37");
    assert_eq!(p.choose([37, 42, 24]).name, "16x9_t37_T");
}

#[test]
#[ignore = "requires the exclusive local Metal token"]
fn native_veda_controls_preserve_carriers_and_capture_published_schedules() {
    use crate::denoise::{
        adaln_schedule, denoise_av, JointGeometry, JointSchedule, PackedLayout, TEXT_TAG,
    };
    use crate::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};
    use crate::dit::{AdaLnResidency, JointDit};
    use crate::reference::ReferenceKind;
    use mlx_gen::gen_core::{CancelFlag, VedaBudget};
    // Tiny native packed inputs cover each carrier mechanism; real models and canvases are S9.
    for (name, evals, shift, mode) in [
        ("base", 50, 12.0, 0),
        ("4step_768p", 4, 6.0, 1),
        ("8step", 8, 12.0, 2),
        ("4step_v01", 4, 12.0, 3),
        ("ref2v_4step", 4, 12.0, 4),
    ] {
        let geometry = JointGeometry::new(5, 2, 2).unwrap();
        let anchors = match mode {
            1 => vec![KeyframeAnchor::First],
            2 => vec![KeyframeAnchor::Last],
            3 => vec![KeyframeAnchor::First, KeyframeAnchor::Last],
            _ => vec![],
        };
        let layout = if mode < 4 {
            PackedLayout::build(geometry, [1, 2, 2], &[TEXT_TAG; 6], 2, &anchors).unwrap()
        } else {
            let refs = [
                ReferenceLatentGeometry {
                    kind: ReferenceKind::Image,
                    num_latent_frames: 1,
                    latent_height: 2,
                    latent_width: 2,
                    num_audio_latents: 0,
                },
                ReferenceLatentGeometry {
                    kind: ReferenceKind::Video,
                    num_latent_frames: 7,
                    latent_height: 2,
                    latent_width: 2,
                    num_audio_latents: 37,
                },
                ReferenceLatentGeometry {
                    kind: ReferenceKind::Audio,
                    num_latent_frames: 0,
                    latent_height: 0,
                    latent_width: 0,
                    num_audio_latents: 3,
                },
            ];
            PackedLayout::build_ref2va(geometry, [1, 2, 2], &[TEXT_TAG; 6], 2, &refs).unwrap()
        };
        let schedule = JointSchedule::with_shifts(evals + 1, shift, 3.0).unwrap();
        let adaln = adaln_schedule(&schedule).unwrap();
        let mut model = JointDit::new(
            dit().unwrap(),
            layout.clone(),
            &context(),
            adaln.clone(),
            AdaLnResidency::PrecomputeAndEvict,
        )
        .unwrap();
        let request = VedaRequest {
            enabled: true,
            generated: VedaBudget::KeyTiles(2),
            // Nonzero reference budget: reference key tiles go through learned selection
            // rather than the 0% (keep-all) dense shortcut.
            reference: VedaBudget::KeyTiles(1),
            full_attention_layers: vec![1],
            full_attention_steps: vec![0, evals - 1],
        };
        request.validate(2, evals).unwrap();
        let report = crate::veda::ReportSink::default();
        model.prepare_veda_fixture(&request, report.clone(), false);
        let video = Array::from_slice(
            &vec![0.02f32; layout.video_indices().len() * 96],
            &[1, layout.video_indices().len() as i32, 96],
        );
        let audio = Array::from_slice(
            &vec![0.03f32; layout.audio_indices().len() * 32],
            &[1, layout.audio_indices().len() as i32, 32],
        );
        let (outv, outa) = denoise_av(
            &mut model,
            &layout,
            &schedule,
            &adaln,
            &video,
            &audio,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(outv.shape(), video.shape(), "{name}");
        assert_eq!(outa.shape(), audio.shape(), "{name}");
        for (input, output, rows) in [
            (&video, &outv, layout.num_condition_video_rows()),
            (&audio, &outa, layout.num_condition_audio_rows()),
        ] {
            if rows > 0 {
                let want = crate::tensor::slice_axis(input, 1, 0, rows)
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap();
                let got = crate::tensor::slice_axis(output, 1, 0, rows)
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap();
                mlx_rs::transforms::eval([&want, &got]).unwrap();
                assert_eq!(want.as_slice::<f32>(), got.as_slice::<f32>(), "{name}");
            }
        }
        let calls = model.veda_calls();
        assert_eq!(calls.len(), evals * 2, "{name}");
        for (index, (layer, step, controls)) in calls.iter().enumerate() {
            assert_eq!((*layer, *step), (index % 2, index / 2), "{name}");
            assert_eq!(controls, &request, "{name}");
        }
        let r = report.borrow();
        // layer1 is always full; first/last evaluation are full in both layers.
        // A fake/default step0 would make every attention call dense and fail this discriminator.
        assert_eq!(r.sparse_calls, (evals - 2) as u64, "{name}");
        assert_eq!(r.dense_calls, (evals + 2) as u64, "{name}");
        assert!(
            r.fallback_reason.is_none(),
            "{name}: {:?}",
            r.fallback_reason
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED veda_fixture::native_veda_controls_preserve_carriers_and_capture_published_schedules");
}
