#![cfg(all(test, target_os = "macos"))]

use super::*;
use mlx_gen::{AdapterKind, AdapterSpec};

struct DeviceGuard(mlx_rs::Device);

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&self.0);
    }
}

fn cpu_only() -> DeviceGuard {
    let guard = DeviceGuard(mlx_rs::Device::try_default().expect("default device"));
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    guard
}

fn write_tiny_prepared_pack(path: &Path, audio_frames: usize) {
    let audio_bytes = 8 * audio_frames * 16 * std::mem::size_of::<f32>();
    let total_bytes = 512 + audio_bytes;
    let header = serde_json::json!({
        "__metadata__": {
            "schemaVersion": "ltx-prepared-v1",
            "videoShape": "[1,128,1,1,1]",
            "audioShape": format!("[1,8,{audio_frames},16]"),
            "fps": "24"
        },
        "video_latents": {"dtype": "F32", "shape": [1,128,1,1,1], "data_offsets": [0,512]},
        "audio_latents": {"dtype": "F32", "shape": [1,8,audio_frames,16], "data_offsets": [512,total_bytes]}
    });
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + total_bytes, 0);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn weights_free_preflight_accepts_a_valid_t2v_prepared_pack() {
    let _cpu = cpu_only();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prepared.safetensors");
    write_tiny_prepared_pack(&path, 1);
    let mut request = super::validate_request_tests::request(1);
    request
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    request.items[0]
        .model_options
        .insert("ltxPreparedBundlePath".into(), serde_json::json!(path));
    validate_ltx25_training_request(&request).unwrap();
}

#[test]
fn weights_free_preflight_rejects_mismatched_av_duration() {
    let _cpu = cpu_only();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prepared.safetensors");
    write_tiny_prepared_pack(&path, 2);
    let mut request = super::validate_request_tests::request(1);
    request
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    request.items[0]
        .model_options
        .insert("ltxPreparedBundlePath".into(), serde_json::json!(path));
    let error = validate_ltx25_training_request(&request)
        .unwrap_err()
        .to_string();
    assert!(error.contains("audioShape"), "{error}");
    assert!(error.contains("expected 1"), "{error}");
}

#[test]
fn weights_free_preflight_rejects_a_missing_mask_tensor_without_fallback() {
    let _cpu = cpu_only();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prepared.safetensors");
    write_tiny_prepared_pack(&path, 1);
    let mut request = super::validate_request_tests::request(1);
    request.config.model_options.insert(
        "ltxWorkflow".into(),
        serde_json::json!("video_inpainting_lora"),
    );
    request.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{"type": "mask", "probability": 1.0, "tensorKey": "video_mask"}]
        }),
    );
    request.items[0]
        .model_options
        .insert("ltxPreparedBundlePath".into(), serde_json::json!(path));
    let error = validate_ltx25_training_request(&request)
        .unwrap_err()
        .to_string();
    assert!(error.contains("video_mask"), "{error}");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn train_changes_output_and_strict_save_reloads_on_fresh_av_model() {
    let _cpu = cpu_only();
    let mut cfg = crate::transformer::rung4_block_window_tests::tiny_cfg();
    cfg.ff_bias = false;
    let map = crate::transformer::rung4_block_window_tests::tiny_weight_map(&cfg);
    let weights = Weights::from_map(map);
    let mut train = AvDiT::from_weights(&weights, &cfg, Precision::quant_f32(8, 64)).unwrap();
    let baseline = AvDiT::from_weights(&weights, &cfg, Precision::quant_f32(8, 64)).unwrap();

    let suffixes = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]
    .map(str::to_string);
    let (targets, mut params) =
        build_targets_25(&mut train, cfg.num_layers, &suffixes, 2, 17).unwrap();
    assert_eq!(targets.len(), cfg.num_layers as usize * 28);

    let video_clean = Array::from_slice(&[0.1f32; 24], &[1, 6, 4]);
    let video_noise = Array::from_slice(&[0.7f32; 24], &[1, 6, 4]);
    let video_context = Array::from_slice(&[0.2f32; 72], &[1, 3, 24]);
    let video_positions = create_position_grid(1, 1, 2, 3);
    let video_plan = build_ltx25_token_plan(
        GENERATED,
        Ltx25TokenGeometry {
            frames: 1,
            height: 2,
            width: 3,
            spatial_scale: 32,
        },
        Ltx25ConditionInputs::default(),
    )
    .unwrap();
    let video = prepare_ltx25_modality(
        &video_clean,
        &video_noise,
        &video_context,
        &video_positions,
        None,
        None,
        0.4,
        &video_plan,
    )
    .unwrap();

    let audio_clean = Array::from_slice(&[0.15f32; 16], &[1, 4, 4]);
    let audio_noise = Array::from_slice(&[0.65f32; 16], &[1, 4, 4]);
    let audio_context = Array::from_slice(&[0.25f32; 24], &[1, 3, 8]);
    let audio_positions = create_audio_position_grid(1, 4);
    let audio_plan = build_ltx25_token_plan(
        GENERATED,
        Ltx25TokenGeometry {
            frames: 4,
            height: 1,
            width: 1,
            spatial_scale: 1,
        },
        Ltx25ConditionInputs::default(),
    )
    .unwrap();
    let audio = prepare_ltx25_modality(
        &audio_clean,
        &audio_noise,
        &audio_context,
        &audio_positions,
        None,
        None,
        0.4,
        &audio_plan,
    )
    .unwrap();
    let batch = Ltx25PreparedBatch {
        video: Some(video.clone()),
        audio: Some(audio.clone()),
    };
    let (base_video, _) = baseline
        .forward(
            &video.noisy,
            &video.timestep,
            &video.context,
            None,
            &video.positions,
            &audio.noisy,
            &audio.timestep,
            &audio.context,
            None,
            &audio.positions,
            None,
            None,
        )
        .unwrap();
    let (_, grads) = compute_ltx25_loss_grads(
        &mut train,
        &params,
        &targets,
        2.0,
        2.0,
        &batch,
        false,
        Dtype::Float32,
    )
    .unwrap();
    let mut optimizer = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
    optimizer.step(&mut params, &grads).unwrap();
    eval(params.values()).unwrap();
    install_train_lora(&mut train, &params, &targets, 2.0, 2.0, None).unwrap();
    let (adapted_video, _) = train
        .forward(
            &video.noisy,
            &video.timestep,
            &video.context,
            None,
            &video.positions,
            &audio.noisy,
            &audio.timestep,
            &audio.context,
            None,
            &audio.positions,
            None,
            None,
        )
        .unwrap();
    let effect = subtract(&adapted_video, &base_video)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(effect > 1e-6, "optimizer update must change AV output");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny_ltx25.safetensors");
    save_lora(&params, &targets, 2.0, 2, &path).unwrap();
    let saved = Weights::from_file(&path).unwrap();
    assert_eq!(saved.metadata("lora_rank"), Some("2"));
    assert_eq!(saved.metadata("lora_alpha"), Some("2"));

    let mut reloaded = AvDiT::from_weights(&weights, &cfg, Precision::quant_f32(8, 64)).unwrap();
    let report = crate::apply_ltx25_adapters(
        &mut reloaded,
        &[AdapterSpec::new(path, 1.0, AdapterKind::Lora)],
        1,
    )
    .unwrap();
    assert_eq!(report.applied, targets.len());
    assert!(report.skipped.is_empty());
    let (reloaded_video, _) = reloaded
        .forward(
            &video.noisy,
            &video.timestep,
            &video.context,
            None,
            &video.positions,
            &audio.noisy,
            &audio.timestep,
            &audio.context,
            None,
            &audio.positions,
            None,
            None,
        )
        .unwrap();
    let roundtrip = subtract(&adapted_video, &reloaded_video)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(
        roundtrip < 1e-6,
        "fresh strict reload must reproduce output"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::ltx25_tiny_production_lifecycle::train_changes_output_and_strict_save_reloads_on_fresh_av_model");
}
