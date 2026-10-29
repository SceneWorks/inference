#![cfg(test)]

use super::validate_request;

/// sc-24161: the LTX-2.5 weights-free preflight keeps the shared floors' typed variant — an
/// edit dataset is a capability gap (`Unsupported`), never flattened to `Msg`.
#[test]
fn ltx25_preflight_keeps_the_edit_refusal_typed() {
    let mut req = request(1);
    req.items = vec![TrainingItem::edit_pair(
        PathBuf::from("target.png"),
        "make it blue".into(),
        vec![PathBuf::from("ref.png")],
    )];
    match super::validate_ltx25_training_request(&req) {
        Err(mlx_gen::Error::Unsupported(message)) => {
            assert!(message.contains("instruction-edit"), "{message}")
        }
        other => panic!("expected a typed Unsupported, got {other:?}"),
    }
}
use mlx_gen::{NetworkType, TrainingConfig, TrainingItem, TrainingRequest};
use std::path::PathBuf;

pub(super) fn request(items: usize) -> TrainingRequest {
    TrainingRequest {
        items: (0..items)
            .map(|i| TrainingItem {
                image_path: PathBuf::from(format!("img{i}.png")),
                caption: "a cat".into(),
                control_image_path: None,
                model_options: serde_json::Map::new(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            })
            .collect(),
        config: TrainingConfig::default(),
        output_dir: PathBuf::from("/tmp/ltx-trainer-test"),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec![],
        cancel: Default::default(),
    }
}

#[test]
fn accepts_valid_and_rejects_bad_requests() {
    assert!(validate_request(&request(1), "ltx_2_3 trainer").is_ok());
    assert!(validate_request(&request(0), "ltx_2_3 trainer").is_err()); // empty dataset

    let mut r = request(1);
    r.config.rank = 0;
    assert!(validate_request(&r, "ltx_2_3 trainer").is_err()); // zero rank

    let mut r = request(1);
    r.config.network_type = NetworkType::Lokr;
    assert!(validate_request(&r, "ltx_2_3 trainer").is_err()); // LoKr is LoRA-only here

    let mut r = request(1);
    r.config.optimizer = "sgd".into();
    assert!(validate_request(&r, "ltx_2_3 trainer").is_err()); // unsupported optimizer
}

fn masked_request() -> TrainingRequest {
    let mut r = request(1);
    r.items[0].subject_mask_path = Some(PathBuf::from("img0.mask.png"));
    r.config.subject_mask_loss = Some(mlx_gen::gen_core::SubjectMaskLoss {
        background_weight: 0.1,
        subject_weight: 1.0,
    });
    r
}

/// sc-24828: LTX-2.3 (decoded, centre-cropped images) declares subject-masked loss; LTX-2.5
/// (preprocessed latent bundles only) does not, and refuses it with a typed `Unsupported` that
/// names the reason — from the shared refusal `validate`/`train` call and from the weights-free
/// preflight, before any bundle is read.
#[test]
fn subject_mask_loss_is_ltx23_only_and_ltx25_refuses_it_by_name() {
    assert!(super::trainer_descriptor().techniques.subject_mask_loss);
    assert!(!super::trainer_descriptor_25().techniques.subject_mask_loss);
    let masked = masked_request();
    assert!(super::refuse_ltx25_subject_mask(super::MODEL_ID, &masked).is_ok());
    assert!(super::refuse_ltx25_subject_mask(super::MODEL_25_ID, &request(1)).is_ok());
    match super::refuse_ltx25_subject_mask(super::MODEL_25_ID, &masked) {
        Err(mlx_gen::gen_core::Error::Unsupported(message)) => {
            assert!(message.contains("preprocessed"), "{message}")
        }
        other => panic!("expected a typed Unsupported, got {other:?}"),
    }
    match super::validate_ltx25_training_request(&masked) {
        Err(mlx_gen::Error::Unsupported(message)) => {
            assert!(message.contains("subject-masked loss"), "{message}")
        }
        other => panic!("expected a typed Unsupported, got {other:?}"),
    }
    // The 2.3 descriptor passes the shared technique floor with the mask on.
    assert!(
        gen_core_train::validate_training_techniques(&super::trainer_descriptor(), &masked).is_ok()
    );
}

/// sc-24828: the LTX-2.3 weight is built on the unpatchified `(1, C, 1, le, le)` latent and run
/// through the trainer's own [`super::flatten_latent`], so it lines up token-for-token with the
/// patchified clean latent: cell `(y, x)` (value `y·10 + x`) lands on every channel of token
/// `y·le_w + x`.
#[test]
fn flattened_subject_weight_lines_up_with_the_patchified_latent() {
    let (c, h, w) = (3i32, 2usize, 3usize);
    let values: Vec<f32> = (0..h * w).map(|i| ((i / w) * 10 + i % w) as f32).collect();
    let shape = [1, c, 1, h as i32, w as i32];
    let weight = mlx_gen::train::loss::subject_mask_weight(&values, h, w, &shape).unwrap();
    let flat = super::flatten_latent(&weight).unwrap();
    assert_eq!(flat.shape(), &[1, (h * w) as i32, c]);
    // Row-major copy (the patchify is a strided view).
    let flat = flat.reshape(&[-1]).unwrap();
    let flat = flat.as_slice::<f32>();
    for token in 0..h * w {
        for ch in 0..c as usize {
            let want = ((token / w) * 10 + token % w) as f32;
            assert_eq!(flat[token * c as usize + ch], want, "token {token} ch {ch}");
        }
    }
}

#[test]
fn ltx25_preset_and_full_workflow_vocabulary_are_not_legacy_aliases() {
    assert_eq!(super::trainer_descriptor_25().id, super::MODEL_25_ID);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.width, 960);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.height, 544);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.frames, 89);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.fps, 24);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.steps, 30);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.stg_blocks, &[28]);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.video_cfg_scale, 3.0);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.audio_cfg_scale, 7.0);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.video_stg_scale, 1.0);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.audio_stg_scale, 1.0);
    assert_eq!(super::LTX25_VALIDATION_DEFAULTS.guidance_rescale, 0.7);
    assert_eq!(
        super::LTX25_VALIDATION_DEFAULTS.video_modality_guidance_scale,
        3.0
    );
    assert_eq!(
        super::LTX25_VALIDATION_DEFAULTS.audio_modality_guidance_scale,
        3.0
    );
    let executable_defaults = super::LTX25_VALIDATION_DEFAULTS;
    assert!(executable_defaults.generate_audio);
    let defaults = super::Ltx25ValidationConfig::from(executable_defaults);
    let rendered = super::build_ltx25_validation_render_plan(&defaults).unwrap();
    assert_eq!(rendered.sigmas.len(), 31);
    assert_eq!(rendered.video_guidance.cfg_scale, 3.0);
    assert_eq!(rendered.audio_guidance.cfg_scale, 7.0);
    assert_eq!(rendered.stg_blocks, vec![28]);
    assert_eq!(super::LTX25_WORKFLOWS.len(), 15);
    assert_eq!(
        super::Ltx25Workflow::parse("av2av_ic_lora").unwrap(),
        super::Ltx25Workflow::Av2avIcLora
    );
    assert_eq!(
        super::Ltx25Workflow::parse("a2a_ic_lora").unwrap(),
        super::Ltx25Workflow::A2aIcLora
    );
}

#[test]
fn validation_overrides_flow_into_the_executable_dev_render_plan() {
    let mut request = request(1);
    request
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    request.config.model_options.insert(
        "ltxValidation".into(),
        serde_json::json!({
            "width": 512,
            "height": 320,
            "frames": 25,
            "fps": 30,
            "steps": 12,
            "videoCfgScale": 4.0,
            "audioCfgScale": 6.0,
            "videoStgScale": 1.5,
            "audioStgScale": 0.75,
            "stgBlocks": [3, 7],
            "guidanceRescale": 0.5,
            "videoModalityGuidanceScale": 2.5,
            "audioModalityGuidanceScale": 4.0,
            "generateAudio": false
        }),
    );
    let parsed = super::Ltx25TrainingPlan::from_request(&request).unwrap();
    let rendered = super::build_ltx25_validation_render_plan(&parsed.validation).unwrap();
    assert_eq!(
        (
            rendered.width,
            rendered.height,
            rendered.frames,
            rendered.fps
        ),
        (512, 320, 25, 30)
    );
    assert_eq!(rendered.sigmas.len(), 13);
    assert_eq!(rendered.stg_blocks, vec![3, 7]);
    assert_eq!(
        rendered.video_guidance,
        super::Ltx25GuidancePlan {
            cfg_scale: 4.0,
            stg_scale: 1.5,
            rescale_scale: 0.5,
            modality_scale: 2.5,
        }
    );
    assert_eq!(
        rendered.audio_guidance,
        super::Ltx25GuidancePlan {
            cfg_scale: 6.0,
            stg_scale: 0.75,
            rescale_scale: 0.5,
            modality_scale: 4.0,
        }
    );
    assert!(!rendered.generate_audio);

    request
        .config
        .model_options
        .insert("ltxValidation".into(), serde_json::json!({"steps": 101}));
    assert!(super::Ltx25TrainingPlan::from_request(&request).is_err());
}

#[test]
fn all_upstream_workflows_resolve_to_executable_modality_plans() {
    let ids: Vec<_> = super::LTX25_WORKFLOWS
        .into_iter()
        .map(super::Ltx25Workflow::id)
        .collect();
    assert_eq!(ids.len(), 15);
    let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
    assert_eq!(unique.len(), 15);
    for workflow in super::LTX25_WORKFLOWS {
        assert_eq!(
            super::Ltx25Workflow::parse(workflow.id()).unwrap(),
            workflow
        );
        let plan = workflow.plan();
        assert!(
            plan.video.is_some_and(|value| value.is_generated)
                || plan.audio.is_some_and(|value| value.is_generated),
            "{} must generate at least one modality",
            workflow.id()
        );
        for modality in [plan.video, plan.audio].into_iter().flatten() {
            let has_mask = modality
                .conditions
                .iter()
                .any(|condition| condition.kind == super::Ltx25ConditionKind::Mask);
            let has_reference = modality
                .conditions
                .iter()
                .any(|condition| condition.kind == super::Ltx25ConditionKind::Reference);
            let mask = [true; 40];
            let token_plan = super::build_ltx25_token_plan(
                modality,
                super::Ltx25TokenGeometry {
                    frames: 10,
                    height: 2,
                    width: 2,
                    spatial_scale: 32,
                },
                super::Ltx25ConditionInputs {
                    conditioning_mask: has_mask.then_some(mask.as_slice()),
                    reference_tokens: usize::from(has_reference) * 3,
                    seed: 11,
                    sample_index: 7,
                },
            )
            .unwrap_or_else(|error| {
                panic!("{} modality plan is not executable: {error}", workflow.id())
            });
            assert_eq!(
                token_plan.loss_mask.len(),
                40 + usize::from(has_reference) * 3
            );
        }
    }
    let v2a = super::Ltx25Workflow::V2aLora.plan();
    assert!(!v2a.video.unwrap().is_generated);
    assert!(v2a.audio.unwrap().is_generated);
    let av_ic = super::Ltx25Workflow::Av2avIcLora.plan();
    assert_eq!(
        av_ic.video.unwrap().conditions[0].kind,
        super::Ltx25ConditionKind::Reference
    );
    assert_eq!(
        av_ic.audio.unwrap().conditions[0].kind,
        super::Ltx25ConditionKind::Reference
    );
}

#[test]
fn intrinsic_conditions_execute_exact_token_and_loss_masks() {
    let geometry = super::Ltx25TokenGeometry {
        frames: 3,
        height: 2,
        width: 2,
        spatial_scale: 32,
    };
    let prefix = super::Ltx25ModalityPlan {
        is_generated: true,
        conditions: &[super::Ltx25ConditionSpec {
            kind: super::Ltx25ConditionKind::Prefix,
            probability: 1.0,
            temporal_boundary: Some(1),
            spatial_region: None,
            spatial_scale_factor: None,
            temporal_scale_factor: None,
        }],
    };
    let plan =
        super::build_ltx25_token_plan(prefix, geometry, super::Ltx25ConditionInputs::default())
            .unwrap();
    assert_eq!(&plan.loss_mask[..4], &[0.0; 4]);
    assert_eq!(&plan.loss_mask[4..], &[1.0; 8]);

    let mask_values = [true, false, false, true];
    let mask = super::build_ltx25_token_plan(
        super::Ltx25ModalityPlan {
            is_generated: true,
            conditions: &[super::Ltx25ConditionSpec {
                kind: super::Ltx25ConditionKind::Mask,
                probability: 1.0,
                temporal_boundary: None,
                spatial_region: None,
                spatial_scale_factor: None,
                temporal_scale_factor: None,
            }],
        },
        super::Ltx25TokenGeometry {
            frames: 1,
            height: 2,
            width: 2,
            spatial_scale: 32,
        },
        super::Ltx25ConditionInputs {
            conditioning_mask: Some(&mask_values),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(mask.loss_mask, vec![0.0, 1.0, 1.0, 0.0]);

    let reference = super::build_ltx25_token_plan(
        super::Ltx25ModalityPlan {
            is_generated: true,
            conditions: super::REFERENCE,
        },
        super::Ltx25TokenGeometry {
            frames: 2,
            height: 1,
            width: 1,
            spatial_scale: 1,
        },
        super::Ltx25ConditionInputs {
            reference_tokens: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(reference.loss_mask, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    assert_eq!(reference.timestep_mask, reference.loss_mask);
}

#[test]
fn av_target_inventory_covers_attention_cross_modal_and_optional_ff() {
    let suffixes = vec![
        "to_q".to_string(),
        "ff.net.0.proj".to_string(),
        "ff.net.2".to_string(),
    ];
    let paths = super::ltx25_target_paths(1, &suffixes);
    for required in [
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.audio_attn1.to_q",
        "transformer_blocks.0.audio_to_video_attn.to_q",
        "transformer_blocks.0.video_to_audio_attn.to_q",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.audio_ff.net.2",
    ] {
        assert!(
            paths.iter().any(|path| path == required),
            "missing {required}"
        );
    }
    assert!(paths.iter().all(|path| !path.ends_with(".bias")));
    assert_eq!(paths.len(), 10);
    let ff = super::ltx25_target_paths(1, &["ff".to_string()]);
    assert_eq!(ff.len(), 4);
    assert!(ff.iter().any(|path| path.ends_with("audio_ff.net.0.proj")));
}

#[test]
fn dev_q4_identity_and_strict_alpha_fail_closed() {
    assert!(
        super::ensure_ltx25_training_variant(crate::dev_sampler::TransformerVariant::Dev).is_ok()
    );
    let distilled =
        super::ensure_ltx25_training_variant(crate::dev_sampler::TransformerVariant::Distilled)
            .unwrap_err()
            .to_string();
    assert!(distilled.contains("dev/q4"), "{distilled}");
    assert!(distilled.contains("distilled"), "{distilled}");
    assert!(super::ensure_ltx25_training_tier(super::SplitModel {
        quantized: true,
        bits: 4,
        group: 64,
    })
    .is_ok());
    for tier in [
        super::SplitModel::dense(),
        super::SplitModel {
            quantized: true,
            bits: 8,
            group: 64,
        },
    ] {
        assert!(super::ensure_ltx25_training_tier(tier).is_err());
    }
    assert!(super::validate_ltx25_adapter_scale(4.0).is_ok());
    for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(super::validate_ltx25_adapter_scale(alpha).is_err());
    }
}

#[test]
fn request_schema_requires_exact_condition_fields_and_tensor_keys() {
    let mut mask = request(1);
    mask.config.model_options.insert(
        "ltxWorkflow".into(),
        serde_json::json!("video_inpainting_lora"),
    );
    let missing_key = super::Ltx25TrainingPlan::from_request(&mask)
        .unwrap_err()
        .to_string();
    assert!(missing_key.contains("tensorKey"), "{missing_key}");
    mask.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{"type": "mask", "probability": 1.0, "tensorKey": "video_mask"}]
        }),
    );
    let parsed = super::Ltx25TrainingPlan::from_request(&mask).unwrap();
    assert_eq!(
        parsed.video.unwrap().conditions[0].tensor_key.as_deref(),
        Some("video_mask")
    );

    let mut prefix = request(1);
    prefix
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("video_extend_lora"));
    prefix.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{"type": "prefix", "probability": 1.0}]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&prefix)
        .unwrap_err()
        .to_string()
        .contains("temporalBoundary"));
    prefix.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{
                "type": "prefix", "probability": 1.0, "temporalBoundary": 8,
                "tensorKey": "must_not_be_ignored"
            }]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&prefix)
        .unwrap_err()
        .to_string()
        .contains("tensorKey"));

    let mut first_frame = request(1);
    first_frame
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("i2v_lora"));
    first_frame.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{"type": "firstFrame", "probability": 0.75}]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&first_frame)
        .unwrap_err()
        .to_string()
        .contains("canonical workflow"));

    let mut boundary = request(1);
    boundary
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("video_extend_lora"));
    boundary.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{"type": "prefix", "probability": 1.0, "temporalBoundary": 7}]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&boundary)
        .unwrap_err()
        .to_string()
        .contains("canonical workflow"));

    let mut crop = request(1);
    crop.config.model_options.insert(
        "ltxWorkflow".into(),
        serde_json::json!("video_outpainting_lora"),
    );
    crop.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [{
                "type": "spatialCrop", "probability": 1.0,
                "spatialRegion": [0, 0, 256, 576]
            }]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&crop)
        .unwrap_err()
        .to_string()
        .contains("canonical workflow"));

    let mut reference = request(1);
    reference
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("v2v_ic_lora"));
    reference.config.model_options.insert(
        "ltxVideo".into(),
        serde_json::json!({
            "isGenerated": true,
            "conditions": [
                {"type": "reference", "probability": 1.0},
                {"type": "firstFrame", "probability": 0.2}
            ]
        }),
    );
    assert!(super::Ltx25TrainingPlan::from_request(&reference)
        .unwrap_err()
        .to_string()
        .contains("tensorKey"));
}

#[test]
fn reference_scales_are_exact_for_each_modality() {
    let mut video = request(1);
    video
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("v2v_ic_lora"));
    video.config.model_options.insert(
            "ltxVideo".into(),
            serde_json::json!({
                "isGenerated": true,
                "conditions": [
                    {"type": "reference", "probability": 1.0, "tensorKey": "video_ref", "spatialScaleFactor": 16, "temporalScaleFactor": 8},
                    {"type": "firstFrame", "probability": 0.2}
                ]
            }),
        );
    let error = super::Ltx25TrainingPlan::from_request(&video)
        .unwrap_err()
        .to_string();
    assert!(error.contains("spatial=32, temporal=8"), "{error}");
    video.config.model_options.insert(
            "ltxVideo".into(),
            serde_json::json!({
                "isGenerated": true,
                "conditions": [
                    {"type": "reference", "probability": 1.0, "tensorKey": "video_ref", "spatialScaleFactor": 32, "temporalScaleFactor": 8},
                    {"type": "firstFrame", "probability": 0.2}
                ]
            }),
        );
    assert!(super::Ltx25TrainingPlan::from_request(&video).is_ok());

    let mut audio = request(1);
    audio
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("a2a_ic_lora"));
    audio.config.model_options.insert(
            "ltxAudio".into(),
            serde_json::json!({
                "isGenerated": true,
                "conditions": [{"type": "reference", "probability": 1.0, "tensorKey": "audio_ref", "spatialScaleFactor": 1, "temporalScaleFactor": 8}]
            }),
        );
    let error = super::Ltx25TrainingPlan::from_request(&audio)
        .unwrap_err()
        .to_string();
    assert!(error.contains("spatial=1, temporal=4"), "{error}");
    audio.config.model_options.insert(
            "ltxAudio".into(),
            serde_json::json!({
                "isGenerated": true,
                "conditions": [{"type": "reference", "probability": 1.0, "tensorKey": "audio_ref", "spatialScaleFactor": 1, "temporalScaleFactor": 4}]
            }),
        );
    assert!(super::Ltx25TrainingPlan::from_request(&audio).is_ok());
}

#[test]
fn weights_free_preflight_rejects_default_and_missing_bundle_requests() {
    let default_error = super::validate_ltx25_training_request(&request(1))
        .unwrap_err()
        .to_string();
    assert!(default_error.contains("ltxWorkflow"), "{default_error}");

    let mut missing = request(1);
    missing
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    let missing_error = super::validate_ltx25_training_request(&missing)
        .unwrap_err()
        .to_string();
    assert!(
        missing_error.contains("ltxPreparedBundlePath"),
        "{missing_error}"
    );

    let mut control = request(1);
    control.config.control_type = Some("pose".into());
    let control_error = super::validate_ltx25_training_request(&control)
        .unwrap_err()
        .to_string();
    assert!(control_error.contains("control_type"), "{control_error}");

    let mut full = request(1);
    full.config.full_finetune = true;
    let full_error = super::validate_ltx25_training_request(&full)
        .unwrap_err()
        .to_string();
    assert!(full_error.contains("full_finetune"), "{full_error}");
}
