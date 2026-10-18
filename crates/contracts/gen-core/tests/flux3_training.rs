//! Pure training-policy tests. All fixture identities are synthetic, not downloaded BFL files.
use gen_core::flux3::{training::*, DitConfig};

#[test]
fn single_pair_training_conditioning_uses_pinned_l_minus_one_grid() {
    let ids = training_condition_ids(2, 3).unwrap();
    assert_eq!(
        ids,
        vec![
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.5, -1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.5, 0.0, -1.0],
            [0.0, 0.5, 0.5, -1.0],
            [0.0, 0.5, 1.0, -1.0]
        ]
    );
    assert!(training_condition_ids(0, 3).is_err());
    assert!(training_condition_ids(usize::MAX, 2).is_err());
}

#[test]
fn shifted_discrete_schedule_is_not_inference_flow_map_or_continuous_uniform() {
    assert_eq!(training_sigma(0).unwrap(), 1.0);
    assert!((training_sigma(500).unwrap() - 0.8738966).abs() < 2e-7);
    assert!((training_sigma(999).unwrap() - 0.0068891468).abs() < 2e-9);
    assert!(training_sigma(1000).is_err());
    assert_eq!(density_index(0.0).unwrap(), 0);
    assert_eq!(density_index(0.9999).unwrap(), 999);
    assert!(density_index(1.0).is_err());
    assert!(density_index(f32::NAN).is_err());
}

#[test]
fn density_and_loss_weighting_follow_distinct_source_branches() {
    assert_eq!(
        Weighting::None.density(0.37, 99.0, 0.0, 1.0, 1.29).unwrap(),
        0.37
    );
    assert_eq!(
        Weighting::SigmaSqrt
            .density(0.37, 99.0, 0.0, 1.0, 1.29)
            .unwrap(),
        0.37
    );
    assert!(
        (Weighting::LogitNormal
            .density(0.0, -0.7, 0.2, 1.3, 1.29)
            .unwrap()
            - 0.32959884)
            .abs()
            < 2e-7
    );
    assert!((Weighting::Mode.density(0.37, 0.0, 0.0, 1.0, 1.29).unwrap() - 0.5415396).abs() < 2e-7);
    assert_eq!(Weighting::SigmaSqrt.loss_weight(0.25).unwrap(), 16.0);
    assert!((Weighting::Cosmap.loss_weight(0.25).unwrap() - 1.0185916).abs() < 2e-7);
    assert_eq!(Weighting::LogitNormal.loss_weight(0.25).unwrap(), 1.0);
    assert!(Weighting::parse("sigmoid").is_err());
    assert!(Weighting::SigmaSqrt.loss_weight(0.0).is_err());
}

#[test]
fn caption_dropout_never_duplicates_batch_or_drops_prior_and_keeps_window() {
    assert_eq!(
        caption_choice(0.09, 0.1, false, 160).unwrap(),
        CaptionChoice::EmptyAtWindow(160)
    );
    assert_eq!(
        caption_choice(0.09, 0.1, true, 160).unwrap(),
        CaptionChoice::Original
    );
    assert_eq!(
        caption_choice(0.1, 0.1, false, 160).unwrap(),
        CaptionChoice::Original
    );
    assert!(caption_choice(0.0, 1.1, false, 160).is_err());
    assert!(caption_choice(0.0, 0.1, false, 0).is_err());
}

#[test]
fn default_targets_roundtrip_the_existing_strict_generation_mapping() {
    let cfg = DitConfig {
        hidden: 32,
        heads: 4,
        depth: 1,
        depth_single: 2,
        mlp_ratio: 2.0,
        axes_dim: vec![2, 2, 2, 2],
        theta: 10000.0,
        image_in_channels: 8,
        image_cond_in_channels: 8,
        text_in_channels: 12,
        out_modes: vec!["image".into()],
        vec_in_dim: None,
        image_rope_scale: 0.5,
        flow_map: false,
    };
    let targets = lora_targets(&cfg, &[]).unwrap();
    assert_eq!(targets.len(), 30);
    for target in &targets {
        let (native, output, input) =
            gen_core::flux3::adapters::native_target(&target.diffusers, &cfg).unwrap();
        assert_eq!(native, target.native);
        assert_eq!((output, input), (target.output, target.input));
    }
    assert!(targets
        .iter()
        .any(|t| t.native == "txt_mode_blocks.0.mlp_in"));
    let selected = lora_targets(&cfg, &["to_q".into(), "ff.linear_out".into()]).unwrap();
    assert_eq!(selected.len(), 10);
    let exact = lora_targets(
        &cfg,
        &["transformer.single_transformer_blocks.1.ff.linear_out".into()],
    )
    .unwrap();
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].native, "single_blocks.1.mlp_out");
    let overlapping = lora_targets(
        &cfg,
        &["to_q".into(), "mode_blocks.image.0.attn.to_q".into()],
    )
    .unwrap();
    assert_eq!(
        overlapping.len(),
        5,
        "literal PEFT suffix/exact overlap must not duplicate trainable factors"
    );
    assert!(lora_targets(&cfg, &[".*to_q".into()]).is_err());
    assert!(lora_targets(&cfg, &["typo".into()]).is_err());
    assert!(lora_targets(&cfg, &["to_q".into(), "to_q".into()]).is_err());
}

#[test]
fn strong_resume_identity_binds_base_artifacts_data_and_every_config_field() {
    use std::collections::BTreeMap;
    let identity = ResumeIdentity {
        base_repo: "synthetic/flux3-tiny-NOT-BFL-weights".into(),
        base_revision: "a".repeat(40),
        artifacts: BTreeMap::from([("model.safetensors".into(), "b".repeat(64))]),
        dataset: "c".repeat(64),
        preprocessing: "d".repeat(64),
        configuration: BTreeMap::from([
            ("optimizer".into(), "adamw".into()),
            ("learning_rate".into(), "0.0003".into()),
            ("rank".into(), "3".into()),
            ("alpha".into(), "1.5".into()),
            ("scaling".into(), "alpha/rank".into()),
            ("precision".into(), "f32".into()),
            ("backend".into(), "candle-cpu".into()),
            ("scheduler".into(), "cosine:30".into()),
            ("accumulation".into(), "2".into()),
            ("rng".into(), "synthetic-counter-v1".into()),
        ]),
        factors: BTreeMap::from([
            ("single_blocks.0.q_proj.lora_A.weight".into(), vec![3, 32]),
            ("single_blocks.0.q_proj.lora_B.weight".into(), vec![32, 3]),
        ]),
    };
    identity.validate().unwrap();
    identity.ensure_matches(&identity).unwrap();
    for key in identity.configuration.keys() {
        let mut changed = identity.clone();
        changed.configuration.insert(key.clone(), "changed".into());
        let error = identity.ensure_matches(&changed).unwrap_err().to_string();
        assert!(error.contains("configuration"), "{key}: {error}");
    }
    let mut changed = identity.clone();
    changed.dataset = "e".repeat(64);
    assert!(identity
        .ensure_matches(&changed)
        .unwrap_err()
        .to_string()
        .contains("dataset"));
    let mut changed = identity.clone();
    changed
        .artifacts
        .insert("model.safetensors".into(), "f".repeat(64));
    assert!(identity
        .ensure_matches(&changed)
        .unwrap_err()
        .to_string()
        .contains("artifacts"));
    let mut changed = identity.clone();
    changed.base_revision = "f".repeat(40);
    assert!(identity
        .ensure_matches(&changed)
        .unwrap_err()
        .to_string()
        .contains("base_revision"));
    let mut changed = identity.clone();
    changed.factors.values_mut().next().unwrap()[0] = 4;
    assert!(identity
        .ensure_matches(&changed)
        .unwrap_err()
        .to_string()
        .contains("factors"));
    let mut malformed = identity;
    malformed.artifacts.values_mut().next().unwrap().clear();
    assert!(malformed.validate().is_err());
}

fn options_request(model_options: serde_json::Value) -> gen_core::train::TrainingRequest {
    gen_core::train::TrainingRequest {
        items: vec![gen_core::train::TrainingItem::captioned(
            "synthetic/0.png".into(),
            "synthetic".into(),
        )],
        config: gen_core::train::TrainingConfig {
            optimizer: "adamw".into(),
            train_dtype: "f32".into(),
            resolution: 512,
            timestep_type: "uniform".into(),
            model_options: model_options.as_object().unwrap().clone(),
            ..Default::default()
        },
        output_dir: "synthetic-out".into(),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec![],
        cancel: gen_core::CancelFlag::new(),
    }
}

/// The source options live in the `model_options["flux3"]` bag (strict), while the host's
/// product keys beside it (SceneWorks forwards its whole `advanced` object) are not FLUX3 options.
#[test]
fn source_options_are_read_strictly_from_the_namespaced_bag() {
    use options::{TrainingOptions, Workflow, OPTIONS_KEY};
    assert_eq!(OPTIONS_KEY, "flux3");
    let product = serde_json::json!({
        "mixedPrecision": "f32", "cacheLatents": true, "qualityPreset": "balanced",
        "flux3": {"weighting_scheme": "cosmap", "caption_dropout": 0.25, "max_grad_norm": 0.5}
    });
    let parsed = TrainingOptions::parse(&options_request(product)).unwrap();
    assert_eq!(parsed.weighting, Weighting::Cosmap);
    assert_eq!(parsed.caption_dropout, 0.25);
    assert_eq!(parsed.max_grad_norm, 0.5);
    assert_eq!(parsed.workflow, Workflow::Captioned);
    // Absent bag: the source defaults (uniform density -> weighting `none`).
    let defaults = TrainingOptions::parse(&options_request(serde_json::json!({}))).unwrap();
    assert_eq!(defaults.weighting, Weighting::None);
    assert_eq!(defaults.max_sequence_length, 512);
    for refused in [
        serde_json::json!({"flux3": {"use_dora": false, "typo_option": 1}}),
        serde_json::json!({"flux3": {"use_dora": true}}),
        serde_json::json!({"flux3": "cosmap"}),
    ] {
        assert!(
            TrainingOptions::parse(&options_request(refused.clone())).is_err(),
            "{refused}"
        );
    }
}

#[test]
fn captioned_runs_never_reach_the_reference_only_projections() {
    use options::Workflow;
    let reference = "content_mode_blocks.image_cond.0.qkv.lora_A.weight";
    let image = "content_mode_blocks.image.0.qkv.lora_A.weight";
    assert!(!Workflow::Captioned.trains(reference));
    assert!(Workflow::Captioned.trains(image));
    assert!(Workflow::Captioned.trains("single_blocks.0.linear1.lora_B.weight"));
    assert!(Workflow::PairedEdit.trains(reference));
    assert!(Workflow::PairedEdit.trains(image));
}
