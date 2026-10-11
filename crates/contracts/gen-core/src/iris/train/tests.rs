use super::*;
use crate::runtime::CancelFlag;

fn tiny_config() -> IrisConfig {
    IrisConfig::parse(
        "model:\n  dual_depth: 2\n  hidden_size: 64\n  depth: 3\n  num_heads: 4\n  num_kv_heads: 2\n  \
         patch_size: 4\n  text_dim: 32\n  text_len: 10\n  text_lap_num_heads: 4\n  pixel:\n    \
         depth: 2\n    hidden_size: 8\n    attn_hidden_size: 32\n    num_heads: 2\n\
         text_encoder:\n  dim: 32\n  max_length: 10\n",
    )
    .unwrap()
}

fn request(config: TrainingConfig) -> TrainingRequest {
    TrainingRequest {
        items: vec![TrainingItem::captioned("a.png".into(), "a fox".into())],
        config,
        output_dir: "/out".into(),
        file_name: "style.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    }
}

fn base_cfg() -> TrainingConfig {
    TrainingConfig {
        resolution: 16,
        steps: 4,
        ..Default::default()
    }
}

fn defaults() -> TrainFlowDefaults {
    TrainFlowDefaults::parse("flow:\n  shift: 4.0\n").unwrap()
}

fn linears() -> Vec<String> {
    [
        "blocks.0.attn.q_proj_x",
        "blocks.0.mlp_x.w1",
        "blocks.2.attn_proj",
        "pixel_blocks.0.adaln",
        "s_embedder.proj",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[test]
fn mix_seed_matches_the_upstream_function() {
    // Values printed by `iris3b.seeding.mix_seed` at the pinned revision.
    assert_eq!(mix_seed(&[1, 0, 1, 0]), 1287795393316269613);
    assert_eq!(mix_seed(&[1, 0, 1, 1]), 1177922293586589958);
    assert_eq!(mix_seed(&[42, 0, 3, 17]), 828574415078863100);
    assert_eq!(mix_seed(&[0]), 7070836379803831727);
    assert_eq!(mix_seed(&[(1u64 << 63) + 5, 7]), 8606081053527696572);
    assert_eq!(batch_seed(42, 3, 17), 828574415078863100);
}

#[test]
fn training_schedule_matches_flow_schedule() {
    // `FlowSchedule(1000, 4.0)`: sigmas (f32) and truncated model times.
    let s = TrainSchedule::new(1000, 4.0);
    let want_sigma = [
        (0, 0.0f32),
        (1, 0.003_988_036),
        (499, 0.799_359_2),
        (500, 0.8),
        (998, 0.999_499_26),
        (999, 0.999_749_84),
    ];
    for (i, w) in want_sigma {
        assert!(
            (s.sigmas[i] - w).abs() <= 1e-7,
            "sigma[{i}] {} vs {w}",
            s.sigmas[i]
        );
    }
    let want_t = [
        (0, 0.0),
        (1, 3.0),
        (499, 799.0),
        (500, 800.0),
        (998, 999.0),
        (999, 999.0),
    ];
    for (i, w) in want_t {
        assert_eq!(s.model_times[i], w, "model_time[{i}]");
    }
}

#[test]
fn resolution_shift_and_scale_lr_match_upstream() {
    assert_eq!(resolution_shift(1024, "sd3", 1.0, 256).unwrap(), 2.0);
    assert!((resolution_shift(4096, "flux", 1.0, 256).unwrap() - 3.158192909689767).abs() < 1e-12);
    assert_eq!(resolution_shift(10, "none", 4.0, 256).unwrap(), 4.0);
    assert!(resolution_shift(10, "bogus", 4.0, 256).is_err());
    assert!((scale_lr(1e-4, "sqrt", 8, 256).unwrap() - 1.767766952966369e-05).abs() < 1e-18);
    assert_eq!(scale_lr(1e-4, "none", 8, 256).unwrap(), 1e-4);
}

#[test]
fn lr_factor_is_lambda_lr() {
    // Warmup ramps from exactly 0 (upstream's first optimizer step runs at lr 0).
    assert_eq!(lr_factor(LrShape::Constant, 0, 4, 100), 0.0);
    assert_eq!(lr_factor(LrShape::Constant, 2, 4, 100), 0.5);
    assert_eq!(lr_factor(LrShape::Constant, 4, 4, 100), 1.0);
    assert_eq!(lr_factor(LrShape::Constant, 0, 0, 100), 1.0);
    assert_eq!(lr_factor(LrShape::Cosine, 4, 4, 104), 1.0);
    assert!(lr_factor(LrShape::Cosine, 104, 4, 104).abs() < 1e-12);
    assert!((lr_factor(LrShape::Cosine, 54, 4, 104) - 0.5).abs() < 1e-12);
    assert!((clip_coefficient(0.5, 2.0) - 0.5 / (2.0 + 1e-6)).abs() < 1e-15);
    assert_eq!(clip_coefficient(0.5, 0.1), 1.0);
}

#[test]
fn data_walk_is_the_ranged_sampler() {
    let w = DataWalk::new(10, 4).unwrap();
    assert_eq!((w.covered, w.batches_per_epoch()), (10, 3));
    assert_eq!(w.batch(2), 8..10, "the last partial batch is kept");
    // 1000 items: 640 chunks of 1 — upstream leaves the 360-sample tail unassigned.
    let w = DataWalk::new(1000, 8).unwrap();
    assert_eq!(w.covered, 640);
    assert_eq!(w.batches_per_epoch(), 80);
    let w = DataWalk::new(1300, 1).unwrap();
    assert_eq!(w.covered, 1280);
    assert!(DataWalk::new(0, 1).is_err());
    // The dropped tail is named at run start: 641 items cover 640, one is never trained on.
    let w = DataWalk::new(641, 4).unwrap();
    assert_eq!((w.covered, w.unused_items()), (640, 1));
    let msg = w.unused_tail_warning().expect("a dropped tail warns");
    assert!(msg.contains("1 of 641 dataset items"), "{msg}");
    assert!(msg.contains("indices 640..641"), "{msg}");
    for n in [10, 640, 1280] {
        let w = DataWalk::new(n, 4).unwrap();
        assert_eq!(w.unused_items(), 0);
        assert_eq!(w.unused_tail_warning(), None, "{n} items use every item");
    }
    assert_eq!(total_optimizer_steps(10, 3, Some(2), 100), 8);
    assert_eq!(total_optimizer_steps(10, 3, Some(2), 5), 5);
    assert_eq!(total_optimizer_steps(10, 3, None, 7), 7);
}

#[test]
fn draws_are_positional_and_ordered() {
    let a = draw_batch(
        7,
        2,
        12,
        0.5,
        TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
        1000,
    );
    let b = draw_batch(
        7,
        2,
        12,
        0.5,
        TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
        1000,
    );
    assert_eq!(a, b, "same key, same draws");
    let c = draw_batch(
        8,
        2,
        12,
        0.5,
        TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
        1000,
    );
    assert_ne!(a.noise, c.noise);
    assert_eq!(a.noise.len(), 24);
    assert!(a.timestep_idx.iter().all(|&t| t < 1000));
    // No dropout draw when text_dropout = 0: the timestep stream starts at the first draw.
    let d = draw_batch(7, 2, 12, 0.0, TimestepSampler::Uniform, 1000);
    assert_eq!(d.drop, vec![false, false]);
    let mut rng = HostRng::new(7);
    assert_eq!(d.timestep_idx[0], rng.below(1000));
    // The normal generator has unit variance.
    let mut rng = HostRng::new(3);
    let xs = rng.normals_f32(20_000);
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    assert!(
        mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
        "{mean} {var}"
    );
}

#[test]
fn captions_select_uniformly_among_present_fields() {
    let mut item = TrainingItem::captioned("a.png".into(), "plain".into());
    item.model_options.insert(
        ITEM_CAPTIONS_KEY.into(),
        json!({"short": "a fox", "long": "a red fox in snow", "empty": ""}),
    );
    let fields = vec!["short".to_string(), "long".to_string(), "empty".to_string()];
    let mut seen = std::collections::BTreeSet::new();
    for idx in 0..64 {
        let mut rng = HostRng::new(caption_seed(1, 1, idx));
        seen.insert(select_caption(&item, "caption", &fields, &mut rng).unwrap());
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        ["a fox", "a red fox in snow"],
        "empty fields are never chosen"
    );
    let mut rng = HostRng::new(0);
    assert_eq!(
        select_caption(&item, "caption", &[], &mut rng).unwrap(),
        "plain"
    );
    assert!(select_caption(&item, "missing", &[], &mut rng).is_err());
    assert_eq!(
        item_caption_variants(&item, "caption", &fields),
        ["a fox", "a red fox in snow", "", "plain"]
    );
}

#[test]
fn preprocessing_resizes_the_short_side_and_center_crops() {
    // A 6x4 (w x h) gradient → size 2: short side h=4 → (3, 2), crop left = round(0.5) = 0.
    let (w, h) = (6usize, 4usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                rgb[(y * w + x) * 3 + c] = (x * 40 + c * 10) as u8;
            }
        }
    }
    let out = preprocess_rgb(&rgb, w, h, 2).unwrap();
    assert_eq!(out.chw.len(), 12);
    assert!(out.chw.iter().all(|v| (-1.0..=1.0).contains(v)));
    // Rows are constant (the gradient runs along x), channels are offset.
    assert_eq!(out.chw[0], out.chw[2], "row 0 and row 1 agree at x=0");
    assert!(out.chw[1] > out.chw[0], "x increases to the right");
    // A uniform image maps to exactly x/127.5 - 1.
    let flat = vec![255u8; 5 * 5 * 3];
    let out = preprocess_rgb(&flat, 5, 5, 4).unwrap();
    assert!(out.chw.iter().all(|&v| v == 1.0));
}

#[test]
fn muon_routing_follows_build_muon_param_groups() {
    let cfg = tiny_config().model;
    let r = |k: &str, nd: usize| full_param_route(k, nd, OptimizerKind::Muon, &cfg);
    assert_eq!(
        r("blocks.0.attn.q_proj_x.weight", 2),
        ParamRoute::Muon { split: None }
    );
    assert_eq!(
        r("blocks.0.attn.qkv_x.weight", 2),
        ParamRoute::Muon { split: Some(3) }
    );
    assert_eq!(
        r("y_embedder.layer_blocks.0.attn.qkv.weight", 2),
        ParamRoute::Muon { split: Some(3) }
    );
    assert_eq!(
        r("pixel_blocks.1.attn.qkv.weight", 2),
        ParamRoute::Muon { split: Some(3) }
    );
    assert_eq!(
        r("modulation_cores.adaln_img.weight", 2),
        ParamRoute::Muon { split: Some(6) }
    );
    for boundary in [
        "s_embedder.proj.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "y_embedder.layer_pool.weight",
        "y_embedder.refiner.proj.weight",
        "pixel_embedder.proj.weight",
        "pixel_blocks.0.adaln.weight",
        "final_layer.linear.weight",
    ] {
        assert_eq!(r(boundary, 2), ParamRoute::AdamW, "{boundary}");
    }
    assert_eq!(r("blocks.0.attn.proj_x.bias", 1), ParamRoute::AdamW);
    assert_eq!(r("blocks.0.norm_x1.weight", 1), ParamRoute::AdamW);
    assert_eq!(r("y_pos_embedding", 3), ParamRoute::AdamW);
    assert_eq!(
        full_param_route("blocks.0.mlp_x.w1.weight", 2, OptimizerKind::AdamW, &cfg),
        ParamRoute::AdamW
    );
    // Dead text tail: only a final DUAL block under `keep`.
    let mut all_dual = cfg.clone();
    all_dual.dual_depth = all_dual.depth;
    assert_eq!(
        full_param_route(
            "blocks.2.mlp_y.w1.weight",
            2,
            OptimizerKind::AdamW,
            &all_dual
        ),
        ParamRoute::Frozen
    );
    assert_ne!(r("blocks.2.mlp.w1.weight", 2), ParamRoute::Frozen);
    assert_eq!(
        adapter_param_route("blocks.0.mlp_x.w1", 2, OptimizerKind::Muon),
        ParamRoute::Muon { split: None }
    );
    assert_eq!(
        adapter_param_route("pixel_blocks.0.adaln", 2, OptimizerKind::Muon),
        ParamRoute::AdamW
    );
    assert_eq!(
        adapter_param_route("blocks.0.mlp_x.w1", 2, OptimizerKind::AdamW),
        ParamRoute::AdamW
    );
    let scales = muon_split_scales(MuonAdjustLr::RmsNorm, &[64; 6], 64);
    assert!((scales[0] - (0.2 * 8.0) / (0.2 * 384f64.sqrt())).abs() < 1e-12);
    assert_eq!(MuonAdjustLr::SpectralNorm.ratio(16, 4), 2.0);
}

#[test]
fn random_init_follows_initialize_weights() {
    assert_eq!(
        init_kind("final_layer.linear.weight", &[3, 8], None, true),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("final_layer.linear.bias", &[3], Some(8), true),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("modulation_cores.adaln_img.weight", &[384, 64], None, true),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("pixel_blocks.0.adaln.bias", &[64], Some(64), true),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("blocks.1.adaln_img.bias", &[384], None, false),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("modulation_cores.adaln_img.weight", &[384, 64], None, false),
        InitKind::Uniform { bound: 0.125 }
    );
    assert_eq!(
        init_kind("blocks.0.norm_x1.weight", &[64], None, true),
        InitKind::Ones
    );
    assert_eq!(
        init_kind("y_pos_embedding", &[1, 10, 64], None, true),
        InitKind::Normal { std: 1.0 }
    );
    assert_eq!(
        init_kind("t_embedder.mlp.0.weight", &[64, 256], None, true),
        InitKind::Normal { std: 0.02 }
    );
    assert_eq!(
        init_kind("s_embedder.proj.bias", &[64], Some(48), true),
        InitKind::Zeros
    );
    assert_eq!(
        init_kind("s_embedder.proj.weight", &[64, 48], None, true),
        InitKind::Uniform {
            bound: (6.0f64 / 112.0).sqrt()
        }
    );
    assert_eq!(
        init_kind("blocks.0.attn.proj_x.bias", &[64], Some(16), true),
        InitKind::Uniform { bound: 0.25 }
    );
}

#[test]
fn adapter_targets_default_to_the_trunk_and_selectors_are_strict() {
    let all = linears();
    assert_eq!(
        select_adapter_targets(&all, &[]).unwrap(),
        [
            "blocks.0.attn.q_proj_x",
            "blocks.0.mlp_x.w1",
            "blocks.2.attn_proj"
        ]
    );
    assert_eq!(
        select_adapter_targets(&all, &["w1".into(), "adaln".into()]).unwrap(),
        ["blocks.0.mlp_x.w1", "pixel_blocks.0.adaln"]
    );
    assert!(select_adapter_targets(&all, &["to_q".into()]).is_err());
    let keys = [
        ("blocks.0.mlp_x.w1.weight", 2),
        ("blocks.0.norm_x1.weight", 1),
        ("y_pos_embedding", 3),
    ];
    assert_eq!(linear_paths_from_keys(keys), ["blocks.0.mlp_x.w1"]);
}

#[test]
fn adapter_metadata_round_trips() {
    let m = AdapterMetadata {
        network_type: "lokr".into(),
        rank: 4,
        alpha: 2.5,
        decompose_factor: Some(-1),
        weights: WeightsSelect::Ema,
        steps: 12,
        base_identity: "abc".into(),
        prediction: Prediction::Velocity,
        shift: 4.0,
        targets: vec!["blocks.0.mlp_x.w1".into()],
    };
    let map = m.to_map();
    assert_eq!(map["family"], "iris");
    assert_eq!(map[META_ARTIFACT], "adapter");
    assert_eq!(AdapterMetadata::from_map(&map).unwrap(), m);
    let mut wrong = map.clone();
    wrong.insert(META_TASK.into(), "depth".into());
    assert!(AdapterMetadata::from_map(&wrong).is_err());
}

#[test]
fn exported_config_round_trips_through_the_reader() {
    let cfg = tiny_config();
    let plan = IrisTrainPlan::resolve(&request(base_cfg()), &cfg, &defaults(), &[]).unwrap();
    let yaml = export_config_yaml(&cfg, &plan.flow, true);
    let back = IrisConfig::parse(&yaml).unwrap();
    assert_eq!(back, cfg);
    let flow = TrainFlowDefaults::parse(&yaml).unwrap();
    assert_eq!(flow, defaults());
    back.validate_supported().unwrap();
}

#[test]
fn plan_defaults_are_upstreams() {
    let cfg = tiny_config();
    let plan = IrisTrainPlan::resolve(&request(base_cfg()), &cfg, &defaults(), &linears()).unwrap();
    assert_eq!(plan.text_dropout, 0.1);
    assert_eq!(plan.gradient_clip, 0.5);
    assert_eq!(plan.ema, Some(0.9999));
    assert_eq!(plan.optimizer.betas, (0.9, 0.95));
    assert_eq!(plan.optimizer.muon_adjust_lr, MuonAdjustLr::RmsNorm);
    assert!(plan.optimizer.muon_nesterov);
    assert_eq!(plan.flow.prediction, Prediction::Velocity);
    assert_eq!(
        plan.flow.sampler,
        TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0
        }
    );
    assert_eq!(plan.mixed_precision, MixedPrecision::Bf16);
    assert_eq!(plan.export_weights, WeightsSelect::Raw);
    assert_eq!(plan.init, InitMode::Weights);
    assert!(plan.preview.prompts.is_empty());
    assert_eq!(plan.preview.every, 0);
    assert!(matches!(plan.artifact, ArtifactPlan::Lora { rank: 16, .. }));
    assert_eq!(plan.artifact.targets().len(), 3);
}

#[test]
fn plan_honours_options_and_refuses_what_it_cannot_honour() {
    let cfg = tiny_config();
    let resolve =
        |c: TrainingConfig| IrisTrainPlan::resolve(&request(c), &cfg, &defaults(), &linears());
    let mut c = base_cfg();
    c.full_finetune = true;
    c.optimizer = "muon".into();
    c.lr_scheduler = LrSchedule::Cosine;
    c.model_options.insert(
        OPTIONS_KEY.into(),
        json!({"init": "random", "prediction": "x", "shift_law": "sd3", "flow_shift": 1.0,
               "text_dropout": 0.0, "ema_enabled": false, "betas": [0.8, 0.9],
               "muon_adjust_lr": "spectral_norm", "auto_lr": "linear", "base_batch_size": 2}),
    );
    let p = resolve(c).unwrap();
    assert_eq!(p.init, InitMode::Random);
    assert_eq!(p.flow.prediction, Prediction::Clean);
    // 16px / patch 4 = 16 tokens; sd3 law anchored at 256 tokens.
    assert!((p.flow.shift - 0.25).abs() < 1e-12);
    assert_eq!(p.export_weights, WeightsSelect::Raw);
    assert_eq!(p.optimizer.kind, OptimizerKind::Muon);
    assert_eq!(
        p.optimizer.lr,
        p.optimizer.base_lr * 0.5,
        "batch 1 / base 2 linear"
    );

    for (field, mutate) in [
        (
            "gradient",
            Box::new(|c: &mut TrainingConfig| c.gradient_checkpointing = true)
                as Box<dyn Fn(&mut TrainingConfig)>,
        ),
        (
            "loss",
            Box::new(|c: &mut TrainingConfig| c.loss_type = "mae".into()),
        ),
        (
            "lr_scheduler",
            Box::new(|c: &mut TrainingConfig| c.lr_scheduler = LrSchedule::Linear),
        ),
        (
            "optimizer",
            Box::new(|c: &mut TrainingConfig| c.optimizer = "prodigy".into()),
        ),
        (
            "train_dtype",
            Box::new(|c: &mut TrainingConfig| c.train_dtype = "fp16".into()),
        ),
        (
            "timestep_type",
            Box::new(|c: &mut TrainingConfig| c.timestep_type = "shift".into()),
        ),
        (
            "timestep_bias",
            Box::new(|c: &mut TrainingConfig| c.timestep_bias = "high_noise".into()),
        ),
        (
            "not an Iris",
            Box::new(|c: &mut TrainingConfig| {
                c.model_options
                    .insert(OPTIONS_KEY.into(), json!({"bogus": 1}));
            }),
        ),
        (
            "init random",
            Box::new(|c: &mut TrainingConfig| {
                c.model_options
                    .insert(OPTIONS_KEY.into(), json!({"init": "random"}));
            }),
        ),
    ] {
        let mut c = base_cfg();
        mutate(&mut c);
        match resolve(c) {
            Err(Error::Unsupported(m)) => assert!(m.contains(field), "{field}: {m}"),
            other => panic!("{field}: expected a typed refusal, got {other:?}"),
        }
    }
    let mut c = base_cfg();
    c.resolution = 18;
    assert!(resolve(c).is_err(), "resolution must be on the patch grid");
    let mut c = base_cfg();
    c.lora_target_modules = vec!["to_q".into()];
    assert!(resolve(c).is_err());
}

#[test]
fn checkpoints_publish_atomically_and_prune() {
    let root = tempfile::tempdir().unwrap();
    let state = |step: u64| CheckpointState {
        step,
        epoch: 1,
        batches_consumed: step as usize,
        scheduler_step: step,
        lr: 1e-4,
        nan_count: 0,
        last_loss: 0.5,
        artifact: "lora".into(),
        optimizer: "adamw".into(),
        ema_decay: Some(0.99),
        seed: u64::MAX,
        backend: "mlx".into(),
        state_identity: "s".into(),
        data_identity: "d".into(),
        dataset_fingerprint: "f".into(),
        base_identity: "b".into(),
    };
    for step in [1u64, 2, 3, 4] {
        let staging = checkpoint_staging_dir(root.path(), step);
        std::fs::create_dir_all(&staging).unwrap();
        atomic_write(
            &staging.join(CKPT_STATE),
            state(step).to_json().to_string().as_bytes(),
        )
        .unwrap();
        publish_checkpoint(root.path(), &staging, step).unwrap();
    }
    // A half-written staging directory is never a resume candidate.
    std::fs::create_dir_all(checkpoint_staging_dir(root.path(), 9)).unwrap();
    std::fs::create_dir_all(root.path().join(checkpoint_dir_name(8))).unwrap();
    let latest = latest_checkpoint(root.path()).unwrap();
    assert!(latest.ends_with("step_00000004"));
    let back = read_checkpoint_state(&latest).unwrap();
    assert_eq!(back, state(4));
    assert_eq!(prune_checkpoints(root.path(), 2, &[1]).unwrap(), vec![2]);
    let left: Vec<u64> = list_checkpoints(root.path())
        .into_iter()
        .map(|(s, _)| s)
        .collect();
    assert_eq!(left, vec![1, 3, 4]);
    // A stale pointer cannot hide a newer published checkpoint.
    atomic_write(&root.path().join(CKPT_LATEST), b"step_00000003").unwrap();
    assert!(latest_checkpoint(root.path())
        .unwrap()
        .ends_with("step_00000004"));
}

#[test]
fn resume_checks_state_and_data_identity() {
    let cfg = tiny_config();
    let plan = IrisTrainPlan::resolve(&request(base_cfg()), &cfg, &defaults(), &linears()).unwrap();
    let saved = CheckpointState {
        step: 2,
        epoch: 1,
        batches_consumed: 2,
        scheduler_step: 2,
        lr: 1e-4,
        nan_count: 0,
        last_loss: 0.1,
        artifact: "lora".into(),
        optimizer: "adamw".into(),
        ema_decay: Some(0.9999),
        seed: 0,
        backend: "mlx".into(),
        state_identity: plan.state_identity(),
        data_identity: plan.data_identity(),
        dataset_fingerprint: "ds".into(),
        base_identity: "base".into(),
    };
    check_resume(&saved, &plan, "ds", "base").unwrap();
    assert!(check_resume(&saved, &plan, "other", "base").is_err());
    assert!(check_resume(&saved, &plan, "ds", "other-base").is_err());
    let mut new_phase = plan.clone();
    new_phase.resume_data_policy = ResumeDataPolicy::NewPhase;
    check_resume(&saved, &new_phase, "other", "base").unwrap();
    let mut other_rank = saved.clone();
    other_rank.state_identity = "artifact=lora:8".into();
    assert!(check_resume(&other_rank, &new_phase, "ds", "base").is_err());
}

#[test]
fn backbone_shapes_enumerate_the_fixture_state_dict_exactly() {
    // The miniature backbone written by upstream `IrisDiT(cfg).state_dict()` (dump_iris_golden.py).
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../media/mlx-gen/mlx-gen-iris/tests/fixtures/tiny-snapshot/iris");
    let cfg = IrisConfig::from_dir(&dir).unwrap();
    let bytes = std::fs::read(dir.join(super::super::BACKBONE_WEIGHTS_FILE)).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut want: Vec<(String, Vec<usize>)> = st
        .tensors()
        .into_iter()
        .map(|(k, v)| (k, v.shape().to_vec()))
        .collect();
    want.sort();
    let mut got = backbone_tensor_shapes(&cfg.model);
    got.sort();
    assert_eq!(got, want);
    // Every key routes and inits.
    for (k, shape) in &got {
        let _ = full_param_route(k, shape.len(), OptimizerKind::Muon, &cfg.model);
        let fan_in = k
            .strip_suffix(".bias")
            .and_then(|m| got.iter().find(|(w, _)| *w == format!("{m}.weight")))
            .map(|(_, s)| s[1]);
        let _ = init_kind(k, shape, fan_in, true);
    }
}

/// Upstream exports a full model's EMA; an adapter exports its raw factors by default (upstream has
/// no adapter EMA, and one seeded from the zero delta lags the run). Previews default to what the
/// run exports.
#[test]
fn export_and_preview_weights_default_per_artifact_kind() {
    let cfg = tiny_config();
    let resolve = |full: bool, network: NetworkType, opts: serde_json::Value| {
        let mut c = base_cfg();
        c.full_finetune = full;
        c.network_type = network;
        c.model_options.insert(OPTIONS_KEY.into(), opts);
        let p = IrisTrainPlan::resolve(&request(c), &cfg, &defaults(), &linears()).unwrap();
        (p.export_weights, p.preview.weights)
    };
    use WeightsSelect::{Ema, Raw};
    assert_eq!(resolve(false, NetworkType::Lora, json!({})), (Raw, Raw));
    assert_eq!(resolve(false, NetworkType::Lokr, json!({})), (Raw, Raw));
    assert_eq!(resolve(true, NetworkType::Lora, json!({})), (Ema, Ema));
    assert_eq!(
        resolve(true, NetworkType::Lora, json!({"ema_enabled": false})),
        (Raw, Raw)
    );
    // Explicit choices still win on either kind.
    assert_eq!(
        resolve(false, NetworkType::Lokr, json!({"export_weights": "ema"})),
        (Ema, Raw)
    );
    assert_eq!(
        resolve(true, NetworkType::Lora, json!({"preview_weights": "raw"})),
        (Ema, Raw)
    );
}

/// The shared preview contract: empty `sample_prompts` disables sampling whatever the cadence, at
/// most `PREVIEW_PROMPT_CAP` prompts render, and upstream's defaults are an explicit opt-in.
#[test]
fn previews_follow_the_shared_sample_contract() {
    let cfg = tiny_config();
    let resolve = |prompts: Vec<&str>, opts: serde_json::Value| {
        let mut c = base_cfg();
        c.sample_every = 2;
        c.sample_steps = 3;
        c.sample_prompts = prompts.into_iter().map(str::to_string).collect();
        c.model_options.insert(OPTIONS_KEY.into(), opts);
        IrisTrainPlan::resolve(&request(c), &cfg, &defaults(), &linears())
    };
    let p = resolve(vec![], json!({})).unwrap();
    assert_eq!((p.preview.every, p.preview.prompts.len()), (0, 0));
    let p = resolve(vec!["a", "b", "c", "d", "e", "f"], json!({})).unwrap();
    assert_eq!(p.preview.every, 2);
    assert_eq!(p.preview.prompts, ["a", "b", "c", "d"]);
    let p = resolve(vec![], json!({"upstream_validation_prompts": true})).unwrap();
    assert_eq!(p.preview.every, 2);
    assert_eq!(p.preview.prompts, DEFAULT_VALIDATION_PROMPTS.to_vec());
    let err = resolve(vec!["a"], json!({"upstream_validation_prompts": true})).unwrap_err();
    assert!(err.to_string().contains("one or the other"), "{err}");
}

/// `steps`, `save_every`, `sample_every` and `lr_warmup_steps` are the shared contract's
/// micro-steps: the plan holds optimizer steps, and a cadence that is not a whole number of
/// accumulation windows is refused rather than rounded.
#[test]
fn micro_step_fields_convert_to_optimizer_steps() {
    let cfg = tiny_config();
    let resolve = |steps: u32, save: u32, sample: u32, warmup: u32| {
        let mut c = base_cfg();
        c.gradient_accumulation = 2;
        c.steps = steps;
        c.save_every = save;
        c.sample_every = sample;
        c.sample_steps = 2;
        c.sample_prompts = vec!["a fox".into()];
        c.lr_warmup_steps = warmup;
        IrisTrainPlan::resolve(&request(c), &cfg, &defaults(), &linears())
    };
    let p = resolve(8, 4, 2, 3).unwrap();
    assert_eq!(
        (p.max_steps, p.save_every, p.preview.every, p.warmup_steps),
        (4, 2, 1, 2)
    );
    for (steps, save, sample, field) in [
        (7, 4, 2, "steps"),
        (8, 3, 2, "save_every"),
        (8, 4, 3, "sample_every"),
    ] {
        let err = resolve(steps, save, sample, 0).unwrap_err().to_string();
        assert!(
            err.contains(field) && err.contains("not a multiple of gradient_accumulation"),
            "{err}"
        );
    }
}
