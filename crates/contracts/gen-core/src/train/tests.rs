use super::*;

#[test]
fn preview_sample_failed_warning_is_coded_and_names_the_sample() {
    let w = preview_sample_failed(40, 2, &"out of memory");
    assert_eq!(w.code, PREVIEW_SAMPLE_FAILED);
    assert_eq!(
        w.message,
        "preview sample 2 at step 40 failed to render: out of memory"
    );
}

#[test]
fn training_item_ctors_set_control() {
    let img = PathBuf::from("a.png");
    let lora = TrainingItem::captioned(img.clone(), "a cat".into());
    assert_eq!(
        lora.control_image_path, None,
        "captioned = no control (LoRA)"
    );

    let ctrl = TrainingItem::with_control(img.clone(), "a cat".into(), PathBuf::from("a.pose.png"));
    assert_eq!(ctrl.control_image_path, Some(PathBuf::from("a.pose.png")));
    assert_eq!(ctrl.image_path, img);
}

#[test]
fn config_default_has_no_control_type() {
    // Additive: LoRA callers building via `..Default::default()` get `control_type: None` and
    // are unaffected; only a control-branch trainer sets it.
    assert_eq!(TrainingConfig::default().control_type, None);
}

#[test]
fn full_finetune_is_opt_in_and_the_flag_discriminates() {
    // Additive (sc-14056): every `..Default::default()` caller and the whole conformance surface
    // trains a LoRA/LoKr adapter; only a full-base-fine-tune run opts in.
    //
    // This deliberately does NOT just assert the default value — a bare
    // `assert!(!TrainingConfig::default().full_finetune)` is a constant that passes no matter what
    // the floor does with the field, and would still pass if `validate_full_finetune_request` were
    // deleted. Instead assert the *behavioral* difference the flag must make: on one and the same
    // adapter-only trainer, two requests differing ONLY by this field land on opposite sides of
    // the floor. Flip the default, or stop reading the field, and exactly one half breaks.
    let desc = trainer_desc(false);
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];

    let adapter_req = train_req(None, items);
    assert!(
        !adapter_req.config.full_finetune,
        "the default request must be the adapter path"
    );
    assert!(
        validate_full_finetune_request(&desc, &adapter_req).is_ok(),
        "an adapter-only trainer must accept the default (non-full) request"
    );

    let mut full_req = adapter_req;
    full_req.config.full_finetune = true;
    assert!(
        validate_full_finetune_request(&desc, &full_req).is_err(),
        "flipping only `full_finetune` must change the outcome — otherwise the field is inert"
    );
}

#[test]
fn validate_training_techniques_floor() {
    // sc-24826 (epic 2123 E3/E5): weight noising is refused unless the descriptor declares it,
    // refused with a full fine-tune, and malformed sigmas are refused outright.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut noisy_desc = trainer_desc_with(false, true);
    noisy_desc.techniques.weight_noise = true;

    // Off (the default) ⇒ no-op everywhere.
    let off = train_req(None, items);
    assert_eq!(off.config.weight_noise_sigma, 0.0);
    assert!(validate_training_techniques(&plain, &off).is_ok());
    assert!(validate_training_techniques(&noisy_desc, &off).is_ok());

    // On ⇒ typed Unsupported on a trainer that does not declare it; accepted where declared.
    let mut on = off.clone();
    on.config.weight_noise_sigma = 0.0125;
    let err = validate_training_techniques(&plain, &on).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("weight noising")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&noisy_desc, &on).is_ok());

    // On + full fine-tune ⇒ refused even where weight noise is declared (E5).
    let mut full = on.clone();
    full.config.full_finetune = true;
    let err = validate_training_techniques(&noisy_desc, &full).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("full base fine-tune")),
        "{err:?}"
    );

    // Malformed ⇒ Msg, regardless of support.
    for bad in [-0.01f32, f32::NAN, f32::INFINITY] {
        let mut r = off.clone();
        r.config.weight_noise_sigma = bad;
        let err = validate_training_techniques(&noisy_desc, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{bad}: {err:?}");
    }
}

#[test]
fn validate_training_techniques_gates_the_training_adapter() {
    // sc-25213 (E3): a training adapter is refused unless the descriptor declares it, and with a
    // full fine-tune even where declared (it would be baked into the fine-tuned weights).
    // Mutations: drop the `!desc.techniques.training_adapter` arm ⇒ the `plain` assert fails;
    // drop the `full_finetune` arm ⇒ the `full` assert fails.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut declaring = trainer_desc_with(false, true);
    declaring.techniques.training_adapter = true;

    let off = train_req(None, items);
    assert!(validate_training_techniques(&plain, &off).is_ok());

    let mut on = off.clone();
    on.config.training_adapter = Some(PathBuf::from("/adapters/de-distill.safetensors"));
    let err = validate_training_techniques(&plain, &on).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("training adapter")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&declaring, &on).is_ok());

    let mut full = on;
    full.config.full_finetune = true;
    let err = validate_training_techniques(&declaring, &full).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("full base fine-tune")),
        "{err:?}"
    );
}

#[test]
fn validate_training_techniques_gates_gradient_noise() {
    // sc-24827 (epic 2123 E3/E5): gradient noise is refused unless declared, refused with a
    // full fine-tune, and malformed eta/gamma are refused outright. Independent of weight noise.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut wn_only = trainer_desc_with(false, true);
    wn_only.techniques.weight_noise = true;
    let mut both = trainer_desc_with(false, true);
    both.techniques = TrainingTechniques::ADAPTER_NOISE;

    let off = train_req(None, items);
    assert_eq!(off.config.gradient_noise_eta, 0.0, "off by default (E1)");
    assert_eq!(
        off.config.gradient_noise_gamma,
        DEFAULT_GRADIENT_NOISE_GAMMA
    );
    assert!(validate_training_techniques(&plain, &off).is_ok());

    let mut on = off.clone();
    on.config.gradient_noise_eta = 0.01;
    for desc in [&plain, &wn_only] {
        let err = validate_training_techniques(desc, &on).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains("gradient noise")),
            "{err:?}"
        );
    }
    assert!(validate_training_techniques(&both, &on).is_ok());

    let mut full = on.clone();
    full.config.full_finetune = true;
    let err = validate_training_techniques(&both, &full).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("full base fine-tune")),
        "{err:?}"
    );

    for bad in [-0.01f32, f32::NAN, f32::INFINITY] {
        let mut r = off.clone();
        r.config.gradient_noise_eta = bad;
        let err = validate_training_techniques(&both, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("gradient_noise_eta")));
        let mut r = off.clone();
        r.config.gradient_noise_gamma = bad;
        let err = validate_training_techniques(&both, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("gradient_noise_gamma")));
    }
}

#[test]
fn gradient_noise_std_anneals_per_the_neelakantan_formula() {
    // sc-24827: σ_t = η / (1 + t)^γ, t the 0-based optimizer update.
    assert_eq!(gradient_noise_std(0.0, 0.55, 0), 0.0, "eta 0 is off");
    assert_eq!(gradient_noise_std(0.01, 0.55, 0), 0.01, "σ_0 = η");
    for t in [1u32, 9, 99, 999] {
        let want = 0.01 / (1.0 + t as f64).powf(0.55);
        let got = gradient_noise_std(0.01, 0.55, t) as f64;
        assert!((got - want).abs() <= want * 1e-6, "t={t}: {got} vs {want}");
        assert!(
            got < gradient_noise_std(0.01, 0.55, t - 1) as f64,
            "shrinks with t"
        );
    }
    // γ = 0 is constant noise.
    assert_eq!(gradient_noise_std(0.02, 0.0, 500), 0.02);
}

#[test]
fn technique_noise_keys_separate_streams_updates_and_tensors() {
    let k = |salt, u, i| technique_noise_key(7, salt, u, i);
    assert_eq!(k(WEIGHT_NOISE_SALT, 3, 2), k(WEIGHT_NOISE_SALT, 3, 2));
    assert_ne!(k(WEIGHT_NOISE_SALT, 3, 2), k(GRADIENT_NOISE_SALT, 3, 2));
    assert_ne!(k(WEIGHT_NOISE_SALT, 3, 2), k(WEIGHT_NOISE_SALT, 4, 2));
    assert_ne!(k(WEIGHT_NOISE_SALT, 3, 2), k(WEIGHT_NOISE_SALT, 3, 1));
    assert_ne!(
        technique_noise_key(7, WEIGHT_NOISE_SALT, 0, 0),
        technique_noise_key(8, WEIGHT_NOISE_SALT, 0, 0)
    );
}

fn rb(resolution: u32, repeats: u32) -> ResolutionBucket {
    ResolutionBucket {
        resolution,
        repeats,
    }
}

#[test]
fn validate_resolution_buckets_floor() {
    // sc-2127 (epic 2123 E3): buckets are refused unless declared; malformed lists are refused
    // regardless of support; an empty list is off everywhere.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut bucketed = trainer_desc(false);
    bucketed.techniques.resolution_buckets = true;

    let off = train_req(None, items);
    assert!(off.config.resolution_buckets.is_empty());
    assert!(validate_training_techniques(&plain, &off).is_ok());

    let mut on = off.clone();
    on.config.resolution_buckets = vec![rb(512, 16), rb(768, 4), rb(1024, 1)];
    let err = validate_training_techniques(&plain, &on).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("resolution_buckets")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&bucketed, &on).is_ok());

    let too_many: Vec<_> = (1..=MAX_RESOLUTION_BUCKETS as u32 + 1)
        .map(|i| rb(256 * i, 1))
        .collect();
    for bad in [
        vec![rb(512, 0)],
        vec![rb(0, 1)],
        vec![rb(512, 1), rb(512, 2)],
        // 520 floors to the 512 edge: it would cache 512 twice and double its weight.
        vec![rb(512, 1), rb(520, 1)],
        vec![rb(520, 1)],
        too_many,
    ] {
        let mut r = off.clone();
        r.config.resolution_buckets = bad.clone();
        let err = validate_training_techniques(&bucketed, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{bad:?}: {err:?}");
    }
    // The cap itself is accepted.
    let mut at_cap = off.clone();
    at_cap.config.resolution_buckets = (1..=MAX_RESOLUTION_BUCKETS as u32)
        .map(|i| rb(256 * i, 1))
        .collect();
    assert!(validate_training_techniques(&bucketed, &at_cap).is_ok());
}

#[test]
fn training_buckets_default_to_the_legacy_resolution() {
    let mut cfg = TrainingConfig {
        resolution: 768,
        ..TrainingConfig::default()
    };
    assert_eq!(cfg.training_buckets(), vec![rb(768, 1)]);
    assert_eq!(cfg.max_training_resolution(), 768);
    cfg.resolution_buckets = vec![rb(512, 16), rb(1024, 1), rb(768, 4)];
    assert_eq!(cfg.training_buckets(), cfg.resolution_buckets);
    assert_eq!(cfg.max_training_resolution(), 1024);
}

#[test]
fn single_bucket_schedule_is_the_legacy_round_robin() {
    // AC: a single bucket equal to today's resolution gives today's sample order — the exact
    // `cache[(step - 1) % cache.len()]` walk — whatever its repeat count.
    for repeats in [1, 3] {
        let s = BucketSchedule::new(5, &[rb(1024, repeats)], 42);
        assert_eq!(s.n_buckets(), 1);
        for k in 0..37 {
            assert_eq!(s.cache_index(k), k % 5, "repeats {repeats} k {k}");
        }
    }
}

#[test]
fn multi_bucket_schedule_mixes_16_4_1_per_image() {
    // AC: buckets 512/768/1024 with repeats 16/4/1 ⇒ each image is sampled 16:4:1 across the
    // three buckets every epoch.
    let buckets = [rb(512, 16), rb(768, 4), rb(1024, 1)];
    let n_items = 3;
    let s = BucketSchedule::new(n_items, &buckets, 7);
    assert_eq!(s.epoch_len(), n_items * 21);
    for epoch in 0..3 {
        let mut counts = vec![[0usize; 3]; n_items];
        for k in epoch * s.epoch_len()..(epoch + 1) * s.epoch_len() {
            let (item, bucket) = s.sample(k);
            counts[item][bucket] += 1;
            assert_eq!(s.cache_index(k), item * 3 + bucket);
        }
        for (item, c) in counts.iter().enumerate() {
            assert_eq!(*c, [16, 4, 1], "epoch {epoch} item {item}");
        }
    }
}

#[test]
fn multi_bucket_schedule_is_seeded_and_reshuffled_per_epoch() {
    // E4: the order is a pure function of the job seed; a different seed reorders; each epoch
    // gets its own permutation.
    let buckets = [rb(512, 2), rb(1024, 1)];
    let order = |seed: u64, range: std::ops::Range<usize>| {
        let s = BucketSchedule::new(4, &buckets, seed);
        range.map(|k| s.sample(k)).collect::<Vec<_>>()
    };
    assert_eq!(order(11, 0..24), order(11, 0..24));
    // The memoized epoch never leaks across epochs: random access matches a sequential walk.
    let s = BucketSchedule::new(4, &buckets, 11);
    let mut backwards: Vec<_> = (0..24).rev().map(|k| s.sample(k)).collect();
    backwards.reverse();
    assert_eq!(backwards, order(11, 0..24));
    assert_ne!(order(11, 0..12), order(12, 0..12));
    assert_ne!(order(11, 0..12), order(11, 12..24));
    // Not the unshuffled multiset order.
    let unshuffled: Vec<_> = (0..2)
        .flat_map(|b| (0..4).flat_map(move |i| std::iter::repeat_n((i, b), [2, 1][b])))
        .collect();
    assert_ne!(order(11, 0..12), unshuffled);
}

/// Epic 2123 E3 (feature-end review): the shared floor refuses EVERY auxiliary loss combined
/// with a full base fine-tune — even on a trainer that declares the loss and supports full
/// fine-tunes (the aux losses train the adapter surface only) — and a trainer that builds its
/// own x0 decoder (`builtin_x0_decoder`) is never asked for `perceptual_decoder_dir`.
/// Mutations: drop the full-fine-tune + aux check ⇒ the full request passes ⇒ red; drop an arm
/// from `enabled_aux_losses` ⇒ red; ignore `builtin_x0_decoder` in `lacks_x0_decoder` ⇒ red.
#[test]
fn aux_losses_refuse_full_finetune_and_a_builtin_decoder_needs_no_dir() {
    let items = vec![TrainingItem::captioned(PathBuf::from("a.png"), "a".into())];
    let mut desc = trainer_desc_with(false, true);
    desc.techniques = TrainingTechniques {
        depth_anchoring: true,
        identity_loss: true,
        face_landmark_loss: true,
        body_proportion_loss: true,
        body_shape_loss: true,
        normal_loss: true,
        vae_anchor_loss: true,
        latent_lpips_loss: true,
        ..TrainingTechniques::NONE
    };
    let dir = |p: &str| Some(PathBuf::from(p));
    type Enable = fn(&mut TrainingConfig);
    let arms: [(&str, Enable); 8] = [
        ("depth", |c| {
            c.depth_anchoring.schedule.weight = 0.1;
            c.depth_anchoring.model_dir = Some(PathBuf::from("/m/da2"));
        }),
        ("identity", |c| {
            c.identity_loss.schedule.weight = 0.1;
            c.face_analysis_dir = Some(PathBuf::from("/m/face"));
        }),
        ("face-landmark", |c| {
            c.face_landmark_loss.schedule.weight = 0.1;
            c.face_landmark_loss.model_dir = Some(PathBuf::from("/m/mesh"));
            c.face_analysis_dir = Some(PathBuf::from("/m/face"));
        }),
        ("body-proportion", |c| {
            c.body_losses.proportion.weight = 0.1;
            c.body_losses.pose_model_dir = Some(PathBuf::from("/m/vitpose"));
        }),
        ("body-shape", |c| {
            c.body_losses.shape.weight = 0.1;
            c.body_losses.pose_model_dir = Some(PathBuf::from("/m/vitpose"));
            c.body_losses.shape_model_dir = Some(PathBuf::from("/m/hybrik"));
        }),
        ("normal", |c| {
            c.body_losses.normal.weight = 0.1;
            c.body_losses.pose_model_dir = Some(PathBuf::from("/m/vitpose"));
            c.body_losses.normal_model_dir = Some(PathBuf::from("/m/sapiens"));
        }),
        ("vae_anchor", |c| {
            c.vae_anchor.schedule.weight = 0.5;
            c.vae_anchor.model_dir = Some(PathBuf::from("/m/flux2-vae"));
        }),
        ("latent_lpips", |c| {
            c.latent_lpips.schedule.weight = 0.5;
            c.latent_lpips.model_dir = Some(PathBuf::from("/m/elpips"));
        }),
    ];
    assert!(enabled_aux_losses(&TrainingConfig::default()).is_empty());
    for (name, enable) in arms {
        let mut on = train_req(None, items.clone());
        enable(&mut on.config);
        on.config.perceptual_decoder_dir = dir("/m/taef1");
        assert_eq!(enabled_aux_losses(&on.config), vec![name]);
        validate_training_techniques(&desc, &on).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut full = on.clone();
        full.config.full_finetune = true;
        let err = validate_training_techniques(&desc, &full).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m)
                    if m.contains("full base fine-tune") && m.contains(name)),
            "{name}: {err:?}"
        );
        // The decoded-x0 losses need the decoder dir — unless the trainer builds its own.
        let mut no_dec = on.clone();
        no_dec.config.perceptual_decoder_dir = None;
        if name != "latent_lpips" {
            let err = validate_training_techniques(&desc, &no_dec).unwrap_err();
            assert!(
                matches!(err, crate::Error::Msg(ref m) if m.contains("perceptual_decoder_dir")),
                "{name}: {err:?}"
            );
        }
        let mut builtin = desc;
        builtin.techniques.builtin_x0_decoder = true;
        validate_training_techniques(&builtin, &no_dec)
            .unwrap_or_else(|e| panic!("{name} (builtin decoder): {e}"));
    }
}

/// sc-24832 (epic 2123 E3): each body loss is refused unless its own flag is declared, needs
/// ViTPose (every loss's person detector), its own checkpoint and the x0 decoder, and the
/// subject-restricted normal loss needs every item's mask. Mutations: check
/// `body_proportion_loss` for the shape loss ⇒ the shape-only descriptor passes ⇒ red; drop the
/// `pose_model_dir` requirement ⇒ red.
#[test]
fn validate_training_techniques_body_loss_floor() {
    let items = vec![TrainingItem::captioned(PathBuf::from("a.png"), "a".into())];
    let off = train_req(None, items);
    assert!(!off.config.body_losses.any_enabled());
    assert!(validate_training_techniques(&trainer_desc(false), &off).is_ok());
    let full = |r: &mut TrainingRequest| {
        r.config.body_losses.pose_model_dir = Some(PathBuf::from("/m/vitpose"));
        r.config.body_losses.shape_model_dir = Some(PathBuf::from("/m/hybrik"));
        r.config.body_losses.normal_model_dir = Some(PathBuf::from("/m/sapiens"));
        r.config.perceptual_decoder_dir = Some(PathBuf::from("/m/taef1"));
    };
    type Pick = fn(&mut TrainingRequest);
    type Flag = fn(&mut TrainingTechniques);
    let cases: [(&str, Pick, Flag); 3] = [
        (
            "proportion",
            |r| r.config.body_losses.proportion.weight = 0.1,
            |t| t.body_proportion_loss = true,
        ),
        (
            "shape",
            |r| r.config.body_losses.shape.weight = 0.1,
            |t| t.body_shape_loss = true,
        ),
        (
            "normal",
            |r| r.config.body_losses.normal.weight = 0.1,
            |t| t.normal_loss = true,
        ),
    ];
    for (name, enable, declare) in cases {
        let mut on = off.clone();
        enable(&mut on);
        full(&mut on);
        // Declaring every OTHER body flag is not enough.
        let mut others = trainer_desc(false);
        for (n2, _, d2) in cases {
            if n2 != name {
                d2(&mut others.techniques);
            }
        }
        let err = validate_training_techniques(&others, &on).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains(name)),
            "{name}: {err:?}"
        );
        let mut desc = trainer_desc(false);
        declare(&mut desc.techniques);
        validate_training_techniques(&desc, &on).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut no_pose = on.clone();
        no_pose.config.body_losses.pose_model_dir = None;
        let err = validate_training_techniques(&desc, &no_pose).unwrap_err();
        assert!(
            matches!(err, crate::Error::Msg(ref m) if m.contains("ViTPose")),
            "{err:?}"
        );
        let mut no_dec = on.clone();
        no_dec.config.perceptual_decoder_dir = None;
        let err = validate_training_techniques(&desc, &no_dec).unwrap_err();
        assert!(
            matches!(err, crate::Error::Msg(ref m) if m.contains("decoder")),
            "{err:?}"
        );
    }
    let mut shape = off.clone();
    shape.config.body_losses.shape.weight = 0.1;
    full(&mut shape);
    shape.config.body_losses.shape_model_dir = None;
    let mut d = trainer_desc(false);
    d.techniques.body_shape_loss = true;
    let err = validate_training_techniques(&d, &shape).unwrap_err();
    assert!(
        matches!(err, crate::Error::Msg(ref m) if m.contains("HybrIK")),
        "{err:?}"
    );
    let mut normal = off.clone();
    normal.config.body_losses.normal.weight = 0.1;
    full(&mut normal);
    let mut d = trainer_desc(false);
    d.techniques.normal_loss = true;
    assert!(validate_training_techniques(&d, &normal).is_ok());
    // Subject-restricted normals need every item's mask (named when missing). Mutation: drop
    // the `lacking` refusal ⇒ the unmasked request passes ⇒ red.
    let mut restricted = normal.clone();
    restricted.config.body_losses.normal_restrict_to_subject = true;
    let err = validate_training_techniques(&d, &restricted).unwrap_err();
    assert!(
        matches!(err, crate::Error::Msg(ref m) if m.contains("subject mask") && m.contains("a.png")),
        "{err:?}"
    );
    restricted.items[0].subject_mask_path = Some(PathBuf::from("a.mask.png"));
    assert!(validate_training_techniques(&d, &restricted).is_ok());
    normal.config.body_losses.normal_model_dir = None;
    let err = validate_training_techniques(&d, &normal).unwrap_err();
    assert!(
        matches!(err, crate::Error::Msg(ref m) if m.contains("Sapiens")),
        "{err:?}"
    );
    // A malformed knob is a Msg regardless of support.
    let mut bad = off.clone();
    bad.config.body_losses.shape_min_cos = f32::NAN;
    let err = validate_training_techniques(&trainer_desc(false), &bad).unwrap_err();
    assert!(
        matches!(err, crate::Error::Msg(ref m) if m.contains("shape_min_cos")),
        "{err:?}"
    );
}

/// sc-2124: the decoded-x0 aux losses whose upstream window was `[0, 1]` (depth, identity,
/// landmark, body proportion) default to the latent losses' low-noise window `[0, 0.5]`; body
/// shape / normal keep upstream's `[0.4, 0.8]`; all alternate every other update. Mutation:
/// default any of them to `AuxLossSchedule::OFF` (window `[0, 1]`) ⇒ red.
#[test]
fn pixel_space_aux_losses_default_to_the_low_noise_window() {
    let c = TrainingConfig::default();
    let pixel = [
        ("depth", c.depth_anchoring.schedule),
        ("identity", c.identity_loss.schedule),
        ("landmark", c.face_landmark_loss.schedule),
        ("proportion", c.body_losses.proportion),
    ];
    for (name, s) in pixel {
        assert_eq!(
            (s.weight, s.t_min, s.t_max, s.every_n),
            (0.0, 0.0, 0.5, 2),
            "{name}"
        );
        assert!(s.validate(name).is_ok());
    }
    assert_eq!(
        PIXEL_PERCEPTUAL_SCHEDULE.t_max,
        LATENT_PERCEPTUAL_SCHEDULE.t_max
    );
    for (name, s) in [
        ("shape", c.body_losses.shape),
        ("normal", c.body_losses.normal),
    ] {
        assert_eq!(
            (s.weight, s.t_min, s.t_max, s.every_n),
            (0.0, 0.4, 0.8, 2),
            "{name}"
        );
    }
}

#[test]
fn validate_training_techniques_depth_anchoring_floor() {
    // sc-2125 (epic 2123 E3): depth anchoring is refused unless the descriptor declares it,
    // needs both aux models named, and a malformed schedule is refused outright.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut depth_desc = trainer_desc(false);
    depth_desc.techniques.depth_anchoring = true;

    // Off (the default) ⇒ no-op everywhere.
    let off = train_req(None, items);
    assert_eq!(
        off.config.depth_anchoring.schedule,
        PIXEL_PERCEPTUAL_SCHEDULE
    );
    assert!(!off.config.depth_anchoring.schedule.is_enabled());
    assert!(validate_training_techniques(&plain, &off).is_ok());

    let mut on = off.clone();
    on.config.depth_anchoring.schedule.weight = 0.1;
    on.config.depth_anchoring.model_dir = Some(PathBuf::from("/m/da2"));
    on.config.perceptual_decoder_dir = Some(PathBuf::from("/m/taef1"));
    let err = validate_training_techniques(&plain, &on).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("depth anchoring")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&depth_desc, &on).is_ok());

    // Missing aux models ⇒ a message naming the missing model.
    let mut no_da2 = on.clone();
    no_da2.config.depth_anchoring.model_dir = None;
    let err = validate_training_techniques(&depth_desc, &no_da2).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("Depth-Anything-V2")));
    let mut no_dec = on.clone();
    no_dec.config.perceptual_decoder_dir = None;
    let err = validate_training_techniques(&depth_desc, &no_dec).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("decoder")));

    // Malformed schedules ⇒ Msg, regardless of support.
    let bad = [
        AuxLossSchedule {
            weight: -0.1,
            ..AuxLossSchedule::OFF
        },
        AuxLossSchedule {
            weight: f32::NAN,
            ..AuxLossSchedule::OFF
        },
        AuxLossSchedule {
            t_min: 0.6,
            t_max: 0.4,
            ..AuxLossSchedule::OFF
        },
        AuxLossSchedule {
            t_max: 1.5,
            ..AuxLossSchedule::OFF
        },
        AuxLossSchedule {
            every_n: 0,
            ..AuxLossSchedule::OFF
        },
    ];
    for schedule in bad {
        let mut r = on.clone();
        r.config.depth_anchoring.schedule = AuxLossSchedule {
            weight: if schedule.weight == 0.0 {
                0.1
            } else {
                schedule.weight
            },
            ..schedule
        };
        let err = validate_training_techniques(&depth_desc, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{schedule:?}: {err:?}");
    }
}

/// sc-24831 (epic 2123 E3): the identity / face-landmark losses are refused unless declared,
/// need their models named, and malformed schedules / min_cos are refused outright.
/// Mutations: skip `validate_face_losses` ⇒ the Unsupported/Msg asserts go red; drop the
/// `min_cos` range check ⇒ the bad-gate loop goes red.
#[test]
fn validate_training_techniques_face_loss_floor() {
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let mut face_desc = trainer_desc(false);
    face_desc.techniques.identity_loss = true;
    face_desc.techniques.face_landmark_loss = true;

    let off = train_req(None, items);
    assert_eq!(off.config.identity_loss.schedule, PIXEL_PERCEPTUAL_SCHEDULE);
    assert_eq!(
        off.config.face_landmark_loss.schedule,
        PIXEL_PERCEPTUAL_SCHEDULE
    );
    assert_eq!(off.config.identity_loss.min_cos, 0.2);
    assert_eq!(
        off.config.identity_loss.reference_mode,
        IdentityReferenceMode::DatasetAverage
    );
    assert!(validate_training_techniques(&plain, &off).is_ok());

    let mut on = off.clone();
    on.config.identity_loss.schedule.weight = 0.1;
    on.config.face_landmark_loss.schedule.weight = 0.05;
    on.config.face_landmark_loss.model_dir = Some(PathBuf::from("/m/facemesh"));
    on.config.face_analysis_dir = Some(PathBuf::from("/m/face"));
    on.config.perceptual_decoder_dir = Some(PathBuf::from("/m/taef1"));
    assert!(validate_training_techniques(&face_desc, &on).is_ok());
    for (identity, name) in [(true, "identity loss"), (false, "face-landmark loss")] {
        let mut d = face_desc;
        if identity {
            d.techniques.identity_loss = false;
        } else {
            d.techniques.face_landmark_loss = false;
        }
        let err = validate_training_techniques(&d, &on).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(ref m) if m.contains(name)),
            "{name}: {err:?}"
        );
    }

    let mut no_face = on.clone();
    no_face.config.face_analysis_dir = None;
    let err = validate_training_techniques(&face_desc, &no_face).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("face_analysis_dir")));
    let mut no_dec = on.clone();
    no_dec.config.perceptual_decoder_dir = None;
    let err = validate_training_techniques(&face_desc, &no_dec).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("decoder")));
    let mut no_mesh = on.clone();
    no_mesh.config.face_landmark_loss.model_dir = None;
    let err = validate_training_techniques(&face_desc, &no_mesh).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("FaceMesh")));

    for bad in [f32::NAN, -1.01, 1.5] {
        let mut r = on.clone();
        r.config.identity_loss.min_cos = bad;
        let err = validate_training_techniques(&face_desc, &r).unwrap_err();
        assert!(
            matches!(err, crate::Error::Msg(ref m) if m.contains("min_cos")),
            "{bad}"
        );
    }
    let mut bad_sched = on.clone();
    bad_sched.config.face_landmark_loss.schedule.every_n = 0;
    assert!(matches!(
        validate_training_techniques(&face_desc, &bad_sched).unwrap_err(),
        crate::Error::Msg(_)
    ));
    assert_eq!(
        IdentityReferenceMode::parse("Per_Image"),
        Some(IdentityReferenceMode::PerImage)
    );
    assert_eq!(IdentityReferenceMode::parse("mean"), None);
}

#[test]
fn validate_training_techniques_latent_perceptual_floor() {
    // sc-24833 (epic 2123 E3): the VAE anchor and E-LatentLPIPS are each refused unless the
    // descriptor declares them, need their models named, and a malformed schedule is refused.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let plain = trainer_desc(false);
    let off = train_req(None, items);
    assert_eq!(off.config.vae_anchor.schedule, LATENT_PERCEPTUAL_SCHEDULE);
    assert_eq!(off.config.latent_lpips.schedule, LATENT_PERCEPTUAL_SCHEDULE);
    assert!(!off.config.vae_anchor.schedule.is_enabled());
    assert!(!off.config.latent_lpips.schedule.is_enabled());
    assert!(validate_training_techniques(&plain, &off).is_ok());

    // VAE anchor.
    let mut va_desc = trainer_desc(false);
    va_desc.techniques.vae_anchor_loss = true;
    let mut va = off.clone();
    va.config.vae_anchor.schedule.weight = 0.5;
    va.config.vae_anchor.model_dir = Some(PathBuf::from("/m/flux2-vae"));
    va.config.perceptual_decoder_dir = Some(PathBuf::from("/m/taef1"));
    let err = validate_training_techniques(&plain, &va).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("VAE anchor")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&va_desc, &va).is_ok());
    let mut no_vae = va.clone();
    no_vae.config.vae_anchor.model_dir = None;
    let err = validate_training_techniques(&va_desc, &no_vae).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("FLUX.2 VAE")));
    let mut no_dec = va.clone();
    no_dec.config.perceptual_decoder_dir = None;
    let err = validate_training_techniques(&va_desc, &no_dec).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("decoder")));
    // The VAE-anchor flag does not license E-LatentLPIPS.
    let mut lp = off.clone();
    lp.config.latent_lpips.schedule.weight = 0.5;
    lp.config.latent_lpips.model_dir = Some(PathBuf::from("/m/elatentlpips"));
    let err = validate_training_techniques(&va_desc, &lp).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("E-LatentLPIPS")),
        "{err:?}"
    );

    // E-LatentLPIPS (no decoder needed: it runs on the latent).
    let mut lp_desc = trainer_desc(false);
    lp_desc.techniques.latent_lpips_loss = true;
    assert!(validate_training_techniques(&lp_desc, &lp).is_ok());
    let mut no_w = lp.clone();
    no_w.config.latent_lpips.model_dir = None;
    let err = validate_training_techniques(&lp_desc, &no_w).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(ref m) if m.contains("weights")));
    assert!(validate_training_techniques(&lp_desc, &va).is_err());

    // Malformed schedules ⇒ Msg even on a declaring trainer.
    for bad in [
        AuxLossSchedule {
            weight: -1.0,
            ..LATENT_PERCEPTUAL_SCHEDULE
        },
        AuxLossSchedule {
            weight: 0.5,
            t_min: 0.6,
            t_max: 0.4,
            every_n: 1,
        },
        AuxLossSchedule {
            weight: 0.5,
            every_n: 0,
            ..LATENT_PERCEPTUAL_SCHEDULE
        },
    ] {
        let mut r = lp.clone();
        r.config.latent_lpips.schedule = bad;
        let err = validate_training_techniques(&lp_desc, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{bad:?}: {err:?}");
        let mut r = va.clone();
        r.config.vae_anchor.schedule = bad;
        let err = validate_training_techniques(&va_desc, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{bad:?}: {err:?}");
    }
}

#[test]
fn latent_lpips_family_table() {
    use LatentLpipsFamily::*;
    let rows: Vec<_> = [Sd15, Sd21, Sdxl, Sd3, Flux]
        .iter()
        .map(|f| (f.as_str(), f.latent_channels(), f.checkpoint_file()))
        .collect();
    assert_eq!(
        rows[2],
        (
            "sdxl",
            4,
            "elatentlpips_ckpt/sdxl_latest_vgg16_tuned.pth".into()
        )
    );
    assert_eq!(rows[3].1, 16);
    assert_eq!(
        rows[4],
        (
            "flux",
            16,
            "elatentlpips_ckpt/flux_latest_vgg16_tuned.pth".into()
        )
    );
}

#[test]
fn aux_loss_schedule_window_is_inclusive() {
    let s = AuxLossSchedule {
        weight: 1.0,
        t_min: 0.2,
        t_max: 0.8,
        every_n: 2,
    };
    assert!(s.in_window(0.2) && s.in_window(0.8) && s.in_window(0.5));
    assert!(!s.in_window(0.19) && !s.in_window(0.81));
}

#[test]
fn depth_model_size_round_trips() {
    for size in [
        DepthModelSize::Small,
        DepthModelSize::Base,
        DepthModelSize::Large,
    ] {
        assert_eq!(DepthModelSize::parse(size.as_str()), Some(size));
    }
    assert_eq!(
        DepthModelSize::parse(" LARGE "),
        Some(DepthModelSize::Large)
    );
    assert_eq!(DepthModelSize::parse("giant"), None);
    assert_eq!(DepthModelSize::default(), DepthModelSize::Small);
}

#[test]
fn validate_training_techniques_subject_mask_loss_floor() {
    // sc-24828 (epic 2123 E3): masked loss is refused unless declared, with malformed weights,
    // and when any item lacks a mask (naming it); off is a no-op everywhere.
    let mut masked = TrainingItem::captioned(PathBuf::from("a.png"), "a cat".into());
    masked.subject_mask_path = Some(PathBuf::from("masks/a.png"));
    let bare = TrainingItem::captioned(PathBuf::from("b.png"), "a dog".into());
    let plain = trainer_desc(false);
    let mut desc = trainer_desc(false);
    desc.techniques.subject_mask_loss = true;

    let off = train_req(None, vec![masked.clone(), bare.clone()]);
    assert_eq!(off.config.subject_mask_loss, None);
    assert!(validate_training_techniques(&plain, &off).is_ok());
    assert!(validate_training_techniques(&desc, &off).is_ok());

    let mut on = train_req(None, vec![masked.clone()]);
    on.config.subject_mask_loss = Some(SubjectMaskLoss {
        background_weight: 0.0,
        subject_weight: 1.0,
    });
    let err = validate_training_techniques(&plain, &on).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(ref m) if m.contains("subject-masked loss")),
        "{err:?}"
    );
    assert!(validate_training_techniques(&desc, &on).is_ok());

    let mut missing = on.clone();
    missing.items.push(bare);
    let err = validate_training_techniques(&desc, &missing)
        .unwrap_err()
        .to_string();
    assert!(err.contains("1 of 2 have none: b.png"), "{err}");

    for (bg, subj) in [
        (-0.1, 1.0),
        (1.1, 1.0),
        (f32::NAN, 1.0),
        (0.0, 0.0),
        (0.0, 1.5),
        (0.0, f32::INFINITY),
    ] {
        let mut r = on.clone();
        r.config.subject_mask_loss = Some(SubjectMaskLoss {
            background_weight: bg,
            subject_weight: subj,
        });
        let err = validate_training_techniques(&desc, &r).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)), "{bg}/{subj}: {err:?}");
    }
}

/// sc-27282: the shared floor refuses an output name that could escape `output_dir` (or is not
/// a `.safetensors` adapter) for every trainer, not only flux3.
#[test]
fn training_floor_refuses_unsafe_output_file_names() {
    assert_eq!(output_stem("my_style.safetensors").unwrap(), "my_style");
    assert_eq!(output_stem("a.b.safetensors").unwrap(), "a.b");
    for bad in [
        "",
        ".safetensors",
        "adapter",
        "adapter.ckpt",
        "../escape.safetensors",
        "..",
        ".",
        "sub/adapter.safetensors",
        "sub\\adapter.safetensors",
        "/abs/adapter.safetensors",
        "C:adapter.safetensors",
        "nul\0.safetensors",
        "line\nbreak.safetensors",
    ] {
        assert!(
            matches!(output_stem(bad), Err(crate::Error::Unsupported(_))),
            "{bad:?} must be refused"
        );
    }
    let desc = trainer_desc(false);
    let mut req = TrainingRequest {
        items: vec![],
        output_dir: PathBuf::from("unused"),
        file_name: "ok.safetensors".into(),
        trigger_words: vec![],
        config: TrainingConfig::default(),
        cancel: crate::CancelFlag::new(),
    };
    validate_training_techniques(&desc, &req).unwrap();
    req.file_name = "../../etc/evil.safetensors".into();
    assert!(matches!(
        validate_training_techniques(&desc, &req),
        Err(crate::Error::Unsupported(message)) if message.contains("single safe file name")
    ));
}

#[test]
fn sd3_refuses_unimplemented_caller_byte_cap_before_preflight() {
    let mut desc = trainer_desc(false);
    desc.family = "sd3";
    let mut req = TrainingRequest {
        items: vec![],
        output_dir: PathBuf::from("unused"),
        file_name: "unused.safetensors".into(),
        trigger_words: vec![],
        config: TrainingConfig::default(),
        cancel: crate::CancelFlag::new(),
    };
    validate_training_techniques(&desc, &req).unwrap();
    for bytes in [1, u64::MAX] {
        req.config.memory_limit_bytes = Some(bytes);
        assert!(matches!(validate_training_techniques(&desc, &req),
                Err(crate::Error::Unsupported(message)) if message.contains("byte cap is not routed")));
    }
    req.config.memory_limit_bytes = None;
    req.config.memory_limit_fraction = Some(0.5);
    assert!(matches!(validate_training_techniques(&desc, &req),
            Err(crate::Error::Unsupported(message)) if message.contains("memory fraction is not routed")));
    desc.family = "moss_image";
    req.config.memory_limit_bytes = Some(1);
    validate_training_techniques(&desc, &req).unwrap();
}

fn trainer_desc(supports_control: bool) -> TrainerDescriptor {
    trainer_desc_with(supports_control, false)
}

fn trainer_desc_with(supports_control: bool, supports_full_finetune: bool) -> TrainerDescriptor {
    TrainerDescriptor {
        id: "fam_trainer",
        family: "fam",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control,
        supports_full_finetune,
        max_reference_images: 0,
        techniques: TrainingTechniques::NONE,
    }
}

fn train_req(control_type: Option<&str>, items: Vec<TrainingItem>) -> TrainingRequest {
    TrainingRequest {
        items,
        config: TrainingConfig {
            control_type: control_type.map(str::to_owned),
            ..Default::default()
        },
        output_dir: PathBuf::from("/out"),
        file_name: "a.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    }
}

#[test]
fn validate_control_request_floor() {
    // F-006: the shared control-training floor.
    let img = PathBuf::from("a.png");
    let ctrl = PathBuf::from("a.pose.png");

    // control_type unset ⇒ no-op on both trainer kinds.
    let lora_items = vec![TrainingItem::captioned(img.clone(), "a cat".into())];
    assert!(
        validate_control_request(&trainer_desc(false), &train_req(None, lora_items.clone()))
            .is_ok()
    );

    // control_type set on a NON-control trainer ⇒ typed Unsupported (must reject, not silently
    // train a plain adapter — the F-055 class).
    let err = validate_control_request(
        &trainer_desc(false),
        &train_req(Some("pose"), lora_items.clone()),
    )
    .unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(_)),
        "control on a LoRA-only trainer is a capability gap → Unsupported, got {err:?}"
    );

    // control_type set on a control-capable trainer, but an item lacks a control image ⇒ Msg.
    let err = validate_control_request(&trainer_desc(true), &train_req(Some("pose"), lora_items))
        .unwrap_err();
    assert!(
        matches!(&err, crate::Error::Msg(_)) && err.to_string().contains("control image"),
        "got {err:?}"
    );

    // control_type set on a control-capable trainer with every item carrying a control image ⇒ ok.
    let ctrl_items = vec![TrainingItem::with_control(img, "a cat".into(), ctrl)];
    assert!(
        validate_control_request(&trainer_desc(true), &train_req(Some("pose"), ctrl_items)).is_ok()
    );
}

#[test]
fn validate_full_finetune_request_floor() {
    // F-006 (sc-14056): the shared full-base-fine-tune floor, mirroring the control floor above.
    let items = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    let full_req = |desc_items: Vec<TrainingItem>| {
        let mut req = train_req(None, desc_items);
        req.config.full_finetune = true;
        req
    };

    // full_finetune unset ⇒ no-op on BOTH trainer kinds (additive; the whole existing surface).
    for supports_full in [false, true] {
        assert!(
            validate_full_finetune_request(
                &trainer_desc_with(false, supports_full),
                &train_req(None, items.clone())
            )
            .is_ok(),
            "a non-full request must pass regardless of supports_full_finetune"
        );
    }

    // full_finetune set on a trainer that does NOT advertise it ⇒ typed Unsupported. It must
    // REJECT, not fall through and silently train a LoRA adapter — the F-055 class.
    let err =
        validate_full_finetune_request(&trainer_desc_with(false, false), &full_req(items.clone()))
            .unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(_)),
        "a full fine-tune on an adapter-only trainer is a capability gap → Unsupported, got \
             {err:?}"
    );
    assert!(
        err.to_string().contains("full_finetune"),
        "the rejection must name the unsupported capability, got {err}"
    );

    // full_finetune set on a full-tune-capable trainer ⇒ ok. This is the half that proves the
    // floor reads the descriptor flag rather than rejecting every full request outright.
    assert!(
        validate_full_finetune_request(&trainer_desc_with(false, true), &full_req(items)).is_ok(),
        "a trainer advertising supports_full_finetune must be allowed through the floor"
    );
}

fn edit_desc(max_reference_images: u32) -> TrainerDescriptor {
    TrainerDescriptor {
        max_reference_images,
        ..trainer_desc(false)
    }
}

/// sc-24161: the edit-pair item shape keeps its references in the order given, is
/// distinguishable from the captioned/control shapes, and the older constructors (and
/// `Default`) carry no references — so every pre-existing caller is unaffected.
#[test]
fn edit_pair_keeps_its_references_in_order_and_older_shapes_carry_none() {
    let refs: Vec<PathBuf> = ["c.png", "a.png", "b.png"]
        .iter()
        .map(PathBuf::from)
        .collect();
    let item = TrainingItem::edit_pair(
        PathBuf::from("target.png"),
        "put the cat from the second image on the sofa of the first".into(),
        refs.clone(),
    );
    assert_eq!(item.image_path, PathBuf::from("target.png"));
    assert_eq!(
        item.caption,
        "put the cat from the second image on the sofa of the first"
    );
    assert_eq!(item.reference_image_paths, refs, "order is semantic");
    assert!(item.is_edit_pair());
    assert_eq!(item.control_image_path, None);

    // Round trip through a request (the shape the worker hands every trainer) and a clone:
    // the order survives untouched.
    let req = train_req(None, vec![item.clone()]);
    assert_eq!(req.clone().items[0].reference_image_paths, refs);
    assert_eq!(req.items[0], item);

    let captioned = TrainingItem::captioned(PathBuf::from("a.png"), "a cat".into());
    let control = TrainingItem::with_control(
        PathBuf::from("a.png"),
        "a cat".into(),
        PathBuf::from("a.pose.png"),
    );
    for older in [&captioned, &control, &TrainingItem::default()] {
        assert!(older.reference_image_paths.is_empty());
        assert!(!older.is_edit_pair());
    }
}

/// sc-24161: a trainer that cannot use references (`max_reference_images == 0`, every trainer
/// shipped before the edit trainer) refuses an edit dataset with a typed `Unsupported` — never
/// a silently trained text-to-image adapter — and passes a captioned dataset untouched.
#[test]
fn a_non_edit_trainer_refuses_an_edit_dataset() {
    let edit = vec![TrainingItem::edit_pair(
        PathBuf::from("t.png"),
        "make it blue".into(),
        vec![PathBuf::from("r.png")],
    )];
    let err = validate_edit_request(&edit_desc(0), &train_req(None, edit)).unwrap_err();
    assert!(
        matches!(err, crate::Error::Unsupported(_)),
        "an edit dataset on a non-edit trainer is a capability gap → Unsupported, got {err:?}"
    );
    assert!(err.to_string().contains("instruction-edit"), "{err}");

    let captioned = vec![TrainingItem::captioned(
        PathBuf::from("a.png"),
        "a cat".into(),
    )];
    for cap in [0, 10] {
        assert!(
            validate_edit_request(&edit_desc(cap), &train_req(None, captioned.clone())).is_ok(),
            "a captioned dataset is the floor's no-op (cap {cap})"
        );
    }
}

/// sc-24161: the reference cap. Exactly `max_reference_images` passes; one more is refused
/// with a message naming the cap and the offending item.
#[test]
fn the_reference_cap_is_enforced_and_named() {
    let refs =
        |n: usize| -> Vec<PathBuf> { (0..n).map(|i| PathBuf::from(format!("r{i}.png"))).collect() };
    let item =
        |n: usize| TrainingItem::edit_pair(PathBuf::from("t.png"), "compose".into(), refs(n));
    let desc = edit_desc(10);
    assert!(validate_edit_request(&desc, &train_req(None, vec![item(1)])).is_ok());
    assert!(validate_edit_request(&desc, &train_req(None, vec![item(10)])).is_ok());
    let err = validate_edit_request(&desc, &train_req(None, vec![item(1), item(11)])).unwrap_err();
    assert!(matches!(err, crate::Error::Msg(_)), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("at most 10"), "{msg}");
    assert!(msg.contains("item 1 carries 11"), "{msg}");
}

/// sc-24161: an edit run is all-edit — a reference-less item in it, or an item that is both an
/// edit pair and a control pair, is refused by name.
#[test]
fn mixed_and_hybrid_edit_datasets_are_refused() {
    let edit = TrainingItem::edit_pair(
        PathBuf::from("t.png"),
        "make it blue".into(),
        vec![PathBuf::from("r.png")],
    );
    let plain = TrainingItem::captioned(PathBuf::from("a.png"), "a cat".into());
    let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![edit.clone(), plain]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("item 1 has none"), "{err}");

    let mut hybrid = edit;
    hybrid.control_image_path = Some(PathBuf::from("pose.png"));
    let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![hybrid]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("one or the other"), "{err}");
}

/// sc-24161: the item-shape floor that makes `TrainingItem: Default` safe — a forgotten
/// (empty) `image_path` or reference path is refused by name on EVERY trainer (cap 0 or not),
/// and an edit pair needs a non-empty instruction. A captioned item may keep an empty caption
/// (trigger-word-only datasets), so that is NOT refused.
#[test]
fn empty_paths_and_empty_edit_instructions_are_refused() {
    for cap in [0, 10] {
        let forgotten = TrainingItem {
            caption: "a cat".into(),
            ..Default::default()
        };
        let err = validate_edit_request(&edit_desc(cap), &train_req(None, vec![forgotten]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("item 0 has an empty image_path"),
            "cap {cap}: {err}"
        );
    }

    let empty_ref = TrainingItem::edit_pair(
        PathBuf::from("t.png"),
        "make it blue".into(),
        vec![PathBuf::from("r.png"), PathBuf::new()],
    );
    let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![empty_ref]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty path at reference 1"), "{err}");

    let silent = TrainingItem::edit_pair(
        PathBuf::from("t.png"),
        "  ".into(),
        vec![PathBuf::from("r.png")],
    );
    let err = validate_edit_request(&edit_desc(10), &train_req(None, vec![silent]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty instruction"), "{err}");

    let trigger_only = TrainingItem::captioned(PathBuf::from("a.png"), String::new());
    assert!(
        validate_edit_request(&edit_desc(0), &train_req(None, vec![trigger_only])).is_ok(),
        "an empty caption on a captioned item stays legal"
    );
}

fn plain_request() -> TrainingRequest {
    TrainingRequest {
        items: vec![TrainingItem::captioned(
            PathBuf::from("a.png"),
            "a cat".into(),
        )],
        config: TrainingConfig::default(),
        output_dir: PathBuf::from("out"),
        file_name: "out.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    }
}

/// *Mutation that reds this:* `model_option_selects_something` treating every non-null value as
/// present, or dropping a key from [`REFERENCE_CONTROL_MODEL_OPTIONS`].
#[test]
fn reference_control_model_options_refuse_on_values_and_ignore_off_values() {
    let mut req = plain_request();
    let off = serde_json::json!({
        "mixedPrecision": "bf16",
        "references": [],
        "referenceImages": "",
        "reference_images": "  ",
        "referenceImagePaths": {},
        "controlType": null,
        "control_type": "none",
        "controlImage": false,
        "control_image": "None",
    });
    req.config.model_options = off.as_object().unwrap().clone();
    req.items[0].model_options = off.as_object().unwrap().clone();
    refuse_reference_control_model_options("t", &req).unwrap();
    for key in REFERENCE_CONTROL_MODEL_OPTIONS {
        let mut on = req.clone();
        on.items[0]
            .model_options
            .insert(key.into(), serde_json::json!("x"));
        let err = refuse_reference_control_model_options("t", &on)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("t: model_options `"), "{err}");
        assert!(err.contains(&format!("`{key}`")), "{err}");
    }
}

/// *Mutation that reds this:* dropping any arm of the `||` in `refuse_trainer_load_overlays`.
#[test]
fn trainer_load_overlays_are_typed_refusals() {
    use crate::runtime::{IdentityWeights, LoadSpec, WeightsSource};

    let dir = || WeightsSource::Dir(PathBuf::from("/x"));
    let base = LoadSpec::new(dir());
    refuse_trainer_load_overlays("t", &base).unwrap();
    let mut identity = base.clone();
    identity.identity = Some(IdentityWeights::default());
    for spec in [
        base.clone().with_control(dir()),
        base.clone().with_extra_control(dir()),
        base.clone().with_ip_adapter(dir()),
        identity,
    ] {
        match refuse_trainer_load_overlays("t", &spec) {
            Err(crate::Error::Unsupported(message)) => assert_eq!(
                message,
                "t: control / IP-adapter / identity overlays are not part of text-to-image \
                     LoRA/LoKr training"
            ),
            other => panic!("an overlay must be a typed Unsupported, got {other:?}"),
        }
    }
}
