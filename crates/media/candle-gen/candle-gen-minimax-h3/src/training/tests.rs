//! Weights-free checks of the candle trainer's bookkeeping: no tensor is read and no model runs.

use super::*;

fn cfg() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig::image()
}

/// ai-toolkit's surface: every DiT-block and token-refiner-block leaf, so a default run refines
/// the caption every step.
#[test]
fn the_default_targets_are_every_dit_and_refiner_block_leaf() {
    let paths = resolve_target_paths(&cfg(), &TrainingConfig::default()).unwrap();
    assert_eq!(paths.len(), (50 + 2) * 6);
    assert_eq!(
        paths.iter().filter(|p| target_block(p).is_some()).count(),
        300
    );
    assert_eq!(refiner_layers(&paths), 2);
    assert!(trains_refiner(&paths));
    for p in &paths {
        leaf_shape(&cfg(), p).unwrap();
    }
}

/// The trainer resolves its targets over exactly the surface the SW-7096 install addresses, so
/// every trained key is one the loader resolves.
#[test]
fn the_trainable_surface_is_the_loaders_surface() {
    let c = cfg();
    let surface = adapter_surface(
        c.num_layers,
        c.num_refiner_layers,
        c.num_pdd_heads().unwrap(),
    );
    assert_eq!(surface, crate::adapters::image_adapter_target_paths(&c));
}

/// A module the run computes once per request is a typed refusal, never a smaller adapter.
#[test]
fn an_untrainable_target_is_refused() {
    let err = resolve_target_paths(
        &cfg(),
        &TrainingConfig {
            lora_target_modules: vec!["adaln_proj.linear".into()],
            ..TrainingConfig::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("precomputed"), "{err}");
}

#[test]
fn the_descriptor_is_the_shared_one_on_candle() {
    let d = descriptor();
    let shared = trainer_descriptor("mlx");
    assert_eq!((d.id, d.backend), (shared.id, "candle"));
    assert_eq!(d.techniques, shared.techniques);
    assert_eq!(d.max_reference_images, shared.max_reference_images);
    assert_eq!(d.id, crate::image_model::IMAGE_MODEL_ID);
}

/// The shared request floor runs on this backend's descriptor: a control image and a technique
/// the trainer does not declare are refused before any weight is read.
#[test]
fn the_request_floor_refuses_control_images_and_undeclared_techniques() {
    let d = descriptor();
    let mut item = TrainingItem::captioned("/nonexistent/a.png".into(), "a cat".into());
    item.control_image_path = Some("/nonexistent/c.png".into());
    let req = |items: Vec<TrainingItem>, config: TrainingConfig| TrainingRequest {
        items,
        config,
        output_dir: "/nonexistent".into(),
        file_name: "x.safetensors".into(),
        trigger_words: vec![],
        cancel: Default::default(),
    };
    assert!(gen_core_train::h3_image::validate_floors(
        &d,
        &req(vec![item], TrainingConfig::default())
    )
    .is_err());
    let plain = TrainingItem::captioned("/nonexistent/a.png".into(), "a cat".into());
    let config = TrainingConfig {
        subject_mask_loss: Some(candle_gen::gen_core::train::SubjectMaskLoss {
            background_weight: 0.5,
            subject_weight: 1.0,
        }),
        ..TrainingConfig::default()
    };
    assert!(gen_core_train::h3_image::validate_floors(&d, &req(vec![plain], config)).is_err());
}

/// The trainable element count is exact for the default rank-16 LoRA, and the LoKr count follows
/// PEFT's `use_w2` rule per leaf.
#[test]
fn adapter_elements_count_what_the_install_trains() {
    let c = cfg();
    let paths = resolve_target_paths(&c, &TrainingConfig::default()).unwrap();
    let lora = TrainingConfig {
        rank: 16,
        ..TrainingConfig::default()
    };
    let e = adapter_elements(&c, &paths, &lora).unwrap();
    // 50 DiT blocks and 2 refiner blocks, the same six leaves at the same widths.
    assert_eq!(
        e.trainable,
        52 * 16 * (3 * (5376 + 7168) + (7168 + 5376) + (5376 + 2 * 14336) + (14336 + 5376))
    );
    assert_eq!((e.lokr_block_max, e.lokr_total, e.lokr_refiner), (0, 0, 0));
    let lokr = TrainingConfig {
        rank: 16,
        network_type: NetworkType::Lokr,
        ..TrainingConfig::default()
    };
    let e = adapter_elements(&c, &paths, &lokr).unwrap();
    let (block_max, total) = (e.lokr_block_max, e.lokr_total);
    assert!(e.trainable > 0);
    // Per row, each leaf's vec-trick intermediate `in_a · out_b` — never the dense `[out, in]`.
    let row = |out_f: usize, in_f: usize| -> u64 {
        let (_, out_b) = factorization(out_f, -1);
        let (in_a, _) = factorization(in_f, -1);
        (in_a * out_b) as u64
    };
    let block = 3 * row(7168, 5376) + row(5376, 7168) + row(2 * 14336, 5376) + row(5376, 14336);
    // The refiner's intermediates are per caption row, counted apart from the DiT blocks' maximum
    // and inside the total.
    assert_eq!(block_max, block);
    assert_eq!(e.lokr_refiner, 2 * block);
    assert_eq!(total, 52 * block);
    assert!(block < 7168 * 5376);
    assert_eq!(
        leaf_shape(&c, "transformer_blocks.3.ff.net.0.proj").unwrap(),
        (2 * 14336, 5376)
    );
    assert_eq!(
        leaf_shape(&c, "token_refiner.refiner_blocks.1.attn.to_out.0").unwrap(),
        (5376, 7168)
    );
    assert!(leaf_shape(&c, "proj_in").is_err());
    // A block-only target list leaves the refiner constant: nothing refined per step.
    let blocks_only: Vec<String> = paths
        .iter()
        .filter(|p| target_block(p).is_some())
        .cloned()
        .collect();
    assert_eq!(refiner_layers(&blocks_only), 0);
    assert_eq!(
        adapter_elements(&c, &blocks_only, &lokr)
            .unwrap()
            .lokr_refiner,
        0
    );
}

/// The six heads a step trains through are the render's six steps: the published sigma grid and
/// the render schedule's timesteps agree index for index.
#[test]
fn every_training_step_is_a_render_step() {
    let schedule = gen_core_h3::h3_schedule::SigmaSchedule::h3_image_distilled().unwrap();
    for (i, &t) in schedule.timesteps().iter().enumerate() {
        let st = gen_core_train::h3_image::distilled_step(i);
        assert_eq!(st.index, i);
        assert!(
            (st.timestep - t).abs() < 1e-6,
            "step {i}: {} vs {t}",
            st.timestep
        );
    }
}

/// The trainer's own timestep draw (`flow_match::sample_unit_timestep` at the per-step seed, the
/// exact path `micro_step` takes) trains every one of the six heads on a real share of the steps
/// under each distribution at the balanced bias — the six-step render runs each head once, and the
/// LoRA in the shared blocks changes all of them. Nearest-σ quantization gave heads 0 and 1 about
/// 0.002 % and 0.5 % of a sigmoid run. Floors: `linear` is the render's own 1/6 per head;
/// `sigmoid` puts `P(z > ln 5) ≈ 5.4 %` on each end head; `weighted` (the mean of a uniform and a
/// sigmoid draw) concentrates on mid noise by design, leaving its end heads ≈ 2.6 % each.
#[test]
fn every_head_is_trained_under_every_timestep_distribution() {
    let steps = 20_000u32;
    for (kind, floor) in [("sigmoid", 0.045), ("linear", 0.155), ("weighted", 0.02)] {
        let mut counts = [0u32; 6];
        for step in 1..=steps {
            let t = flow_match::sample_unit_timestep(
                kind,
                "balanced",
                flow_match::timestep_seed(42, step),
            );
            counts[training_step(t).index] += 1;
        }
        for (head, &n) in counts.iter().enumerate() {
            let share = f64::from(n) / f64::from(steps);
            assert!(
                share > floor,
                "{kind}: head {head} trained {share:.4} of steps, floor {floor} ({counts:?})"
            );
        }
    }
}

/// A resume continues only the mode it was written in.
#[test]
fn a_resume_refuses_the_other_training_mode() {
    assert!(check_resumed_mode(None, false).is_ok());
    assert!(check_resumed_mode(Some("edit"), true).is_ok());
    assert!(check_resumed_mode(Some("edit"), false).is_err());
    assert!(check_resumed_mode(None, true).is_err());
}

/// The reference-augmentation seed is a pure function of (seed, step): a resumed run's step `k`
/// draws what an uninterrupted run's step `k` drew, and neighbouring steps differ.
#[test]
fn the_step_seed_depends_only_on_the_run_seed_and_the_step() {
    assert_eq!(step_seed(7, 12), step_seed(7, 12));
    assert_ne!(step_seed(7, 12), step_seed(7, 13));
    assert_ne!(step_seed(7, 12), step_seed(8, 12));
}

#[test]
fn optimizer_state_follows_the_optimizer() {
    assert_eq!(optimizer_state_per_element("adamw"), 2);
    assert_eq!(optimizer_state_per_element("Prodigy-Opt"), 4);
    assert_eq!(optimizer_state_per_element("rose"), 0);
}

/// Every saved adapter carries the shared provenance, and an edit adapter the edit marker.
#[test]
fn saved_adapters_carry_the_provenance() {
    let t2i = provenance(false);
    for (k, v) in ADAPTER_PROVENANCE {
        assert_eq!(t2i.get(k).map(String::as_str), Some(v));
    }
    assert!(!t2i.contains_key(EDIT_ADAPTER_MARKER.0));
    assert_eq!(
        provenance(true)
            .get(EDIT_ADAPTER_MARKER.0)
            .map(String::as_str),
        Some(EDIT_ADAPTER_MARKER.1)
    );
}

/// A preview is the generator's own request: the training canvas, the seed, the references as
/// plain references, and both transformers streamed one block at a time.
#[test]
fn the_preview_request_streams_both_transformers() {
    let image = Image {
        width: 4,
        height: 4,
        pixels: vec![0; 48],
    };
    let req = preview_request("a cat", 9, (2048, 1536), &[image], CancelFlag::new());
    assert_eq!(
        (req.width, req.height, req.count, req.seed),
        (2048, 1536, 1, Some(9))
    );
    assert_eq!(req.conditioning.len(), 1);
    let memory = req.memory.unwrap();
    assert!(memory.stream_transformer_blocks && memory.stage_residency);
    assert_eq!(memory.transformer_window_size, Some(1));
    assert_eq!(
        memory.transformer_window_component,
        Some(TransformerComponent::Both)
    );
}

/// The crate's registry carries the trainer at the generator's id.
#[test]
fn the_registry_registers_the_trainer() {
    let registry = crate::provider_registry().unwrap();
    let ids: Vec<&str> = registry.trainers().map(|t| (t.descriptor)().id).collect();
    assert_eq!(ids, vec![TRAINER_ID]);
}

/// The training adapter is a declared technique on candle: the shared floor admits a run naming one
/// (validate never reads the file), and refuses it on a full fine-tune.
#[test]
fn the_training_adapter_is_declared_and_passes_the_floor() {
    let d = descriptor();
    assert!(d.techniques.training_adapter);
    let req = |config: TrainingConfig| TrainingRequest {
        items: vec![TrainingItem::captioned(
            "/nonexistent/a.png".into(),
            "a cat".into(),
        )],
        config,
        output_dir: "/nonexistent".into(),
        file_name: "x.safetensors".into(),
        trigger_words: vec![],
        cancel: Default::default(),
    };
    let with_adapter = TrainingConfig {
        training_adapter: Some("/nonexistent/fizgig.safetensors".into()),
        ..gen_core_train::h3_image::default_training_config()
    };
    gen_core_train::h3_image::validate_floors(&d, &req(with_adapter.clone())).unwrap();
    let full = TrainingConfig {
        full_finetune: true,
        ..with_adapter
    };
    assert!(gen_core_train::h3_image::validate_floors(&d, &req(full)).is_err());
}

/// No adapter configured prices nothing and reads nothing (the path would not exist).
#[test]
fn no_training_adapter_prices_nothing() {
    let bytes = training_adapter::training_adapter_bytes(
        Path::new("/nonexistent-dit"),
        &cfg(),
        &[],
        None,
        training_adapter::TRAINING_ADAPTER_STRENGTH,
    )
    .unwrap();
    assert_eq!(
        bytes,
        gen_core_train::h3_image::TrainingAdapterBytes::default()
    );
}

/// A preview loads the trainer's snapshot with the in-progress factors as its only adapter — the
/// frozen training adapter never reaches a preview render, so the samples show what the saved
/// adapter renders on the bare distilled base. Mutation: append the training adapter to the
/// preview load ⇒ the adapter-list assert fails.
#[test]
fn a_preview_loads_only_the_trained_adapter() {
    let setup = PreviewSetup {
        spec: LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            "/nonexistent-minimax-h3-image".into(),
        )),
        prompts: vec!["a cat".into()],
        size: (1024, 1024),
        references: vec![],
        adapter_kind: AdapterKind::Lora,
    };
    let preview = Path::new("/out/.run-preview.safetensors");
    let spec = setup.generator_spec(preview);
    assert_eq!(spec.adapters.len(), 1);
    assert_eq!(spec.adapters[0].path, preview);
    assert_eq!(spec.adapters[0].scale, 1.0);
}
