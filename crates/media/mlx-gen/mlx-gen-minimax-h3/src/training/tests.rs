//! Weights-free checks of the MLX trainer's bookkeeping: no tensor is read and no model runs.

use super::*;

#[test]
fn the_default_targets_are_every_block_leaf() {
    let paths = resolve_target_paths(&TrainingConfig::default()).unwrap();
    assert_eq!(paths.len(), 50 * 6);
    assert!(paths.iter().all(|p| target_block(p).is_some()));
}

#[test]
fn the_descriptor_is_the_shared_one_on_mlx() {
    let d = descriptor();
    let shared = trainer_descriptor("mlx");
    assert_eq!((d.id, d.backend), (shared.id, "mlx"));
    assert_eq!(d.max_reference_images, shared.max_reference_images);
    assert_eq!(d.id, crate::h3_image::IMAGE_MODEL_ID);
    assert_eq!(d.family, crate::h3_image::model::FAMILY);
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

/// The image registry carries the trainer at the generator's id.
#[test]
fn the_image_registry_registers_the_trainer() {
    let registry =
        crate::h3_image::register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new())
            .build()
            .unwrap();
    let ids: Vec<&str> = registry.trainers().map(|t| (t.descriptor)().id).collect();
    assert_eq!(ids, vec![TRAINER_ID]);
}

/// The training adapter is a declared technique on MLX: the shared floor admits a run naming one
/// (validate never reads the file).
#[test]
fn the_training_adapter_is_declared_and_passes_the_floor() {
    let d = descriptor();
    assert!(d.techniques.training_adapter);
    let req = TrainingRequest {
        items: vec![TrainingItem::captioned(
            "/nonexistent/a.png".into(),
            "a cat".into(),
        )],
        config: TrainingConfig {
            training_adapter: Some("/nonexistent/fizgig.safetensors".into()),
            ..gen_core_train::h3_image::default_training_config()
        },
        output_dir: "/nonexistent".into(),
        file_name: "x.safetensors".into(),
        trigger_words: vec![],
        cancel: Default::default(),
    };
    gen_core_train::h3_image::validate_floors(&d, &req).unwrap();
}

/// A host of plain dense linears keyed by dotted path.
struct Leaves(std::collections::BTreeMap<String, mlx_gen::adapters::AdaptableLinear>);

impl AdaptableHost for Leaves {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut mlx_gen::adapters::AdaptableLinear> {
        self.0.get_mut(&path.join("."))
    }
}

/// The trainable install replaces a target leaf's stack; the frozen training adapter is appended
/// back on exactly the leaves of the block being installed (or of every block), once per install —
/// so the leaf computes base + trainable + frozen and never accumulates a second frozen copy.
/// Mutations: skip the re-attach ⇒ the stack-length and forward asserts fail; ignore the block
/// filter ⇒ block 1's stack assert fails.
#[test]
fn the_frozen_adapter_is_reattached_after_each_trainable_install() {
    use mlx_gen::adapters::{AdaptableLinear, Adapter};
    use mlx_rs::ops::{add, matmul, subtract};
    let q0 = ["transformer_blocks", "0", "attn", "to_q"];
    let q1 = ["transformer_blocks", "1", "attn", "to_q"];
    let weight = Array::from_slice(&[0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5], &[3, 2]);
    let mut host = Leaves(
        [q0, q1]
            .iter()
            .map(|p| (p.join("."), AdaptableLinear::dense(weight.clone(), None)))
            .collect(),
    );
    let lora = |a: &[f32], b: &[f32]| Adapter::Lora {
        a: Array::from_slice(a, &[2, 1]),
        b: Array::from_slice(b, &[1, 3]),
        scale: 1.0,
    };
    let frozen = lora(&[1.0, 2.0], &[0.5, -0.5, 1.0]);
    let trainable = lora(&[-1.0, 0.5], &[2.0, 1.0, -1.0]);
    let residuals = super::training_adapter::FrozenResiduals::from_paths(
        [q0, q1]
            .iter()
            .map(|p| (p.join("."), vec![frozen.clone()]))
            .collect(),
    );
    // What a window rebuild leaves on block 1: the frozen residual alone.
    host.adaptable_mut(&q1).unwrap().push(frozen.clone());
    for _ in 0..2 {
        // The trainable install's replacement on block 0, then block 0's re-attach.
        host.adaptable_mut(&q0)
            .unwrap()
            .set_training_adapters(vec![trainable.clone()]);
        residuals.reattach(&mut host, Some(0)).unwrap();
        assert_eq!(host.adaptable_mut(&q0).unwrap().adapters().len(), 2);
        assert_eq!(host.adaptable_mut(&q1).unwrap().adapters().len(), 1);
    }
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5, 3.0], &[2, 2]);
    let residual = |adapter: &Adapter| match adapter {
        Adapter::Lora { a, b, .. } => matmul(&matmul(&x, a).unwrap(), b).unwrap(),
        _ => unreachable!(),
    };
    let want = add(
        &add(&matmul(&x, weight.t()).unwrap(), &residual(&trainable)).unwrap(),
        &residual(&frozen),
    )
    .unwrap();
    let got = host.adaptable_mut(&q0).unwrap().forward(&x).unwrap();
    let diff = subtract(&got, &want)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(diff < 1e-5, "{diff}");
    // The resident DiT installs and re-attaches every block's targets at once.
    for q in [q0, q1] {
        host.adaptable_mut(&q)
            .unwrap()
            .set_training_adapters(vec![trainable.clone()]);
    }
    residuals.reattach(&mut host, None).unwrap();
    for q in [q0, q1] {
        assert_eq!(host.adaptable_mut(&q).unwrap().adapters().len(), 2);
    }
}
