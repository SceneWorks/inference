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
