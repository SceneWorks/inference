//! # mlx-gen-krea
//!
//! The **Krea 2** provider crate for [`mlx-gen`](mlx_gen). Krea 2 is Krea AI's first from-scratch
//! foundation image model (released 2026-06-22). Two surfaces share **one architecture**:
//! - **Krea 2 Turbo** (`krea_2_turbo`) — the user-facing text-to-image model (TDM-distilled few-step,
//!   CFG-free, 8 steps, up to 2048²),
//! - **Krea 2 Raw** (`krea_2_raw`) — the undistilled base. Both a **generation model** (full-CFG
//!   text-to-image: real guidance + a user negative prompt, 52 steps, resolution-dynamic mu — epic
//!   9992) AND the **LoRA-training base** (LoRAs train on Raw and apply at Turbo inference, the Lens /
//!   Z-Image precedent — epic 7565 P3). One id, both roles (generator + trainer registries).
//!
//! ## Architecture (verified against the real `krea/Krea-2-Turbo` configs + safetensors index)
//! - **DiT** — `Krea2Transformer2DModel`, a **dense single-stream** rectified-flow / v-param
//!   transformer: text + image tokens concatenated through 28 gated single-stream `transformer_blocks`
//!   (hidden 6144, GQA 48Q/12KV, head_dim 128, SwiGLU 16384, 3-axis RoPE `[32,48,48]`), a
//!   `DoubleSharedModulation` (one shared 6-factor `time_mod_proj` + per-block `scale_shift_table`),
//!   and a `text_fusion` (`TextFusionTransformer`) front-end that aggregates the 12 selected Qwen3-VL
//!   hidden layers (2 layerwise cross-layer-axis blocks → learned `projector` 12→1 → 2 token-axis
//!   refiner blocks).
//! - **Text encoder** — `Qwen3-VL-4B-Instruct` (`Qwen3VLModel`): the pipeline stacks the
//!   `text_encoder_select_layers` `[2,5,…,35]` hidden states and feeds them to the DiT's `text_fusion`.
//! - **VAE** — `AutoencoderKLQwenImage` (z_dim 16, per-channel `latents_mean`/`latents_std` de-norm) —
//!   direct reuse of `mlx-gen-qwen-image`'s `QwenVae`.
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler`, v-param, dynamic exponential time-shift; Turbo
//!   fixes mu 1.15 / 8 steps / CFG 0.
//!
//! ## Surfaces (all landed)
//! The core Turbo t2i vertical (provider scaffold, `krea_2_turbo` registration, architecture-validated
//! [`model::load`], offline Q4/Q8 [`convert`]; the single-stream DiT [`transformer`] reusing
//! `mlx-gen-boogu`'s 3-axis-RoPE blocks; the Qwen3-VL-4B [`text_encoder`]; the [`vae`] reusing
//! `mlx-gen-qwen-image`'s `QwenVae` over the core [`mlx_gen::FlowMatchSampler`] [`schedule`]) landed in
//! epic 7565 P1 (sc-7567…sc-7571). Since then the crate grew four more registered generators plus a
//! trainer and the residency split — all in this crate:
//! - **`krea_2_raw`** ([`raw_descriptor`] / [`load_raw`]) — the undistilled full-CFG base (epic 9992):
//!   real guidance + a user negative prompt, resolution-dynamic mu, 52 steps.
//! - **`krea_2_edit`** ([`edit_descriptor`] / [`load_edit`]) — Kontext-style image edit on the Raw path
//!   (epic 10871): dual conditioning (in-context VAE reference tokens + the Qwen3-VL grounded encode),
//!   one source `Reference` or a scene+person `MultiReference`.
//! - **`krea_2_turbo_edit`** ([`turbo_edit_descriptor`] / [`load_turbo_edit`]) — the same edit surface
//!   on the distilled CFG-free few-step schedule (sc-11640).
//! - **`krea_2_turbo_control`** ([`KreaTurboControl`], `model_control::load`) — pose-ControlNet on
//!   Turbo (epic 8459), a `control_scale`-scaled RMS-clamped residual branch ([`control`]).
//! - **Raw LoRA/LoKr trainer** ([`KreaRawTrainer`] / [`load_trainer`]) — LoRAs train on Raw and apply at
//!   Turbo inference (the Lens / Z-Image precedent, epic 7565 P3).
//! - **Component residency** (epic 10834 / sc-11101) — the [`KreaText`] + [`KreaHeavy`] phase split that
//!   bounds peak unified memory under `Sequential`; the img2img, PiD-decode (`mlx-gen-pid`) and
//!   `from_ldm` early-stop seams thread through it.

pub mod block_memory_strategy;
mod block_stream;
pub mod config;
pub mod control;
pub mod convert;
pub mod loader;
pub mod memory;
pub mod memory_strategy;
pub mod model;
pub mod model_control;
pub mod multiphase;
pub mod native_remap;
pub mod pipeline;
mod quant;
pub mod schedule;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vae;

pub use block_stream::{
    block_stream_diagnostics, reset_block_stream_diagnostics, BlockStreamDiagnostics,
};
pub use config::Krea2Config;
pub use control::Krea2ControlBranch;
pub use loader::{
    last_native_file_receipt, load_text_encoder, load_transformer,
    load_transformer_from_native_file, reset_native_file_receipt,
};
pub use memory::{control_geometry_fits, require_control_geometry};
pub use model::{
    descriptor, edit_descriptor, load, load_edit, load_from_native_dit_file, load_raw,
    load_turbo_edit, raw_descriptor, turbo_edit_descriptor, Krea, KREA_2_EDIT_ID, KREA_2_RAW_ID,
    KREA_2_TURBO_EDIT_ID, KREA_2_TURBO_ID, RES_MULTIPLE,
};
pub use model_control::{
    load_control_from_native_dit_file, KreaTurboControl, KREA_2_TURBO_CONTROL_ID,
};
pub use multiphase::{
    any_phase_uses_cfg, phase_spec_subset, phase_uses_cfg, resolve_phase_adapters,
    resolve_phase_slices, resolve_phases, total_phase_steps, PhaseSlice, ResolvedPhase,
    ResolvedPhaseAdapter,
};
pub use native_remap::{
    native_dit_key_to_diffusers, remap_native_dit_to_diffusers, KreaNativeToDiffusersMapping,
};
pub use pipeline::{KreaHeavy, KreaPipeline, KreaText, MultiPhasePlan, TurboOptions};
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder, KreaTokenizer};
pub use training::{load_trainer, KreaRawTrainer, KREA_2_RAW_TRAINER_ID};
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae};

/// sc-16195 Apple-Silicon warm sweep: Krea 2 Turbo q8 and dense both peaked below 7.67 GiB
/// at 1024². Activations stay bf16 across weight tiers; Raw/Edit are distinct unmeasured routes.
pub const TURBO_ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: KREA_2_TURBO_ID,
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 8_235_599_791,
        },
    };

/// Add all MLX Krea generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::TURBO_REGISTRATION)
        .register_activation_memory(TURBO_ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model::RAW_REGISTRATION)
        .register_generator(model::EDIT_REGISTRATION)
        .register_generator(model::TURBO_EDIT_REGISTRATION)
        .register_memory_strategy(model::TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: KREA_2_TURBO_ID,
            contract: |spec| {
                block_memory_strategy::weights_free_memory_strategy_contract(KREA_2_TURBO_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: KREA_2_TURBO_ID,
                contract: |surface| {
                    block_memory_strategy::weights_free_memory_strategy_surface_contract(
                        KREA_2_TURBO_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model::TURBO_MEMORY_BEHAVIOR)
        .register_memory_strategy(model::RAW_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: KREA_2_RAW_ID,
            contract: |spec| {
                block_memory_strategy::weights_free_memory_strategy_contract(KREA_2_RAW_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: KREA_2_RAW_ID,
                contract: |surface| {
                    block_memory_strategy::weights_free_memory_strategy_surface_contract(
                        KREA_2_RAW_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model::RAW_MEMORY_BEHAVIOR)
        .register_memory_strategy(model::EDIT_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: KREA_2_EDIT_ID,
            contract: |spec| {
                block_memory_strategy::weights_free_memory_strategy_contract(KREA_2_EDIT_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: KREA_2_EDIT_ID,
                contract: |surface| {
                    block_memory_strategy::weights_free_memory_strategy_surface_contract(
                        KREA_2_EDIT_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model::EDIT_MEMORY_BEHAVIOR)
        .register_memory_strategy(model::TURBO_EDIT_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: KREA_2_TURBO_EDIT_ID,
            contract: |spec| {
                block_memory_strategy::weights_free_memory_strategy_contract(
                    KREA_2_TURBO_EDIT_ID,
                    spec,
                )
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: KREA_2_TURBO_EDIT_ID,
                contract: |surface| {
                    block_memory_strategy::weights_free_memory_strategy_surface_contract(
                        KREA_2_TURBO_EDIT_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model::TURBO_EDIT_MEMORY_BEHAVIOR)
        .register_generator(model_control::CONTROL_REGISTRATION)
        .register_memory_strategy(model_control::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: KREA_2_TURBO_CONTROL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(
                    KREA_2_TURBO_CONTROL_ID,
                    spec,
                )
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: KREA_2_TURBO_CONTROL_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_strategy_surface_contract(
                        KREA_2_TURBO_CONTROL_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model_control::MEMORY_BEHAVIOR_REGISTRATION)
        .register_checkpoint_adapter(mlx_gen::gen_core::CheckpointAdapterRegistration {
            backend_bindings: &[
                mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                    backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                    source: mlx_gen::gen_core::ImportedModelSource::TransformerFile,
                    operation: mlx_gen::gen_core::ImportedModelOperation::Generate,
                    provider_id: KREA_2_TURBO_ID,
                    required_components: Some(&[mlx_gen::BASE_SNAPSHOT_COMPONENT]),
                    inherit_adapters: true,
                },
                mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                    backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                    source: mlx_gen::gen_core::ImportedModelSource::TransformerFile,
                    operation: mlx_gen::gen_core::ImportedModelOperation::Edit,
                    provider_id: KREA_2_TURBO_EDIT_ID,
                    required_components: Some(&[mlx_gen::BASE_SNAPSHOT_COMPONENT]),
                    inherit_adapters: true,
                },
                mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                    backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                    source: mlx_gen::gen_core::ImportedModelSource::TransformerFile,
                    operation: mlx_gen::gen_core::ImportedModelOperation::Pose,
                    provider_id: KREA_2_TURBO_CONTROL_ID,
                    required_components: Some(&[mlx_gen::BASE_SNAPSHOT_COMPONENT]),
                    inherit_adapters: true,
                },
                mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                    backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                    source: mlx_gen::gen_core::ImportedModelSource::TransformerFile,
                    operation: mlx_gen::gen_core::ImportedModelOperation::MultiPhase,
                    provider_id: KREA_2_RAW_ID,
                    required_components: Some(&[mlx_gen::BASE_SNAPSHOT_COMPONENT]),
                    inherit_adapters: true,
                },
            ],
            ..mlx_gen::gen_core::KREA_2_CHECKPOINT_ADAPTER
        })
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX Krea provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn snapshot(tmp: &tempfile::TempDir, tag: &str) -> std::path::PathBuf {
        let root = tmp.path().join(format!("krea-{tag}"));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::model::ENCODER_CONTRACT,
            crate::model::VISION_ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
        root
    }

    #[test]
    fn every_base_variant_resolves_the_rung_four_contract_through_the_registry() {
        let tmp = tempfile::tempdir().unwrap();
        use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};

        let registry = super::provider_registry().unwrap();
        let root = snapshot(&tmp, "registry-memory");
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.clone()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        for id in [
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
        ] {
            let contract = registry
                .memory_strategy_contract(id, &spec)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} must register a memory contract"));
            assert_eq!(contract.provider_id, id);
            let rung = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert_eq!(rung.support, MemoryStrategySupport::Implemented, "{id}");
            assert_eq!(
                rung.parameters.transformer_window_sizes,
                [crate::block_memory_strategy::TRANSFORMER_WINDOW_SIZE]
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn every_registered_file_route_has_load_footprint_and_memory_contract_parity() {
        use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};

        let tmp = tempfile::tempdir().unwrap();
        let registry = super::provider_registry().unwrap();
        let base = snapshot(&tmp, "registry-file-matrix");
        let dit = tmp.path().join("imported.safetensors");
        let control = tmp.path().join("control.safetensors");
        write_minimal_safetensors(&dit);
        write_minimal_safetensors(&control);
        let base_spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::File(dit.clone()))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                mlx_gen::WeightsSource::Dir(base.clone()),
            )
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);

        for id in [
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
        ] {
            let footprint = registry
                .footprint(id, &base_spec)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} must expose a footprint"));
            let selected = crate::model::ENCODER_CONTRACT
                .source_for_load(&base_spec, &base)
                .unwrap();
            let language = crate::model::selected_language_resident_bytes(
                &selected,
                crate::model::native_text_encoder_expected_quant_bits(&base).unwrap(),
                id,
            )
            .unwrap();
            let vision = if matches!(id, "krea_2_edit" | "krea_2_turbo_edit") {
                let builtin = crate::model::ENCODER_CONTRACT
                    .validate_source_against_base(
                        &mlx_gen::WeightsSource::Dir(base.join("text_encoder")),
                        &base,
                    )
                    .unwrap();
                let headers = builtin
                    .materialized_vision_tensor_headers(
                        &crate::model::VISION_ENCODER_CONTRACT,
                        &crate::model::ENCODER_CONTRACT,
                    )
                    .unwrap();
                mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |_| {
                    mlx_gen::asset_facts::ResidentProjection::Stored
                })
                .unwrap()
            } else {
                0
            };
            assert_eq!(footprint.text_encoder, language + vision, "{id}");
            assert_eq!(footprint.dit, mlx_gen::safetensors_path_bytes(&dit), "{id}");
            assert_eq!(
                footprint.vae,
                mlx_gen::safetensors_path_bytes(base.join("vae")),
                "{id}"
            );
            let contract = registry
                .memory_strategy_contract(id, &base_spec)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} must expose a memory contract"));
            assert_eq!(contract.provider_id, id);
            assert_eq!(contract.asset_facts.transformer_bytes, 2, "{id}");
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "{id}"
            );
            let loaded = registry.load(id, &base_spec).unwrap_or_else(|error| {
                panic!("{id} File load must agree with its public contracts: {error}")
            });
            assert_eq!(
                loaded
                    .memory_strategy_contract()
                    .expect("loaded File generator memory contract")
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "{id} loaded File generator must keep the unpromoted rung eager"
            );
        }

        let control_spec = base_spec
            .clone()
            .with_control(mlx_gen::WeightsSource::File(control));
        let id = "krea_2_turbo_control";
        assert!(registry.footprint(id, &control_spec).unwrap().is_some());
        let contract = registry
            .memory_strategy_contract(id, &control_spec)
            .unwrap()
            .expect("control memory contract");
        assert_eq!(contract.provider_id, id);
        assert!(contract.asset_facts.overlay_bytes > 0);
        let loaded = registry
            .load(id, &control_spec)
            .expect("control File load must agree with its public contracts");
        assert_eq!(
            loaded
                .memory_strategy_contract()
                .expect("loaded control File generator memory contract")
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn every_registered_file_route_rejects_unrealized_fields_and_accepts_selected_encoder() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = super::provider_registry().unwrap();
        let base = snapshot(&tmp, "registry-file-typed-fields");
        let dit = tmp.path().join("imported.safetensors");
        let control = tmp.path().join("control.safetensors");
        write_minimal_safetensors(&dit);
        write_minimal_safetensors(&control);
        let external_text_encoder = tmp.path().join("external-text-encoder");
        gen_core_testkit::write_encoder_contract_fixture(
            &external_text_encoder,
            crate::model::ENCODER_CONTRACT,
        )
        .expect("validation-complete selected text encoder fixture");
        let base_spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::File(dit))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                mlx_gen::WeightsSource::Dir(base),
            )
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential);

        for id in [
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
            "krea_2_turbo_control",
        ] {
            let route_spec = if id == "krea_2_turbo_control" {
                base_spec
                    .clone()
                    .with_control(mlx_gen::WeightsSource::File(control.clone()))
            } else {
                base_spec.clone()
            };

            let mut identity = route_spec.clone();
            identity.identity = Some(mlx_gen::IdentityWeights::default());
            let error = registry
                .load(id, &identity)
                .err()
                .expect("identity field must be rejected")
                .to_string();
            assert!(error.contains("identity"), "{id}: {error}");
            let contract_error = registry
                .memory_strategy_contract(id, &identity)
                .expect_err("identity field must be rejected by the memory contract")
                .to_string();
            assert!(
                contract_error.contains("identity"),
                "{id}: {contract_error}"
            );

            let mut text_encoder = route_spec;
            text_encoder.text_encoder =
                Some(mlx_gen::WeightsSource::Dir(external_text_encoder.clone()));
            registry
                .load(id, &text_encoder)
                .unwrap_or_else(|error| panic!("{id}: selected text encoder rejected: {error}"));
            assert!(
                registry
                    .memory_strategy_contract(id, &text_encoder)
                    .unwrap_or_else(|error| panic!("{id}: selected contract rejected: {error}"))
                    .is_some(),
                "{id} must retain its memory contract with a compatible selected encoder"
            );
        }
    }

    #[test]
    fn raw_directory_load_contract_and_footprint_reject_ignored_axes_and_preserve_supported_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = super::provider_registry().unwrap();
        let root = snapshot(&tmp, "raw-dir-typed-fields");
        let base = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.clone()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);

        let mut identity = base.clone();
        identity.identity = Some(mlx_gen::IdentityWeights::default());
        let unknown_component = base.clone().with_component(
            "unknown",
            mlx_gen::WeightsSource::File("/unknown.safetensors".into()),
        );
        let unsupported = [
            base.clone()
                .with_control(mlx_gen::WeightsSource::File("/control.safetensors".into())),
            base.clone()
                .with_extra_control(mlx_gen::WeightsSource::File(
                    "/extra-control.safetensors".into(),
                )),
            base.clone().with_ip_adapter(mlx_gen::WeightsSource::File(
                "/ip-adapter.safetensors".into(),
            )),
            identity,
            unknown_component,
        ];
        for spec in unsupported {
            for error in [
                registry
                    .memory_strategy_contract(crate::KREA_2_RAW_ID, &spec)
                    .expect_err("contract must reject unsupported Raw axis")
                    .to_string(),
                registry
                    .footprint(crate::KREA_2_RAW_ID, &spec)
                    .expect_err("footprint must reject unsupported Raw axis")
                    .to_string(),
                registry
                    .load(crate::KREA_2_RAW_ID, &spec)
                    .err()
                    .expect("load must reject unsupported Raw axis")
                    .to_string(),
            ] {
                assert!(!error.is_empty(), "fail-closed rejection must be explicit");
            }
        }

        let low_rank = root.join("low-rank.safetensors");
        write_minimal_safetensors(&low_rank);
        let lora = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            low_rank,
            1.0,
            mlx_gen::AdapterKind::Lora,
        )]);
        let lokr_file = root.join("low-rank-lokr.safetensors");
        write_minimal_safetensors(&lokr_file);
        let lokr = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            lokr_file,
            1.0,
            mlx_gen::AdapterKind::Lokr,
        )]);

        let external_text_encoder = tmp.path().join("raw-external-text-encoder");
        gen_core_testkit::write_encoder_contract_fixture(
            &external_text_encoder,
            crate::model::ENCODER_CONTRACT,
        )
        .expect("validation-complete selected text encoder fixture");
        let mut selected_encoder = base.clone();
        selected_encoder.text_encoder = Some(mlx_gen::WeightsSource::Dir(external_text_encoder));

        let pid = base.clone().with_pid(
            mlx_gen::WeightsSource::File("/pid.safetensors".into()),
            mlx_gen::WeightsSource::Dir("/pid-text-encoder".into()),
        );

        let wan_vae = tmp.path().join("wan-vae.safetensors");
        write_minimal_safetensors(&wan_vae);
        let alternate_decoder = base.clone().with_component(
            mlx_gen::VAE_COMPONENT,
            mlx_gen::WeightsSource::File(wan_vae),
        );

        let behavior = registry
            .memory_behavior_registrations()
            .find(|registration| registration.provider_id == crate::KREA_2_RAW_ID)
            .expect("Raw memory behavior registration");
        for (profile, spec) in [
            ("plain", base),
            ("lora", lora),
            ("lokr", lokr),
            ("external_text_encoder", selected_encoder),
            ("pid", pid),
            ("wan_vae", alternate_decoder),
        ] {
            let contract = registry
                .memory_strategy_contract(crate::KREA_2_RAW_ID, &spec)
                .unwrap()
                .expect("supported Raw load must retain its memory contract");
            assert_eq!(
                contract
                    .capability(mlx_gen::gen_core::MemoryStrategy::BoundedTransformerResidency)
                    .expect("complete Raw ladder")
                    .support,
                mlx_gen::gen_core::MemoryStrategySupport::Implemented,
                "{profile}"
            );
            let fixtures = (behavior.valid_fixtures)(
                &spec,
                &contract,
                mlx_gen::gen_core::MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap();
            for (mode, references) in [
                (mlx_gen::gen_core::MemoryMode::TextToImage, 0),
                (mlx_gen::gen_core::MemoryMode::ImageToImage, 1),
            ] {
                for use_pid in [false, true] {
                    assert!(
                        fixtures.iter().any(|fixture| {
                            fixture.context.mode == mode
                                && fixture.context.geometry.reference_count == references
                                && fixture.context.use_pid == use_pid
                        }),
                        "{profile} must expose {mode:?} with use_pid={use_pid}"
                    );
                }
            }
            assert!(
                fixtures.iter().any(|fixture| fixture.context.use_pid),
                "{profile}: provider-owned behavior inventory must preserve the PiD route"
            );
            registry
                .load(crate::KREA_2_RAW_ID, &spec)
                .unwrap_or_else(|error| panic!("supported Raw {profile} load rejected: {error}"));
        }
    }

    /// sc-18451: the pose-control route must publish the SAME registry-load surface the four base
    /// routes do, and every rung it declares there must be executable through its own registered
    /// behavior — contract → admission → request scope → the request controls the pipeline reads.
    /// Before the selector-aware resolver was registered, the q4 witness below reported rung 4
    /// `Missing` and none of this was reachable at all.
    #[test]
    fn control_route_surface_walks_contract_safety_and_scope_into_request_controls() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemorySafetyDecision, MemoryStrategy,
            MemoryStrategySupport, OffloadPolicy, TransformerComponent,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let control: Vec<_> = surfaces
            .iter()
            .filter(|surface| surface.contract.provider_id == crate::KREA_2_TURBO_CONTROL_ID)
            .collect();
        assert!(control.iter().all(|surface| !surface.composed));
        for surface in &control {
            let streamable = surface.selector.offload_policy == OffloadPolicy::Sequential
                && surface.selector.load_shape == LoadShape::DeferredMaterialization;
            assert_eq!(
                surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .expect("complete pose-control ladder")
                    .support,
                if streamable {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                "{}",
                surface.selector.id()
            );
            for strategy in [MemoryStrategy::Resident, MemoryStrategy::BoundedDecode] {
                assert_eq!(
                    surface.contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{}: {strategy:?}",
                    surface.selector.id()
                );
            }
            assert_eq!(surface.contract.asset_facts, Default::default());
            assert_ne!(
                surface.contract.calibration.as_ref().unwrap().fingerprint,
                crate::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
                "{}: the declaration surface must not publish the measured q4 identity",
                surface.selector.id()
            );
        }

        // Q4 is the tier the pose-control evidence covers, and a shipped q4 turnkey is prepacked, so
        // this is the exact witness the missing resolver used to strand at rung 4 `Missing`.
        let surface = control
            .iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == MemoryContractSurfaceTier::Q4
                    && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            })
            .expect("q4 sequential deferred pose-control surface");
        let behavior = registry
            .memory_behavior_registrations()
            .find(|behavior| behavior.provider_id == crate::KREA_2_TURBO_CONTROL_ID)
            .expect("pose-control memory behavior");
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == crate::KREA_2_TURBO_CONTROL_ID)
            .expect("pose-control memory registration");
        let mut fixtures = (behavior.valid_fixtures)(
            &surface.spec,
            &surface.contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        assert!(!fixtures.is_empty());
        let fixture = &mut fixtures[0];
        assert_eq!(fixture.context.overlay.as_deref(), Some("pose-control"));
        assert!(!fixture.context.use_pid);
        assert_eq!(
            (registration.safety_check)(&surface.spec, &surface.contract, &fixture.context),
            MemorySafetyDecision::Accept
        );

        let mut scope =
            (behavior.begin_request)(&surface.spec, &surface.contract, &fixture.context)
                .unwrap()
                .expect("the declared rung must open a request scope");
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("configured request memory");
        // The SELECTED parameters, not defaults: the engine reads exactly these.
        assert!(memory.stream_transformer_blocks);
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert_eq!(
            memory.transformer_window_size,
            Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
        );
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Dit)
        );
        assert_eq!(
            memory.decode_tile_edge,
            Some(crate::memory_strategy::DECODE_TILE_EDGE)
        );
        assert_eq!(
            memory.decode_overlap,
            Some(crate::memory_strategy::DECODE_OVERLAP)
        );
        assert_eq!(
            memory.attention_chunk_size,
            Some(crate::memory_strategy::ATTENTION_CHUNK_SIZE)
        );

        // Each admission guard mutated ALONE, so none can hide behind another.
        let reject = |context: &mlx_gen::gen_core::MemoryRunContext, case: &str| {
            assert!(
                matches!(
                    (registration.safety_check)(&surface.spec, &surface.contract, context),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{case} must be refused at admission"
            );
            assert!(
                (behavior.begin_request)(&surface.spec, &surface.contract, context).is_err(),
                "{case} must not open a request scope"
            );
        };
        let mut pid = fixture.context.clone();
        pid.use_pid = true;
        reject(&pid, "pid");
        let mut edge = fixture.context.clone();
        edge.selection.parameters.decode_tile_edge = Some(384);
        reject(&edge, "decode_tile_edge");
        let mut overlap = fixture.context.clone();
        overlap.selection.parameters.decode_overlap = Some(32);
        reject(&overlap, "decode_overlap");
        let mut tier = fixture.context.clone();
        tier.selection.tier.quant = None;
        reject(&tier, "tier");
        let mut handshake = fixture.context.clone();
        handshake.calibration_fingerprint = "krea-control-mlx-not-this-one".to_owned();
        reject(&handshake, "calibration_fingerprint");
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = super::provider_registry().unwrap();
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();
        let explicit_generators: Vec<_> =
            descriptors.iter().map(|descriptor| descriptor.id).collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            explicit_generators,
            [
                "krea_2_turbo",
                "krea_2_raw",
                "krea_2_edit",
                "krea_2_turbo_edit",
                "krea_2_turbo_control",
            ]
        );
        assert_eq!(explicit_trainers, ["krea_2_raw"]);
        assert!(descriptors
            .iter()
            .all(|descriptor| descriptor.capabilities.supports_preview));

        let root = snapshot(&tmp, "registry-control-memory");
        let control = root.join("control.safetensors");
        write_minimal_safetensors(&control);
        let contract = registry
            .memory_strategy_contract(
                "krea_2_turbo_control",
                // Q4 is the one tier SC-15517 measured, so it is the one route that carries the
                // measured identity the run context below hands back in its handshake.
                &mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.clone()))
                    .with_quant(mlx_gen::Quant::Q4)
                    .with_control(mlx_gen::WeightsSource::File(control))
                    .with_offload_policy(mlx_gen::OffloadPolicy::Sequential),
            )
            .unwrap()
            .expect("Krea control memory contract");
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            crate::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            contract
                .capability(mlx_gen::gen_core::MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            crate::memory_strategy::DECODE_TILE_EDGES
        );
        assert_eq!(
            crate::memory_strategy::DECODE_TILE_EDGES,
            [crate::memory_strategy::DECODE_TILE_EDGE],
            "only the exact real-weight-verified 512 px decode edge may be advertised"
        );
        let routes = mlx_gen_pid::assert_decode_routes(
            "krea_2_turbo_control",
            crate::memory_strategy::DECODE_TILE_EDGES,
            crate::memory_strategy::DECODE_OVERLAP,
        );
        assert_eq!(
            routes.native_edges(),
            crate::memory_strategy::DECODE_TILE_EDGES
        );
        gen_core_testkit::check_memory_strategy_contract(&contract)
            .expect("Krea control memory contract conformance");
        std::fs::remove_dir_all(root).ok();

        let context = mlx_gen::gen_core::MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: mlx_gen::gen_core::MemorySelection {
                strategy: mlx_gen::gen_core::MemoryStrategy::BoundedDecode,
                parameters: mlx_gen::gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(crate::memory_strategy::DECODE_TILE_EDGE),
                    decode_overlap: Some(crate::memory_strategy::DECODE_OVERLAP),
                    ..Default::default()
                },
                tier: mlx_gen::gen_core::MemoryNumericTier {
                    precision: mlx_gen::Precision::Bf16,
                    quant: Some(mlx_gen::Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: crate::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned(),
            load_shape: mlx_gen::LoadShape::EagerMaterialization,
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: true,
            has_phases: false,
            geometry: mlx_gen::gen_core::MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: Some("control:1".to_owned()),
            budget: mlx_gen::gen_core::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: mlx_gen::gen_core::MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        };
        assert!(matches!(
            crate::memory_strategy::safety_check(
                &contract,
                mlx_gen::Precision::Bf16,
                Some(mlx_gen::Quant::Q4),
                &context,
            ),
            mlx_gen::gen_core::MemorySafetyDecision::Reject { reason }
                if reason.contains("PiD decode is not implemented")
        ));
    }
}

#[cfg(test)]
mod reexport_tests {
    //! F-077: the crate-root re-exports must cover the FULL registered surface — including the edit API
    //! (`load_edit`, `edit_descriptor`, `KREA_2_EDIT_ID`, and the Turbo-edit trio) which was previously
    //! reachable only via the `model::` path. Referencing each at the crate root pins the re-export.
    #[test]
    fn edit_surface_is_reexported_at_crate_root() {
        // Ids.
        assert_eq!(crate::KREA_2_EDIT_ID, "krea_2_edit");
        assert_eq!(crate::KREA_2_TURBO_EDIT_ID, "krea_2_turbo_edit");
        // Descriptors + loaders: referencing each function item at the crate root fails to compile if
        // the re-export is missing, and their ids must match.
        let _ = crate::load_edit;
        let _ = crate::load_turbo_edit;
        assert_eq!(crate::edit_descriptor().id, crate::KREA_2_EDIT_ID);
        assert_eq!(
            crate::turbo_edit_descriptor().id,
            crate::KREA_2_TURBO_EDIT_ID
        );
    }
}
