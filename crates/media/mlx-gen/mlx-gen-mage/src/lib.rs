//! # mlx-gen-mage
//!
//! The **Mage-Flow** (`microsoft/Mage`, MIT) provider crate for [`mlx-gen`](mlx_gen): a compact 4B
//! text→image + instruction-editing stack made of a one-step **Mage-VAE** codec (128-channel,
//! 16×-downsampled latents), a 12-block **NR-MMDiT** rectified-flow denoiser, and a **Qwen3-VL-4B**
//! text encoder.
//!
//! ## Status
//!
//! The RL checkpoint is a complete registered text-to-image provider through the normal
//! [`mlx_gen::Generator`] surface. The separate full Base and Turbo checkpoints are also registered
//! at `mage_flow_base` and `mage_flow_turbo`; all three instruction-editing checkpoints share the
//! reviewed edit pipeline under distinct checkpoint-identified provider IDs.
//!
//! ## Reuse lineage
//!
//! Mage's own `transformer/config.json` declares `schedule_mode: "z-image"`, `rope_type: "msrope"`,
//! `time_type: "qwen_proj"` and `double_block_type: "double_stream"`: the NR-MMDiT is a
//! reparameterised Z-Image (Tongyi) S3-DiT, so `mlx-gen-z-image`-shaped module boundaries are the
//! right template. **Two inherited assumptions are wrong for Mage and are pinned as constants in
//! [`config`] so they cannot leak back in:** the DiT FFN is `gelu-approximate`, *not* SwiGLU
//! ([`config::FFN_ACTIVATION`]), and the text conditioning is the **final** post-RMSNorm hidden
//! state, *not* the penultimate layer.
//!
//! ## Ground truth
//!
//! The frozen PyTorch reference is vendored byte-identically at
//! `crates/media/mlx-gen/_vendor/mage_flow/` (upstream `microsoft/Mage @ df7f84d9`); the six
//! architectural questions it answers are written up in `_vendor/MAGE_FLOW_GAPS.md`, and CPU
//! boundary goldens plus their hardened checker live under `crates/media/mlx-gen/tools/`. Read
//! those before porting anything — several of them correct the original epic description.
//!
//! ## Deliberate divergence
//!
//! The reference runs a **mandatory, fail-closed content-moderation gate** over every prompt
//! (`TextEncoder.screen_text` / `screen_edit`). It is **not** ported (decision recorded on
//! sc-14105): SceneWorks ships no content classifier for any family, by documented product
//! posture. The text-encoder port needs the *embedding* forward only — no `lm_head`, no KV-cache
//! decode loop, no `.generate()`. The Gaussian-Shading watermark ([`latent`]) is kept: provenance
//! marking stays, blocking does not.
//!
//! [`mlx-gen-catalog`]: https://docs.rs/mlx-gen-catalog

// ---------------------------------------------------------------------------------------------
// DECISION (sc-14037): there is deliberately **no `loader.rs`**.
//
// The sibling convention (`mlx-gen-z-image/src/loader.rs`) puts `load_transformer` /
// `load_text_encoder` / `load_vae` in one shared file. That is a three-way collision here, because
// sc-14038, sc-14039 and sc-14040 land concurrently and would each create the same file *and* each
// add the same `pub mod loader;` line. **Each component's weight loading lives inside the module
// that owns it** — `text_encoder::load`, `vae::load`, `transformer::load` — so a story touches only
// its own files. sc-14041, which assembles them, may add a thin `loader.rs` that re-exports the
// three (restoring the sibling's public shape) once all three exist; it must not move the bodies.
// ---------------------------------------------------------------------------------------------

pub mod config;
pub mod memory;
pub mod model;

// --- Physical per-tier quant artifacts (sc-14980) --------------------------------------------
// `quant` is the Group-B packed-load template (sc-8669); `convert` is the offline producer that
// writes the `q4/`/`q8/`/`bf16/` artifacts the SceneWorks mirrors host.
pub mod convert;
pub(crate) mod quant;

// --- NR-MMDiT (sc-14040) -------------------------------------------------------------------
pub mod attention;
pub(crate) mod block_stream;
pub mod feed_forward;
pub mod final_layer;
pub mod rope_embedder;
pub mod timestep_embedder;
pub mod transformer;
pub mod transformer_block;

// --- Qwen3-VL text encoder (sc-14038; vision tower sc-14048) ---------------------------------
pub mod text_encoder;

// --- Mage-VAE one-step codec (sc-14039) ------------------------------------------------------
pub mod vae;

// --- Gaussian-Shading watermarked initial noise (sc-14104) -----------------------------------
pub mod latent;

// --- Rectified-flow sampler + native-resolution packing (sc-14041) ---------------------------
pub mod pipeline;

// --- LoRA/LoKr adapter reload (sc-14055) + rectified-flow LoRA trainer (sc-14055) -------------
pub mod adapters;
pub mod training;

// ---------------------------------------------------------------------------------------------
// Re-export surface.
//
// PARALLEL-EXECUTION CONTRACT (sc-14037): the four P1 ports land concurrently, so each owns ONE
// pre-seeded line below. **Replace your own placeholder in place — do not append, reorder, or
// touch a neighbouring line** and the four diffs stay in separate hunks. `lib.rs` already declares
// every module above, so no story needs to add a `mod` line either.
// ---------------------------------------------------------------------------------------------
pub use config::{MageFlowConfig, QwenVlTextConfig, FAMILY};
pub use model::{descriptor_for, MageVariant, MODEL_IDS};
// sc-15036: the fine-tuned-checkpoint entrypoint + the component ids a caller must stage for it.
pub use model::{load_finetuned, COMPONENT_TEXT_ENCODER, COMPONENT_VAE, REQUIRED_COMPONENTS};
pub use text_encoder::{Conditioning, MageTextEncoder, PromptKind, Qwen3VlTextEncoder};

pub use vae::{MageVae, VaePart};

pub use attention::{DualStream, MageJointAttention};
pub use feed_forward::MageFeedForward;
pub use final_layer::MageFinalLayer;
pub use rope_embedder::{ImgShape, MsRope, PackContext, PackLayout, RopeTable};
pub use timestep_embedder::MageTimestepEmbedder;
pub use transformer::{Linear, MageTransformer};
pub use transformer_block::MageTransformerBlock;

pub use latent::{
    decode_bits, encode_noise, invert_to_noise, resolve_gs_key, GsKey, WatermarkReport,
}; // sc-14104 (Gaussian-Shading noise)

// sc-14041 (pipeline + the loaded model) re-exports here:
pub use pipeline::{
    mage_flow_sigmas, BatchGenerationTrace, EditTrace, GenerationPack, GenerationSample,
    GenerationTrace, MageComponentDirs, MageFlowPipeline, MAX_PACKED_IMAGE_TOKENS, STATIC_SHIFT,
};

// sc-14055 (LoRA/LoKr reload + rectified-flow trainer):
pub use adapters::apply_mage_adapters;
pub use training::{MageFlowTrainer, MODEL_ID as TRAINER_MODEL_ID};

// Later phases add their own modules rather than growing these: `quant` (Q4/Q8 tiers, sc-14046),
// `convert` (offline pre-quantisation, sc-14046), `adapters` (LoRA/LoKr routing, sc-14057) and
// `training` (LoRA + base fine-tune, sc-14055/sc-14056). They are not stubbed here because their
// shape is decided by those stories, not by this one.

/// sc-16209 Apple-Silicon warm sweep: Mage Flow bf16 peaked below 1.57 GiB at 1024².
/// A two-step control set the published high-water mark.
pub const ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: "mage_flow",
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 1_685_774_664,
        },
    };

/// Add every Mage-Flow MLX provider to an explicit media registry builder.
///
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_activation_memory(ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model::REGISTRATION_BASE)
        .register_checkpoint_adapter(mlx_gen::gen_core::CheckpointAdapterRegistration {
            backend_bindings: &[mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                source: mlx_gen::gen_core::ImportedModelSource::TransformerDirectory,
                operation: mlx_gen::gen_core::ImportedModelOperation::Generate,
                provider_id: "mage_flow_base",
                required_components: Some(model::REQUIRED_COMPONENTS),
                inherit_adapters: false,
            }],
            ..mlx_gen::gen_core::MAGE_FLOW_CHECKPOINT_ADAPTER
        })
        .register_generator(model::REGISTRATION_TURBO)
        .register_generator(model::REGISTRATION_EDIT)
        .register_generator(model::REGISTRATION_EDIT_BASE)
        .register_generator(model::REGISTRATION_EDIT_TURBO)
        .register_memory_strategy(model::MEMORY_REGISTRATION)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow",
            contract: |spec| model::weights_free_memory_strategy_contract("mage_flow", spec),
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow", surface)
                },
            },
        )
        .register_memory_strategy(model::MEMORY_REGISTRATION_BASE)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION_BASE)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow_base",
            contract: |spec| model::weights_free_memory_strategy_contract("mage_flow_base", spec),
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow_base",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow_base", surface)
                },
            },
        )
        .register_memory_strategy(model::MEMORY_REGISTRATION_TURBO)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION_TURBO)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow_turbo",
            contract: |spec| model::weights_free_memory_strategy_contract("mage_flow_turbo", spec),
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow_turbo",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow_turbo", surface)
                },
            },
        )
        .register_memory_strategy(model::MEMORY_REGISTRATION_EDIT)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION_EDIT)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow_edit",
            contract: |spec| model::weights_free_memory_strategy_contract("mage_flow_edit", spec),
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow_edit",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow_edit", surface)
                },
            },
        )
        .register_memory_strategy(model::MEMORY_REGISTRATION_EDIT_BASE)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION_EDIT_BASE)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow_edit_base",
            contract: |spec| {
                model::weights_free_memory_strategy_contract("mage_flow_edit_base", spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow_edit_base",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow_edit_base", surface)
                },
            },
        )
        .register_memory_strategy(model::MEMORY_REGISTRATION_EDIT_TURBO)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION_EDIT_TURBO)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: "mage_flow_edit_turbo",
            contract: |spec| {
                model::weights_free_memory_strategy_contract("mage_flow_edit_turbo", spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "mage_flow_edit_turbo",
                contract: |surface| {
                    model::weights_free_memory_surface_contract("mage_flow_edit_turbo", surface)
                },
            },
        )
        // The rectified-flow LoRA/LoKr trainer targets the Base checkpoint (sc-14055).
        .register_trainer(training::REGISTRATION)
}

/// Build the explicit Mage-Flow MLX provider catalog (this crate only).
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

    fn snapshot(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        let root = tmp.path().join("mage-registry");
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        root
    }

    use super::*;

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(
            generators,
            [
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo"
            ]
        );
        // The rectified-flow LoRA/LoKr trainer targets the Base checkpoint (sc-14055); no
        // captioner/embedder surface.
        let trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(trainers, ["mage_flow_base"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        let root = snapshot(&tmp);
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.clone()));
        for id in MODEL_IDS {
            assert!(registry
                .memory_strategy_contract(id, &spec)
                .unwrap()
                .is_some());
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn every_optimized_mage_route_registers_a_weights_free_behavior_seam() {
        let registry = provider_registry().unwrap();
        let mut ids = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        let mut expected = MODEL_IDS.to_vec();
        expected.sort_unstable();
        assert_eq!(ids, expected);
    }

    /// Every id is prefixed with the family id, matching the image-family convention
    /// (`flux2_*`, `krea_2_*`, `z_image_*`). SceneWorks routes and groups on this prefix.
    #[test]
    fn every_model_id_is_family_prefixed() {
        for id in MODEL_IDS {
            assert!(
                id.starts_with(FAMILY),
                "{id} does not carry the '{FAMILY}' family prefix"
            );
        }
    }

    #[test]
    fn imported_transformer_directory_route_executes_the_finetuned_loader() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), b"{}").unwrap();
        write_minimal_safetensors(&root.path().join("diffusion_pytorch_model.safetensors"));
        let registry = provider_registry().unwrap();
        let descriptor = registry
            .imported_model_descriptor(
                "mage-flow",
                mlx_gen::gen_core::ImportedModelSource::TransformerDirectory,
                mlx_gen::gen_core::ImportedModelOperation::Generate,
            )
            .expect("exact Mage fine-tune route");
        assert_eq!(descriptor.id, "mage_flow_base");
        assert_eq!(descriptor.required_components, model::REQUIRED_COMPONENTS);
        assert!(!descriptor.capabilities.supports_lora);
        assert!(!descriptor.capabilities.supports_lokr);

        let error = registry
            .load(
                descriptor.id,
                &mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(root.path().to_path_buf())),
            )
            .err()
            .expect("unstaged fine-tune must fail at its component gate")
            .to_string();
        assert!(
            error.contains(model::COMPONENT_TEXT_ENCODER),
            "the selected registry loader must enter the fine-tune component gate, got: {error}"
        );
        assert!(!error.contains("checkpoint fingerprint"), "{error}");
    }
}
