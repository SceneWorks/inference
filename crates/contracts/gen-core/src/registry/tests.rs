use super::*;
use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
use crate::audio_transform::{
    AudioTarget, AudioTransform, AudioTransformCapabilities, AudioTransformDescriptor,
    AudioTransformKind, AudioTransformRequest,
};
use crate::caption::{
    CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
};
use crate::generator::{
    reject_unsupported_adapters, ActivationMemoryAnchor, Capabilities, GenerationOutput,
    GenerationRequest, Modality, ModelDescriptor, SizeFloor,
};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::media::{AudioTrack, Image};
use crate::runtime::{Progress, WeightsSource};
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest};
use crate::transcribe::{
    TranscribeCapabilities, TranscribeRequest, Transcriber, TranscriberDescriptor, TranscriptOutput,
};
use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
use std::path::PathBuf;

struct DummyGen {
    desc: ModelDescriptor,
}

impl Generator for DummyGen {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }
    fn validate(&self, _req: &GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Ok(GenerationOutput::Images(vec![Image::default()]))
    }
}

/// Small-but-coherent capabilities for the dummy registrations: the descriptor sweep runs over
/// the explicit fixture catalog, so the dummies must carry real bounds (a
/// `Capabilities::default()` has the F-084 all-zero bounds the sweep exists to reject).
fn dummy_caps() -> Capabilities {
    Capabilities {
        min_size: 64,
        max_size: 512,
        max_count: 1,
        ..Default::default()
    }
}

fn dummy_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "dummy_test_model",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(DummyGen {
        desc: dummy_descriptor(),
    }))
}

crate::register_generators! {
    const DUMMY_GENERATOR_REGISTRATION = dummy_descriptor => dummy_load
}

fn dummy_candle_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "dummy_candle_test_model",
        backend: "candle",
        ..dummy_descriptor()
    }
}

crate::register_generators! {
    const DUMMY_CANDLE_GENERATOR_REGISTRATION = dummy_candle_descriptor => dummy_load
}

fn dummy_other_candle_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "dummy_other_candle_model",
        family: "other",
        backend: "candle",
        ..dummy_descriptor()
    }
}

crate::register_generators! {
    const DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION = dummy_other_candle_descriptor => dummy_load
}

fn dummy_mage_mlx_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "mage_flow_base",
        family: "mage_flow",
        backend: "mlx",
        ..dummy_descriptor()
    }
}

crate::register_generators! {
    const DUMMY_MAGE_MLX_GENERATOR_REGISTRATION = dummy_mage_mlx_descriptor => dummy_load
}

fn dummy_legacy_mage_candle_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "dummy_legacy_mage_candle_model",
        family: "mage-flow",
        backend: "candle",
        ..dummy_descriptor()
    }
}

crate::register_generators! {
    const DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION =
        dummy_legacy_mage_candle_descriptor => dummy_load
}

struct DummyDelegatedGen {
    descriptor: ModelDescriptor,
}

impl DummyDelegatedGen {
    fn generate_impl(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Ok(GenerationOutput::Images(vec![Image::default()]))
    }
}

crate::impl_generator!(DummyDelegatedGen {
    validate: |_s, _req| Ok::<(), Error>(()),
    generate: generate_impl,
});

fn dummy_delegated_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "dummy_delegated_test_model",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

fn dummy_delegated_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(DummyDelegatedGen {
        descriptor: dummy_delegated_descriptor(),
    }))
}

crate::register_generators! {
    const DUMMY_DELEGATED_GENERATOR_REGISTRATION =
        dummy_delegated_descriptor => dummy_delegated_load
}

// A dummy generator that DECLARES a per-component footprint (sc-10894), exercising the
// `; footprint = …` macro arm and the [`footprint`] entry point. Its text encoder is under a
// non-standard `mllm/` subdir (the real boogu layout) — a naming a `text_encoder*` guesser would
// read as ZERO — so the provider-owned split is what finds it.
fn dummy_footprint_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "dummy_footprint_model",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

fn dummy_footprint_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(DummyGen {
        desc: dummy_footprint_descriptor(),
    }))
}

fn dummy_footprint(spec: &LoadSpec) -> Result<PerComponentBytes> {
    PerComponentBytes::from_spec_subdirs(spec, &["mllm"], &["transformer"], &["vae"])
}

crate::register_generators! {
    const DUMMY_FOOTPRINT_GENERATOR_REGISTRATION =
        dummy_footprint_descriptor => dummy_footprint_load;
    footprint = dummy_footprint
}

#[cfg(unix)]
static PREPARED_CALLBACK_REBIND: std::sync::Mutex<Option<(PathBuf, PathBuf, PathBuf)>> =
    std::sync::Mutex::new(None);

#[cfg(unix)]
fn rebind_prepared_callback_file() -> Result<()> {
    let (selected, staged_b, staged_a) = PREPARED_CALLBACK_REBIND
        .lock()
        .expect("callback rebinding lock")
        .take()
        .expect("callback rebinding fixture");
    std::fs::rename(staged_b, &selected)?;
    std::fs::rename(staged_a, &selected)?;
    Ok(())
}

#[cfg(unix)]
fn prepared_callback_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "prepared_callback_model",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

#[cfg(unix)]
fn prepared_callback_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    rebind_prepared_callback_file()?;
    Ok(Box::new(DummyGen {
        desc: prepared_callback_descriptor(),
    }))
}

#[cfg(unix)]
fn prepared_callback_footprint(_spec: &LoadSpec) -> Result<PerComponentBytes> {
    rebind_prepared_callback_file()?;
    Ok(PerComponentBytes::default())
}

#[cfg(unix)]
fn prepared_callback_memory_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    rebind_prepared_callback_file()?;
    Ok(MemoryProviderContract::compatibility_default(
        "prepared_callback_model",
        crate::memory_strategy::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

struct DummyTrainer {
    desc: TrainerDescriptor,
}

impl Trainer for DummyTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.desc
    }

    fn validate(&self, _req: &TrainingRequest) -> Result<()> {
        Ok(())
    }

    fn train(
        &mut self,
        _req: &TrainingRequest,
        _on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        Ok(TrainingOutput {
            adapter_path: PathBuf::from("/tmp/dummy.safetensors"),
            steps: 0,
            final_loss: 0.0,
        })
    }
}

fn dummy_trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: "dummy_test_trainer",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: crate::train::TrainingTechniques::NONE,
    }
}

fn dummy_trainer_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(DummyTrainer {
        desc: dummy_trainer_descriptor(),
    }))
}

crate::register_trainer! {
    const DUMMY_TRAINER_REGISTRATION = dummy_trainer_descriptor => dummy_trainer_load
}

// Multi-provider fixtures verify that independently named constants compose into one catalog.
fn dummy_multi_gen_a_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "dummy_multi_gen_a",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

fn dummy_multi_gen_b_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "dummy_multi_gen_b",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: dummy_caps(),
    }
}

fn dummy_multi_gen_a_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(DummyGen {
        desc: dummy_multi_gen_a_descriptor(),
    }))
}

fn dummy_multi_gen_b_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(DummyGen {
        desc: dummy_multi_gen_b_descriptor(),
    }))
}

crate::register_generators! {
    const DUMMY_MULTI_GENERATOR_A_REGISTRATION =
        dummy_multi_gen_a_descriptor => dummy_multi_gen_a_load
}
crate::register_generators! {
    const DUMMY_MULTI_GENERATOR_B_REGISTRATION =
        dummy_multi_gen_b_descriptor => dummy_multi_gen_b_load
}

fn dummy_multi_trainer_a_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: "dummy_multi_trainer_a",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: crate::train::TrainingTechniques::NONE,
    }
}

fn dummy_multi_trainer_b_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: "dummy_multi_trainer_b",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: crate::train::TrainingTechniques::NONE,
    }
}

fn dummy_multi_trainer_a_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(DummyTrainer {
        desc: dummy_multi_trainer_a_descriptor(),
    }))
}

fn dummy_multi_trainer_b_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(DummyTrainer {
        desc: dummy_multi_trainer_b_descriptor(),
    }))
}

crate::register_trainer! {
    const DUMMY_MULTI_TRAINER_A_REGISTRATION =
        dummy_multi_trainer_a_descriptor => dummy_multi_trainer_a_load
}
crate::register_trainer! {
    const DUMMY_MULTI_TRAINER_B_REGISTRATION =
        dummy_multi_trainer_b_descriptor => dummy_multi_trainer_b_load
}

struct DummyCaptioner {
    desc: CaptionerDescriptor,
}

impl Captioner for DummyCaptioner {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.desc
    }
    fn validate(&self, _req: &CaptionRequest) -> Result<()> {
        Ok(())
    }
    fn caption(
        &self,
        _req: &CaptionRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CaptionOutput> {
        Ok(CaptionOutput {
            text: "caption".to_owned(),
            generated_tokens: Some(1),
            finish_reason: None,
        })
    }
}

fn dummy_captioner_descriptor() -> CaptionerDescriptor {
    CaptionerDescriptor {
        id: "dummy_test_captioner",
        family: "test",
        backend: "mlx",
        capabilities: CaptionCapabilities {
            min_image_size: 1,
            max_image_size: 4096,
            max_prompt_chars: 4000,
            max_name_chars: 120,
            max_extra_options: 16,
            max_extra_option_chars: 500,
            max_trigger_words: 32,
            max_trigger_word_chars: 120,
            max_new_tokens: 1024,
            ..Default::default()
        },
    }
}

fn dummy_captioner_load(_spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
    Ok(Box::new(DummyCaptioner {
        desc: dummy_captioner_descriptor(),
    }))
}

crate::register_captioner! {
    const DUMMY_CAPTIONER_REGISTRATION = dummy_captioner_descriptor => dummy_captioner_load
}

struct DummyTranscriber {
    desc: TranscriberDescriptor,
}

impl Transcriber for DummyTranscriber {
    fn descriptor(&self) -> &TranscriberDescriptor {
        &self.desc
    }
    fn validate(&self, _req: &TranscribeRequest) -> Result<()> {
        Ok(())
    }
    fn transcribe(
        &self,
        _req: &TranscribeRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<TranscriptOutput> {
        Ok(TranscriptOutput {
            text: "transcript".to_owned(),
            generated_tokens: Some(1),
            ..Default::default()
        })
    }
}

fn dummy_transcriber_descriptor() -> TranscriberDescriptor {
    TranscriberDescriptor {
        id: "dummy_test_transcriber",
        family: "test",
        backend: "candle",
        capabilities: TranscribeCapabilities {
            languages: vec!["en"],
            supports_segment_timestamps: true,
            max_audio_seconds: 30.0,
            max_new_tokens: 448,
            ..Default::default()
        },
    }
}

fn dummy_transcriber_load(_spec: &LoadSpec) -> Result<Box<dyn Transcriber>> {
    Ok(Box::new(DummyTranscriber {
        desc: dummy_transcriber_descriptor(),
    }))
}

crate::register_transcriber! {
    const DUMMY_TRANSCRIBER_REGISTRATION =
        dummy_transcriber_descriptor => dummy_transcriber_load
}

fn dummy_registry() -> ProviderRegistry {
    ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_DELEGATED_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_FOOTPRINT_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_MULTI_GENERATOR_A_REGISTRATION)
        .register_generator(DUMMY_MULTI_GENERATOR_B_REGISTRATION)
        .register_trainer(DUMMY_TRAINER_REGISTRATION)
        .register_trainer(DUMMY_MULTI_TRAINER_A_REGISTRATION)
        .register_trainer(DUMMY_MULTI_TRAINER_B_REGISTRATION)
        .register_captioner(DUMMY_CAPTIONER_REGISTRATION)
        .register_transcriber(DUMMY_TRANSCRIBER_REGISTRATION)
        .register_text_embedder(DUMMY_TEXT_EMBEDDER_REGISTRATION)
        .register_image_embedder(DUMMY_IMAGE_EMBEDDER_REGISTRATION)
        .register_voice_embedder(DUMMY_VOICE_EMBEDDER_REGISTRATION)
        .register_audio_embedder(DUMMY_AUDIO_EMBEDDER_REGISTRATION)
        .build()
        .unwrap()
}

#[test]
fn registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let g = registry
        .load("dummy_test_model", &spec)
        .expect("dummy is registered");
    assert_eq!(g.descriptor().id, "dummy_test_model");
    assert_eq!(g.descriptor().modality, Modality::Image);
}

#[cfg(unix)]
#[test]
fn prepared_file_identity_guards_load_footprint_and_memory_callbacks() {
    use std::os::unix::fs::symlink;

    fn spec_with_rebinding_callback(tag: &str) -> (tempfile::TempDir, LoadSpec) {
        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join(format!("{tag}-blob-a"));
        let second = dir.path().join(format!("{tag}-blob-b"));
        let selected = dir.path().join(format!("{tag}.safetensors"));
        let staged_b = dir.path().join(format!("{tag}-staged-b.safetensors"));
        let staged_a = dir.path().join(format!("{tag}-staged-a.safetensors"));
        std::fs::write(&first, b"same-size-a").expect("write A");
        std::fs::write(&second, b"same-size-b").expect("write B");
        symlink(&first, &selected).expect("select A");
        symlink(&second, &staged_b).expect("stage B link");
        symlink(&first, &staged_a).expect("stage replacement A link");
        let mut spec = LoadSpec::new(WeightsSource::File(selected.clone()));
        spec.prepare_file_sources().expect("prepare A identity");
        *PREPARED_CALLBACK_REBIND
            .lock()
            .expect("callback rebinding lock") = Some((selected, staged_b, staged_a));
        (dir, spec)
    }

    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: prepared_callback_descriptor,
            load: prepared_callback_load,
            footprint: Some(prepared_callback_footprint),
        })
        .register_memory_strategy(MemoryRegistration {
            provider_id: "prepared_callback_model",
            contract: prepared_callback_memory_contract,
            safety_check: crate::memory_strategy::default_registered_memory_strategy_safety_check,
        })
        .build()
        .expect("callback-boundary registry");

    let (_load_dir, load_spec) = spec_with_rebinding_callback("load");
    let load_error = registry
        .load("prepared_callback_model", &load_spec)
        .err()
        .expect("load callback A -> B -> recreated A must fail");
    // The one crate-wide definition (`runtime::assert_artifact_seal_mismatch`, `#[cfg(test)]
    // pub(crate)`): this test is `#[cfg(unix)]`, so a helper defined only here reads as dead code
    // on Windows and was deleted there once — which broke every Unix lane (SC-22667 review).
    crate::runtime::assert_artifact_seal_mismatch(load_error);

    let (_footprint_dir, footprint_spec) = spec_with_rebinding_callback("footprint");
    let footprint_error = registry
        .footprint("prepared_callback_model", &footprint_spec)
        .expect_err("footprint callback A -> B -> recreated A must fail");
    crate::runtime::assert_artifact_seal_mismatch(footprint_error);

    let (_memory_dir, memory_spec) = spec_with_rebinding_callback("memory");
    let memory_error = registry
        .memory_strategy_contract("prepared_callback_model", &memory_spec)
        .expect_err("memory callback A -> B -> recreated A must fail");
    crate::runtime::assert_artifact_seal_mismatch(memory_error);
}

#[test]
fn explicit_registry_resolves_minimal_catalog() {
    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        })
        .build()
        .unwrap();
    assert_eq!(registry.generators().len(), 1);
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/tmp")));
    assert_eq!(
        registry
            .load("dummy_test_model", &spec)
            .unwrap()
            .descriptor()
            .id,
        "dummy_test_model"
    );
    assert!(registry.trainers().next().is_none());
}

#[test]
fn encoder_contract_route_registration_fails_closed() {
    let missing_target = ProviderRegistryBuilder::new()
        .register_encoder_contract_route(EncoderContractRouteRegistration {
            route_id: "dummy_route",
            provider_id: "missing_provider",
        })
        .build()
        .err()
        .expect("an encoder route cannot target an unregistered provider")
        .to_string();
    assert!(missing_target.contains("targets unregistered generator"));

    let target_without_contract = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_encoder_contract_route(EncoderContractRouteRegistration {
            route_id: "dummy_route",
            provider_id: "dummy_test_model",
        })
        .build()
        .err()
        .expect("an encoder route cannot target a provider without a contract")
        .to_string();
    assert!(target_without_contract.contains("with no encoder contract"));

    let duplicate = ProviderRegistryBuilder::new()
        .register_encoder_contract_route(EncoderContractRouteRegistration {
            route_id: "dummy_route",
            provider_id: "missing_a",
        })
        .register_encoder_contract_route(EncoderContractRouteRegistration {
            route_id: "dummy_route",
            provider_id: "missing_b",
        })
        .build()
        .err()
        .expect("duplicate encoder routes must fail before resolution")
        .to_string();
    assert!(duplicate.contains("duplicate encoder-contract route id"));

    let ordinary = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(ordinary.provider_encoder_contract("dummy_test_model"), None);
    assert_eq!(ordinary.provider_encoder_contract("unknown_route"), None);
}

fn composed_memory_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(MemoryProviderContract::compatibility_default(
        "dummy_composed_route",
        crate::memory_strategy::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

const DUMMY_COMPOSED_MEMORY_REGISTRATION: MemoryRegistration = MemoryRegistration {
    provider_id: "dummy_composed_route",
    contract: composed_memory_contract,
    safety_check: crate::memory_strategy::default_registered_memory_strategy_safety_check,
};

fn production_contract_requires_assets(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    Err(Error::Msg("production factory requires assets".to_owned()))
}

fn weights_free_fixture_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(MemoryProviderContract::compatibility_default(
        "dummy_weights_free_route",
        crate::memory_strategy::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

const DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION: MemoryRegistration = MemoryRegistration {
    provider_id: "dummy_weights_free_route",
    contract: production_contract_requires_assets,
    safety_check: crate::memory_strategy::default_registered_memory_strategy_safety_check,
};

const DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE: MemoryContractFixtureRegistration =
    MemoryContractFixtureRegistration {
        provider_id: "dummy_weights_free_route",
        contract: weights_free_fixture_contract,
        surface_specs: mlx_memory_contract_surface_specs,
    };

fn selector_aware_fixture_contract(
    surface: &MemoryContractSurfaceSpec,
) -> Result<MemoryProviderContract> {
    let mut contract = weights_free_fixture_contract(&surface.spec)?;
    contract.load_shape = surface.spec.load_shape;
    if surface.selector.tier == MemoryContractSurfaceTier::Q4
        && surface.selector.load_shape == crate::LoadShape::DeferredMaterialization
    {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters.transformer_window_sizes = vec![1];
        capability.parameters.transformer_window_components =
            vec![crate::memory_strategy::TransformerComponent::Dit];
        contract.lifecycle.transformer_window_materialization = true;
    }
    Ok(contract)
}

const DUMMY_SURFACE_RESOLVER: MemoryContractSurfaceResolverRegistration =
    MemoryContractSurfaceResolverRegistration {
        provider_id: "dummy_weights_free_route",
        contract: selector_aware_fixture_contract,
    };

const DUMMY_RESIDENT_ONLY_WITNESS: ResidentOnlyMemoryContractRegistration =
    ResidentOnlyMemoryContractRegistration {
        provider_id: "dummy_weights_free_route",
        contract: weights_free_fixture_contract,
        surface_specs: mlx_memory_contract_surface_specs,
    };

fn false_resident_only_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let mut contract = weights_free_fixture_contract(spec)?;
    contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .unwrap()
        .support = MemoryStrategySupport::Implemented;
    Ok(contract)
}

fn empty_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    Vec::new()
}

fn duplicate_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    let first = mlx_memory_contract_surface_specs().remove(0);
    let duplicate = mlx_memory_contract_surface_specs().remove(0);
    vec![first, duplicate]
}

fn mismatched_surface_spec() -> Vec<MemoryContractSurfaceSpec> {
    let mut surface = mlx_memory_contract_surface_specs().remove(0);
    surface.selector.tier = MemoryContractSurfaceTier::Q4;
    vec![surface]
}

fn wrong_provider_fixture_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(MemoryProviderContract::compatibility_default(
        "wrong_provider",
        crate::memory_strategy::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

#[test]
fn production_registry_resolution_never_uses_the_weights_free_factory() {
    let registry = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .build()
        .unwrap();
    let error = registry
        .memory_strategy_contract(
            "dummy_weights_free_route",
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("production factory requires assets"),
        "{error}"
    );
}

#[test]
fn weights_free_contract_fixtures_are_unique_and_paired() {
    let orphan = ProviderRegistryBuilder::new()
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .build()
        .err()
        .expect("an orphan fixture must fail")
        .to_string();
    assert!(
        orphan.contains("has no matching memory strategy"),
        "{orphan}"
    );

    let duplicate = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .build()
        .err()
        .expect("duplicate fixtures must fail")
        .to_string();
    assert!(
        duplicate.contains("duplicate memory-contract fixture provider id"),
        "{duplicate}"
    );

    let orphan_resolver = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
        .build()
        .err()
        .expect("a selector-aware resolver without a finite fixture must fail")
        .to_string();
    assert!(orphan_resolver.contains("has no matching contract-surface fixture"));

    let duplicate_resolver = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
        .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
        .build()
        .err()
        .expect("duplicate selector-aware resolvers must fail")
        .to_string();
    assert!(duplicate_resolver.contains("duplicate memory-contract surface resolver"));
}

#[test]
fn selector_aware_resolver_receives_the_explicit_resolved_artifact_tier() {
    let registry = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
        .build()
        .unwrap();
    let surfaces = registry.memory_contract_surfaces().unwrap();
    let implemented: Vec<_> = surfaces
        .iter()
        .filter(|surface| {
            surface
                .contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support
                == MemoryStrategySupport::Implemented
        })
        .map(|surface| surface.selector)
        .collect();
    assert_eq!(implemented.len(), 2);
    assert!(implemented.iter().all(|selector| {
        selector.tier == MemoryContractSurfaceTier::Q4
            && selector.load_shape == crate::LoadShape::DeferredMaterialization
    }));
    assert_eq!(
        registry
            .memory_contract_surface_resolver_registrations()
            .len(),
        1
    );
}

#[test]
fn contract_surface_inventory_fails_closed_on_missing_empty_duplicate_and_wrong_provider() {
    let missing = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .build()
        .unwrap()
        .memory_contract_surfaces()
        .err()
        .expect("a missing fixture must fail")
        .to_string();
    assert!(
        missing.contains(
            "neither a weights-free contract-surface fixture nor a resident-only witness"
        ),
        "{missing}"
    );

    for (surface_specs, expected) in [
        (
            empty_surface_specs as fn() -> Vec<MemoryContractSurfaceSpec>,
            "publishes no surface selectors",
        ),
        (duplicate_surface_specs, "repeats surface selector"),
        (mismatched_surface_spec, "does not match its LoadSpec"),
    ] {
        let error = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(MemoryContractFixtureRegistration {
                provider_id: "dummy_weights_free_route",
                contract: weights_free_fixture_contract,
                surface_specs,
            })
            .build()
            .unwrap()
            .memory_contract_surfaces()
            .err()
            .expect("an invalid surface inventory must fail")
            .to_string();
        assert!(error.contains(expected), "{error}");
    }

    let wrong_provider = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(MemoryContractFixtureRegistration {
            provider_id: "dummy_weights_free_route",
            contract: wrong_provider_fixture_contract,
            surface_specs: mlx_memory_contract_surface_specs,
        })
        .build()
        .unwrap()
        .memory_contract_surfaces()
        .err()
        .expect("a wrong-provider fixture must fail")
        .to_string();
    assert!(
        wrong_provider.contains("returned contract for 'wrong_provider'"),
        "{wrong_provider}"
    );
}

#[test]
fn resident_only_witnesses_are_explicit_excluded_and_mutation_checked() {
    let orphan = ProviderRegistryBuilder::new()
        .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
        .build()
        .err()
        .expect("a resident-only witness must pair with a memory strategy")
        .to_string();
    assert!(
        orphan.contains("has no matching memory strategy"),
        "{orphan}"
    );

    let registry = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
        .build()
        .unwrap();
    assert_eq!(
        registry.resident_only_memory_contract_registrations().len(),
        1
    );
    assert!(registry.memory_contract_surfaces().unwrap().is_empty());

    let overlap = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
        .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
        .build()
        .err()
        .expect("a route cannot be both enumerated and resident-only")
        .to_string();
    assert!(overlap.contains("has both a contract-surface fixture and a resident-only witness"));

    let duplicate = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
        .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
        .build()
        .err()
        .expect("resident-only witnesses must be unique")
        .to_string();
    assert!(
        duplicate.contains("duplicate resident-only memory-contract witness provider id"),
        "{duplicate}"
    );

    let mutated = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
        .register_resident_only_memory_contract(ResidentOnlyMemoryContractRegistration {
            contract: false_resident_only_contract,
            ..DUMMY_RESIDENT_ONLY_WITNESS
        })
        .build()
        .unwrap()
        .memory_contract_surfaces()
        .err()
        .expect("an optimized rung cannot hide behind a resident-only witness")
        .to_string();
    assert!(
        mutated.contains("exposes BoundedDecode as Implemented"),
        "{mutated}"
    );
}

#[test]
fn unmatched_memory_strategy_requires_explicit_composed_route_registration() {
    let error = ProviderRegistryBuilder::new()
        .register_memory_strategy(DUMMY_COMPOSED_MEMORY_REGISTRATION)
        .build()
        .err()
        .expect("an ordinary memory registration must still match a generator");
    assert_eq!(
        error.to_string(),
        "memory-strategy contract 'dummy_composed_route' has no matching generator registration"
    );

    let registry = ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(DUMMY_COMPOSED_MEMORY_REGISTRATION)
        .build()
        .expect("an explicitly composed route is a valid memory-contract owner");
    let contract = registry
        .memory_strategy_contract(
            "dummy_composed_route",
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        )
        .expect("the composed route resolves")
        .expect("the composed route has a contract");
    assert_eq!(contract.provider_id, "dummy_composed_route");
    assert_eq!(
        registry
            .activation_memory_bytes_1024("dummy_composed_route")
            .expect("a known composed route has a truthful unmeasured activation state"),
        None
    );
    assert!(registry
        .activation_memory_bytes_1024("unknown_composed_route")
        .is_err());
}

/// A tier the platform declared unimplemented is rejected at the load boundary — loudly, naming
/// the tier, the id, and the platform's reason (epic 11037 SC#5: a quant tier is a creative
/// choice, never silently substituted). `dummy_load` would otherwise *succeed*, so this pins that
/// the guard fires ahead of the provider rather than leaving the coercion to the backend.
#[test]
fn rejected_quant_tier_fails_loudly_at_load() {
    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        })
        .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
        .build()
        .unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    spec.quantize = Some(Quant::Nvfp4);
    let error = registry
        .load("dummy_test_model", &spec)
        .err()
        .expect("a rejected tier must not reach the provider");
    match error {
        Error::Unsupported(message) => assert_eq!(
            message,
            "quant tier Nvfp4 is not implemented by this runtime's backend \
                 (requested for 'dummy_test_model'): no FP4 quantizer on this backend. Refusing to \
                 load rather than silently serving a different tier's numerics."
        ),
        other => panic!("a rejected quant tier is a capability gap, got {other:?}"),
    }
}

#[test]
fn rowwise_fp8_requires_exact_provider_capability_not_q8_width() {
    fn q8_only() -> ModelDescriptor {
        let mut descriptor = dummy_candle_descriptor();
        descriptor.capabilities.supported_quants = &[Quant::Q8];
        descriptor
    }
    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: q8_only,
            load: dummy_load,
            footprint: None,
        })
        .build()
        .unwrap();
    let q8 = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q8);
    assert!(registry.load("dummy_candle_test_model", &q8).is_ok());
    let fp8 = q8.with_quant(Quant::Fp8Rowwise);
    for error in [
        registry.load("dummy_candle_test_model", &fp8).err(),
        registry.footprint("dummy_candle_test_model", &fp8).err(),
        registry
            .memory_strategy_contract("dummy_candle_test_model", &fp8)
            .err(),
    ] {
        assert!(
            matches!(error,Some(Error::Unsupported(message)) if message.contains("fp8r")&&message.contains("not Q8"))
        );
    }
}

#[test]
fn flux3_fp8_is_not_advertised_until_explicitly_listed() {
    fn without_fp8() -> ModelDescriptor {
        ModelDescriptor {
            id: crate::flux3::DISTILLED_MODEL_ID,
            family: "flux3",
            backend: "candle",
            ..dummy_descriptor()
        }
    }
    fn with_fp8() -> ModelDescriptor {
        let mut descriptor = without_fp8();
        descriptor.capabilities.supported_quants = &[Quant::Fp8Rowwise];
        descriptor
    }
    fn load_fp8(_: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen { desc: with_fp8() }))
    }
    let spec =
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Fp8Rowwise);
    let unlisted = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: without_fp8,
            load: load_fp8,
            footprint: None,
        })
        .build()
        .unwrap();
    assert!(matches!(
        unlisted.load(crate::flux3::DISTILLED_MODEL_ID, &spec),
        Err(Error::Unsupported(_))
    ));
    let listed = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: with_fp8,
            load: load_fp8,
            footprint: None,
        })
        .build()
        .unwrap();
    assert!(listed.load(crate::flux3::DISTILLED_MODEL_ID, &spec).is_ok());
    assert_eq!(
        (listed.generators().next().unwrap().descriptor)()
            .capabilities
            .supported_quants,
        &[Quant::Fp8Rowwise]
    );
}

/// A model-specific AR execution choice cannot be silently ignored by another provider.
#[test]
fn yue2_ar_mode_is_scoped_to_yue2() {
    let registry = dummy_registry();
    let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert_eq!(base.yue2_ar_mode, crate::Yue2ArMode::Native);
    assert!(registry.load("dummy_test_model", &base).is_ok());
    let fp8 = base
        .clone()
        .with_yue2_ar_mode(crate::Yue2ArMode::ExperimentalFp8);
    let err = registry.load("dummy_test_model", &fp8).err().unwrap();
    assert!(
        matches!(&err, Error::Unsupported(message)
        if message.contains("YuE2 AR mode") && message.contains("dummy_test_model")),
        "{err}"
    );
    assert_eq!(base.yue2_compute_policy, crate::Yue2ComputePolicy::Legacy);
    let strict = base.with_yue2_compute_policy(crate::Yue2ComputePolicy::Bf16);
    let err = registry.load("dummy_test_model", &strict).err().unwrap();
    assert!(
        matches!(&err, Error::Unsupported(message)
        if message.contains("YuE2 compute policy") && message.contains("dummy_test_model")),
        "{err}"
    );
}

/// The guard is scoped to the declared tiers: an unrejected tier (and a dense, `None` load) still
/// reaches the provider untouched, and a catalog that declares nothing rejects nothing.
#[test]
fn unrejected_quant_tiers_still_load() {
    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        })
        .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
        .build()
        .unwrap();

    for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = quant;
        assert!(
            registry.load("dummy_test_model", &spec).is_ok(),
            "{quant:?} must still load"
        );
    }

    // A catalog whose backend implements every tier (the CUDA candle catalog) declares no
    // rejection and is unaffected.
    let permissive = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        })
        .build()
        .unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    spec.quantize = Some(Quant::Nvfp4);
    assert!(permissive.load("dummy_test_model", &spec).is_ok());
}

/// An unknown id reports as an unknown id even when the spec also carries a rejected tier — the
/// guard runs after id resolution so the caller sees the primary fault.
#[test]
fn unknown_id_wins_over_rejected_quant() {
    let registry = ProviderRegistryBuilder::new()
        .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
        .build()
        .unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    spec.quantize = Some(Quant::Nvfp4);
    let error = registry
        .load("nope", &spec)
        .err()
        .expect("unknown id must fail")
        .to_string();
    assert!(
        error.contains("no generator registered for id 'nope'"),
        "{error}"
    );
}

#[test]
fn explicit_registry_rejects_duplicate_ids_deterministically() {
    let registration = ModelRegistration {
        descriptor: dummy_descriptor,
        load: dummy_load,
        footprint: None,
    };
    let error = ProviderRegistryBuilder::new()
        .register_generator(registration)
        .register_generator(registration)
        .build()
        .err()
        .expect("duplicate registry must fail");
    assert_eq!(
        error.to_string(),
        "duplicate generator id 'dummy_test_model' in explicit registry"
    );
}

#[test]
fn checkpoint_codecs_register_once_per_catalog_and_duplicate_rows_fail_closed() {
    use crate::checkpoint_codec::{WeightEncoding, DENSE_BF16_CODEC};

    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_codec(DENSE_BF16_CODEC)
        .build()
        .expect("one baseline codec row builds");
    assert_eq!(
        registry
            .checkpoint_codecs()
            .codecs()
            .copied()
            .collect::<Vec<_>>(),
        [DENSE_BF16_CODEC]
    );
    assert_eq!(
        registry
            .checkpoint_codecs()
            .for_encoding(WeightEncoding::DenseBf16),
        Some(&DENSE_BF16_CODEC)
    );
    assert!(registry
        .checkpoint_codecs()
        .for_encoding(WeightEncoding::Fp8E4M3)
        .is_none());

    // A composed catalog that registers the same portable row twice (the withdrawn sc-20638
    // draft's duplicate codec-suite rows) is refused at build, not deduplicated silently.
    let duplicated = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_codec(DENSE_BF16_CODEC)
        .register_checkpoint_codec(DENSE_BF16_CODEC)
        .build();
    assert!(
        duplicated
            .as_ref()
            .err()
            .is_some_and(|error| error.to_string().contains("duplicate checkpoint codec id")),
        "duplicate codec rows must fail the build: {:?}",
        duplicated.as_ref().err().map(ToString::to_string)
    );

    let empty = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .build()
        .expect("a catalog without codecs still builds");
    assert!(empty.checkpoint_codecs().is_empty());
}

#[test]
fn imported_routes_are_exact_and_project_the_selected_provider() {
    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .expect("exact imported route");

    let selected = registry
        .imported_model_descriptor(
            "test",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Generate,
        )
        .expect("registered shape and operation resolve");
    assert_eq!(selected.id, "dummy_test_model");
    assert_eq!(selected.required_components, &["imported_component"]);
    assert!(registry
        .imported_model_descriptor(
            "test",
            ImportedModelSource::FusedCheckpoint,
            ImportedModelOperation::Generate,
        )
        .is_none());
    assert!(registry
        .imported_model_descriptor(
            "test",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Edit,
        )
        .is_none());
}

#[test]
fn imported_route_can_withdraw_structurally_invalid_adapter_inheritance() {
    fn adapter_descriptor() -> ModelDescriptor {
        let mut descriptor = dummy_descriptor();
        descriptor.id = "dummy_adapter_model";
        descriptor.capabilities.supports_lora = true;
        descriptor.capabilities.supports_lokr = true;
        descriptor
    }
    fn adapter_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: adapter_descriptor(),
        }))
    }

    // TWO routes on the same fixture: `Generate` withholds adapter inheritance, `Edit` grants it.
    // The contrast is what proves the gate keys off the binding rather than blanket-refusing.
    const NON_INHERITING: CheckpointBackendBindingRegistration =
        CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Mlx,
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Generate,
            provider_id: "dummy_adapter_model",
            required_components: None,
            inherit_adapters: false,
        };
    const RESTRICTED_BINDING: &[CheckpointBackendBindingRegistration] = &[
        NON_INHERITING,
        CheckpointBackendBindingRegistration {
            operation: ImportedModelOperation::Edit,
            inherit_adapters: true,
            ..NON_INHERITING
        },
    ];
    const WITHHELD: CheckpointAdapterCapabilityRegistration =
        CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: false,
        };
    const RESTRICTED_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] = &[
        WITHHELD,
        CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Edit,
            supports_adapter_inheritance: true,
            ..WITHHELD
        },
    ];
    const RESTRICTED_OPERATIONS: &[ImportedModelOperation] = &[
        ImportedModelOperation::Generate,
        ImportedModelOperation::Edit,
    ];
    let mut adapter = fixture_adapter(RESTRICTED_BINDING);
    adapter.capabilities = RESTRICTED_CAPABILITIES;
    adapter.operations = RESTRICTED_OPERATIONS;

    let registry = ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor: adapter_descriptor,
            load: adapter_load,
            footprint: None,
        })
        .register_checkpoint_adapter(adapter)
        .build()
        .expect("structurally restricted imported route");

    let ordinary = (registry.generators().next().unwrap().descriptor)();
    assert!(ordinary.capabilities.supports_lora);
    assert!(ordinary.capabilities.supports_lokr);
    let imported = registry
        .imported_model_descriptor(
            "test",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Generate,
        )
        .unwrap();
    assert!(!imported.capabilities.supports_lora);
    assert!(!imported.capabilities.supports_lokr);

    // sc-21483 (epic 11037 E6): the withdrawn capability must be *observable*. An adapter-bearing
    // request against this route is a typed capability refusal, never a silently dropped adapter.
    let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("weights")));
    spec.adapters = vec![crate::runtime::AdapterSpec::new(
        std::path::PathBuf::from("adapter.safetensors"),
        1.0,
        crate::runtime::AdapterKind::Lora,
    )];
    // The refusal is exercised through the *production* shape: resolve the route's descriptor,
    // then hand it to the provider-build seam gate (`reject_unsupported_adapters`, the same call
    // `candle-gen-krea`'s `build` makes). There is deliberately no registry-side wrapper — a
    // second gate that no load route calls would be indistinguishable from no gate at all.
    let error =
        reject_unsupported_adapters(imported.id, &imported.capabilities, spec.adapters.len())
            .expect_err("an adapter-bearing request on a non-inheriting route is refused");
    assert!(
        matches!(error, Error::Unsupported(_)),
        "the refusal must be a typed capability error, got {error:?}"
    );
    assert!(
        error.to_string().contains("does not inherit adapters"),
        "{error}"
    );

    // The same route with no adapter selected loads normally…
    reject_unsupported_adapters(imported.id, &imported.capabilities, 0)
        .expect("an adapter-free request is unaffected");

    // …and so does an adapter-bearing request against a route that DOES inherit adapters, so the
    // gate keys off the binding rather than blanket-refusing adapters. This second route is
    // registered on the SAME fixture below, and it must resolve — otherwise the assertion would
    // be vacuous.
    let inheriting = registry
        .imported_model_descriptor(
            "test",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Edit,
        )
        .expect("the inheriting route must resolve, or this assertion proves nothing");
    assert!(inheriting.capabilities.supports_lora);
    reject_unsupported_adapters(inheriting.id, &inheriting.capabilities, spec.adapters.len())
        .expect("an inheriting route admits adapters");
}

/// sc-21483: the shared refusal is capability-shaped, not route-shaped — a descriptor that
/// advertises either adapter form admits, one that advertises neither refuses.
#[test]
fn adapter_refusal_keys_off_the_advertised_capability() {
    let mut capabilities = dummy_descriptor().capabilities;
    capabilities.supports_lora = false;
    capabilities.supports_lokr = false;
    assert!(reject_unsupported_adapters("m", &capabilities, 0).is_ok());
    assert!(reject_unsupported_adapters("m", &capabilities, 1).is_err());

    capabilities.supports_lokr = true;
    assert!(reject_unsupported_adapters("m", &capabilities, 2).is_ok());
    capabilities.supports_lokr = false;
    capabilities.supports_lora = true;
    assert!(reject_unsupported_adapters("m", &capabilities, 2).is_ok());
}

const FIXTURE_DIALECTS: &[CheckpointDialectRegistration] = &[CheckpointDialectRegistration {
    id: "fixture-diffusers",
    source: ImportedModelSource::TransformerFile,
}];
const FIXTURE_SIGNATURES: &[CheckpointSignatureRegistration] = &[CheckpointSignatureRegistration {
    id: "fixture-transformer-v1",
    dialect: "fixture-diffusers",
    required_tensor_names: &["transformer.weight"],
}];
const FIXTURE_COMPONENTS: &[CheckpointComponentRegistration] = &[CheckpointComponentRegistration {
    role: "transformer",
    min_count: 1,
    max_count: 1,
}];
const FIXTURE_BASES: &[CheckpointBaseCompatibilityRegistration] =
    &[CheckpointBaseCompatibilityRegistration {
        component_role: "transformer",
        compatible_families: &["test"],
    }];
const FIXTURE_MAPPINGS: &[CheckpointCanonicalMappingRegistration] =
    &[CheckpointCanonicalMappingRegistration {
        dialect: "fixture-diffusers",
        mapping_id: "fixture-identity-v1",
        plan_driven_backends: &[],
    }];
const FIXTURE_RECOVERY: &[CheckpointConfigRecoveryRegistration] =
    &[CheckpointConfigRecoveryRegistration {
        field: "hidden-size",
        recovery_id: "fixture-tensor-shape-v1",
    }];
const FIXTURE_OPERATIONS: &[ImportedModelOperation] = &[ImportedModelOperation::Generate];
const FIXTURE_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] =
    &[CheckpointAdapterCapabilityRegistration {
        operation: ImportedModelOperation::Generate,
        inherit_provider_capabilities: true,
        supports_adapter_inheritance: true,
    }];
const FIXTURE_MLX_BINDINGS: &[CheckpointBackendBindingRegistration] =
    &[CheckpointBackendBindingRegistration {
        backend: CheckpointBackend::Mlx,
        source: ImportedModelSource::TransformerFile,
        operation: ImportedModelOperation::Generate,
        provider_id: "dummy_test_model",
        required_components: Some(&["imported_component"]),
        inherit_adapters: true,
    }];
const FIXTURE_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
    &[CheckpointBackendBindingRegistration {
        backend: CheckpointBackend::Candle,
        provider_id: "dummy_candle_test_model",
        ..FIXTURE_MLX_BINDINGS[0]
    }];
const FIXTURE_TWO_OPERATIONS: &[ImportedModelOperation] = &[
    ImportedModelOperation::Generate,
    ImportedModelOperation::Edit,
];
const FIXTURE_TWO_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] = &[
    FIXTURE_CAPABILITIES[0],
    CheckpointAdapterCapabilityRegistration {
        operation: ImportedModelOperation::Edit,
        ..FIXTURE_CAPABILITIES[0]
    },
];
const FIXTURE_CANDLE_GENERATE_AND_EDIT_BINDINGS: &[CheckpointBackendBindingRegistration] = &[
    FIXTURE_CANDLE_BINDINGS[0],
    CheckpointBackendBindingRegistration {
        operation: ImportedModelOperation::Edit,
        ..FIXTURE_CANDLE_BINDINGS[0]
    },
];
const FIXTURE_OTHER_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
    &[CheckpointBackendBindingRegistration {
        provider_id: "dummy_other_candle_model",
        ..FIXTURE_CANDLE_BINDINGS[0]
    }];
const FIXTURE_MAGE_MLX_BINDINGS: &[CheckpointBackendBindingRegistration] =
    &[CheckpointBackendBindingRegistration {
        backend: CheckpointBackend::Mlx,
        source: ImportedModelSource::TransformerDirectory,
        operation: ImportedModelOperation::Generate,
        provider_id: "mage_flow_base",
        required_components: Some(&["base_snapshot"]),
        inherit_adapters: false,
    }];
const FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
    &[CheckpointBackendBindingRegistration {
        backend: CheckpointBackend::Candle,
        provider_id: "dummy_legacy_mage_candle_model",
        ..FIXTURE_MLX_BINDINGS[0]
    }];

fn fixture_adapter(
    bindings: &'static [CheckpointBackendBindingRegistration],
) -> CheckpointAdapterRegistration {
    CheckpointAdapterRegistration {
        adapter_id: "fixture-family-v1",
        family: "test",
        compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
            family: "test",
        },
        signatures: FIXTURE_SIGNATURES,
        dialects: FIXTURE_DIALECTS,
        component_topology: FIXTURE_COMPONENTS,
        base_compatibility: FIXTURE_BASES,
        canonical_mappings: FIXTURE_MAPPINGS,
        config_recovery: FIXTURE_RECOVERY,
        eligible_backends: &[CheckpointBackend::Mlx],
        backend_bindings: bindings,
        operations: FIXTURE_OPERATIONS,
        capabilities: FIXTURE_CAPABILITIES,
    }
}

fn cross_platform_fixture_adapter(
    bindings: &'static [CheckpointBackendBindingRegistration],
) -> CheckpointAdapterRegistration {
    CheckpointAdapterRegistration {
        eligible_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
        backend_bindings: bindings,
        ..fixture_adapter(bindings)
    }
}

fn two_operation_cross_platform_fixture_adapter(
    bindings: &'static [CheckpointBackendBindingRegistration],
) -> CheckpointAdapterRegistration {
    CheckpointAdapterRegistration {
        operations: FIXTURE_TWO_OPERATIONS,
        capabilities: FIXTURE_TWO_CAPABILITIES,
        ..cross_platform_fixture_adapter(bindings)
    }
}

#[test]
fn checkpoint_adapter_is_the_authority_and_projects_legacy_import_routes() {
    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .expect("one adapter registration is a complete fixture-family addition");

    let adapter = registry.checkpoint_adapters().next().unwrap();
    assert_eq!(adapter.adapter_id, "fixture-family-v1");
    assert_eq!(adapter.signatures, FIXTURE_SIGNATURES);
    assert_eq!(adapter.dialects, FIXTURE_DIALECTS);
    assert_eq!(adapter.component_topology, FIXTURE_COMPONENTS);
    assert_eq!(adapter.base_compatibility, FIXTURE_BASES);
    assert_eq!(adapter.canonical_mappings, FIXTURE_MAPPINGS);
    assert_eq!(adapter.config_recovery, FIXTURE_RECOVERY);
    assert_eq!(adapter.operations, FIXTURE_OPERATIONS);
    assert_eq!(adapter.capabilities, FIXTURE_CAPABILITIES);

    let projected: Vec<_> = registry.imported_models().copied().collect();
    assert_eq!(
        projected,
        [ImportedModelRegistration {
            family: "test",
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Generate,
            provider_id: "dummy_test_model",
            required_components: Some(&["imported_component"]),
            inherit_adapters: true,
        }]
    );
    assert_eq!(
        registry
            .imported_model_descriptor(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
            )
            .unwrap()
            .id,
        "dummy_test_model"
    );
}

#[test]
fn mage_adapter_binds_provider_truth_and_preserves_the_exact_legacy_family() {
    let adapter = CheckpointAdapterRegistration {
        backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
        ..MAGE_FLOW_CHECKPOINT_ADAPTER
    };
    assert_eq!(adapter.family, "mage_flow");

    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(adapter)
        .build()
        .expect("provider-truth Mage family must bind the real MLX route");

    assert_eq!(
        registry.imported_models().copied().collect::<Vec<_>>(),
        [ImportedModelRegistration {
            family: "mage-flow",
            source: ImportedModelSource::TransformerDirectory,
            operation: ImportedModelOperation::Generate,
            provider_id: "mage_flow_base",
            required_components: Some(&["base_snapshot"]),
            inherit_adapters: false,
        }]
    );
}

/// sc-20644 Wan row — the dual-expert backbone is DECLARED, distinctly, and the declaration is
/// what a later inspector/plan-layer change has to satisfy.
///
/// The two experts are two single-count roles, not one role with `max_count: 2`. That
/// distinction is the whole point: a high-noise and a low-noise expert are selected per denoise
/// step and are not interchangeable, so a plan that recorded them as two instances of one role
/// could not say which is which — which is exactly the state SceneWorks' inspector is in today
/// (both map to the single path role `transformer`).
///
/// Also pins the two things about this family that are easy to get wrong: it is Candle-only, and
/// it does NOT inherit adapter flags, because `load_from_comfyui_experts` has no LoRA seam.
///
/// Failing mutations: collapse the two roles into one `transformer` with `max_count: 2`; set
/// `supports_adapter_inheritance: true`; add `CheckpointBackend::Mlx`.
#[test]
fn the_wan_adapter_declares_two_distinct_single_count_experts() {
    let topology = WAN_CHECKPOINT_ADAPTER.component_topology;
    let backbones: Vec<&CheckpointComponentRegistration> = topology
        .iter()
        .filter(|component| component.role.starts_with("transformer"))
        .collect();
    assert_eq!(
        backbones.len(),
        2,
        "Wan declares TWO backbones; got {:?}",
        topology.iter().map(|c| c.role).collect::<Vec<_>>()
    );
    for component in &backbones {
        assert_eq!(
            (component.min_count, component.max_count),
            (1, 1),
            "{}: each expert is exactly one artifact, not one role holding two",
            component.role
        );
    }
    let mut roles: Vec<&str> = backbones.iter().map(|c| c.role).collect();
    roles.sort_unstable();
    assert_eq!(roles, ["transformer-high", "transformer-low"]);

    assert_eq!(
        WAN_CHECKPOINT_ADAPTER.eligible_backends,
        [CheckpointBackend::Candle],
        "the ComfyUI Wan expert pair loads on Candle only"
    );

    // The inference half of the cross-repo spelling tie (sc-20644 review minor 8). These
    // topology roles are hyphenated; SceneWorks' plan-layer roles are underscored, and the two
    // are joined by one projection that nothing structural enforces. Pinned in the `mapping_id`
    // posture: this asserts the projection, SceneWorks asserts the projected literals are what
    // its inspector emits. Either side drifting alone turns a Wan plan into two unrecognized
    // roles.
    let project = |topology_role: &str| topology_role.replace('-', "_");
    assert_eq!(project("transformer-high"), "transformer_high");
    assert_eq!(project("transformer-low"), "transformer_low");
    assert_ne!(
        "transformer-high", "transformer_high",
        "fixture check: the two spellings genuinely differ, so the projection is not vacuous"
    );
    // The precedent this follows rather than invents — the component id both repos already
    // share is spelled the same two ways.
    assert_eq!(
        project("base-snapshot"),
        crate::runtime::BASE_SNAPSHOT_COMPONENT
    );
    assert_eq!(
        WAN_CHECKPOINT_ADAPTER
            .dialects
            .iter()
            .map(|dialect| dialect.source)
            .collect::<Vec<_>>(),
        [ImportedModelSource::ComfyUiTree]
    );
    let generate = WAN_CHECKPOINT_ADAPTER
        .capabilities
        .iter()
        .find(|capability| capability.operation == ImportedModelOperation::Generate)
        .expect("Wan binds Generate");
    assert!(
        !generate.supports_adapter_inheritance,
        "`load_from_comfyui_experts` takes no adapters, so the imported route must not \
             advertise the provider's LoRA flags"
    );
}

#[test]
fn portable_adapter_families_and_legacy_projections_are_explicit_for_every_family() {
    let identities = [
        (&KREA_2_CHECKPOINT_ADAPTER, "krea_2", "krea_2"),
        (&SDXL_CHECKPOINT_ADAPTER, "sdxl", "sdxl"),
        (&MAGE_FLOW_CHECKPOINT_ADAPTER, "mage_flow", "mage-flow"),
        (&Z_IMAGE_CHECKPOINT_ADAPTER, "z-image", "z-image"),
        (&QWEN_IMAGE_CHECKPOINT_ADAPTER, "qwen-image", "qwen-image"),
        (&FLUX2_CHECKPOINT_ADAPTER, "flux2", "flux2"),
        (&WAN_CHECKPOINT_ADAPTER, "wan", "wan-video"),
    ];

    for (adapter, portable_family, legacy_family) in identities {
        assert_eq!(adapter.family, portable_family, "{}", adapter.adapter_id);
        assert_eq!(
            adapter.compatibility_projection.family, legacy_family,
            "{}",
            adapter.adapter_id
        );
    }
    assert_eq!(
        MAGE_FLOW_CHECKPOINT_ADAPTER.base_compatibility,
        [CheckpointBaseCompatibilityRegistration {
            component_role: "base-snapshot",
            compatible_families: &["mage_flow"],
        }]
    );
}

#[test]
fn compatibility_projection_rejects_alias_collisions_and_duplicate_legacy_families() {
    let mage = CheckpointAdapterRegistration {
        backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
        ..MAGE_FLOW_CHECKPOINT_ADAPTER
    };
    let mut legacy_family_owner = fixture_adapter(FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS);
    legacy_family_owner.adapter_id = "legacy-mage-family-v1";
    legacy_family_owner.family = "mage-flow";
    legacy_family_owner.compatibility_projection.family = "legacy-mage";
    legacy_family_owner.eligible_backends = &[CheckpointBackend::Candle];

    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(mage)
        .register_checkpoint_adapter(legacy_family_owner)
        .build()
        .err()
        .expect("an explicit legacy alias cannot shadow another portable authority")
        .to_string();
    assert_eq!(
        error,
        "checkpoint-adapter compatibility family 'mage-flow' for 'mage-flow-diffusers-v1' collides with portable family owned by 'legacy-mage-family-v1'"
    );

    let first = fixture_adapter(FIXTURE_MLX_BINDINGS);
    let mut second = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
    second.adapter_id = "other-family-v1";
    second.family = "other";
    second.eligible_backends = &[CheckpointBackend::Candle];
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(first)
        .register_checkpoint_adapter(second)
        .build()
        .err()
        .expect("legacy compatibility families must be unique")
        .to_string();
    assert_eq!(
        error,
        "duplicate checkpoint-adapter compatibility family 'test'"
    );
}

#[test]
fn checkpoint_adapter_registry_rejects_duplicate_malformed_and_implementation_free_entries() {
    let duplicate = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .err()
        .expect("duplicate adapters fail")
        .to_string();
    assert!(
        duplicate.contains("duplicate checkpoint-adapter id"),
        "{duplicate}"
    );

    let implementation_free = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(&[]))
        .build()
        .err()
        .expect("implementation-free adapters fail")
        .to_string();
    assert!(
        implementation_free.contains("has no backend bindings"),
        "{implementation_free}"
    );

    let mut malformed = fixture_adapter(FIXTURE_MLX_BINDINGS);
    malformed.signatures = &[];
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(malformed)
        .build()
        .err()
        .expect("signature-free adapter fails")
        .to_string();
    assert!(error.contains("signatures must not be empty"), "{error}");

    let malformed_surfaces: [fn(&mut CheckpointAdapterRegistration); 8] = [
        |adapter| adapter.dialects = &[],
        |adapter| adapter.component_topology = &[],
        |adapter| adapter.base_compatibility = &[],
        |adapter| adapter.canonical_mappings = &[],
        |adapter| adapter.config_recovery = &[],
        |adapter| adapter.eligible_backends = &[],
        |adapter| adapter.operations = &[],
        |adapter| adapter.capabilities = &[],
    ];
    for mutate in malformed_surfaces {
        let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
        mutate(&mut adapter);
        assert!(
            ProviderRegistryBuilder::new()
                .register_generator(DUMMY_GENERATOR_REGISTRATION)
                .register_checkpoint_adapter(adapter)
                .build()
                .is_err(),
            "removing a required portable adapter surface must fail"
        );
    }

    let mut same_family = fixture_adapter(FIXTURE_MLX_BINDINGS);
    same_family.adapter_id = "fixture-family-v2";
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .register_checkpoint_adapter(same_family)
        .build()
        .err()
        .expect("two adapter ids must not claim one family")
        .to_string();
    assert!(
        error.contains("duplicate checkpoint-adapter family 'test'"),
        "{error}"
    );

    let mut orphan_operation = fixture_adapter(FIXTURE_MLX_BINDINGS);
    orphan_operation.operations = FIXTURE_TWO_OPERATIONS;
    orphan_operation.capabilities = FIXTURE_TWO_CAPABILITIES;
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(orphan_operation)
        .build()
        .err()
        .expect("a sole-backend operation without a binding must fail")
        .to_string();
    assert_eq!(
        error,
        "checkpoint-adapter 'fixture-family-v1' operation Edit has no binding on sole eligible backend 'mlx'"
    );
}

#[test]
fn checkpoint_adapter_registry_rejects_mutated_metadata_and_provider_routes() {
    const EMPTY_COMPONENTS_BINDING: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            required_components: Some(&[]),
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const DUPLICATE_COMPONENTS_BINDING: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            required_components: Some(&["same", "same"]),
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const MALFORMED_PROVIDER_BINDING: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            provider_id: "Bad Provider",
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const UNKNOWN_SOURCE_BINDING: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            source: ImportedModelSource::FusedCheckpoint,
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const WHITESPACE_TENSOR_SIGNATURE: &[CheckpointSignatureRegistration] =
        &[CheckpointSignatureRegistration {
            required_tensor_names: &["transformer bad.weight"],
            ..FIXTURE_SIGNATURES[0]
        }];
    const INVALID_CARDINALITY: &[CheckpointComponentRegistration] =
        &[CheckpointComponentRegistration {
            min_count: 0,
            ..FIXTURE_COMPONENTS[0]
        }];
    const DUPLICATE_OPERATIONS: &[ImportedModelOperation] = &[
        ImportedModelOperation::Generate,
        ImportedModelOperation::Generate,
    ];
    const DUPLICATE_BACKENDS: &[CheckpointBackend] =
        &[CheckpointBackend::Mlx, CheckpointBackend::Mlx];
    const NON_PROJECTABLE_CAPABILITY: &[CheckpointAdapterCapabilityRegistration] =
        &[CheckpointAdapterCapabilityRegistration {
            inherit_provider_capabilities: false,
            ..FIXTURE_CAPABILITIES[0]
        }];
    const NON_INHERITING_CAPABILITY: &[CheckpointAdapterCapabilityRegistration] =
        &[CheckpointAdapterCapabilityRegistration {
            supports_adapter_inheritance: false,
            ..FIXTURE_CAPABILITIES[0]
        }];

    type AdapterMutation = (&'static str, fn(&mut CheckpointAdapterRegistration));
    let mutations: [AdapterMutation; 11] = [
        ("identity", |adapter| adapter.family = "Bad Family"),
        ("compatibility projection", |adapter| {
            adapter.compatibility_projection.family = "Bad Legacy Family"
        }),
        ("tensor name", |adapter| {
            adapter.signatures = WHITESPACE_TENSOR_SIGNATURE
        }),
        ("cardinality", |adapter| {
            adapter.component_topology = INVALID_CARDINALITY
        }),
        ("eligible backend", |adapter| {
            adapter.eligible_backends = DUPLICATE_BACKENDS
        }),
        ("operation", |adapter| {
            adapter.operations = DUPLICATE_OPERATIONS
        }),
        ("capability projection", |adapter| {
            adapter.capabilities = NON_PROJECTABLE_CAPABILITY
        }),
        ("source dialect", |adapter| {
            adapter.backend_bindings = UNKNOWN_SOURCE_BINDING
        }),
        ("provider id", |adapter| {
            adapter.backend_bindings = MALFORMED_PROVIDER_BINDING
        }),
        ("empty required components", |adapter| {
            adapter.backend_bindings = EMPTY_COMPONENTS_BINDING
        }),
        ("duplicate required components", |adapter| {
            adapter.backend_bindings = DUPLICATE_COMPONENTS_BINDING
        }),
    ];
    for (surface, mutate) in mutations {
        let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
        mutate(&mut adapter);
        assert!(
            ProviderRegistryBuilder::new()
                .register_generator(DUMMY_GENERATOR_REGISTRATION)
                .register_checkpoint_adapter(adapter)
                .build()
                .is_err(),
            "mutating {surface} must fail registry construction"
        );
    }

    let mut wrong_family = fixture_adapter(FIXTURE_MLX_BINDINGS);
    wrong_family.family = "other";
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(wrong_family)
        .build()
        .err()
        .expect("a provider from another family must fail")
        .to_string();
    assert!(error.contains("does not match generator"), "{error}");

    let mut contradictory_inheritance = fixture_adapter(FIXTURE_MLX_BINDINGS);
    contradictory_inheritance.capabilities = NON_INHERITING_CAPABILITY;
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(contradictory_inheritance)
        .build()
        .err()
        .expect("binding inheritance must obey portable capability policy")
        .to_string();
    assert!(
        error.contains("inherits adapters contrary to capability policy"),
        "{error}"
    );
}

#[test]
fn checkpoint_adapter_registry_rejects_dangling_or_contradictory_metadata() {
    const UNKNOWN_PROVIDER: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            provider_id: "not_registered",
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const WRONG_BACKEND: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Candle,
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const DUPLICATE_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[FIXTURE_MLX_BINDINGS[0], FIXTURE_MLX_BINDINGS[0]];

    for (bindings, expected) in [
        (UNKNOWN_PROVIDER, "targets unregistered generator"),
        (DUPLICATE_BINDINGS, "duplicate checkpoint-adapter binding"),
    ] {
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(bindings))
            .build()
            .err()
            .expect("contradictory binding fails")
            .to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?}, got: {error}"
        );
    }
    let mut wrong_backend = fixture_adapter(WRONG_BACKEND);
    wrong_backend.eligible_backends = &[CheckpointBackend::Mlx, CheckpointBackend::Candle];
    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(wrong_backend)
        .build()
        .err()
        .expect("backend/provider mismatch fails")
        .to_string();
    assert!(
        error.contains("does not match generator backend"),
        "{error}"
    );

    const UNKNOWN_DIALECT_SIGNATURE: &[CheckpointSignatureRegistration] =
        &[CheckpointSignatureRegistration {
            dialect: "missing",
            ..FIXTURE_SIGNATURES[0]
        }];
    const UNKNOWN_DIALECT_MAPPING: &[CheckpointCanonicalMappingRegistration] =
        &[CheckpointCanonicalMappingRegistration {
            dialect: "missing",
            ..FIXTURE_MAPPINGS[0]
        }];
    const UNKNOWN_COMPONENT_BASE: &[CheckpointBaseCompatibilityRegistration] =
        &[CheckpointBaseCompatibilityRegistration {
            component_role: "missing",
            ..FIXTURE_BASES[0]
        }];
    const UNDECLARED_OPERATION_BINDING: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            operation: ImportedModelOperation::Edit,
            ..FIXTURE_MLX_BINDINGS[0]
        }];

    let mutations: [fn(&mut CheckpointAdapterRegistration); 4] = [
        |adapter| adapter.signatures = UNKNOWN_DIALECT_SIGNATURE,
        |adapter| adapter.canonical_mappings = UNKNOWN_DIALECT_MAPPING,
        |adapter| adapter.base_compatibility = UNKNOWN_COMPONENT_BASE,
        |adapter| adapter.backend_bindings = UNDECLARED_OPERATION_BINDING,
    ];
    for mutate in mutations {
        let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
        mutate(&mut adapter);
        assert!(
            ProviderRegistryBuilder::new()
                .register_generator(DUMMY_GENERATOR_REGISTRATION)
                .register_checkpoint_adapter(adapter)
                .build()
                .is_err(),
            "dangling portable metadata must fail"
        );
    }

    let error = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .err()
        .expect("a shipped eligible backend without a family binding must fail")
        .to_string();
    assert!(
        error.contains("no binding for shipped eligible backend 'candle'"),
        "{error}"
    );
}

/// A canonical mapping may only claim a plan-driven backend the adapter is eligible for, and
/// may not claim one twice. Each catalog's conformance test reads
/// `plan_driven_backends` as the exact set of backends that must ship a `LogicalKeyMapping`
/// with that id, so an ineligible or repeated entry would make the reachability proof
/// unsatisfiable (or vacuous) rather than merely untidy.
#[test]
fn canonical_mapping_plan_driven_backends_must_be_eligible_and_unique() {
    const INELIGIBLE: &[CheckpointCanonicalMappingRegistration] =
        &[CheckpointCanonicalMappingRegistration {
            // The fixture adapter is MLX-eligible only.
            plan_driven_backends: &[CheckpointBackend::Candle],
            ..FIXTURE_MAPPINGS[0]
        }];
    const REPEATED: &[CheckpointCanonicalMappingRegistration] =
        &[CheckpointCanonicalMappingRegistration {
            plan_driven_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Mlx],
            ..FIXTURE_MAPPINGS[0]
        }];
    for mapping in [INELIGIBLE, REPEATED] {
        let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
        adapter.canonical_mappings = mapping;
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(adapter)
            .build()
            .err()
            .expect("a repeated or ineligible plan-driven backend must fail the build")
            .to_string();
        assert!(
            error.contains("repeated or ineligible plan-driven backend"),
            "{error}"
        );
    }

    // And the honest declaration still builds.
    const ELIGIBLE: &[CheckpointCanonicalMappingRegistration] =
        &[CheckpointCanonicalMappingRegistration {
            plan_driven_backends: &[CheckpointBackend::Mlx],
            ..FIXTURE_MAPPINGS[0]
        }];
    let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
    adapter.canonical_mappings = ELIGIBLE;
    ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(adapter)
        .build()
        .expect("an eligible plan-driven backend builds");
}

/// The shipped posture, pinned per (adapter, dialect): which mapping id is the authority and
/// which backends actually implement it. Two facts this guards, both of which were live
/// defects before sc-20651:
///
/// * the Krea 2 `diffusers` dialect must NOT be `identity-v1` — that mapping accepts every
///   on-disk key and decodes undescribed fp8 at unit scale, silently wrong (its own doc comment
///   forbids the use); and
/// * a `mapping_id` with no implementation anywhere must say so (`plan_driven_backends: &[]`)
///   rather than reading like a backed route. The catalog tests then prove the non-empty
///   entries resolve and the empty ones have nothing masquerading behind them.
#[test]
fn shipped_canonical_mapping_posture_is_pinned_per_dialect() {
    use crate::checkpoint_codec::IdentityKeyMapping;

    /// One pinned mapping row: `(dialect, mapping_id, plan_driven_backends)`.
    type MappingRow = (&'static str, &'static str, &'static [CheckpointBackend]);

    let expected: &[(&str, &[MappingRow])] = &[
        (
            KREA_2_CHECKPOINT_ADAPTER.adapter_id,
            &[
                (
                    // Implemented on BOTH engines since sc-20651 — one dialect, one canonical
                    // mapping id, two implementations.
                    "krea-native",
                    "krea-native-to-diffusers-v1",
                    &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
                ),
                (
                    "diffusers",
                    "krea-2-diffusers-v1",
                    &[CheckpointBackend::Mlx],
                ),
            ],
        ),
        (
            SDXL_CHECKPOINT_ADAPTER.adapter_id,
            &[("ldm", "sdxl-ldm-to-diffusers-v1", &[])],
        ),
        (
            MAGE_FLOW_CHECKPOINT_ADAPTER.adapter_id,
            &[("diffusers", "identity-v1", &[])],
        ),
        (
            Z_IMAGE_CHECKPOINT_ADAPTER.adapter_id,
            &[("comfyui", "z-image-comfyui-to-diffusers-v1", &[])],
        ),
        (
            QWEN_IMAGE_CHECKPOINT_ADAPTER.adapter_id,
            &[("comfyui", "qwen-image-comfyui-to-diffusers-v1", &[])],
        ),
        (
            FLUX2_CHECKPOINT_ADAPTER.adapter_id,
            &[
                ("comfyui", "flux2-comfyui-to-diffusers-v1", &[]),
                // sc-21485: the klein universal single file plans through the Candle
                // `Flux2BflToDiffusersMapping`.
                (
                    "bfl",
                    "flux2-bfl-to-diffusers-v1",
                    &[CheckpointBackend::Candle],
                ),
            ],
        ),
        (
            WAN_CHECKPOINT_ADAPTER.adapter_id,
            &[(
                "comfyui",
                "wan-comfyui-to-diffusers-v1",
                &[CheckpointBackend::Candle],
            )],
        ),
    ];
    let shipped: &[&CheckpointAdapterRegistration] = &[
        &KREA_2_CHECKPOINT_ADAPTER,
        &SDXL_CHECKPOINT_ADAPTER,
        &MAGE_FLOW_CHECKPOINT_ADAPTER,
        &Z_IMAGE_CHECKPOINT_ADAPTER,
        &QWEN_IMAGE_CHECKPOINT_ADAPTER,
        &FLUX2_CHECKPOINT_ADAPTER,
        &WAN_CHECKPOINT_ADAPTER,
    ];
    assert_eq!(
        shipped.len(),
        expected.len(),
        "every shipped adapter's mapping posture must be pinned here"
    );
    for (adapter, (adapter_id, rows)) in shipped.iter().zip(expected) {
        assert_eq!(adapter.adapter_id, *adapter_id);
        let actual: Vec<_> = adapter
            .canonical_mappings
            .iter()
            .map(|mapping| {
                (
                    mapping.dialect,
                    mapping.mapping_id,
                    mapping.plan_driven_backends,
                )
            })
            .collect();
        let expected_rows: Vec<_> = rows.to_vec();
        assert_eq!(actual, expected_rows, "adapter {adapter_id}");
        assert!(
            !adapter
                .canonical_mappings
                .iter()
                .any(
                    |mapping| mapping.mapping_id == IdentityKeyMapping::MAPPING_ID
                        && !mapping.plan_driven_backends.is_empty()
                ),
            "adapter {adapter_id} routes a plan through `identity-v1`, which decodes \
                 undescribed fp8 at unit scale — silently wrong rather than refused"
        );
    }
}

/// sc-21485 review. A signature is an all-present name list, so `flux2-klein-bfl-v1` must be
/// discriminated by a name only the klein community export carries. A bare-keyed
/// FLUX.2-**dev** BFL file shares the fused-qkv and AdaLN names; it differs by spelling the
/// per-head norms `.scale` (and by carrying `guidance_in.*`, which no all-present list can
/// assert the absence of).
///
/// Mutation witness: drop `double_blocks.0.img_attn.norm.query_norm.weight` from the
/// registration and the dev key set below starts claiming the klein signature — the
/// misrouting this test forbids.
#[test]
fn the_klein_bfl_signature_does_not_claim_a_bare_keyed_dev_file() {
    /// The rule a signature declares: every required name is present in the file's key set.
    fn claims(signature: &CheckpointSignatureRegistration, keys: &[&str]) -> bool {
        signature
            .required_tensor_names
            .iter()
            .all(|name| keys.contains(name))
    }
    let klein_signature = FLUX2_CHECKPOINT_ADAPTER
        .signatures
        .iter()
        .find(|signature| signature.id == "flux2-klein-bfl-v1")
        .expect("the klein bfl signature is registered");

    // The klein community export (`wikeeyang/Flux2-Klein-9B-True-V2`): bare BFL keys, `.weight`
    // per-head norms, no guidance embedder.
    let klein: &[&str] = &[
        "img_in.weight",
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.img_attn.norm.query_norm.weight",
        "double_blocks.0.img_attn.norm.key_norm.weight",
        "single_blocks.0.linear1.weight",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.linear.weight",
    ];
    assert!(claims(klein_signature, klein), "klein must be claimed");

    // The BFL-official FLUX.2-dev export: the SAME fused-qkv and AdaLN names, the `.scale`
    // per-head-norm spelling, and the guidance embedder klein does not have.
    let dev: &[&str] = &[
        "img_in.weight",
        "guidance_in.in_layer.weight",
        "guidance_in.out_layer.weight",
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.img_attn.norm.query_norm.scale",
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "single_blocks.0.linear1.weight",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.linear.weight",
    ];
    assert!(
        !claims(klein_signature, dev),
        "a bare-keyed FLUX.2-dev file must NOT claim the klein signature (it would import a \
             32B dev checkpoint as flux2_klein_9b and fail only at plan time)"
    );

    // The comment on the registration must not claim a discrimination the names cannot make.
    assert!(
        klein_signature
            .required_tensor_names
            .contains(&"double_blocks.0.img_attn.norm.query_norm.weight"),
        "the klein-only per-head-norm spelling is what carries the discrimination"
    );
}

#[test]
fn checkpoint_adapter_catalog_conformance_preserves_truthful_backend_asymmetry() {
    let mlx = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .unwrap();
    let candle = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS))
        .build()
        .unwrap();
    assert_eq!(
        mlx.checkpoint_adapter_catalog_conformance_errors(&candle),
        Vec::<String>::new()
    );

    let mlx_only = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .unwrap();
    let candle_without_adapter = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(
        mlx_only.checkpoint_adapter_catalog_conformance_errors(&candle_without_adapter),
        Vec::<String>::new(),
        "an explicitly MLX-only family must not fabricate a Candle route"
    );

    let mlx_generate_only = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
            FIXTURE_MLX_BINDINGS,
        ))
        .build()
        .unwrap();
    let candle_generate_only = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
            FIXTURE_CANDLE_BINDINGS,
        ))
        .build()
        .unwrap();
    assert_eq!(
        mlx_generate_only
            .checkpoint_adapter_catalog_conformance_errors(&candle_generate_only),
        ["checkpoint-adapter family 'test' (id 'fixture-family-v1') operation Edit has no binding across eligible catalogs"]
    );

    let candle_implements_edit = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
            FIXTURE_CANDLE_GENERATE_AND_EDIT_BINDINGS,
        ))
        .build()
        .unwrap();
    assert_eq!(
        mlx_generate_only.checkpoint_adapter_catalog_conformance_errors(&candle_implements_edit),
        Vec::<String>::new(),
        "one eligible backend may omit an operation when another real binding implements it"
    );
}

#[test]
fn checkpoint_adapter_catalog_conformance_rejects_family_id_drift_deterministically() {
    let mlx = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .unwrap();

    let mut different_id = fixture_adapter(FIXTURE_CANDLE_BINDINGS);
    different_id.adapter_id = "fixture-family-v2";
    different_id.eligible_backends = &[CheckpointBackend::Candle];
    let candle_different_id = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(different_id)
        .build()
        .unwrap();
    let expected_id_drift = ["checkpoint-adapter family 'test' maps to different adapter ids 'fixture-family-v1' and 'fixture-family-v2' across catalogs"];
    assert_eq!(
        mlx.checkpoint_adapter_catalog_conformance_errors(&candle_different_id),
        expected_id_drift
    );
    assert_eq!(
        candle_different_id.checkpoint_adapter_catalog_conformance_errors(&mlx),
        expected_id_drift,
        "family/id drift diagnostics must not depend on catalog argument order"
    );

    let mut different_family = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
    different_family.family = "other";
    different_family.eligible_backends = &[CheckpointBackend::Candle];
    let candle_different_family = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(different_family)
        .build()
        .unwrap();
    let expected_family_drift = ["checkpoint-adapter id 'fixture-family-v1' maps to different families 'other' and 'test' across catalogs"];
    assert_eq!(
        mlx.checkpoint_adapter_catalog_conformance_errors(&candle_different_family),
        expected_family_drift
    );
    assert_eq!(
        candle_different_family.checkpoint_adapter_catalog_conformance_errors(&mlx),
        expected_family_drift,
        "adapter-id/family drift diagnostics must not depend on catalog argument order"
    );

    let mage_mlx = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(CheckpointAdapterRegistration {
            backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
            ..MAGE_FLOW_CHECKPOINT_ADAPTER
        })
        .build()
        .unwrap();
    let mut normalized_legacy_family = fixture_adapter(FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS);
    normalized_legacy_family.adapter_id = MAGE_FLOW_CHECKPOINT_ADAPTER.adapter_id;
    normalized_legacy_family.family = "mage-flow";
    normalized_legacy_family.compatibility_projection.family = "mage-flow";
    normalized_legacy_family.eligible_backends = &[CheckpointBackend::Candle];
    let mage_candle_with_normalized_family = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(normalized_legacy_family)
        .build()
        .unwrap();
    let expected_normalization_drift = ["checkpoint-adapter id 'mage-flow-diffusers-v1' maps to different families 'mage-flow' and 'mage_flow' across catalogs"];
    assert_eq!(
        mage_mlx.checkpoint_adapter_catalog_conformance_errors(&mage_candle_with_normalized_family),
        expected_normalization_drift,
        "hyphen/underscore normalization must not hide portable-family drift"
    );
    assert_eq!(
        mage_candle_with_normalized_family.checkpoint_adapter_catalog_conformance_errors(&mage_mlx),
        expected_normalization_drift,
        "normalization-drift diagnostics must not depend on catalog argument order"
    );

    let mut mlx_shared_projection = fixture_adapter(FIXTURE_MLX_BINDINGS);
    mlx_shared_projection.compatibility_projection.family = "shared-legacy-family";
    let mlx_shared_projection = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(mlx_shared_projection)
        .build()
        .unwrap();
    let mut candle_shared_projection = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
    candle_shared_projection.adapter_id = "other-family-v1";
    candle_shared_projection.family = "other";
    candle_shared_projection.compatibility_projection.family = "shared-legacy-family";
    candle_shared_projection.eligible_backends = &[CheckpointBackend::Candle];
    let candle_shared_projection = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(candle_shared_projection)
        .build()
        .unwrap();
    let expected_alias_ownership_drift = ["checkpoint-adapter compatibility family 'shared-legacy-family' maps to different portable authorities 'other' (id 'other-family-v1') and 'test' (id 'fixture-family-v1') across catalogs"];
    assert_eq!(
        mlx_shared_projection
            .checkpoint_adapter_catalog_conformance_errors(&candle_shared_projection),
        expected_alias_ownership_drift
    );
    assert_eq!(
        candle_shared_projection
            .checkpoint_adapter_catalog_conformance_errors(&mlx_shared_projection),
        expected_alias_ownership_drift,
        "compatibility-family ownership diagnostics must not depend on catalog argument order"
    );
}

#[test]
fn checkpoint_adapter_catalog_conformance_detects_metadata_and_binding_mutations() {
    const MUTATED_RECOVERY: &[CheckpointConfigRecoveryRegistration] =
        &[CheckpointConfigRecoveryRegistration {
            field: "hidden-size",
            recovery_id: "mutated-tensor-shape-v1",
        }];

    let mlx = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
        .build()
        .unwrap();
    let mut mutated = cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS);
    mutated.config_recovery = MUTATED_RECOVERY;
    let candle_with_metadata_drift = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(mutated)
        .build()
        .unwrap();
    let expected_metadata_drift = vec![
        "checkpoint-adapter family 'test' (id 'fixture-family-v1') portable metadata differs across catalogs"
            .to_owned(),
    ];
    assert_eq!(
        mlx.checkpoint_adapter_catalog_conformance_errors(&candle_with_metadata_drift),
        expected_metadata_drift
    );
    assert_eq!(
        candle_with_metadata_drift.checkpoint_adapter_catalog_conformance_errors(&mlx),
        expected_metadata_drift
    );

    let mut mutated_projection = cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS);
    mutated_projection.compatibility_projection.family = "test-legacy";
    let candle_with_projection_drift = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .register_checkpoint_adapter(mutated_projection)
        .build()
        .unwrap();
    assert_eq!(
        mlx.checkpoint_adapter_catalog_conformance_errors(&candle_with_projection_drift),
        expected_metadata_drift,
        "legacy projection drift is portable-metadata drift"
    );
    assert_eq!(
        candle_with_projection_drift.checkpoint_adapter_catalog_conformance_errors(&mlx),
        expected_metadata_drift,
        "projection-drift diagnostics must not depend on catalog argument order"
    );

    let candle_without_adapter = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
        .build()
        .unwrap();
    let errors = mlx.checkpoint_adapter_catalog_conformance_errors(&candle_without_adapter);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("missing from a catalog")),
        "{errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("no binding for shipped eligible backend 'candle'")),
        "{errors:?}"
    );
}

#[test]
fn unknown_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry.load("no_such_model", &spec).is_err());
}

#[test]
fn dummy_appears_in_iteration() {
    assert!(dummy_registry()
        .generators()
        .any(|r| (r.descriptor)().id == "dummy_test_model"));
}

#[test]
fn macro_delegated_generator_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let g = registry
        .load("dummy_delegated_test_model", &spec)
        .expect("dummy is registered");
    assert_eq!(g.descriptor().id, "dummy_delegated_test_model");
    g.validate(&GenerationRequest::default()).unwrap();
}

#[test]
fn macro_registered_trainer_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let t = registry
        .load_trainer("dummy_test_trainer", &spec)
        .expect("dummy trainer is registered");
    assert_eq!(t.descriptor().id, "dummy_test_trainer");
    assert!(registry
        .trainers()
        .any(|r| (r.descriptor)().id == "dummy_test_trainer"));
}

#[test]
fn multiple_generator_constants_compose() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    for id in ["dummy_multi_gen_a", "dummy_multi_gen_b"] {
        let g = registry
            .load(id, &spec)
            .unwrap_or_else(|_| panic!("{id} is registered"));
        assert_eq!(g.descriptor().id, id);
    }
}

#[test]
fn multiple_trainer_constants_compose() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    for id in ["dummy_multi_trainer_a", "dummy_multi_trainer_b"] {
        let t = registry
            .load_trainer(id, &spec)
            .unwrap_or_else(|_| panic!("{id} is registered"));
        assert_eq!(t.descriptor().id, id);
    }
}

#[test]
fn captioner_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let c = registry
        .load_captioner("dummy_test_captioner", &spec)
        .expect("dummy captioner is registered");
    assert_eq!(c.descriptor().id, "dummy_test_captioner");
}

#[test]
fn unknown_captioner_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry.load_captioner("no_such_captioner", &spec).is_err());
}

#[test]
fn dummy_captioner_appears_in_iteration() {
    assert!(dummy_registry()
        .captioners()
        .any(|r| (r.descriptor)().id == "dummy_test_captioner"));
}

#[test]
fn transcriber_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let t = registry
        .load_transcriber("dummy_test_transcriber", &spec)
        .expect("dummy transcriber is registered");
    assert_eq!(t.descriptor().id, "dummy_test_transcriber");
}

#[test]
fn unknown_transcriber_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_transcriber("no_such_transcriber", &spec)
        .is_err());
}

#[test]
fn dummy_transcriber_appears_in_iteration() {
    assert!(dummy_registry()
        .transcribers()
        .any(|r| (r.descriptor)().id == "dummy_test_transcriber"));
}

struct DummyTextEmbedder {
    desc: TextEmbedderDescriptor,
}

impl TextEmbedder for DummyTextEmbedder {
    fn descriptor(&self) -> &TextEmbedderDescriptor {
        &self.desc
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        Ok(vec![text.len() as f32, 1.0])
    }
}

fn dummy_text_embedder_descriptor() -> TextEmbedderDescriptor {
    TextEmbedderDescriptor {
        id: "dummy_test_text_embedder",
        family: "test",
        backend: "mlx",
        embedding_dim: 2,
        space: "test-space",
        mac_only: true,
    }
}

fn dummy_text_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn TextEmbedder>> {
    Ok(Box::new(DummyTextEmbedder {
        desc: dummy_text_embedder_descriptor(),
    }))
}

crate::register_text_embedder! {
    const DUMMY_TEXT_EMBEDDER_REGISTRATION =
        dummy_text_embedder_descriptor => dummy_text_embedder_load
}

#[test]
fn text_embedder_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let e = registry
        .load_text_embedder("dummy_test_text_embedder", &spec)
        .expect("dummy text embedder is registered");
    assert_eq!(e.descriptor().id, "dummy_test_text_embedder");
    assert_eq!(e.embed_text("clip").unwrap(), vec![4.0, 1.0]);
    assert_eq!(
        e.embed_text_batch(&["a", "abcd"]).unwrap(),
        vec![vec![1.0, 1.0], vec![4.0, 1.0]]
    );
}

#[test]
fn unknown_text_embedder_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_text_embedder("no_such_text_embedder", &spec)
        .is_err());
}

#[test]
fn dummy_text_embedder_appears_in_iteration() {
    assert!(dummy_registry()
        .text_embedders()
        .any(|r| (r.descriptor)().id == "dummy_test_text_embedder"));
}

struct DummyImageEmbedder {
    desc: ImageEmbedderDescriptor,
}

impl ImageEmbedder for DummyImageEmbedder {
    fn descriptor(&self) -> &ImageEmbedderDescriptor {
        &self.desc
    }

    fn embed(&self, image: &Image) -> Result<Vec<f32>> {
        Ok(vec![image.width as f32, image.height as f32])
    }
}

fn dummy_image_embedder_descriptor() -> ImageEmbedderDescriptor {
    ImageEmbedderDescriptor {
        id: "dummy_test_image_embedder",
        family: "test",
        backend: "mlx",
        embedding_dim: 2,
        space: "test-space",
        mac_only: true,
    }
}

fn dummy_image_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn ImageEmbedder>> {
    Ok(Box::new(DummyImageEmbedder {
        desc: dummy_image_embedder_descriptor(),
    }))
}

crate::register_image_embedder! {
    const DUMMY_IMAGE_EMBEDDER_REGISTRATION =
        dummy_image_embedder_descriptor => dummy_image_embedder_load
}

#[test]
fn image_embedder_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let e = registry
        .load_image_embedder("dummy_test_image_embedder", &spec)
        .expect("dummy image embedder is registered");
    assert_eq!(e.descriptor().id, "dummy_test_image_embedder");
    let img = Image {
        width: 7,
        height: 3,
        pixels: vec![0; 7 * 3 * 3],
    };
    assert_eq!(e.embed(&img).unwrap(), vec![7.0, 3.0]);
}

#[test]
fn unknown_image_embedder_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_image_embedder("no_such_image_embedder", &spec)
        .is_err());
}

#[test]
fn dummy_image_embedder_appears_in_iteration() {
    assert!(dummy_registry()
        .image_embedders()
        .any(|r| (r.descriptor)().id == "dummy_test_image_embedder"));
}

// ---- audio transforms (sc-12839) --------------------------------------------------------

/// A weights-free stub audio transform: `apply` returns `out_tracks` copies of the source clip
/// (one for the single-output kinds, one per stem for separation), retargeted to the request's
/// sample rate — enough to exercise register→resolve→apply for all three shapes without a tensor
/// backend.
struct DummyAudioTransform {
    desc: AudioTransformDescriptor,
    out_tracks: usize,
}

impl AudioTransform for DummyAudioTransform {
    fn descriptor(&self) -> &AudioTransformDescriptor {
        &self.desc
    }
    fn validate(&self, _req: &AudioTransformRequest) -> Result<()> {
        Ok(())
    }
    fn apply(
        &self,
        req: &AudioTransformRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<AudioTrack>> {
        let rate = match req.target {
            AudioTarget::Preserve => req.audio.sample_rate,
            AudioTarget::SampleRate(r) => r,
        };
        Ok(vec![
            AudioTrack {
                sample_rate: rate,
                ..req.audio.clone()
            };
            self.out_tracks
        ])
    }
}

fn dummy_voice_conversion_descriptor() -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id: "dummy_voice_conversion",
        family: "audio",
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind: AudioTransformKind::VoiceConversion,
            ..Default::default()
        },
    }
}

fn dummy_stem_separation_descriptor() -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id: "dummy_stem_separation",
        family: "audio",
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind: AudioTransformKind::StemSeparation,
            stem_count: 4,
            ..Default::default()
        },
    }
}

fn dummy_super_resolution_descriptor() -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id: "dummy_super_resolution",
        family: "audio",
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind: AudioTransformKind::SuperResolution,
            is_diffusion: true,
            supports_resample: true,
            ..Default::default()
        },
    }
}

fn dummy_voice_conversion_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
    Ok(Box::new(DummyAudioTransform {
        desc: dummy_voice_conversion_descriptor(),
        out_tracks: 1,
    }))
}

fn dummy_stem_separation_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
    Ok(Box::new(DummyAudioTransform {
        desc: dummy_stem_separation_descriptor(),
        out_tracks: 4,
    }))
}

fn dummy_super_resolution_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
    Ok(Box::new(DummyAudioTransform {
        desc: dummy_super_resolution_descriptor(),
        out_tracks: 1,
    }))
}

crate::register_audio_transform! {
    const DUMMY_VOICE_CONVERSION_REGISTRATION =
        dummy_voice_conversion_descriptor => dummy_voice_conversion_load
}
crate::register_audio_transform! {
    const DUMMY_STEM_SEPARATION_REGISTRATION =
        dummy_stem_separation_descriptor => dummy_stem_separation_load
}
crate::register_audio_transform! {
    const DUMMY_SUPER_RESOLUTION_REGISTRATION =
        dummy_super_resolution_descriptor => dummy_super_resolution_load
}

fn audio_transform_registry() -> ProviderRegistry {
    ProviderRegistryBuilder::new()
        .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
        .register_audio_transform(DUMMY_STEM_SEPARATION_REGISTRATION)
        .register_audio_transform(DUMMY_SUPER_RESOLUTION_REGISTRATION)
        .build()
        .unwrap()
}

#[test]
fn audio_transform_registry_resolves_by_id() {
    let registry = audio_transform_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let t = registry
        .load_audio_transform("dummy_voice_conversion", &spec)
        .expect("dummy voice conversion is registered");
    assert_eq!(t.descriptor().id, "dummy_voice_conversion");
}

#[test]
fn unknown_audio_transform_id_errors() {
    let registry = audio_transform_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_audio_transform("no_such_audio_transform", &spec)
        .is_err());
}

#[test]
fn audio_transforms_appear_in_iteration() {
    let registry = audio_transform_registry();
    assert_eq!(registry.audio_transforms().len(), 3);
    assert!(registry
        .audio_transforms()
        .any(|r| (r.descriptor)().id == "dummy_stem_separation"));
}

#[test]
fn duplicate_audio_transform_id_is_rejected() {
    let err = ProviderRegistryBuilder::new()
        .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
        .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
        .build()
        .err()
        .expect("duplicate audio transform id must fail");
    assert_eq!(
        err.to_string(),
        "duplicate audio transform id 'dummy_voice_conversion' in explicit registry"
    );
}

#[test]
fn audio_transform_descriptors_pass_conformance() {
    assert!(audio_transform_registry()
        .descriptor_conformance_errors()
        .is_empty());
}

/// The sc-12839 acceptance path end to end, weights-free: register the three non-prompt
/// audio→audio shapes, then resolve + `apply` each and assert the output shape.
#[test]
fn all_three_audio_transform_shapes_resolve_and_apply() {
    let registry = audio_transform_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let source = dummy_audio_track(1024, 16_000);

    // audio→audio: voice conversion, rate-preserving, single output.
    let vc = registry
        .load_audio_transform("dummy_voice_conversion", &spec)
        .unwrap();
    let converted = vc
        .apply(
            &AudioTransformRequest {
                audio: source.clone(),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].sample_rate, 16_000);

    // audio→Vec<audio>: stem separation, multi output.
    let stems = registry
        .load_audio_transform("dummy_stem_separation", &spec)
        .unwrap();
    let separated = stems
        .apply(
            &AudioTransformRequest {
                audio: source.clone(),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(separated.len(), 4);
    assert_eq!(
        stems.descriptor().capabilities.stem_count as usize,
        separated.len()
    );

    // audio→audio: super-resolution / bandwidth extension to a higher rate, single output.
    let sr = registry
        .load_audio_transform("dummy_super_resolution", &spec)
        .unwrap();
    let restored = sr
        .apply(
            &AudioTransformRequest {
                audio: source,
                target: AudioTarget::SampleRate(48_000),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].sample_rate, 48_000);
}

fn dummy_audio_track(samples: usize, rate: u32) -> AudioTrack {
    AudioTrack {
        samples: vec![0.0; samples],
        sample_rate: rate,
        channels: 1,
        ..Default::default()
    }
}

// Malformed audio-transform descriptors exercising the kind/stem_count coherence guard's
// *rejection* branches (an inverted condition would otherwise pass the whole suite, since the
// positive test only asserts an empty sweep). The `load` fn is never invoked by the sweep.
fn bad_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
    Ok(Box::new(DummyAudioTransform {
        desc: dummy_voice_conversion_descriptor(),
        out_tracks: 1,
    }))
}

fn separator_stem_count(id: &'static str, stem_count: u16) -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id,
        family: "audio",
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind: AudioTransformKind::StemSeparation,
            stem_count,
            ..Default::default()
        },
    }
}
fn single_output_stem_count(
    id: &'static str,
    kind: AudioTransformKind,
) -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id,
        family: "audio",
        backend: "candle",
        capabilities: AudioTransformCapabilities {
            kind,
            stem_count: 3,
            ..Default::default()
        },
    }
}

fn bad_stems_zero_descriptor() -> AudioTransformDescriptor {
    separator_stem_count("bad_stems_zero", 0)
}
fn bad_stems_one_descriptor() -> AudioTransformDescriptor {
    separator_stem_count("bad_stems_one", 1)
}
fn bad_vc_stems_descriptor() -> AudioTransformDescriptor {
    single_output_stem_count("bad_vc_stems", AudioTransformKind::VoiceConversion)
}
fn bad_sr_stems_descriptor() -> AudioTransformDescriptor {
    single_output_stem_count("bad_sr_stems", AudioTransformKind::SuperResolution)
}

crate::register_audio_transform! {
    const BAD_STEMS_ZERO_REGISTRATION = bad_stems_zero_descriptor => bad_load
}
crate::register_audio_transform! {
    const BAD_STEMS_ONE_REGISTRATION = bad_stems_one_descriptor => bad_load
}
crate::register_audio_transform! {
    const BAD_VC_STEMS_REGISTRATION = bad_vc_stems_descriptor => bad_load
}
crate::register_audio_transform! {
    const BAD_SR_STEMS_REGISTRATION = bad_sr_stems_descriptor => bad_load
}

#[test]
fn audio_transform_kind_stem_count_incoherence_is_rejected() {
    let errs = ProviderRegistryBuilder::new()
        .register_audio_transform(BAD_STEMS_ZERO_REGISTRATION)
        .register_audio_transform(BAD_STEMS_ONE_REGISTRATION)
        .register_audio_transform(BAD_VC_STEMS_REGISTRATION)
        .register_audio_transform(BAD_SR_STEMS_REGISTRATION)
        .build()
        .unwrap()
        .descriptor_conformance_errors();
    let has = |needle: &str| errs.iter().any(|e| e.contains(needle));

    // A separator advertising < 2 stems (0 and 1) is rejected with the specific message.
    assert!(
        has("audio transform 'bad_stems_zero': StemSeparation advertises stem_count 0 (a separator must produce ≥ 2 stems)"),
        "{errs:?}"
    );
    assert!(
        has("audio transform 'bad_stems_one': StemSeparation advertises stem_count 1 (a separator must produce ≥ 2 stems)"),
        "{errs:?}"
    );
    // A single-output kind advertising any stems is rejected.
    assert!(
        has("audio transform 'bad_vc_stems': VoiceConversion advertises stem_count 3 — only StemSeparation produces stems"),
        "{errs:?}"
    );
    assert!(
        has("audio transform 'bad_sr_stems': SuperResolution advertises stem_count 3 — only StemSeparation produces stems"),
        "{errs:?}"
    );
}

struct DummyVoiceEmbedder {
    desc: VoiceEmbedderDescriptor,
}

impl VoiceEmbedder for DummyVoiceEmbedder {
    fn descriptor(&self) -> &VoiceEmbedderDescriptor {
        &self.desc
    }

    fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding> {
        Ok(vec![audio.samples.len() as f32; self.desc.embedding_dim])
    }
}

fn dummy_voice_embedder_descriptor() -> VoiceEmbedderDescriptor {
    VoiceEmbedderDescriptor {
        id: "dummy_test_voice_embedder",
        family: "voice",
        backend: "candle",
        embedding_dim: 4,
        mac_only: false,
    }
}

fn dummy_voice_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn VoiceEmbedder>> {
    Ok(Box::new(DummyVoiceEmbedder {
        desc: dummy_voice_embedder_descriptor(),
    }))
}

crate::register_voice_embedder! {
    const DUMMY_VOICE_EMBEDDER_REGISTRATION =
        dummy_voice_embedder_descriptor => dummy_voice_embedder_load
}

fn dummy_audio(samples: usize) -> AudioTrack {
    AudioTrack {
        samples: vec![0.0; samples],
        sample_rate: 24_000,
        channels: 1,
        ..Default::default()
    }
}

#[test]
fn voice_embedder_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let e = registry
        .load_voice_embedder("dummy_test_voice_embedder", &spec)
        .expect("dummy voice embedder is registered");
    assert_eq!(e.descriptor().id, "dummy_test_voice_embedder");
    assert_eq!(e.embed(&dummy_audio(3)).unwrap(), vec![3.0; 4]);
}

#[test]
fn unknown_voice_embedder_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_voice_embedder("no_such_voice_embedder", &spec)
        .is_err());
}

#[test]
fn dummy_voice_embedder_appears_in_iteration() {
    assert!(dummy_registry()
        .voice_embedders()
        .any(|r| (r.descriptor)().id == "dummy_test_voice_embedder"));
}

// ---- audio embedders (sc-12851) ---------------------------------------------------------

/// A weights-free joint audio-text embedder: both `embed` and `embed_text` return a **one-hot**
/// unit vector, keyed off the audio clip's length (a proxy for its semantic "category") and off
/// the text's length respectively. That is deliberately enough structure to drive the DoD
/// cross-modal ranking test without a tensor backend: a text query lands on the one-hot index of
/// exactly one clip, so cosine ranks that clip first — and the test FAILS if the stub ignored the
/// audio (every clip identical) or if the audio/text vectors were not the same length (not joint).
struct DummyAudioEmbedder {
    desc: AudioEmbedderDescriptor,
}

impl DummyAudioEmbedder {
    fn one_hot(&self, index: usize) -> Vec<f32> {
        let mut v = vec![0.0; self.desc.embedding_dim];
        v[index % self.desc.embedding_dim] = 1.0;
        v
    }
}

impl AudioEmbedder for DummyAudioEmbedder {
    fn descriptor(&self) -> &AudioEmbedderDescriptor {
        &self.desc
    }
    fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>> {
        Ok(self.one_hot(audio.samples.len()))
    }
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.one_hot(text.len()))
    }
}

fn dummy_audio_embedder_descriptor() -> AudioEmbedderDescriptor {
    AudioEmbedderDescriptor {
        id: "dummy_test_audio_embedder",
        family: "audio-embed",
        backend: "candle",
        embedding_dim: 4,
        space: "test-space",
        mac_only: false,
    }
}

fn dummy_audio_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn AudioEmbedder>> {
    Ok(Box::new(DummyAudioEmbedder {
        desc: dummy_audio_embedder_descriptor(),
    }))
}

crate::register_audio_embedder! {
    const DUMMY_AUDIO_EMBEDDER_REGISTRATION =
        dummy_audio_embedder_descriptor => dummy_audio_embedder_load
}

#[test]
fn audio_embedder_registry_resolves_by_id() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let e = registry
        .load_audio_embedder("dummy_test_audio_embedder", &spec)
        .expect("dummy audio embedder is registered");
    assert_eq!(e.descriptor().id, "dummy_test_audio_embedder");
    // audio and text embeddings share the joint dim.
    assert_eq!(e.embed(&dummy_audio(1)).unwrap().len(), 4);
    assert_eq!(e.embed_text("q").unwrap().len(), 4);
}

#[test]
fn unknown_audio_embedder_id_errors() {
    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert!(registry
        .load_audio_embedder("no_such_audio_embedder", &spec)
        .is_err());
}

#[test]
fn dummy_audio_embedder_appears_in_iteration() {
    assert!(dummy_registry()
        .audio_embedders()
        .any(|r| (r.descriptor)().id == "dummy_test_audio_embedder"));
}

/// The sc-12851 acceptance path end to end, weights-free: resolve the joint embedder, embed a
/// SET of audio clips spanning "categories", embed a TEXT query, and assert the semantically
/// matching clip ranks HIGHEST by cosine over the others — the cross-modal retrieval the DoD
/// pins. Designed to fail if the embedder ignored the audio (all clips equidistant) or the audio
/// and text vectors were not in one joint space.
#[test]
fn audio_text_query_ranks_the_matching_clip_highest() {
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let e = registry
        .load_audio_embedder("dummy_test_audio_embedder", &spec)
        .expect("registered");

    // Three clips whose lengths map to distinct one-hot categories (1, 2, 3).
    let clips = [dummy_audio(1), dummy_audio(2), dummy_audio(3)];
    let clip_vecs: Vec<Vec<f32>> = clips.iter().map(|c| e.embed(c).unwrap()).collect();

    // A 3-char query lands on category 3 → the third clip is the match.
    let query = e.embed_text("abc").unwrap();
    let scores: Vec<f32> = clip_vecs.iter().map(|v| cosine(&query, v)).collect();

    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        best, 2,
        "the matching clip must rank first, scores={scores:?}"
    );
    assert!(scores[2] > scores[0] && scores[2] > scores[1], "{scores:?}");
}

/// The sc-12838 acceptance path end to end, weights-free: resolve a registered voice embedder,
/// drive `embed()` over a reference clip, then feed the resulting embedding into a stub audio
/// [`Generator`]'s conditioning (`Conditioning::VoiceEmbedding`) and validate — a cloned voice
/// driving TTS, the audio mirror of a face embedding conditioning InstantID/PuLID.
#[test]
fn voice_embedding_resolves_embeds_and_conditions_a_generator() {
    use crate::generator::{Conditioning, ConditioningKind};

    let registry = dummy_registry();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));

    // resolve → embed
    let embedder = registry
        .load_voice_embedder("dummy_test_voice_embedder", &spec)
        .expect("registered");
    let embedding = embedder.embed(&dummy_audio(5)).unwrap();
    assert_eq!(embedding.len(), 4);

    // A stub audio generator that advertises VoiceEmbedding conditioning.
    let tts = DummyGen {
        desc: ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_tts",
            family: "test",
            backend: "mlx",
            modality: Modality::Audio,
            capabilities: Capabilities {
                max_count: 1,
                conditioning: vec![ConditioningKind::VoiceEmbedding],
                audio_sample_rates: vec![24_000],
                ..Default::default()
            },
        },
    };

    // feed the embedding into the generator's conditioning and validate (size-skipping audio floor)
    let req = GenerationRequest {
        conditioning: vec![Conditioning::VoiceEmbedding {
            embedding,
            strength: None,
        }],
        steps: Some(1),
        ..Default::default()
    };
    tts.descriptor()
        .capabilities
        .validate_request_audio("dummy_tts", &req)
        .expect("a cloned-voice embedding is accepted conditioning for the TTS generator");

    // A generator that does NOT advertise it rejects the same conditioning.
    let no_voice = Capabilities {
        max_count: 1,
        audio_sample_rates: vec![24_000],
        ..Default::default()
    };
    assert!(no_voice.validate_request_audio("dummy_tts", &req).is_err());
}

/// The sweep (sc-9098, F-009) is clean over the explicit dummy catalog.
#[test]
fn descriptor_sweep_is_clean_over_dummy_catalog() {
    let errs = dummy_registry().descriptor_conformance_errors();
    assert!(
        errs.is_empty(),
        "descriptor conformance FAILED:\n  - {}",
        errs.join("\n  - ")
    );
}

#[test]
fn activation_memory_anchor_selection_is_provider_route_wide() {
    const REGISTRATION: ActivationMemoryRegistration = ActivationMemoryRegistration {
        provider_id: "dummy_test_model",
        anchor: ActivationMemoryAnchor { bytes_1024: 8 },
    };
    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_activation_memory(REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(
        registry
            .activation_memory_bytes_1024("dummy_test_model")
            .unwrap(),
        Some(8)
    );
}

/// *Mutations that red this:* pricing `None` facts through the provider, treating an
/// unregistered provider as zero, or accepting duplicate/orphan registrations.
#[test]
fn reference_workspace_registration_is_additive_and_fails_closed() {
    fn price(_: &LoadSpec, facts: &crate::MemoryReferenceResourceFacts) -> Result<u64> {
        Ok(facts.conditioning_tokens * 10)
    }
    const REGISTRATION: ReferenceWorkspaceRegistration = ReferenceWorkspaceRegistration {
        provider_id: "dummy_test_model",
        reference_workspace_bytes: price,
    };
    let spec = LoadSpec::new(crate::WeightsSource::Dir("/unused".into()));
    let facts = crate::MemoryReferenceResourceFacts {
        conditioning_tokens: 3,
        prepared_canvases: vec![(48, 16)],
    };
    let registry = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_reference_workspace(REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(
        registry
            .reference_workspace_bytes("dummy_test_model", &spec, None)
            .unwrap(),
        Some(0)
    );
    assert_eq!(
        registry
            .reference_workspace_bytes("dummy_test_model", &spec, Some(&facts))
            .unwrap(),
        Some(30)
    );
    assert!(registry
        .reference_workspace_bytes("no_such_model", &spec, None)
        .is_err());
    let unpriced = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(
        unpriced
            .reference_workspace_bytes("dummy_test_model", &spec, Some(&facts))
            .unwrap(),
        None
    );
    assert!(ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_reference_workspace(REGISTRATION)
        .register_reference_workspace(REGISTRATION)
        .build()
        .is_err());
    assert!(ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_reference_workspace(ReferenceWorkspaceRegistration {
            provider_id: "no_such_model",
            reference_workspace_bytes: price,
        })
        .build()
        .is_err());
}

#[test]
fn activation_memory_registration_fails_closed_and_preserves_unmeasured_state() {
    const REGISTRATION: ActivationMemoryRegistration = ActivationMemoryRegistration {
        provider_id: "dummy_test_model",
        anchor: ActivationMemoryAnchor { bytes_1024: 8 },
    };
    let duplicate = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_activation_memory(REGISTRATION)
        .register_activation_memory(REGISTRATION)
        .build()
        .err()
        .expect("duplicate activation registration must fail")
        .to_string();
    assert!(duplicate.contains("duplicate activation-memory provider id"));

    let orphan = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_activation_memory(ActivationMemoryRegistration {
            provider_id: "no_such_model",
            anchor: ActivationMemoryAnchor { bytes_1024: 8 },
        })
        .build()
        .err()
        .expect("orphan activation registration must fail")
        .to_string();
    assert!(orphan.contains("has no matching generator"));

    let zero = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .register_activation_memory(ActivationMemoryRegistration {
            provider_id: "dummy_test_model",
            anchor: ActivationMemoryAnchor { bytes_1024: 0 },
        })
        .build()
        .err()
        .expect("zero activation registration must fail")
        .to_string();
    assert!(zero.contains("is zero"));

    let unmeasured = ProviderRegistryBuilder::new()
        .register_generator(DUMMY_GENERATOR_REGISTRATION)
        .build()
        .unwrap();
    assert_eq!(
        unmeasured
            .activation_memory_bytes_1024("dummy_test_model")
            .unwrap(),
        None
    );
    assert!(unmeasured
        .activation_memory_bytes_1024("no_such_model")
        .is_err());
}

/// sc-19559 — an unsatisfiable step declaration is a descriptor mistake, not a runtime
/// surprise. Each shape refuses EVERY step count, and the shared floor would enforce it on
/// every request, so the sweep has to name it before a catalog ships.
///
/// Each case is built and asserted individually: a single descriptor carrying all of them at
/// once would pass even if only one check fired.
#[test]
fn model_descriptor_errors_flags_an_unsatisfiable_step_declaration() {
    let with = |steps: StepSupport| ModelDescriptor {
        capabilities: Capabilities {
            supported_steps: steps,
            ..dummy_descriptor().capabilities
        },
        ..dummy_descriptor()
    };
    let errs = |steps: StepSupport| model_descriptor_errors(&with(steps));

    // The baseline: the coherent descriptor this varies from is clean, so every message below
    // is attributable to the step declaration alone.
    assert!(errs(StepSupport::Unconstrained).is_empty());
    assert!(errs(StepSupport::Exact(vec![8])).is_empty());
    assert!(errs(StepSupport::Range { min: 1, max: 200 }).is_empty());

    let empty_menu = errs(StepSupport::Exact(Vec::new()));
    assert!(
        empty_menu.iter().any(|e| e.contains("EMPTY exact menu")),
        "{empty_menu:?}"
    );
    let zero_menu = errs(StepSupport::Exact(vec![0, 8]));
    assert!(
        zero_menu.iter().any(|e| e.contains("advertises 0 steps")),
        "{zero_menu:?}"
    );
    let dupe_menu = errs(StepSupport::Exact(vec![8, 8]));
    assert!(
        dupe_menu
            .iter()
            .any(|e| e.contains("duplicate supported step count 8")),
        "{dupe_menu:?}"
    );
    let zero_range = errs(StepSupport::Range { min: 0, max: 200 });
    assert!(
        zero_range.iter().any(|e| e.contains("range starts at 0")),
        "{zero_range:?}"
    );
    let inverted = errs(StepSupport::Range { min: 30, max: 8 });
    assert!(
        inverted.iter().any(|e| e.contains("30..=8 is empty")),
        "{inverted:?}"
    );
}

/// Each per-descriptor invariant fires: identity shape, zero/inverted bounds, duplicate or
/// malformed curated names, duplicate conditioning, video conditioning on an Image model.
#[test]
fn model_descriptor_errors_flags_each_violation() {
    // A fully-coherent descriptor produces no errors.
    assert!(model_descriptor_errors(&dummy_descriptor()).is_empty());

    let broken = ModelDescriptor {
        encoder_contract: Some(crate::EncoderContract {
            architecture: "qwen3",
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 3,
            head_dim: 4,
            vocab_size: 16,
            output_width: 8,
            loaded_hidden_layers: 2,
            requires_final_norm: true,
            requires_lm_head: false,
            hidden_activation: "silu",
            attention_dropout: crate::EncoderConfigFloat::new(0.0),
            rms_norm_eps: crate::EncoderConfigFloat::new(1e-6),
            qk_norm_eps: Some(crate::EncoderConfigFloat::new(1e-6)),
            rope_theta: crate::EncoderConfigFloat::new(1_000_000.0),
            max_position_embeddings: 4_096,
            attention_bias: crate::EncoderConfigBool::Required(false),
            tie_word_embeddings: crate::EncoderConfigBool::Required(true),
            tokenizer: crate::EncoderTokenizerContract {
                family: "test-qwen3",
                binding: crate::EncoderTokenizerBinding::RetainBase,
                artifact_candidates: &["tokenizer/tokenizer.json"],
                required_tokens: &[],
            },
            prompt_executions: &[crate::EncoderPromptExecutionContract {
                purpose: "test",
                template: crate::EncoderPromptTemplate::QwenInstruct,
                add_special_tokens: true,
                length: crate::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 8 },
                padding: crate::EncoderPromptPadding::None,
                prefix_trim: 0,
            }],
            bos_token_id: None,
            eos_token_id: None,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            mrope_section: &[],
            mrope_interleaved: None,
            selected_hidden_layers: &[2],
            packing: None,
            dense_storage_dtype_probe: None,
        }),
        denoiser_output_latent_space: None,
        control_kinds: None,
        // Blank + duplicate required-component ids (sc-13658) — unstageable / ambiguous keys.
        required_components: &["", "voice_embedding", "voice_embedding"],
        id: "Bad Id", // uppercase + whitespace
        family: "",   // empty
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            min_size: 512,
            max_size: 256, // inverted
            max_count: 0,  // zero
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 0 },
            samplers: vec!["euler", "euler", "bad name"], // duplicate + whitespace
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Reference, // duplicate
                ConditioningKind::VideoClip, // video kind on an Image model
            ],
            ..Default::default()
        },
    };
    let errs = model_descriptor_errors(&broken);
    let has = |needle: &str| errs.iter().any(|e| e.contains(needle));
    assert!(has("id \"Bad Id\""), "{errs:?}");
    assert!(has("family \"\""), "{errs:?}");
    assert!(has("invalid text encoder contract"), "{errs:?}");
    assert!(has("max_count is 0"), "{errs:?}");
    assert!(has("min_size 512 > max_size 256"), "{errs:?}");
    assert!(has("explicit-size multiple is 0"), "{errs:?}");
    assert!(has("duplicate sampler entry \"euler\""), "{errs:?}");
    assert!(has("sampler[2] \"bad name\""), "{errs:?}");
    assert!(has("duplicate conditioning kind Reference"), "{errs:?}");
    assert!(has("video conditioning VideoClip"), "{errs:?}");
    // The required-component id list is validated like the curated name lists: a blank id and a
    // duplicate id are both flagged (sc-13658).
    assert!(has("required_component[0] \"\""), "{errs:?}");
    assert!(
        has("duplicate required_component entry \"voice_embedding\""),
        "{errs:?}"
    );

    // All-zero bounds report the Default-0 message (F-084), not the inverted-bounds one.
    let zeroed = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "zeroed",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities::default(),
    };
    assert!(model_descriptor_errors(&zeroed)
        .iter()
        .any(|e| e.contains("left at the Default 0")));
}

/// The multi-turn path-A cross-check (sc-14150): `supports_conversation_history` without the
/// matching `ConditioningKind::ConversationHistory` in `conditioning` is flagged (the "flag on,
/// kind missing" footgun where every path-A request would be rejected by the allowlist), and a
/// descriptor that wires both is conformant.
#[test]
fn model_descriptor_errors_flags_conversation_history_flag_without_kind() {
    let half_wired = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "convo",
        family: "test",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            max_count: 1,
            supports_conversation_history: true,
            // ConditioningKind::ConversationHistory deliberately NOT advertised.
            ..Default::default()
        },
    };
    assert!(model_descriptor_errors(&half_wired)
        .iter()
        .any(|e| e.contains("supports_conversation_history is set but")));

    // Wiring both the flag and the kind is conformant.
    let wired = ModelDescriptor {
        capabilities: Capabilities {
            max_count: 1,
            supports_conversation_history: true,
            conditioning: vec![ConditioningKind::ConversationHistory],
            ..Default::default()
        },
        ..half_wired
    };
    assert!(model_descriptor_errors(&wired).is_empty());
}

/// The size-bounds floor is exempt for `Modality::Audio` (sc-13314): a pure-audio generator has
/// no width/height, so `min_size`/`max_size` left at the unused `Default` 0 must NOT be flagged —
/// mirroring the size-skipping `validate_request_audio` floor. The exemption is scoped to the
/// size axis: every other invariant (identity, `max_count`, curated-name shape) still fires for an
/// audio descriptor.
#[test]
fn audio_descriptor_with_zero_size_bounds_passes_sweep() {
    let audio = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "zeroed_audio",
        family: "test",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            // No spatial dims — bounds stay at the natural unused 0.
            min_size: 0,
            max_size: 0,
            max_count: 1,
            audio_sample_rates: vec![24_000],
            ..Default::default()
        },
    };
    assert!(
        model_descriptor_errors(&audio).is_empty(),
        "an audio descriptor with unused (0) size bounds must pass the sweep: {:?}",
        model_descriptor_errors(&audio)
    );

    // The exemption is only the size axis: a broken `max_count` on the same audio descriptor is
    // still reported.
    let audio_bad_count = ModelDescriptor {
        required_components: &[],
        capabilities: Capabilities {
            max_count: 0,
            ..audio.capabilities.clone()
        },
        ..audio.clone()
    };
    assert!(
        model_descriptor_errors(&audio_bad_count)
            .iter()
            .any(|e| e.contains("max_count is 0")),
        "the audio exemption must not relax the non-size invariants"
    );
}

/// The strictness the sweep exists for is preserved for the visual modalities: an `Image`/`Video`
/// descriptor with invalid size bounds is STILL flagged (the audio exemption above does not weaken
/// the check for modalities that genuinely carry a spatial size range).
#[test]
fn visual_descriptor_with_invalid_size_bounds_still_fails_sweep() {
    // Video, zero bounds → the Default-0 footgun still fires.
    let video_zero = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "video_zero",
        family: "test",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            min_size: 0,
            max_size: 0,
            max_count: 1,
            ..Default::default()
        },
    };
    assert!(
        model_descriptor_errors(&video_zero)
            .iter()
            .any(|e| e.contains("left at the Default 0")),
        "a Video descriptor with zero size bounds must still be rejected"
    );

    // Image, inverted bounds → the inverted-bounds message still fires.
    let image_inverted = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "image_inverted",
        family: "test",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            min_size: 512,
            max_size: 256,
            max_count: 1,
            ..Default::default()
        },
    };
    assert!(
        model_descriptor_errors(&image_inverted)
            .iter()
            .any(|e| e.contains("min_size 512 > max_size 256")),
        "an Image descriptor with inverted size bounds must still be rejected"
    );

    // `Both` (emits image or video) is a visual modality too — zero bounds still fail.
    let both_zero = ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: "both_zero",
        family: "test",
        backend: "mlx",
        modality: Modality::Both,
        capabilities: Capabilities {
            min_size: 0,
            max_size: 0,
            max_count: 1,
            ..Default::default()
        },
    };
    assert!(
        model_descriptor_errors(&both_zero)
            .iter()
            .any(|e| e.contains("left at the Default 0")),
        "a Both-modality descriptor with zero size bounds must still be rejected"
    );
}

/// Build a synthetic diffusers-style snapshot with a `bytes`-sized `model.safetensors` under each
/// named subdir.
///
/// Returns the `TempDir` guard, not a bare `PathBuf` (sc-17755): the tree leaves on `Drop`,
/// including out of a panicking test, which the trailing `remove_dir_all(..).ok()` lines this
/// replaced could not do. Callers bind it for the whole test and read `tmp.path()`.
fn synthetic_snapshot(tag: &str, subdirs: &[(&str, usize)]) -> tempfile::TempDir {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("gencore_footprint_{tag}_"))
        .tempdir()
        .expect("fixture temp dir");
    let root = tmp.path();
    for (sub, bytes) in subdirs {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; *bytes]).unwrap();
    }
    tmp
}

/// Guards the sc-17755 fix for [`synthetic_snapshot`]: the assembled tree leaves with the guard.
/// Revert it to a bare `create_dir_all` on a `temp_dir()` join and this goes RED.
#[test]
fn synthetic_snapshot_is_removed_on_drop() {
    let (root, shard) = {
        let tmp = synthetic_snapshot("drop-guard", &[("transformer", 16)]);
        let root = tmp.path().to_path_buf();
        let shard = root.join("transformer/model.safetensors");
        assert!(shard.is_file(), "snapshot shard not written");
        (root, shard)
    };
    assert!(!shard.exists(), "shard survived: {}", shard.display());
    assert!(!root.exists(), "snapshot root survived: {}", root.display());
}

/// sc-10894: a provider that declared a footprint returns the per-component on-disk split, resolved
/// from the exact subdirs its loader uses — including a text encoder under a NON-`text_encoder`
/// subdir (`mllm/`, the boogu layout) that a name-guessing consumer would read as zero.
#[test]
fn footprint_returns_provider_component_split() {
    let root = synthetic_snapshot(
        "split",
        &[("mllm", 1500), ("transformer", 9000), ("vae", 400)],
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));

    let fp = dummy_registry()
        .footprint("dummy_footprint_model", &spec)
        .expect("registered + declares a footprint")
        .expect("Some — the provider computed the split");
    assert_eq!(
        fp,
        PerComponentBytes {
            text_encoder: 1500,
            dit: 9000,
            vae: 400,
        }
    );
    // The whole point: the text encoder is NON-zero even though it is not under `text_encoder*`.
    assert!(fp.text_encoder > 0, "mllm/ text encoder must be measured");
}

/// A registered generator that declares NO footprint yields `Ok(None)` (the consumer falls back);
/// an unknown id is an `Err`.
#[test]
fn footprint_is_none_without_declaration_and_errs_on_unknown_id() {
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    // `dummy_test_model` is registered but declares no footprint.
    let registry = dummy_registry();
    assert_eq!(registry.footprint("dummy_test_model", &spec).unwrap(), None);
    // Unknown id → Err (a fail-open consumer treats it like None).
    assert!(registry.footprint("no_such_model", &spec).is_err());
}

/// sc-10894: `from_spec_subdirs` sums each component's subdir(s) (SD3's three text encoders here),
/// treats a missing subdir as `0`, and errors on a single-`File` source (no tree to split).
#[test]
fn per_component_bytes_from_spec_subdirs_and_file_guard() {
    let root = synthetic_snapshot(
        "sd3",
        &[
            ("text_encoder", 100),
            ("text_encoder_2", 200),
            ("text_encoder_3", 4000),
            ("transformer", 8000),
            ("vae", 300),
        ],
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
    let fp = PerComponentBytes::from_spec_subdirs(
        &spec,
        &["text_encoder", "text_encoder_2", "text_encoder_3"],
        &["transformer"],
        &["vae"],
    )
    .unwrap();
    assert_eq!(fp.text_encoder, 4300); // 100 + 200 + 4000
    assert_eq!(fp.dit, 8000);
    assert_eq!(fp.vae, 300);

    // A named-but-absent subdir contributes 0 (does not error).
    let fp_missing =
        PerComponentBytes::from_spec_subdirs(&spec, &["nope"], &["transformer"], &["vae"]).unwrap();
    assert_eq!(fp_missing.text_encoder, 0);

    // A single-file source has no component tree → Err (consumer falls back to whole-file).
    let file_spec = LoadSpec::new(WeightsSource::File(
        root.path().join("transformer/model.safetensors"),
    ));
    assert!(PerComponentBytes::from_spec_subdirs(&file_spec, &["te"], &["dit"], &["vae"]).is_err());
}

/// sc-10894: `from_root_subdirs` sums a component named by a flat FILE (the bernini/anima layout —
/// `t5_encoder.safetensors` at the root, not a `text_encoder/` subdir) as well as a subdir, against
/// an already-resolved root.
#[test]
fn per_component_bytes_from_root_subdirs_handles_flat_files() {
    let tmp = tempfile::Builder::new()
        .prefix("gencore_footprint_flat_")
        .tempdir()
        .expect("fixture temp dir");
    let root = tmp.path();
    // bernini-style flat component files at the root.
    std::fs::write(root.join("t5_encoder.safetensors"), vec![0u8; 2000]).unwrap();
    std::fs::write(root.join("low_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
    std::fs::write(root.join("high_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
    std::fs::write(root.join("vae.safetensors"), vec![0u8; 500]).unwrap();

    let fp = PerComponentBytes::from_root_subdirs(
        root,
        &["t5_encoder.safetensors"],
        &[
            "low_noise_model.safetensors",
            "high_noise_model.safetensors",
        ],
        &["vae.safetensors"],
    );
    assert_eq!(
        fp,
        PerComponentBytes {
            text_encoder: 2000,
            dit: 12000,
            vae: 500,
        }
    );
}
