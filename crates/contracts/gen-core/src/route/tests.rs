//! Tier 0 weights-free unit & conformance tests for `RouteRecord`, one-call `register_route`,
//! `HOOK-4` precondition derivation, negative conformance fixtures, and `AP18` descriptor indexing.

use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::control::AcceptedControlKinds;
use crate::driver::{DriverRefusalReasons, DriverSupport};
use crate::feature::{
    FeatureInapplicabilityCode, FeatureRegistry, Pass0ConditioningRequirement,
};
use crate::generator::{
    ActivationMemoryAnchor, ConditioningKind, ControlKind, GenerationRequest, Generator, Modality,
    ModelDescriptor, SizeFloor, StepSupport,
};
use crate::latent::FLUX1_LATENT_SPACE;
use crate::memory_strategy::{
    MemoryProviderContract, MemoryReferenceResourceFacts, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy,
};
use crate::pag::{PagCapability, DEFAULT_PAG_MAX_SCALE};
use crate::registry::{
    mlx_memory_contract_surface_specs, MemoryBehaviorFixture, MemoryContractSurfaceSpec,
    MemoryRegistration, PerComponentBytes, ProviderRegistryBuilder,
};
use crate::render::{PackedImagePass, RenderPlan};
use crate::runtime::{LoadSpec, Quant, WeightsSource};
use crate::{Error, Result};

use super::*;

fn sample_pag() -> PagCapability {
    PagCapability {
        num_blocks: 57,
        block_domain: Cow::Borrowed("flux_blocks"),
        default_layers: vec![28],
        default_start: 0.0,
        default_stop: 1.0,
        recommended_scale: 1.5,
        max_scale: DEFAULT_PAG_MAX_SCALE,
    }
}

fn stub_compile_plan(_spec: &LoadSpec, req: &GenerationRequest) -> Result<Option<RenderPlan>> {
    let steps = req.steps.unwrap_or(25) as usize;
    let sigmas: Vec<f32> = (0..=steps)
        .map(|i| 1.0 - (i as f32 / steps.max(1) as f32))
        .collect();
    RenderPlan::base_image(
        &sample_flux_descriptor(),
        req,
        PackedImagePass {
            pixel_stride: 16,
            token_width: 64,
            sampler: "euler",
            guidance: 3.5,
            sigmas,
            executed_steps: steps,
            capture_sigma: 0.0,
        },
    )
    .map(Some)
}

fn stub_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Err(Error::Msg("stub loader not invoked in Tier 0 tests".into()))
}

fn stub_footprint(_spec: &LoadSpec) -> Result<PerComponentBytes> {
    Ok(PerComponentBytes {
        text_encoder: 100,
        dit: 200,
        vae: 50,
    })
}

fn stub_memory_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(MemoryProviderContract::compatibility_default(
        "pilot_flux_dev",
        crate::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

fn stub_memory_surface_contract(
    _surface: &MemoryContractSurfaceSpec,
) -> Result<MemoryProviderContract> {
    Ok(MemoryProviderContract::compatibility_default(
        "pilot_flux_dev",
        crate::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    ))
}

fn stub_safety_check(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _ctx: &MemoryRunContext,
) -> MemorySafetyDecision {
    MemorySafetyDecision::Accept
}

fn stub_valid_fixtures(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _strategy: MemoryStrategy,
) -> Result<Vec<MemoryBehaviorFixture>> {
    Ok(Vec::new())
}

fn stub_begin_request(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _ctx: &MemoryRunContext,
) -> Result<Option<Box<dyn crate::MemoryRequestScope>>> {
    Ok(None)
}

fn stub_ref_workspace(
    _spec: &LoadSpec,
    _facts: &MemoryReferenceResourceFacts,
) -> Result<u64> {
    Ok(4096)
}

fn sample_flux_route() -> RouteRecord {
    RouteRecord {
        route_id: "pilot_flux_dev",
        family: "flux",
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![
                    TextEncoderDeclaration::new("clip_l", Some(77)),
                    TextEncoderDeclaration::new("t5_xxl", Some(512)),
                ],
                ..Default::default()
            },
            conditioning: ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                modules: vec!["xlabs_ip_adapter"],
                ..Default::default()
            },
            sampling: SamplingSurface {
                samplers: vec!["euler", "heun", "flow_match"],
                schedulers: vec!["simple", "linear"],
                supports_guidance: true,
                requires_sigma_shift: true,
                ..Default::default()
            },
            denoiser: ModelBinding::family_model(
                FamilyModelSpec::new("flux1_mmdit")
                    .with_low_res_start()
                    .with_pag(sample_pag()),
            ),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&FLUX1_LATENT_SPACE),
                pixel_stride: Some(16),
                token_width: Some(64),
                supports_diffusion_decoder: false,
            },
            upsamplers: UpsamplerSurface {
                components: vec!["pid_decoder"],
                max_temporal_upsample_rounds: 0,
            },
            adapters: AdapterSurface::lora_and_lokr(AdapterInstallPolicy::ForwardResidual),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::PromptEncode,
                        ["clip_l", "t5_xxl"],
                        true,
                    ),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Denoise, ["flux1_mmdit"], false),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Decode, ["flux1_ae"], false),
                ],
                supports_sequential_offload: true,
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                ..Default::default()
            },
            output: OutputSurface {
                preview: PreviewSupport::LatentProjection,
                ..Default::default()
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 16 },
            supported_steps: StepSupport::Range { min: 1, max: 100 },
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(25),
            guidance: Some(3.5),
            sampler: Some("flow_match"),
            schedule: Some("linear"),
            shift: Some(1.15),
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(sample_flux_descriptor, stub_load)
            .with_footprint(stub_footprint)
            .with_render_plan(stub_compile_plan)
            .with_memory(stub_memory_contract, stub_safety_check)
            .with_memory_fixture(mlx_memory_contract_surface_specs, stub_memory_contract)
            .with_memory_surface_resolver(stub_memory_surface_contract)
            .with_memory_behavior(stub_valid_fixtures, stub_begin_request)
            .with_activation_memory(ActivationMemoryAnchor {
                bytes_1024: 15_096_810_046,
            })
            .with_reference_workspace(stub_ref_workspace),
    }
}

fn sample_flux_descriptor() -> ModelDescriptor {
    sample_flux_route().compute_descriptor()
}

fn sample_audio_route() -> RouteRecord {
    RouteRecord {
        route_id: "pilot_kokoro_82m",
        family: "kokoro",
        backend: "candle",
        modality: Modality::Audio,
        mac_only: false,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("albert", Some(512))],
                ..Default::default()
            },
            denoiser: ModelBinding::non_diffusion("styletts2_istftnet"),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("istftnet_vocoder"),
                ..Default::default()
            },
            adapters: AdapterSurface::unsupported(),
            residency: ResidencySurface {
                lifetime_graph: vec![ResidencyPhaseNode::new(
                    ResidencyPhaseKind::AudioSynthesize,
                    ["albert", "predictor", "istftnet_vocoder"],
                    false,
                )],
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                ..Default::default()
            },
            audio: AudioSurface {
                sample_rates: vec![24_000],
                voices: vec!["af_heart", "bm_george"],
                languages: vec!["en", "en-us", "en-gb"],
                ..Default::default()
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: 0,
            max_size: 0,
            max_audio_duration_secs: Some(30.0),
            ..Default::default()
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::DeterministicSynthesis,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            audio_voice: Some("af_heart"),
            audio_sample_rate: Some(24_000),
            audio_language: Some("en"),
            ..Default::default()
        },
        bindings: RouteBindings::generator(sample_audio_descriptor, stub_load),
    }
}

fn sample_audio_descriptor() -> ModelDescriptor {
    sample_audio_route().compute_descriptor()
}

#[test]
fn register_route_one_call_registers_all_tables_and_computes_capabilities() {
    let route = sample_flux_route();
    verify_route_conformance(&route).expect("sample flux route must conform");

    let desc = route.compute_descriptor();
    assert_eq!(desc.id, "pilot_flux_dev");
    assert_eq!(desc.family, "flux");
    assert_eq!(desc.backend, "mlx");
    assert_eq!(desc.modality, Modality::Image);
    assert!(desc.capabilities.supports_guidance);
    assert!(desc.capabilities.supports_lora);
    assert!(desc.capabilities.supports_lokr);
    assert!(desc.capabilities.supports_preview);
    assert!(desc.capabilities.supports_low_res_start);
    assert!(!desc.capabilities.supports_phases);
    assert!(desc.capabilities.mac_only);
    assert_eq!(desc.capabilities.min_size, 256);
    assert_eq!(desc.capabilities.max_size, 2048);
    assert_eq!(desc.capabilities.max_count, 8);
    assert_eq!(desc.capabilities.pag, Some(sample_pag()));

    let registry = ProviderRegistryBuilder::new()
        .register_route(route)
        .register_route(sample_audio_route())
        .build()
        .expect("one-call register_route builds cleanly");

    assert_eq!(registry.route_records().len(), 2);
    assert_eq!(registry.generators().len(), 2);
    assert_eq!(registry.generator_descriptors().len(), 2);
    assert!(registry.route_record("pilot_flux_dev").is_some());
    assert!(registry.route_record("pilot_kokoro_82m").is_some());
    assert!(registry.render_plan_registration("pilot_flux_dev").is_some());
    assert_eq!(registry.memory_strategy_registrations().len(), 1);
    assert_eq!(registry.memory_contract_fixture_registrations().len(), 1);
    assert_eq!(
        registry.memory_contract_surface_resolver_registrations().len(),
        1
    );
    assert_eq!(registry.memory_behavior_registrations().len(), 1);
    assert_eq!(
        registry
            .activation_memory_bytes_1024("pilot_flux_dev")
            .unwrap(),
        Some(15_096_810_046)
    );
    let dummy_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    assert_eq!(
        registry
            .footprint("pilot_flux_dev", &dummy_spec)
            .unwrap(),
        Some(PerComponentBytes {
            text_encoder: 100,
            dit: 200,
            vae: 50,
        })
    );
    assert_eq!(
        registry
            .reference_workspace_bytes("pilot_flux_dev", &dummy_spec, None)
            .unwrap(),
        Some(0)
    );
    assert!(registry.route_conformance_errors().is_empty());
    assert!(registry.descriptor_conformance_errors().is_empty());

    // Audio route defaults & limits serialize cleanly.
    let audio_json = registry
        .route_record("pilot_kokoro_82m")
        .unwrap()
        .to_json();
    assert_eq!(audio_json["defaults"]["audio_voice"], "af_heart");
    assert_eq!(audio_json["defaults"]["audio_sample_rate"], 24_000);
    fn has_expected_audio_limit(json: &serde_json::Value) -> bool {
        json["limits"]["max_audio_duration_secs"] == 30.0
    }
    assert!(has_expected_audio_limit(&audio_json));
}

#[test]
fn hook4_removing_structural_precondition_flips_applicability_with_zero_table_edits() {
    let features = FeatureRegistry::standard();
    let base = sample_flux_route();

    // Baseline: low_res_start, hires_fix, and pag are applicable; krea_multiphase and wan_expert_switch are not.
    let base_apps = features.evaluate_route_record(&base);
    assert!(base_apps["low_res_start"].applicable);
    assert!(base_apps["hires_fix"].applicable);
    assert!(base_apps["pag"].applicable);
    assert!(!base_apps["krea_multiphase"].applicable);
    assert!(!base_apps["wan_expert_switch"].applicable);

    // 1. Removing output_latent_space flips low_res_start off with MissingSpatialUnpackRepack.
    let mut no_latent = base.clone();
    no_latent.structure.latent_codec.output_latent_space = None;
    assert!(!no_latent.compute_capabilities().supports_low_res_start);
    let no_latent_apps = features.evaluate_route_record(&no_latent);
    assert!(!no_latent_apps["low_res_start"].applicable);
    assert!(no_latent_apps["low_res_start"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::MissingSpatialUnpackRepack));

    // 2. Removing denoiser geometry_change flips low_res_start off with MissingCleanEstimateRenoiseTransition.
    let mut no_geom = base.clone();
    if let ModelBinding::FamilyModel(ref mut spec) = no_geom.structure.denoiser {
        spec.supports_geometry_change = false;
    }
    assert!(!no_geom.compute_capabilities().supports_low_res_start);
    let no_geom_apps = features.evaluate_route_record(&no_geom);
    assert!(!no_geom_apps["low_res_start"].applicable);
    assert!(no_geom_apps["low_res_start"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::MissingCleanEstimateRenoiseTransition));

    // 3. Removing render_plan flips low_res_start off with MissingRenderPlan.
    let mut no_plan = base.clone();
    no_plan.bindings.render_plan = None;
    let no_plan_apps = features.evaluate_route_record(&no_plan);
    assert!(!no_plan_apps["low_res_start"].applicable);
    assert!(no_plan_apps["low_res_start"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::MissingRenderPlan));

    // 4. Changing pass0_requirement to RequiresControl flips low_res_start off with ConflictsWithPass0ControlConditioning
    //    (ZERO route_id string matching!).
    let mut requires_ctrl = base.clone();
    requires_ctrl.structure.conditioning.pass0_requirement =
        Pass0ConditioningRequirement::RequiresControl;
    assert!(!requires_ctrl.compute_capabilities().supports_low_res_start);
    let ctrl_apps = features.evaluate_route_record(&requires_ctrl);
    assert!(!ctrl_apps["low_res_start"].applicable);
    assert!(ctrl_apps["low_res_start"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));

    // 5. Setting defaults.steps to 1 flips low_res_start off with InsufficientExecutedSteps.
    let mut one_step = base.clone();
    one_step.defaults.steps = Some(1);
    let one_step_apps = features.evaluate_route_record(&one_step);
    assert!(!one_step_apps["low_res_start"].applicable);
    assert!(one_step_apps["low_res_start"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::InsufficientExecutedSteps));

    // 6. Removing PAG from denoiser flips pag off with MissingPagCapability.
    let mut no_pag = base.clone();
    if let ModelBinding::FamilyModel(ref mut spec) = no_pag.structure.denoiser {
        spec.pag = None;
    }
    let no_pag_apps = features.evaluate_route_record(&no_pag);
    assert!(!no_pag_apps["pag"].applicable);
    assert!(no_pag_apps["pag"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::MissingPagCapability));

    // 7. Removing Reference conditioning flips hires_fix off with MissingProducedReferenceSupport.
    let mut no_ref = base.clone();
    no_ref.structure.conditioning.kinds.clear();
    let no_ref_apps = features.evaluate_route_record(&no_ref);
    assert!(!no_ref_apps["hires_fix"].applicable);
    assert!(no_ref_apps["hires_fix"]
        .reasons
        .iter()
        .any(|r| r.code == FeatureInapplicabilityCode::MissingProducedReferenceSupport));

    // 8. Enabling multiphase_denoise or expert_switching flips those features on automatically.
    let mut with_phases = base.clone();
    if let ModelBinding::FamilyModel(ref mut spec) = with_phases.structure.denoiser {
        spec.supports_multiphase_denoise = true;
        spec.supports_expert_switching = true;
    }
    assert!(with_phases.compute_capabilities().supports_phases);
    let phase_apps = features.evaluate_route_record(&with_phases);
    assert!(phase_apps["krea_multiphase"].applicable);
    assert!(phase_apps["wan_expert_switch"].applicable);
}

#[test]
fn tier0_conformance_rejects_negative_fixtures_for_all_behavioural_declarations() {
    let base = sample_flux_route();

    // Helper: drop static descriptor fn parity check when mutating structural fields in isolation.
    let mut det = base.clone();
    det.bindings.generator = None;

    // 1. Defaults outside limits: steps out of range.
    let mut bad_steps = det.clone();
    bad_steps.defaults.steps = Some(999);
    assert!(verify_route_conformance(&bad_steps).is_err());

    // 2. Defaults.guidance set when sampling.supports_guidance == false.
    let mut bad_guidance = det.clone();
    bad_guidance.structure.sampling.supports_guidance = false;
    bad_guidance.defaults.guidance = Some(3.5);
    assert!(verify_route_conformance(&bad_guidance).is_err());

    // 3. Defaults.dimensions off grid or out of bounds.
    let mut bad_dims = det.clone();
    bad_dims.defaults.dimensions = Some((1000, 1000)); // not multiple of 16
    assert!(verify_route_conformance(&bad_dims).is_err());

    // 4. Defaults.strength out of [0, 1].
    let mut bad_strength = det.clone();
    bad_strength.defaults.strength = Some(1.5);
    assert!(verify_route_conformance(&bad_strength).is_err());

    // 5. Defaults.sampler not in sampling.samplers.
    let mut bad_sampler = det.clone();
    bad_sampler.defaults.sampler = Some("nonexistent_sampler");
    assert!(verify_route_conformance(&bad_sampler).is_err());

    // 6. FamilyModel geometry_change without renoise_from_clean.
    let mut bad_model = det.clone();
    bad_model.structure.denoiser = ModelBinding::FamilyModel(FamilyModelSpec {
        model_id: "broken_model",
        supports_geometry_change: true,
        supports_renoise_from_clean: false,
        supports_read_only_boundary_estimate: true,
        supports_multiphase_denoise: false,
        supports_expert_switching: false,
        pag: None,
    });
    assert!(verify_route_conformance(&bad_model).is_err());

    // 7. FamilyDriver DriverSupport carry_history without arbitrary_slice.
    let mut bad_driver = det.clone();
    bad_driver.structure.denoiser = ModelBinding::FamilyDriver(FamilyDriverSpec::new(
        DriverSupport {
            driver_id: "broken_driver",
            arbitrary_slice: false,
            carry_history: true,
            reset_history: false,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons::default(),
        },
    ));
    assert!(verify_route_conformance(&bad_driver).is_err());

    // 8. Adapter surface vs install_policy mismatch.
    let mut bad_adapters = det.clone();
    bad_adapters.structure.adapters = AdapterSurface {
        supports_lora: true,
        supports_lokr: false,
        install_policy: AdapterInstallPolicy::Unsupported,
    };
    assert!(verify_route_conformance(&bad_adapters).is_err());

    // 9. Pass0 RequiresControl without ConditioningKind::Control.
    let mut bad_ctrl = det.clone();
    bad_ctrl.structure.conditioning.pass0_requirement =
        Pass0ConditioningRequirement::RequiresControl;
    assert!(verify_route_conformance(&bad_ctrl).is_err());

    // 10. control_kinds without ConditioningKind::Control.
    let mut bad_ctrl_kinds = det.clone();
    bad_ctrl_kinds.structure.conditioning.control_kinds =
        Some(AcceptedControlKinds::Only(vec![ControlKind::Pose]));
    assert!(verify_route_conformance(&bad_ctrl_kinds).is_err());

    // 11. Preview on Modality::Audio rejected.
    let mut bad_audio_preview = sample_audio_route();
    bad_audio_preview.bindings.generator = None;
    bad_audio_preview.structure.output.preview = PreviewSupport::LatentProjection;
    assert!(verify_route_conformance(&bad_audio_preview).is_err());

    // 12. Audio default voice not in declared voices.
    let mut bad_voice = sample_audio_route();
    bad_voice.bindings.generator = None;
    bad_voice.defaults.audio_voice = Some("unknown_voice");
    assert!(verify_route_conformance(&bad_voice).is_err());

    // 13. Audio max_speakers without supports_multi_speaker.
    let mut bad_speakers = sample_audio_route();
    bad_speakers.bindings.generator = None;
    bad_speakers.structure.audio.max_speakers = Some(4);
    assert!(verify_route_conformance(&bad_speakers).is_err());

    // 14. Static descriptor drift caught by both conformance and ProviderRegistryBuilder::build().
    let mut drifted = base.clone();
    drifted.limits.max_size = 4096; // diverges from sample_flux_descriptor() which has 2048
    assert!(verify_route_conformance(&drifted).is_err());
    assert!(ProviderRegistryBuilder::new()
        .register_route(drifted)
        .build()
        .is_err());

    // 15. Mismatched provider_id in memory registration.
    let mut bad_mem = det.clone();
    bad_mem.bindings = bad_mem
        .bindings
        .with_memory_registration(MemoryRegistration {
            provider_id: "other_route_id",
            contract: stub_memory_contract,
            safety_check: stub_safety_check,
        });
    assert!(verify_route_conformance(&bad_mem).is_err());
}

#[test]
fn ap18_registry_indexes_descriptors_without_reinvoking_descriptor_fn_on_lookups() {
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn counted_route() -> RouteRecord {
        let mut r = sample_audio_route();
        r.route_id = "counted_route";
        r.bindings = RouteBindings::generator(counted_descriptor, stub_load);
        r
    }

    fn counted_descriptor() -> ModelDescriptor {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        counted_route().compute_descriptor()
    }

    CALL_COUNT.store(0, Ordering::SeqCst);
    let registry = ProviderRegistryBuilder::new()
        .register_route(counted_route())
        .build()
        .expect("counted route builds");

    let count_after_build = CALL_COUNT.load(Ordering::SeqCst);
    assert!(count_after_build > 0);

    // Every indexed lookup below must read the cached descriptor/index without calling descriptor().
    let _ = registry.generator_descriptor("counted_route").unwrap();
    let _ = registry.route_record("counted_route").unwrap();
    let _ = registry.provider_encoder_contract("counted_route");
    let _ = registry.activation_memory_bytes_1024("counted_route").unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let _ = registry.footprint("counted_route", &spec).unwrap();
    let _ = registry
        .memory_strategy_contract("counted_route", &spec)
        .unwrap();
    let _ = registry
        .reference_workspace_bytes("counted_route", &spec, None)
        .unwrap();
    let _ = registry.route_feature_applicability("counted_route").unwrap();
    let _ = registry.load("counted_route", &spec);

    assert_eq!(
        CALL_COUNT.load(Ordering::SeqCst),
        count_after_build,
        "indexed ProviderRegistry lookups must not re-invoke descriptor() after build()"
    );
}
