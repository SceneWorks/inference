//! The LTX-2.3 and LTX-2.5 model routes and descriptors.

use mlx_gen::gen_core::{
    AdapterInstallPolicy, AdapterSurface, ConditioningSurface, FamilyDriverSpec,
    LatentCodecDeclaration, ModelBinding, OutputSurface, Pass0ConditioningRequirement,
    PreviewSupport, ProductLimits, ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface,
    RouteBindings, RouteDefaults, RouteRecord, RouteStructure, SamplingSurface,
    SeedAndBatchSemantics, SeedScheme, SizeFloor, TextEncoderDeclaration, TextEncoderSurface,
    UpsamplerSurface, VideoSurface, WeightPlanDeclaration, WeightSourceKind,
    LTX_VIDEO_LATENT_SPACE,
};

use super::*;

/// Primary typed route declaration for LTX-2.3 AudioVideo (`REG-1`, `AP5`).
pub fn route() -> RouteRecord {
    RouteRecord {
        route_id: MODEL_ID,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("gemma3_12b", Some(1024))],
                ..Default::default()
            },
            conditioning: ConditioningSurface {
                kinds: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                    ConditioningKind::Keyframe,
                    ConditioningKind::VideoClip,
                    ConditioningKind::ControlClip,
                ],
                control_kinds: None,
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: false,
                modules: vec![],
                ..Default::default()
            },
            sampling: SamplingSurface {
                samplers: curated_sampler_names(),
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: false,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: ModelBinding::family_driver(FamilyDriverSpec::new(
                crate::drivers::ltx_driver_support("ltx_2_3"),
            )),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("ltx_video_vae"),
                output_latent_space: Some(&LTX_VIDEO_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: UpsamplerSurface {
                components: vec!["ltx_spatial_upsampler"],
                max_temporal_upsample_rounds: 0,
            },
            adapters: AdapterSurface::lora_and_lokr(AdapterInstallPolicy::ForwardResidual),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::PromptEncode,
                        ["gemma3_12b", "ltx_connector"],
                        true,
                    ),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Denoise, ["ltx_av_dit"], true),
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::Upsample,
                        ["ltx_spatial_upsampler"],
                        true,
                    ),
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::Decode,
                        ["ltx_video_vae", "ltx_audio_vae", "ltx_vocoder"],
                        false,
                    ),
                ],
                supports_sequential_offload: false,
                unconditionally_engages_staged_residency: true,
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                supported_quants: &[],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
                ..Default::default()
            },
            output: OutputSurface {
                preview: PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            video: VideoSurface {
                supports_auto_duration: false,
                supports_generated_keyframes: false,
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            size_floor: SizeFloor::RangeChecked,
            supported_steps: StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS]),
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(crate::pipeline::NATIVE_STEPS),
            guidance: None,
            sampler: Some("euler"),
            schedule: None,
            shift: None,
            strength: None,
            dimensions: Some((768, 512)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(descriptor, |spec| load(spec).map_err(Into::into))
            .with_memory_registration(crate::memory_strategy::MEMORY_REGISTRATION)
            .with_memory_fixture(
                crate::memory_strategy::memory_contract_surface_specs,
                crate::memory_strategy::weights_free_memory_strategy_contract,
            )
            .with_memory_behavior_registration(crate::memory_strategy::MEMORY_BEHAVIOR),
    }
}

/// Primary typed route declaration for LTX-2.5 AudioVideo (`REG-1`, `AP5`).
pub fn route_25() -> RouteRecord {
    let mut out = route();
    out.route_id = MODEL_25_ID;
    out.structure.text_encoders.encoders = vec![TextEncoderDeclaration::new(
        "gemma4_text_encoder",
        Some(1024),
    )];
    out.structure.text_encoders.supports_prompt_enhancement = true;
    out.structure.conditioning.supports_negative_prompt = true;
    out.structure.denoiser = ModelBinding::family_driver(FamilyDriverSpec::new(
        crate::drivers::ltx_driver_support("ltx_2_5"),
    ));
    out.structure.latent_codec.supports_diffusion_decoder = true;
    out.structure.upsamplers.components = vec!["ltx_spatial_upsampler", "ltx_temporal_upsampler"];
    out.structure.upsamplers.max_temporal_upsample_rounds =
        gen_core_ltx::ltx_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS;
    out.structure.adapters = AdapterSurface {
        supports_lora: true,
        supports_lokr: false,
        install_policy: AdapterInstallPolicy::ForwardResidual,
    };
    out.structure.residency.lifetime_graph[0].components =
        vec!["gemma4_text_encoder", "ltx_connector"];
    out.structure.video = VideoSurface {
        supports_auto_duration: true,
        supports_generated_keyframes: true,
    };
    out.limits.supported_steps = StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS, 30]);
    out.bindings = RouteBindings::generator(descriptor_25, |spec| {
        load_25(spec).map_err(Into::into)
    })
    .with_memory_registration(crate::memory_strategy_2_5::LTX_2_5_MEMORY_REGISTRATION)
    .with_memory_fixture_registration(crate::memory_strategy_2_5::LTX_2_5_MEMORY_FIXTURE)
    .with_memory_behavior_registration(crate::memory_strategy_2_5::LTX_2_5_MEMORY_BEHAVIOR);
    out
}

/// Stable identity + advertised capabilities for the LTX-2.3 AudioVideo model, computed from
/// [`route`].
pub fn descriptor() -> ModelDescriptor {
    route().compute_descriptor()
}

/// Stable identity and capability surface of the LTX-2.5 MLX route, computed from [`route_25`].
pub fn descriptor_25() -> ModelDescriptor {
    route_25().compute_descriptor()
}

pub(super) fn descriptor_25_for_variant(variant: TransformerVariant) -> ModelDescriptor {
    let mut out = descriptor_25();
    match variant {
        TransformerVariant::Distilled => {
            out.capabilities.supports_negative_prompt = false;
            out.capabilities.supported_steps =
                StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS]);
        }
        TransformerVariant::Dev => {
            out.capabilities.supports_negative_prompt = true;
            out.capabilities.supported_steps = StepSupport::Exact(vec![30]);
            // The non-distilled checkpoint has one native 30-step guided Euler pipeline.  The
            // curated menu swaps integrators over the baked distilled waypoints and is therefore
            // not a valid dev-checkpoint control.
            out.capabilities.samplers.clear();
        }
    }
    out
}
