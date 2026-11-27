//! The `minimax_h3` model route and descriptor.

use mlx_gen::gen_core::{
    AdapterInstallPolicy, AdapterSurface, ConditioningSurface, FamilyDriverSpec,
    LatentCodecDeclaration, ModelBinding, OutputSurface, Pass0ConditioningRequirement,
    PreviewSupport, ProductLimits, ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface,
    RouteBindings, RouteDefaults, RouteRecord, RouteStructure, SamplingSurface,
    SeedAndBatchSemantics, SeedScheme, SizeFloor, TextEncoderDeclaration, TextEncoderSurface,
    WeightPlanDeclaration, WeightSourceKind,
};

use super::*;

/// Primary typed route declaration for MiniMax-H3 AudioVideo (`REG-1`, `AP5`).
pub fn route() -> RouteRecord {
    RouteRecord {
        route_id: MODEL_ID,
        family: "minimax_h3",
        backend: "mlx",
        modality: Modality::Video,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("qwen3_vl_32b", None)],
                encoder_contract: None,
                max_prompt_tokens: None,
                supports_caption_overflow_policy: false,
                supports_prompt_enhancement: false,
            },
            conditioning: ConditioningSurface {
                kinds: vec![
                    ConditioningKind::Keyframe,
                    ConditioningKind::Reference,
                    ConditioningKind::ReferenceVideo,
                    ConditioningKind::ReferenceAudio,
                ],
                control_kinds: None,
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: false,
                modules: vec![],
            },
            sampling: SamplingSurface {
                samplers: vec![],
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: false,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: ModelBinding::family_driver(FamilyDriverSpec::new(
                crate::drivers::h3_driver_support(MODEL_ID),
            )),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("minimax_h3_video_vae"),
                output_latent_space: None,
                pixel_stride: Some(SPATIAL_STRIDE),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            adapters: AdapterSurface::lora_and_lokr(AdapterInstallPolicy::ForwardResidual),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::PromptEncode,
                        ["qwen3_vl_32b"],
                        true,
                    ),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Denoise, ["minimax_h3_dit"], true),
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::Decode,
                        ["minimax_h3_video_vae", "minimax_h3_audio_vae"],
                        false,
                    ),
                ],
                supports_sequential_offload: false,
                unconditionally_engages_staged_residency: true,
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: OutputSurface {
                preview: PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: SPATIAL_STRIDE,
            max_size: MAX_CANVAS_EDGE,
            size_floor: SizeFloor::RangeChecked,
            supported_steps: StepSupport::Range {
                min: 1,
                max: MAX_STEPS,
            },
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(DEFAULT_STEPS),
            guidance: None,
            sampler: None,
            schedule: None,
            shift: Some(VIDEO_SIGMA_SHIFT),
            strength: None,
            dimensions: Some((1280, 704)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(descriptor, |spec| load(spec).map_err(Into::into))
            .with_memory_registration(crate::memory_strategy::MEMORY_REGISTRATION)
            .with_memory_fixture_registration(crate::memory_strategy::MEMORY_CONTRACT_FIXTURE)
            .with_memory_behavior_registration(crate::memory_strategy::MEMORY_BEHAVIOR),
    }
}

/// The identity, modality and capability surface, computed from [`route`].
pub fn descriptor() -> ModelDescriptor {
    route().compute_descriptor()
}
