//! FLUX.1 family configuration, lifted from the frozen Python mflux fork's
//! `ModelConfig.{schnell,dev}` and `FluxWeightDefinition.get_tokenizers`.

use mlx_gen::gen_core::{
    AdapterInstallPolicy, AdapterSurface, ConditioningSurface, FamilyModelSpec,
    LatentCodecDeclaration, ModelBinding, OutputSurface, Pass0ConditioningRequirement,
    PreviewSupport, ProductLimits, ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface,
    RouteBindings, RouteDefaults, RouteRecord, RouteStructure, SamplingSurface,
    SeedAndBatchSemantics, SeedScheme, SizeFloor, StepSupport, TextEncoderDeclaration,
    TextEncoderSurface, UpsamplerSurface, WeightPlanDeclaration, WeightSourceKind,
    FLUX1_LATENT_SPACE,
};
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, ConditioningKind, Modality, ModelDescriptor,
    Quant,
};

pub const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
pub const FLUX1_DEV_ID: &str = "flux1_dev";
/// FLUX.1-dev **Fun-Controlnet-Union** variant (sc-8238/sc-8239): the dev base + the Shakker
/// `FLUX.1-dev-ControlNet-Union-Pro-2.0` control branch, registered as its own generator by E2.
pub const FLUX1_DEV_CONTROL_ID: &str = "flux1_dev_control";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
pub const DEFAULT_GUIDANCE: f32 = 3.5;

/// The base flow-match sampler name in the capability surface (sc-2908). An unset `req.sampler`
/// resolves to this — the standard FLUX flow-match Euler denoise over `build_linear_sigmas`.
pub const DEFAULT_SAMPLER: &str = "flow_match";
/// The Hyper-FLUX few-step acceleration profile (sc-2908): the SAME flow-match schedule at a reduced
/// step count (8) and guidance 3.5, paired with the ByteDance Hyper-FLUX 8-step LoRA loaded at
/// `scale≈0.125` (`spec.adapters`). FLUX.1-dev-only — it is a dev LoRA, so schnell never advertises
/// it. Selecting it without the LoRA loaded just runs 8 base steps (undertrained noise).
pub const HYPER_SAMPLER: &str = "hyper";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxVariant {
    Schnell,
    Dev,
}

impl FluxVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Schnell => FLUX1_SCHNELL_ID,
            Self::Dev => FLUX1_DEV_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Schnell => "black-forest-labs/FLUX.1-schnell",
            Self::Dev => "black-forest-labs/FLUX.1-dev",
        }
    }

    pub fn default_steps(self) -> u32 {
        match self {
            Self::Schnell => 4,
            Self::Dev => 25,
        }
    }

    pub fn max_sequence_length(self) -> usize {
        match self {
            Self::Schnell => 256,
            Self::Dev => 512,
        }
    }

    pub fn supports_guidance(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn requires_sigma_shift(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn pag_capability(self) -> mlx_gen::gen_core::PagCapability {
        let (default_stop, recommended_scale) = match self {
            Self::Schnell => (0.75, 1.2),
            Self::Dev => (1.0, 1.5),
        };
        mlx_gen::gen_core::PagCapability {
            num_blocks: 57,
            block_domain: std::borrow::Cow::Borrowed("flux_blocks"),
            default_layers: vec![28],
            default_start: 0.0,
            default_stop,
            recommended_scale,
            max_scale: mlx_gen::gen_core::DEFAULT_PAG_MAX_SCALE,
        }
    }

    /// Primary typed route declaration for this FLUX.1 base variant (`REG-1`, `AP5`).
    pub fn route_record(self) -> RouteRecord {
        let mut samplers = curated_sampler_names();
        samplers.push(DEFAULT_SAMPLER);
        if matches!(self, Self::Dev) {
            samplers.push(HYPER_SAMPLER);
        }
        let mut schedulers = curated_scheduler_names();
        schedulers.push("linear");

        RouteRecord {
            route_id: self.id(),
            family: "flux",
            backend: "mlx",
            modality: Modality::Image,
            mac_only: true,
            structure: RouteStructure {
                text_encoders: TextEncoderSurface {
                    encoders: vec![
                        TextEncoderDeclaration::new("clip_l", Some(77)),
                        TextEncoderDeclaration::new(
                            "t5_xxl",
                            Some(self.max_sequence_length() as u32),
                        ),
                    ],
                    ..Default::default()
                },
                conditioning: ConditioningSurface {
                    // FLUX.1 reference-image conditioning is the XLabs IP-Adapter (epic 3621): a
                    // single `Reference` rides `Conditioning::Reference { image, strength=ipAdapterScale }`,
                    // exactly as SDXL exposes its IP-Adapter. Only wired when a `LoadSpec::ip_adapter`
                    // is supplied at load time; a `Reference` request without it errors loudly.
                    kinds: vec![ConditioningKind::Reference],
                    control_kinds: None,
                    pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                    supports_negative_prompt: false,
                    modules: vec!["xlabs_ip_adapter"],
                    low_res_start_forbids_reference_conditioning: true,
                },
                sampling: SamplingSurface {
                    samplers,
                    schedulers,
                    guidance_methods: vec![],
                    supports_guidance: self.supports_guidance(),
                    supports_true_cfg: false,
                    supports_cfg_interval: false,
                    requires_sigma_shift: self.requires_sigma_shift(),
                    supports_h3_image_sampler: false,
                },
                denoiser: ModelBinding::family_model(
                    FamilyModelSpec::new("flux1_mmdit")
                        .with_low_res_start()
                        .with_pag(self.pag_capability()),
                ),
                latent_codec: LatentCodecDeclaration {
                    codec_id: Some("flux1_ae"),
                    output_latent_space: Some(&FLUX1_LATENT_SPACE),
                    pixel_stride: Some(crate::SIZE_MULTIPLE),
                    token_width: Some(crate::PACKED_TOKEN_WIDTH as u32),
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
                        ResidencyPhaseNode::new(
                            ResidencyPhaseKind::Denoise,
                            ["flux1_mmdit"],
                            false,
                        ),
                        ResidencyPhaseNode::new(ResidencyPhaseKind::Decode, ["flux1_ae"], false),
                    ],
                    supports_sequential_offload: true,
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
                    ..Default::default()
                },
                output: OutputSurface {
                    preview: PreviewSupport::LatentProjection,
                    supports_hdr: false,
                    supports_alpha_output: false,
                },
                ..Default::default()
            },
            limits: ProductLimits {
                min_size: 256,
                max_size: 2048,
                size_floor: SizeFloor::RangeChecked,
                supported_steps: StepSupport::Unconstrained,
                max_audio_duration_secs: None,
            },
            seed_and_batch: SeedAndBatchSemantics {
                seed_scheme: SeedScheme::BackendGlobalSeeded,
                max_count: 8,
                supports_prompt_batch: false,
            },
            defaults: RouteDefaults {
                steps: Some(self.default_steps()),
                guidance: self.supports_guidance().then_some(DEFAULT_GUIDANCE),
                sampler: Some(DEFAULT_SAMPLER),
                schedule: Some("linear"),
                shift: self.requires_sigma_shift().then_some(1.15),
                strength: None,
                dimensions: Some((DEFAULT_WIDTH, DEFAULT_HEIGHT)),
                ..Default::default()
            },
            bindings: RouteBindings::default(),
        }
    }

    pub fn descriptor(self) -> ModelDescriptor {
        self.route_record().compute_descriptor()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxTokenizerKind {
    Clip,
    T5,
}

impl FluxTokenizerKind {
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Clip => "tokenizer",
            Self::T5 => "tokenizer_2",
        }
    }

    pub fn max_length(self, variant: FluxVariant) -> usize {
        match self {
            Self::Clip => 77,
            Self::T5 => variant.max_sequence_length(),
        }
    }

    pub fn pad_token_id(self) -> i32 {
        match self {
            // CLIP's `<|endoftext|>` id in the FLUX.1 tokenizer.
            Self::Clip => 49407,
            // T5's `<pad>` id.
            Self::T5 => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pag_capability_matches_flux1_contract() {
        let dev = FluxVariant::Dev
            .descriptor()
            .capabilities
            .pag
            .expect("dev pag");
        assert_eq!(dev.num_blocks, 57);
        assert_eq!(dev.block_domain.as_ref(), "flux_blocks");
        assert_eq!(dev.default_layers, vec![28]);
        assert_eq!(dev.default_start, 0.0);
        assert_eq!(dev.default_stop, 1.0);
        assert_eq!(dev.recommended_scale, 1.5);
        assert_eq!(dev.max_scale, 15.0);
        assert_eq!(
            crate::model_control::descriptor_dev_control()
                .capabilities
                .pag,
            Some(dev)
        );

        let schnell = FluxVariant::Schnell
            .descriptor()
            .capabilities
            .pag
            .expect("schnell pag");
        assert_eq!(schnell.num_blocks, 57);
        assert_eq!(schnell.block_domain.as_ref(), "flux_blocks");
        assert_eq!(schnell.default_layers, vec![28]);
        assert_eq!(schnell.default_start, 0.0);
        assert_eq!(schnell.default_stop, 0.75);
        assert_eq!(schnell.recommended_scale, 1.2);
        assert_eq!(schnell.max_scale, 15.0);
    }
}
