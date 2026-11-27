//! Chroma family configuration (epic 3531), lifted from the diffusers `ChromaTransformer2DModel` /
//! `ChromaPipeline` reference (the parity source — mflux has no Chroma port).
//!
//! Chroma is a FLUX.1-schnell-derived DiT: same MMDiT skeleton (19 dual + 38 single blocks,
//! `inner_dim = 3072`, 24 heads × 128, FluxPosEmbed RoPE), but the FLUX `time_text_embed`
//! (timestep + guidance + pooled-CLIP) is replaced by a `distilled_guidance_layer` (Approximator)
//! that generates *all* per-block modulation, conditioning is **T5-XXL only** (no CLIP / no pooled),
//! and attention is **masked** by the T5 padding mask. Generation uses **true CFG** (real negative
//! prompt), not FLUX's distilled guidance.

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
    curated_sampler_names, curated_scheduler_names, Modality, ModelDescriptor, Quant,
};

pub const CHROMA1_HD_ID: &str = "chroma1_hd";
pub const CHROMA1_BASE_ID: &str = "chroma1_base";
pub const CHROMA1_FLASH_ID: &str = "chroma1_flash";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;

/// Flow-match Euler sampler name. Flash's unset sampler resolves to Heun in the generate path, but
/// Euler stays available for parity tests and manual fallback.
pub const DEFAULT_SAMPLER: &str = "euler";
pub const HEUN_SAMPLER: &str = "heun";

/// T5 sequence length Chroma conditions on (diffusers `_get_t5_prompt_embeds(max_sequence_length=512)`).
pub const MAX_SEQUENCE_LENGTH: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaVariant {
    /// `lodestones/Chroma1-HD` — the high-detail full-CFG model.
    Hd,
    /// `lodestones/Chroma1-Base` — the base full-CFG model.
    Base,
    /// `lodestones/Chroma1-Flash` — the few-step distilled model.
    Flash,
}

impl ChromaVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Hd => CHROMA1_HD_ID,
            Self::Base => CHROMA1_BASE_ID,
            Self::Flash => CHROMA1_FLASH_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Hd => "lodestones/Chroma1-HD",
            Self::Base => "lodestones/Chroma1-Base",
            Self::Flash => "lodestones/Chroma1-Flash",
        }
    }

    /// Default denoise steps. HD/Base run full CFG; Flash is a few-step distilled checkpoint. Exact
    /// values are reconciled against each repo's scheduler/README in sc-3840.
    pub fn default_steps(self) -> u32 {
        match self {
            Self::Hd | Self::Base => 28,
            Self::Flash => 12,
        }
    }

    /// Default true-CFG scale. Flash is distilled toward CFG≈1 (single forward); refined in sc-3840.
    pub fn default_true_cfg(self) -> f32 {
        match self {
            Self::Hd | Self::Base => 4.0,
            Self::Flash => 1.0,
        }
    }

    /// Static flow-match time `shift` (diffusers `FlowMatchEulerDiscreteScheduler`,
    /// `use_dynamic_shifting=false`): `σ' = shift·σ / (1 + (shift-1)·σ)`. HD's `scheduler_config.json`
    /// pins `shift=3.0`; Flash pins `1.0`. **Base** ships `use_beta_sigmas=true` (a beta-spaced
    /// schedule, not linspace — see [`crate::beta`] and [`Self::use_beta_sigmas`]); the beta schedule
    /// itself shapes the spacing, so Base's static `shift` is the identity `1.0` (not a placeholder —
    /// it is the value the reference applies after the beta sigmas).
    pub fn sigma_shift(self) -> f32 {
        match self {
            Self::Hd => 3.0,
            Self::Base | Self::Flash => 1.0,
        }
    }

    /// Base ships `use_beta_sigmas=true` — a beta-distribution sigma spacing (see [`crate::beta`])
    /// instead of the shifted linspace HD/Flash use.
    pub fn use_beta_sigmas(self) -> bool {
        matches!(self, Self::Base)
    }

    /// Primary typed route declaration for this Chroma variant (`REG-1`, `AP5`).
    pub fn route_record(self) -> RouteRecord {
        let mut samplers = curated_sampler_names();
        samplers.push("flow_match");
        let mut schedulers = curated_scheduler_names();
        schedulers.push("linear");

        RouteRecord {
            route_id: self.id(),
            family: "chroma",
            backend: "mlx",
            modality: Modality::Image,
            mac_only: true,
            structure: RouteStructure {
                text_encoders: TextEncoderSurface {
                    encoders: vec![TextEncoderDeclaration::new(
                        "t5_xxl",
                        Some(MAX_SEQUENCE_LENGTH as u32),
                    )],
                    ..Default::default()
                },
                conditioning: ConditioningSurface {
                    kinds: vec![],
                    control_kinds: None,
                    pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                    supports_negative_prompt: true,
                    modules: vec![],
                },
                sampling: SamplingSurface {
                    samplers,
                    schedulers,
                    guidance_methods: vec![],
                    supports_guidance: false,
                    supports_true_cfg: true,
                    supports_cfg_interval: false,
                    requires_sigma_shift: true,
                    supports_h3_image_sampler: false,
                },
                denoiser: ModelBinding::family_model(
                    FamilyModelSpec::new("chroma_mmdit").with_low_res_start(),
                ),
                latent_codec: LatentCodecDeclaration {
                    codec_id: Some("flux1_ae"),
                    output_latent_space: Some(&FLUX1_LATENT_SPACE),
                    pixel_stride: Some(crate::SIZE_MULTIPLE),
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
                        ResidencyPhaseNode::new(ResidencyPhaseKind::PromptEncode, ["t5_xxl"], true),
                        ResidencyPhaseNode::new(
                            ResidencyPhaseKind::Denoise,
                            ["chroma_mmdit"],
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
                guidance: None,
                sampler: Some(if matches!(self, Self::Flash) {
                    HEUN_SAMPLER
                } else {
                    DEFAULT_SAMPLER
                }),
                schedule: Some("linear"),
                shift: Some(self.sigma_shift()),
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

/// Static dims of `ChromaTransformer2DModel` (diffusers defaults — identical across the three
/// variants; only the weights and sampling profile differ).
#[derive(Clone, Copy, Debug)]
pub struct ChromaTransformerConfig {
    pub in_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub approximator_num_channels: usize,
    pub approximator_hidden_dim: usize,
    pub approximator_layers: usize,
}

impl Default for ChromaTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            num_layers: 19,
            num_single_layers: 38,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 4096,
            axes_dims_rope: [16, 56, 56],
            approximator_num_channels: 64,
            approximator_hidden_dim: 5120,
            approximator_layers: 5,
        }
    }
}

impl ChromaTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The modulation index length `out_dim = 3·N_single + 2·6·N_double + 2` — the number of
    /// `[inner_dim]` modulation rows the Approximator emits (`pooled_temb` has shape `[B, out_dim, inner]`).
    pub fn mod_index_len(&self) -> usize {
        3 * self.num_single_layers + 2 * 6 * self.num_layers + 2
    }
}
