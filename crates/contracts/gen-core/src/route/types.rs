//! Primary typed declaration types for [`RouteRecord`] (`REG-1`, `AP5`).

use serde::Serialize;

use crate::control::AcceptedControlKinds;
use crate::driver::DriverSupport;
use crate::encoder_contract::EncoderContract;
use crate::feature::Pass0ConditioningRequirement;
use crate::generator::{
    ActivationMemoryAnchor, AudioEditMode, ComponentPrecisionFloor, ConditioningKind,
    FamilyExtensionKind, GenerationRequest, Generator, Modality, ModelDescriptor, SizeFloor,
    StepSupport,
};
use crate::latent::{LatentPatchLayout, LatentSpace};
use crate::memory_strategy::{
    MemoryProviderContract, MemoryReferenceResourceFacts, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy,
};
use crate::pag::PagCapability;
use crate::registry::{
    MemoryBehaviorBeginRequest, MemoryBehaviorFixture, MemoryBehaviorRegistration,
    MemoryContractFixtureRegistration, MemoryContractSurfaceResolverRegistration,
    MemoryContractSurfaceSpec, MemoryRegistration, PerComponentBytes,
    ResidentOnlyMemoryContractRegistration,
};
use crate::render::video_plan::VideoPlanRegistration;
use crate::render::RenderPlan;
use crate::runtime::{LoadSpec, Quant};
use crate::{ApproximationSurface, ExecutionSurface, Result};

// -------------------------------------------------------------------------------------------------
// Denoiser / Model Binding (`REG-1`: `FamilyModel` or `FamilyDriver`, plus `NonDiffusion`)
// -------------------------------------------------------------------------------------------------

/// Primary behavioural declaration for a standard denoiser driven by shared solver/scheduler code
/// (`ModelBinding::FamilyModel`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyModelSpec {
    /// Stable architecture/model identifier (e.g. `"flux1_mmdit"`, `"sdxl_unet"`).
    pub model_id: &'static str,
    /// Supports changing spatial geometry across phase boundaries (e.g. `low_res_start`).
    pub supports_geometry_change: bool,
    /// Supports entering a phase by re-noising a clean estimate (`BoundaryState::CleanEstimate`).
    pub supports_renoise_from_clean: bool,
    /// Supports read-only boundary clean-estimate evaluation without mutating state, RNG, or progress.
    pub supports_read_only_boundary_estimate: bool,
    /// Supports multi-phase denoise (`GenerationRequest::phases`).
    pub supports_multiphase_denoise: bool,
    /// Supports MoE expert switching across denoise phases (e.g. Wan A14B).
    pub supports_expert_switching: bool,
    /// Perturbed-Attention Guidance structural capability on this denoiser's attention blocks.
    pub pag: Option<PagCapability>,
}

impl FamilyModelSpec {
    /// Construct a base [`FamilyModelSpec`] with all optional phase/PAG capabilities off.
    pub const fn new(model_id: &'static str) -> Self {
        Self {
            model_id,
            supports_geometry_change: false,
            supports_renoise_from_clean: false,
            supports_read_only_boundary_estimate: false,
            supports_multiphase_denoise: false,
            supports_expert_switching: false,
            pag: None,
        }
    }

    /// Declare full low-res-start boundary transition support (geometry change + re-noise from
    /// clean estimate + read-only boundary estimate).
    pub const fn with_low_res_start(mut self) -> Self {
        self.supports_geometry_change = true;
        self.supports_renoise_from_clean = true;
        self.supports_read_only_boundary_estimate = true;
        self
    }

    /// Declare multi-phase denoise (`req.phases`) support.
    pub const fn with_multiphase_denoise(mut self) -> Self {
        self.supports_multiphase_denoise = true;
        self
    }

    /// Declare MoE expert-switching support across phase boundaries.
    pub const fn with_expert_switching(mut self) -> Self {
        self.supports_expert_switching = true;
        self
    }

    /// Attach a [`PagCapability`] declaration for this denoiser's attention blocks.
    pub fn with_pag(mut self, pag: PagCapability) -> Self {
        self.pag = Some(pag);
        self
    }
}

/// Primary behavioural declaration for an essential bespoke denoising loop implementing
/// [`crate::FamilyDriver`] (`ModelBinding::FamilyDriver`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyDriverSpec {
    /// The driver's primary [`DriverSupport`] declaration verified in Tier 0.
    pub support: DriverSupport,
    /// Supports multi-phase denoise (`GenerationRequest::phases`).
    pub supports_multiphase_denoise: bool,
    /// Supports MoE expert switching across denoise phases.
    pub supports_expert_switching: bool,
    /// Perturbed-Attention Guidance structural capability on this driver's denoiser blocks.
    pub pag: Option<PagCapability>,
}

impl FamilyDriverSpec {
    /// Construct a [`FamilyDriverSpec`] wrapping a primary [`DriverSupport`].
    pub const fn new(support: DriverSupport) -> Self {
        Self {
            support,
            supports_multiphase_denoise: false,
            supports_expert_switching: false,
            pag: None,
        }
    }

    /// Declare multi-phase denoise (`req.phases`) support.
    pub const fn with_multiphase_denoise(mut self) -> Self {
        self.supports_multiphase_denoise = true;
        self
    }

    /// Declare MoE expert-switching support across phase boundaries.
    pub const fn with_expert_switching(mut self) -> Self {
        self.supports_expert_switching = true;
        self
    }

    /// Attach a [`PagCapability`] declaration for this driver's denoiser blocks.
    pub fn with_pag(mut self, pag: PagCapability) -> Self {
        self.pag = Some(pag);
        self
    }
}

/// Primary denoiser/model execution binding for a route (`REG-1`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelBinding {
    /// Standard diffusion/flow model executed through shared solver/scheduler loops.
    FamilyModel(FamilyModelSpec),
    /// Bespoke denoising loop executed through [`crate::FamilyDriver`].
    FamilyDriver(FamilyDriverSpec),
    /// Non-diffusion generative route (e.g. StyleTTS2/autoregressive audio synthesis).
    NonDiffusion { architecture: &'static str },
}

impl Default for ModelBinding {
    fn default() -> Self {
        Self::NonDiffusion {
            architecture: "unspecified",
        }
    }
}

impl ModelBinding {
    /// Convenience constructor for [`Self::FamilyModel`].
    pub fn family_model(spec: FamilyModelSpec) -> Self {
        Self::FamilyModel(spec)
    }

    /// Convenience constructor for [`Self::FamilyDriver`].
    pub fn family_driver(spec: FamilyDriverSpec) -> Self {
        Self::FamilyDriver(spec)
    }

    /// Convenience constructor for [`Self::NonDiffusion`].
    pub const fn non_diffusion(architecture: &'static str) -> Self {
        Self::NonDiffusion { architecture }
    }

    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::FamilyModel(_) => "family_model",
            Self::FamilyDriver(_) => "family_driver",
            Self::NonDiffusion { .. } => "non_diffusion",
        }
    }

    pub const fn supports_geometry_change(&self) -> bool {
        match self {
            Self::FamilyModel(spec) => spec.supports_geometry_change,
            Self::FamilyDriver(spec) => spec.support.geometry_change,
            Self::NonDiffusion { .. } => false,
        }
    }

    pub const fn supports_renoise_from_clean(&self) -> bool {
        match self {
            Self::FamilyModel(spec) => spec.supports_renoise_from_clean,
            Self::FamilyDriver(spec) => spec.support.renoise_from_clean,
            Self::NonDiffusion { .. } => false,
        }
    }

    pub const fn supports_read_only_boundary_estimate(&self) -> bool {
        match self {
            Self::FamilyModel(spec) => spec.supports_read_only_boundary_estimate,
            Self::FamilyDriver(spec) => spec.support.read_only_boundary_estimate,
            Self::NonDiffusion { .. } => false,
        }
    }

    /// Whether the denoiser binding supports the complete clean-estimate spatial resize + re-noise
    /// boundary transition required by `low_res_start`.
    pub const fn supports_low_res_transition(&self) -> bool {
        self.supports_geometry_change()
            && self.supports_renoise_from_clean()
            && self.supports_read_only_boundary_estimate()
    }

    pub const fn supports_multiphase_denoise(&self) -> bool {
        match self {
            Self::FamilyModel(spec) => spec.supports_multiphase_denoise,
            Self::FamilyDriver(spec) => spec.supports_multiphase_denoise,
            Self::NonDiffusion { .. } => false,
        }
    }

    pub const fn supports_expert_switching(&self) -> bool {
        match self {
            Self::FamilyModel(spec) => spec.supports_expert_switching,
            Self::FamilyDriver(spec) => spec.supports_expert_switching,
            Self::NonDiffusion { .. } => false,
        }
    }

    pub fn pag(&self) -> Option<&PagCapability> {
        match self {
            Self::FamilyModel(spec) => spec.pag.as_ref(),
            Self::FamilyDriver(spec) => spec.pag.as_ref(),
            Self::NonDiffusion { .. } => None,
        }
    }

    pub fn driver_support(&self) -> Option<&DriverSupport> {
        match self {
            Self::FamilyDriver(spec) => Some(&spec.support),
            Self::FamilyModel(_) | Self::NonDiffusion { .. } => None,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Structural Component Declarations (`REG-1`)
// -------------------------------------------------------------------------------------------------

/// One text encoder component attached to a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TextEncoderDeclaration {
    /// Stable component role/name (e.g. `"clip_l"`, `"t5_xxl"`, `"gemma3"`, `"albert"`).
    pub name: &'static str,
    /// Optional token context length padded/consumed by this encoder (e.g. `Some(77)`, `Some(512)`).
    pub max_sequence_length: Option<u32>,
}

impl TextEncoderDeclaration {
    pub const fn new(name: &'static str, max_sequence_length: Option<u32>) -> Self {
        Self {
            name,
            max_sequence_length,
        }
    }
}

/// Primary text-encoder and prompt-policy declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TextEncoderSurface {
    /// Ordered text encoders participating in prompt conditioning.
    pub encoders: Vec<TextEncoderDeclaration>,
    /// Substitutable text-encoder contract (`ModelDescriptor::encoder_contract`).
    #[serde(skip)]
    pub encoder_contract: Option<EncoderContract>,
    /// Advisory longest prompt in model tokens (`Capabilities::max_prompt_tokens`).
    pub max_prompt_tokens: Option<u32>,
    /// Optional provider-specific prompt-enhancement token ceiling (`Capabilities::max_enhance_tokens`).
    /// `None` uses the shared default [`crate::generator::MAX_ENHANCE_TOKENS`] (`2048`).
    pub max_enhance_tokens: Option<u32>,
    /// Whether `GenerationRequest::caption_overflow` is honored.
    pub supports_caption_overflow_policy: bool,
    /// Whether `GenerationRequest::enhance_prompt` modifies the consumed prompt.
    pub supports_prompt_enhancement: bool,
}

/// Primary conditioning-module and pass-0 conditioning declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ConditioningSurface {
    /// Accepted conditioning input kinds (`Capabilities::conditioning`).
    #[serde(skip)]
    pub kinds: Vec<ConditioningKind>,
    /// Accepted `ControlKind`s when control conditioning is wired (`ModelDescriptor::control_kinds`).
    #[serde(skip)]
    pub control_kinds: Option<AcceptedControlKinds>,
    /// Primary pass-0 conditioning requirement used by `HOOK-4` applicability derivation.
    pub pass0_requirement: Pass0ConditioningRequirement,
    /// When `true`, single-reference `Conditioning::Reference` img2img cannot combine with
    /// `low_res_start` (the route requires `GenerationRequest::source_image` instead). Replaces
    /// shared model-id string heuristics (`REG-4`).
    pub low_res_start_forbids_reference_conditioning: bool,
    /// Whether this route honors `GenerationRequest::negative_prompt`.
    pub supports_negative_prompt: bool,
    /// Named conditioning modules wired into this route (e.g. `"xlabs_ip_adapter"`, `"flux_controlnet_union"`).
    pub modules: Vec<&'static str>,
}

/// Primary sampler, scheduler, and guidance axis declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SamplingSurface {
    /// Selectable sampler names (`Capabilities::samplers`).
    pub samplers: Vec<&'static str>,
    /// Selectable scheduler names (`Capabilities::schedulers`).
    pub schedulers: Vec<&'static str>,
    /// Selectable guidance method names (`Capabilities::supported_guidance_methods`).
    pub guidance_methods: Vec<&'static str>,
    /// Whether `GenerationRequest::guidance` is honored.
    pub supports_guidance: bool,
    /// Whether `GenerationRequest::true_cfg` is honored.
    pub supports_true_cfg: bool,
    /// Whether `GenerationRequest::cfg_interval` is honored.
    pub supports_cfg_interval: bool,
    /// Whether this route's schedule requires sigma shift.
    pub requires_sigma_shift: bool,
    /// Whether `GenerationRequest::h3_image_sampler` is honored.
    pub supports_h3_image_sampler: bool,
}

/// Primary latent-codec declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LatentCodecDeclaration {
    /// Optional codec identifier (e.g. `"flux1_ae"`, `"sdxl_vae"`, `"kokoro_istftnet"`).
    pub codec_id: Option<&'static str>,
    /// Exact latent tensor emitted by the denoiser at its decoder boundary.
    #[serde(skip)]
    pub output_latent_space: Option<&'static LatentSpace>,
    /// Packed 2D image trajectory stride in pixels (e.g. `Some(16)` for FLUX.1).
    pub pixel_stride: Option<u32>,
    /// Packed 2D image trajectory token width (e.g. `Some(64)` for FLUX.1).
    pub token_width: Option<u32>,
    /// Whether this route can decode through a staged diffusion VAE decoder (DiffVAE).
    pub supports_diffusion_decoder: bool,
}

impl LatentCodecDeclaration {
    /// Resolve the effective spatial pixel stride from explicit declaration, `SizeFloor`, or `LatentSpace`.
    pub fn effective_pixel_stride(&self, size_floor: SizeFloor) -> Option<u32> {
        self.pixel_stride
            .or_else(|| size_floor.explicit_size_multiple())
            .or_else(|| {
                self.output_latent_space.map(|space| match space.patch_layout {
                    LatentPatchLayout::Packed { patch_width, .. } => {
                        u32::from(space.spatial_compression.width) * u32::from(patch_width.max(1))
                    }
                    LatentPatchLayout::Unpacked => u32::from(space.spatial_compression.width) * 2,
                })
            })
    }

    /// Resolve the effective packed token width from explicit declaration or packed `LatentSpace`.
    pub fn effective_token_width(&self) -> Option<u32> {
        self.token_width.or_else(|| {
            self.output_latent_space.and_then(|space| match space.patch_layout {
                LatentPatchLayout::Packed { .. } => Some(u32::from(space.channels)),
                LatentPatchLayout::Unpacked => None,
            })
        })
    }

    /// Whether this codec + packed trajectory geometry supports isotropic 2D spatial unpack/repack.
    pub fn supports_spatial_unpack_repack(&self) -> bool {
        let Some(space) = self.output_latent_space else {
            return false;
        };
        if !space.is_well_formed()
            || space.spatial_compression.width != space.spatial_compression.height
        {
            return false;
        }
        let vae_scale = u32::from(space.spatial_compression.width);
        match space.patch_layout {
            LatentPatchLayout::Packed {
                patch_height,
                patch_width,
            } => {
                if patch_height != patch_width || patch_width == 0 {
                    return false;
                }
                let default_stride = vae_scale * u32::from(patch_width);
                let default_width = u32::from(space.channels);
                self.pixel_stride.unwrap_or(default_stride) > 0
                    && self.token_width.unwrap_or(default_width) > 0
            }
            LatentPatchLayout::Unpacked => {
                let ch = u32::from(space.channels);
                let Some(stride) = self.pixel_stride else {
                    return false;
                };
                let Some(width) = self.token_width else {
                    return false;
                };
                stride >= vae_scale
                    && stride.is_multiple_of(vae_scale)
                    && {
                        let ratio = stride / vae_scale;
                        width == ch || width == ch * ratio * ratio
                    }
            }
        }
    }
}

/// Primary upsampler and auxiliary component declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct UpsamplerSurface {
    /// Named upsampler or post-latent components (e.g. `"pid_decoder"`, `"ltx_spatial_upsampler"`).
    pub components: Vec<&'static str>,
    /// Largest `GenerationRequest::temporal_upsample_rounds` honored (`Capabilities::max_temporal_upsample_rounds`).
    pub max_temporal_upsample_rounds: u32,
}

/// How runtime LoRA/LoKr adapters are installed onto a route's denoiser (`REG-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterInstallPolicy {
    /// No runtime adapter installation supported; non-empty `LoadSpec::adapters` is refused.
    #[default]
    Unsupported,
    /// Installed as forward-time residuals over base (dense or quantized) linear projections.
    ForwardResidual,
    /// Merged into dense base weights at load time.
    FusedMerge,
    /// Installed on the base denoiser only while auxiliary branches (e.g. ControlNet) stay un-adapted.
    BaseDenoiserOnly,
}

/// Primary adapter surface and install policy for a route (`REG-1`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AdapterSurface {
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub install_policy: AdapterInstallPolicy,
}

impl AdapterSurface {
    pub const fn lora_and_lokr(install_policy: AdapterInstallPolicy) -> Self {
        Self {
            supports_lora: true,
            supports_lokr: true,
            install_policy,
        }
    }

    pub const fn unsupported() -> Self {
        Self {
            supports_lora: false,
            supports_lokr: false,
            install_policy: AdapterInstallPolicy::Unsupported,
        }
    }
}

/// Logical lifecycle phase in a route's residency lifetime graph (`REG-1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyPhaseKind {
    PromptEncode,
    ConditioningEncode,
    Denoise,
    Upsample,
    Decode,
    AudioSynthesize,
}

/// One node in a route's ordered residency lifetime graph (`REG-1`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResidencyPhaseNode {
    pub phase: ResidencyPhaseKind,
    pub components: Vec<&'static str>,
    /// Whether components owned by this phase can be dropped before the next phase loads.
    pub droppable_after_phase: bool,
}

impl ResidencyPhaseNode {
    pub fn new(
        phase: ResidencyPhaseKind,
        components: impl Into<Vec<&'static str>>,
        droppable_after_phase: bool,
    ) -> Self {
        Self {
            phase,
            components: components.into(),
            droppable_after_phase,
        }
    }
}

/// Primary residency lifetime graph and offload surface (`REG-1`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ResidencySurface {
    /// Ordered residency lifetime graph of phases and resident components.
    pub lifetime_graph: Vec<ResidencyPhaseNode>,
    /// Whether this route honors `OffloadPolicy::Sequential`.
    pub supports_sequential_offload: bool,
    /// Whether every generation physically stages eligible heavyweight components unconditionally.
    pub unconditionally_engages_staged_residency: bool,
    /// Whether this route uses KV-cache residency.
    pub supports_kv_cache: bool,
    /// Per-request `GenerationMemory` rungs honored without a `MemoryProviderContract` (audio lane).
    #[serde(skip)]
    pub request_memory_strategies: &'static [MemoryStrategy],
}

/// Primary on-disk weight source shape accepted by a route's loader (`REG-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightSourceKind {
    /// Component snapshot directory (`WeightsSource::Dir`).
    #[default]
    SnapshotDirectory,
    /// Single checkpoint file (`WeightsSource::File`).
    SingleFile,
    /// Accepts either a snapshot directory or a single checkpoint file.
    DirectoryOrFile,
}

/// Primary weight plan and quantization contract for a route (`REG-1`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct WeightPlanDeclaration {
    pub source_kind: WeightSourceKind,
    /// Exact native quantization tiers loadable by this route (`Capabilities::supported_quants`).
    #[serde(skip)]
    pub supported_quants: &'static [Quant],
    /// Component-local precision floors above the selected tier.
    #[serde(skip)]
    pub component_precision_floors: &'static [ComponentPrecisionFloor],
    /// Named extra artifacts callers must stage in `LoadSpec::components`.
    pub required_components: &'static [&'static str],
    /// Whether this route requires a control checkpoint in `LoadSpec::control`.
    pub requires_control_overlay: bool,
    /// Whether this route requires an IP-Adapter checkpoint in `LoadSpec::ip_adapter`.
    pub requires_ip_adapter_overlay: bool,
    /// Whether this route requires identity weights in `LoadSpec::identity`.
    pub requires_identity_overlay: bool,
    /// Whether this route honors autoregressive load-time mode/compute policies (e.g. YuE2 AR mode
    /// and compute policy). Replaces model-id string gating in registry load admission (`REG-4`).
    pub supports_ar_load_controls: bool,
}

/// How intermediate denoise previews are produced by a route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewSupport {
    /// Intermediate denoise previews are not emitted.
    #[default]
    Unsupported,
    /// Emitted by projecting the denoiser's intermediate latent tensor to RGB.
    LatentProjection,
    /// Emitted directly from a pixel-space denoiser trajectory (no VAE projection).
    PixelSpace,
}

impl PreviewSupport {
    pub const fn is_supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Primary output-channel, HDR, and preview declaration for a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct OutputSurface {
    pub preview: PreviewSupport,
    pub supports_hdr: bool,
    pub supports_alpha_output: bool,
}

/// Primary video-specific generation axes (e.g. LTX 2.5 duration head and generated keyframes).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VideoSurface {
    pub supports_auto_duration: bool,
    pub supports_generated_keyframes: bool,
}

/// Primary audio synthesis, streaming, dialogue, and song-planning surface.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AudioSurface {
    pub sample_rates: Vec<u32>,
    pub voices: Vec<&'static str>,
    pub languages: Vec<&'static str>,
    #[serde(skip)]
    pub edit_modes: Vec<AudioEditMode>,
    pub supports_streaming: bool,
    pub supports_multi_speaker: bool,
    pub max_speakers: Option<u32>,
    pub supports_segmented_lyrics: bool,
    pub supports_repetition_penalty: bool,
    pub supports_reference_region: bool,
    pub supports_output_limiter: bool,
    pub supports_symbolic_song: bool,
    pub supports_audio_artifacts: bool,
    pub supports_song_plan_only: bool,
    pub supports_song_cover: bool,
    pub supports_conversation_history: bool,
    pub supports_conversation_session: bool,
}

/// Complete structural composition of one route (`REG-1`, `AP5`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RouteStructure {
    pub text_encoders: TextEncoderSurface,
    pub conditioning: ConditioningSurface,
    pub sampling: SamplingSurface,
    pub denoiser: ModelBinding,
    pub latent_codec: LatentCodecDeclaration,
    pub upsamplers: UpsamplerSurface,
    pub adapters: AdapterSurface,
    pub residency: ResidencySurface,
    pub weight_plan: WeightPlanDeclaration,
    pub output: OutputSurface,
    pub video: VideoSurface,
    pub audio: AudioSurface,
    /// Typed family request extensions explicitly supported by this route (`REG-4`).
    pub supported_family_extensions: Vec<FamilyExtensionKind>,
    #[serde(skip)]
    pub execution: ExecutionSurface,
    #[serde(skip)]
    pub approximation: ApproximationSurface,
}

// -------------------------------------------------------------------------------------------------
// Product Limits, Seed/Batch Semantics, and Model Defaults (`REG-1`)
// -------------------------------------------------------------------------------------------------

/// Primary product limits for a route (`REG-1`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ProductLimits {
    /// Minimum spatial dimension in pixels (`0` for `Modality::Audio`).
    pub min_size: u32,
    /// Maximum spatial dimension in pixels (`0` for `Modality::Audio`).
    pub max_size: u32,
    /// Spatial size enforcement floor and grid policy.
    #[serde(skip)]
    pub size_floor: SizeFloor,
    /// Legal denoise step counts (`Unconstrained`, `Exact`, or `Range`).
    #[serde(skip)]
    pub supported_steps: StepSupport,
    /// Longest audio clip advertised in seconds (`None` for visual or uncapped).
    pub max_audio_duration_secs: Option<f32>,
}

/// Primary seed derivation scheme for a route (`REG-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedScheme {
    /// Deterministic per-sample seed (`base_seed + sample_index`) on portable CPU/host RNG.
    #[default]
    DeterministicOffset,
    /// Deterministic backend-global RNG seeded per trajectory (e.g. MLX `random::seed`).
    BackendGlobalSeeded,
    /// Deterministic per-request synthesis without stochastic latent noise (e.g. StyleTTS2).
    DeterministicSynthesis,
}

/// Primary seed, count, and prompt-batch semantics for a route (`REG-1`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SeedAndBatchSemantics {
    pub seed_scheme: SeedScheme,
    /// Largest `GenerationRequest::count` admitted (`Capabilities::max_count`, must be `>= 1`).
    pub max_count: u32,
    /// Whether `GenerationRequest::prompt_batch` renders as one batched denoise.
    pub supports_prompt_batch: bool,
}

impl Default for SeedAndBatchSemantics {
    fn default() -> Self {
        Self {
            seed_scheme: SeedScheme::DeterministicOffset,
            max_count: 1,
            supports_prompt_batch: false,
        }
    }
}

/// Primary model defaults for a route (`REG-1`: native steps, guidance, schedule, shift, strength).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RouteDefaults {
    /// Native default denoise step count (e.g. `Some(4)` for FLUX.1-schnell, `Some(25)` for FLUX.1-dev).
    pub steps: Option<u32>,
    /// Native default guidance scale (e.g. `Some(3.5)` for FLUX.1-dev).
    pub guidance: Option<f32>,
    /// Native default sampler name (e.g. `Some("flow_match")` or `Some("euler")`).
    pub sampler: Option<&'static str>,
    /// Native default schedule / scheduler name (e.g. `Some("linear")`).
    pub schedule: Option<&'static str>,
    /// Native default schedule shift (e.g. `Some(1.15)` or `Some(3.0)`).
    pub shift: Option<f32>,
    /// Native default denoising/control/reference strength in `0.0..=1.0`.
    pub strength: Option<f32>,
    /// Native default spatial `(width, height)` in pixels.
    pub dimensions: Option<(u32, u32)>,
    /// Native default audio voice id (e.g. `Some("af_heart")`).
    pub audio_voice: Option<&'static str>,
    /// Native default audio sample rate in Hz (e.g. `Some(24_000)`).
    pub audio_sample_rate: Option<u32>,
    /// Native default audio language code (e.g. `Some("en")`).
    pub audio_language: Option<&'static str>,
}

// -------------------------------------------------------------------------------------------------
// Executable & Registry Bindings (`REG-1`: one call registers generator, memory, fixtures, behavior)
// -------------------------------------------------------------------------------------------------

/// Pre-load render-plan compiler function pointer.
pub type RenderPlanCompiler = fn(&LoadSpec, &GenerationRequest) -> Result<Option<RenderPlan>>;

/// Structural scope of a route's pre-load [`RenderPlanCompiler`] binding (`HOOK-4`, `AP5`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderPlanScope {
    /// Compiles ordinary single-pass / pass-0 packed trajectories and (when `Reference`
    /// conditioning is declared) multi-pass produced-reference trajectories.
    #[default]
    SingleAndMultiPass,
    /// Compiles only explicit multi-phase denoise requests (`req.phases.is_some()`, e.g.
    /// `krea_multiphase`) and returns `Ok(None)` for ordinary single-phase / multi-pass requests.
    MultiphaseOnly,
}

/// Standalone generator factory binding attached to a [`RouteRecord`].
#[derive(Clone, Copy)]
pub struct RouteGeneratorBinding {
    /// Static descriptor function exported by the provider crate.
    /// Build-time conformance verifies `(descriptor)() == route.compute_descriptor()`.
    pub descriptor: fn() -> ModelDescriptor,
    /// Lazy generator constructor registered into [`crate::ProviderRegistry`].
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    /// Optional per-component on-disk footprint estimator.
    pub footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>,
}

impl std::fmt::Debug for RouteGeneratorBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteGeneratorBinding")
            .field("descriptor", &"<fn>")
            .field("load", &"<fn>")
            .field("footprint", &self.footprint.map(|_| "<fn>"))
            .finish()
    }
}

/// Conditioned / composed feature sub-route binding attached to a [`RouteRecord`].
#[derive(Clone, Copy)]
pub struct RouteFeatureBinding {
    /// Owning parent provider id (e.g. `Some("flux1_dev")` for `flux1_dev_ip_adapter`).
    pub parent_provider_id: Option<&'static str>,
    /// Static descriptor function for this feature route.
    pub descriptor: fn() -> ModelDescriptor,
}

impl std::fmt::Debug for RouteFeatureBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteFeatureBinding")
            .field("parent_provider_id", &self.parent_provider_id)
            .field("descriptor", &"<fn>")
            .finish()
    }
}

/// Memory contract builder function pointer.
pub type MemoryContractFn = fn(&LoadSpec) -> Result<MemoryProviderContract>;
/// Memory safety-check function pointer.
pub type MemorySafetyCheckFn =
    fn(&LoadSpec, &MemoryProviderContract, &MemoryRunContext) -> MemorySafetyDecision;
/// Memory surface-specs enumerator function pointer.
pub type MemorySurfaceSpecsFn = fn() -> Vec<MemoryContractSurfaceSpec>;
/// Selector-aware memory surface resolver function pointer.
pub type MemorySurfaceResolverFn = fn(&MemoryContractSurfaceSpec) -> Result<MemoryProviderContract>;
/// Memory behavior valid-fixture generator function pointer.
pub type MemoryValidFixturesFn =
    fn(&LoadSpec, &MemoryProviderContract, MemoryStrategy) -> Result<Vec<MemoryBehaviorFixture>>;

/// Bound memory contract entry `(optional_explicit_provider_id, contract, safety_check)`.
pub type RouteMemoryContractEntry = (
    Option<&'static str>,
    MemoryContractFn,
    MemorySafetyCheckFn,
);
/// Bound memory fixture / resident-only entry `(optional_explicit_provider_id, surface_specs, contract)`.
pub type RouteMemoryFixtureEntry = (
    Option<&'static str>,
    MemorySurfaceSpecsFn,
    MemoryContractFn,
);
/// Bound selector-aware memory surface resolver entry `(optional_explicit_provider_id, resolver)`.
pub type RouteMemoryResolverEntry = (Option<&'static str>, MemorySurfaceResolverFn);
/// Bound memory behavior entry `(optional_explicit_provider_id, valid_fixtures, begin_request)`.
pub type RouteMemoryBehaviorEntry = (
    Option<&'static str>,
    MemoryValidFixturesFn,
    MemoryBehaviorBeginRequest,
);

/// Memory contract, weights-free fixture, surface resolver, resident-only witness, and behavior
/// bindings attached to a [`RouteRecord`].
///
/// Any `provider_id` field left `None` automatically binds to `RouteRecord::route_id` on
/// registration (`AP5`). Explicit registrations (`with_*_registration`) are verified to match
/// `RouteRecord::route_id`.
#[derive(Clone, Copy, Default)]
pub struct RouteMemoryBindings {
    pub contract: Option<RouteMemoryContractEntry>,
    pub composed: bool,
    pub fixture: Option<RouteMemoryFixtureEntry>,
    pub surface_resolver: Option<RouteMemoryResolverEntry>,
    pub resident_only: Option<RouteMemoryFixtureEntry>,
    pub behavior: Option<RouteMemoryBehaviorEntry>,
}

impl RouteMemoryBindings {
    pub fn is_empty(&self) -> bool {
        self.contract.is_none()
            && self.fixture.is_none()
            && self.surface_resolver.is_none()
            && self.resident_only.is_none()
            && self.behavior.is_none()
    }
}

impl std::fmt::Debug for RouteMemoryBindings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteMemoryBindings")
            .field("has_contract", &self.contract.is_some())
            .field("composed", &self.composed)
            .field("has_fixture", &self.fixture.is_some())
            .field("has_surface_resolver", &self.surface_resolver.is_some())
            .field("has_resident_only", &self.resident_only.is_some())
            .field("has_behavior", &self.behavior.is_some())
            .finish()
    }
}

/// Complete executable and registry bindings registered atomically by
/// [`crate::ProviderRegistryBuilder::register_route`].
#[derive(Clone, Default)]
pub struct RouteBindings {
    pub generator: Option<RouteGeneratorBinding>,
    pub render_plan: Option<RenderPlanCompiler>,
    pub render_plan_scope: RenderPlanScope,
    pub video_plan: Option<VideoPlanRegistration>,
    pub feature_route: Option<RouteFeatureBinding>,
    pub encoder_contract_parent: Option<&'static str>,
    pub memory: RouteMemoryBindings,
    pub activation_memory: Option<ActivationMemoryAnchor>,
    pub reference_workspace: Option<fn(&LoadSpec, &MemoryReferenceResourceFacts) -> Result<u64>>,
}

impl std::fmt::Debug for RouteBindings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteBindings")
            .field("generator", &self.generator)
            .field("has_render_plan", &self.render_plan.is_some())
            .field("render_plan_scope", &self.render_plan_scope)
            .field("has_video_plan", &self.video_plan.is_some())
            .field("feature_route", &self.feature_route)
            .field("encoder_contract_parent", &self.encoder_contract_parent)
            .field("memory", &self.memory)
            .field("activation_memory", &self.activation_memory)
            .field("has_reference_workspace", &self.reference_workspace.is_some())
            .finish()
    }
}

impl RouteBindings {
    /// Start bindings for a standalone registered generator route.
    pub fn generator(
        descriptor: fn() -> ModelDescriptor,
        load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    ) -> Self {
        Self {
            generator: Some(RouteGeneratorBinding {
                descriptor,
                load,
                footprint: None,
            }),
            ..Default::default()
        }
    }

    /// Start bindings for a conditioned / bespoke feature sub-route (no standalone generator).
    pub fn feature_route(
        parent_provider_id: Option<&'static str>,
        descriptor: fn() -> ModelDescriptor,
    ) -> Self {
        Self {
            feature_route: Some(RouteFeatureBinding {
                parent_provider_id,
                descriptor,
            }),
            ..Default::default()
        }
    }

    /// Attach a per-component on-disk footprint estimator to the generator binding.
    pub fn with_footprint(mut self, footprint: fn(&LoadSpec) -> Result<PerComponentBytes>) -> Self {
        if let Some(gen) = self.generator.as_mut() {
            gen.footprint = Some(footprint);
        }
        self
    }

    /// Attach a pre-load render-plan compiler covering single-pass and multi-pass trajectories.
    pub fn with_render_plan(mut self, compile: RenderPlanCompiler) -> Self {
        self.render_plan = Some(compile);
        self.render_plan_scope = RenderPlanScope::SingleAndMultiPass;
        self
    }

    /// Attach a pre-load render-plan compiler that compiles only explicit multi-phase requests
    /// (`req.phases.is_some()`) and returns `Ok(None)` for ordinary single-phase / multi-pass
    /// requests.
    pub fn with_multiphase_render_plan(mut self, compile: RenderPlanCompiler) -> Self {
        self.render_plan = Some(compile);
        self.render_plan_scope = RenderPlanScope::MultiphaseOnly;
        self
    }

    /// Override the structural [`RenderPlanScope`] of an attached render-plan compiler.
    pub fn with_render_plan_scope(mut self, scope: RenderPlanScope) -> Self {
        self.render_plan_scope = scope;
        self
    }

    /// Attach a pre-load video-plan compiler registration for this route.
    pub fn with_video_plan(mut self, video_plan: VideoPlanRegistration) -> Self {
        self.video_plan = Some(video_plan);
        self
    }

    /// Also register this route as a conditioned feature sub-route under `parent_provider_id`.
    pub fn with_feature_route(
        mut self,
        parent_provider_id: Option<&'static str>,
        descriptor: fn() -> ModelDescriptor,
    ) -> Self {
        self.feature_route = Some(RouteFeatureBinding {
            parent_provider_id,
            descriptor,
        });
        self
    }

    /// Alias this route's encoder contract lookup to an owning base generator id.
    pub fn with_encoder_contract_parent(mut self, provider_id: &'static str) -> Self {
        self.encoder_contract_parent = Some(provider_id);
        self
    }

    /// Attach a memory-strategy contract and safety check (auto-bound to `route_id`).
    pub fn with_memory(
        mut self,
        contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
        safety_check: fn(
            &LoadSpec,
            &MemoryProviderContract,
            &MemoryRunContext,
        ) -> MemorySafetyDecision,
    ) -> Self {
        self.memory.contract = Some((None, contract, safety_check));
        self.memory.composed = false;
        self
    }

    /// Attach an existing [`MemoryRegistration`] constant (verified against `route_id`).
    pub fn with_memory_registration(mut self, registration: MemoryRegistration) -> Self {
        self.memory.contract = Some((
            Some(registration.provider_id),
            registration.contract,
            registration.safety_check,
        ));
        self.memory.composed = false;
        self
    }

    /// Attach a composed memory-strategy contract for a route with no standalone generator.
    pub fn with_composed_memory(
        mut self,
        contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
        safety_check: fn(
            &LoadSpec,
            &MemoryProviderContract,
            &MemoryRunContext,
        ) -> MemorySafetyDecision,
    ) -> Self {
        self.memory.contract = Some((None, contract, safety_check));
        self.memory.composed = true;
        self
    }

    /// Attach an existing [`MemoryRegistration`] constant as a composed memory route.
    pub fn with_composed_memory_registration(mut self, registration: MemoryRegistration) -> Self {
        self.memory.contract = Some((
            Some(registration.provider_id),
            registration.contract,
            registration.safety_check,
        ));
        self.memory.composed = true;
        self
    }

    /// Attach a weights-free memory-contract fixture (auto-bound to `route_id`).
    pub fn with_memory_fixture(
        mut self,
        surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
        contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    ) -> Self {
        self.memory.fixture = Some((None, surface_specs, contract));
        self
    }

    /// Attach an existing [`MemoryContractFixtureRegistration`] (verified against `route_id`).
    pub fn with_memory_fixture_registration(
        mut self,
        fixture: MemoryContractFixtureRegistration,
    ) -> Self {
        self.memory.fixture = Some((
            Some(fixture.provider_id),
            fixture.surface_specs,
            fixture.contract,
        ));
        self
    }

    /// Attach a selector-aware memory-contract surface resolver (auto-bound to `route_id`).
    pub fn with_memory_surface_resolver(
        mut self,
        contract: fn(&MemoryContractSurfaceSpec) -> Result<MemoryProviderContract>,
    ) -> Self {
        self.memory.surface_resolver = Some((None, contract));
        self
    }

    /// Attach an existing [`MemoryContractSurfaceResolverRegistration`] (verified against `route_id`).
    pub fn with_memory_surface_resolver_registration(
        mut self,
        resolver: MemoryContractSurfaceResolverRegistration,
    ) -> Self {
        self.memory.surface_resolver = Some((Some(resolver.provider_id), resolver.contract));
        self
    }

    /// Attach a resident-only memory-contract witness (auto-bound to `route_id`).
    pub fn with_resident_only_memory(
        mut self,
        surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
        contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    ) -> Self {
        self.memory.resident_only = Some((None, surface_specs, contract));
        self
    }

    /// Attach an existing [`ResidentOnlyMemoryContractRegistration`] (verified against `route_id`).
    pub fn with_resident_only_memory_registration(
        mut self,
        witness: ResidentOnlyMemoryContractRegistration,
    ) -> Self {
        self.memory.resident_only = Some((
            Some(witness.provider_id),
            witness.surface_specs,
            witness.contract,
        ));
        self
    }

    /// Attach executable memory-behavior conformance fixtures (auto-bound to `route_id`).
    pub fn with_memory_behavior(
        mut self,
        valid_fixtures: fn(
            &LoadSpec,
            &MemoryProviderContract,
            MemoryStrategy,
        ) -> Result<Vec<MemoryBehaviorFixture>>,
        begin_request: MemoryBehaviorBeginRequest,
    ) -> Self {
        self.memory.behavior = Some((None, valid_fixtures, begin_request));
        self
    }

    /// Attach an existing [`MemoryBehaviorRegistration`] constant (verified against `route_id`).
    pub fn with_memory_behavior_registration(
        mut self,
        behavior: MemoryBehaviorRegistration,
    ) -> Self {
        self.memory.behavior = Some((
            Some(behavior.provider_id),
            behavior.valid_fixtures,
            behavior.begin_request,
        ));
        self
    }

    /// Attach a warm 1024² activation-memory anchor for this route.
    pub fn with_activation_memory(mut self, anchor: ActivationMemoryAnchor) -> Self {
        self.activation_memory = Some(anchor);
        self
    }

    /// Attach a pre-fit reference-workspace byte pricer for this route.
    pub fn with_reference_workspace(
        mut self,
        reference_workspace_bytes: fn(&LoadSpec, &MemoryReferenceResourceFacts) -> Result<u64>,
    ) -> Self {
        self.reference_workspace = Some(reference_workspace_bytes);
        self
    }
}

// -------------------------------------------------------------------------------------------------
// Top-level RouteRecord (`REG-1`, `REG-2`, `AP5`)
// -------------------------------------------------------------------------------------------------

/// The primary typed composition record for one inference route (`REG-1`, `REG-2`, `AP5`).
///
/// One call to [`crate::ProviderRegistryBuilder::register_route`] registers the route record,
/// generator factory, computed descriptor/capabilities, render/video plans, feature sub-route,
/// memory contract, fixtures, behavior, activation memory, and reference workspace.
#[derive(Clone, Debug)]
pub struct RouteRecord {
    /// Unique stable route identifier (same key as `ModelDescriptor::id` / `FeatureRouteRegistration::route_id`).
    pub route_id: &'static str,
    /// Stable model family identifier (e.g. `"flux"`, `"sdxl"`, `"kokoro"`).
    pub family: &'static str,
    /// Tensor backend label (`"mlx"` or `"candle"`).
    pub backend: &'static str,
    /// Output modality (`Image`, `Video`, `Both`, or `Audio`).
    pub modality: Modality,
    /// Whether this route is macOS/Apple-Silicon-only (`Capabilities::mac_only`).
    pub mac_only: bool,
    /// Primary component and behavioural structure of the route.
    pub structure: RouteStructure,
    /// Product bounds (spatial range/grid, legal steps, audio duration cap).
    pub limits: ProductLimits,
    /// Seed scheme, per-request output count ceiling, and prompt-batch semantics.
    pub seed_and_batch: SeedAndBatchSemantics,
    /// Model defaults (native steps, guidance, sampler, schedule, shift, strength, dimensions, audio defaults).
    pub defaults: RouteDefaults,
    /// Executable and registry bindings fanned out by `ProviderRegistryBuilder::register_route`.
    pub bindings: RouteBindings,
}
