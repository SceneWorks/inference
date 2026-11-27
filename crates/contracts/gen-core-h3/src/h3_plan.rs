//! Weights-free pre-load MiniMax-H3 execution and CUDA envelope plan (`DEC-3`, `DEC-13`, `PIPE-6`,
//! `MEM-3`, `SW-6973`).
//!
//! Models both MiniMax-H3 multi-stream video+audio pipelines (`t2va`, `fl2va`, `ref2va`) and
//! MiniMax-H3-Image pipelines (`minimax_h3_image` T2I and Edit + PAG) before opening or allocating
//! any weights:
//!
//! * **Paired schedules**: distinct shifts (`VIDEO_SIGMA_SHIFT = 12.0` default or overridable via
//!   `scheduler_shift`, `AUDIO_SIGMA_SHIFT = 3.0`) on explicit [`StreamId::Video`] and
//!   [`StreamId::Audio`] streams resolved into [`ResolvedSchedule`] tables.
//! * **Veda sparse-attention vs dense-fallback policy** and matched-policy 24 GiB (`M5`) vs 95 GiB
//!   control envelope metadata (`1344x768`, 345 frames, 24 fps, Turbo 8-step, `21 GiB` pool +
//!   `3 GiB` reserve = `24 GiB`, `32 MiB` cuBLAS workspace, `TransformerComponent::Both`, `window = 1`).
//! * **`MEM-3` concurrent live-set pricing** via [`build_multistream_phase_live_set`] summing
//!   concurrent Video + Audio + joint workspace (with deduplicated DiT trunk aliases), plus direct
//!   verification against [`WindowAllocation::bounded_cuda_denoise_peak_bytes`] and
//!   [`WindowAllocation::bounded_cuda_denoise_with_veda_bytes`].

use serde::{Deserialize, Serialize};

use gen_core::h3_image::{
    edit_reference_images, generation_canvas, minimax_h3_image_pag_capability, reference_vae_size,
    request_sampler, request_scheduler, validate_min_generation_area,
    validate_request as validate_image_contract, FIXED_STEPS as H3_IMAGE_FIXED_STEPS,
    MIN_GENERATION_AREA_DEFAULT, SIZE_STRIDE as IMAGE_STRIDE,
};
use gen_core::h3_memory::window::WindowAllocation;
use gen_core::multistream::{
    build_multistream_phase_live_set, PerTokenTimePolicy, ResolvedSchedule, ScheduleSource,
    StreamBundleSpec, StreamGeometry, StreamId, StreamLatentDescriptor, StreamMemorySpec,
    StreamSpec,
};
use gen_core::veda::workspace::WorkspaceBound;
use gen_core::{
    effective_reference_image_resize_mode, effective_reference_image_short_edge, resolve_pag,
    validate_reference_image_short_edge, Conditioning, Error, GenerationRequest,
    MemoryBudgetDomain, MemoryPhase, PagPolicyErrorKind, PricedAllocation, RequestLiveSetPlan,
    Result, TransformerComponent, VedaRequest,
};

use crate::h3_keyframe::{resolve_keyframe_canvas, resolve_keyframe_slots};
use crate::h3_memory::{
    resolve_request_frames, validate_request_geometry, NativeBackend, NativeRows,
    NativeTaskAllocation, AUDIO_OUTPUT_CHANNELS, SPATIAL_STRIDE,
};
use crate::h3_reference::{
    normalized_clip_frame_count, reference_audio_max_samples_per_channel, Ref2VaReferenceRef,
    Ref2VaReferences, REFERENCE_IMAGE_MULTIPLE,
};
use crate::h3_schedule::{
    JointSchedule, SigmaSchedule, AUDIO_SIGMA_SHIFT, MIN_INFERENCE_STEPS, VIDEO_SIGMA_SHIFT,
};

pub const H3_DEFAULT_VIDEO_STEPS: u32 = 50;
pub const H3_MAX_VIDEO_STEPS: u32 = 200;
pub const H3_TURBO_8_STEPS: u32 = 8;
pub const H3_DIT_LAYERS: usize = 50;
pub const H3_VIDEO_LATENT_CHANNELS: u16 = 24;
pub const H3_AUDIO_LATENT_CHANNELS: u16 = 32;
pub const H3_PATCH_SIZE: [u32; 3] = [1, 2, 2];
pub const H3_CANVAS_DEFAULT_WIDTH: u32 = 1344;
pub const H3_CANVAS_DEFAULT_HEIGHT: u32 = 768;
pub use crate::h3_memory::LEGAL_FRAME_COUNTS as H3_LEGAL_FRAME_COUNTS;
pub const H3_FPS: f32 = 24.0;
pub const H3_CUBLASLT_WORKSPACE_BYTES: u64 = 32 * 1024 * 1024;
pub const H3_M5_POOL_BUDGET_BYTES: u64 = 21 * 1024 * 1024 * 1024;
pub const H3_M5_RESERVE_BYTES: u64 = 3 * 1024 * 1024 * 1024;
pub const H3_M5_TOTAL_ENVELOPE_BYTES: u64 = 24 * 1024 * 1024 * 1024;
pub const H3_CONTROL_95GIB_ENVELOPE_BYTES: u64 = 95 * 1024 * 1024 * 1024;
pub const H3_CONTROL_95GIB_POOL_BYTES: u64 = 92 * 1024 * 1024 * 1024;
pub const H3_DEFAULT_STREAM_WINDOW_SIZE: u32 = 1;

const VIDEO_ROW_FEATURE_DIM: usize = 96; // 24 channels * 1 * 2 * 2 patch

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H3RouteKind {
    T2va,
    Fl2va,
    Ref2va,
    ImageT2i,
    ImageEdit,
}

impl H3RouteKind {
    pub const fn is_video(self) -> bool {
        matches!(self, Self::T2va | Self::Fl2va | Self::Ref2va)
    }

    pub const fn is_image(self) -> bool {
        matches!(self, Self::ImageT2i | Self::ImageEdit)
    }

    pub const fn checkpoint_partition(self) -> &'static str {
        match self {
            Self::T2va | Self::Fl2va => "transformer",
            Self::Ref2va => "transformer_ref",
            Self::ImageT2i | Self::ImageEdit => "transformer_diffusers",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H3BudgetDomainKey {
    DeviceVram,
    UnifiedMemory,
}

impl H3BudgetDomainKey {
    pub const fn to_domain(self) -> MemoryBudgetDomain {
        match self {
            Self::DeviceVram => MemoryBudgetDomain::DeviceVram,
            Self::UnifiedMemory => MemoryBudgetDomain::UnifiedMemory,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H3StreamComponentScope {
    Dit,
    TextEncoder,
    Both,
}

impl From<TransformerComponent> for H3StreamComponentScope {
    fn from(value: TransformerComponent) -> Self {
        match value {
            TransformerComponent::Dit => Self::Dit,
            TransformerComponent::TextEncoder => Self::TextEncoder,
            TransformerComponent::Both => Self::Both,
        }
    }
}

impl H3StreamComponentScope {
    pub const fn to_gen_core(self) -> TransformerComponent {
        match self {
            Self::Dit => TransformerComponent::Dit,
            Self::TextEncoder => TransformerComponent::TextEncoder,
            Self::Both => TransformerComponent::Both,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H3EnvelopeArmKind {
    M5Streamed24GiB,
    Control95GiB,
    Custom,
}

/// Hardware/runtime envelope specification (`M5` 24 GiB vs `95 GiB` control arm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3EnvelopeSpec {
    pub arm: H3EnvelopeArmKind,
    pub arm_id: String,
    pub canonical_width: u32,
    pub canonical_height: u32,
    pub canonical_frames: u32,
    pub canonical_fps: f32,
    pub canonical_steps: u32,
    pub pool_budget_bytes: u64,
    pub reserve_bytes: u64,
    pub total_envelope_bytes: u64,
    pub cublas_workspace_bytes: u64,
    pub stream_component: Option<H3StreamComponentScope>,
    pub stream_window_size: Option<u32>,
}

impl H3EnvelopeSpec {
    /// Canonical M5 envelope: `1344x768`, `345` frames, `24` fps, Turbo `8`-step,
    /// `21 GiB` pool + `3 GiB` reserve = `24 GiB` envelope, `32 MiB` cuBLAS workspace,
    /// both text-encoder and DiT streamed windows (`window = 1`).
    pub fn m5_24gib() -> Self {
        Self {
            arm: H3EnvelopeArmKind::M5Streamed24GiB,
            arm_id: "m5_1344x768_345f_turbo8_24gib".to_owned(),
            canonical_width: H3_CANVAS_DEFAULT_WIDTH,
            canonical_height: H3_CANVAS_DEFAULT_HEIGHT,
            canonical_frames: 345,
            canonical_fps: H3_FPS,
            canonical_steps: H3_TURBO_8_STEPS,
            pool_budget_bytes: H3_M5_POOL_BUDGET_BYTES,
            reserve_bytes: H3_M5_RESERVE_BYTES,
            total_envelope_bytes: H3_M5_TOTAL_ENVELOPE_BYTES,
            cublas_workspace_bytes: H3_CUBLASLT_WORKSPACE_BYTES,
            stream_component: Some(H3StreamComponentScope::Both),
            stream_window_size: Some(H3_DEFAULT_STREAM_WINDOW_SIZE),
        }
    }

    /// 95 GiB workstation control arm (`92 GiB` pool + `3 GiB` reserve = `95 GiB`), sharing
    /// the same matched sampling/streaming policy when compared against [`Self::m5_24gib`].
    pub fn control_95gib() -> Self {
        Self {
            arm: H3EnvelopeArmKind::Control95GiB,
            arm_id: "control_95gib".to_owned(),
            canonical_width: H3_CANVAS_DEFAULT_WIDTH,
            canonical_height: H3_CANVAS_DEFAULT_HEIGHT,
            canonical_frames: 345,
            canonical_fps: H3_FPS,
            canonical_steps: H3_TURBO_8_STEPS,
            pool_budget_bytes: H3_CONTROL_95GIB_POOL_BYTES,
            reserve_bytes: H3_M5_RESERVE_BYTES,
            total_envelope_bytes: H3_CONTROL_95GIB_ENVELOPE_BYTES,
            cublas_workspace_bytes: H3_CUBLASLT_WORKSPACE_BYTES,
            stream_component: Some(H3StreamComponentScope::Both),
            stream_window_size: Some(H3_DEFAULT_STREAM_WINDOW_SIZE),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.pool_budget_bytes == 0 || self.total_envelope_bytes == 0 {
            return Err(Error::Unsupported(
                "minimax_h3 plan: envelope pool and total bytes must be non-zero".into(),
            ));
        }
        let sum = self
            .pool_budget_bytes
            .checked_add(self.reserve_bytes)
            .ok_or_else(|| Error::Msg("minimax_h3 plan: envelope byte sum overflow".into()))?;
        if sum != self.total_envelope_bytes {
            return Err(Error::Unsupported(format!(
                "minimax_h3 plan: envelope pool ({}) + reserve ({}) != total ({})",
                self.pool_budget_bytes, self.reserve_bytes, self.total_envelope_bytes
            )));
        }
        if let Some(component) = self.stream_component {
            if component != H3StreamComponentScope::Both {
                return Err(Error::Unsupported(format!(
                    "minimax_h3 plan: streamed envelope requires TransformerComponent::Both, got {component:?}"
                )));
            }
            let win = self
                .stream_window_size
                .unwrap_or(H3_DEFAULT_STREAM_WINDOW_SIZE);
            if win != H3_DEFAULT_STREAM_WINDOW_SIZE {
                return Err(Error::Unsupported(format!(
                    "minimax_h3 plan: streamed window must be {H3_DEFAULT_STREAM_WINDOW_SIZE}, got {win}"
                )));
            }
        }
        Ok(())
    }
}

/// Veda sparse-attention vs dense-fallback execution declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H3VedaExecutionPolicy {
    Disabled,
    SparseWithDenseFallback {
        generated_budget: String,
        reference_budget: String,
        dense_fallback_supported: bool,
    },
}

impl H3VedaExecutionPolicy {
    pub fn from_veda_request(veda: Option<&VedaRequest>) -> Self {
        match veda.filter(|v| v.enabled) {
            None => Self::Disabled,
            Some(v) => Self::SparseWithDenseFallback {
                generated_budget: format!("{:?}", v.generated),
                reference_budget: format!("{:?}", v.reference),
                dense_fallback_supported: true,
            },
        }
    }

    pub const fn is_veda_enabled(&self) -> bool {
        matches!(self, Self::SparseWithDenseFallback { .. })
    }
}

/// Streamed text-encoder & DiT window policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct H3StreamedWindowPolicy {
    pub enabled: bool,
    pub component: Option<H3StreamComponentScope>,
    pub window_size: Option<u32>,
}

impl H3StreamedWindowPolicy {
    pub fn from_request_memory(req: &GenerationRequest) -> Result<Self> {
        let Some(memory) = req.memory.filter(|m| m.stream_transformer_blocks) else {
            return Ok(Self {
                enabled: false,
                component: None,
                window_size: None,
            });
        };
        let component = memory
            .transformer_window_component
            .unwrap_or(TransformerComponent::Dit);
        if component != TransformerComponent::Both {
            return Err(Error::Unsupported(format!(
                "minimax_h3: transformer streaming requires TransformerComponent::Both, got {component:?}"
            )));
        }
        let window_size = memory
            .transformer_window_size
            .unwrap_or(H3_DEFAULT_STREAM_WINDOW_SIZE);
        if window_size != H3_DEFAULT_STREAM_WINDOW_SIZE {
            return Err(Error::Unsupported(format!(
                "minimax_h3: transformer window {window_size} is outside the published singleton domain [{}]",
                H3_DEFAULT_STREAM_WINDOW_SIZE
            )));
        }
        Ok(Self {
            enabled: true,
            component: Some(H3StreamComponentScope::Both),
            window_size: Some(window_size),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.component != Some(H3StreamComponentScope::Both) {
                return Err(Error::Unsupported(
                    "minimax_h3 plan: enabled block streaming requires TransformerComponent::Both"
                        .into(),
                ));
            }
            if self.window_size != Some(H3_DEFAULT_STREAM_WINDOW_SIZE) {
                return Err(Error::Unsupported(format!(
                    "minimax_h3 plan: enabled block streaming requires window size {H3_DEFAULT_STREAM_WINDOW_SIZE}"
                )));
            }
        }
        Ok(())
    }
}

/// Policy equivalence key ensuring comparisons (`M5` vs `95 GiB`, sparse vs dense) compare against
/// matched policy only (`E2 / AC1`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct H3MatchedPolicyKey {
    pub route: H3RouteKind,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps_bits: u32,
    pub steps: u32,
    pub video_shift_bits: u32,
    pub audio_shift_bits: u32,
    pub veda_policy: H3VedaExecutionPolicy,
    pub streamed_window: H3StreamedWindowPolicy,
    pub cublas_workspace_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3ScheduleSpec {
    pub stream: StreamId,
    pub modality_name: String,
    pub shift: f32,
    pub schedule: ResolvedSchedule,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3MemoryFootprintSpec {
    pub domain: H3BudgetDomainKey,
    pub host_staging_bytes: u64,
    pub dit_trunk_bytes: u64,
    pub adaln_table_bytes: u64,
    pub joint_attention_workspace_bytes: u64,
    pub cublas_workspace_bytes: u64,
    pub veda_sparse_device_bytes: u64,
    pub veda_dense_fallback_device_bytes: u64,
}

impl H3MemoryFootprintSpec {
    /// Conservative Q4 streamed footprint sized for M5 (`1344x768`, 345f, window=1) with `32 MiB`
    /// cuBLAS workspace so sparse & dense fallback paths both sit under the 21 GiB pool.
    pub fn streamed_default(
        domain: H3BudgetDomainKey,
        envelope: &H3EnvelopeSpec,
        veda_enabled: bool,
    ) -> Self {
        Self {
            domain,
            host_staging_bytes: 512 * 1024 * 1024,
            // Resident root + 1-block Q4 window (~5.9 GiB streamed DiT footprint).
            dit_trunk_bytes: 6_000_000_000,
            adaln_table_bytes: 128 * 1024 * 1024,
            joint_attention_workspace_bytes: 512 * 1024 * 1024,
            cublas_workspace_bytes: envelope.cublas_workspace_bytes,
            veda_sparse_device_bytes: if veda_enabled { 2_400_000_000 } else { 0 },
            veda_dense_fallback_device_bytes: if veda_enabled { 1_800_000_000 } else { 0 },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct H3DomainMemorySummary {
    pub domain: H3BudgetDomainKey,
    pub host_ram_peak_bytes: u64,
    pub video_stream_activation_bytes: u64,
    pub audio_stream_activation_bytes: u64,
    pub joint_workspace_bytes: u64,
    pub primary_operation_peak_bytes: u64,
    pub veda_sparse_peak_bytes: Option<u64>,
    pub veda_dense_fallback_peak_bytes: Option<u64>,
    pub fits_pool_budget: bool,
    pub fits_total_envelope: bool,
}

/// Result of checking a concrete header-derived [`WindowAllocation`] (+ optional Veda
/// [`WorkspaceBound`]) against an [`H3RenderPlan`]'s envelope under matched policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3CudaWindowFitResult {
    pub arm_id: String,
    pub matched_policy: H3MatchedPolicyKey,
    pub window_size: usize,
    pub bounded_cuda_denoise_peak_bytes: u64,
    pub bounded_cuda_denoise_with_veda_bytes: Option<u64>,
    pub veda_sparse_device_bytes: Option<u64>,
    pub veda_dense_fallback_device_bytes: Option<u64>,
    pub pool_budget_bytes: u64,
    pub reserve_bytes: u64,
    pub total_envelope_bytes: u64,
    pub sparse_fits_pool: bool,
    pub dense_fallback_fits_pool: bool,
    pub fits_envelope: bool,
}

mod option_veda_request_serde {
    use gen_core::{VedaBudget, VedaRequest};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum SerdeVedaBudget {
        SparsityPercent(f64),
        KeyTiles(u32),
    }

    impl From<VedaBudget> for SerdeVedaBudget {
        fn from(value: VedaBudget) -> Self {
            match value {
                VedaBudget::SparsityPercent(p) => Self::SparsityPercent(p),
                VedaBudget::KeyTiles(n) => Self::KeyTiles(n),
            }
        }
    }

    impl From<SerdeVedaBudget> for VedaBudget {
        fn from(value: SerdeVedaBudget) -> Self {
            match value {
                SerdeVedaBudget::SparsityPercent(p) => Self::SparsityPercent(p),
                SerdeVedaBudget::KeyTiles(n) => Self::KeyTiles(n),
            }
        }
    }

    #[derive(Serialize, Deserialize)]
    struct SerdeVedaRequest {
        enabled: bool,
        generated: SerdeVedaBudget,
        reference: SerdeVedaBudget,
        full_attention_layers: Vec<usize>,
        full_attention_steps: Vec<usize>,
    }

    pub fn serialize<S>(value: &Option<VedaRequest>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mapped = value.as_ref().map(|v| SerdeVedaRequest {
            enabled: v.enabled,
            generated: v.generated.into(),
            reference: v.reference.into(),
            full_attention_layers: v.full_attention_layers.clone(),
            full_attention_steps: v.full_attention_steps.clone(),
        });
        mapped.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<VedaRequest>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mapped = Option::<SerdeVedaRequest>::deserialize(deserializer)?;
        Ok(mapped.map(|v| VedaRequest {
            enabled: v.enabled,
            generated: v.generated.into(),
            reference: v.reference.into(),
            full_attention_layers: v.full_attention_layers,
            full_attention_steps: v.full_attention_steps,
        }))
    }
}

/// Pre-load inputs validated by [`H3RenderPlan::build`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3RenderPlanInput {
    pub route: H3RouteKind,
    pub domain: H3BudgetDomainKey,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: f32,
    pub steps: u32,
    pub video_shift: f32,
    pub audio_shift: f32,
    pub seed: u64,
    pub condition_video_tokens: usize,
    pub condition_audio_rows: usize,
    pub reference_count: usize,
    pub min_generation_area: Option<u64>,
    pub image_sampler: Option<String>,
    pub image_scheduler: Option<String>,
    pub pag_active_steps: u32,
    #[serde(default, with = "option_veda_request_serde")]
    pub veda: Option<VedaRequest>,
    pub streamed_window: H3StreamedWindowPolicy,
    pub envelope: H3EnvelopeSpec,
    pub footprint: H3MemoryFootprintSpec,
}

/// Complete serializable, weights-free pre-load MiniMax-H3 execution and `MEM-3` plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H3RenderPlan {
    pub route: H3RouteKind,
    pub width: u32,
    pub height: u32,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub frames: u32,
    pub fps: f32,
    pub steps: u32,
    pub video_shift: f32,
    pub audio_shift: f32,
    pub patch_size: [u32; 3],
    pub seed: u64,
    pub condition_video_tokens: usize,
    pub condition_audio_rows: usize,
    pub reference_count: usize,
    pub image_sampler: Option<String>,
    pub image_scheduler: Option<String>,
    pub pag_active_steps: u32,
    pub bundle: StreamBundleSpec,
    pub schedules: Vec<H3ScheduleSpec>,
    pub veda_policy: H3VedaExecutionPolicy,
    pub streamed_window: H3StreamedWindowPolicy,
    pub envelope: H3EnvelopeSpec,
    pub memory: H3DomainMemorySummary,
    #[serde(skip)]
    cached_footprint: Option<H3MemoryFootprintSpec>,
}

impl H3RenderPlan {
    /// Canonical M5 24 GiB baseline (`1344x768`, 345 frames, 24 fps, Turbo 8-step, streamed window=1
    /// on `TransformerComponent::Both`, `32 MiB` cuBLAS workspace, `21 GiB` pool + `3 GiB` reserve).
    pub fn m5_canonical_t2va_turbo8(veda: Option<VedaRequest>) -> Result<Self> {
        let envelope = H3EnvelopeSpec::m5_24gib();
        let veda_enabled = veda.as_ref().is_some_and(|v| v.enabled);
        let footprint = H3MemoryFootprintSpec::streamed_default(
            H3BudgetDomainKey::DeviceVram,
            &envelope,
            veda_enabled,
        );
        Self::build(H3RenderPlanInput {
            route: H3RouteKind::T2va,
            domain: H3BudgetDomainKey::DeviceVram,
            width: envelope.canonical_width,
            height: envelope.canonical_height,
            frames: envelope.canonical_frames,
            fps: envelope.canonical_fps,
            steps: envelope.canonical_steps,
            video_shift: VIDEO_SIGMA_SHIFT,
            audio_shift: AUDIO_SIGMA_SHIFT,
            seed: 1,
            condition_video_tokens: 0,
            condition_audio_rows: 0,
            reference_count: 0,
            min_generation_area: None,
            image_sampler: None,
            image_scheduler: None,
            pag_active_steps: 0,
            veda,
            streamed_window: H3StreamedWindowPolicy {
                enabled: true,
                component: Some(H3StreamComponentScope::Both),
                window_size: Some(H3_DEFAULT_STREAM_WINDOW_SIZE),
            },
            envelope,
            footprint,
        })
    }

    /// Build and validate a pre-load video plan directly from [`GenerationRequest`], resolving
    /// unset `(0, 0)` canvas geometry and `frames`/`duration` on the shared H3 lattice.
    pub fn from_video_generation_request(
        req: &GenerationRequest,
        domain: H3BudgetDomainKey,
        envelope: H3EnvelopeSpec,
    ) -> Result<Self> {
        let frames = resolve_request_frames(req.frames, req.duration)?;
        let keyframes = req.keyframes();
        let (width, height) = if req.width == 0 && req.height == 0 {
            if let Some(first_kf) = keyframes.first() {
                resolve_keyframe_canvas(first_kf.image.width, first_kf.image.height)?
            } else {
                (H3_CANVAS_DEFAULT_WIDTH, H3_CANVAS_DEFAULT_HEIGHT)
            }
        } else {
            (req.width, req.height)
        };
        Self::from_video_request_with_geometry(req, width, height, frames, domain, envelope)
    }

    /// Build and validate a pre-load video plan from [`GenerationRequest`] and already-resolved
    /// lattice `(width, height, frames)` geometry so engine `validate()` and `generate()` stay in
    /// lockstep.
    pub fn from_video_request_with_geometry(
        req: &GenerationRequest,
        width: u32,
        height: u32,
        frames: u32,
        domain: H3BudgetDomainKey,
        envelope: H3EnvelopeSpec,
    ) -> Result<Self> {
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg("minimax_h3: prompt must not be empty".into()));
        }
        if req
            .negative_prompt
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty())
        {
            return Err(Error::Unsupported(
                "minimax_h3: negative_prompt is not supported (guidance-distilled checkpoint)"
                    .into(),
            ));
        }
        if let Some(guidance) = req.guidance {
            return Err(Error::Unsupported(format!(
                "minimax_h3: guidance is not supported (got {guidance})"
            )));
        }
        if let Some(fps) = req.fps {
            if f64::from(fps) != f64::from(H3_FPS) {
                return Err(Error::Msg(format!(
                    "minimax_h3: the released model generates at {H3_FPS} fps only, got {fps}"
                )));
            }
        }
        let steps = req.steps.unwrap_or(H3_DEFAULT_VIDEO_STEPS);
        if steps == 0 || steps > H3_MAX_VIDEO_STEPS {
            return Err(Error::Msg(format!(
                "minimax_h3: steps must be in 1..={H3_MAX_VIDEO_STEPS}, got {steps}"
            )));
        }
        if let Some(veda) = &req.veda {
            veda.validate(H3_DIT_LAYERS, steps as usize)
                .map_err(Error::Msg)?;
        }
        let video_shift = req.scheduler_shift.unwrap_or(VIDEO_SIGMA_SHIFT);
        if !video_shift.is_finite() || video_shift <= 0.0 {
            return Err(Error::Msg(format!(
                "minimax_h3: scheduler_shift must be finite and positive, got {video_shift}"
            )));
        }

        let keyframes = req.keyframes();
        validate_request_geometry(width, height, u64::from(frames), req.count)?;

        for kf in &keyframes {
            if kf.strength != 1.0 {
                return Err(Error::Msg(format!(
                    "minimax_h3: keyframe conditioning strength is not supported (got {})",
                    kf.strength
                )));
            }
        }
        if !keyframes.is_empty() {
            resolve_keyframe_slots(keyframes.iter().map(|kf| kf.frame_idx), frames as usize)?;
        }

        let mut refs: Vec<Ref2VaReferenceRef<'_>> = Vec::new();
        for c in &req.conditioning {
            match c {
                Conditioning::Keyframe { .. } => {}
                Conditioning::Reference { image, .. } => {
                    refs.push(Ref2VaReferenceRef::Image(image));
                }
                Conditioning::ReferenceAudio { audio, .. } => {
                    refs.push(Ref2VaReferenceRef::Audio(audio));
                }
                Conditioning::ReferenceVideo { frames, fps, audio } => {
                    refs.push(Ref2VaReferenceRef::Video {
                        frames,
                        fps: f64::from(*fps),
                        audio: audio.as_ref(),
                    });
                }
                _ => {
                    return Err(Error::Unsupported(
                        "minimax_h3: unsupported conditioning kind on video route".into(),
                    ));
                }
            }
        }
        if !keyframes.is_empty() && !refs.is_empty() {
            return Err(Error::Msg(
                "minimax_h3: cannot mix keyframes (fl2va) and unpositioned references (ref2va)"
                    .into(),
            ));
        }
        if !refs.is_empty() {
            Ref2VaReferences::validate_borrowed(&refs)?;
        }
        validate_reference_image_short_edge("minimax_h3", req)?;

        let (route, condition_video_tokens, condition_audio_rows, reference_count) =
            if !refs.is_empty() {
                let mut ref_video_tokens = 0usize;
                let mut ref_audio_rows = 0usize;
                for item in &refs {
                    match item {
                        Ref2VaReferenceRef::Image(img) => {
                            let (rw, rh) = crate::resolve_reference_image_size(
                                img.width,
                                img.height,
                                width,
                                height,
                                effective_reference_image_resize_mode(req),
                                effective_reference_image_short_edge(req),
                                REFERENCE_IMAGE_MULTIPLE,
                            )?;
                            let gw = ((rw / SPATIAL_STRIDE) as usize).max(1);
                            let gh = ((rh / SPATIAL_STRIDE) as usize).max(1);
                            ref_video_tokens = ref_video_tokens.saturating_add(gw * gh);
                        }
                        Ref2VaReferenceRef::Video {
                            frames: ref_frames,
                            fps,
                            audio,
                        } => {
                            let kept = normalized_clip_frame_count(
                                ref_frames.len(),
                                *fps,
                                frames as usize,
                                f64::from(H3_FPS),
                            )?;
                            let latent_f = (kept.saturating_sub(5)) / 17 * 5 + 2;
                            let first = &ref_frames[0];
                            let rw = (first.width / SPATIAL_STRIDE).max(1) as usize;
                            let rh = (first.height / SPATIAL_STRIDE).max(1) as usize;
                            ref_video_tokens = ref_video_tokens.saturating_add(latent_f * rw * rh);
                            if let Some(a) = audio {
                                let per_ch = reference_audio_max_samples_per_channel(
                                    frames as usize,
                                    f64::from(H3_FPS),
                                    a.sample_rate,
                                );
                                let latents = per_ch.div_ceil(800);
                                ref_audio_rows = ref_audio_rows
                                    .saturating_add(latents * usize::from(a.channels));
                            }
                        }
                        Ref2VaReferenceRef::Audio(a) => {
                            let per_ch = reference_audio_max_samples_per_channel(
                                frames as usize,
                                f64::from(H3_FPS),
                                a.sample_rate,
                            );
                            let latents = per_ch.div_ceil(800);
                            ref_audio_rows =
                                ref_audio_rows.saturating_add(latents * usize::from(a.channels));
                        }
                    }
                }
                (
                    H3RouteKind::Ref2va,
                    ref_video_tokens,
                    ref_audio_rows,
                    refs.len(),
                )
            } else if !keyframes.is_empty() {
                let grid_w = (width / SPATIAL_STRIDE) as usize;
                let grid_h = (height / SPATIAL_STRIDE) as usize;
                (H3RouteKind::Fl2va, keyframes.len() * grid_w * grid_h, 0, 0)
            } else {
                (H3RouteKind::T2va, 0, 0, 0)
            };

        let streamed_window = H3StreamedWindowPolicy::from_request_memory(req)?;
        let veda_enabled = req.veda.as_ref().is_some_and(|v| v.enabled);
        let footprint = H3MemoryFootprintSpec::streamed_default(domain, &envelope, veda_enabled);

        Self::build(H3RenderPlanInput {
            route,
            domain,
            width,
            height,
            frames,
            fps: H3_FPS,
            steps,
            video_shift,
            audio_shift: AUDIO_SIGMA_SHIFT,
            seed: req.seed.unwrap_or(1),
            condition_video_tokens,
            condition_audio_rows,
            reference_count,
            min_generation_area: None,
            image_sampler: None,
            image_scheduler: None,
            pag_active_steps: 0,
            veda: req.veda.clone(),
            streamed_window,
            envelope,
            footprint,
        })
    }

    /// Build and validate a pre-load image plan from [`GenerationRequest`] and resolved output size.
    pub fn from_image_request_with_size(
        req: &GenerationRequest,
        width: u32,
        height: u32,
        domain: H3BudgetDomainKey,
        envelope: H3EnvelopeSpec,
    ) -> Result<Self> {
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(
                "minimax_h3_image: prompt must not be empty".into(),
            ));
        }
        let refs = edit_reference_images(&req.conditioning).map_err(Error::from)?;
        let steps = req.steps.unwrap_or(H3_IMAGE_FIXED_STEPS);
        validate_image_contract(
            width,
            height,
            None,
            steps,
            req.guidance,
            req.negative_prompt
                .as_deref()
                .is_some_and(|n| !n.trim().is_empty()),
            refs.len(),
        )
        .map_err(Error::from)?;
        let sampler = request_sampler(req).map_err(Error::from)?;
        let scheduler = request_scheduler(req).map_err(Error::from)?;
        let pag_active_steps = resolve_pag(
            Some(&minimax_h3_image_pag_capability()),
            "minimax_h3_image",
            req.pag.as_ref(),
        )
        .map_err(|e| match e.kind {
            PagPolicyErrorKind::Unsupported => Error::Unsupported(e.message),
            PagPolicyErrorKind::Invalid => Error::Msg(e.message),
        })?
        .map_or(0, |p| p.active_steps(steps));
        if req.veda.as_ref().is_some_and(|v| v.enabled) {
            return Err(Error::Unsupported(
                "minimax_h3_image: Veda sparse attention is a video-only control".into(),
            ));
        }

        let streamed_window = H3StreamedWindowPolicy::from_request_memory(req)?;
        let footprint = H3MemoryFootprintSpec::streamed_default(domain, &envelope, false);
        let ref_tokens: usize = refs
            .iter()
            .map(|r| {
                let (rw, rh) = reference_vae_size(r.width, r.height);
                ((rw / IMAGE_STRIDE).max(1) as usize) * ((rh / IMAGE_STRIDE).max(1) as usize)
            })
            .sum();

        Self::build(H3RenderPlanInput {
            route: if refs.is_empty() {
                H3RouteKind::ImageT2i
            } else {
                H3RouteKind::ImageEdit
            },
            domain,
            width,
            height,
            frames: 1,
            fps: H3_FPS,
            steps,
            video_shift: 1.0,
            audio_shift: 1.0,
            seed: req.seed.unwrap_or(1),
            condition_video_tokens: ref_tokens,
            condition_audio_rows: 0,
            reference_count: refs.len(),
            min_generation_area: Some(MIN_GENERATION_AREA_DEFAULT),
            image_sampler: Some(sampler.name().to_owned()),
            image_scheduler: Some(scheduler.name().to_owned()),
            pag_active_steps,
            veda: None,
            streamed_window,
            envelope,
            footprint,
        })
    }

    pub fn from_image_generation_request(
        req: &GenerationRequest,
        domain: H3BudgetDomainKey,
        envelope: H3EnvelopeSpec,
    ) -> Result<Self> {
        Self::from_image_request_with_size(req, req.width, req.height, domain, envelope)
    }

    /// Build a validated pre-load [`H3RenderPlan`] from explicit inputs.
    pub fn build(input: H3RenderPlanInput) -> Result<Self> {
        input.envelope.validate()?;
        input.streamed_window.validate()?;
        if input.envelope.cublas_workspace_bytes != input.footprint.cublas_workspace_bytes {
            return Err(Error::Unsupported(format!(
                "minimax_h3 plan: footprint cublas_workspace_bytes ({}) disagrees with envelope ({})",
                input.footprint.cublas_workspace_bytes, input.envelope.cublas_workspace_bytes
            )));
        }

        let veda_policy = H3VedaExecutionPolicy::from_veda_request(input.veda.as_ref());

        match input.route {
            H3RouteKind::T2va | H3RouteKind::Fl2va | H3RouteKind::Ref2va => {
                if input.steps == 0 || input.steps > H3_MAX_VIDEO_STEPS {
                    return Err(Error::Unsupported(format!(
                        "minimax_h3: steps must be in 1..={H3_MAX_VIDEO_STEPS}, got {}",
                        input.steps
                    )));
                }
                if !input.fps.is_finite() || f64::from(input.fps) != f64::from(H3_FPS) {
                    return Err(Error::Unsupported(format!(
                        "minimax_h3: fps must be {H3_FPS}, got {}",
                        input.fps
                    )));
                }
                validate_request_geometry(input.width, input.height, u64::from(input.frames), 1)?;
                if let Some(veda) = &input.veda {
                    veda.validate(H3_DIT_LAYERS, input.steps as usize)
                        .map_err(Error::Msg)?;
                }
                if input.route == H3RouteKind::T2va
                    && (input.condition_video_tokens > 0
                        || input.condition_audio_rows > 0
                        || input.reference_count > 0)
                {
                    return Err(Error::Unsupported(
                        "minimax_h3 plan: t2va route cannot carry keyframe or reference conditioning"
                            .into(),
                    ));
                }
                if input.route == H3RouteKind::Fl2va && input.reference_count > 0 {
                    return Err(Error::Unsupported(
                        "minimax_h3 plan: fl2va route cannot carry ref2va references".into(),
                    ));
                }

                let joint = JointSchedule::with_shifts(
                    input.steps as usize + 1,
                    input.video_shift,
                    input.audio_shift,
                )?;
                if joint.num_evals() < MIN_INFERENCE_STEPS - 1 {
                    return Err(Error::Unsupported(
                        "minimax_h3 plan: schedule collapsed to zero model evaluations".into(),
                    ));
                }
                let video_schedule = ScheduleSource::ExplicitTable {
                    table_id: format!("h3_video_shift_{:.1}", input.video_shift),
                    sigmas: joint.video().sigmas().to_vec(),
                }
                .resolve()?;
                let audio_schedule = ScheduleSource::ExplicitTable {
                    table_id: format!("h3_audio_shift_{:.1}", input.audio_shift),
                    sigmas: joint.audio().sigmas().to_vec(),
                }
                .resolve()?;

                let latent_frames = ((input.frames as usize - 5) / 17) * 5 + 2;
                let patch_h = (input.height / SPATIAL_STRIDE) as usize;
                let patch_w = (input.width / SPATIAL_STRIDE) as usize;
                let target_tokens = latent_frames * patch_h * patch_w;
                let audio_frames =
                    ((f64::from(input.frames) / f64::from(H3_FPS)) * 40.0).round() as usize;
                let duration_millis =
                    ((f64::from(input.frames) / f64::from(H3_FPS)) * 1000.0).round() as u64;

                let video_time_policy = match input.route {
                    H3RouteKind::T2va => PerTokenTimePolicy::Uniform,
                    H3RouteKind::Fl2va | H3RouteKind::Ref2va => PerTokenTimePolicy::PerTokenMask,
                    _ => unreachable!(),
                };
                let audio_time_policy = if input.condition_audio_rows > 0 {
                    PerTokenTimePolicy::PerTokenMask
                } else {
                    PerTokenTimePolicy::Uniform
                };

                let video_spec = StreamSpec {
                    id: StreamId::Video,
                    latent_space: StreamLatentDescriptor {
                        channels: H3_VIDEO_LATENT_CHANNELS,
                        spatial_compression: (16, 16),
                        temporal_scale: 4,
                        patch_layout: "packed(2x2)".to_owned(),
                        normalization: "per_channel(minimax_h3)".to_owned(),
                    },
                    geometry: StreamGeometry::VideoTokens {
                        width: input.width,
                        height: input.height,
                        latent_frames,
                        target_tokens,
                        conditioning_tokens: input.condition_video_tokens,
                        channels: H3_VIDEO_LATENT_CHANNELS,
                    },
                    token_time_policy: video_time_policy,
                };
                let audio_spec = StreamSpec {
                    id: StreamId::Audio,
                    latent_space: StreamLatentDescriptor::audio(
                        AUDIO_OUTPUT_CHANNELS,
                        H3_AUDIO_LATENT_CHANNELS,
                        "identity",
                    ),
                    geometry: StreamGeometry::AudioGrid {
                        audio_frames,
                        mel_bins: H3_AUDIO_LATENT_CHANNELS,
                        channels: AUDIO_OUTPUT_CHANNELS,
                        duration_millis,
                    },
                    token_time_policy: audio_time_policy,
                };
                let bundle = StreamBundleSpec::new("h3_joint_av", vec![video_spec, audio_spec])?;
                let schedules = vec![
                    H3ScheduleSpec {
                        stream: StreamId::Video,
                        modality_name: "video".to_owned(),
                        shift: input.video_shift,
                        schedule: video_schedule,
                    },
                    H3ScheduleSpec {
                        stream: StreamId::Audio,
                        modality_name: "audio".to_owned(),
                        shift: input.audio_shift,
                        schedule: audio_schedule,
                    },
                ];
                let memory = compute_memory_summary(
                    &bundle,
                    input.condition_audio_rows,
                    &input.footprint,
                    &input.envelope,
                    &veda_policy,
                )?;
                Ok(Self {
                    route: input.route,
                    width: input.width,
                    height: input.height,
                    canvas_width: input.width,
                    canvas_height: input.height,
                    frames: input.frames,
                    fps: input.fps,
                    steps: input.steps,
                    video_shift: input.video_shift,
                    audio_shift: input.audio_shift,
                    patch_size: H3_PATCH_SIZE,
                    seed: input.seed,
                    condition_video_tokens: input.condition_video_tokens,
                    condition_audio_rows: input.condition_audio_rows,
                    reference_count: input.reference_count,
                    image_sampler: None,
                    image_scheduler: None,
                    pag_active_steps: 0,
                    bundle,
                    schedules,
                    veda_policy,
                    streamed_window: input.streamed_window,
                    envelope: input.envelope,
                    memory,
                    cached_footprint: Some(input.footprint),
                })
            }
            H3RouteKind::ImageT2i | H3RouteKind::ImageEdit => {
                if input.steps != H3_IMAGE_FIXED_STEPS {
                    return Err(Error::Unsupported(format!(
                        "minimax_h3_image: steps is fixed at {H3_IMAGE_FIXED_STEPS}, got {}",
                        input.steps
                    )));
                }
                if input.veda.as_ref().is_some_and(|v| v.enabled) {
                    return Err(Error::Unsupported(
                        "minimax_h3_image: Veda sparse attention is not supported on image routes"
                            .into(),
                    ));
                }
                let area = input
                    .min_generation_area
                    .unwrap_or(MIN_GENERATION_AREA_DEFAULT);
                validate_min_generation_area(Some(area)).map_err(Error::from)?;
                if input.width == 0 || input.height == 0 {
                    return Err(Error::Unsupported(format!(
                        "minimax_h3_image: size {}x{} must be positive",
                        input.width, input.height
                    )));
                }
                let (canvas_width, canvas_height) =
                    generation_canvas(input.width, input.height, area);
                let scheduler_name = input.image_scheduler.as_deref().unwrap_or("published");
                let scheduler = match scheduler_name {
                    "published" => gen_core::h3_image::H3ImageScheduler::Published,
                    other => {
                        gen_core::h3_image::H3ImageScheduler::from_name(other).ok_or_else(|| {
                            Error::Unsupported(format!(
                                "minimax_h3_image: unknown scheduler {other:?}"
                            ))
                        })?
                    }
                };
                let sigma_schedule = SigmaSchedule::h3_image(scheduler)?;
                let resolved_schedule = ScheduleSource::ExplicitTable {
                    table_id: format!("h3_image_{}", scheduler.name()),
                    sigmas: sigma_schedule.sigmas().to_vec(),
                }
                .resolve()?;
                let patch_h = (canvas_height / IMAGE_STRIDE).max(1) as usize;
                let patch_w = (canvas_width / IMAGE_STRIDE).max(1) as usize;
                let target_tokens = patch_h * patch_w;
                let policy = if input.route == H3RouteKind::ImageEdit {
                    PerTokenTimePolicy::PerTokenMask
                } else {
                    PerTokenTimePolicy::Uniform
                };
                let image_stream = StreamSpec {
                    id: StreamId::Video,
                    latent_space: StreamLatentDescriptor {
                        channels: H3_VIDEO_LATENT_CHANNELS,
                        spatial_compression: (16, 16),
                        temporal_scale: 1,
                        patch_layout: "packed(2x2)".to_owned(),
                        normalization: "per_channel(minimax_h3_image)".to_owned(),
                    },
                    geometry: StreamGeometry::VideoTokens {
                        width: canvas_width,
                        height: canvas_height,
                        latent_frames: 1,
                        target_tokens,
                        conditioning_tokens: input.condition_video_tokens,
                        channels: H3_VIDEO_LATENT_CHANNELS,
                    },
                    token_time_policy: policy,
                };
                let bundle = StreamBundleSpec::new("h3_image", vec![image_stream])?;
                let schedules = vec![H3ScheduleSpec {
                    stream: StreamId::Video,
                    modality_name: "image".to_owned(),
                    shift: 1.0,
                    schedule: resolved_schedule,
                }];
                let memory = compute_memory_summary(
                    &bundle,
                    0,
                    &input.footprint,
                    &input.envelope,
                    &H3VedaExecutionPolicy::Disabled,
                )?;
                Ok(Self {
                    route: input.route,
                    width: input.width,
                    height: input.height,
                    canvas_width,
                    canvas_height,
                    frames: 1,
                    fps: input.fps,
                    steps: input.steps,
                    video_shift: 1.0,
                    audio_shift: 1.0,
                    patch_size: H3_PATCH_SIZE,
                    seed: input.seed,
                    condition_video_tokens: input.condition_video_tokens,
                    condition_audio_rows: 0,
                    reference_count: input.reference_count,
                    image_sampler: Some(input.image_sampler.unwrap_or_else(|| "lcm".to_owned())),
                    image_scheduler: Some(scheduler.name().to_owned()),
                    pag_active_steps: input.pag_active_steps,
                    bundle,
                    schedules,
                    veda_policy: H3VedaExecutionPolicy::Disabled,
                    streamed_window: input.streamed_window,
                    envelope: input.envelope,
                    memory,
                    cached_footprint: Some(input.footprint),
                })
            }
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::Msg(format!("h3_plan serialize: {e}")))
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| Error::Msg(format!("h3_plan deserialize: {e}")))
    }

    pub fn matched_policy_key(&self) -> H3MatchedPolicyKey {
        H3MatchedPolicyKey {
            route: self.route,
            width: self.width,
            height: self.height,
            frames: self.frames,
            fps_bits: self.fps.to_bits(),
            steps: self.steps,
            video_shift_bits: self.video_shift.to_bits(),
            audio_shift_bits: self.audio_shift.to_bits(),
            veda_policy: self.veda_policy.clone(),
            streamed_window: self.streamed_window,
            cublas_workspace_bytes: self.envelope.cublas_workspace_bytes,
        }
    }

    /// Enforce `E2 / AC1` "Compare against matched policy only" before comparing two envelope runs.
    pub fn assert_matched_policy(&self, other: &Self) -> Result<()> {
        let a = self.matched_policy_key();
        let b = other.matched_policy_key();
        if a != b {
            return Err(Error::Unsupported(format!(
                "minimax_h3 plan: refusing unmatched-policy comparison between arm '{}' and '{}' ({a:?} vs {b:?})",
                self.envelope.arm_id, other.envelope.arm_id
            )));
        }
        Ok(())
    }

    /// Reconstruct the complete `MEM-3` [`RequestLiveSetPlan`] for domain inspection.
    pub fn live_set_plan(&self) -> Result<RequestLiveSetPlan> {
        let footprint = self.cached_footprint.clone().unwrap_or_else(|| {
            H3MemoryFootprintSpec::streamed_default(
                self.memory.domain,
                &self.envelope,
                self.veda_policy.is_veda_enabled(),
            )
        });
        build_h3_live_set_plan(&self.bundle, self.condition_audio_rows, &footprint)
    }

    /// Price bounded CUDA window execution (`bounded_cuda_denoise_peak_bytes` and, when Veda is
    /// enabled, `bounded_cuda_denoise_with_veda_bytes`) against this plan's envelope pool budget.
    pub fn evaluate_cuda_window_fit(
        &self,
        profile: &WindowAllocation,
        task: &NativeTaskAllocation,
        rows: NativeRows,
        veda_bound: Option<&WorkspaceBound>,
    ) -> Result<H3CudaWindowFitResult> {
        let window_size = self
            .streamed_window
            .window_size
            .unwrap_or(H3_DEFAULT_STREAM_WINDOW_SIZE) as usize;
        let backend = NativeBackend::Candle {
            tile_rows: 16,
            workspace_bytes: self.envelope.cublas_workspace_bytes,
        };
        let dense = profile.bounded_cuda_denoise_peak_bytes(task, backend, rows, window_size)?;
        let (with_veda, sparse_dev, fallback_dev) = match veda_bound {
            Some(bound) => {
                if !self.veda_policy.is_veda_enabled() {
                    return Err(Error::Unsupported(
                        "minimax_h3 plan: WorkspaceBound passed while plan disables Veda".into(),
                    ));
                }
                let combined = profile.bounded_cuda_denoise_with_veda_bytes(
                    task,
                    backend,
                    rows,
                    window_size,
                    bound,
                )?;
                (
                    Some(combined),
                    Some(bound.lifetimes.sparse.device),
                    Some(bound.lifetimes.dense_fallback.device),
                )
            }
            None => {
                if self.veda_policy.is_veda_enabled() {
                    return Err(Error::Unsupported(
                        "minimax_h3 plan: Veda-enabled plan requires WorkspaceBound for CUDA window fit"
                            .into(),
                    ));
                }
                (None, None, None)
            }
        };
        let sparse_fits_pool = with_veda.unwrap_or(dense) <= self.envelope.pool_budget_bytes;
        let dense_fallback_fits_pool = dense <= self.envelope.pool_budget_bytes
            && fallback_dev.is_none_or(|d| {
                profile
                    .denoise_resident_bytes(task, backend, rows, window_size)
                    .is_ok_and(|res| res.saturating_add(d) <= self.envelope.pool_budget_bytes)
            });
        Ok(H3CudaWindowFitResult {
            arm_id: self.envelope.arm_id.clone(),
            matched_policy: self.matched_policy_key(),
            window_size,
            bounded_cuda_denoise_peak_bytes: dense,
            bounded_cuda_denoise_with_veda_bytes: with_veda,
            veda_sparse_device_bytes: sparse_dev,
            veda_dense_fallback_device_bytes: fallback_dev,
            pool_budget_bytes: self.envelope.pool_budget_bytes,
            reserve_bytes: self.envelope.reserve_bytes,
            total_envelope_bytes: self.envelope.total_envelope_bytes,
            sparse_fits_pool,
            dense_fallback_fits_pool,
            fits_envelope: sparse_fits_pool && dense_fallback_fits_pool,
        })
    }
}

fn stream_activation_bytes(stream: &StreamSpec, condition_audio_rows: usize) -> u64 {
    match &stream.geometry {
        StreamGeometry::VideoTokens {
            target_tokens,
            conditioning_tokens,
            ..
        } => {
            // Packed f32 latent rows entering the step sampler (`[1, rows, 96]`).
            ((target_tokens + conditioning_tokens) as u64)
                .saturating_mul(VIDEO_ROW_FEATURE_DIM as u64)
                .saturating_mul(4)
        }
        StreamGeometry::AudioGrid {
            audio_frames,
            mel_bins,
            channels,
            ..
        } => {
            let gen_rows = (*audio_frames as u64).saturating_mul(u64::from(*channels));
            let total_rows = gen_rows.saturating_add(condition_audio_rows as u64);
            total_rows
                .saturating_mul(u64::from(*mel_bins))
                .saturating_mul(4)
        }
        StreamGeometry::VideoGrid {
            latent_frames,
            latent_height,
            latent_width,
            channels,
            ..
        } => (latent_frames * latent_height * latent_width * usize::from(*channels) * 4) as u64,
        StreamGeometry::Custom { shape } => {
            (shape.iter().copied().product::<usize>().max(1) * 4) as u64
        }
    }
}

fn build_h3_live_set_plan(
    bundle: &StreamBundleSpec,
    condition_audio_rows: usize,
    footprint: &H3MemoryFootprintSpec,
) -> Result<RequestLiveSetPlan> {
    let domain = footprint.domain.to_domain();
    let mut stream_specs = Vec::with_capacity(bundle.streams.len());
    let mut resident_weights = Vec::with_capacity(bundle.streams.len() + 1);
    for stream in &bundle.streams {
        let act_bytes = stream_activation_bytes(stream, condition_audio_rows);
        let retained = if matches!(stream.id, StreamId::Video) && footprint.host_staging_bytes > 0 {
            vec![PricedAllocation::new(
                "h3_host_staging",
                MemoryBudgetDomain::HostRam,
                footprint.host_staging_bytes,
            )]
        } else {
            Vec::new()
        };
        stream_specs.push(StreamMemorySpec {
            spec: stream.clone(),
            activations: vec![PricedAllocation::new(
                format!("h3_{}_latent_rows", stream.id),
                domain,
                act_bytes,
            )],
            retained,
        });
        // Video and Audio share the single physical DiT trunk via `alias_group` (`MEM-3`).
        resident_weights.push(
            PricedAllocation::new(
                format!("h3_{}_dit_ref", stream.id),
                domain,
                footprint.dit_trunk_bytes,
            )
            .with_alias_group("h3_joint_dit_trunk"),
        );
    }
    if footprint.adaln_table_bytes > 0 {
        resident_weights.push(PricedAllocation::new(
            "h3_adaln_table",
            domain,
            footprint.adaln_table_bytes,
        ));
    }
    let mut joint_ws = Vec::new();
    let total_ws = footprint
        .joint_attention_workspace_bytes
        .saturating_add(footprint.cublas_workspace_bytes)
        .saturating_add(footprint.veda_sparse_device_bytes);
    if total_ws > 0 {
        joint_ws.push(PricedAllocation::new(
            "h3_joint_attention_and_cublas_ws",
            domain,
            total_ws,
        ));
    }
    let op = build_multistream_phase_live_set(
        "h3_denoise",
        MemoryPhase::Denoise,
        0,
        0,
        &stream_specs,
        resident_weights,
        Vec::new(),
        joint_ws,
        Vec::new(),
    )?;
    Ok(RequestLiveSetPlan {
        operations: vec![op],
        transitions: Vec::new(),
    })
}

fn compute_memory_summary(
    bundle: &StreamBundleSpec,
    condition_audio_rows: usize,
    footprint: &H3MemoryFootprintSpec,
    envelope: &H3EnvelopeSpec,
    veda_policy: &H3VedaExecutionPolicy,
) -> Result<H3DomainMemorySummary> {
    let plan = build_h3_live_set_plan(bundle, condition_audio_rows, footprint)?;
    let primary = plan.price_domain(footprint.domain.to_domain())?;
    let host = plan.price_domain(MemoryBudgetDomain::HostRam)?;
    let video_act = bundle
        .stream(&StreamId::Video)
        .map_or(0, |s| stream_activation_bytes(s, 0));
    let audio_act = bundle
        .stream(&StreamId::Audio)
        .map_or(0, |s| stream_activation_bytes(s, condition_audio_rows));
    let base_joint_ws = footprint
        .joint_attention_workspace_bytes
        .saturating_add(footprint.cublas_workspace_bytes);
    let joint_ws = base_joint_ws.saturating_add(footprint.veda_sparse_device_bytes);

    let (veda_sparse_peak, veda_dense_fallback_peak) = if veda_policy.is_veda_enabled() {
        let sparse = primary.max_operation_peak_bytes;
        let dense_fallback = primary
            .max_operation_peak_bytes
            .saturating_sub(footprint.veda_sparse_device_bytes)
            .saturating_add(footprint.veda_dense_fallback_device_bytes);
        (Some(sparse), Some(dense_fallback))
    } else {
        (None, None)
    };
    let worst_device = veda_sparse_peak
        .unwrap_or(primary.max_operation_peak_bytes)
        .max(veda_dense_fallback_peak.unwrap_or(primary.max_operation_peak_bytes));
    let fits_pool = worst_device <= envelope.pool_budget_bytes;
    let fits_total =
        worst_device.saturating_add(envelope.reserve_bytes) <= envelope.total_envelope_bytes;

    Ok(H3DomainMemorySummary {
        domain: footprint.domain,
        host_ram_peak_bytes: host.request_peak_bytes,
        video_stream_activation_bytes: video_act,
        audio_stream_activation_bytes: audio_act,
        joint_workspace_bytes: joint_ws,
        primary_operation_peak_bytes: primary.max_operation_peak_bytes,
        veda_sparse_peak_bytes: veda_sparse_peak,
        veda_dense_fallback_peak_bytes: veda_dense_fallback_peak,
        fits_pool_budget: fits_pool,
        fits_total_envelope: fits_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::veda::workspace::for_request;
    use gen_core::h3_memory::window::CudaWorkingSet;
    use gen_core::h3_memory::NativeTaskSelection;
    use gen_core::veda::workspace::Backend;
    use gen_core::{GenerationMemory, PagConfig, VedaBudget};

    #[test]
    fn h3_plan_round_trips_json_and_preserves_paired_va_shifts() {
        let m5 = H3RenderPlan::m5_canonical_t2va_turbo8(Some(VedaRequest {
            enabled: true,
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(m5.width, 1344);
        assert_eq!(m5.height, 768);
        assert_eq!(m5.frames, 345);
        assert_eq!(m5.fps, 24.0);
        assert_eq!(m5.steps, 8);
        assert_eq!(m5.video_shift, VIDEO_SIGMA_SHIFT);
        assert_eq!(m5.audio_shift, AUDIO_SIGMA_SHIFT);
        assert_eq!(m5.schedules.len(), 2);
        assert_eq!(m5.schedules[0].stream, StreamId::Video);
        assert_eq!(m5.schedules[1].stream, StreamId::Audio);
        assert_ne!(
            m5.schedules[0].schedule.sigmas(),
            m5.schedules[1].schedule.sigmas(),
            "paired Video (12.0) and Audio (3.0) schedules must remain distinct"
        );
        assert!(m5.streamed_window.enabled);
        assert_eq!(
            m5.streamed_window.component,
            Some(H3StreamComponentScope::Both)
        );
        assert_eq!(m5.streamed_window.window_size, Some(1));
        assert_eq!(m5.envelope.pool_budget_bytes, 21 * 1024 * 1024 * 1024);
        assert_eq!(m5.envelope.reserve_bytes, 3 * 1024 * 1024 * 1024);
        assert_eq!(m5.envelope.total_envelope_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(m5.envelope.cublas_workspace_bytes, 32 * 1024 * 1024);
        assert!(m5.memory.fits_pool_budget);
        assert!(m5.memory.fits_total_envelope);

        let json = m5.to_json().unwrap();
        let restored = H3RenderPlan::from_json(&json).unwrap();
        assert_eq!(restored.matched_policy_key(), m5.matched_policy_key());
        assert_eq!(restored.schedules, m5.schedules);
        assert_eq!(restored.envelope, m5.envelope);
    }

    #[test]
    fn mem3_sums_concurrent_video_and_audio_streams_plus_joint_workspace() {
        let with_audio = H3RenderPlan::m5_canonical_t2va_turbo8(Some(VedaRequest {
            enabled: true,
            ..Default::default()
        }))
        .unwrap();
        // Image plan uses single stream without audio or joint Veda workspace.
        let img_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        let image_plan = H3RenderPlan::from_image_generation_request(
            &img_req,
            H3BudgetDomainKey::DeviceVram,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(image_plan.route, H3RouteKind::ImageT2i);
        assert_eq!(image_plan.schedules.len(), 1);

        // Concurrent sums strictly exceed both trunk-alone and single-stream components.
        let fp = with_audio.cached_footprint.as_ref().unwrap();
        let single_max = with_audio
            .memory
            .video_stream_activation_bytes
            .max(with_audio.memory.audio_stream_activation_bytes);
        assert!(with_audio.memory.audio_stream_activation_bytes > 0);
        assert!(with_audio.memory.video_stream_activation_bytes > 0);
        assert!(
            with_audio.memory.primary_operation_peak_bytes > fp.dit_trunk_bytes + single_max,
            "MEM-3 must sum concurrent Video + Audio + joint workspace, never max"
        );
        assert_eq!(
            with_audio.memory.primary_operation_peak_bytes,
            fp.dit_trunk_bytes
                + fp.adaln_table_bytes
                + with_audio.memory.video_stream_activation_bytes
                + with_audio.memory.audio_stream_activation_bytes
                + with_audio.memory.joint_workspace_bytes
        );

        // Removing Veda lowers joint workspace peak strictly cleanly.
        let dense_only = H3RenderPlan::m5_canonical_t2va_turbo8(None).unwrap();
        assert!(
            with_audio.memory.primary_operation_peak_bytes
                > dense_only.memory.primary_operation_peak_bytes
        );
        // Unmatched policy comparison between Veda-on and Veda-off is refused.
        assert!(matches!(
            with_audio.assert_matched_policy(&dense_only),
            Err(Error::Unsupported(_))
        ));

        // Matched policy comparison between M5 24 GiB and 95 GiB control arm succeeds!
        let mut control_input = H3RenderPlanInput {
            route: H3RouteKind::T2va,
            domain: H3BudgetDomainKey::DeviceVram,
            width: 1344,
            height: 768,
            frames: 345,
            fps: 24.0,
            steps: 8,
            video_shift: VIDEO_SIGMA_SHIFT,
            audio_shift: AUDIO_SIGMA_SHIFT,
            seed: 1,
            condition_video_tokens: 0,
            condition_audio_rows: 0,
            reference_count: 0,
            min_generation_area: None,
            image_sampler: None,
            image_scheduler: None,
            pag_active_steps: 0,
            veda: Some(VedaRequest {
                enabled: true,
                ..Default::default()
            }),
            streamed_window: with_audio.streamed_window,
            envelope: H3EnvelopeSpec::control_95gib(),
            footprint: H3MemoryFootprintSpec::streamed_default(
                H3BudgetDomainKey::DeviceVram,
                &H3EnvelopeSpec::control_95gib(),
                true,
            ),
        };
        control_input.envelope.arm = H3EnvelopeArmKind::Control95GiB;
        let control = H3RenderPlan::build(control_input).unwrap();
        with_audio.assert_matched_policy(&control).unwrap();
    }

    #[test]
    fn cuda_window_fit_verifies_sparse_and_dense_fallback_under_24gib_vs_95gib() {
        // Synthetic weights-free 2-layer WindowAllocation sized so M5 (21 GiB pool) accommodates
        // streamed blocks + Veda workspace + dense fallback cleanly.
        let data = vec![
            ("token_refiner.proj.weight".into(), 64 * 1024 * 1024),
            (
                "transformer_blocks.0.attn.to_q.weight".into(),
                512 * 1024 * 1024,
            ),
            (
                "transformer_blocks.0.adaln_proj.linear.weight".into(),
                64 * 1024 * 1024,
            ),
            (
                "transformer_blocks.1.attn.to_q.weight".into(),
                512 * 1024 * 1024,
            ),
            (
                "transformer_blocks.1.adaln_proj.linear.weight".into(),
                64 * 1024 * 1024,
            ),
        ];
        let profile = WindowAllocation::from_resident_tensors(&data, &[], 0, false)
            .unwrap()
            .with_cuda_working_set(CudaWorkingSet {
                hidden: 5376,
                heads: 56,
                head_dim: 128,
                rotary_dim: 96,
                ffn_inner: 14336,
                element_bytes: 2,
                projection_decode_bytes: 256 * 1024 * 1024,
            })
            .unwrap();
        let task = NativeTaskAllocation {
            adapter: None,
            resident_bytes: 1_200_000_000,
            constructor_bytes: 0,
            patch_size: [1, 2, 2],
            reference_task: false,
            evicted_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: Vec::new(),
            window: None,
        };
        let mut req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 1344,
            height: 768,
            frames: Some(345),
            steps: Some(8),
            veda: Some(VedaRequest {
                enabled: true,
                generated: VedaBudget::SparsityPercent(70.0),
                ..Default::default()
            }),
            memory: Some(GenerationMemory {
                stage_residency: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                ..Default::default()
            }),
            ..Default::default()
        };
        let bound = for_request(&req, Backend::Cuda).unwrap().unwrap();
        let rows = NativeRows {
            task: NativeTaskSelection::Base,
            packed: 104_030,
            conditioning: 64,
            schedule: 18,
            visual_elements: 102 * 24 * 42 * 4,
            global: 1214,
        };
        let m5 = H3RenderPlan::from_video_generation_request(
            &req,
            H3BudgetDomainKey::DeviceVram,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        let fit = m5
            .evaluate_cuda_window_fit(&profile, &task, rows, Some(&bound))
            .unwrap();
        assert!(fit.bounded_cuda_denoise_with_veda_bytes.is_some());
        assert!(fit.veda_sparse_device_bytes.is_some());
        assert!(fit.veda_dense_fallback_device_bytes.is_some());
        assert!(fit.sparse_fits_pool);
        assert!(fit.dense_fallback_fits_pool);
        assert!(fit.fits_envelope);

        // Control arm with exact matched policy also fits.
        let control = H3RenderPlan::from_video_generation_request(
            &req,
            H3BudgetDomainKey::DeviceVram,
            H3EnvelopeSpec::control_95gib(),
        )
        .unwrap();
        m5.assert_matched_policy(&control).unwrap();

        // Removing Veda fails matched-policy comparison (`Compare against matched policy only`).
        req.veda = None;
        let m5_dense = H3RenderPlan::from_video_generation_request(
            &req,
            H3BudgetDomainKey::DeviceVram,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert!(m5.assert_matched_policy(&m5_dense).is_err());
    }

    #[test]
    fn plan_refuses_invalid_combinations_before_weight_load() {
        let base_video = || GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 1344,
            height: 768,
            frames: Some(124),
            ..Default::default()
        };
        // Off-lattice frame count.
        let mut bad_frames = base_video();
        bad_frames.frames = Some(125);
        assert!(H3RenderPlan::from_video_generation_request(
            &bad_frames,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());

        // Over area cap (1344x1344 > 768*1344).
        let mut over_area = base_video();
        over_area.width = 1344;
        over_area.height = 1344;
        assert!(H3RenderPlan::from_video_generation_request(
            &over_area,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());

        // Guidance refused on guidance-distilled video.
        let mut guided = base_video();
        guided.guidance = Some(3.0);
        assert!(matches!(
            H3RenderPlan::from_video_generation_request(
                &guided,
                H3BudgetDomainKey::UnifiedMemory,
                H3EnvelopeSpec::m5_24gib()
            ),
            Err(Error::Unsupported(_))
        ));

        // Non-positive shift refused.
        let mut bad_shift = base_video();
        bad_shift.scheduler_shift = Some(0.0);
        assert!(H3RenderPlan::from_video_generation_request(
            &bad_shift,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());

        // Keyframe + Reference mix refused, and keyframe strength != 1.0 refused.
        let img = gen_core::Image {
            width: 64,
            height: 64,
            pixels: vec![0; 64 * 64 * 3],
        };
        let mut mixed = base_video();
        mixed.conditioning = vec![
            Conditioning::Keyframe {
                image: img.clone(),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::Reference {
                image: img.clone(),
                strength: None,
            },
        ];
        assert!(H3RenderPlan::from_video_generation_request(
            &mixed,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());

        // Keyframe strength != 1.0 refused.
        let mut bad_strength = base_video();
        bad_strength.conditioning = vec![Conditioning::Keyframe {
            image: img,
            frame_idx: 0,
            strength: 0.5,
        }];
        assert!(H3RenderPlan::from_video_generation_request(
            &bad_strength,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());

        // Streamed Dit-only or window != 1 refused before weight load.
        let mut dit_only = base_video();
        dit_only.memory = Some(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_component: Some(TransformerComponent::Dit),
            transformer_window_size: Some(1),
            ..Default::default()
        });
        assert!(matches!(
            H3RenderPlan::from_video_generation_request(
                &dit_only,
                H3BudgetDomainKey::UnifiedMemory,
                H3EnvelopeSpec::m5_24gib()
            ),
            Err(Error::Unsupported(_))
        ));

        // Image route: invalid steps / PAG / guidance refused cleanly.
        let mut bad_img = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 2048,
            height: 2048,
            steps: Some(8),
            pag: Some(PagConfig {
                scale: 1.2,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(H3RenderPlan::from_image_generation_request(
            &bad_img,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib()
        )
        .is_err());
        bad_img.steps = Some(6);
        let ok_img = H3RenderPlan::from_image_generation_request(
            &bad_img,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(ok_img.pag_active_steps, 4);
    }

    #[test]
    fn oversized_reference_image_is_priced_at_resized_vae_dimensions() {
        let oversized = gen_core::Image {
            width: 4096,
            height: 3000,
            pixels: Vec::new(),
        };

        // Video Ref2VA: default `Match` resize mode prices against resolved 1344x768 canvas -> 1184x864.
        let video_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: H3_CANVAS_DEFAULT_WIDTH,
            height: H3_CANVAS_DEFAULT_HEIGHT,
            frames: Some(124),
            conditioning: vec![Conditioning::Reference {
                image: oversized.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        let (expected_vw, expected_vh) = crate::resolve_reference_image_size(
            4096,
            3000,
            H3_CANVAS_DEFAULT_WIDTH,
            H3_CANVAS_DEFAULT_HEIGHT,
            effective_reference_image_resize_mode(&video_req),
            effective_reference_image_short_edge(&video_req),
            REFERENCE_IMAGE_MULTIPLE,
        )
        .unwrap();
        assert_eq!((expected_vw, expected_vh), (1184, 864));
        let expected_video_tokens =
            (expected_vw / SPATIAL_STRIDE) as usize * (expected_vh / SPATIAL_STRIDE) as usize;
        let raw_tokens = (4096 / SPATIAL_STRIDE) as usize * (3000 / SPATIAL_STRIDE) as usize;
        let video_plan = H3RenderPlan::from_video_generation_request(
            &video_req,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(video_plan.route, H3RouteKind::Ref2va);
        assert_eq!(video_plan.condition_video_tokens, expected_video_tokens);
        assert!(video_plan.condition_video_tokens < raw_tokens);

        // Video Ref2VA with `Max` mode (short edge capped at 2048 -> 2784x2048).
        let mut video_max_req = video_req;
        video_max_req.reference_image_resize_mode = Some(gen_core::ReferenceImageResizeMode::Max);
        let (max_vw, max_vh) = crate::resolve_reference_image_size(
            4096,
            3000,
            H3_CANVAS_DEFAULT_WIDTH,
            H3_CANVAS_DEFAULT_HEIGHT,
            effective_reference_image_resize_mode(&video_max_req),
            effective_reference_image_short_edge(&video_max_req),
            REFERENCE_IMAGE_MULTIPLE,
        )
        .unwrap();
        assert_eq!((max_vw, max_vh), (2784, 2048));
        let video_max_plan = H3RenderPlan::from_video_generation_request(
            &video_max_req,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(
            video_max_plan.condition_video_tokens,
            (max_vw / SPATIAL_STRIDE) as usize * (max_vh / SPATIAL_STRIDE) as usize
        );
        assert!(video_max_plan.condition_video_tokens < raw_tokens);

        // Image edit: `reference_vae_size(4096, 3000)` caps short edge at 2048 -> (2784, 2048).
        let (expected_iw, expected_ih) = reference_vae_size(4096, 3000);
        assert_eq!((expected_iw, expected_ih), (2784, 2048));
        let expected_image_tokens =
            (expected_iw / IMAGE_STRIDE) as usize * (expected_ih / IMAGE_STRIDE) as usize;
        let image_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 2048,
            height: 2048,
            conditioning: vec![Conditioning::Reference {
                image: oversized,
                strength: None,
            }],
            ..Default::default()
        };
        let image_plan = H3RenderPlan::from_image_generation_request(
            &image_req,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(image_plan.route, H3RouteKind::ImageEdit);
        assert_eq!(image_plan.condition_video_tokens, expected_image_tokens);
        assert!(image_plan.condition_video_tokens < raw_tokens);
    }

    #[test]
    fn ref2va_video_planning_validates_borrowed_media_without_cloning_buffers() {
        // Large-video metadata Ref2VA clip (120 frames, 1280x720 declared geometry, plus stereo
        // audio) validated strictly via borrowed `Ref2VaReferenceRef` slices without cloning
        // pixel or waveform buffers.
        let frames: Vec<gen_core::Image> = (0..120)
            .map(|_| gen_core::Image {
                width: 1280,
                height: 720,
                pixels: vec![0u8; 3],
            })
            .collect();
        let audio = gen_core::AudioTrack {
            sample_rate: 32_000,
            channels: 2,
            samples: vec![0.0f32; 64],
            stems: Vec::new(),
        };
        let req = GenerationRequest {
            prompt: "a sailboat gliding past cliffs".into(),
            width: H3_CANVAS_DEFAULT_WIDTH,
            height: H3_CANVAS_DEFAULT_HEIGHT,
            frames: Some(124),
            steps: Some(H3_TURBO_8_STEPS),
            conditioning: vec![Conditioning::ReferenceVideo {
                frames,
                fps: 24.0,
                audio: Some(audio),
            }],
            ..Default::default()
        };

        let plan = H3RenderPlan::from_video_generation_request(
            &req,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .unwrap();
        assert_eq!(plan.route, H3RouteKind::Ref2va);
        assert_eq!(plan.reference_count, 1);
        assert!(plan.condition_video_tokens > 0);
        assert!(plan.condition_audio_rows > 0);

        // Borrowed validator must still reject invalid Ref2VA payloads (e.g., audio-only).
        let audio_only = GenerationRequest {
            prompt: "audio only".into(),
            width: H3_CANVAS_DEFAULT_WIDTH,
            height: H3_CANVAS_DEFAULT_HEIGHT,
            frames: Some(124),
            steps: Some(H3_TURBO_8_STEPS),
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: gen_core::AudioTrack {
                    sample_rate: 32_000,
                    channels: 2,
                    samples: vec![0.0f32; 64],
                    stems: Vec::new(),
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(H3RenderPlan::from_video_generation_request(
            &audio_only,
            H3BudgetDomainKey::UnifiedMemory,
            H3EnvelopeSpec::m5_24gib(),
        )
        .is_err());
    }
}
