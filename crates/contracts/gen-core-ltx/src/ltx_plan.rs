//! Weights-free pre-load LTX execution plan (`DEC-3`, `DEC-13`, `PIPE-4`, `PIPE-6`, `MEM-3`).
//!
//! Models both LTX-2.3 and LTX-2.5 multi-stream video+audio pipelines across Distilled (8+3),
//! Dev (30-step 4-pass guided stage 1 + 3-step distilled stage 2), IC-LoRA token-native clips, and
//! DFR generated-keyframe + temporal-upsample rounds (`TEMPORAL_SIGMAS`), along with explicit
//! stream-targeted transitions and `MEM-3` concurrent/transition peak accounting.

use serde::{Deserialize, Serialize};

use gen_core::multistream::{
    build_multistream_phase_live_set, build_stream_transition_live_set, LearnedUpsamplerComponent,
    PerTokenTimePolicy, ResolvedSchedule, ScheduleSource, StreamBundleSpec, StreamGeometry,
    StreamId, StreamLatentDescriptor, StreamMemorySpec, StreamSpec, StreamTransition,
    StreamTransitionKind,
};
use gen_core::{
    Conditioning, Error, GenerationRequest, MemoryBudgetDomain, MemoryPhase, PricedAllocation,
    RequestLiveSetPlan, Result, LTX_VIDEO_LATENT_SPACE,
};

use crate::ltx_dfr::{resolve_canvas, MAX_TEMPORAL_UPSAMPLE_ROUNDS, TEMPORAL_ANCESTRAL_ETA};

/// Half-resolution stage-one distilled rectified-flow sigma schedule (`8` outer steps).
pub const LTX_STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875, 0.0,
];

/// Full-resolution stage-two distilled refinement sigma schedule (`3` outer steps).
pub const LTX_STAGE2_SIGMAS: [f32; 4] = [0.909_375, 0.725, 0.421_875, 0.0];

/// DFR temporal-round sigma schedule (`4` outer steps, RF ancestral `eta = 0.5`).
pub const LTX_TEMPORAL_SIGMAS: [f32; 5] = [0.975, 0.909_375, 0.725, 0.421_875, 0.0];

pub const LTX_NATIVE_DISTILLED_STEPS: u32 = LTX_STAGE1_SIGMAS.len() as u32 - 1;
pub const LTX_STAGE2_STEPS: u32 = LTX_STAGE2_SIGMAS.len() as u32 - 1;
pub const LTX_DEV_STEPS: u32 = 30;
pub const LTX_SPATIAL_SCALE: u32 = 32;
pub const LTX_TEMPORAL_SCALE: u32 = 8;
pub const LTX_SIZE_MULTIPLE: u32 = 2 * LTX_SPATIAL_SCALE;
pub const LTX_MAX_FRAMES: u32 = 1025;
pub const LTX_LATENT_CHANNELS: u16 = 128;
pub const LTX_AUDIO_LATENT_CHANNELS: u16 = 8;
pub const LTX_AUDIO_MEL_BINS: u16 = 16;

pub const LTX_SPATIAL_UPSAMPLER_COMPONENT_NAME: &str = "ltx_spatial_upsampler_x2";
pub const LTX_TEMPORAL_UPSAMPLER_COMPONENT_NAME: &str = "ltx_temporal_upsampler_x2";
pub const LTX_UPSAMPLER_NORMALIZATION: &str = "vae_per_channel_denorm_renorm";

const AUDIO_LATENTS_PER_SECOND: f64 = 25.0;

fn py_round(x: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Official LTX audio latent frame count matching `compute_audio_frames` (`25 Hz`).
pub fn compute_audio_frames(num_video_frames: usize, fps: f64) -> usize {
    let duration = num_video_frames as f64 / fps;
    py_round(duration * AUDIO_LATENTS_PER_SECOND).max(1) as usize
}

/// Generate the canonical Dev linear stage-1 sigma table (`N=30` steps from `1.0 -> 0.0`, `DEC-3`).
pub fn ltx_dev_sigmas(steps: u32) -> Result<Vec<f32>> {
    if steps != LTX_DEV_STEPS {
        return Err(Error::Unsupported(format!(
            "ltx_dev: stage-1 step count is fixed at {LTX_DEV_STEPS}, got {steps}"
        )));
    }
    Ok((0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxModelFamily {
    Ltx23,
    Ltx25,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LtxVariant {
    Distilled,
    Dev,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxOrchestrationRoute {
    VideoOnlyGrid,
    AvGrid,
    AvTokensIcLora,
    AvDfr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxConditioningProfile {
    T2v,
    ReplaceLatentKeyframes {
        count: usize,
    },
    InContextClips {
        count: usize,
        clip_tokens: usize,
    },
    DfrGeneratedKeyframes {
        slot_count: usize,
        temporal_upsample_rounds: u32,
    },
}

/// Physical component and workspace bytes used by [`LtxRenderPlan`] `MEM-3` accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LtxMemoryFootprintSpec {
    pub domain: MemoryBudgetDomainKey,
    pub host_staging_bytes: u64,
    pub av_dit_trunk_bytes: u64,
    pub spatial_upsampler_weights_bytes: u64,
    pub spatial_upsampler_workspace_bytes: u64,
    pub temporal_upsampler_weights_bytes: u64,
    pub temporal_upsampler_workspace_bytes: u64,
    pub joint_cross_attention_workspace_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBudgetDomainKey {
    DeviceVram,
    UnifiedMemory,
}

impl MemoryBudgetDomainKey {
    pub const fn to_domain(self) -> MemoryBudgetDomain {
        match self {
            Self::DeviceVram => MemoryBudgetDomain::DeviceVram,
            Self::UnifiedMemory => MemoryBudgetDomain::UnifiedMemory,
        }
    }
}

impl Default for LtxMemoryFootprintSpec {
    fn default() -> Self {
        Self {
            domain: MemoryBudgetDomainKey::UnifiedMemory,
            host_staging_bytes: 64 * 1024 * 1024,
            // Representative pre-packed Q4 AvDiT trunk (alias-deduplicated across Video & Audio).
            av_dit_trunk_bytes: 11_000_000_000,
            spatial_upsampler_weights_bytes: 996_000_000,
            spatial_upsampler_workspace_bytes: 384 * 1024 * 1024,
            temporal_upsampler_weights_bytes: 262_000_000,
            temporal_upsampler_workspace_bytes: 256 * 1024 * 1024,
            joint_cross_attention_workspace_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Pre-load inputs validated by [`LtxRenderPlan::build`] before opening or allocating model weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxRenderPlanInput {
    pub family: LtxModelFamily,
    pub variant: LtxVariant,
    pub route: LtxOrchestrationRoute,
    pub conditioning: LtxConditioningProfile,
    pub sampler: Option<String>,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: f32,
    pub seed: u64,
    pub explicit_steps: Option<u32>,
    pub has_temporal_upsampler: bool,
    pub footprint: LtxMemoryFootprintSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxStageScheduleEntry {
    pub stream: StreamId,
    pub schedule: ResolvedSchedule,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxStagePlanSpec {
    pub stage_id: String,
    pub pass_index: usize,
    pub bundle: StreamBundleSpec,
    pub schedules: Vec<LtxStageScheduleEntry>,
    pub adapter_pass: usize,
    pub four_pass_guidance: bool,
    pub rf_ancestral_eta: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LtxDomainMemorySummary {
    pub domain: MemoryBudgetDomainKey,
    pub host_ram_peak_bytes: u64,
    pub primary_operation_peak_bytes: u64,
    pub primary_transition_peak_bytes: u64,
    pub primary_request_peak_bytes: u64,
    pub dominant_stage_id: String,
}

/// Complete serializable, pre-load LTX execution and `MEM-3` memory plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxRenderPlan {
    pub family: LtxModelFamily,
    pub variant: LtxVariant,
    pub route: LtxOrchestrationRoute,
    pub conditioning: LtxConditioningProfile,
    pub sampler: Option<String>,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub canvas_frames: u32,
    pub fps: f32,
    pub seed: u64,
    pub audio_enabled: bool,
    pub total_outer_steps: usize,
    pub stages: Vec<LtxStagePlanSpec>,
    pub transitions: Vec<StreamTransition>,
    pub memory: LtxDomainMemorySummary,
    #[serde(skip)]
    cached_footprint: Option<LtxMemoryFootprintSpec>,
}

fn is_default_euler_sampler(sampler: Option<&str>) -> bool {
    matches!(sampler, None | Some("euler") | Some("rectified-flow"))
}

fn bf16_latent_tensor_bytes(elements: usize) -> u64 {
    (elements as u64).saturating_mul(2)
}

impl LtxRenderPlanInput {
    /// Construct a backend-neutral pre-load plan input directly from a [`GenerationRequest`].
    pub fn from_generation_request(
        family: LtxModelFamily,
        variant: LtxVariant,
        domain: MemoryBudgetDomainKey,
        req: &GenerationRequest,
        resolved_frames: u32,
        fps: f32,
        has_temporal_upsampler: bool,
    ) -> Self {
        let generated_slots = req.num_generated_keyframes.unwrap_or(0);
        let temporal_rounds = req.temporal_upsample_rounds.unwrap_or(0);
        let uses_dfr = generated_slots > 0 || temporal_rounds > 0;
        let clip_count = req.video_clips().len() + usize::from(req.control_clip().is_some());
        let keyframe_count = req
            .conditioning
            .iter()
            .filter(|entry| {
                matches!(
                    entry,
                    Conditioning::Reference { .. }
                        | Conditioning::Keyframe { .. }
                        | Conditioning::MultiReference { .. }
                )
            })
            .count();

        let (route, conditioning) = if uses_dfr && clip_count == 0 {
            (
                LtxOrchestrationRoute::AvDfr,
                LtxConditioningProfile::DfrGeneratedKeyframes {
                    slot_count: generated_slots as usize,
                    temporal_upsample_rounds: temporal_rounds,
                },
            )
        } else if clip_count > 0 {
            (
                if uses_dfr {
                    LtxOrchestrationRoute::AvDfr
                } else {
                    LtxOrchestrationRoute::AvTokensIcLora
                },
                LtxConditioningProfile::InContextClips {
                    count: clip_count,
                    clip_tokens: 0,
                },
            )
        } else if keyframe_count > 0 {
            (
                LtxOrchestrationRoute::AvGrid,
                LtxConditioningProfile::ReplaceLatentKeyframes {
                    count: keyframe_count,
                },
            )
        } else {
            (LtxOrchestrationRoute::AvGrid, LtxConditioningProfile::T2v)
        };

        Self {
            family,
            variant,
            route,
            conditioning,
            sampler: req.sampler.clone(),
            width: req.width,
            height: req.height,
            frames: resolved_frames,
            fps,
            seed: req.seed.unwrap_or(0),
            explicit_steps: req.steps,
            has_temporal_upsampler,
            footprint: LtxMemoryFootprintSpec {
                domain,
                ..LtxMemoryFootprintSpec::default()
            },
        }
    }
}

impl LtxRenderPlan {
    /// Validate and build a pre-load [`LtxRenderPlan`] directly from a [`GenerationRequest`].
    pub fn from_generation_request(
        family: LtxModelFamily,
        variant: LtxVariant,
        domain: MemoryBudgetDomainKey,
        req: &GenerationRequest,
        resolved_frames: u32,
        fps: f32,
        has_temporal_upsampler: bool,
    ) -> Result<Self> {
        Self::build(LtxRenderPlanInput::from_generation_request(
            family,
            variant,
            domain,
            req,
            resolved_frames,
            fps,
            has_temporal_upsampler,
        ))
    }

    /// Validate geometry, variant, sampler, conditioning, and upsampler requirements BEFORE
    /// loading weights, refusing unsupported combinations with typed errors (`DEC-13`).
    pub fn build(input: LtxRenderPlanInput) -> Result<Self> {
        if input.width == 0
            || input.height == 0
            || !input.width.is_multiple_of(LTX_SIZE_MULTIPLE)
            || !input.height.is_multiple_of(LTX_SIZE_MULTIPLE)
        {
            return Err(Error::Msg(format!(
                "ltx: width/height must be positive multiples of {LTX_SIZE_MULTIPLE}, got {}x{}",
                input.width, input.height
            )));
        }
        if input.frames == 0 || input.frames % LTX_TEMPORAL_SCALE != 1 {
            return Err(Error::Msg(format!(
                "ltx: frames must satisfy frames % {LTX_TEMPORAL_SCALE} == 1, got {}",
                input.frames
            )));
        }
        if input.frames > LTX_MAX_FRAMES {
            return Err(Error::Msg(format!(
                "ltx: frames {} exceeds maximum {LTX_MAX_FRAMES}",
                input.frames
            )));
        }
        if !input.fps.is_finite() || input.fps <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx: fps must be finite and > 0.0, got {}",
                input.fps
            )));
        }
        if input.footprint.spatial_upsampler_workspace_bytes == 0 {
            return Err(Error::Unsupported(
                "ltx: spatial upsampler transition requires non-zero learned workspace_bytes (MEM-3)"
                    .into(),
            ));
        }

        // Dev vs Distilled trajectory and curated-sampler rules.
        match input.variant {
            LtxVariant::Dev => {
                if input.family == LtxModelFamily::Ltx23 {
                    return Err(Error::Unsupported(
                        "ltx: Dev variant requires an LTX-2.5 split bundle".into(),
                    ));
                }
                if input.sampler.is_some() {
                    return Err(Error::Unsupported(
                        "ltx_2_5: Dev transformer executes its 30-step guided Euler trajectory; curated samplers are unsupported"
                            .into(),
                    ));
                }
                if let Some(steps) = input.explicit_steps {
                    if steps != LTX_DEV_STEPS {
                        return Err(Error::Unsupported(format!(
                            "ltx_2_5: Dev variant requires {LTX_DEV_STEPS} steps, got {steps}"
                        )));
                    }
                }
            }
            LtxVariant::Distilled => {
                if let Some(steps) = input.explicit_steps {
                    if steps != LTX_NATIVE_DISTILLED_STEPS {
                        return Err(Error::Unsupported(format!(
                            "ltx: Distilled variant requires fixed {LTX_NATIVE_DISTILLED_STEPS} stage-1 steps, got {steps}"
                        )));
                    }
                }
            }
        }

        let is_curated_non_euler = !is_default_euler_sampler(input.sampler.as_deref());
        let (canvas_frames, temporal_rounds, per_token_video) = match (
            &input.route,
            &input.conditioning,
        ) {
            (
                LtxOrchestrationRoute::VideoOnlyGrid,
                LtxConditioningProfile::DfrGeneratedKeyframes { .. },
            ) => {
                return Err(Error::Unsupported(
                    "ltx: VideoOnlyGrid does not support DFR generated-keyframe temporal orchestration"
                        .into(),
                ));
            }
            (LtxOrchestrationRoute::AvDfr, LtxConditioningProfile::InContextClips { .. }) => {
                return Err(Error::Unsupported(
                    "ltx_2_5: DFR cannot be combined with IC-LoRA in-context clips".into(),
                ));
            }
            (
                _,
                LtxConditioningProfile::DfrGeneratedKeyframes {
                    slot_count,
                    temporal_upsample_rounds,
                },
            ) => {
                if input.family == LtxModelFamily::Ltx23 {
                    return Err(Error::Unsupported(
                        "ltx_2_3: DFR generated keyframes and temporal_upsample_rounds require LTX >= 2.5"
                            .into(),
                    ));
                }
                if *temporal_upsample_rounds > MAX_TEMPORAL_UPSAMPLE_ROUNDS {
                    return Err(Error::Unsupported(format!(
                        "ltx_2_5: temporal_upsample_rounds={temporal_upsample_rounds} exceeds maximum {MAX_TEMPORAL_UPSAMPLE_ROUNDS}"
                    )));
                }
                if *temporal_upsample_rounds > 0 && !input.has_temporal_upsampler {
                    return Err(Error::Unsupported(
                        "ltx_2_5: temporal_upsample_rounds > 0 requires the temporal latent upsampler component"
                            .into(),
                    ));
                }
                if *temporal_upsample_rounds > 0
                    && input.footprint.temporal_upsampler_workspace_bytes == 0
                {
                    return Err(Error::Unsupported(
                        "ltx_2_5: temporal upsampler transition requires non-zero workspace_bytes (MEM-3)"
                            .into(),
                    ));
                }
                if *slot_count == 0 && *temporal_upsample_rounds == 0 {
                    return Err(Error::Unsupported(
                        "ltx_2_5: DFR profile requires slot_count > 0 or temporal_upsample_rounds > 0"
                            .into(),
                    ));
                }
                if input.sampler.is_some() && !is_default_euler_sampler(input.sampler.as_deref()) {
                    return Err(Error::Unsupported(
                        "ltx_2_5: DFR routes do not support curated samplers".into(),
                    ));
                }
                let canvas = if *temporal_upsample_rounds > 0 {
                    let (canvas, _, positions) =
                        resolve_canvas(i64::from(input.frames), i64::from(LTX_TEMPORAL_SCALE))?;
                    if positions.is_empty() && *slot_count == 0 {
                        return Err(Error::Unsupported(
                            "ltx_2_5: DFR canvas resolved zero keyframe slot positions".into(),
                        ));
                    }
                    canvas as u32
                } else {
                    input.frames
                };
                (canvas, *temporal_rounds_ref(temporal_upsample_rounds), true)
            }
            (
                LtxOrchestrationRoute::AvTokensIcLora,
                LtxConditioningProfile::InContextClips { count, .. },
            ) => {
                if *count == 0 {
                    return Err(Error::Unsupported(
                        "ltx: AvTokensIcLora requires at least one conditioning clip".into(),
                    ));
                }
                if is_curated_non_euler {
                    return Err(Error::Unsupported(
                        "ltx: curated samplers apply to T2V only; IC-LoRA clips require native Euler"
                            .into(),
                    ));
                }
                (input.frames, 0, true)
            }
            (_, LtxConditioningProfile::ReplaceLatentKeyframes { count }) => {
                if *count == 0 {
                    return Err(Error::Unsupported(
                        "ltx: ReplaceLatentKeyframes requires count > 0".into(),
                    ));
                }
                if is_curated_non_euler {
                    return Err(Error::Unsupported(
                        "ltx: curated samplers apply to T2V only; I2V/keyframes require per-token-sigma native Euler"
                            .into(),
                    ));
                }
                (input.frames, 0, true)
            }
            (_, LtxConditioningProfile::T2v) => {
                if matches!(input.route, LtxOrchestrationRoute::AvTokensIcLora) {
                    return Err(Error::Unsupported(
                        "ltx: AvTokensIcLora requires InContextClips conditioning".into(),
                    ));
                }
                (input.frames, 0, false)
            }
            (_, LtxConditioningProfile::InContextClips { .. }) => {
                return Err(Error::Unsupported(
                    "ltx: InContextClips conditioning requires AvTokensIcLora route".into(),
                ));
            }
        };

        let audio_enabled = input.route != LtxOrchestrationRoute::VideoOnlyGrid;
        let lf = ((canvas_frames - 1) / LTX_TEMPORAL_SCALE + 1) as usize;
        let h2 = (input.height / LTX_SPATIAL_SCALE) as usize;
        let w2 = (input.width / LTX_SPATIAL_SCALE) as usize;
        let h1 = h2 / 2;
        let w1 = w2 / 2;
        let audio_frames = compute_audio_frames(canvas_frames as usize, f64::from(input.fps));
        let audio_duration_millis = ((f64::from(canvas_frames) / f64::from(input.fps)) * 1000.0)
            .round()
            .max(1.0) as u64;

        let stage1_table = match input.variant {
            LtxVariant::Distilled => ScheduleSource::ExplicitTable {
                table_id: "ltx_stage1_sigmas".into(),
                sigmas: LTX_STAGE1_SIGMAS.to_vec(),
            }
            .resolve()?,
            LtxVariant::Dev => ScheduleSource::FreshPerPass {
                table_id: "ltx_dev_stage1_linear_30".into(),
                sigmas: ltx_dev_sigmas(LTX_DEV_STEPS)?,
            }
            .resolve()?,
        };
        let stage2_table = ScheduleSource::ExplicitTable {
            table_id: "ltx_stage2_sigmas".into(),
            sigmas: LTX_STAGE2_SIGMAS.to_vec(),
        }
        .resolve()?;

        let video_policy = if per_token_video {
            PerTokenTimePolicy::PerTokenMask
        } else {
            PerTokenTimePolicy::Uniform
        };

        let make_audio_stream = || StreamSpec {
            id: StreamId::Audio,
            latent_space: StreamLatentDescriptor::audio(
                LTX_AUDIO_LATENT_CHANNELS,
                LTX_AUDIO_MEL_BINS,
                "identity",
            ),
            geometry: StreamGeometry::AudioGrid {
                audio_frames,
                mel_bins: LTX_AUDIO_MEL_BINS,
                channels: LTX_AUDIO_LATENT_CHANNELS,
                duration_millis: audio_duration_millis,
            },
            // Audio stays strictly uniform-sigma even when video uses per-token masks (PIPE-6).
            token_time_policy: PerTokenTimePolicy::Uniform,
        };

        let extra_tokens = match &input.conditioning {
            LtxConditioningProfile::InContextClips { clip_tokens, .. } => *clip_tokens,
            LtxConditioningProfile::DfrGeneratedKeyframes { slot_count, .. } => {
                slot_count.saturating_mul(h1 * w1)
            }
            _ => 0,
        };

        let s1_video_geom = match input.route {
            LtxOrchestrationRoute::AvTokensIcLora | LtxOrchestrationRoute::AvDfr => {
                StreamGeometry::VideoTokens {
                    width: input.width / 2,
                    height: input.height / 2,
                    latent_frames: lf,
                    target_tokens: lf * h1 * w1,
                    conditioning_tokens: extra_tokens,
                    channels: LTX_LATENT_CHANNELS,
                }
            }
            LtxOrchestrationRoute::VideoOnlyGrid | LtxOrchestrationRoute::AvGrid => {
                StreamGeometry::VideoGrid {
                    width: input.width / 2,
                    height: input.height / 2,
                    latent_frames: lf,
                    latent_height: h1,
                    latent_width: w1,
                    channels: LTX_LATENT_CHANNELS,
                }
            }
        };
        let s1_video = StreamSpec::from_latent_space(
            StreamId::Video,
            &LTX_VIDEO_LATENT_SPACE,
            s1_video_geom,
            video_policy,
        )?;

        let mut s1_streams = vec![s1_video];
        let mut s1_schedules = vec![LtxStageScheduleEntry {
            stream: StreamId::Video,
            schedule: stage1_table.clone(),
        }];
        if audio_enabled {
            s1_streams.push(make_audio_stream());
            s1_schedules.push(LtxStageScheduleEntry {
                stream: StreamId::Audio,
                schedule: stage1_table.clone(),
            });
        }
        let stage1 = LtxStagePlanSpec {
            stage_id: "stage1_base".into(),
            pass_index: 0,
            bundle: StreamBundleSpec::new("stage1_bundle", s1_streams)?,
            schedules: s1_schedules,
            adapter_pass: 0,
            four_pass_guidance: input.variant == LtxVariant::Dev,
            rf_ancestral_eta: None,
        };

        let s2_video_geom = match input.route {
            LtxOrchestrationRoute::AvDfr => StreamGeometry::VideoTokens {
                width: input.width,
                height: input.height,
                latent_frames: lf,
                target_tokens: lf * h2 * w2,
                conditioning_tokens: extra_tokens,
                channels: LTX_LATENT_CHANNELS,
            },
            _ => StreamGeometry::VideoGrid {
                width: input.width,
                height: input.height,
                latent_frames: lf,
                latent_height: h2,
                latent_width: w2,
                channels: LTX_LATENT_CHANNELS,
            },
        };
        let s2_video = StreamSpec::from_latent_space(
            StreamId::Video,
            &LTX_VIDEO_LATENT_SPACE,
            s2_video_geom,
            video_policy,
        )?;
        let mut s2_streams = vec![s2_video];
        let mut s2_schedules = vec![LtxStageScheduleEntry {
            stream: StreamId::Video,
            schedule: stage2_table.clone(),
        }];
        if audio_enabled {
            s2_streams.push(make_audio_stream());
            s2_schedules.push(LtxStageScheduleEntry {
                stream: StreamId::Audio,
                schedule: stage2_table.clone(),
            });
        }
        let stage2 = LtxStagePlanSpec {
            stage_id: "stage2_refine".into(),
            pass_index: 1,
            bundle: StreamBundleSpec::new("stage2_bundle", s2_streams)?,
            schedules: s2_schedules,
            adapter_pass: 1,
            four_pass_guidance: false,
            rf_ancestral_eta: None,
        };

        // Stream-targeted boundary transitions (PIPE-4).
        let mut transitions = vec![StreamTransition::new(
            "stage1_to_stage2_video_spatial_upsample_renoise",
            vec![StreamId::Video],
            StreamTransitionKind::Composed(vec![
                StreamTransitionKind::LatentResample {
                    upsampler: Some(LearnedUpsamplerComponent {
                        name: LTX_SPATIAL_UPSAMPLER_COMPONENT_NAME.into(),
                        normalization: LTX_UPSAMPLER_NORMALIZATION.into(),
                        workspace_bytes: input.footprint.spatial_upsampler_workspace_bytes,
                    }),
                    spatial_factor: (2, 2),
                    temporal_factor: 1,
                },
                StreamTransitionKind::Renoise {
                    sigma: LTX_STAGE2_SIGMAS[0],
                },
            ]),
        )?];
        transitions[0].validate_against_bundle(&stage1.bundle)?;

        if audio_enabled {
            let audio_tr = StreamTransition::new(
                "stage1_to_stage2_audio_renoise",
                vec![StreamId::Audio],
                StreamTransitionKind::Renoise {
                    sigma: LTX_STAGE2_SIGMAS[0],
                },
            )?;
            audio_tr.validate_against_bundle(&stage1.bundle)?;
            transitions.push(audio_tr);
        }

        let mut stages = vec![stage1, stage2];
        let mut total_outer_steps = stage1_table.num_steps() + stage2_table.num_steps();

        let temporal_table = ScheduleSource::ExplicitTable {
            table_id: "ltx_temporal_sigmas".into(),
            sigmas: LTX_TEMPORAL_SIGMAS.to_vec(),
        }
        .resolve()?;
        let mut round_lf = lf;
        for round in 1..=temporal_rounds {
            round_lf = (round_lf * 2).saturating_sub(1).max(1);
            let round_video = StreamSpec::from_latent_space(
                StreamId::Video,
                &LTX_VIDEO_LATENT_SPACE,
                StreamGeometry::VideoTokens {
                    width: input.width,
                    height: input.height,
                    latent_frames: round_lf,
                    target_tokens: round_lf * h2 * w2,
                    conditioning_tokens: extra_tokens,
                    channels: LTX_LATENT_CHANNELS,
                },
                PerTokenTimePolicy::PerTokenMask,
            )?;
            let round_stage = LtxStagePlanSpec {
                stage_id: format!("temporal_round_{round}"),
                pass_index: 1 + round as usize,
                bundle: StreamBundleSpec::new(
                    format!("temporal_round_{round}_bundle"),
                    vec![round_video],
                )?,
                schedules: vec![LtxStageScheduleEntry {
                    stream: StreamId::Video,
                    schedule: temporal_table.clone(),
                }],
                adapter_pass: 0,
                four_pass_guidance: false,
                rf_ancestral_eta: Some(TEMPORAL_ANCESTRAL_ETA),
            };
            let tr = StreamTransition::new(
                format!("temporal_upsample_round_{round}"),
                vec![StreamId::Video],
                StreamTransitionKind::Composed(vec![
                    StreamTransitionKind::LatentResample {
                        upsampler: Some(LearnedUpsamplerComponent {
                            name: LTX_TEMPORAL_UPSAMPLER_COMPONENT_NAME.into(),
                            normalization: LTX_UPSAMPLER_NORMALIZATION.into(),
                            workspace_bytes: input.footprint.temporal_upsampler_workspace_bytes,
                        }),
                        spatial_factor: (1, 1),
                        temporal_factor: 2,
                    },
                    StreamTransitionKind::Renoise {
                        sigma: LTX_TEMPORAL_SIGMAS[0],
                    },
                ]),
            )?;
            tr.validate_against_bundle(&round_stage.bundle)?;
            transitions.push(tr);
            stages.push(round_stage);
            total_outer_steps += temporal_table.num_steps();
        }

        let memory = compute_memory_summary(&stages, &transitions, &input.footprint)?;

        Ok(Self {
            family: input.family,
            variant: input.variant,
            route: input.route,
            conditioning: input.conditioning,
            sampler: input.sampler,
            width: input.width,
            height: input.height,
            frames: input.frames,
            canvas_frames,
            fps: input.fps,
            seed: input.seed,
            audio_enabled,
            total_outer_steps,
            stages,
            transitions,
            memory,
            cached_footprint: Some(input.footprint),
        })
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::Msg(format!("ltx_plan serialize: {e}")))
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| Error::Msg(format!("ltx_plan deserialize: {e}")))
    }

    /// Reconstruct the complete `MEM-3` [`RequestLiveSetPlan`] for explicit domain inspection.
    pub fn live_set_plan(&self) -> Result<RequestLiveSetPlan> {
        let footprint = self.cached_footprint.clone().unwrap_or_default();
        build_request_live_set(&self.stages, &self.transitions, &footprint)
    }
}

fn temporal_rounds_ref(rounds: &u32) -> &u32 {
    rounds
}

fn stream_activation_bytes(geom: &StreamGeometry) -> u64 {
    match *geom {
        StreamGeometry::VideoGrid {
            latent_frames,
            latent_height,
            latent_width,
            channels,
            ..
        } => bf16_latent_tensor_bytes(
            latent_frames * latent_height * latent_width * usize::from(channels),
        ),
        StreamGeometry::VideoTokens {
            target_tokens,
            conditioning_tokens,
            channels,
            ..
        } => {
            bf16_latent_tensor_bytes((target_tokens + conditioning_tokens) * usize::from(channels))
        }
        StreamGeometry::AudioGrid {
            audio_frames,
            mel_bins,
            channels,
            ..
        } => bf16_latent_tensor_bytes(audio_frames * usize::from(channels) * usize::from(mel_bins)),
        StreamGeometry::Custom { ref shape } => {
            bf16_latent_tensor_bytes(shape.iter().copied().product::<usize>().max(1))
        }
    }
}

fn build_request_live_set(
    stages: &[LtxStagePlanSpec],
    transitions: &[StreamTransition],
    footprint: &LtxMemoryFootprintSpec,
) -> Result<RequestLiveSetPlan> {
    let domain = footprint.domain.to_domain();
    let mut operations = Vec::with_capacity(stages.len());

    for (idx, stage) in stages.iter().enumerate() {
        let mut stream_specs = Vec::with_capacity(stage.bundle.streams.len());
        let mut resident_weights = Vec::new();
        for stream in &stage.bundle.streams {
            let act_bytes = stream_activation_bytes(&stream.geometry);
            let retained = if matches!(stream.id, StreamId::Video)
                && idx == 0
                && footprint.host_staging_bytes > 0
            {
                vec![PricedAllocation::new(
                    format!("{}_host_staging", stage.stage_id),
                    MemoryBudgetDomain::HostRam,
                    footprint.host_staging_bytes,
                )]
            } else {
                Vec::new()
            };
            stream_specs.push(StreamMemorySpec {
                spec: stream.clone(),
                activations: vec![PricedAllocation::new(
                    format!("{}_{}_latent_act", stage.stage_id, stream.id),
                    domain,
                    act_bytes,
                )],
                retained,
            });
            // Both Video and Audio share the single physical AvDiT trunk via `alias_group`,
            // so shared weights are never double-counted (`MEM-3`).
            resident_weights.push(
                PricedAllocation::new(
                    format!("{}_{}_dit_ref", stage.stage_id, stream.id),
                    domain,
                    footprint.av_dit_trunk_bytes,
                )
                .with_alias_group("ltx_av_dit_trunk"),
            );
        }

        let joint_ws = if stage.bundle.contains(&StreamId::Audio)
            && footprint.joint_cross_attention_workspace_bytes > 0
        {
            vec![PricedAllocation::new(
                format!("{}_av_cross_attn_ws", stage.stage_id),
                domain,
                footprint.joint_cross_attention_workspace_bytes,
            )]
        } else {
            Vec::new()
        };

        operations.push(build_multistream_phase_live_set(
            &stage.stage_id,
            MemoryPhase::Denoise,
            stage.pass_index,
            idx,
            &stream_specs,
            resident_weights,
            Vec::new(),
            joint_ws,
            Vec::new(),
        )?);
    }

    let mut live_transitions = Vec::new();
    for tr in transitions {
        let (from_op, to_op, weights_bytes, ws_bytes) =
            if tr.transition_id.starts_with("stage1_to_stage2") {
                let ws = tr.kind.learned_workspace_bytes().unwrap_or(0);
                let upsampler_w = if tr.targets.contains(&StreamId::Video) {
                    footprint.spatial_upsampler_weights_bytes
                } else {
                    0
                };
                (
                    "stage1_base".to_owned(),
                    "stage2_refine".to_owned(),
                    upsampler_w,
                    ws,
                )
            } else {
                let ws = tr.kind.learned_workspace_bytes().unwrap_or(0);
                let round_num = tr
                    .transition_id
                    .strip_prefix("temporal_upsample_round_")
                    .and_then(|r| r.parse::<u32>().ok())
                    .unwrap_or(1);
                let from_stage = if round_num <= 1 {
                    "stage2_refine".to_owned()
                } else {
                    format!("temporal_round_{}", round_num - 1)
                };
                let to_stage = format!("temporal_round_{round_num}");
                (
                    from_stage,
                    to_stage,
                    footprint.temporal_upsampler_weights_bytes,
                    ws,
                )
            };
        let mut resident = vec![PricedAllocation::new(
            format!("{}_resident_dit", tr.transition_id),
            domain,
            footprint.av_dit_trunk_bytes,
        )
        .with_alias_group("ltx_av_dit_trunk")];
        if weights_bytes > 0 {
            resident.push(PricedAllocation::new(
                format!("{}_upsampler_weights", tr.transition_id),
                domain,
                weights_bytes,
            ));
        }
        let workspace = if ws_bytes > 0 {
            vec![PricedAllocation::new(
                format!("{}_upsampler_workspace", tr.transition_id),
                domain,
                ws_bytes,
            )]
        } else {
            Vec::new()
        };
        let handoff = vec![PricedAllocation::new(
            format!("{}_handoff_latent", tr.transition_id),
            domain,
            16 * 1024 * 1024,
        )];
        live_transitions.push(build_stream_transition_live_set(
            tr,
            from_op,
            to_op,
            resident,
            Vec::new(),
            workspace,
            handoff,
        )?);
    }

    Ok(RequestLiveSetPlan {
        operations,
        transitions: live_transitions,
    })
}

fn compute_memory_summary(
    stages: &[LtxStagePlanSpec],
    transitions: &[StreamTransition],
    footprint: &LtxMemoryFootprintSpec,
) -> Result<LtxDomainMemorySummary> {
    let plan = build_request_live_set(stages, transitions, footprint)?;
    let primary = plan.price_domain(footprint.domain.to_domain())?;
    let host = plan.price_domain(MemoryBudgetDomain::HostRam)?;
    Ok(LtxDomainMemorySummary {
        domain: footprint.domain,
        host_ram_peak_bytes: host.request_peak_bytes,
        primary_operation_peak_bytes: primary.max_operation_peak_bytes,
        primary_transition_peak_bytes: primary.max_transition_peak_bytes,
        primary_request_peak_bytes: primary.request_peak_bytes,
        dominant_stage_id: primary.dominant_stage_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::multistream::TransitionBoundaryGuard;
    use gen_core::{BoundaryState, EstimateSource};

    fn sample_input(variant: LtxVariant, route: LtxOrchestrationRoute) -> LtxRenderPlanInput {
        LtxRenderPlanInput {
            family: match variant {
                LtxVariant::Dev => LtxModelFamily::Ltx25,
                LtxVariant::Distilled => LtxModelFamily::Ltx25,
            },
            variant,
            route,
            conditioning: match route {
                LtxOrchestrationRoute::AvDfr => LtxConditioningProfile::DfrGeneratedKeyframes {
                    slot_count: 2,
                    temporal_upsample_rounds: 2,
                },
                LtxOrchestrationRoute::AvTokensIcLora => LtxConditioningProfile::InContextClips {
                    count: 1,
                    clip_tokens: 128,
                },
                _ => LtxConditioningProfile::T2v,
            },
            sampler: None,
            width: 512,
            height: 512,
            frames: 17,
            fps: 24.0,
            seed: 42,
            explicit_steps: None,
            has_temporal_upsampler: true,
            footprint: LtxMemoryFootprintSpec::default(),
        }
    }

    #[test]
    fn plan_builds_serializes_and_inspects_without_weights_for_distilled_dev_iclora_and_dfr() {
        let distilled = LtxRenderPlan::build(sample_input(
            LtxVariant::Distilled,
            LtxOrchestrationRoute::AvGrid,
        ))
        .unwrap();
        assert_eq!(distilled.total_outer_steps, 11);
        assert_eq!(distilled.stages.len(), 2);
        assert_eq!(
            distilled.stages[0].schedules[0].schedule.sigmas(),
            &LTX_STAGE1_SIGMAS
        );
        assert_eq!(
            distilled.stages[1].schedules[0].schedule.sigmas(),
            &LTX_STAGE2_SIGMAS
        );
        assert_eq!(distilled.transitions[0].targets, vec![StreamId::Video]);
        assert_eq!(distilled.transitions[1].targets, vec![StreamId::Audio]);
        let round_trip = LtxRenderPlan::from_json(&distilled.to_json().unwrap()).unwrap();
        assert_eq!(round_trip.total_outer_steps, 11);

        let dev =
            LtxRenderPlan::build(sample_input(LtxVariant::Dev, LtxOrchestrationRoute::AvGrid))
                .unwrap();
        assert_eq!(dev.total_outer_steps, 33);
        assert!(dev.stages[0].four_pass_guidance);
        assert!(!dev.stages[1].four_pass_guidance);
        assert_eq!(dev.stages[0].schedules[0].schedule.num_steps(), 30);
        assert_eq!(
            dev.stages[1].schedules[0].schedule.sigmas(),
            &LTX_STAGE2_SIGMAS
        );

        let dfr = LtxRenderPlan::build(sample_input(LtxVariant::Dev, LtxOrchestrationRoute::AvDfr))
            .unwrap();
        assert_eq!(dfr.stages.len(), 4); // stage1 (30) + stage2 (3) + round1 (4) + round2 (4)
        assert_eq!(dfr.total_outer_steps, 41);
        assert_eq!(
            dfr.stages[2].schedules[0].schedule.sigmas(),
            &LTX_TEMPORAL_SIGMAS
        );
        assert_eq!(dfr.stages[2].rf_ancestral_eta, Some(TEMPORAL_ANCESTRAL_ETA));
        assert!(!dfr.stages[2].bundle.contains(&StreamId::Audio));
    }

    #[test]
    fn plan_refuses_invalid_combinations_before_weight_load() {
        // 1. Curated sampler on Dev.
        let mut dev_curated = sample_input(LtxVariant::Dev, LtxOrchestrationRoute::AvGrid);
        dev_curated.sampler = Some("dpmpp_2m".into());
        assert!(matches!(
            LtxRenderPlan::build(dev_curated),
            Err(Error::Unsupported(_))
        ));

        // 2. Non-native steps on Distilled.
        let mut bad_steps = sample_input(LtxVariant::Distilled, LtxOrchestrationRoute::AvGrid);
        bad_steps.explicit_steps = Some(20);
        assert!(matches!(
            LtxRenderPlan::build(bad_steps),
            Err(Error::Unsupported(_))
        ));

        // 3. LTX-2.3 DFR refusal.
        let mut ltx23_dfr = sample_input(LtxVariant::Distilled, LtxOrchestrationRoute::AvDfr);
        ltx23_dfr.family = LtxModelFamily::Ltx23;
        assert!(matches!(
            LtxRenderPlan::build(ltx23_dfr),
            Err(Error::Unsupported(_))
        ));

        // 4. Temporal rounds without temporal upsampler.
        let mut missing_temporal =
            sample_input(LtxVariant::Distilled, LtxOrchestrationRoute::AvDfr);
        missing_temporal.has_temporal_upsampler = false;
        assert!(matches!(
            LtxRenderPlan::build(missing_temporal),
            Err(Error::Unsupported(_))
        ));

        // 5. Curated non-euler sampler on I2V keyframes.
        let mut curated_i2v = LtxRenderPlanInput {
            conditioning: LtxConditioningProfile::ReplaceLatentKeyframes { count: 1 },
            sampler: Some("heun".into()),
            ..sample_input(LtxVariant::Distilled, LtxOrchestrationRoute::AvGrid)
        };
        assert!(matches!(
            LtxRenderPlan::build(curated_i2v.clone()),
            Err(Error::Unsupported(_))
        ));
        curated_i2v.sampler = Some("euler".into());
        let i2v_plan = LtxRenderPlan::build(curated_i2v).unwrap();
        // Per-token time applies to Video while Audio stays strictly Uniform!
        assert_eq!(
            i2v_plan.stages[0]
                .bundle
                .stream(&StreamId::Video)
                .unwrap()
                .token_time_policy,
            PerTokenTimePolicy::PerTokenMask
        );
        assert_eq!(
            i2v_plan.stages[0]
                .bundle
                .stream(&StreamId::Audio)
                .unwrap()
                .token_time_policy,
            PerTokenTimePolicy::Uniform
        );
    }

    #[test]
    fn mem3_sums_concurrent_streams_and_prices_upsampler_transition_peak() {
        let av_plan = LtxRenderPlan::build(sample_input(
            LtxVariant::Distilled,
            LtxOrchestrationRoute::AvGrid,
        ))
        .unwrap();
        let video_only_plan = LtxRenderPlan::build(sample_input(
            LtxVariant::Distilled,
            LtxOrchestrationRoute::VideoOnlyGrid,
        ))
        .unwrap();
        // Adding the concurrent Audio stream + joint workspace MUST strictly increase operation peak
        // (summed, never maxed, while shared AvDiT trunk is deduplicated!).
        assert!(
            av_plan.memory.primary_operation_peak_bytes
                > video_only_plan.memory.primary_operation_peak_bytes
        );
        // HostRam staging stays strictly separate from UnifiedMemory.
        assert_eq!(
            av_plan.memory.host_ram_peak_bytes,
            LtxMemoryFootprintSpec::default().host_staging_bytes
        );
        // Spatial upsampler transition includes component weights + workspace above latent operations.
        assert!(
            av_plan.memory.primary_transition_peak_bytes
                > av_plan.memory.primary_operation_peak_bytes
        );

        // Transition double-noise guard: entering video transition with XSigma fails closed.
        let guard =
            TransitionBoundaryGuard::new(StreamId::Video, BoundaryState::XSigma(vec![0.0_f32]));
        let err = guard.apply(&av_plan.transitions[0], Ok).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));

        // CleanEstimate succeeds into XSigma.
        let guard_ok = TransitionBoundaryGuard::new(
            StreamId::Video,
            BoundaryState::CleanEstimate {
                latent: vec![0.0_f32],
                source: EstimateSource::EndpointEvaluation,
            },
        );
        let out = guard_ok
            .apply(&av_plan.transitions[0], |state| match state {
                BoundaryState::CleanEstimate { latent, .. } => Ok(BoundaryState::XSigma(latent)),
                _ => unreachable!(),
            })
            .unwrap();
        assert!(matches!(out, BoundaryState::XSigma(_)));
    }
}
