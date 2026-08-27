//! Candle LTX LoRA/QLoRA training (sc-13867, sc-18779).
//!
//! The recipe matches `mlx-gen-ltx`: single-frame VAE latents and Gemma video features are cached
//! once, the encoder stack is dropped before the DiT loads, and the trainable video-only DiT regresses
//! raw velocity against `noise - clean` at a seeded uniform sigma. The legacy LTX-2.3 route is
//! retained unchanged; the LTX-2.5 route loads the split Gemma-4 bundle and its config-derived
//! QLoRA base. Adapter factors, loss reductions, and optimizer state are f32.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{
    self, safetensors_file_metadata, safetensors_path_tensor_headers, CancelFlag, Image, LoadSpec,
    Modality, Progress, SafetensorsTensorHeader, WeightsSource,
};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, run_flow_match_training, velocity_loss, FlowMatchTrainer, SamplePlan,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::LoraSet;
use candle_gen::train::merge::read_adapter;
use candle_gen::{CandleError, Result};

use crate::config::{
    compute_audio_frames, AvConfig, ConnectorConfig, GemmaConfig, DEFAULT_FPS, LATENT_CHANNELS,
    SPATIAL_SCALE, STAGE1_SIGMAS, TRAINER_ID,
};
use crate::dit_train::{LtxDiT, LTX_ATTN_TARGETS};
use crate::gemma4_te::Ltx25TextEncoder;
use crate::params::GuiderParams;
use crate::pipeline::{flatten_audio_latent, flatten_latent, frames_to_images, unflatten_latent};
use crate::rope::{create_audio_position_grid, create_position_grid};
use crate::text_encoder::LtxTextEncoder;
use crate::tier::TierPaths;
use crate::tier::{Ltx25Component, Ltx25Tier};
use crate::tokenizer::Ltx25Tokenizer;
use crate::transformer::AvDiT;
use crate::vae::LtxVideoVae;
use crate::MODEL_25_ID;
use candle_gen::gen_core::ltx_checkpoint::{LtxBundle, LtxCheckpointLayout, LtxComponent};
use candle_gen::train::lora::{LoraHost, LoraLinear};

const LABEL: &str = "ltx trainer";
const SAMPLE_PROMPT_CAP: usize = 4;
const TRAIN_TEXT_MAX_LENGTH: usize = 128;
const SAFE_MEMORY_FRACTION: f64 = 0.85;
const LTX25_VIDEO_LATENT_CHANNELS: usize = 128;
const LTX25_AUDIO_FLAT_CHANNELS: usize = 128;

/// The real trainable AV target surface. Attention suffixes reach video/audio self + text
/// attention and both cross-modal directions; FF suffixes reach both configured stream FFNs.
pub const LTX_AV_LORA_TARGETS: &[&str] = &[
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
];

/// The upstream 2.5 validation preset.  The generic training request deliberately has no
/// video-shaped preview fields, so providers expose this exact typed contract instead of silently
/// treating its image-preview defaults as LTX-2.5 video validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25ValidationDefaults {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub video_cfg_scale: f32,
    pub audio_cfg_scale: f32,
    pub video_stg_scale: f32,
    pub audio_stg_scale: f32,
    pub stg_blocks: &'static [usize],
    pub guidance_rescale: f32,
    pub video_modality_guidance_scale: f32,
    pub audio_modality_guidance_scale: f32,
    pub generate_video: bool,
    pub generate_audio: bool,
}

pub const LTX25_VALIDATION_DEFAULTS: Ltx25ValidationDefaults = Ltx25ValidationDefaults {
    width: 960,
    height: 544,
    frames: 89,
    fps: 24,
    steps: 30,
    video_cfg_scale: 3.0,
    audio_cfg_scale: 7.0,
    video_stg_scale: 1.0,
    audio_stg_scale: 1.0,
    stg_blocks: &[28],
    guidance_rescale: 0.7,
    video_modality_guidance_scale: 3.0,
    audio_modality_guidance_scale: 3.0,
    generate_video: true,
    generate_audio: true,
};

/// Request-owned validation settings.  Defaults are intentionally typed separately from the
/// static preset because worker/API callers may omit fields or provide bounded overrides.
#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25ValidationConfig {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub video_cfg_scale: f32,
    pub audio_cfg_scale: f32,
    pub video_stg_scale: f32,
    pub audio_stg_scale: f32,
    pub stg_blocks: Vec<usize>,
    pub guidance_rescale: f32,
    pub video_modality_guidance_scale: f32,
    pub audio_modality_guidance_scale: f32,
    pub generate_video: bool,
    pub generate_audio: bool,
}

impl From<Ltx25ValidationDefaults> for Ltx25ValidationConfig {
    fn from(value: Ltx25ValidationDefaults) -> Self {
        Self {
            width: value.width,
            height: value.height,
            frames: value.frames,
            fps: value.fps,
            steps: value.steps,
            video_cfg_scale: value.video_cfg_scale,
            audio_cfg_scale: value.audio_cfg_scale,
            video_stg_scale: value.video_stg_scale,
            audio_stg_scale: value.audio_stg_scale,
            stg_blocks: value.stg_blocks.to_vec(),
            guidance_rescale: value.guidance_rescale,
            video_modality_guidance_scale: value.video_modality_guidance_scale,
            audio_modality_guidance_scale: value.audio_modality_guidance_scale,
            generate_video: value.generate_video,
            generate_audio: value.generate_audio,
        }
    }
}

/// Exact validated preview route.  Keeping this concrete makes the 2.5 preview use its Dev
/// schedule and branch controls instead of falling through to the distilled image sampler.
#[derive(Clone, Debug, PartialEq)]
struct Ltx25ValidationRenderPlan {
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    sigmas: Vec<f32>,
    stg_blocks: Vec<usize>,
    video_cfg_scale: f32,
    audio_cfg_scale: f32,
    video_stg_scale: f32,
    audio_stg_scale: f32,
    guidance_rescale: f32,
    video_modality_guidance_scale: f32,
    audio_modality_guidance_scale: f32,
    generate_audio: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx25Workflow {
    I2vLora,
    T2vLora,
    V2aLora,
    A2vLora,
    T2aLora,
    VideoExtendLora,
    VideoInpaintingLora,
    VideoOutpaintingLora,
    VideoSuffixLora,
    AudioExtendLora,
    AudioInpaintingLora,
    AudioSuffixLora,
    Av2avIcLora,
    V2vIcLora,
    A2aIcLora,
}

impl Ltx25Workflow {
    pub const fn id(self) -> &'static str {
        match self {
            Self::I2vLora => "i2v_lora",
            Self::T2vLora => "t2v_lora",
            Self::V2aLora => "v2a_lora",
            Self::A2vLora => "a2v_lora",
            Self::T2aLora => "t2a_lora",
            Self::VideoExtendLora => "video_extend_lora",
            Self::VideoInpaintingLora => "video_inpainting_lora",
            Self::VideoOutpaintingLora => "video_outpainting_lora",
            Self::VideoSuffixLora => "video_suffix_lora",
            Self::AudioExtendLora => "audio_extend_lora",
            Self::AudioInpaintingLora => "audio_inpainting_lora",
            Self::AudioSuffixLora => "audio_suffix_lora",
            Self::Av2avIcLora => "av2av_ic_lora",
            Self::V2vIcLora => "v2v_ic_lora",
            Self::A2aIcLora => "a2a_ic_lora",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "i2v_lora" => Ok(Self::I2vLora),
            "t2v_lora" => Ok(Self::T2vLora),
            "v2a_lora" => Ok(Self::V2aLora),
            "a2v_lora" => Ok(Self::A2vLora),
            "t2a_lora" => Ok(Self::T2aLora),
            "video_extend_lora" => Ok(Self::VideoExtendLora),
            "video_inpainting_lora" => Ok(Self::VideoInpaintingLora),
            "video_outpainting_lora" => Ok(Self::VideoOutpaintingLora),
            "video_suffix_lora" => Ok(Self::VideoSuffixLora),
            "audio_extend_lora" => Ok(Self::AudioExtendLora),
            "audio_inpainting_lora" => Ok(Self::AudioInpaintingLora),
            "audio_suffix_lora" => Ok(Self::AudioSuffixLora),
            "av2av_ic_lora" => Ok(Self::Av2avIcLora),
            "v2v_ic_lora" => Ok(Self::V2vIcLora),
            "a2a_ic_lora" => Ok(Self::A2aIcLora),
            other => Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: unsupported upstream workflow `{other}`"
            ))),
        }
    }

    fn canonical(self) -> (WorkflowModalityRule, WorkflowModalityRule) {
        match self {
            Self::I2vLora => (generated_rule(&FIRST_FRAME_50), GENERATED),
            Self::T2vLora => (GENERATED, GENERATED),
            Self::V2aLora => (FROZEN, GENERATED),
            Self::A2vLora => (GENERATED, FROZEN),
            Self::T2aLora => (ABSENT, GENERATED),
            Self::VideoExtendLora => (generated_rule(&PREFIX_8), GENERATED),
            Self::VideoInpaintingLora => (generated_rule(&MASK), ABSENT),
            Self::VideoOutpaintingLora => (generated_rule(&SPATIAL_CROP), ABSENT),
            Self::VideoSuffixLora => (generated_rule(&SUFFIX_8), GENERATED),
            Self::AudioExtendLora => (ABSENT, generated_rule(&PREFIX_8)),
            Self::AudioInpaintingLora => (ABSENT, generated_rule(&MASK)),
            Self::AudioSuffixLora => (ABSENT, generated_rule(&SUFFIX_8)),
            Self::Av2avIcLora => (generated_rule(&REFERENCE), generated_rule(&REFERENCE)),
            Self::V2vIcLora => (generated_rule(&REFERENCE_FIRST_FRAME_20), ABSENT),
            Self::A2aIcLora => (ABSENT, generated_rule(&REFERENCE)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CanonicalCondition {
    kind: IntrinsicCondition,
    probability: f32,
    temporal_boundary: Option<u32>,
    spatial_region: Option<[u32; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WorkflowModalityRule {
    is_generated: bool,
    present: bool,
    conditions: &'static [CanonicalCondition],
}

const GENERATED: WorkflowModalityRule = WorkflowModalityRule {
    is_generated: true,
    present: true,
    conditions: &[],
};
const FROZEN: WorkflowModalityRule = WorkflowModalityRule {
    is_generated: false,
    present: true,
    conditions: &[],
};
const ABSENT: WorkflowModalityRule = WorkflowModalityRule {
    is_generated: false,
    present: false,
    conditions: &[],
};
const fn generated_rule(conditions: &'static [CanonicalCondition]) -> WorkflowModalityRule {
    WorkflowModalityRule {
        is_generated: true,
        present: true,
        conditions,
    }
}
const FIRST_FRAME_50: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::First,
    probability: 0.5,
    temporal_boundary: None,
    spatial_region: None,
}];
const PREFIX_8: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::Prefix,
    probability: 1.0,
    temporal_boundary: Some(8),
    spatial_region: None,
}];
const SUFFIX_8: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::Suffix,
    probability: 1.0,
    temporal_boundary: Some(8),
    spatial_region: None,
}];
const MASK: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::Mask,
    probability: 1.0,
    temporal_boundary: None,
    spatial_region: None,
}];
const SPATIAL_CROP: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::SpatialCrop,
    probability: 1.0,
    temporal_boundary: None,
    spatial_region: Some([0, 0, 288, 576]),
}];
const REFERENCE: [CanonicalCondition; 1] = [CanonicalCondition {
    kind: IntrinsicCondition::None,
    probability: 1.0,
    temporal_boundary: None,
    spatial_region: None,
}];
const REFERENCE_FIRST_FRAME_20: [CanonicalCondition; 2] = [
    CanonicalCondition {
        kind: IntrinsicCondition::None,
        probability: 1.0,
        temporal_boundary: None,
        spatial_region: None,
    },
    CanonicalCondition {
        kind: IntrinsicCondition::First,
        probability: 0.2,
        temporal_boundary: None,
        spatial_region: None,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntrinsicCondition {
    None,
    First,
    Prefix,
    Suffix,
    Mask,
    SpatialCrop,
}

impl IntrinsicCondition {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "firstFrame" => Ok(Self::First),
            "prefix" => Ok(Self::Prefix),
            "suffix" => Ok(Self::Suffix),
            "mask" => Ok(Self::Mask),
            "spatialCrop" => Ok(Self::SpatialCrop),
            "reference" => Ok(Self::None),
            other => Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: unsupported condition type `{other}`"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
struct Ltx25Condition {
    kind: IntrinsicCondition,
    probability: f32,
    temporal_boundary: Option<u32>,
    spatial_region: Option<[u32; 4]>,
    tensor_key: Option<String>,
    spatial_scale_factor: Option<u32>,
    temporal_scale_factor: Option<u32>,
    reference: bool,
}

#[derive(Clone, Debug)]
struct ModalityConditioningPlan {
    present: bool,
    is_generated: bool,
    conditions: Vec<Ltx25Condition>,
}

#[derive(Clone, Debug)]
struct Ltx25ConditioningPlan {
    workflow: Ltx25Workflow,
    video: ModalityConditioningPlan,
    audio: ModalityConditioningPlan,
    validation: Ltx25ValidationConfig,
}

fn condition_from_value(value: &serde_json::Value) -> Result<Ltx25Condition> {
    let object = value
        .as_object()
        .ok_or_else(|| CandleError::Msg("ltx_2_5 trainer: condition must be an object".into()))?;
    let kind_name = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CandleError::Msg("ltx_2_5 trainer: condition.type is required".into()))?;
    let probability = object
        .get("probability")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            CandleError::Msg("ltx_2_5 trainer: condition.probability is required".into())
        })? as f32;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: condition probability {probability} must be in [0, 1]"
        )));
    }
    let spatial_region =
        object
            .get("spatialRegion")
            .map_or(Ok::<Option<[u32; 4]>, CandleError>(None), |value| {
                let values = value.as_array().ok_or_else(|| {
                    CandleError::Msg(
                        "ltx_2_5 trainer: spatialRegion must be a four-element array".into(),
                    )
                })?;
                let values: Vec<u32> = values
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .ok_or_else(|| {
                                CandleError::Msg(
                                    "ltx_2_5 trainer: spatialRegion entries must be u32".into(),
                                )
                            })
                    })
                    .collect::<Result<_>>()?;
                let region: [u32; 4] = values.try_into().map_err(|_| {
                    CandleError::Msg("ltx_2_5 trainer: spatialRegion must have four entries".into())
                })?;
                Ok(Some(region))
            })?;
    let key = |name: &str| -> Result<Option<u32>> {
        object.get(name).map_or(Ok(None), |value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .map(Some)
                .ok_or_else(|| CandleError::Msg(format!("ltx_2_5 trainer: {name} must be u32")))
        })
    };
    let kind = IntrinsicCondition::parse(kind_name)?;
    let reference = matches!(kind, IntrinsicCondition::None);
    let tensor_key = object
        .get("tensorKey")
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    CandleError::Msg(
                        "ltx_2_5 trainer: condition.tensorKey must be a non-empty string".into(),
                    )
                })
        })
        .transpose()?;
    if (reference || matches!(kind, IntrinsicCondition::Mask)) && tensor_key.is_none() {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {kind_name} condition requires tensorKey"
        )));
    }
    if !reference && !matches!(kind, IntrinsicCondition::Mask) && tensor_key.is_some() {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {kind_name} conditions operate on the main clean modality and must not declare tensorKey"
        )));
    }
    let temporal_boundary = key("temporalBoundary")?;
    let spatial_scale_factor = key("spatialScaleFactor")?;
    let temporal_scale_factor = key("temporalScaleFactor")?;
    if spatial_scale_factor == Some(0) || temporal_scale_factor == Some(0) {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: spatialScaleFactor and temporalScaleFactor must be > 0".into(),
        ));
    }
    let reject = |field: &str| -> Result<()> {
        Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {field} is not valid for condition type `{kind_name}`"
        )))
    };
    match kind {
        IntrinsicCondition::First => {
            if temporal_boundary.is_some() {
                reject("temporalBoundary")?;
            }
            if spatial_region.is_some() {
                reject("spatialRegion")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        IntrinsicCondition::Prefix | IntrinsicCondition::Suffix => {
            if temporal_boundary.is_none() || temporal_boundary == Some(0) {
                return Err(CandleError::Msg(
                    "ltx_2_5 trainer: prefix/suffix conditions require temporalBoundary > 0".into(),
                ));
            }
            if spatial_region.is_some() {
                reject("spatialRegion")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        IntrinsicCondition::SpatialCrop => {
            let [top, left, bottom, right] = spatial_region.ok_or_else(|| {
                CandleError::Msg("ltx_2_5 trainer: spatialCrop requires spatialRegion".into())
            })?;
            if bottom <= top || right <= left {
                return Err(CandleError::Msg(
                    "ltx_2_5 trainer: spatialRegion must have positive area".into(),
                ));
            }
            if temporal_boundary.is_some() {
                reject("temporalBoundary")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        IntrinsicCondition::Mask => {
            if temporal_boundary.is_some() || spatial_region.is_some() {
                reject("temporalBoundary/spatialRegion")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        IntrinsicCondition::None => {
            if temporal_boundary.is_some() || spatial_region.is_some() {
                reject("temporalBoundary/spatialRegion")?;
            }
        }
    }
    if matches!(
        kind,
        IntrinsicCondition::Prefix | IntrinsicCondition::Suffix
    ) && temporal_boundary == Some(0)
    {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: temporalBoundary must be > 0".into(),
        ));
    }
    Ok(Ltx25Condition {
        kind,
        probability,
        temporal_boundary,
        spatial_region,
        tensor_key,
        spatial_scale_factor,
        temporal_scale_factor,
        reference,
    })
}

fn modality_from_value(
    value: Option<&serde_json::Value>,
    name: &str,
    canonical: WorkflowModalityRule,
    is_video: bool,
) -> Result<ModalityConditioningPlan> {
    let canonical_conditions = || {
        canonical
            .conditions
            .iter()
            .map(|condition| Ltx25Condition {
                kind: condition.kind,
                probability: condition.probability,
                temporal_boundary: condition.temporal_boundary,
                spatial_region: condition.spatial_region,
                tensor_key: None,
                spatial_scale_factor: None,
                temporal_scale_factor: None,
                reference: matches!(condition.kind, IntrinsicCondition::None),
            })
            .collect::<Vec<_>>()
    };
    if !canonical.present {
        if value.is_some() {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: selected workflow does not carry {name}, but it was provided"
            )));
        }
        return Ok(ModalityConditioningPlan {
            present: false,
            is_generated: false,
            conditions: Vec::new(),
        });
    }
    let (is_generated, mut conditions) = match value {
        None => (canonical.is_generated, canonical_conditions()),
        Some(value) => {
            let object = value.as_object().ok_or_else(|| {
                CandleError::Msg(format!("ltx_2_5 trainer: {name} must be an object"))
            })?;
            let is_generated = object
                .get("isGenerated")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    CandleError::Msg(format!("ltx_2_5 trainer: {name}.isGenerated is required"))
                })?;
            let conditions = object
                .get("conditions")
                .map(|value| {
                    value
                        .as_array()
                        .ok_or_else(|| {
                            CandleError::Msg(format!(
                                "ltx_2_5 trainer: {name}.conditions must be an array"
                            ))
                        })?
                        .iter()
                        .map(condition_from_value)
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_else(canonical_conditions);
            (is_generated, conditions)
        }
    };
    if is_generated != canonical.is_generated {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {name}.isGenerated={is_generated} contradicts the selected workflow"
        )));
    }
    if conditions.len() != canonical.conditions.len() {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {name} must contain the workflow's exact intrinsic condition set"
        )));
    }
    for expected in canonical.conditions {
        let Some(actual) = conditions
            .iter_mut()
            .find(|actual| actual.kind == expected.kind)
        else {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: {name} omits a workflow-required {:?} condition",
                expected.kind
            )));
        };
        if actual.probability != expected.probability
            || actual.temporal_boundary != expected.temporal_boundary
            || actual.spatial_region != expected.spatial_region
        {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: {name} condition contradicts the selected workflow"
            )));
        }
        if matches!(
            actual.kind,
            IntrinsicCondition::Mask | IntrinsicCondition::None
        ) && actual.tensor_key.is_none()
        {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: {name} {:?} condition requires tensorKey",
                actual.kind
            )));
        }
        if actual.reference {
            let (spatial, temporal) = if is_video {
                (SPATIAL_SCALE as u32, 8)
            } else {
                (1, 4)
            };
            if actual
                .spatial_scale_factor
                .is_some_and(|scale| scale != spatial)
                || actual
                    .temporal_scale_factor
                    .is_some_and(|scale| scale != temporal)
            {
                return Err(CandleError::Msg(format!(
                    "ltx_2_5 trainer: {name} reference scales must be spatial={spatial}, temporal={temporal}"
                )));
            }
            actual.spatial_scale_factor = Some(spatial);
            actual.temporal_scale_factor = Some(temporal);
        }
    }
    if conditions.iter().any(|actual| {
        !canonical
            .conditions
            .iter()
            .any(|expected| expected.kind == actual.kind)
    }) {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: {name} conditions contradict the selected workflow"
        )));
    }
    Ok(ModalityConditioningPlan {
        present: true,
        is_generated,
        conditions,
    })
}

fn validation_from_value(value: Option<&serde_json::Value>) -> Result<Ltx25ValidationConfig> {
    let Some(value) = value else {
        return Ok(LTX25_VALIDATION_DEFAULTS.into());
    };
    let object = value.as_object().ok_or_else(|| {
        CandleError::Msg("ltx_2_5 trainer: ltxValidation must be an object".into())
    })?;
    let number = |name: &str, default: u32| -> Result<u32> {
        object.get(name).map_or(Ok(default), |value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    CandleError::Msg(format!("ltx_2_5 trainer: ltxValidation.{name} must be u32"))
                })
        })
    };
    let float = |name: &str, default: f32| -> Result<f32> {
        let value = object.get(name).map_or(Ok(default), |value| {
            value.as_f64().map(|value| value as f32).ok_or_else(|| {
                CandleError::Msg(format!(
                    "ltx_2_5 trainer: ltxValidation.{name} must be numeric"
                ))
            })
        })?;
        if !value.is_finite() {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: ltxValidation.{name} must be finite"
            )));
        }
        Ok(value)
    };
    let blocks = match object.get("stgBlocks") {
        None => LTX25_VALIDATION_DEFAULTS.stg_blocks.to_vec(),
        Some(value) => value
            .as_array()
            .ok_or_else(|| {
                CandleError::Msg("ltx_2_5 trainer: ltxValidation.stgBlocks must be an array".into())
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        CandleError::Msg(
                            "ltx_2_5 trainer: ltxValidation.stgBlocks entries must be usize".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let generate_audio = match object.get("generateAudio") {
        None => LTX25_VALIDATION_DEFAULTS.generate_audio,
        Some(value) => value.as_bool().ok_or_else(|| {
            CandleError::Msg("ltx_2_5 trainer: ltxValidation.generateAudio must be boolean".into())
        })?,
    };
    let parsed = Ltx25ValidationConfig {
        width: number("width", LTX25_VALIDATION_DEFAULTS.width)?,
        height: number("height", LTX25_VALIDATION_DEFAULTS.height)?,
        frames: number("frames", LTX25_VALIDATION_DEFAULTS.frames)?,
        fps: number("fps", LTX25_VALIDATION_DEFAULTS.fps)?,
        steps: number("steps", LTX25_VALIDATION_DEFAULTS.steps)?,
        video_cfg_scale: float("videoCfgScale", LTX25_VALIDATION_DEFAULTS.video_cfg_scale)?,
        audio_cfg_scale: float("audioCfgScale", LTX25_VALIDATION_DEFAULTS.audio_cfg_scale)?,
        video_stg_scale: float("videoStgScale", LTX25_VALIDATION_DEFAULTS.video_stg_scale)?,
        audio_stg_scale: float("audioStgScale", LTX25_VALIDATION_DEFAULTS.audio_stg_scale)?,
        stg_blocks: blocks,
        guidance_rescale: float(
            "guidanceRescale",
            LTX25_VALIDATION_DEFAULTS.guidance_rescale,
        )?,
        video_modality_guidance_scale: float(
            "videoModalityGuidanceScale",
            LTX25_VALIDATION_DEFAULTS.video_modality_guidance_scale,
        )?,
        audio_modality_guidance_scale: float(
            "audioModalityGuidanceScale",
            LTX25_VALIDATION_DEFAULTS.audio_modality_guidance_scale,
        )?,
        generate_video: true,
        generate_audio,
    };
    build_ltx25_validation_render_plan(&parsed)?;
    Ok(parsed)
}

fn build_ltx25_validation_render_plan(
    config: &Ltx25ValidationConfig,
) -> Result<Ltx25ValidationRenderPlan> {
    if !config.generate_video {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: validation must generate video".into(),
        ));
    }
    if !(32..=4096).contains(&config.width)
        || !(32..=4096).contains(&config.height)
        || !config.width.is_multiple_of(32)
        || !config.height.is_multiple_of(32)
    {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: validation width/height must be 32..=4096 and 32px-aligned".into(),
        ));
    }
    if config.frames == 0 || config.frames > 257 || config.frames % 8 != 1 {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: validation frames must be 1..=257 with frames % 8 == 1".into(),
        ));
    }
    if !(1..=120).contains(&config.fps) || !(1..=100).contains(&config.steps) {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: validation fps must be 1..=120 and steps must be 1..=100".into(),
        ));
    }
    if config.stg_blocks.is_empty()
        || config.stg_blocks.len() > 8
        || config.stg_blocks.iter().any(|&block| block >= 48)
    {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: stgBlocks must contain 1..=8 unique indices in 0..48".into(),
        ));
    }
    let mut unique = config.stg_blocks.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != config.stg_blocks.len() {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: stgBlocks must not contain duplicates".into(),
        ));
    }
    for (name, value, max) in [
        ("videoCfgScale", config.video_cfg_scale, 20.0),
        ("audioCfgScale", config.audio_cfg_scale, 20.0),
        ("videoStgScale", config.video_stg_scale, 20.0),
        ("audioStgScale", config.audio_stg_scale, 20.0),
        (
            "videoModalityGuidanceScale",
            config.video_modality_guidance_scale,
            20.0,
        ),
        (
            "audioModalityGuidanceScale",
            config.audio_modality_guidance_scale,
            20.0,
        ),
        ("guidanceRescale", config.guidance_rescale, 1.0),
    ] {
        if !value.is_finite() || !(0.0..=max).contains(&value) {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: {name} must be finite and in [0,{max}]"
            )));
        }
    }
    Ok(Ltx25ValidationRenderPlan {
        width: config.width,
        height: config.height,
        frames: config.frames,
        fps: config.fps,
        sigmas: (0..=config.steps)
            .map(|index| 1.0 - index as f32 / config.steps as f32)
            .collect(),
        stg_blocks: config.stg_blocks.clone(),
        video_cfg_scale: config.video_cfg_scale,
        audio_cfg_scale: config.audio_cfg_scale,
        video_stg_scale: config.video_stg_scale,
        audio_stg_scale: config.audio_stg_scale,
        guidance_rescale: config.guidance_rescale,
        video_modality_guidance_scale: config.video_modality_guidance_scale,
        audio_modality_guidance_scale: config.audio_modality_guidance_scale,
        generate_audio: config.generate_audio,
    })
}

impl Ltx25ConditioningPlan {
    fn from_request(req: &TrainingRequest) -> Result<Self> {
        let options = &req.config.model_options;
        let workflow = Ltx25Workflow::parse(
            options
                .get("ltxWorkflow")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    CandleError::Msg("ltx_2_5 trainer: ltxWorkflow is required".into())
                })?,
        )?;
        let (video_rule, audio_rule) = workflow.canonical();
        let video = modality_from_value(options.get("ltxVideo"), "ltxVideo", video_rule, true)?;
        let audio = modality_from_value(options.get("ltxAudio"), "ltxAudio", audio_rule, false)?;
        if !video.is_generated && !audio.is_generated {
            return Err(CandleError::Msg(
                "ltx_2_5 trainer: at least one modality must be generated".into(),
            ));
        }
        Ok(Self {
            workflow,
            video,
            audio,
            validation: validation_from_value(options.get("ltxValidation"))?,
        })
    }
}

/// Build executable generated/conditioning/loss masks for one flattened modality.  The policy is
/// evaluated per example with a deterministic Bernoulli draw, matching upstream's flexible
/// strategy while keeping it independent of the training-noise RNG.
fn modality_masks(
    plan: &ModalityConditioningPlan,
    condition_tensors: &HashMap<String, Tensor>,
    geometry: LtxTokenGeometry,
    seed: u64,
    sample_index: u64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let tokens = geometry.tokens();
    let mut generated = vec![f32::from(plan.is_generated); tokens];
    let frame_tokens = geometry.height * geometry.width;
    for (ordinal, condition) in plan.conditions.iter().enumerate() {
        if condition_draw(seed, sample_index, ordinal) >= condition.probability {
            continue;
        }
        match condition.kind {
            IntrinsicCondition::None => {
                // Reference tokens are appended by `prepare_av_modality`; the target stream
                // remains generated and its loss mask does not spuriously hide a prefix.
                if condition.reference {
                    let key = condition
                        .tensor_key
                        .as_deref()
                        .expect("parser requires reference tensorKey");
                    if !condition_tensors.contains_key(key) {
                        return Err(CandleError::Msg(format!(
                            "ltx_2_5 trainer: prepared bundle is missing reference tensor `{key}`"
                        )));
                    }
                }
            }
            IntrinsicCondition::First => generated[..frame_tokens].fill(0.0),
            IntrinsicCondition::Prefix => {
                let frames = condition.temporal_boundary.unwrap_or(1) as usize;
                generated[..frames.min(geometry.frames) * frame_tokens].fill(0.0);
            }
            IntrinsicCondition::Suffix => {
                let frames = condition.temporal_boundary.unwrap_or(1) as usize;
                let start =
                    geometry.frames.saturating_sub(frames.min(geometry.frames)) * frame_tokens;
                generated[start..].fill(0.0);
            }
            IntrinsicCondition::Mask => {
                // The prepared bundle supplies the real per-token mask.  The coarse fallback that
                // used to alternate tokens made the named workflow surface decorative, so a mask
                // condition is now only valid when its exact tensor is present and shape-aligned.
                let key = condition
                    .tensor_key
                    .as_deref()
                    .expect("parser requires mask tensorKey");
                let mask = condition_tensors.get(key).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "ltx_2_5 trainer: prepared bundle is missing mask tensor `{key}`"
                    ))
                })?;
                if mask.elem_count() != tokens {
                    return Err(CandleError::Msg(format!(
                        "ltx_2_5 trainer: mask tensor `{key}` must contain {tokens} tokens, got {:?}", mask.dims()
                    )));
                }
                let values = mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (output, value) in generated.iter_mut().zip(values) {
                    if !value.is_finite() {
                        return Err(CandleError::Msg(format!(
                            "ltx_2_5 trainer: mask tensor `{key}` must contain finite values"
                        )));
                    }
                    if value > 0.5 {
                        *output = 0.0;
                    }
                }
            }
            IntrinsicCondition::SpatialCrop => {
                let [top, left, bottom, right] = condition.spatial_region.ok_or_else(|| {
                    CandleError::Msg("ltx_2_5 trainer: spatialCrop requires spatialRegion".into())
                })?;
                let scale = condition
                    .spatial_scale_factor
                    .unwrap_or(geometry.spatial_scale as u32) as usize;
                if scale == 0 {
                    return Err(CandleError::Msg(
                        "ltx_2_5 trainer: spatialScaleFactor must be > 0".into(),
                    ));
                }
                let y1 = (top as usize / scale).min(geometry.height);
                let x1 = (left as usize / scale).min(geometry.width);
                let y2 = (bottom as usize).div_ceil(scale).min(geometry.height);
                let x2 = (right as usize).div_ceil(scale).min(geometry.width);
                for frame in 0..geometry.frames {
                    for y in y1..y2 {
                        for x in x1..x2 {
                            generated[frame * frame_tokens + y * geometry.width + x] = 0.0;
                        }
                    }
                }
            }
        }
    }
    let conditioning: Vec<f32> = generated.iter().map(|value| 1.0 - value).collect();
    let loss = generated.clone();
    Ok((
        Tensor::from_vec(generated, (1, tokens, 1), device)?,
        Tensor::from_vec(conditioning, (1, tokens, 1), device)?,
        Tensor::from_vec(loss, (1, tokens, 1), device)?,
    ))
}

fn apply_modality_conditioning(
    noisy: &Tensor,
    clean: &Tensor,
    plan: &ModalityConditioningPlan,
    condition_tensors: &HashMap<String, Tensor>,
    geometry: LtxTokenGeometry,
    seed: u64,
    sample_index: u64,
) -> Result<(Tensor, Tensor)> {
    let (_, tokens, _) = noisy.dims3()?;
    if tokens != geometry.tokens() {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: modality geometry and latent tokens disagree".into(),
        ));
    }
    let (generated, conditioning, loss) = modality_masks(
        plan,
        condition_tensors,
        geometry,
        seed,
        sample_index,
        noisy.device(),
    )?;
    // Every intrinsic condition (first/prefix/suffix/crop/mask) reads from the clean target
    // stream.  A reference is handled exclusively by `append_active_reference`, where it gets
    // independent positions and a zero-loss suffix; substituting it here would erase target
    // tokens and change the IC objective.
    let input = (noisy.broadcast_mul(&generated)? + clean.broadcast_mul(&conditioning)?)?;
    Ok((input, loss))
}

/// Upstream IC workflows concatenate clean reference tokens after the generated target stream.
/// Their zero loss mask preserves the references as context without training the model to recreate
/// them; concatenating their own RoPE positions is the bit that prevents a reference from merely
/// replacing arbitrary target tokens.
#[allow(clippy::too_many_arguments)]
fn append_active_reference(
    plan: &ModalityConditioningPlan,
    condition_tensors: &HashMap<String, Tensor>,
    condition_positions: &HashMap<String, Tensor>,
    seed: u64,
    sample_index: u64,
    input: Tensor,
    target: Tensor,
    loss: Tensor,
    positions: Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let active = plan
        .conditions
        .iter()
        .enumerate()
        .find(|(ordinal, condition)| {
            condition.reference
                && condition_draw(seed, sample_index, *ordinal) < condition.probability
        });
    let Some((_, condition)) = active else {
        return Ok((input, target, loss, positions));
    };
    let key = condition
        .tensor_key
        .as_deref()
        .expect("parser requires reference tensorKey");
    let reference = condition_tensors.get(key).ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle is missing reference tensor `{key}`"
        ))
    })?;
    let reference_positions = condition_positions.get(key).ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle is missing reference positions for `{key}`"
        ))
    })?;
    let (_, _, channels) = input.dims3()?;
    let (batch, reference_tokens, reference_channels) = reference.dims3()?;
    if batch != input.dim(0)? || reference_channels != channels {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: reference `{key}` shape {:?} does not match target {:?}",
            reference.dims(),
            input.dims()
        )));
    }
    if reference_positions.dim(2)? != reference_tokens {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: reference `{key}` positions do not match its token count"
        )));
    }
    let zeros = Tensor::zeros(
        (batch, reference_tokens, channels),
        target.dtype(),
        target.device(),
    )?;
    let zero_loss = Tensor::zeros((1, reference_tokens, 1), loss.dtype(), loss.device())?;
    Ok((
        Tensor::cat(&[&input, reference], 1)?,
        Tensor::cat(&[&target, &zeros], 1)?,
        Tensor::cat(&[&loss, &zero_loss], 1)?,
        Tensor::cat(&[&positions, reference_positions], 2)?,
    ))
}

fn masked_velocity_loss(
    prediction: &Tensor,
    target: &Tensor,
    loss_mask: &Tensor,
) -> Result<Tensor> {
    let sq = (prediction - target)?.sqr()?.broadcast_mul(loss_mask)?;
    let (batch, _, channels) = prediction.dims3()?;
    let denom = loss_mask.sum_all()?.to_scalar::<f32>()?.max(1.0) * batch as f32 * channels as f32;
    Ok((sq.sum_all()? / denom as f64)?)
}

fn masked_mae_loss(prediction: &Tensor, target: &Tensor, loss_mask: &Tensor) -> Result<Tensor> {
    let diff = (prediction - target)?.abs()?.broadcast_mul(loss_mask)?;
    let (batch, _, channels) = prediction.dims3()?;
    let denom = loss_mask.sum_all()?.to_scalar::<f32>()?.max(1.0) * batch as f32 * channels as f32;
    Ok((diff.sum_all()? / denom as f64)?)
}

/// Convert the executable target/reference mask into the `[B,S]` table fed to adaLN.  This is
/// intentionally not inferred from whether the modality is generated: intrinsic conditions and
/// appended references are present within a generated stream but still remain at timestep zero.
fn masked_token_timesteps(loss_mask: &Tensor, sigma: f64) -> Result<Tensor> {
    Ok((loss_mask.squeeze(2)? * sigma)?)
}

/// Upstream v1.2.0 LTX-2.5 adapter workflow vocabulary.  This typed declaration intentionally
/// matches the MLX sibling, so the workspace parity gate sees one canonical fifteen-mode surface.
pub const LTX25_WORKFLOWS: [Ltx25Workflow; 15] = [
    Ltx25Workflow::I2vLora,
    Ltx25Workflow::T2vLora,
    Ltx25Workflow::V2aLora,
    Ltx25Workflow::A2vLora,
    Ltx25Workflow::T2aLora,
    Ltx25Workflow::VideoExtendLora,
    Ltx25Workflow::VideoInpaintingLora,
    Ltx25Workflow::VideoOutpaintingLora,
    Ltx25Workflow::VideoSuffixLora,
    Ltx25Workflow::AudioExtendLora,
    Ltx25Workflow::AudioInpaintingLora,
    Ltx25Workflow::AudioSuffixLora,
    Ltx25Workflow::Av2avIcLora,
    Ltx25Workflow::V2vIcLora,
    Ltx25Workflow::A2aIcLora,
];

enum TrainingTokenizer {
    Gemma3(tokenizers::Tokenizer),
    Gemma4(Ltx25Tokenizer),
}

impl TrainingTokenizer {
    fn encode(&self, text: &str, device: &Device) -> Result<(Tensor, Vec<u32>)> {
        match self {
            Self::Gemma3(tokenizer) => tokenize(tokenizer, text, device),
            Self::Gemma4(tokenizer) => tokenizer.encode(text, TRAIN_TEXT_MAX_LENGTH, device),
        }
    }
}

enum TrainingTextEncoder {
    Gemma3(Box<LtxTextEncoder>),
    Gemma4(Box<Ltx25TextEncoder>),
}

impl TrainingTextEncoder {
    fn encode(&self, ids: &Tensor, mask: &[u32]) -> Result<Tensor> {
        match self {
            Self::Gemma3(encoder) => Ok(encoder.encode(ids, mask)?),
            Self::Gemma4(encoder) => Ok(encoder.encode(ids, mask)?),
        }
    }

    fn encode_both(&self, ids: &Tensor, mask: &[u32]) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Gemma3(encoder) => Ok(encoder.encode_both(ids, mask)?),
            Self::Gemma4(encoder) => Ok(encoder.encode_both(ids, mask)?),
        }
    }
}

enum TrainingRoute {
    Ltx23 {
        root: PathBuf,
        gemma_override: Option<PathBuf>,
    },
    Ltx25 {
        bundle: LtxBundle,
        tier: Ltx25Tier,
    },
}

/// One preview prompt's already-encoded distilled conditioning.
pub struct LtxSampleState {
    contexts: Vec<SampleContext>,
    vae: Arc<LtxVideoVae>,
    aux: TrainingAux,
    latent_edge: usize,
}

enum SampleContext {
    Ltx23(Tensor),
    Ltx25 {
        workflow: Ltx25Workflow,
        video: Tensor,
        audio: Tensor,
        negative_video: Tensor,
        negative_audio: Tensor,
        validation: Ltx25ValidationRenderPlan,
    },
}

#[derive(Clone)]
pub struct AvTrainingPositions {
    video: Option<Tensor>,
    audio: Option<Tensor>,
}

#[derive(Clone)]
pub enum TrainingAux {
    Ltx23(Tensor),
    Ltx25(AvTrainingPositions),
}

pub enum TrainingDiT {
    Ltx23(Box<LtxDiT>),
    Ltx25(Box<AvDiT>),
}

impl LoraHost for TrainingDiT {
    fn visit_lora_mut(&mut self, f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>) -> Result<()> {
        match self {
            Self::Ltx23(dit) => dit.visit_lora_mut(f),
            Self::Ltx25(dit) => dit.visit_lora_mut(f),
        }
    }
}

pub enum TrainingCached {
    Ltx23 { clean: Tensor, context: Tensor },
    Ltx25(Box<AvTrainingExample>),
}

#[derive(Clone)]
pub struct AvTrainingExample {
    video_clean: Option<Tensor>,
    audio_clean: Option<Tensor>,
    video_context: Option<Tensor>,
    audio_context: Option<Tensor>,
    conditioning: Ltx25ConditioningPlan,
    condition_tensors: HashMap<String, Tensor>,
    condition_positions: HashMap<String, Tensor>,
    positions: AvTrainingPositions,
    video_geometry: Option<LtxTokenGeometry>,
    audio_geometry: Option<LtxTokenGeometry>,
    condition_seed: u64,
    sample_index: u64,
}

/// Lazy trainer: caching loads Gemma + VAE first and drops Gemma before `build_dit`.
pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    route: TrainingRoute,
    device: Device,
}

impl LtxTrainer {
    fn label(&self) -> &'static str {
        match self.route {
            TrainingRoute::Ltx23 { .. } => "ltx_2_3 trainer",
            TrainingRoute::Ltx25 { .. } => "ltx_2_5 trainer",
        }
    }
}

pub fn trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(TRAINER_ID)
}

pub fn trainer_descriptor_25() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_25_ID)
}

fn trainer_descriptor_for(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "ltx",
        backend: "candle",
        modality: Modality::Video,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: false,
    }
}

pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: expects the split q4/q8 LTX tier directory, not a single checkpoint"
            )))
        }
    };
    let gemma_override = match spec.text_encoder.as_ref() {
        Some(WeightsSource::Dir(path)) => Some(path.clone()),
        Some(WeightsSource::File(_)) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: text_encoder must be a Gemma snapshot directory"
            )))
        }
        None => None,
    };
    Ok(Box::new(LtxTrainer {
        descriptor: trainer_descriptor(),
        route: TrainingRoute::Ltx23 {
            root,
            gemma_override,
        },
        device: candle_gen::default_device()?,
    }))
}

/// Construct the QLoRA-capable LTX-2.5 trainer from the same split bundle used by the ordinary
/// generator.  `AvConfig::from_bundle` carries the real `ff_bias:false` shape into `LtxDiT`; the
/// bundle resolver and Gemma-version assertion make a mismatched 2.3/Gemma-3 layout fail before
/// any trainable factor is built.
pub fn load_trainer_25(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer: expects a split-component directory, not a single checkpoint"
        )))
        }
    };
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    // Keep the original root check in the error path: explicit components may be outside root, but
    // a bare `LoadSpec` must still name a directory so the whole training lifecycle is reproducible.
    if !root.is_dir() {
        return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer: weights directory does not exist: {}",
            root.display()
        )));
    }
    let tier = resolve_ltx25_dev_q4_tier(root, &bundle)?;
    Ok(Box::new(LtxTrainer {
        descriptor: trainer_descriptor_25(),
        route: TrainingRoute::Ltx25 { bundle, tier },
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION_25 = trainer_descriptor_25 => load_trainer_25
}

/// LTX always trains f32. Bf16 was measured to decorrelate gradients through the deep distilled DiT.
fn validate_ltx_request(req: &TrainingRequest, label: &str) -> Result<()> {
    flow_match::validate_flow_match_request(req, label)?;
    if req.config.network_type != NetworkType::Lora {
        return Err(CandleError::Msg(format!(
            "{label}: LoKr training is unsupported; LTX training is LoRA-only"
        )));
    }
    let dtype = req.config.train_dtype.trim();
    if dtype.eq_ignore_ascii_case("bf16") || dtype.eq_ignore_ascii_case("bfloat16") {
        return Err(CandleError::Msg(format!(
            "{label}: bf16 training is rejected; LTX LoRA training requires f32"
        )));
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if matches!(self.route, TrainingRoute::Ltx25 { .. }) {
            return validate_ltx25_training_request(req).map_err(Into::into);
        }
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        validate_ltx_request(req, self.label()).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

/// Dense-backward peak projection inherited from the MLX real-weight sweep (16.9 GiB resident plus
/// 0.0251 GiB/token). This is a conservative fail-fast policy input, not a claim of Candle-calibrated
/// VRAM parity; a dedicated Candle real-weight sweep remains follow-up validation.
pub fn projected_dense_peak_gb(latent_tokens: usize) -> f64 {
    16.9 + 0.0251 * latent_tokens as f64
}

fn preflight_against_budget(resolution: u32, available_bytes: u64) -> Result<()> {
    let edge = bucket_resolution(resolution);
    let latent_edge = edge as usize / SPATIAL_SCALE;
    let projected = projected_dense_peak_gb(latent_edge * latent_edge);
    let available_gb = available_bytes as f64 / 1024f64.powi(3);
    let safe = available_gb * SAFE_MEMORY_FRACTION;
    if projected > safe {
        return Err(CandleError::Msg(format!(
            "{LABEL}: a dense first step at {edge}px projects ~{projected:.1} GiB, exceeding the \
             {safe:.1} GiB safe budget ({available_gb:.1} GiB free x {SAFE_MEMORY_FRACTION:.2}); \
             enable gradient checkpointing or reduce resolution"
        )));
    }
    Ok(())
}

fn available_device_bytes() -> u64 {
    #[cfg(feature = "cuda")]
    {
        use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
        if let Ok((free, _)) = cuda::mem_get_info() {
            return free as u64;
        }
    }
    // CPU/Metal tests use a conservative 64 GiB logical budget; production candle LTX is CUDA.
    64 * 1024 * 1024 * 1024
}

fn tier(root: &std::path::Path, gemma_override: Option<&std::path::Path>) -> Result<TierPaths> {
    TierPaths::detect(root, gemma_override).ok_or_else(|| {
        CandleError::Msg(format!(
            "{LABEL}: {} is not a split packed LTX tier (missing transformer.safetensors or \
             quantize_config.json)",
            root.display()
        ))
    })
}

/// Validate the complete packed 2.5 tier and locate its independently packaged connector.  The
/// connector cannot be recovered from the packed transformer: q4/q8 tiers intentionally split it
/// so the Gemma projection and both AV connectors stay dense and identity-addressable.
fn validate_ltx25_dev_q4_manifest(manifest: &crate::tier::Ltx25TierManifest) -> Result<()> {
    if !manifest.model_version.starts_with("2.5")
        || manifest.tier != "q4"
        || !manifest.quantized
        || manifest.quant.bits != 4
        || manifest.quant.group != crate::quant::GROUP_SIZE
    {
        return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer requires the packed LTX-2.5 dev/q4 tier (q4, 4-bit, group {}); got model_version={} tier={} quantized={} bits={} group={}",
            crate::quant::GROUP_SIZE,
            manifest.model_version,
            manifest.tier,
            manifest.quantized,
            manifest.quant.bits,
            manifest.quant.group,
        )));
    }
    Ok(())
}

/// The QLoRA trainer is deliberately narrower than inference: it trains only the exact split
/// LTX-2.5 Dev/q4 release.  A dense, q8, or distilled artifact has a different memory/trajectory
/// contract and must fail before a model/connector is materialized.
fn resolve_ltx25_dev_q4_tier(root: &std::path::Path, bundle: &LtxBundle) -> Result<Ltx25Tier> {
    if bundle.layout() != LtxCheckpointLayout::Split {
        return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer requires a split LTX-2.5 bundle, got {}",
            bundle.layout().id()
        )));
    }
    if crate::dev_sampler::TransformerVariant::from_bundle(bundle)?
        != crate::dev_sampler::TransformerVariant::Dev
    {
        return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer requires the dev transformer; distilled checkpoints cannot execute QLoRA validation"
        )));
    }
    let tier = Ltx25Tier::detect(root)?.ok_or_else(|| {
        CandleError::Msg(format!(
            "{MODEL_25_ID} trainer requires a split_model.json LTX-2.5 tier manifest under {}",
            root.display()
        ))
    })?;
    validate_ltx25_dev_q4_manifest(tier.manifest())?;
    tier.validate()
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    let selected_transformer = bundle
        .require(LtxComponent::Transformer)
        .map_err(|error| CandleError::Msg(error.to_string()))?
        .path();
    let tier_transformer = tier
        .file(Ltx25Component::Transformer)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    if selected_transformer != tier_transformer {
        return Err(CandleError::Msg(format!(
            "{MODEL_25_ID} trainer refuses transformer {} outside the selected dev/q4 tier {}",
            selected_transformer.display(),
            tier_transformer.display(),
        )));
    }
    Ok(tier)
}

fn pad_training_ids(mut ids: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    ids.truncate(TRAIN_TEXT_MAX_LENGTH);
    let pad = TRAIN_TEXT_MAX_LENGTH - ids.len();
    let mut padded = vec![0u32; pad];
    padded.extend_from_slice(&ids);
    let mut mask = vec![0u32; pad];
    mask.extend(std::iter::repeat_n(1u32, ids.len()));
    (padded, mask)
}

fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    device: &Device,
) -> Result<(Tensor, Vec<u32>)> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| CandleError::Msg(format!("{LABEL}: tokenize: {e}")))?;
    let (padded, mask) = pad_training_ids(enc.get_ids().to_vec());
    Ok((
        Tensor::from_vec(padded, (1, TRAIN_TEXT_MAX_LENGTH), device)?,
        mask,
    ))
}

fn encode_context(
    tokenizer: &TrainingTokenizer,
    encoder: &TrainingTextEncoder,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let (ids, mask) = tokenizer.encode(text, device)?;
    Ok(encoder.encode(&ids, &mask)?.to_dtype(DType::F32)?)
}

fn encode_contexts(
    tokenizer: &TrainingTokenizer,
    encoder: &TrainingTextEncoder,
    text: &str,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (ids, mask) = tokenizer.encode(text, device)?;
    let (video, audio) = encoder.encode_both(&ids, &mask)?;
    Ok((video.to_dtype(DType::F32)?, audio.to_dtype(DType::F32)?))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LtxTokenGeometry {
    frames: usize,
    height: usize,
    width: usize,
    spatial_scale: usize,
}

impl LtxTokenGeometry {
    fn tokens(self) -> usize {
        self.frames * self.height * self.width
    }
}

fn condition_draw(seed: u64, sample_index: u64, ordinal: usize) -> f32 {
    let mut x = seed
        .wrapping_add(sample_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((ordinal as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 40) as f32) / ((1_u32 << 24) as f32)
}

/// One actual LTX-2.5 prepared example.  We do not re-encode a thumbnail image on this route: the
/// worker has already selected the upstream workflow and emitted the exact AV latent pair plus any
/// intrinsic conditioning tensors into this safetensors-v1 pack.
struct PreparedLtx25Example {
    video_clean: Option<Tensor>,
    audio_clean: Option<Tensor>,
    positions: AvTrainingPositions,
    video_geometry: Option<LtxTokenGeometry>,
    audio_geometry: Option<LtxTokenGeometry>,
    condition_tensors: HashMap<String, Tensor>,
    condition_positions: HashMap<String, Tensor>,
}

fn required_prepared_path(item: &candle_gen::gen_core::train::TrainingItem) -> Result<&str> {
    item.model_options
        .get("ltxPreparedBundlePath")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| {
            CandleError::Msg(
                "ltx_2_5 trainer: every item requires model_options.ltxPreparedBundlePath".into(),
            )
        })
}

fn prepared_metadata_shape(raw: Option<&str>, key: &str, rank: usize) -> Result<Vec<usize>> {
    let raw = raw.ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle metadata `{key}` is required"
        ))
    })?;
    let values: Vec<usize> = serde_json::from_str(raw).map_err(|error| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: invalid `{key}` metadata: {error}"
        ))
    })?;
    if values.len() != rank || values.contains(&0) {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: `{key}` must be a non-zero {rank}-axis shape, got {values:?}"
        )));
    }
    Ok(values)
}

fn prepared_shape(meta: &HashMap<String, String>, key: &str, rank: usize) -> Result<Vec<usize>> {
    prepared_metadata_shape(meta.get(key).map(String::as_str), key, rank)
}

fn prepared_header<'a>(
    headers: &'a [SafetensorsTensorHeader],
    key: &str,
    expected_shape: &[usize],
    path: &std::path::Path,
) -> Result<&'a SafetensorsTensorHeader> {
    let header = headers
        .iter()
        .find(|header| header.name == key)
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx_2_5 trainer: prepared bundle `{}` is missing `{key}`",
                path.display()
            ))
        })?;
    if !header.is_float() || header.shape != expected_shape {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared `{key}` in `{}` must be a floating tensor with shape {expected_shape:?}, got {:?} {:?}",
            path.display(), header.dtype, header.shape
        )));
    }
    Ok(header)
}

fn header_elements(header: &SafetensorsTensorHeader, key: &str) -> Result<usize> {
    header.shape.iter().try_fold(1_usize, |total, &axis| {
        total.checked_mul(axis).ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx_2_5 trainer: prepared `{key}` shape {:?} overflows its element count",
                header.shape
            ))
        })
    })
}

fn prepared_condition_header(
    headers: &[SafetensorsTensorHeader],
    path: &std::path::Path,
    condition: &Ltx25Condition,
    is_video: bool,
    video_shape: &[usize],
    audio_shape: &[usize],
) -> Result<()> {
    let key = condition
        .tensor_key
        .as_deref()
        .expect("parser requires tensorKey for tensor-backed conditions");
    let header = headers
        .iter()
        .find(|header| header.name == key)
        .ok_or_else(|| {
            CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle `{}` is missing selected condition tensor `{key}`",
            path.display()
        ))
        })?;
    if !header.is_float() || header.shape.contains(&0) {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared condition `{key}` in `{}` must be a non-empty floating tensor",
            path.display()
        )));
    }
    if !condition.reference {
        let expected_tokens = if is_video {
            video_shape[2] * video_shape[3] * video_shape[4]
        } else {
            audio_shape[2]
        };
        if header_elements(header, key)? != expected_tokens {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: prepared mask `{key}` has {} elements; {expected_tokens} are required for the selected modality",
                header_elements(header, key)?
            )));
        }
        return Ok(());
    }

    match header.shape.as_slice() {
        [batch, channels, ..] if header.shape.len() == 5 && is_video => {
            if *batch != video_shape[0] || *channels != video_shape[1] {
                return Err(CandleError::Msg(format!(
                    "ltx_2_5 trainer: video reference `{key}` must retain batch/channel {:?}, got {:?}",
                    &video_shape[..2], &header.shape[..2]
                )));
            }
            if condition
                .spatial_scale_factor
                .unwrap_or(SPATIAL_SCALE as u32)
                != SPATIAL_SCALE as u32
                || condition.temporal_scale_factor.unwrap_or(8) != 8
            {
                return Err(CandleError::Msg(format!(
                    "ltx_2_5 trainer: video reference `{key}` uses unsupported position scale factors"
                )));
            }
        }
        [batch, channels, _, feature] if header.shape.len() == 4 && !is_video => {
            if *batch != audio_shape[0] || *channels != audio_shape[1] || *feature != audio_shape[3]
            {
                return Err(CandleError::Msg(format!(
                    "ltx_2_5 trainer: audio reference `{key}` must retain batch/channel/feature {:?}, got {:?}",
                    [audio_shape[0], audio_shape[1], audio_shape[3]], header.shape
                )));
            }
            if condition.spatial_scale_factor.unwrap_or(1) != 1
                || condition.temporal_scale_factor.unwrap_or(4) != 4
            {
                return Err(CandleError::Msg(format!(
                    "ltx_2_5 trainer: audio reference `{key}` uses unsupported position scale factors"
                )));
            }
        }
        _ => {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: reference `{key}` must be a {} latent tensor, got shape {:?}",
                if is_video {
                    "5-axis video"
                } else {
                    "4-axis audio"
                },
                header.shape
            )));
        }
    }
    Ok(())
}

/// Weights-free LTX-2.5 QLoRA admission.  This is intentionally callable before a trainer/model is
/// loaded: it validates the exact selected upstream workflow and checks each prepared safetensors-v1
/// pack through its header only, never materializing a base-model or latent tensor.
pub fn validate_ltx25_training_request(req: &TrainingRequest) -> Result<()> {
    let descriptor = trainer_descriptor_25();
    gen_core::train::validate_control_request(&descriptor, req)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    gen_core::train::validate_full_finetune_request(&descriptor, req)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    validate_ltx_request(req, MODEL_25_ID)?;
    if !req.config.alpha.is_finite() || req.config.alpha <= 0.0 {
        return Err(CandleError::Msg(
            "ltx_2_5 trainer: alpha must be finite and > 0".into(),
        ));
    }
    let conditioning = Ltx25ConditioningPlan::from_request(req)?;
    for item in &req.items {
        validate_prepared_ltx25_bundle(item, &conditioning)?;
    }
    Ok(())
}

fn validate_prepared_ltx25_bundle(
    item: &candle_gen::gen_core::train::TrainingItem,
    conditioning: &Ltx25ConditioningPlan,
) -> Result<()> {
    let path = PathBuf::from(required_prepared_path(item)?);
    let metadata = safetensors_file_metadata(&path).map_err(|error| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: could not read prepared bundle header `{}`: {error}",
            path.display()
        ))
    })?;
    if metadata.get("schemaVersion").map(String::as_str) != Some("ltx-prepared-v1") {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle {} must declare schemaVersion=ltx-prepared-v1",
            path.display()
        )));
    }
    let headers = safetensors_path_tensor_headers(&path).map_err(|error| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: could not read prepared bundle tensors `{}`: {error}",
            path.display()
        ))
    })?;
    let video_shape = if conditioning.video.present {
        let shape = prepared_metadata_shape(
            metadata.get("videoShape").map(String::as_str),
            "videoShape",
            5,
        )?;
        let [batch, channels, frames, height, width]: [usize; 5] =
            shape.clone().try_into().map_err(|_| {
                CandleError::Msg("ltx_2_5 trainer: videoShape must be [B,C,F,H,W]".into())
            })?;
        if batch != 1
            || channels != LTX25_VIDEO_LATENT_CHANNELS
            || frames == 0
            || height == 0
            || width == 0
        {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: videoShape must be [1,{LTX25_VIDEO_LATENT_CHANNELS},F,H,W] with positive F/H/W, got {shape:?}"
            )));
        }
        prepared_header(&headers, "video_latents", &shape, &path)?;
        Some(shape)
    } else {
        None
    };
    let audio_shape = if conditioning.audio.present {
        let shape = prepared_metadata_shape(
            metadata.get("audioShape").map(String::as_str),
            "audioShape",
            4,
        )?;
        let [batch, channels, frames, features]: [usize; 4] =
            shape.clone().try_into().map_err(|_| {
                CandleError::Msg("ltx_2_5 trainer: audioShape must be [B,C,T,F]".into())
            })?;
        if batch != 1
            || channels == 0
            || frames == 0
            || features == 0
            || channels.checked_mul(features) != Some(LTX25_AUDIO_FLAT_CHANNELS)
        {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: audioShape must have B=1, positive C/T/F, and C*F={LTX25_AUDIO_FLAT_CHANNELS}; got {shape:?}"
            )));
        }
        prepared_header(&headers, "audio_latents", &shape, &path)?;
        Some(shape)
    } else {
        None
    };
    let fps = if video_shape.is_some() {
        let fps: f64 = metadata
            .get("fps")
            .ok_or_else(|| {
                CandleError::Msg(
                    "ltx_2_5 trainer: prepared bundle metadata `fps` is required for video".into(),
                )
            })?
            .parse()
            .map_err(|error| {
                CandleError::Msg(format!("ltx_2_5 trainer: invalid prepared `fps`: {error}"))
            })?;
        if !fps.is_finite() || fps <= 0.0 {
            return Err(CandleError::Msg(
                "ltx_2_5 trainer: prepared `fps` must be finite and positive".into(),
            ));
        }
        Some(fps)
    } else {
        None
    };
    if let (Some(video_shape), Some(audio_shape), Some(fps)) = (&video_shape, &audio_shape, fps) {
        let output_frames = video_shape[2]
            .checked_sub(1)
            .and_then(|frames| frames.checked_mul(8))
            .and_then(|frames| frames.checked_add(1))
            .ok_or_else(|| {
                CandleError::Msg(
                    "ltx_2_5 trainer: videoShape duration overflows frame mapping".into(),
                )
            })?;
        let expected_audio_frames = compute_audio_frames(output_frames, fps);
        if audio_shape[2] != expected_audio_frames {
            return Err(CandleError::Msg(format!(
                "ltx_2_5 trainer: prepared audio frames {} disagree with video shape/fps expectation {expected_audio_frames}",
                audio_shape[2]
            )));
        }
    }
    for (is_video, modality, shape) in [
        (true, &conditioning.video, video_shape.as_deref()),
        (false, &conditioning.audio, audio_shape.as_deref()),
    ] {
        let Some(shape) = shape else {
            continue;
        };
        for condition in &modality.conditions {
            if condition.tensor_key.is_some() {
                prepared_condition_header(
                    &headers,
                    &path,
                    condition,
                    is_video,
                    if is_video {
                        shape
                    } else {
                        video_shape.as_deref().unwrap_or(&[])
                    },
                    if is_video {
                        audio_shape.as_deref().unwrap_or(&[])
                    } else {
                        shape
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn prepared_tensor(
    tensors: &HashMap<String, Tensor>,
    key: &str,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let tensor = tensors.get(key).ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle is missing `{key}`"
        ))
    })?;
    if tensor.dims() != shape {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared `{key}` shape {:?} disagrees with metadata {shape:?}",
            tensor.dims()
        )));
    }
    Ok(tensor.to_dtype(DType::F32)?.to_device(device)?)
}

fn prepare_ltx25_example(
    item: &candle_gen::gen_core::train::TrainingItem,
    conditioning: &Ltx25ConditioningPlan,
    device: &Device,
) -> Result<PreparedLtx25Example> {
    let path = PathBuf::from(required_prepared_path(item)?);
    let file = read_adapter(&path)?;
    if file.meta.get("schemaVersion").map(String::as_str) != Some("ltx-prepared-v1") {
        return Err(CandleError::Msg(format!(
            "ltx_2_5 trainer: prepared bundle {} must declare schemaVersion=ltx-prepared-v1",
            path.display()
        )));
    }
    // Call the same header-only contract as the public preflight before materializing tensors:
    // an API caller and an in-process trainer must reject the same absent stream / bad shape.
    validate_prepared_ltx25_bundle(item, conditioning)?;
    let video_shape = conditioning
        .video
        .present
        .then(|| prepared_shape(&file.meta, "videoShape", 5))
        .transpose()?;
    let audio_shape = conditioning
        .audio
        .present
        .then(|| prepared_shape(&file.meta, "audioShape", 4))
        .transpose()?;
    let fps = if video_shape.is_some() {
        let value: f64 = file
            .meta
            .get("fps")
            .ok_or_else(|| {
                CandleError::Msg(
                    "ltx_2_5 trainer: prepared bundle metadata `fps` is required for video".into(),
                )
            })?
            .parse()
            .map_err(|error| {
                CandleError::Msg(format!("ltx_2_5 trainer: invalid prepared `fps`: {error}"))
            })?;
        if !value.is_finite() || value <= 0.0 {
            return Err(CandleError::Msg(
                "ltx_2_5 trainer: prepared `fps` must be finite and positive".into(),
            ));
        }
        value
    } else {
        DEFAULT_FPS as f64
    };
    let video_clean = video_shape
        .as_ref()
        .map(|shape| -> Result<Tensor> {
            let latent = prepared_tensor(&file.tensors, "video_latents", shape, device)?;
            Ok(flatten_latent(&latent)?)
        })
        .transpose()?;
    let audio_clean = audio_shape
        .as_ref()
        .map(|shape| -> Result<Tensor> {
            let latent = prepared_tensor(&file.tensors, "audio_latents", shape, device)?;
            Ok(flatten_audio_latent(&latent)?)
        })
        .transpose()?;
    let video_positions = video_shape
        .as_ref()
        .map(|shape| create_position_grid(shape[2], shape[3], shape[4], fps as f32, device))
        .transpose()?;
    let audio_positions = audio_shape
        .as_ref()
        .map(|shape| create_audio_position_grid(shape[2], device))
        .transpose()?;

    let mut condition_tensors = HashMap::new();
    let mut condition_positions = HashMap::new();
    for (is_video, modality) in [(true, &conditioning.video), (false, &conditioning.audio)] {
        if !modality.present {
            continue;
        }
        for condition in &modality.conditions {
            let Some(key) = condition.tensor_key.as_deref() else {
                continue;
            };
            let source = file.tensors.get(key).ok_or_else(|| {
                CandleError::Msg(format!(
                    "ltx_2_5 trainer: prepared bundle `{}` is missing selected condition tensor `{key}`",
                    path.display()
                ))
            })?;
            let normalized = if condition.reference {
                match source.dims().len() {
                    5 => {
                        if !is_video
                            || condition.spatial_scale_factor != Some(SPATIAL_SCALE as u32)
                            || condition.temporal_scale_factor != Some(8)
                        {
                            return Err(CandleError::Msg(format!(
                                "ltx_2_5 trainer: video reference `{key}` uses unsupported position scale factors"
                            )));
                        }
                        let shape = source.dims();
                        condition_positions.insert(
                            key.to_owned(),
                            create_position_grid(shape[2], shape[3], shape[4], fps as f32, device)?,
                        );
                        flatten_latent(&source.to_dtype(DType::F32)?.to_device(device)?)?
                    }
                    4 => {
                        if is_video
                            || condition.spatial_scale_factor != Some(1)
                            || condition.temporal_scale_factor != Some(4)
                        {
                            return Err(CandleError::Msg(format!(
                                "ltx_2_5 trainer: audio reference `{key}` uses unsupported position scale factors"
                            )));
                        }
                        let shape = source.dims();
                        condition_positions.insert(
                            key.to_owned(),
                            create_audio_position_grid(shape[2], device)?,
                        );
                        flatten_audio_latent(&source.to_dtype(DType::F32)?.to_device(device)?)?
                    }
                    rank => return Err(CandleError::Msg(format!(
                        "ltx_2_5 trainer: reference `{key}` must be a video/audio latent with executable RoPE positions, got rank {rank}"
                    ))),
                }
            } else {
                source.to_dtype(DType::F32)?.to_device(device)?
            };
            condition_tensors.insert(key.to_owned(), normalized);
        }
    }
    Ok(PreparedLtx25Example {
        video_clean,
        audio_clean,
        positions: AvTrainingPositions {
            video: video_positions,
            audio: audio_positions,
        },
        video_geometry: video_shape.map(|shape| LtxTokenGeometry {
            frames: shape[2],
            height: shape[3],
            width: shape[4],
            spatial_scale: SPATIAL_SCALE,
        }),
        audio_geometry: audio_shape.map(|shape| LtxTokenGeometry {
            frames: shape[2],
            height: 1,
            width: 1,
            spatial_scale: 1,
        }),
        condition_tensors,
        condition_positions,
    })
}

/// Seeded sigma for LTX: always uniform and strictly inside `(1e-3, 1-1e-3)`. Timestep type/bias
/// knobs intentionally do not participate in this family recipe.
pub fn sample_ltx_sigma(seed: u64, step: u32) -> f64 {
    let lower = f32::from_bits(1e-3f32.to_bits() + 1);
    let upper = f32::from_bits((1.0f32 - 1e-3).to_bits() - 1);
    flow_match::sample_uniform_range(flow_match::timestep_seed(seed, step), lower, upper) as f64
}

#[allow(clippy::too_many_arguments)]
fn compute_av_loss_grads(
    dit: &AvDiT,
    _vars: &[Var],
    cached: &AvTrainingExample,
    _positions: &AvTrainingPositions,
    sigma: f64,
    video_noise: Option<&Tensor>,
    audio_noise: Option<&Tensor>,
    mae: bool,
) -> Result<(f32, GradStore)> {
    struct ActiveModality {
        input: Tensor,
        target: Tensor,
        loss_mask: Tensor,
        timesteps: Tensor,
        context: Tensor,
        positions: Tensor,
        generated: bool,
    }
    let prepare = |clean: Option<&Tensor>,
                   context: Option<&Tensor>,
                   geometry: Option<LtxTokenGeometry>,
                   positions: Option<&Tensor>,
                   plan: &ModalityConditioningPlan,
                   noise: Option<&Tensor>|
     -> Result<Option<ActiveModality>> {
        if !plan.present {
            // Presence is an executable routing control, not merely validation metadata.  An
            // absent stream cannot be resurrected by a stale cache entry or a caller-supplied
            // zero tensor.
            return Ok(None);
        }
        let (Some(clean), Some(context), Some(geometry), Some(positions)) =
            (clean, context, geometry, positions)
        else {
            return Ok(None);
        };
        let (noisy, target) = if plan.is_generated {
            let noise = noise.ok_or_else(|| {
                CandleError::Msg(
                    "ltx_2_5 trainer: generated modality is missing training noise".into(),
                )
            })?;
            flow_match::build_batch(clean, noise, sigma)?
        } else {
            // A present conditioning modality enters the AV transformer clean and contributes no
            // velocity objective. It is not a synthetic noised stream.
            (clean.clone(), clean.clone())
        };
        let (input, loss_mask) = apply_modality_conditioning(
            &noisy,
            clean,
            plan,
            &cached.condition_tensors,
            geometry,
            cached.condition_seed,
            cached.sample_index,
        )?;
        let (input, target, loss_mask, positions) = append_active_reference(
            plan,
            &cached.condition_tensors,
            &cached.condition_positions,
            cached.condition_seed,
            cached.sample_index,
            input,
            target,
            loss_mask,
            positions.clone(),
        )?;
        // The flexible strategy's generated/loss mask is also the token-wise sigma multiplier:
        // intrinsic, clean/frozen, and appended reference tokens all remain at timestep zero.
        // The prepared bundle is deliberately batch-one, so squeezing the final channel yields
        // the exact `[B,S]` table consumed by the AV transformer APIs.
        let timesteps = masked_token_timesteps(&loss_mask, sigma)?;
        Ok(Some(ActiveModality {
            input,
            target,
            loss_mask,
            timesteps,
            context: context.clone(),
            positions,
            generated: plan.is_generated,
        }))
    };

    let video = prepare(
        cached.video_clean.as_ref(),
        cached.video_context.as_ref(),
        cached.video_geometry,
        cached.positions.video.as_ref(),
        &cached.conditioning.video,
        video_noise,
    )?;
    let audio = prepare(
        cached.audio_clean.as_ref(),
        cached.audio_context.as_ref(),
        cached.audio_geometry,
        cached.positions.audio.as_ref(),
        &cached.conditioning.audio,
        audio_noise,
    )?;
    for (name, modality) in [("video", video.as_ref()), ("audio", audio.as_ref())] {
        if let Some(modality) = modality {
            if modality.generated {
                let generated_tokens = modality.loss_mask.sum_all()?.to_scalar::<f32>()?;
                if !generated_tokens.is_finite() || generated_tokens <= 0.0 {
                    return Err(CandleError::Msg(format!(
                        "ltx_2_5 trainer: generated {name} modality has no active loss tokens"
                    )));
                }
            }
        }
    }
    let mut losses = Vec::with_capacity(2);
    match (video, audio) {
        (Some(video), Some(audio)) => {
            let (video_velocity, audio_velocity) = dit.forward_token_timed(
                &video.input,
                &audio.input,
                &video.timesteps,
                &audio.timesteps,
                &video.context,
                &audio.context,
                &video.positions,
                &audio.positions,
            )?;
            if video.generated {
                losses.push(if mae {
                    masked_mae_loss(&video_velocity, &video.target, &video.loss_mask)?
                } else {
                    masked_velocity_loss(&video_velocity, &video.target, &video.loss_mask)?
                });
            }
            if audio.generated {
                losses.push(if mae {
                    masked_mae_loss(&audio_velocity, &audio.target, &audio.loss_mask)?
                } else {
                    masked_velocity_loss(&audio_velocity, &audio.target, &audio.loss_mask)?
                });
            }
        }
        (Some(video), None) => {
            if video.generated {
                let velocity = dit.forward_video_only_token_timed(
                    &video.input,
                    &video.timesteps,
                    &video.context,
                    &video.positions,
                )?;
                losses.push(if mae {
                    masked_mae_loss(&velocity, &video.target, &video.loss_mask)?
                } else {
                    masked_velocity_loss(&velocity, &video.target, &video.loss_mask)?
                });
            }
        }
        (None, Some(audio)) => {
            if audio.generated {
                let velocity = dit.forward_audio_only_token_timed(
                    &audio.input,
                    &audio.timesteps,
                    &audio.context,
                    &audio.positions,
                )?;
                losses.push(if mae {
                    masked_mae_loss(&velocity, &audio.target, &audio.loss_mask)?
                } else {
                    masked_velocity_loss(&velocity, &audio.target, &audio.loss_mask)?
                });
            }
        }
        (None, None) => {
            return Err(CandleError::Msg(
                "ltx_2_5 trainer: workflow selected no executable modality".into(),
            ));
        }
    }
    let active = losses.len();
    let mut losses = losses.into_iter();
    let mut loss = losses.next().ok_or_else(|| {
        CandleError::Msg("ltx_2_5 trainer: no generated modality contributed to the loss".into())
    })?;
    for next in losses {
        loss = (loss + next)?;
    }
    let loss = (loss / active as f64)?;
    let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    Ok((value, loss.backward()?))
}

/// CPU-only production-lifecycle modes.  Keep both shapes live: a joint objective exercises the
/// dual-stream cross-modal trunk, while a single-video objective proves an absent stream cannot be
/// replaced by a synthetic zero tensor.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ProductionAvLifecycle {
    Joint,
    VideoOnly,
    AudioOnly,
    ZeroVideoLoss,
}

/// CPU-only lifecycle hook for the AV QLoRA regression suite.  It deliberately enters the same
/// `compute_av_loss_grads` production function used by `micro_step`, with a first-frame mask, so
/// a future helper-only loss path cannot make the lifecycle suite green while the trainer ignores
/// generated masks or joint audio/video gradients.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn production_masked_av_lifecycle(
    dit: &AvDiT,
    video_clean: &Tensor,
    audio_clean: &Tensor,
    video_context: &Tensor,
    audio_context: &Tensor,
    video_positions: &Tensor,
    audio_positions: &Tensor,
    mode: ProductionAvLifecycle,
) -> Result<(f32, GradStore)> {
    let video_tokens = video_clean.dim(1)?;
    let audio_tokens = audio_clean.dim(1)?;
    let generated_video = |conditions| ModalityConditioningPlan {
        present: true,
        is_generated: true,
        conditions,
    };
    let generated_audio = || ModalityConditioningPlan {
        present: true,
        is_generated: true,
        conditions: Vec::new(),
    };
    let absent = || ModalityConditioningPlan {
        present: false,
        is_generated: false,
        conditions: Vec::new(),
    };
    let first = || Ltx25Condition {
        kind: IntrinsicCondition::First,
        probability: 1.0,
        temporal_boundary: None,
        spatial_region: None,
        tensor_key: None,
        spatial_scale_factor: None,
        temporal_scale_factor: None,
        reference: false,
    };
    let (workflow, video_plan, audio_plan) = match mode {
        ProductionAvLifecycle::Joint => (
            Ltx25Workflow::T2vLora,
            generated_video(vec![first()]),
            generated_audio(),
        ),
        ProductionAvLifecycle::VideoOnly => (
            Ltx25Workflow::I2vLora,
            generated_video(vec![first()]),
            absent(),
        ),
        ProductionAvLifecycle::AudioOnly => (Ltx25Workflow::T2aLora, absent(), generated_audio()),
        ProductionAvLifecycle::ZeroVideoLoss => (
            Ltx25Workflow::I2vLora,
            generated_video(vec![Ltx25Condition {
                kind: IntrinsicCondition::Prefix,
                probability: 1.0,
                temporal_boundary: Some(video_tokens as u32),
                spatial_region: None,
                tensor_key: None,
                spatial_scale_factor: None,
                temporal_scale_factor: None,
                reference: false,
            }]),
            absent(),
        ),
    };
    let (video_present, video_generated) = (video_plan.present, video_plan.is_generated);
    let (audio_present, audio_generated) = (audio_plan.present, audio_plan.is_generated);
    let conditioning = Ltx25ConditioningPlan {
        workflow,
        video: video_plan,
        audio: audio_plan,
        validation: LTX25_VALIDATION_DEFAULTS.into(),
    };
    let cached = AvTrainingExample {
        video_clean: video_present.then(|| video_clean.clone()),
        audio_clean: audio_present.then(|| audio_clean.clone()),
        video_context: video_present.then(|| video_context.clone()),
        audio_context: audio_present.then(|| audio_context.clone()),
        conditioning,
        condition_tensors: HashMap::new(),
        condition_positions: HashMap::new(),
        positions: AvTrainingPositions {
            video: video_present.then(|| video_positions.clone()),
            audio: audio_present.then(|| audio_positions.clone()),
        },
        video_geometry: video_present.then_some(LtxTokenGeometry {
            frames: video_tokens,
            height: 1,
            width: 1,
            spatial_scale: SPATIAL_SCALE,
        }),
        audio_geometry: audio_present.then_some(LtxTokenGeometry {
            frames: audio_tokens,
            height: 1,
            width: 1,
            spatial_scale: 1,
        }),
        condition_seed: 0,
        sample_index: 0,
    };
    let video_noise = video_generated
        .then(|| Tensor::zeros_like(video_clean))
        .transpose()?;
    let audio_noise = audio_generated
        .then(|| Tensor::zeros_like(audio_clean))
        .transpose()?;
    compute_av_loss_grads(
        dit,
        &[],
        &cached,
        &cached.positions,
        0.37,
        video_noise.as_ref(),
        audio_noise.as_ref(),
        false,
    )
}

/// Kept byte-for-byte in the LTX-2.3 branch: introducing the AV 2.5 trainer must not change the
/// legacy video-only loss or its optional checkpointed backward path.
#[allow(clippy::too_many_arguments)]
fn compute_ltx23_loss_grads(
    dit: &LtxDiT,
    vars: &[Var],
    clean: &Tensor,
    context: &Tensor,
    positions: &Tensor,
    sigma: f64,
    noise: &Tensor,
    mae: bool,
    checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (x_t, target) = flow_match::build_batch(clean, noise, sigma)?;
    if checkpoint {
        let (hidden, ctx) = dit.forward_pre_main(&x_t, sigma, context, positions)?;
        let mut segments = dit.main_block_segments(&ctx);
        let target = target.clone();
        let ctx_ref = &ctx;
        segments.push(Box::new(move |state: &[Tensor]| {
            let velocity = dit.velocity_out(&state[0], ctx_ref)?;
            Ok(vec![velocity_loss(&velocity, &target, mae)?])
        }));
        checkpointed_backward(&segments, &[hidden.detach()], vars)
    } else {
        let velocity = dit.forward(&x_t, sigma, context, positions)?;
        let loss = velocity_loss(&velocity, &target, mae)?;
        let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        Ok((value, loss.backward()?))
    }
}

impl FlowMatchTrainer for LtxTrainer {
    type Dit = TrainingDiT;
    type Cached = TrainingCached;
    type Aux = TrainingAux;
    type SampleState = LtxSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        match self.route {
            TrainingRoute::Ltx23 { .. } => &LTX_ATTN_TARGETS,
            TrainingRoute::Ltx25 { .. } => LTX_AV_LORA_TARGETS,
        }
    }

    fn preflight(&self, req: &TrainingRequest) -> Result<()> {
        if matches!(self.route, TrainingRoute::Ltx25 { .. }) && req.config.gradient_checkpointing {
            return Err(CandleError::Msg(format!(
                "{}: AV QLoRA gradient checkpointing is not implemented; refusing to ignore the requested memory mode",
                self.label()
            )));
        }
        if !req.config.gradient_checkpointing {
            preflight_against_budget(req.config.resolution, available_device_bytes())?;
        }
        if matches!(self.route, TrainingRoute::Ltx25 { .. }) {
            Ltx25ConditioningPlan::from_request(req)?;
        }
        Ok(())
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, Self::Aux, SamplePlan<Self::SampleState>)> {
        match &self.route {
            TrainingRoute::Ltx23 {
                root,
                gemma_override,
            } => {
                let paths = tier(root, gemma_override.as_deref())?;
                paths.validate_group_size()?;
                let connector = paths.connector_vb(DType::BF16, device)?;
                let connector_root = connector.pp("model.diffusion_model");
                let encoder = LtxTextEncoder::new(
                    paths.gemma_vb(DType::BF16, device)?,
                    connector_root.clone(),
                    connector_root,
                    &GemmaConfig::gemma_3_12b(),
                    &ConnectorConfig::ltx_2_3(),
                )?;
                let tokenizer = tokenizers::Tokenizer::from_file(paths.tokenizer_path())
                    .map_err(|e| CandleError::Msg(format!("{LABEL}: load tokenizer: {e}")))?;
                let vae = LtxVideoVae::new_with_encoder(
                    paths.vae_vb(DType::F32, device)?.pp("vae"),
                    paths.vae_encoder_vb(DType::F32, device)?.pp("vae"),
                    LATENT_CHANNELS,
                    4,
                )?;
                let tokenizer = TrainingTokenizer::Gemma3(tokenizer);
                let encoder = TrainingTextEncoder::Gemma3(Box::new(encoder));
                let edge = bucket_resolution(req.config.resolution);
                let latent_edge = edge as usize / SPATIAL_SCALE;
                let positions =
                    create_position_grid(1, latent_edge, latent_edge, DEFAULT_FPS as f32, device)?;
                let mut cached = Vec::with_capacity(req.items.len());
                for (i, item) in req.items.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    on_progress(TrainingProgress::Caching {
                        current: i as u32 + 1,
                        total: req.items.len() as u32,
                    });
                    let image = load_image_tensor(&item.image_path, edge, device)?;
                    let video = image.unsqueeze(2)?;
                    let clean = flatten_latent(&vae.encode(&video)?)?.to_dtype(DType::F32)?;
                    let context = encode_context(&tokenizer, &encoder, &item.caption, device)?;
                    cached.push(TrainingCached::Ltx23 { clean, context });
                }
                let sample_plan =
                    if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
                        let prompts: Vec<String> = req
                            .config
                            .sample_prompts
                            .iter()
                            .take(SAMPLE_PROMPT_CAP)
                            .cloned()
                            .collect();
                        let mut contexts = Vec::with_capacity(prompts.len());
                        for prompt in &prompts {
                            if req.cancel.is_cancelled() {
                                return Err(CandleError::Canceled);
                            }
                            contexts.push(SampleContext::Ltx23(encode_context(
                                &tokenizer, &encoder, prompt, device,
                            )?));
                        }
                        SamplePlan {
                            prompts,
                            state: Some(LtxSampleState {
                                contexts,
                                vae: Arc::new(vae),
                                aux: TrainingAux::Ltx23(positions.clone()),
                                latent_edge,
                            }),
                        }
                    } else {
                        SamplePlan::disabled()
                    };
                Ok((cached, TrainingAux::Ltx23(positions), sample_plan))
            }
            TrainingRoute::Ltx25 { bundle, tier, .. } => {
                // A packed tier is a complete release unit: even though this trainer consumes
                // prepared latents, validate every split VAE/upscaler/duration asset and use its
                // separate connector file rather than accidentally loading projection weights from
                // the q4 transformer.
                for component in LtxComponent::ALL {
                    bundle
                        .require(*component)
                        .map_err(|e| CandleError::Msg(e.to_string()))?;
                }
                let av_cfg = AvConfig::from_bundle(bundle)?;
                let conn_cfg = ConnectorConfig::from_bundle(bundle)?;
                let audio_conn_cfg = ConnectorConfig::audio_from_bundle(bundle)?;
                let connector_path = tier
                    .file(Ltx25Component::Connector)
                    .map_err(|error| CandleError::Msg(error.to_string()))?;
                let connector =
                    candle_gen::mmap_var_builder(&[connector_path], DType::BF16, device)?;
                let connector_root = connector.pp("model.diffusion_model");
                let encoder = Ltx25TextEncoder::from_bundle_av(
                    bundle,
                    connector.clone(),
                    connector_root,
                    &av_cfg,
                    &conn_cfg,
                    &audio_conn_cfg,
                )?;
                let tokenizer = Ltx25Tokenizer::from_packed_te_file(
                    bundle
                        .require(LtxComponent::TextEncoder)
                        .map_err(|e| CandleError::Msg(e.to_string()))?
                        .path(),
                )?;
                let tokenizer = TrainingTokenizer::Gemma4(tokenizer);
                let encoder = TrainingTextEncoder::Gemma4(Box::new(encoder));
                let conditioning = Ltx25ConditioningPlan::from_request(req)?;
                // The selected plan remains live in every cached example, including its typed
                // validation controls.  This makes the sample route consume the same contract
                // checked by public preflight rather than a descriptor-only default.
                let mut cached = Vec::with_capacity(req.items.len());
                let mut aux: Option<AvTrainingPositions> = None;
                for (i, item) in req.items.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    on_progress(TrainingProgress::Caching {
                        current: i as u32 + 1,
                        total: req.items.len() as u32,
                    });
                    let prepared = prepare_ltx25_example(item, &conditioning, device)?;
                    if let Some(existing) = &aux {
                        let same_dims = |left: &Option<Tensor>, right: &Option<Tensor>| {
                            left.as_ref().map(Tensor::dims) == right.as_ref().map(Tensor::dims)
                        };
                        if !same_dims(&existing.video, &prepared.positions.video)
                            || !same_dims(&existing.audio, &prepared.positions.audio)
                        {
                            return Err(CandleError::Msg(
                                "ltx_2_5 trainer: all items in one run must have identical prepared AV geometry".into(),
                            ));
                        }
                    } else {
                        aux = Some(prepared.positions.clone());
                    }
                    let (video_context, audio_context) =
                        encode_contexts(&tokenizer, &encoder, &item.caption, device)?;
                    cached.push(TrainingCached::Ltx25(Box::new(AvTrainingExample {
                        video_clean: prepared.video_clean,
                        audio_clean: prepared.audio_clean,
                        video_context: conditioning.video.present.then_some(video_context),
                        audio_context: conditioning.audio.present.then_some(audio_context),
                        conditioning: conditioning.clone(),
                        condition_tensors: prepared.condition_tensors,
                        condition_positions: prepared.condition_positions,
                        positions: prepared.positions,
                        video_geometry: prepared.video_geometry,
                        audio_geometry: prepared.audio_geometry,
                        condition_seed: req.config.seed,
                        sample_index: i as u64,
                    })));
                }
                let aux = aux.ok_or(CandleError::Canceled)?;
                let sample_plan = if req.config.sample_every > 0
                    && !req.config.sample_prompts.is_empty()
                {
                    let validation = build_ltx25_validation_render_plan(&conditioning.validation)?;
                    let vae_path = bundle
                        .require(LtxComponent::ConvVideoVae)
                        .map_err(|error| CandleError::Msg(error.to_string()))?
                        .path()
                        .to_path_buf();
                    let vae_vb = candle_gen::mmap_var_builder(&[vae_path], DType::F32, device)?;
                    let vae = LtxVideoVae::new(vae_vb.pp("vae"), LATENT_CHANNELS, 4)?;
                    let prompts: Vec<String> = req
                        .config
                        .sample_prompts
                        .iter()
                        .take(SAMPLE_PROMPT_CAP)
                        .cloned()
                        .collect();
                    let mut contexts = Vec::with_capacity(prompts.len());
                    for prompt in &prompts {
                        if req.cancel.is_cancelled() {
                            return Err(CandleError::Canceled);
                        }
                        let (video, audio) = encode_contexts(&tokenizer, &encoder, prompt, device)?;
                        let (negative_video, negative_audio) =
                            encode_contexts(&tokenizer, &encoder, "", device)?;
                        contexts.push(SampleContext::Ltx25 {
                            workflow: conditioning.workflow,
                            video,
                            audio,
                            negative_video,
                            negative_audio,
                            validation: validation.clone(),
                        });
                    }
                    SamplePlan {
                        prompts,
                        state: Some(LtxSampleState {
                            contexts,
                            vae: Arc::new(vae),
                            aux: TrainingAux::Ltx25(aux.clone()),
                            latent_edge: 0,
                        }),
                    }
                } else {
                    SamplePlan::disabled()
                };
                Ok((cached, TrainingAux::Ltx25(aux), sample_plan))
            }
        }
    }

    fn build_dit(&self, _req: &TrainingRequest, device: &Device) -> Result<TrainingDiT> {
        match &self.route {
            TrainingRoute::Ltx23 {
                root,
                gemma_override,
            } => {
                let paths = tier(root, gemma_override.as_deref())?;
                paths.validate_group_size()?;
                Ok(TrainingDiT::Ltx23(Box::new(LtxDiT::new(
                    paths
                        .dit_vb(DType::F32, device)?
                        .pp("model.diffusion_model"),
                    &AvConfig::ltx_2_3().video,
                )?)))
            }
            TrainingRoute::Ltx25 { bundle, tier, .. } => {
                let av_cfg = AvConfig::from_bundle(bundle)?;
                // The config-driven `ff_bias` is crucial here: taking `ltx_2_3().video` would
                // require the absent LTX-2.5 FFN bias tensors and make the QLoRA path unusable.
                let dit = tier.component_vb(Ltx25Component::Transformer, DType::F32, device)?;
                Ok(TrainingDiT::Ltx25(Box::new(AvDiT::new(
                    dit.pp("model.diffusion_model"),
                    &av_cfg,
                )?)))
            }
        }
    }

    fn micro_step(
        &self,
        dit: &TrainingDiT,
        vars: &[Var],
        cached: &Self::Cached,
        positions: &Self::Aux,
        cfg: &TrainingConfig,
        step: u32,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let sigma = sample_ltx_sigma(cfg.seed, step);
        match (dit, cached, positions) {
            (
                TrainingDiT::Ltx23(dit),
                TrainingCached::Ltx23 { clean, context },
                TrainingAux::Ltx23(positions),
            ) => {
                let noise = flow_match::sample_noise(
                    clean.dims(),
                    flow_match::noise_seed(cfg.seed, step),
                    device,
                )?;
                compute_ltx23_loss_grads(
                    dit,
                    vars,
                    clean,
                    context,
                    positions,
                    sigma,
                    &noise,
                    flow_match::is_mae(cfg),
                    cfg.gradient_checkpointing,
                )
            }
            (
                TrainingDiT::Ltx25(dit),
                TrainingCached::Ltx25(cached),
                TrainingAux::Ltx25(positions),
            ) => {
                let video_noise = if cached.conditioning.video.is_generated {
                    cached
                        .video_clean
                        .as_ref()
                        .map(|clean| {
                            flow_match::sample_noise(
                                clean.dims(),
                                flow_match::noise_seed(cfg.seed, step),
                                device,
                            )
                        })
                        .transpose()?
                } else {
                    None
                };
                let audio_noise = if cached.conditioning.audio.is_generated {
                    cached
                        .audio_clean
                        .as_ref()
                        .map(|clean| {
                            flow_match::sample_noise(
                                clean.dims(),
                                flow_match::noise_seed(cfg.seed, step).wrapping_add(2),
                                device,
                            )
                        })
                        .transpose()?
                } else {
                    None
                };
                compute_av_loss_grads(
                    dit,
                    vars,
                    cached,
                    positions,
                    sigma,
                    video_noise.as_ref(),
                    audio_noise.as_ref(),
                    flow_match::is_mae(cfg),
                )
            }
            _ => Err(CandleError::Msg(
                "ltx trainer: cached route and trainable DiT disagree".into(),
            )),
        }
    }

    fn render_sample(
        &self,
        dit: &TrainingDiT,
        state: &LtxSampleState,
        index: usize,
        _cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        match (dit, &state.contexts[index], &state.aux) {
            (
                TrainingDiT::Ltx23(dit),
                SampleContext::Ltx23(context),
                TrainingAux::Ltx23(positions),
            ) => {
                let edge = state.latent_edge;
                let latent = crate::pipeline::create_noise(seed, 1, edge, edge, &self.device)?;
                let noise = flatten_latent(&latent)?;
                let cancel = CancelFlag::new();
                let mut progress = |_: Progress| {};
                let out = candle_gen::run_flow_sampler(
                    None,
                    TimestepConvention::Sigma,
                    &STAGE1_SIGMAS,
                    noise,
                    seed,
                    &cancel,
                    &mut progress,
                    None,
                    |x, sigma| Ok(dit.forward(x, sigma as f64, context, positions)?),
                )?;
                let latent = unflatten_latent(&out.to_dtype(DType::F32)?, 1, edge, edge)?;
                frames_to_images(&state.vae.decode(&latent)?)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        CandleError::Msg(format!("{LABEL}: preview decode produced no frame"))
                    })
            }
            (
                TrainingDiT::Ltx25(dit),
                SampleContext::Ltx25 {
                    workflow,
                    video,
                    audio,
                    negative_video,
                    negative_audio,
                    validation,
                },
                TrainingAux::Ltx25(_),
            ) => {
                // A validation preview is an actual non-distilled Dev trajectory. The workflow
                // remains part of the typed cache identity even though a prompt-only preview has
                // no per-item IC tensor to attach.
                let _selected_workflow = workflow;
                let (frames, height, width) =
                    (validation.frames, validation.height, validation.width);
                let (t_lat, h_lat, w_lat) = crate::pipeline::latent_dims(frames, width, height);
                let video_grid =
                    create_position_grid(t_lat, h_lat, w_lat, validation.fps as f32, &self.device)?;
                let video_noise =
                    crate::pipeline::create_noise(seed, t_lat, h_lat, w_lat, &self.device)?;
                let token_state =
                    crate::conditioning::VideoTokenState::base(&video_noise, &video_grid)?;
                let video_guider = GuiderParams {
                    cfg_scale: validation.video_cfg_scale,
                    stg_scale: validation.video_stg_scale,
                    stg_blocks: &[],
                    rescale_scale: validation.guidance_rescale,
                    modality_scale: validation.video_modality_guidance_scale,
                };
                let audio_guider = GuiderParams {
                    cfg_scale: validation.audio_cfg_scale,
                    stg_scale: validation.audio_stg_scale,
                    stg_blocks: &[],
                    rescale_scale: validation.guidance_rescale,
                    modality_scale: validation.audio_modality_guidance_scale,
                };
                let cancel = CancelFlag::new();
                let mut forwards = || Ok(());
                let mut progress = |_: Progress| {};
                let generated_state = if validation.generate_audio {
                    let audio_frames =
                        compute_audio_frames(frames as usize, validation.fps as f64).max(1);
                    let audio_grid = create_audio_position_grid(audio_frames, &self.device)?;
                    let audio_noise = crate::pipeline::create_audio_noise(
                        seed.wrapping_add(1),
                        audio_frames,
                        &self.device,
                    )?;
                    let (state, _audio) = crate::pipeline::denoise_av_dev_conditioned(
                        dit,
                        &token_state,
                        &audio_noise,
                        video,
                        audio,
                        negative_video,
                        negative_audio,
                        audio_frames,
                        &audio_grid,
                        &validation.sigmas,
                        &validation.stg_blocks,
                        video_guider,
                        audio_guider,
                        &cancel,
                        &mut forwards,
                        &mut progress,
                    )?;
                    state
                } else {
                    crate::pipeline::denoise_video_dev_conditioned(
                        dit,
                        &token_state,
                        video,
                        negative_video,
                        &validation.sigmas,
                        &validation.stg_blocks,
                        video_guider,
                        &cancel,
                        &mut forwards,
                        &mut progress,
                    )?
                };
                let generated =
                    generated_state
                        .latent
                        .narrow(1, 0, generated_state.target_tokens)?;
                let latent =
                    unflatten_latent(&generated.to_dtype(DType::F32)?, t_lat, h_lat, w_lat)?;
                frames_to_images(&state.vae.decode(&latent)?)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        CandleError::Msg(format!("{LABEL}: preview decode produced no frame"))
                    })
            }
            _ => Err(CandleError::Msg(
                "ltx trainer: preview route does not match its cached state".into(),
            )),
        }
    }

    fn save(&self, set: &LoraSet, path: &std::path::Path) -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("lora_rank".into(), set.rank.to_string());
        metadata.insert("lora_alpha".into(), set.alpha.to_string());
        metadata.insert("family".into(), "ltx".into());
        metadata.insert("baseModel".into(), self.descriptor.id.to_string());
        flow_match::save_adapter(set, &metadata, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::TrainingItem;
    use serde_json::json;
    use std::path::PathBuf;

    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                PathBuf::from("image.png"),
                "caption".into(),
            )],
            config: TrainingConfig {
                steps: 4,
                train_dtype: "f32".into(),
                timestep_type: "sigmoid".into(),
                timestep_bias: "high_noise".into(),
                ..Default::default()
            },
            output_dir: PathBuf::from("out"),
            file_name: "ltx.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    fn validation_json() -> serde_json::Value {
        json!({
            "width": 960, "height": 544, "frames": 89, "fps": 24, "steps": 30,
            "videoCfgScale": 3.0, "audioCfgScale": 7.0,
            "videoStgScale": 1.0, "audioStgScale": 1.0, "stgBlocks": [28],
            "guidanceRescale": 0.7, "videoModalityGuidanceScale": 3.0,
            "audioModalityGuidanceScale": 3.0, "generateAudio": true,
        })
    }

    fn canonical_conditions(name: &str, video: bool) -> serde_json::Value {
        let conditions = match (name, video) {
            ("i2v_lora", true) => json!([{"type":"firstFrame","probability":0.5}]),
            ("video_extend_lora", true) | ("audio_extend_lora", false) => {
                json!([{"type":"prefix","probability":1.0,"temporalBoundary":8}])
            }
            ("video_suffix_lora", true) | ("audio_suffix_lora", false) => {
                json!([{"type":"suffix","probability":1.0,"temporalBoundary":8}])
            }
            ("video_inpainting_lora", true) | ("audio_inpainting_lora", false) => {
                json!([{"type":"mask","probability":1.0,"tensorKey":"mask"}])
            }
            ("video_outpainting_lora", true) => json!([{
                "type":"spatialCrop","probability":1.0,"spatialRegion":[0,0,288,576]
            }]),
            ("av2av_ic_lora", _) | ("a2a_ic_lora", false) => {
                json!([{"type":"reference","probability":1.0,"tensorKey":"reference"}])
            }
            ("v2v_ic_lora", true) => json!([
                {"type":"reference","probability":1.0,"tensorKey":"reference"},
                {"type":"firstFrame","probability":0.2},
            ]),
            _ => json!([]),
        };
        json!({"isGenerated": workflow_generated(name, video), "conditions": conditions})
    }

    fn workflow_generated(name: &str, video: bool) -> bool {
        matches!(
            (name, video),
            ("i2v_lora", _)
                | ("t2v_lora", _)
                | ("v2a_lora", false)
                | ("a2v_lora", true)
                | ("t2a_lora", false)
                | ("video_extend_lora", true)
                | ("video_inpainting_lora", true)
                | ("video_outpainting_lora", true)
                | ("video_suffix_lora", true)
                | ("video_extend_lora", false)
                | ("video_suffix_lora", false)
                | ("audio_extend_lora", false)
                | ("audio_inpainting_lora", false)
                | ("audio_suffix_lora", false)
                | ("av2av_ic_lora", _)
                | ("v2v_ic_lora", true)
                | ("a2a_ic_lora", false)
        )
    }

    fn ltx25_request(workflow: &str) -> TrainingRequest {
        let mut req = request();
        let selected = Ltx25Workflow::parse(workflow).unwrap();
        let (video, audio) = selected.canonical();
        let mut options = serde_json::Map::from_iter([
            ("ltxWorkflow".into(), json!(workflow)),
            ("ltxValidation".into(), validation_json()),
        ]);
        if video.present {
            options.insert("ltxVideo".into(), canonical_conditions(workflow, true));
        }
        if audio.present {
            options.insert("ltxAudio".into(), canonical_conditions(workflow, false));
        }
        req.config.model_options = options;
        req
    }

    fn write_prepared_bundle(path: &std::path::Path, schema_version: &str) {
        let video_bytes = 128 * 4;
        let audio_bytes = 8 * 16 * 4;
        let header = json!({
            "__metadata__": {
                "schemaVersion": schema_version,
                "videoShape": "[1,128,1,1,1]",
                "audioShape": "[1,8,1,16]",
                "fps": "24",
            },
            "video_latents": {
                "dtype": "F32", "shape": [1, 128, 1, 1, 1],
                "data_offsets": [0, video_bytes],
            },
            "audio_latents": {
                "dtype": "F32", "shape": [1, 8, 1, 16],
                "data_offsets": [video_bytes, video_bytes + audio_bytes],
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + video_bytes + audio_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_audio_only_prepared_bundle(path: &std::path::Path) {
        let audio_bytes = 8 * 16 * 4;
        let header = json!({
            "__metadata__": {"schemaVersion": "ltx-prepared-v1", "audioShape": "[1,8,1,16]"},
            "audio_latents": {
                "dtype": "F32", "shape": [1, 8, 1, 16], "data_offsets": [0, audio_bytes],
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + audio_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_malformed_duration_bundle(path: &std::path::Path) {
        let video_bytes = 128 * 4;
        let audio_bytes = 2 * 8 * 16 * 4;
        let header = json!({
            "__metadata__": {
                "schemaVersion": "ltx-prepared-v1",
                "videoShape": "[1,128,1,1,1]",
                "audioShape": "[1,8,2,16]",
                "fps": "24",
            },
            "video_latents": {
                "dtype": "F32", "shape": [1, 128, 1, 1, 1], "data_offsets": [0, video_bytes],
            },
            "audio_latents": {
                "dtype": "F32", "shape": [1, 8, 2, 16], "data_offsets": [video_bytes, video_bytes + audio_bytes],
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + video_bytes + audio_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_f64_fps_duration_bundle(path: &std::path::Path) {
        // `9 / 64.28571428571429 * 25` lies on the rounding boundary: retaining the metadata as
        // f64 produces three audio frames, while a premature f32 cast predicts four.
        let fps = 64.285_714_285_714_29_f64;
        let video_frames = 2usize;
        let output_frames = (video_frames - 1) * 8 + 1;
        let audio_frames = compute_audio_frames(output_frames, fps);
        assert_eq!(audio_frames, 3);
        let video_bytes = 128 * video_frames * 4;
        let audio_bytes = 8 * audio_frames * 16 * 4;
        let header = json!({
            "__metadata__": {
                "schemaVersion": "ltx-prepared-v1",
                "videoShape": "[1,128,2,1,1]",
                "audioShape": format!("[1,8,{audio_frames},16]"),
                "fps": fps.to_string(),
            },
            "video_latents": {
                "dtype": "F32", "shape": [1, 128, 2, 1, 1],
                "data_offsets": [0, video_bytes],
            },
            "audio_latents": {
                "dtype": "F32", "shape": [1, 8, audio_frames, 16],
                "data_offsets": [video_bytes, video_bytes + audio_bytes],
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + video_bytes + audio_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn attach_prepared_bundle(req: &mut TrainingRequest, path: &std::path::Path) {
        req.items[0].model_options.insert(
            "ltxPreparedBundlePath".into(),
            json!(path.to_string_lossy()),
        );
    }

    #[test]
    fn uniform_sigma_is_deterministic_strictly_interior_and_config_invariant() {
        let lower = f32::from_bits(1e-3f32.to_bits() + 1) as f64;
        let upper = f32::from_bits((1.0f32 - 1e-3).to_bits() - 1) as f64;
        let mut endpoint_hits = 0;
        for seed in 0..20_000 {
            let sigma = sample_ltx_sigma(seed, 1);
            assert!(sigma >= lower && sigma <= upper, "{sigma}");
            endpoint_hits += usize::from(sigma == lower || sigma == upper);
        }
        assert!(
            endpoint_hits <= 1,
            "affine sampling must not pile clamped mass onto inward endpoints: {endpoint_hits}"
        );
        for seed in [0, 1, 42, u64::MAX] {
            for step in 1..20 {
                let a = sample_ltx_sigma(seed, step);
                let b = sample_ltx_sigma(seed, step);
                assert_eq!(a, b);
                assert!(a > 1e-3 && a < 1.0 - 1e-3, "{a}");
            }
        }
        // The function accepts no timestep_type/bias: family behavior cannot shift with config.
        let mut a = request();
        let mut b = request();
        b.config.timestep_type = "weighted".into();
        b.config.timestep_bias = "low_noise".into();
        assert_eq!(
            sample_ltx_sigma(a.config.seed, 3),
            sample_ltx_sigma(b.config.seed, 3)
        );
        a.config.timestep_type = "uniform".into();
        assert_eq!(
            sample_ltx_sigma(a.config.seed, 3),
            sample_ltx_sigma(b.config.seed, 3)
        );
    }

    #[test]
    fn training_tokenization_truncates_and_masks_at_128_tokens() {
        let (ids, mask) = pad_training_ids((0..200).collect());
        assert_eq!(ids.len(), TRAIN_TEXT_MAX_LENGTH);
        assert_eq!(mask.len(), TRAIN_TEXT_MAX_LENGTH);
        assert_eq!(ids, (0..128).collect::<Vec<_>>());
        assert!(mask.iter().all(|&value| value == 1));

        let (ids, mask) = pad_training_ids(vec![7, 8, 9]);
        assert_eq!(&ids[125..], &[7, 8, 9]);
        assert!(ids[..125].iter().all(|&value| value == 0));
        assert!(mask[..125].iter().all(|&value| value == 0));
        assert_eq!(&mask[125..], &[1, 1, 1]);
    }

    #[test]
    fn flow_recipe_and_losses_match_reference() {
        let dev = Device::Cpu;
        let clean = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target) = flow_match::build_batch(&clean, &noise, 0.25).unwrap();
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        let prediction = Tensor::zeros((1, 2), DType::BF16, &dev).unwrap();
        let mse = velocity_loss(&prediction, &target, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mae = velocity_loss(&prediction, &target, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((mse - 8.5).abs() < 1e-6);
        assert!((mae - 2.5).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_bf16_lokr_and_bad_core_configs() {
        let mut req = request();
        validate_ltx_request(&req, LABEL).unwrap();
        req.config.train_dtype = "bf16".into();
        assert!(validate_ltx_request(&req, LABEL)
            .unwrap_err()
            .to_string()
            .contains("requires f32"));
        req.config.train_dtype = "f32".into();
        req.config.network_type = NetworkType::Lokr;
        assert!(validate_ltx_request(&req, LABEL)
            .unwrap_err()
            .to_string()
            .contains("LoRA-only"));
        req.config.network_type = NetworkType::Lora;
        req.config.rank = 0;
        assert!(validate_ltx_request(&req, LABEL).is_err());
    }

    #[test]
    fn memory_projection_and_guard_are_testable_before_cache() {
        assert!((projected_dense_peak_gb(1024) - 42.6024).abs() < 1e-4);
        preflight_against_budget(512, 64 * 1024 * 1024 * 1024).unwrap();
        let err = preflight_against_budget(2048, 16 * 1024 * 1024 * 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("gradient checkpointing"), "{err}");
    }

    #[test]
    fn descriptor_and_default_targets_are_exact() {
        let d = trainer_descriptor();
        assert_eq!(d.id, TRAINER_ID);
        assert_eq!(d.backend, "candle");
        assert!(d.supports_lora);
        assert!(!d.supports_lokr);
        assert_eq!(LTX_ATTN_TARGETS, ["to_q", "to_k", "to_v", "to_out.0"]);
    }

    #[test]
    fn ltx25_descriptor_and_validation_preset_are_not_aliased_to_ltx23() {
        let d = trainer_descriptor_25();
        assert_eq!(d.id, MODEL_25_ID);
        assert_eq!(d.backend, "candle");
        assert!(d.supports_lora);
        assert_eq!(
            LTX25_VALIDATION_DEFAULTS,
            Ltx25ValidationDefaults {
                width: 960,
                height: 544,
                frames: 89,
                fps: 24,
                steps: 30,
                video_cfg_scale: 3.0,
                audio_cfg_scale: 7.0,
                video_stg_scale: 1.0,
                audio_stg_scale: 1.0,
                stg_blocks: &[28],
                guidance_rescale: 0.7,
                video_modality_guidance_scale: 3.0,
                audio_modality_guidance_scale: 3.0,
                generate_video: true,
                generate_audio: true,
            }
        );
        assert_ne!(
            LTX25_VALIDATION_DEFAULTS.steps,
            STAGE1_SIGMAS.len() as u32 - 1
        );
    }

    #[test]
    fn ltx25_workflow_contract_preserves_all_upstream_modes() {
        assert_eq!(LTX25_WORKFLOWS.len(), 15);
        for required in [
            "i2v_lora",
            "t2v_lora",
            "v2a_lora",
            "a2v_lora",
            "t2a_lora",
            "video_extend_lora",
            "video_inpainting_lora",
            "video_outpainting_lora",
            "video_suffix_lora",
            "audio_extend_lora",
            "audio_inpainting_lora",
            "audio_suffix_lora",
            "av2av_ic_lora",
            "v2v_ic_lora",
            "a2a_ic_lora",
        ] {
            assert!(
                LTX25_WORKFLOWS
                    .iter()
                    .any(|workflow| workflow.id() == required),
                "missing {required}"
            );
        }
    }

    #[test]
    fn every_upstream_workflow_requires_its_canonical_av_plan() {
        for workflow in LTX25_WORKFLOWS {
            Ltx25ConditioningPlan::from_request(&ltx25_request(workflow.id())).unwrap_or_else(
                |error| panic!("canonical {} unexpectedly rejected: {error}", workflow.id()),
            );
        }
        let mut bad = ltx25_request("t2v_lora");
        bad.config.model_options["ltxAudio"]["isGenerated"] = json!(false);
        assert!(
            Ltx25ConditioningPlan::from_request(&bad).is_err(),
            "T2V must be joint AV generation"
        );
        let mut missing = ltx25_request("video_outpainting_lora");
        missing.config.model_options["ltxVideo"]["conditions"] = json!([]);
        assert!(
            Ltx25ConditioningPlan::from_request(&missing).is_err(),
            "crop workflow must not collapse to plain T2V"
        );
    }

    #[test]
    fn absent_workflow_streams_are_optional_but_never_synthetic() {
        for workflow in LTX25_WORKFLOWS {
            let mut req = ltx25_request(workflow.id());
            let (video_rule, audio_rule) = workflow.canonical();
            if !video_rule.present {
                req.config.model_options.remove("ltxVideo");
            }
            if !audio_rule.present {
                req.config.model_options.remove("ltxAudio");
            }
            let plan = Ltx25ConditioningPlan::from_request(&req).unwrap_or_else(|error| {
                panic!(
                    "{} did not retain its canonical absent stream: {error}",
                    workflow.id()
                )
            });
            assert_eq!(
                plan.video.present,
                video_rule.present,
                "{} video presence",
                workflow.id()
            );
            assert_eq!(
                plan.audio.present,
                audio_rule.present,
                "{} audio presence",
                workflow.id()
            );
            if !video_rule.present {
                assert!(plan.video.conditions.is_empty());
                assert!(!plan.video.is_generated);
            }
            if !audio_rule.present {
                assert!(plan.audio.conditions.is_empty());
                assert!(!plan.audio.is_generated);
            }
        }
    }

    #[test]
    fn intrinsic_masks_are_seeded_and_project_spatial_regions_through_video_geometry() {
        let dev = Device::Cpu;
        let crop = ModalityConditioningPlan {
            present: true,
            is_generated: true,
            conditions: vec![Ltx25Condition {
                kind: IntrinsicCondition::SpatialCrop,
                probability: 1.0,
                temporal_boundary: None,
                spatial_region: Some([32, 32, 64, 64]),
                tensor_key: None,
                spatial_scale_factor: Some(32),
                temporal_scale_factor: None,
                reference: false,
            }],
        };
        let geometry = LtxTokenGeometry {
            frames: 2,
            height: 3,
            width: 3,
            spatial_scale: 32,
        };
        let (generated, _, _) =
            modality_masks(&crop, &HashMap::new(), geometry, 7, 0, &dev).unwrap();
        let values = generated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for frame in 0..2 {
            assert_eq!(
                values[frame * 9 + 4],
                0.0,
                "central 32px crop must condition each frame"
            );
            assert_eq!(values[frame * 9], 1.0, "outside crop stays generated");
        }
        let first = ModalityConditioningPlan {
            present: true,
            is_generated: true,
            conditions: vec![Ltx25Condition {
                kind: IntrinsicCondition::First,
                probability: 0.5,
                temporal_boundary: None,
                spatial_region: None,
                tensor_key: None,
                spatial_scale_factor: None,
                temporal_scale_factor: None,
                reference: false,
            }],
        };
        let active = (0..1024)
            .find(|seed| condition_draw(*seed, 0, 0) < 0.5)
            .unwrap();
        let inactive = (0..1024)
            .find(|seed| condition_draw(*seed, 0, 0) >= 0.5)
            .unwrap();
        let active_mask = modality_masks(&first, &HashMap::new(), geometry, active, 0, &dev)
            .unwrap()
            .0;
        let inactive_mask = modality_masks(&first, &HashMap::new(), geometry, inactive, 0, &dev)
            .unwrap()
            .0;
        assert_ne!(
            active_mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            inactive_mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "condition probability must drive a per-example Bernoulli, not a deterministic >0 switch"
        );
    }

    #[test]
    fn validation_controls_accept_bounded_overrides_and_reject_invalid_geometry() {
        let mut req = ltx25_request("t2v_lora");
        req.config.model_options["ltxValidation"]["videoCfgScale"] = json!(4.0);
        req.config.model_options["ltxValidation"]["audioStgScale"] = json!(2.0);
        req.config.model_options["ltxValidation"]["stgBlocks"] = json!([12]);
        let plan = Ltx25ConditioningPlan::from_request(&req).unwrap();
        assert_eq!(plan.validation.video_cfg_scale, 4.0);
        assert_eq!(plan.validation.audio_stg_scale, 2.0);
        assert_eq!(plan.validation.stg_blocks, vec![12]);
        req.config.model_options["ltxValidation"]["frames"] = json!(88);
        assert!(Ltx25ConditioningPlan::from_request(&req).is_err());
    }

    #[test]
    fn validation_defaults_partial_values_and_dynamic_stg_blocks_are_executable() {
        let mut req = ltx25_request("t2v_lora");
        req.config.model_options.remove("ltxValidation");
        let defaults = Ltx25ConditioningPlan::from_request(&req).unwrap();
        assert_eq!(defaults.validation, LTX25_VALIDATION_DEFAULTS.into());

        req.config.model_options.insert(
            "ltxValidation".into(),
            json!({"steps": 31, "stgBlocks": [4, 28], "guidanceRescale": 0.5, "generateAudio": false}),
        );
        let plan = Ltx25ConditioningPlan::from_request(&req).unwrap();
        let render = build_ltx25_validation_render_plan(&plan.validation).unwrap();
        assert_eq!(render.sigmas.len(), 32);
        assert_eq!(render.stg_blocks, vec![4, 28]);
        assert_eq!(render.guidance_rescale, 0.5);
        assert!(!render.generate_audio);
        let source = include_str!("training.rs");
        assert!(source.contains("if validation.generate_audio {"));
        assert!(source.contains("denoise_video_dev_conditioned("));

        req.config.model_options["ltxValidation"]["stgBlocks"] = json!([28, 28]);
        assert!(Ltx25ConditioningPlan::from_request(&req).is_err());
        req.config.model_options["ltxValidation"]["stgBlocks"] = json!("28");
        assert!(Ltx25ConditioningPlan::from_request(&req).is_err());
        req.config.model_options["ltxValidation"]["stgBlocks"] = json!([28]);
        req.config.model_options["ltxValidation"]["generateAudio"] = json!("true");
        assert!(Ltx25ConditioningPlan::from_request(&req).is_err());
    }

    #[test]
    fn condition_schema_rejects_irrelevant_scale_fields() {
        assert!(condition_from_value(&json!({
            "type":"firstFrame", "probability":1.0, "spatialScaleFactor":32
        }))
        .is_err());
        assert!(condition_from_value(&json!({
            "type":"mask", "probability":1.0, "tensorKey":"m", "temporalScaleFactor":4
        }))
        .is_err());
        assert!(condition_from_value(&json!({
            "type":"reference", "probability":1.0, "tensorKey":"", "spatialScaleFactor":32
        }))
        .is_err());
    }

    #[test]
    fn masked_losses_normalize_token_masks_by_batch_and_channels() {
        let dev = Device::Cpu;
        let prediction = Tensor::ones((2, 2, 3), DType::F32, &dev).unwrap();
        let target = Tensor::zeros((2, 2, 3), DType::F32, &dev).unwrap();
        let mask = Tensor::from_vec(vec![1f32, 0.0], (1, 2, 1), &dev).unwrap();
        let mse = masked_velocity_loss(&prediction, &target, &mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mae = masked_mae_loss(&prediction, &target, &mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((mse - 1.0).abs() < 1e-6, "{mse}");
        assert!((mae - 1.0).abs() < 1e-6, "{mae}");
    }

    #[test]
    fn public_ltx25_preflight_requires_valid_defaults_and_every_prepared_pack() {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join("prepared.safetensors");
        write_prepared_bundle(&pack, "ltx-prepared-v1");

        let mut valid = ltx25_request("t2v_lora");
        attach_prepared_bundle(&mut valid, &pack);
        validate_ltx25_training_request(&valid).unwrap();

        let mut invalid_default = valid.clone();
        invalid_default.config.model_options["ltxValidation"]["frames"] = json!(88);
        assert!(validate_ltx25_training_request(&invalid_default).is_err());

        let missing_bundle = ltx25_request("t2v_lora");
        let error = validate_ltx25_training_request(&missing_bundle)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ltxPreparedBundlePath"), "{error}");

        write_prepared_bundle(&pack, "wrong-schema");
        let error = validate_ltx25_training_request(&valid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("schemaVersion=ltx-prepared-v1"), "{error}");
    }

    #[test]
    fn canonical_present_modalities_default_while_absent_or_contradictory_overrides_refuse() {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join("prepared.safetensors");
        write_prepared_bundle(&pack, "ltx-prepared-v1");

        // A normal t2v request carries only its workflow selection.  Both present modalities and
        // their empty canonical condition sets are synthesized before the public pack preflight.
        let mut t2v = request();
        t2v.config.model_options =
            serde_json::from_value(json!({"ltxWorkflow":"t2v_lora"})).unwrap();
        attach_prepared_bundle(&mut t2v, &pack);
        validate_ltx25_training_request(&t2v).unwrap();
        let plan = Ltx25ConditioningPlan::from_request(&t2v).unwrap();
        assert!(plan.video.present && plan.video.is_generated && plan.video.conditions.is_empty());
        assert!(plan.audio.present && plan.audio.is_generated && plan.audio.conditions.is_empty());

        // A supplied present modality may omit `conditions`; it still resolves to the workflow
        // default instead of imposing an artificial field requirement.
        let mut omitted_conditions = t2v.clone();
        omitted_conditions
            .config
            .model_options
            .insert("ltxVideo".into(), json!({"isGenerated": true}));
        let plan = Ltx25ConditioningPlan::from_request(&omitted_conditions).unwrap();
        assert!(plan.video.conditions.is_empty());

        // This cannot be a vacuous empty-list default: i2v's canonical video stream carries
        // first-frame conditioning, which a supplied object may omit while retaining its exact
        // workflow-defined values.
        let mut i2v = request();
        i2v.config.model_options = serde_json::from_value(json!({
            "ltxWorkflow": "i2v_lora",
            "ltxVideo": {"isGenerated": true}
        }))
        .unwrap();
        let plan = Ltx25ConditioningPlan::from_request(&i2v).unwrap();
        assert_eq!(plan.video.conditions.len(), 1);
        assert_eq!(plan.video.conditions[0].kind, IntrinsicCondition::First);
        assert_eq!(plan.video.conditions[0].probability, 0.5);

        let mut t2a = request();
        t2a.config.model_options = serde_json::from_value(json!({
            "ltxWorkflow":"t2a_lora",
            "ltxVideo":{"isGenerated":false,"conditions":[]}
        }))
        .unwrap();
        assert!(Ltx25ConditioningPlan::from_request(&t2a).is_err());

        let mut contradictory = t2v;
        contradictory
            .config
            .model_options
            .insert("ltxAudio".into(), json!({"isGenerated": false}));
        assert!(Ltx25ConditioningPlan::from_request(&contradictory).is_err());
    }

    #[test]
    fn public_ltx25_preflight_accepts_absent_stream_packs_and_rejects_bad_duration_and_alpha() {
        let dir = tempfile::tempdir().unwrap();
        let audio_only = dir.path().join("audio-only.safetensors");
        write_audio_only_prepared_bundle(&audio_only);
        let mut audio_request = ltx25_request("t2a_lora");
        audio_request.config.model_options.remove("ltxVideo");
        attach_prepared_bundle(&mut audio_request, &audio_only);
        validate_ltx25_training_request(&audio_request).unwrap();

        let malformed = dir.path().join("bad-duration.safetensors");
        write_malformed_duration_bundle(&malformed);
        let mut joint = ltx25_request("t2v_lora");
        attach_prepared_bundle(&mut joint, &malformed);
        let error = validate_ltx25_training_request(&joint)
            .unwrap_err()
            .to_string();
        assert!(error.contains("audio frames"), "{error}");

        joint.config.alpha = 0.0;
        let error = validate_ltx25_training_request(&joint)
            .unwrap_err()
            .to_string();
        assert!(error.contains("alpha"), "{error}");
    }

    #[test]
    fn prepared_duration_mapping_keeps_metadata_fps_as_f64() {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join("f64-fps.safetensors");
        write_f64_fps_duration_bundle(&pack);
        let mut req = ltx25_request("t2v_lora");
        attach_prepared_bundle(&mut req, &pack);
        validate_ltx25_training_request(&req).unwrap();
    }

    #[test]
    fn reference_condition_appends_clean_context_and_zero_loss_tokens() {
        let dev = Device::Cpu;
        let plan = ModalityConditioningPlan {
            present: true,
            is_generated: true,
            conditions: vec![Ltx25Condition {
                kind: IntrinsicCondition::None,
                probability: 1.0,
                temporal_boundary: None,
                spatial_region: None,
                tensor_key: Some("ref".into()),
                spatial_scale_factor: None,
                temporal_scale_factor: None,
                reference: true,
            }],
        };
        let input = Tensor::zeros((1, 2, 1), DType::F32, &dev).unwrap();
        let target = Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap();
        let loss = Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap();
        let positions = Tensor::zeros((1, 3, 2, 2), DType::F32, &dev).unwrap();
        let reference = Tensor::from_vec(vec![7f32], (1, 1, 1), &dev).unwrap();
        let reference_positions = Tensor::ones((1, 3, 1, 2), DType::F32, &dev).unwrap();
        let (input, target, loss, positions) = append_active_reference(
            &plan,
            &HashMap::from([("ref".into(), reference)]),
            &HashMap::from([("ref".into(), reference_positions)]),
            11,
            0,
            input,
            target,
            loss,
            positions,
        )
        .unwrap();
        assert_eq!(input.dims(), &[1, 3, 1]);
        assert_eq!(input.to_vec3::<f32>().unwrap()[0][2][0], 7.0);
        assert_eq!(target.to_vec3::<f32>().unwrap()[0][2][0], 0.0);
        assert_eq!(loss.to_vec3::<f32>().unwrap()[0][2][0], 0.0);
        assert_eq!(
            masked_token_timesteps(&loss, 0.37)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![0.37, 0.37, 0.0]],
            "reference tokens must be appended clean with timestep zero"
        );
        assert_eq!(positions.dims(), &[1, 3, 3, 2]);
    }

    #[test]
    fn intrinsic_conditioning_uses_clean_target_and_reference_only_appends() {
        let dev = Device::Cpu;
        let plan = ModalityConditioningPlan {
            present: true,
            is_generated: true,
            conditions: vec![
                Ltx25Condition {
                    kind: IntrinsicCondition::First,
                    probability: 1.0,
                    temporal_boundary: None,
                    spatial_region: None,
                    tensor_key: None,
                    spatial_scale_factor: None,
                    temporal_scale_factor: None,
                    reference: false,
                },
                Ltx25Condition {
                    kind: IntrinsicCondition::None,
                    probability: 1.0,
                    temporal_boundary: None,
                    spatial_region: None,
                    tensor_key: Some("ref".into()),
                    spatial_scale_factor: Some(32),
                    temporal_scale_factor: Some(8),
                    reference: true,
                },
            ],
        };
        let noisy = Tensor::zeros((1, 2, 1), DType::F32, &dev).unwrap();
        let clean = Tensor::from_vec(vec![2f32, 2.0], (1, 2, 1), &dev).unwrap();
        let reference = Tensor::from_vec(vec![7f32, 7.0], (1, 2, 1), &dev).unwrap();
        let (input, loss) = apply_modality_conditioning(
            &noisy,
            &clean,
            &plan,
            &HashMap::from([("ref".into(), reference)]),
            LtxTokenGeometry {
                frames: 2,
                height: 1,
                width: 1,
                spatial_scale: 32,
            },
            0,
            0,
        )
        .unwrap();
        assert_eq!(
            input.to_vec3::<f32>().unwrap(),
            vec![vec![vec![2.0], vec![0.0]]]
        );
        assert_eq!(
            masked_token_timesteps(&loss, 0.37)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![0.0, 0.37]],
            "intrinsic clean tokens must not receive the sampled sigma"
        );
    }

    #[test]
    fn tensor_backed_conditions_and_dev_q4_tier_are_fail_closed() {
        assert!(condition_from_value(&json!({
            "type":"firstFrame", "probability":1.0, "tensorKey":"wrong"
        }))
        .is_err());
        assert!(condition_from_value(&json!({"type":"mask", "probability":1.0})).is_err());
        let source = PathBuf::from("synthetic-split-model.json");
        let mut manifest = crate::tier::Ltx25TierManifest::from_value(
            &source,
            &json!({
                "tier":"q4", "model_version":"2.5.0", "quantized":true,
                "quantization_bits":4, "quantization_group_size":64,
                "component_detail":[]
            }),
        )
        .unwrap();
        validate_ltx25_dev_q4_manifest(&manifest).unwrap();
        manifest.tier = "q8".into();
        assert!(validate_ltx25_dev_q4_manifest(&manifest).is_err());
        manifest.tier = "q4".into();
        manifest.quantized = false;
        assert!(validate_ltx25_dev_q4_manifest(&manifest).is_err());
        manifest.quantized = true;
        manifest.quant.bits = 8;
        assert!(validate_ltx25_dev_q4_manifest(&manifest).is_err());
        manifest.quant.bits = 4;
        manifest.model_version = "2.3.0".into();
        assert!(validate_ltx25_dev_q4_manifest(&manifest).is_err());
    }
}
