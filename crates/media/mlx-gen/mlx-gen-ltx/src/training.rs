//! LTX-2.3/2.5 LoRA **training** in pure Rust on mlx-rs. The original 2.3 route is the Rust port of
//! SceneWorks' pure-MLX `_LtxMlxLoraBackend` / `LtxMlxLoraTrainer` (`training_adapters.py:3249-3628`),
//! realizing the core [`Trainer`] contract (epic 3039). Retiring the Python version removes the last
//! Python-MLX trainer (blocks sc-3049 cutover → sc-3242 `mlx-video` drop).
//!
//! Built on the same functional-autograd mechanism the Z-Image spike proved (sc-3042) and the
//! sc-3043 runtime glue, but LTX has its **own** adapter seam: its [`crate::transformer::Linear`]
//! carries a per-pass [`LoraStack`](crate::transformer) (not the core `AdaptableLinear`), so this
//! module uses the LTX-local `Linear::set_train_lora` training seam and its own target
//! enumeration / save, while reusing the core [`LoraParams`] + grad-accumulation helpers and the
//! runtime (schedule / dataset / checkpoint).
//!
//! **What is LTX-specific:**
//!   * **2.3 video-only forward over `LtxDiT`.** The reference loads the AV model and trains with
//!     `audio=None`; [`LtxDiT`] is exactly that video-only reduction (the AV checkpoint embeds the
//!     same `transformer_blocks.{i}` video blocks), and the trained video-attention adapter reloads
//!     onto the AvDiT inference path unchanged.
//!   * **2.5 flexible AV forward over `AvDiT`.** All fifteen upstream workflows select generated,
//!     frozen or absent video/audio modalities, intrinsic and tensor-backed conditions, and separate
//!     loss masks. The dev/q4 base is required so preview validation can run the 30-step CFG/STG route.
//!   * **Rectified-flow target = `noise - clean`.** LTX denoises with `x_t - σ·v` over
//!     `x_t = (1-σ)·x0 + σ·noise` and feeds the **raw** transformer output straight to `to_denoised`
//!     (no negation, unlike Z-Image), so the velocity that recovers `x0` is `v = noise - x0`. The
//!     **timestep fed to the DiT is the raw σ** (broadcast over tokens), σ ~ U(1e-3, 1-1e-3). MSE.
//!   * **2.3 latent layout.** A still image VAE-encodes (single frame T=1) to a normalized latent
//!     `(1,128,1,h,w)`, flattened to the patchified `(1, S, 128)` the DiT consumes; the position
//!     grid is built once per resolution bucket (sc-2127; one grid when buckets are off). The 24 GB Gemma text encoder is freed
//!     after the one-time prompt-embed cache (mirroring the reference), before the train loop.
//!     2.5 instead consumes schema-checked prepared video/audio safetensors and caches Gemma-4 AV
//!     contexts before freeing the encoder.
//!   * **Adapter surface.** 2.3 retains `attn1`/`attn2` `to_q/k/v/to_out.0`; 2.5 expands those
//!     suffixes over video/audio self/text attention and both cross-modal directions, with optional
//!     video/audio FF projections and no synthetic FF bias target. Residual LoRA over the (Q4) base — the base
//!     is frozen, gradients flow only through the trainable factors (functional autograd handles the
//!     `quantized_matmul` base as a constant). Saved as `{module}.lora_A/B.weight` + `.alpha` (the
//!     `to_out.0` diffusers spelling the inference loader normalizes), so it round-trips through
//!     the generation-specific strict adapter loader.
//!   * **LoRA-only.** The reference LTX MLX trainer has no LoKr (LTX *inference* supports LoKr via
//!     sc-2393, but no LoKr trainer exists); LoKr requests are rejected with that explanation.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{accumulate_grads, adapter_optimizer_update, average_grads, LoraParams};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxDriver, AuxModelFootprint, Parameterization, PerceptualPath, StepPlan,
    X0Decoder,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::taehv::{TaehvConfig, TaehvDecoder};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, divide, multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{LtxConfig, LtxVaeConfig, SplitModel};
use crate::gemma::GemmaConfig;
use crate::gemma4_te::Ltx25TextEncoder;
use crate::model::{MODEL_25_ID, MODEL_ID};
use crate::pipeline::preprocess_conditioning_image;
use crate::positions::{
    create_audio_position_grid, create_audio_position_grid_with, create_position_grid,
    create_position_grid_with, SPATIAL_SCALE,
};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::{Ltx25Tokenizer, LtxTokenizer};
use crate::transformer::{AvDiT, AvPerturbation, BlockLoraRef, LtxAdaptable, LtxDiT, Precision};
use crate::vae::LtxVideoVae;
use mlx_gen::gen_core::ltx_checkpoint::LtxComponent;
use mlx_gen::gen_core::BucketSchedule;

/// Gemma prompt token budget for caption encoding (the captions are short; padding tokens are
/// attended with `mask=None`, matching the reference `Modality(context_mask=None)`).
const MAX_PROMPT_TOKENS: usize = 128;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;
const LTX25_VIDEO_LATENT_CHANNELS: i32 = 128;
const LTX25_AUDIO_FLAT_CHANNELS: i32 = 128;

/// Executable upstream LTX-2.5 validation settings.  The individual guidance terms deliberately
/// stay separate: CFG, STG and modality isolation are different transformer evaluations in the dev
/// sampler and collapsing them to one generic image-preview scale changes the algorithm.
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25GuidancePlan {
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub rescale_scale: f32,
    pub modality_scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25ValidationRenderPlan {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub sigmas: Vec<f32>,
    pub stg_blocks: Vec<usize>,
    pub video_guidance: Ltx25GuidancePlan,
    pub audio_guidance: Ltx25GuidancePlan,
    pub generate_audio: bool,
}

pub fn build_ltx25_validation_render_plan(
    config: &Ltx25ValidationConfig,
) -> Result<Ltx25ValidationRenderPlan> {
    if !config.generate_video {
        return Err("ltx_2_5 trainer: validation must generate video".into());
    }
    if config.width < 32
        || config.width > 4096
        || config.height < 32
        || config.height > 4096
        || !config.width.is_multiple_of(32)
        || !config.height.is_multiple_of(32)
    {
        return Err(
            "ltx_2_5 trainer: validation width/height must be 32..=4096 and 32px-aligned".into(),
        );
    }
    if config.frames == 0 || config.frames > 257 || config.frames % 8 != 1 {
        return Err(
            "ltx_2_5 trainer: validation frames must be 1..=257 with frames % 8 == 1".into(),
        );
    }
    if !(1..=120).contains(&config.fps) || !(1..=100).contains(&config.steps) {
        return Err(
            "ltx_2_5 trainer: validation fps must be 1..=120 and steps must be 1..=100".into(),
        );
    }
    if config.stg_blocks.is_empty()
        || config.stg_blocks.len() > 8
        || config.stg_blocks.iter().any(|&block| block >= 48)
    {
        return Err("ltx_2_5 trainer: stgBlocks must contain 1..=8 unique indices in 0..48".into());
    }
    let mut unique = config.stg_blocks.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != config.stg_blocks.len() {
        return Err("ltx_2_5 trainer: stgBlocks must not contain duplicates".into());
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
        if !value.is_finite() || value < 0.0 || value > max {
            return Err(format!("ltx_2_5 trainer: {name} must be finite and in [0,{max}]").into());
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
        video_guidance: Ltx25GuidancePlan {
            cfg_scale: config.video_cfg_scale,
            stg_scale: config.video_stg_scale,
            rescale_scale: config.guidance_rescale,
            modality_scale: config.video_modality_guidance_scale,
        },
        audio_guidance: Ltx25GuidancePlan {
            cfg_scale: config.audio_cfg_scale,
            stg_scale: config.audio_stg_scale,
            rescale_scale: config.guidance_rescale,
            modality_scale: config.audio_modality_guidance_scale,
        },
        generate_audio: config.generate_audio,
    })
}

fn ensure_ltx25_training_variant(variant: crate::dev_sampler::TransformerVariant) -> Result<()> {
    if variant != crate::dev_sampler::TransformerVariant::Dev {
        return Err(
            "ltx_2_5 trainer requires the dev transformer (the dev/q4 training tier); a distilled \
             transformer cannot execute the canonical 30-step CFG/STG validation route"
                .into(),
        );
    }
    Ok(())
}

fn ensure_ltx25_training_tier(split: SplitModel) -> Result<()> {
    if !split.quantized || split.bits != 4 || split.group != 64 {
        return Err(format!(
            "ltx_2_5 trainer requires the dev/q4 tier (quantized=true, bits=4, group=64), got \
             quantized={}, bits={}, group={}",
            split.quantized, split.bits, split.group
        )
        .into());
    }
    Ok(())
}

fn validate_ltx25_adapter_scale(alpha: f32) -> Result<()> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(
            "ltx_2_5 trainer: alpha must be finite and > 0 for strict adapter metadata".into(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ltx25ConditionKind {
    FirstFrame,
    Prefix,
    Suffix,
    SpatialCrop,
    Mask,
    Reference,
}

/// One condition from the upstream flexible strategy.  `temporal_boundary` and `spatial_region`
/// are populated only by the condition kinds that consume them.  Masks and reference latents are
/// supplied per item by the shared request bridge; the strategy still records their executable
/// role here rather than reducing the workflow to a display string.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25ConditionSpec {
    pub kind: Ltx25ConditionKind,
    pub probability: f32,
    pub temporal_boundary: Option<usize>,
    pub spatial_region: Option<(usize, usize, usize, usize)>,
    pub spatial_scale_factor: Option<usize>,
    pub temporal_scale_factor: Option<usize>,
}

impl Ltx25ConditionSpec {
    const fn simple(kind: Ltx25ConditionKind, probability: f32) -> Self {
        Self {
            kind,
            probability,
            temporal_boundary: None,
            spatial_region: None,
            spatial_scale_factor: None,
            temporal_scale_factor: None,
        }
    }

    const fn temporal(kind: Ltx25ConditionKind, boundary: usize) -> Self {
        Self {
            kind,
            probability: 1.0,
            temporal_boundary: Some(boundary),
            spatial_region: None,
            spatial_scale_factor: None,
            temporal_scale_factor: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25ModalityPlan {
    /// `false` means a clean conditioning modality: sigma/timestep/loss are all zero.
    pub is_generated: bool,
    pub conditions: &'static [Ltx25ConditionSpec],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25WorkflowPlan {
    pub video: Option<Ltx25ModalityPlan>,
    pub audio: Option<Ltx25ModalityPlan>,
}

const GENERATED: Ltx25ModalityPlan = Ltx25ModalityPlan {
    is_generated: true,
    conditions: &[],
};
const FROZEN: Ltx25ModalityPlan = Ltx25ModalityPlan {
    is_generated: false,
    conditions: &[],
};
const fn generated(conditions: &'static [Ltx25ConditionSpec]) -> Ltx25ModalityPlan {
    Ltx25ModalityPlan {
        is_generated: true,
        conditions,
    }
}
const FIRST_FRAME_50: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(
    Ltx25ConditionKind::FirstFrame,
    0.5,
)];
const PREFIX_8: &[Ltx25ConditionSpec] =
    &[Ltx25ConditionSpec::temporal(Ltx25ConditionKind::Prefix, 8)];
const SUFFIX_8: &[Ltx25ConditionSpec] =
    &[Ltx25ConditionSpec::temporal(Ltx25ConditionKind::Suffix, 8)];
const MASK: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(Ltx25ConditionKind::Mask, 1.0)];
const SPATIAL_CROP: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec {
    kind: Ltx25ConditionKind::SpatialCrop,
    probability: 1.0,
    temporal_boundary: None,
    spatial_region: Some((0, 0, 288, 576)),
    spatial_scale_factor: None,
    temporal_scale_factor: None,
}];
const REFERENCE: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(
    Ltx25ConditionKind::Reference,
    1.0,
)];
const REFERENCE_FIRST_FRAME_20: &[Ltx25ConditionSpec] = &[
    Ltx25ConditionSpec::simple(Ltx25ConditionKind::Reference, 1.0),
    Ltx25ConditionSpec::simple(Ltx25ConditionKind::FirstFrame, 0.2),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

    pub fn parse(id: &str) -> Result<Self> {
        LTX25_WORKFLOWS
            .into_iter()
            .find(|workflow| workflow.id() == id)
            .ok_or_else(|| mlx_gen::Error::Msg(format!("ltx_2_5 trainer: unknown workflow `{id}`")))
    }

    /// Exact modality/condition semantics from the upstream v1.2.0 YAMLs.
    pub const fn plan(self) -> Ltx25WorkflowPlan {
        match self {
            Self::I2vLora => Ltx25WorkflowPlan {
                video: Some(generated(FIRST_FRAME_50)),
                audio: Some(GENERATED),
            },
            Self::T2vLora => Ltx25WorkflowPlan {
                video: Some(GENERATED),
                audio: Some(GENERATED),
            },
            Self::V2aLora => Ltx25WorkflowPlan {
                video: Some(FROZEN),
                audio: Some(GENERATED),
            },
            Self::A2vLora => Ltx25WorkflowPlan {
                video: Some(GENERATED),
                audio: Some(FROZEN),
            },
            Self::T2aLora => Ltx25WorkflowPlan {
                video: None,
                audio: Some(GENERATED),
            },
            Self::VideoExtendLora => Ltx25WorkflowPlan {
                video: Some(generated(PREFIX_8)),
                audio: Some(GENERATED),
            },
            Self::VideoInpaintingLora => Ltx25WorkflowPlan {
                video: Some(generated(MASK)),
                audio: None,
            },
            Self::VideoOutpaintingLora => Ltx25WorkflowPlan {
                video: Some(generated(SPATIAL_CROP)),
                audio: None,
            },
            Self::VideoSuffixLora => Ltx25WorkflowPlan {
                video: Some(generated(SUFFIX_8)),
                audio: Some(GENERATED),
            },
            Self::AudioExtendLora => Ltx25WorkflowPlan {
                video: None,
                audio: Some(generated(PREFIX_8)),
            },
            Self::AudioInpaintingLora => Ltx25WorkflowPlan {
                video: None,
                audio: Some(generated(MASK)),
            },
            Self::AudioSuffixLora => Ltx25WorkflowPlan {
                video: None,
                audio: Some(generated(SUFFIX_8)),
            },
            Self::Av2avIcLora => Ltx25WorkflowPlan {
                video: Some(generated(REFERENCE)),
                audio: Some(generated(REFERENCE)),
            },
            Self::V2vIcLora => Ltx25WorkflowPlan {
                video: Some(generated(REFERENCE_FIRST_FRAME_20)),
                audio: None,
            },
            Self::A2aIcLora => Ltx25WorkflowPlan {
                video: None,
                audio: Some(generated(REFERENCE)),
            },
        }
    }
}

/// Token geometry used to execute intrinsic-condition masks. Video tokens are flattened in
/// `(frame, height, width)` order; audio uses `height = width = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ltx25TokenGeometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    /// Number of source pixels represented by one spatial token (32 for the ConvVAE path).
    pub spatial_scale: usize,
}

impl Ltx25TokenGeometry {
    pub const fn tokens(self) -> usize {
        self.frames * self.height * self.width
    }
}

/// Per-item inputs that are intentionally not hard-coded in a workflow YAML: an inpainting mask,
/// optional reference-token count and the stable seed used for probabilistic conditions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ltx25ConditionInputs<'a> {
    /// `true` means clean conditioning/no loss, matching upstream mask=1 semantics.
    pub conditioning_mask: Option<&'a [bool]>,
    pub reference_tokens: usize,
    pub seed: u64,
    pub sample_index: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25TokenPlan {
    pub target_tokens: usize,
    pub reference_tokens: usize,
    /// `1` for generated target tokens, `0` for intrinsic/reference/frozen conditioning tokens.
    pub loss_mask: Vec<f32>,
    /// Per-token multiplier for sigma/timestep; identical to `loss_mask` for the flexible strategy.
    pub timestep_mask: Vec<f32>,
}

fn condition_draw(seed: u64, sample_index: u64, ordinal: usize) -> f32 {
    // SplitMix64: deterministic across platforms and independent of MLX's device RNG.  This mirrors
    // the upstream per-sample Bernoulli decision without consuming the noise generator's stream.
    let mut x = seed
        .wrapping_add(sample_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((ordinal as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 40) as f32) / ((1_u32 << 24) as f32)
}

/// Execute one modality's flexible-strategy conditions into the exact timestep/loss masks consumed
/// by the training kernel. Reference tokens are appended after target tokens and are always clean,
/// timestep zero and loss-free.
pub fn build_ltx25_token_plan(
    modality: Ltx25ModalityPlan,
    geometry: Ltx25TokenGeometry,
    inputs: Ltx25ConditionInputs<'_>,
) -> Result<Ltx25TokenPlan> {
    build_ltx25_token_plan_from_specs(modality.is_generated, modality.conditions, geometry, inputs)
}

fn build_ltx25_token_plan_from_specs(
    is_generated: bool,
    conditions: &[Ltx25ConditionSpec],
    geometry: Ltx25TokenGeometry,
    inputs: Ltx25ConditionInputs<'_>,
) -> Result<Ltx25TokenPlan> {
    let target_tokens = geometry.tokens();
    if target_tokens == 0 || geometry.spatial_scale == 0 {
        return Err("ltx_2_5 trainer: modality geometry must be non-zero".into());
    }
    let mut conditioned = vec![!is_generated; target_tokens];
    let frame_tokens = geometry.height * geometry.width;
    for (ordinal, condition) in conditions.iter().enumerate() {
        if condition_draw(inputs.seed, inputs.sample_index, ordinal) >= condition.probability {
            continue;
        }
        match condition.kind {
            Ltx25ConditionKind::FirstFrame => conditioned[..frame_tokens].fill(true),
            Ltx25ConditionKind::Prefix => {
                let frames = condition
                    .temporal_boundary
                    .unwrap_or(1)
                    .min(geometry.frames);
                conditioned[..frames * frame_tokens].fill(true);
            }
            Ltx25ConditionKind::Suffix => {
                let frames = condition
                    .temporal_boundary
                    .unwrap_or(1)
                    .min(geometry.frames);
                let start = (geometry.frames - frames) * frame_tokens;
                conditioned[start..].fill(true);
            }
            Ltx25ConditionKind::SpatialCrop => {
                let (y1, x1, y2, x2) = condition.spatial_region.ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: spatial_crop condition has no spatial_region".into(),
                    )
                })?;
                let spatial_scale = condition
                    .spatial_scale_factor
                    .unwrap_or(geometry.spatial_scale);
                if spatial_scale == 0 {
                    return Err("ltx_2_5 trainer: spatialScaleFactor must be > 0".into());
                }
                let (ly1, lx1) = (y1 / spatial_scale, x1 / spatial_scale);
                let ly2 = y2.div_ceil(spatial_scale).min(geometry.height);
                let lx2 = x2.div_ceil(spatial_scale).min(geometry.width);
                for t in 0..geometry.frames {
                    for y in ly1.min(geometry.height)..ly2 {
                        for x in lx1.min(geometry.width)..lx2 {
                            conditioned[t * frame_tokens + y * geometry.width + x] = true;
                        }
                    }
                }
            }
            Ltx25ConditionKind::Mask => {
                let mask = inputs.conditioning_mask.ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: mask workflow requires a per-item conditioning mask"
                            .into(),
                    )
                })?;
                if mask.len() != target_tokens {
                    return Err(format!(
                        "ltx_2_5 trainer: conditioning mask has {} tokens, expected {target_tokens}",
                        mask.len()
                    )
                    .into());
                }
                for (dst, &src) in conditioned.iter_mut().zip(mask) {
                    *dst |= src;
                }
            }
            Ltx25ConditionKind::Reference => {
                if inputs.reference_tokens == 0 {
                    return Err(
                        "ltx_2_5 trainer: reference workflow requires reference latents".into(),
                    );
                }
            }
        }
    }
    let mut loss_mask: Vec<f32> = conditioned
        .into_iter()
        .map(|is_condition| if is_condition { 0.0 } else { 1.0 })
        .collect();
    loss_mask.resize(target_tokens + inputs.reference_tokens, 0.0);
    Ok(Ltx25TokenPlan {
        target_tokens,
        reference_tokens: inputs.reference_tokens,
        timestep_mask: loss_mask.clone(),
        loss_mask,
    })
}

/// Fully prepared modality consumed by the joint AV loss. `noisy`/`target`/`timestep` include any
/// appended reference tokens; the mask keeps those tokens and every intrinsic condition out of loss.
#[derive(Clone)]
pub struct Ltx25PreparedModality {
    pub noisy: Array,
    pub target: Array,
    pub timestep: Array,
    pub context: Array,
    pub positions: Array,
    pub loss_mask: Array,
    pub loss_denominator: f32,
    pub target_tokens: usize,
}

/// Apply a token plan to target clean/noise tensors.  This is the executable bridge shared by all
/// 15 workflows: generated tokens use flow matching, intrinsic/frozen tokens stay clean, and
/// reference tokens are concatenated clean with zero target/timestep/loss.
#[allow(clippy::too_many_arguments)]
pub fn prepare_ltx25_modality(
    clean: &Array,
    noise: &Array,
    context: &Array,
    target_positions: &Array,
    reference: Option<&Array>,
    reference_positions: Option<&Array>,
    sigma: f32,
    plan: &Ltx25TokenPlan,
) -> Result<Ltx25PreparedModality> {
    if clean.shape() != noise.shape() || clean.ndim() != 3 {
        return Err("ltx_2_5 trainer: clean/noise modalities must share shape (B,S,C)".into());
    }
    let (batch, tokens, channels) = (clean.shape()[0], clean.shape()[1], clean.shape()[2]);
    if tokens as usize != plan.target_tokens {
        return Err(format!(
            "ltx_2_5 trainer: modality carries {tokens} target tokens, plan expects {}",
            plan.target_tokens
        )
        .into());
    }
    let target_mask = Array::from_slice(&plan.loss_mask[..plan.target_tokens], &[1, tokens, 1])
        .as_dtype(clean.dtype())?;
    let target_mask = broadcast_to(&target_mask, &[batch, tokens, 1])?;
    let one = Array::from_slice(&[1.0f32], &[1]).as_dtype(clean.dtype())?;
    let inv_mask = subtract(&one, &target_mask)?;
    let sigma_a = Array::from_slice(&[sigma], &[1]).as_dtype(clean.dtype())?;
    let one_minus_sigma = Array::from_slice(&[1.0 - sigma], &[1]).as_dtype(clean.dtype())?;
    let flow_noisy = add(
        &multiply(clean, &one_minus_sigma)?,
        &multiply(noise, &sigma_a)?,
    )?;
    let mut noisy = add(
        &multiply(&flow_noisy, &target_mask)?,
        &multiply(clean, &inv_mask)?,
    )?;
    let mut target = multiply(&subtract(noise, clean)?, &target_mask)?;
    let mut positions = target_positions.clone();
    if plan.reference_tokens > 0 {
        let reference = reference.ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: token plan requires reference latents".into())
        })?;
        let reference_positions = reference_positions.ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: reference latents require positions".into())
        })?;
        if reference.shape()[0] != batch
            || reference.shape()[1] as usize != plan.reference_tokens
            || reference.shape()[2] != channels
        {
            return Err(format!(
                "ltx_2_5 trainer: reference shape {:?} does not match (B,{},C)",
                reference.shape(),
                plan.reference_tokens
            )
            .into());
        }
        noisy = concatenate_axis(&[&noisy, reference], 1)?;
        let zero_target = Array::zeros::<f32>(reference.shape())?.as_dtype(clean.dtype())?;
        target = concatenate_axis(&[&target, &zero_target], 1)?;
        positions = concatenate_axis(&[&positions, reference_positions], 2)?;
    }
    let all_tokens = plan.loss_mask.len() as i32;
    let timestep =
        Array::from_slice(&plan.timestep_mask, &[1, all_tokens]).as_dtype(clean.dtype())?;
    let timestep = multiply(&broadcast_to(&timestep, &[batch, all_tokens])?, &sigma_a)?;
    let loss_mask =
        Array::from_slice(&plan.loss_mask, &[1, all_tokens, 1]).as_dtype(clean.dtype())?;
    let loss_mask = broadcast_to(&loss_mask, &[batch, all_tokens, 1])?;
    Ok(Ltx25PreparedModality {
        noisy,
        target,
        timestep,
        context: context.clone(),
        positions,
        loss_mask,
        loss_denominator: plan.loss_mask.iter().sum::<f32>() * batch as f32 * channels as f32,
        target_tokens: plan.target_tokens,
    })
}

#[derive(Clone, Default)]
pub struct Ltx25PreparedBatch {
    pub video: Option<Ltx25PreparedModality>,
    pub audio: Option<Ltx25PreparedModality>,
}

#[derive(Clone, Debug)]
pub struct Ltx25ResolvedCondition {
    pub spec: Ltx25ConditionSpec,
    pub tensor_key: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Ltx25ResolvedModality {
    pub is_generated: bool,
    pub conditions: Vec<Ltx25ResolvedCondition>,
}

#[derive(Clone, Debug)]
pub struct Ltx25TrainingPlan {
    pub workflow: Ltx25Workflow,
    pub video: Option<Ltx25ResolvedModality>,
    pub audio: Option<Ltx25ResolvedModality>,
    pub validation: Ltx25ValidationConfig,
}

fn canonical_modality(plan: Ltx25ModalityPlan) -> Ltx25ResolvedModality {
    Ltx25ResolvedModality {
        is_generated: plan.is_generated,
        conditions: plan
            .conditions
            .iter()
            .map(|&spec| Ltx25ResolvedCondition {
                spec,
                tensor_key: None,
            })
            .collect(),
    }
}

fn json_u32(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<Option<u32>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    mlx_gen::Error::Msg(format!("ltx_2_5 trainer: `{key}` must be a u32"))
                })
        })
        .transpose()
}

fn parse_resolved_condition(value: &serde_json::Value) -> Result<Ltx25ResolvedCondition> {
    let object = value.as_object().ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_5 trainer: each condition must be an object".into())
    })?;
    let kind = match object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_5 trainer: condition.type is required".into()))?
    {
        "firstFrame" => Ltx25ConditionKind::FirstFrame,
        "prefix" => Ltx25ConditionKind::Prefix,
        "suffix" => Ltx25ConditionKind::Suffix,
        "spatialCrop" => Ltx25ConditionKind::SpatialCrop,
        "mask" => Ltx25ConditionKind::Mask,
        "reference" => Ltx25ConditionKind::Reference,
        other => {
            return Err(format!("ltx_2_5 trainer: unsupported condition type `{other}`").into())
        }
    };
    let probability = object
        .get("probability")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: condition.probability is required".into())
        })? as f32;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err("ltx_2_5 trainer: condition probability must be finite and in [0,1]".into());
    }
    let temporal_boundary = json_u32(object, "temporalBoundary")?.map(|value| value as usize);
    if matches!(
        kind,
        Ltx25ConditionKind::Prefix | Ltx25ConditionKind::Suffix
    ) && temporal_boundary == Some(0)
    {
        return Err("ltx_2_5 trainer: temporalBoundary must be > 0".into());
    }
    let spatial_region = object
        .get("spatialRegion")
        .map(|value| {
            let values = value.as_array().ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: spatialRegion must be a four-element array".into(),
                )
            })?;
            let parsed: Vec<usize> = values
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: spatialRegion values must be u32".into(),
                            )
                        })
                })
                .collect::<Result<_>>()?;
            let [y1, x1, y2, x2]: [usize; 4] = parsed.try_into().map_err(|_| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: spatialRegion must contain four values".into(),
                )
            })?;
            if y2 <= y1 || x2 <= x1 {
                return Err(mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: spatialRegion must have positive area".into(),
                ));
            }
            Ok((y1, x1, y2, x2))
        })
        .transpose()?;
    let tensor_key = object
        .get("tensorKey")
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: condition.tensorKey must be a non-empty string".into(),
                    )
                })
        })
        .transpose()?;
    if matches!(
        kind,
        Ltx25ConditionKind::Mask | Ltx25ConditionKind::Reference
    ) && tensor_key.is_none()
    {
        return Err(format!("ltx_2_5 trainer: {:?} condition requires tensorKey", kind).into());
    }
    let spatial_scale_factor = json_u32(object, "spatialScaleFactor")?.map(|value| value as usize);
    let temporal_scale_factor =
        json_u32(object, "temporalScaleFactor")?.map(|value| value as usize);
    if spatial_scale_factor == Some(0) || temporal_scale_factor == Some(0) {
        return Err(
            "ltx_2_5 trainer: spatialScaleFactor and temporalScaleFactor must be > 0".into(),
        );
    }
    let reject = |field: &str| -> Result<()> {
        Err(format!(
            "ltx_2_5 trainer: {field} is not valid for condition type `{}`",
            object["type"].as_str().unwrap_or("unknown")
        )
        .into())
    };
    match kind {
        Ltx25ConditionKind::FirstFrame => {
            if temporal_boundary.is_some() {
                reject("temporalBoundary")?;
            }
            if spatial_region.is_some() {
                reject("spatialRegion")?;
            }
            if tensor_key.is_some() {
                reject("tensorKey")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        Ltx25ConditionKind::Prefix | Ltx25ConditionKind::Suffix => {
            if temporal_boundary.is_none() {
                return Err(
                    "ltx_2_5 trainer: prefix/suffix conditions require temporalBoundary".into(),
                );
            }
            if spatial_region.is_some() {
                reject("spatialRegion")?;
            }
            if tensor_key.is_some() {
                reject("tensorKey")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        Ltx25ConditionKind::SpatialCrop => {
            if spatial_region.is_none() {
                return Err("ltx_2_5 trainer: spatialCrop conditions require spatialRegion".into());
            }
            if temporal_boundary.is_some() {
                reject("temporalBoundary")?;
            }
            if tensor_key.is_some() {
                reject("tensorKey")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        Ltx25ConditionKind::Mask => {
            if temporal_boundary.is_some() || spatial_region.is_some() {
                reject("temporalBoundary/spatialRegion")?;
            }
            if spatial_scale_factor.is_some() || temporal_scale_factor.is_some() {
                reject("spatialScaleFactor/temporalScaleFactor")?;
            }
        }
        Ltx25ConditionKind::Reference => {
            if temporal_boundary.is_some() || spatial_region.is_some() {
                reject("temporalBoundary/spatialRegion")?;
            }
        }
    }
    Ok(Ltx25ResolvedCondition {
        spec: Ltx25ConditionSpec {
            kind,
            probability,
            temporal_boundary,
            spatial_region,
            spatial_scale_factor,
            temporal_scale_factor,
        },
        tensor_key,
    })
}

fn resolve_modality(
    value: Option<&serde_json::Value>,
    canonical: Option<Ltx25ModalityPlan>,
    field: &str,
    is_video: bool,
) -> Result<Option<Ltx25ResolvedModality>> {
    let Some(canonical) = canonical else {
        if value.is_some() {
            return Err(format!(
                "ltx_2_5 trainer: workflow does not carry {field}, but `{field}` was provided"
            )
            .into());
        }
        return Ok(None);
    };
    let (is_generated, mut conditions): (bool, Vec<Ltx25ResolvedCondition>) = match value {
        None => {
            let resolved = canonical_modality(canonical);
            (resolved.is_generated, resolved.conditions)
        }
        Some(value) => {
            let object = value.as_object().ok_or_else(|| {
                mlx_gen::Error::Msg(format!("ltx_2_5 trainer: `{field}` must be an object"))
            })?;
            let is_generated = object
                .get("isGenerated")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| {
                    mlx_gen::Error::Msg(format!(
                        "ltx_2_5 trainer: `{field}.isGenerated` is required"
                    ))
                })?;
            let conditions = object
                .get("conditions")
                .and_then(serde_json::Value::as_array)
                .map(|values| values.iter().map(parse_resolved_condition).collect())
                .transpose()?
                .unwrap_or_else(|| canonical_modality(canonical).conditions);
            (is_generated, conditions)
        }
    };
    if is_generated != canonical.is_generated {
        return Err(format!(
            "ltx_2_5 trainer: `{field}.isGenerated={is_generated}` contradicts workflow value {}",
            canonical.is_generated
        )
        .into());
    }
    if conditions.len() != canonical.conditions.len() {
        return Err(format!(
            "ltx_2_5 trainer: `{field}` must contain the workflow's exact condition set"
        )
        .into());
    }
    for required in canonical.conditions {
        let Some(condition) = conditions
            .iter_mut()
            .find(|condition| condition.spec.kind == required.kind)
        else {
            return Err(format!(
                "ltx_2_5 trainer: `{field}` omits workflow-required {:?} condition",
                required.kind
            )
            .into());
        };
        if condition.spec.probability != required.probability
            || condition.spec.temporal_boundary != required.temporal_boundary
            || condition.spec.spatial_region != required.spatial_region
        {
            return Err(format!(
                "ltx_2_5 trainer: `{field}` {:?} condition values contradict the canonical workflow",
                condition.spec.kind
            )
            .into());
        }
        if matches!(
            condition.spec.kind,
            Ltx25ConditionKind::Mask | Ltx25ConditionKind::Reference
        ) && condition.tensor_key.is_none()
        {
            return Err(format!(
                "ltx_2_5 trainer: `{field}` {:?} condition requires tensorKey",
                condition.spec.kind
            )
            .into());
        }
        if condition.spec.kind == Ltx25ConditionKind::Reference {
            let (spatial, temporal) = if is_video {
                (
                    SPATIAL_SCALE as usize,
                    crate::positions::TEMPORAL_SCALE as usize,
                )
            } else {
                (1, crate::positions::AUDIO_LATENT_DOWNSAMPLE_FACTOR as usize)
            };
            if condition
                .spec
                .spatial_scale_factor
                .is_some_and(|value| value != spatial)
                || condition
                    .spec
                    .temporal_scale_factor
                    .is_some_and(|value| value != temporal)
            {
                return Err(format!(
                    "ltx_2_5 trainer: `{field}` reference scales must be spatial={spatial}, temporal={temporal}"
                )
                .into());
            }
            condition.spec.spatial_scale_factor = Some(spatial);
            condition.spec.temporal_scale_factor = Some(temporal);
        }
    }
    if conditions.iter().any(|condition| {
        !canonical
            .conditions
            .iter()
            .any(|allowed| allowed.kind == condition.spec.kind)
    }) {
        return Err(format!(
            "ltx_2_5 trainer: `{field}` conditions contradict the selected workflow"
        )
        .into());
    }
    Ok(Some(Ltx25ResolvedModality {
        is_generated,
        conditions,
    }))
}

fn parse_validation(value: Option<&serde_json::Value>) -> Result<Ltx25ValidationConfig> {
    let Some(value) = value else {
        return Ok(LTX25_VALIDATION_DEFAULTS.into());
    };
    let object = value.as_object().ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_5 trainer: `ltxValidation` must be an object".into())
    })?;
    let u32_or =
        |key: &str, default: u32| -> Result<u32> { Ok(json_u32(object, key)?.unwrap_or(default)) };
    let f32_or = |key: &str, default: f32| -> Result<f32> {
        let value = object
            .get(key)
            .map(|value| {
                value.as_f64().map(|value| value as f32).ok_or_else(|| {
                    mlx_gen::Error::Msg(format!("ltx_2_5 trainer: `{key}` must be numeric"))
                })
            })
            .transpose()?
            .unwrap_or(default);
        if !value.is_finite() {
            return Err(format!("ltx_2_5 trainer: `{key}` must be finite").into());
        }
        Ok(value)
    };
    let blocks: Vec<usize> = object
        .get("stgBlocks")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| {
                    mlx_gen::Error::Msg("ltx_2_5 trainer: `stgBlocks` must be an array".into())
                })?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: `stgBlocks` values must be u32".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![28]);
    let generate_audio = object
        .get("generateAudio")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: `generateAudio` must be boolean".into())
            })
        })
        .transpose()?
        .unwrap_or(true);
    let parsed = Ltx25ValidationConfig {
        width: u32_or("width", 960)?,
        height: u32_or("height", 544)?,
        frames: u32_or("frames", 89)?,
        fps: u32_or("fps", 24)?,
        steps: u32_or("steps", 30)?,
        video_cfg_scale: f32_or("videoCfgScale", 3.0)?,
        audio_cfg_scale: f32_or("audioCfgScale", 7.0)?,
        video_stg_scale: f32_or("videoStgScale", 1.0)?,
        audio_stg_scale: f32_or("audioStgScale", 1.0)?,
        stg_blocks: blocks,
        guidance_rescale: f32_or("guidanceRescale", 0.7)?,
        video_modality_guidance_scale: f32_or("videoModalityGuidanceScale", 3.0)?,
        audio_modality_guidance_scale: f32_or("audioModalityGuidanceScale", 3.0)?,
        generate_video: true,
        generate_audio,
    };
    build_ltx25_validation_render_plan(&parsed)?;
    Ok(parsed)
}

impl Ltx25TrainingPlan {
    pub fn from_request(req: &TrainingRequest) -> Result<Self> {
        let options = &req.config.model_options;
        let workflow = Ltx25Workflow::parse(
            options
                .get("ltxWorkflow")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    mlx_gen::Error::Msg("ltx_2_5 trainer: `ltxWorkflow` is required".into())
                })?,
        )?;
        let canonical = workflow.plan();
        let video = resolve_modality(options.get("ltxVideo"), canonical.video, "ltxVideo", true)?;
        let audio = resolve_modality(options.get("ltxAudio"), canonical.audio, "ltxAudio", false)?;
        if !video.as_ref().is_some_and(|value| value.is_generated)
            && !audio.as_ref().is_some_and(|value| value.is_generated)
        {
            return Err("ltx_2_5 trainer: at least one modality must be generated".into());
        }
        Ok(Self {
            workflow,
            video,
            audio,
            validation: parse_validation(options.get("ltxValidation"))?,
        })
    }
}

#[derive(Clone)]
struct CachedLtx25Modality {
    clean: Array,
    context: Array,
    positions: Array,
    reference: Option<Array>,
    reference_positions: Option<Array>,
    token_plan: Ltx25TokenPlan,
}

#[derive(Clone, Default)]
struct CachedLtx25Example {
    video: Option<CachedLtx25Modality>,
    audio: Option<CachedLtx25Modality>,
    /// The generated video clip's depth-anchoring frame selection (sc-24830); `None` when depth
    /// anchoring is off (or the example generates no video).
    depth: Option<Ltx25DepthFrames>,
}

/// Which latent frames of one example's video clip the depth loss decodes, and the clip's
/// unpatchified grid (sc-24830).
#[derive(Clone, Debug, PartialEq)]
struct Ltx25DepthFrames {
    /// `(C, F, H, W)` of the patchified target video tokens.
    grid: [i32; 4],
    /// Selected latent frame indices (ascending).
    frames: Vec<i32>,
}

impl Ltx25DepthFrames {
    /// The latent frames [`Ltx25DepthDecoder`] decodes for the selected (ascending) frames: frame
    /// `0` alone (a `T = 1` clip — it decodes to one pixel frame), every frame `k > 0` preceded by
    /// `k − 1` (a `T = 2` clip, so TAEHV's MemBlocks see the frame its 8-pixel-frame group
    /// follows). Frame 0 can only lead, so the layout is `[0]? ++ [k−1, k]*`.
    fn decode_frames(&self) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(2 * self.frames.len());
        for (i, &k) in self.frames.iter().enumerate() {
            match k {
                0 if i == 0 => out.push(0),
                k if k > 0 => out.extend([k - 1, k]),
                _ => {
                    return Err(format!(
                        "ltx_2_5 trainer: depth frames must be ascending and non-negative, got \
                         {:?}",
                        self.frames
                    )
                    .into())
                }
            }
        }
        Ok(out)
    }

    /// The selected frames of the patchified target tokens `(1, F·H·W, C)` as the decoder's NCHW
    /// batch — the model-space (normalized) latent TAELTX2.3 decodes, in the
    /// [`decode_frames`](Self::decode_frames) layout [`Ltx25DepthDecoder`] reads back as clips (one
    /// decoded frame per selected frame).
    fn nchw(&self, tokens: &Array) -> Result<Array> {
        let [c, f, h, w] = self.grid;
        let grid = crate::conditioning::unpatchify_grid(tokens, c, f, h, w)?;
        let decode = self.decode_frames()?;
        let n = decode.len() as i32;
        let index = Array::from_slice(&decode, &[n]);
        Ok(grid
            .take_axis(&index, 2)?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[n, c, h, w])?)
    }
}

/// LTX-2.5's TAELTX2.3 depth decoder (sc-24830). A video latent frame `k > 0` encodes a group of
/// 8 pixel frames that TAEHV decodes with temporal memory of frame `k − 1`; decoding it as a lone
/// `T = 1` clip would be out of distribution. So each selected frame `k > 0` decodes together with
/// its predecessor as a `T = 2` clip and keeps only the clip's last output frame (latent `k`'s last
/// grown frame — the same frame a `T = 1` decode keeps), while frame 0 stays a `T = 1` clip. Input:
/// the [`Ltx25DepthFrames::nchw`] batch `[0]? ++ [k−1, k]*` (odd length ⇔ a leading lone frame 0);
/// output: NHWC, one frame per selected frame, in order. Applied identically to the live x0 and to
/// the clean reference, so their shapes match.
struct Ltx25DepthDecoder(TaehvDecoder);

impl X0Decoder for Ltx25DepthDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        use mlx_rs::ops::indexing::IndexOp;
        let sh = latents.shape().to_vec();
        if sh.len() != 4 {
            return Err(format!(
                "ltx_2_5 trainer: the depth decoder expects NCHW latents, got shape {sh:?}"
            )
            .into());
        }
        let (n, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let lone = n % 2;
        let pairs = n / 2;
        let mut parts = Vec::with_capacity(2);
        if lone == 1 {
            parts.push(self.0.decode_frames(&latents.index(..1))?);
        }
        if pairs > 0 {
            let clips = latents.index(lone..).reshape(&[pairs, 2, c, h, w])?;
            parts.push(self.0.decode_clip_last_frames(&clips)?);
        }
        let parts: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&parts, 0)?)
    }
}

/// The [`Ltx25DepthDecoder`] for the shared builder: TAELTX2.3 from `perceptual_decoder_dir`, and a
/// per-selected-frame working set of one `T = 2` clip decode (an upper bound: frame 0 decodes as
/// `T = 1`).
struct Ltx25DepthDecoderSpec;

impl mlx_gen_perceptual::CustomDecoder for Ltx25DepthDecoderSpec {
    fn name(&self) -> &'static str {
        "TAELTX2.3"
    }

    fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint {
        let cfg = TaehvConfig::taeltx2_3();
        AuxModelFootprint {
            working_set_bytes: cfg.clip_training_working_set_bytes(h, w, 2),
            ..cfg.footprint(h, w)
        }
    }

    fn load(&self, dir: Option<&Path>) -> Result<Box<dyn X0Decoder>> {
        let dir = dir.ok_or_else(|| {
            mlx_gen::Error::Msg(
                "the perceptual losses need the TAELTX2.3 decoder (perceptual_decoder_dir)".into(),
            )
        })?;
        let dec = TaehvDecoder::from_path(dir, TaehvConfig::taeltx2_3())
            .map_err(|e| mlx_gen::Error::Msg(format!("{}: {e}", dir.display())))?;
        Ok(Box::new(Ltx25DepthDecoder(dec)))
    }
}

/// LTX-2.5's decoder for the shared aux-loss builder (sc-24830): TAELTX2.3 through
/// [`Ltx25DepthDecoder`].
fn ltx25_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Custom(Box::new(Ltx25DepthDecoderSpec))
}

fn parse_shape_metadata(weights: &Weights, key: &str, tensor: &Array) -> Result<Vec<i32>> {
    let raw = weights.metadata(key).ok_or_else(|| {
        mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: prepared bundle is missing `{key}` metadata"
        ))
    })?;
    let values: Vec<i32> = serde_json::from_str(raw).map_err(|error| {
        mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: prepared bundle `{key}` is not a JSON shape: {error}"
        ))
    })?;
    if values != tensor.shape() {
        return Err(format!(
            "ltx_2_5 trainer: prepared bundle `{key}` {values:?} does not match tensor shape {:?}",
            tensor.shape()
        )
        .into());
    }
    Ok(values)
}

fn prepared_fps(weights: &Weights) -> Result<f64> {
    let fps = weights
        .metadata("fps")
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: video bundle is missing `fps` metadata".into())
        })?
        .parse::<f64>()
        .map_err(|_| mlx_gen::Error::Msg("ltx_2_5 trainer: `fps` must be numeric".into()))?;
    if !fps.is_finite() || fps <= 0.0 || fps > f32::MAX as f64 {
        return Err("ltx_2_5 trainer: `fps` must be finite, > 0, and representable as f32".into());
    }
    Ok(fps)
}

fn validate_prepared_conditions(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    main_shape: &[i32],
    target_tokens: usize,
    is_video: bool,
) -> Result<()> {
    for condition in &modality.conditions {
        let Some(key) = condition.tensor_key.as_deref() else {
            continue;
        };
        let tensor = weights.require(key)?;
        match condition.spec.kind {
            Ltx25ConditionKind::Mask => {
                if tensor.size() != target_tokens {
                    return Err(format!(
                        "ltx_2_5 trainer: mask tensor `{key}` has {} values, expected {target_tokens}",
                        tensor.size()
                    )
                    .into());
                }
            }
            Ltx25ConditionKind::Reference if is_video => {
                let shape = tensor.shape();
                if shape.len() != 5 || shape[0] != main_shape[0] || shape[1] != main_shape[1] {
                    return Err(format!(
                        "ltx_2_5 trainer: video reference `{key}` must be [B,C,F,H,W] with B/C matching video_latents, got {shape:?}"
                    )
                    .into());
                }
            }
            Ltx25ConditionKind::Reference => {
                let shape = tensor.shape();
                if shape.len() != 4
                    || shape[0] != main_shape[0]
                    || shape[1] != main_shape[1]
                    || shape[3] != main_shape[3]
                {
                    return Err(format!(
                        "ltx_2_5 trainer: audio reference `{key}` must be [B,C,T,F] with B/C/F matching audio_latents, got {shape:?}"
                    )
                    .into());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_prepared_bundle_contract(
    weights: &Weights,
    path: &Path,
    plan: &Ltx25TrainingPlan,
) -> Result<()> {
    if weights.metadata("schemaVersion") != Some("ltx-prepared-v1") {
        return Err(format!(
            "ltx_2_5 trainer: prepared bundle {} must declare schemaVersion=ltx-prepared-v1",
            path.display()
        )
        .into());
    }
    let video_timing = if let Some(modality) = &plan.video {
        let raw = weights.require("video_latents")?;
        let shape = parse_shape_metadata(weights, "videoShape", raw)?;
        let [batch, channels, frames, height, width]: [i32; 5] =
            shape.clone().try_into().map_err(|_| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: videoShape must be [B,C,F,H,W]".into())
            })?;
        if batch != 1 || channels <= 0 || frames <= 0 || height <= 0 || width <= 0 {
            return Err(
                "ltx_2_5 trainer: videoShape must be positive with batch dimension 1".into(),
            );
        }
        if channels != LTX25_VIDEO_LATENT_CHANNELS {
            return Err(format!(
                "ltx_2_5 trainer: video_latents has {channels} channels, transformer expects {}",
                LTX25_VIDEO_LATENT_CHANNELS
            )
            .into());
        }
        let fps = prepared_fps(weights)?;
        validate_prepared_conditions(
            weights,
            modality,
            &shape,
            frames as usize * height as usize * width as usize,
            true,
        )?;
        Some((frames as usize, fps))
    } else {
        None
    };
    let audio_frames = if let Some(modality) = &plan.audio {
        let raw = weights.require("audio_latents")?;
        let shape = parse_shape_metadata(weights, "audioShape", raw)?;
        let [batch, channels, frames, bins]: [i32; 4] = shape.clone().try_into().map_err(|_| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: audioShape must be [B,C,T,F]".into())
        })?;
        if batch != 1 || channels <= 0 || frames <= 0 || bins <= 0 {
            return Err(
                "ltx_2_5 trainer: audioShape must be positive with batch dimension 1".into(),
            );
        }
        if channels * bins != LTX25_AUDIO_FLAT_CHANNELS {
            return Err(format!(
                "ltx_2_5 trainer: flattened audio_latents has {} channels, transformer expects {}",
                channels * bins,
                LTX25_AUDIO_FLAT_CHANNELS
            )
            .into());
        }
        validate_prepared_conditions(weights, modality, &shape, frames as usize, false)?;
        Some(frames as usize)
    } else {
        None
    };
    if let (Some((video_latent_frames, fps)), Some(audio_frames)) = (video_timing, audio_frames) {
        let output_frames = (video_latent_frames - 1)
            .checked_mul(crate::positions::TEMPORAL_SCALE as usize)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: videoShape duration overflows frame mapping".into(),
                )
            })?;
        let expected = crate::positions::compute_audio_frames(output_frames, fps);
        if audio_frames != expected {
            return Err(format!(
                "ltx_2_5 trainer: audioShape duration has {audio_frames} latent frames, expected {expected} for videoShape {video_latent_frames} latent frames at {fps} fps"
            )
            .into());
        }
    }
    Ok(())
}

fn prepared_bundle_path(item: &mlx_gen::TrainingItem) -> Result<&Path> {
    item.model_options
        .get("ltxPreparedBundlePath")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.is_empty())
        .map(Path::new)
        .ok_or_else(|| {
            mlx_gen::Error::Msg(
                "ltx_2_5 trainer: every item requires `ltxPreparedBundlePath`".into(),
            )
        })
}

fn reference_condition(modality: &Ltx25ResolvedModality) -> Option<&Ltx25ResolvedCondition> {
    modality
        .conditions
        .iter()
        .find(|condition| condition.spec.kind == Ltx25ConditionKind::Reference)
}

fn mask_condition(modality: &Ltx25ResolvedModality) -> Option<&Ltx25ResolvedCondition> {
    modality
        .conditions
        .iter()
        .find(|condition| condition.spec.kind == Ltx25ConditionKind::Mask)
}

fn condition_mask_values(
    weights: &Weights,
    condition: Option<&Ltx25ResolvedCondition>,
    target_tokens: usize,
) -> Result<Option<Vec<bool>>> {
    let Some(condition) = condition else {
        return Ok(None);
    };
    let key = condition.tensor_key.as_deref().ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_5 trainer: mask condition has no tensorKey".into())
    })?;
    let values = to_dtype(weights.require(key)?, Dtype::Float32)?;
    eval([&values])?;
    if values.size() != target_tokens {
        return Err(format!(
            "ltx_2_5 trainer: mask tensor `{key}` has {} values, expected {target_tokens}",
            values.size()
        )
        .into());
    }
    Ok(Some(
        values
            .as_slice::<f32>()
            .iter()
            .map(|&value| value > 0.5)
            .collect(),
    ))
}

fn prepared_video_modality(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    context: &Array,
    fps: f32,
    seed: u64,
    sample_index: u64,
) -> Result<CachedLtx25Modality> {
    let raw = to_dtype(weights.require("video_latents")?, Dtype::Float32)?;
    let shape = parse_shape_metadata(weights, "videoShape", &raw)?;
    let [batch, _channels, frames, height, width]: [i32; 5] = shape.try_into().map_err(|_| {
        mlx_gen::Error::Msg(
            "ltx_2_5 trainer: videoShape must be [B,C,F,H,W] for patchification".into(),
        )
    })?;
    if batch != 1 || frames <= 0 || height <= 0 || width <= 0 {
        return Err("ltx_2_5 trainer: prepared video must have positive shape and B=1".into());
    }
    let clean = crate::conditioning::patchify_grid(&raw)?;
    let geometry = Ltx25TokenGeometry {
        frames: frames as usize,
        height: height as usize,
        width: width as usize,
        spatial_scale: SPATIAL_SCALE as usize,
    };
    let mask = condition_mask_values(weights, mask_condition(modality), geometry.tokens())?;
    let reference_condition = reference_condition(modality);
    let reference_active = reference_condition.is_some_and(|condition| {
        condition_draw(seed, sample_index, 0) < condition.spec.probability
    });
    let (reference, reference_positions, reference_tokens) = if reference_active {
        let condition = reference_condition.expect("checked");
        let key = condition.tensor_key.as_deref().ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: reference condition has no tensorKey".into())
        })?;
        let raw_reference = to_dtype(weights.require(key)?, Dtype::Float32)?;
        let shape = raw_reference.shape();
        if shape.len() != 5 || shape[0] != batch || shape[1] != raw.shape()[1] {
            return Err(format!(
                "ltx_2_5 trainer: video reference `{key}` must be [B,C,F,H,W], got {shape:?}"
            )
            .into());
        }
        let tokens = crate::conditioning::patchify_grid(&raw_reference)?;
        let positions = create_position_grid_with(
            batch as usize,
            shape[2] as usize,
            shape[3] as usize,
            shape[4] as usize,
            condition
                .spec
                .temporal_scale_factor
                .unwrap_or(crate::positions::TEMPORAL_SCALE as usize) as i64,
            condition
                .spec
                .spatial_scale_factor
                .unwrap_or(SPATIAL_SCALE as usize) as i64,
            fps,
            true,
        );
        let count = tokens.shape()[1] as usize;
        (Some(tokens), Some(positions), count)
    } else {
        (None, None, 0)
    };
    let specs: Vec<Ltx25ConditionSpec> = modality
        .conditions
        .iter()
        .map(|condition| condition.spec)
        .collect();
    let token_plan = build_ltx25_token_plan_from_specs(
        modality.is_generated,
        &specs,
        geometry,
        Ltx25ConditionInputs {
            conditioning_mask: mask.as_deref(),
            reference_tokens,
            seed,
            sample_index,
        },
    )?;
    Ok(CachedLtx25Modality {
        clean,
        context: context.clone(),
        positions: create_position_grid_with(
            batch as usize,
            frames as usize,
            height as usize,
            width as usize,
            crate::positions::TEMPORAL_SCALE,
            SPATIAL_SCALE,
            fps,
            true,
        ),
        reference,
        reference_positions,
        token_plan,
    })
}

fn prepared_audio_modality(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    context: &Array,
    seed: u64,
    sample_index: u64,
) -> Result<CachedLtx25Modality> {
    let raw = to_dtype(weights.require("audio_latents")?, Dtype::Float32)?;
    let shape = parse_shape_metadata(weights, "audioShape", &raw)?;
    let [batch, channels, frames, bins]: [i32; 4] = shape
        .try_into()
        .map_err(|_| mlx_gen::Error::Msg("ltx_2_5 trainer: audioShape must be [B,C,T,F]".into()))?;
    if batch != 1 || channels <= 0 || frames <= 0 || bins <= 0 {
        return Err("ltx_2_5 trainer: prepared audio must have positive shape and B=1".into());
    }
    let clean = raw
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, frames, channels * bins])?;
    let geometry = Ltx25TokenGeometry {
        frames: frames as usize,
        height: 1,
        width: 1,
        spatial_scale: 1,
    };
    let mask = condition_mask_values(weights, mask_condition(modality), geometry.tokens())?;
    let reference_condition = reference_condition(modality);
    let reference_active = reference_condition.is_some_and(|condition| {
        condition_draw(seed, sample_index, 0) < condition.spec.probability
    });
    let (reference, reference_positions, reference_tokens) = if reference_active {
        let condition = reference_condition.expect("checked");
        let key = condition.tensor_key.as_deref().ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: reference condition has no tensorKey".into())
        })?;
        let raw_reference = to_dtype(weights.require(key)?, Dtype::Float32)?;
        let shape = raw_reference.shape();
        if shape.len() != 4 || shape[0] != batch || shape[1] != channels || shape[3] != bins {
            return Err(format!(
                "ltx_2_5 trainer: audio reference `{key}` must be [B,C,T,F], got {shape:?}"
            )
            .into());
        }
        let tokens = raw_reference.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            shape[2],
            channels * bins,
        ])?;
        let positions = create_audio_position_grid_with(
            batch as usize,
            shape[2] as usize,
            crate::positions::AUDIO_LATENT_SAMPLE_RATE,
            crate::positions::AUDIO_HOP_LENGTH,
            condition
                .spec
                .temporal_scale_factor
                .unwrap_or(crate::positions::AUDIO_LATENT_DOWNSAMPLE_FACTOR as usize)
                as i64,
            true,
        );
        let count = tokens.shape()[1] as usize;
        (Some(tokens), Some(positions), count)
    } else {
        (None, None, 0)
    };
    let specs: Vec<Ltx25ConditionSpec> = modality
        .conditions
        .iter()
        .map(|condition| condition.spec)
        .collect();
    let token_plan = build_ltx25_token_plan_from_specs(
        modality.is_generated,
        &specs,
        geometry,
        Ltx25ConditionInputs {
            conditioning_mask: mask.as_deref(),
            reference_tokens,
            seed,
            sample_index,
        },
    )?;
    Ok(CachedLtx25Modality {
        clean,
        context: context.clone(),
        positions: create_audio_position_grid(batch as usize, frames as usize),
        reference,
        reference_positions,
        token_plan,
    })
}

fn load_prepared_example(
    item: &mlx_gen::TrainingItem,
    plan: &Ltx25TrainingPlan,
    video_context: &Array,
    audio_context: Option<&Array>,
    seed: u64,
    sample_index: u64,
    depth_frames: Option<usize>,
) -> Result<CachedLtx25Example> {
    let path = prepared_bundle_path(item)?;
    let weights = Weights::from_file(path)?;
    validate_prepared_bundle_contract(&weights, path, plan)?;
    let fps = if plan.video.is_some() {
        prepared_fps(&weights)? as f32
    } else {
        crate::positions::DEFAULT_FPS
    };
    let video = plan
        .video
        .as_ref()
        .map(|modality| {
            prepared_video_modality(&weights, modality, video_context, fps, seed, sample_index)
        })
        .transpose()?;
    let audio = plan
        .audio
        .as_ref()
        .map(|modality| {
            let context = audio_context.ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: audio workflow has no audio text context".into(),
                )
            })?;
            prepared_audio_modality(&weights, modality, context, seed, sample_index)
        })
        .transpose()?;
    // sc-24830: the depth frames come from the generated video's own token plan (frames with loss
    // tokens), fixed per example.
    let depth = match (depth_frames, video.as_ref()) {
        (Some(k), Some(video)) if plan.video.as_ref().is_some_and(|v| v.is_generated) => {
            let shape = parse_shape_metadata(
                &weights,
                "videoShape",
                &weights.require("video_latents")?.clone(),
            )?;
            let (c, f, h, w) = (shape[1], shape[2], shape[3], shape[4]);
            let frames = select_depth_frames(
                &video.token_plan.loss_mask[..video.token_plan.target_tokens],
                f as usize,
                (h * w) as usize,
                k,
            );
            if frames.is_empty() {
                return Err(format!(
                    "ltx_2_5 trainer: depth anchoring found no generated video frame in example \
                     {sample_index}"
                )
                .into());
            }
            Some(Ltx25DepthFrames {
                grid: [c, f, h, w],
                frames,
            })
        }
        _ => None,
    };
    let example = CachedLtx25Example {
        video,
        audio,
        depth,
    };
    let mut arrays: Vec<&Array> = Vec::new();
    for modality in [&example.video, &example.audio].into_iter().flatten() {
        arrays.extend([&modality.clean, &modality.context, &modality.positions]);
        if let Some(reference) = &modality.reference {
            arrays.push(reference);
        }
        if let Some(positions) = &modality.reference_positions {
            arrays.push(positions);
        }
    }
    eval(arrays)?;
    Ok(example)
}

fn prepare_cached_ltx25(
    cached: &CachedLtx25Example,
    sigma: f32,
    seed: u64,
) -> Result<Ltx25PreparedBatch> {
    let prepare = |modality: &CachedLtx25Modality, salt: u64| -> Result<Ltx25PreparedModality> {
        let noise = random::normal::<f32>(
            modality.clean.shape(),
            None,
            None,
            Some(&random::key(seed.wrapping_add(salt))?),
        )?;
        prepare_ltx25_modality(
            &modality.clean,
            &noise,
            &modality.context,
            &modality.positions,
            modality.reference.as_ref(),
            modality.reference_positions.as_ref(),
            sigma,
            &modality.token_plan,
        )
    };
    Ok(Ltx25PreparedBatch {
        video: cached
            .video
            .as_ref()
            .map(|value| prepare(value, 1))
            .transpose()?,
        audio: cached
            .audio
            .as_ref()
            .map(|value| prepare(value, 2))
            .transpose()?,
    })
}

/// The reference `inject_video_attention_lora` default targets (`DEFAULT_LORA_TARGET_MODULES`,
/// `training_adapters.py:72`), restricted to `attn1`/`attn2`. `to_out.0` is the diffusers spelling
/// the inference loader normalizes to the checkpoint's `to_out`.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// One LoRA-trained attention `Linear`: its diffusers save spelling (e.g. `…attn1.to_out.0`), the
/// resolution segments after the `to_out.0`→`to_out` normalization, and the factor-map keys.
#[derive(Clone)]
struct LtxLoraTarget {
    save_path: String,
    segs: Vec<String>,
    a_key: Rc<str>,
    b_key: Rc<str>,
}

/// LoRA trainer for LTX-2.3 and the split Gemma-4 LTX-2.5 route, implementing the core
/// [`Trainer`] surface: a frozen LtxDiT (f32 activations × Q4/Q8 weights) + VAE + text encoder +
/// tokenizer that caches a captioned image dataset to (normalized latent, prompt-embed) pairs,
/// then runs the functional-autograd rectified-flow loop with the sc-3043 runtime glue, and writes
/// a LoRA that round-trips through [`crate::apply_ltx_adapters`].
///
/// **Single-use** (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB) after the
/// embed cache, so the instance cannot run a second job — `validate` (hence `train`) rejects a reuse
/// up front. Construct a fresh trainer (via [`load_trainer`]) per job.
enum TrainingTokenizer {
    Gemma3(LtxTokenizer),
    Gemma4(Ltx25Tokenizer),
}

impl TrainingTokenizer {
    fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        match self {
            Self::Gemma3(tokenizer) => tokenizer.encode(prompt, max_length),
            Self::Gemma4(tokenizer) => tokenizer.encode(prompt, max_length),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum TrainingTextEncoder {
    Gemma3(LtxTextEncoder),
    Gemma4(Ltx25TextEncoder),
}

impl TrainingTextEncoder {
    fn encode(&self, ids: &Array, mask: &Array) -> Result<Array> {
        match self {
            Self::Gemma3(encoder) => encoder.encode(ids, mask),
            Self::Gemma4(encoder) => encoder.encode(ids, mask),
        }
    }

    fn encode_av(&self, ids: &Array, mask: &Array) -> Result<(Array, Option<Array>)> {
        match self {
            Self::Gemma3(encoder) => Ok((encoder.encode(ids, mask)?, None)),
            Self::Gemma4(encoder) => {
                let (video, audio) = encoder.encode_av(ids, mask)?;
                Ok((video, Some(audio)))
            }
        }
    }
}

pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    /// Freed after the one-time prompt-embed cache (the 24 GB Gemma backbone), before the loop.
    tokenizer: Option<TrainingTokenizer>,
    text_encoder: Option<TrainingTextEncoder>,
    vae: LtxVideoVae,
    transformer: TrainingTransformer,
    cfg: LtxConfig,
}

#[allow(clippy::large_enum_variant)]
enum TrainingTransformer {
    Ltx23(LtxDiT),
    Ltx25(AvDiT),
}

impl TrainingTransformer {
    #[cfg(test)]
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        match self {
            Self::Ltx23(dit) => dit.set_sdpa_checkpoint(on),
            Self::Ltx25(dit) => dit.set_sdpa_checkpoint(on),
        }
    }

    #[cfg(test)]
    fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        match self {
            Self::Ltx23(dit) => dit.cast_weights(dtype),
            Self::Ltx25(_) => Err("ltx_2_5 test helper cannot cast the full AV base".into()),
        }
    }
}

#[cfg(test)]
impl std::ops::Deref for TrainingTransformer {
    type Target = LtxDiT;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

#[cfg(test)]
impl std::ops::DerefMut for TrainingTransformer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

fn trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_ID)
}

fn trainer_descriptor_25() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_25_ID)
}

fn trainer_descriptor_for(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        // The reference LTX MLX trainer is LoRA-only; LoKr training is unsupported (see `validate`).
        supports_lokr: false,
        // No control-branch training path (F-006). The LTX trainer must reject a control request
        // rather than silently training a plain LoRA (F-055).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets on the 2.3 still-image path only (one cached
        // latent + RoPE grid per spatial edge). 2.5 trains on externally prepared latent packs
        // (`ltxPreparedBundlePath`) whose `videoShape`/`audioShape`, condition masks and token plan
        // are fixed by the pack — there are no pixels to re-encode at another edge and
        // `cfg.resolution` is unused — so it keeps `NONE` and the shared floor refuses buckets.
        // sc-24828 (epic 2123): LTX-2.3 trains on decoded, centre-cropped images and honours the
        // subject-masked loss (one weight per cached (item, bucket) latent). LTX-2.5 trains only on
        // preprocessed latent bundles with no image aligned to the latent, so it does not declare
        // it (refused with the reason by [`refuse_ltx25_subject_mask`]).
        // sc-24830 (epic 2123): depth anchoring through TAELTX2.3 (upstream's tiny decoder for
        // LTX-2.3 and LTX-2.5) — on the LTX-2.3 still-image path (dense and block-checkpointed)
        // and on every LTX-2.5 workflow that generates video (the prepared bundle's clean video
        // latent's selected frames decoded, each `k > 0` with its predecessor; audio ignored). The
        // six 2.5 workflows with no generated video are
        // refused with the reason by [`refuse_ltx25_depth_anchoring`].
        // sc-24833 (epic 2123): the VAE anchor (same TAELTX2.3 decode → FLUX.2 encoder taps, per
        // decoded frame) wherever depth anchoring runs — through the same builder and the same
        // video-workflow refusal. No E-LatentLPIPS: no published weights match this latent family.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: id != MODEL_25_ID,
            subject_mask_loss: id == MODEL_ID,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring.
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// sc-24828 — LTX-2.5 cannot honour subject-masked loss: every one of its workflows trains on a
/// preprocessed video/audio latent bundle (`model_options`), with no decodable image whose crop is
/// aligned to the latent, so a single image mask has nothing to line up with. A typed
/// `Unsupported` naming that reason, raised by `validate`, the start of `train` (before any
/// caching) and the weights-free [`validate_ltx25_training_request`] preflight — ahead of the
/// generic technique floor, so the caller sees why.
fn refuse_ltx25_subject_mask(id: &str, req: &TrainingRequest) -> gen_core::Result<()> {
    if id == MODEL_25_ID && req.config.subject_mask_loss.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: subject-masked loss needs a training image aligned to the latent, but \
             LTX-2.5 trains on preprocessed video/audio latent bundles that carry none; turn \
             subject-masked loss off for this trainer"
        )));
    }
    // sc-24832: the subject-restricted normal loss reads the same (absent) aligned mask.
    if id == MODEL_25_ID
        && mlx_gen::train::subject_mask::PerceptualSubjectMasks::needed(&req.config)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: the normal loss restricted to the subject needs a training image aligned \
             to the latent, but LTX-2.5 trains on preprocessed latent bundles that carry none; turn \
             body_losses.normal_restrict_to_subject off for this trainer"
        )));
    }
    Ok(())
}

mod ltx25_restricted_normal_tests;

/// sc-24830 — LTX-2.5 runs depth anchoring on its generated video stream, so it refuses it for the
/// six workflows that generate no video (`v2a_lora`, whose video is frozen conditioning, and the
/// audio-only `t2a_lora`, `audio_extend_lora`, `audio_inpainting_lora`, `audio_suffix_lora`,
/// `a2a_ic_lora`): a typed `Unsupported` naming the workflow, raised by `validate`, the start of
/// `train` (before any caching) and the weights-free [`validate_ltx25_training_request`] preflight
/// — ahead of the generic technique floor, so the caller sees why. Also validates the
/// [`DEPTH_ANCHORING_FRAMES_KEY`] knob.
fn refuse_ltx25_depth_anchoring(id: &str, req: &TrainingRequest) -> gen_core::Result<()> {
    // Every decoded-x0 perceptual loss (depth anchoring, the sc-24831 identity / face-landmark
    // losses, the sc-24833 VAE anchor, the sc-24832 body losses) decodes the same generated video
    // frames, so the refusal covers whichever ones are on and names them.
    let pixel_losses = mlx_gen_perceptual::enabled_pixel_aux_losses(&req.config);
    if id != MODEL_25_ID || pixel_losses.is_empty() {
        return Ok(());
    }
    depth_anchoring_frames(&req.config)?;
    // The resolved plan (per-request modality overrides applied) when it parses; else the named
    // workflow's canonical plan — an otherwise-malformed request is reported by the plan
    // validation that follows, but never slips a video-less workflow past this refusal.
    let (workflow, video) = match Ltx25TrainingPlan::from_request(req) {
        Ok(plan) => (plan.workflow, plan.video.map(|v| v.is_generated)),
        Err(_) => {
            let Some(workflow) = req
                .config
                .model_options
                .get("ltxWorkflow")
                .and_then(serde_json::Value::as_str)
                .and_then(|id| Ltx25Workflow::parse(id).ok())
            else {
                return Ok(());
            };
            (workflow, workflow.plan().video.map(|v| v.is_generated))
        }
    };
    if video != Some(true) {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: depth anchoring and the other decoded-x0 losses ({}) decode the \
             generated video stream, but workflow `{}` generates no video (the video stream is \
             {}); turn them off for this workflow",
            pixel_losses.join(", "),
            workflow.id(),
            if video.is_some() {
                "frozen conditioning"
            } else {
                "absent"
            }
        )));
    }
    Ok(())
}

/// `model_options` key selecting how many latent frames of an LTX-2.5 video clip the depth loss
/// decodes per step (sc-24830). Each selected latent frame decodes through TAELTX2.3 to one pixel
/// frame — frame 0 as a `T = 1` clip, a frame `k > 0` together with its predecessor `k − 1` as a
/// `T = 2` clip, keeping its last output frame; the frames are evenly spaced over the clip's latent
/// frames that carry generated (loss) tokens, fixed per example. Positive integer; default
/// [`DEFAULT_DEPTH_ANCHORING_FRAMES`]. Bounds the per-step decode + Depth-Anything-V2 memory.
pub const DEPTH_ANCHORING_FRAMES_KEY: &str = "depthAnchoringFrames";

/// The default [`DEPTH_ANCHORING_FRAMES_KEY`]: two latent frames per step.
pub const DEFAULT_DEPTH_ANCHORING_FRAMES: usize = 2;

/// The validated [`DEPTH_ANCHORING_FRAMES_KEY`] of `cfg` (default when absent).
fn depth_anchoring_frames(cfg: &TrainingConfig) -> Result<usize> {
    match cfg.model_options.get(DEPTH_ANCHORING_FRAMES_KEY) {
        None => Ok(DEFAULT_DEPTH_ANCHORING_FRAMES),
        Some(value) => value
            .as_u64()
            .filter(|&n| n >= 1)
            .map(|n| n as usize)
            .ok_or_else(|| {
                mlx_gen::Error::Msg(format!(
                    "ltx_2_5 trainer: `{DEPTH_ANCHORING_FRAMES_KEY}` must be a positive integer, \
                     got {value}"
                ))
            }),
    }
}

/// The latent frames of a `(F, H, W)`-token clip the depth loss decodes (sc-24830): `k` frames
/// evenly spaced over the frames holding at least one generated (`loss_mask > 0`) target token —
/// conditioning frames would decode their clean latent and train nothing. Empty when no frame is
/// generated.
fn select_depth_frames(
    loss_mask: &[f32],
    frames: usize,
    frame_tokens: usize,
    k: usize,
) -> Vec<i32> {
    let candidates: Vec<i32> = (0..frames)
        .filter(|&f| {
            loss_mask[f * frame_tokens..(f + 1) * frame_tokens]
                .iter()
                .any(|&m| m > 0.0)
        })
        .map(|f| f as i32)
        .collect();
    let n = candidates.len();
    if k >= n {
        return candidates;
    }
    if k == 1 {
        return vec![candidates[n / 2]];
    }
    let mut picked: Vec<i32> = (0..k)
        .map(|i| candidates[(i * (n - 1) + (k - 1) / 2) / (k - 1)])
        .collect();
    picked.dedup();
    picked
}

/// Construct the trainer from an LTX-2.3 split-weight snapshot directory (transformer / VAE /
/// connector + the Gemma-3-12B text-encoder snapshot resolved like inference). The transformer loads
/// at **f32 activations × quantized weights** (`quant_f32`) for clean autograd — the base is frozen,
/// gradients flow only through the trainable LoRA factors. Registered via [`mlx_gen::TrainerRegistration`].
///
/// The weights load lazily (sc-2124): construction only checks the spec (and that the Gemma-3
/// override exists), so `validate` and `train`'s refusal floors never read weights; see
/// [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_23(spec)?;
    crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_23(&spec)
        },
    )))
}

/// The LTX-2.3 snapshot directory a trainer spec names — a single file is refused.
fn snapshot_root_23(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_3 trainer expects a split-weight snapshot directory (transformer.safetensors \
             / vae_*.safetensors / connector.safetensors), not a single file"
                .into(),
        )),
    }
}

/// The LTX-2.3 weight load behind [`load_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights_23(spec: &LoadSpec) -> Result<LtxTrainer> {
    load_trainer_from_dir(snapshot_root_23(spec)?, spec.text_encoder.as_ref())
}

/// Construct the LTX-2.5 trainer from its split Gemma-4 bundle.  This follows the ordinary
/// provider's component resolver so training cannot quietly combine a 2.5 DiT with a Gemma-3
/// override or a 2.3-shaped VAE configuration.
///
/// The weights load lazily (sc-2124): construction only checks the spec, so `validate` and `train`'s
/// refusal floors never read weights; see [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer_25(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_25(spec)?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor_25(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_25(&spec)
        },
    )))
}

/// The LTX-2.5 split-component directory a trainer spec names — a single checkpoint is refused.
fn snapshot_root_25(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(path) => Ok(path),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_5 trainer expects a split-component directory, not a single checkpoint".into(),
        )),
    }
}

/// The LTX-2.5 weight load behind [`load_trainer_25`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights_25(spec: &LoadSpec) -> Result<LtxTrainer> {
    let root = snapshot_root_25(spec)?;
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    ensure_ltx25_training_variant(crate::dev_sampler::from_bundle(&bundle)?)?;
    let split = SplitModel::from_model_dir(root)?;
    ensure_ltx25_training_tier(split)?;
    let cfg = LtxConfig::from_bundle(&bundle)?;
    let vae_cfg = LtxVaeConfig::from_bundle(&bundle, LtxComponent::ConvVideoVae)?;
    let transformer_w = Weights::from_file(bundle.require(LtxComponent::Transformer)?.path())?;
    let video_w = Weights::from_file(bundle.require(LtxComponent::ConvVideoVae)?.path())?;
    let connector_path = bundle
        .require(LtxComponent::Transformer)?
        .path()
        .with_file_name("connector.safetensors");
    if !connector_path.is_file() {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: missing connector.safetensors beside {}",
            bundle.require(LtxComponent::Transformer)?.path().display()
        )));
    }
    let connector_w = Weights::from_file(&connector_path)?;
    let text_encoder = Ltx25TextEncoder::from_bundle_av(
        &bundle,
        &connector_w,
        &cfg,
        Precision::quant_bf16(split.bits, split.group),
        mlx_gen::gen_core::OffloadPolicy::Resident,
    )?;
    let tokenizer =
        Ltx25Tokenizer::from_packed_te_file(bundle.require(LtxComponent::TextEncoder)?.path())?;
    let transformer = AvDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    // Converted 2.5 tiers split the encoder into a sibling file; raw upstream split bundles keep
    // it inside the ConvVAE component.  Match the ordinary provider's documented preference rather
    // than making training depend on one publication layout.
    let split_encoder = root.join("vae_encoder.safetensors");
    let encoder_path = if split_encoder.is_file() {
        split_encoder
    } else {
        bundle
            .require(LtxComponent::ConvVideoVae)?
            .path()
            .to_path_buf()
    };
    let vae = LtxVideoVae::from_weights_lazy_encoder(&video_w, encoder_path, &vae_cfg)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [&connector_w, &transformer_w, &video_w] {
        w.materialize_accessed()?;
    }
    Ok(LtxTrainer {
        descriptor: trainer_descriptor_25(),
        tokenizer: Some(TrainingTokenizer::Gemma4(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma4(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx25(transformer),
        cfg,
    })
}

/// The concrete-typed loader behind [`load_trainer`] (sc-4942 — the first-step memory harness needs
/// the concrete [`LtxTrainer`] to reach `.transformer` / `.vae`, which a `Box<dyn Trainer>` hides).
///
/// `te_override` is `LoadSpec::text_encoder` — the bundled Gemma-3 dir the self-contained LTX install
/// ships beside the tier weights (sc-8827/sc-9989). Threaded into [`resolve_gemma_dir`] exactly as the
/// inference path does (`model.rs`), which as of sc-13664 **requires** the slot: `None` is a load-time
/// error (no env / HF-cache fallback), so the trainer must be handed an explicit TE dir.
fn load_trainer_from_dir(root: &Path, te_override: Option<&WeightsSource>) -> Result<LtxTrainer> {
    // Resolve (and validate) the Gemma-3 TE location up front — an absent/bad `LoadSpec::text_encoder`
    // override fails fast here, ahead of the split-weight load, and keeps the wiring unit-testable
    // without the ~24 GB base snapshot (sc-9989). Only the path is resolved here; the heavy
    // `Weights::from_dir` load stays below with the rest of the component loads.
    let gemma_dir = crate::model::resolve_gemma_dir(te_override)?;

    let split = SplitModel::from_model_dir(root)?;
    let cfg = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let gemma_quant = crate::model::resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let vae_dec_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // Video-only text encoder (bf16, the reference TE dtype); we cast its embeds to f32 per-item for
    // the f32 training forward.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &cfg,
        // bf16 activations at the checkpoint's own quant geometry: the connector and the
        // feature-extractor Linear take the packed arm on an LTX-2.5 tier and the dense arm on
        // LTX-2.3's dense `connector.safetensors`, decided per tensor by `.scales` presence.
        Precision::quant_bf16(split.bits, split.group),
    )?;
    let transformer = LtxDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    let vae = LtxVideoVae::from_weights(&vae_dec_w, Some(&vae_enc_w), &vae_config)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [
        &gemma_w,
        &connector_w,
        &transformer_w,
        &vae_dec_w,
        &vae_enc_w,
    ] {
        w.materialize_accessed()?;
    }
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(LtxTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: Some(TrainingTokenizer::Gemma3(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma3(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx23(transformer),
        cfg,
    })
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION_25 = trainer_descriptor_25 => load_trainer_25
}

/// Capability-free request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer. Rejects an empty dataset, zero rank, LoKr (LoRA-only
/// trainer), and unsupported optimizers. The single-use / text-encoder-present check stays in
/// [`Trainer::validate`], which has the trainer state to inspect (F-055).
fn validate_request(req: &TrainingRequest, label: &str) -> Result<()> {
    if req.items.is_empty() {
        return Err(format!("{label}: dataset is empty").into());
    }
    if req.config.rank == 0 {
        return Err(format!("{label}: rank must be > 0").into());
    }
    if req.config.network_type == NetworkType::Lokr {
        return Err(format!(
            "{label}: LoKr training is not supported — the reference LTX MLX \
                    trainer is LoRA-only. (LTX *inference* supports LoKr via sc-2393, but no \
                    LoKr trainer exists yet; that would be a separate extension.)"
        )
        .into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "{label}: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

/// Weights-free (with respect to the multi-gigabyte base model) LTX-2.5 request preflight. Product
/// callers can run this before constructing a trainer: it resolves the complete workflow and Dev
/// validation contract, requires every prepared-pack path, and validates each safetensors header,
/// schema, modality shape and tensor-backed condition key. The same function is the request half of
/// [`LtxTrainer::validate`], so preflight and execution cannot drift.
pub fn validate_ltx25_training_request(req: &TrainingRequest) -> Result<()> {
    let descriptor = trainer_descriptor_25();
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // The shared floors keep their typed variant across the seam (`?` maps
    // `gen_core::Error::Unsupported` 1:1): a capability gap must stay `Unsupported` for the worker,
    // never be flattened to a message (sc-24161).
    gen_core::train::validate_control_request(&descriptor, req)?;
    gen_core::train::validate_full_finetune_request(&descriptor, req)?;
    gen_core::train::validate_training_techniques(&descriptor, req)?;
    gen_core::train::validate_edit_request(&descriptor, req)?;
    validate_request(req, "ltx_2_5 trainer")?;
    validate_ltx25_adapter_scale(req.config.alpha)?;
    let plan = Ltx25TrainingPlan::from_request(req)?;
    for item in &req.items {
        let path = prepared_bundle_path(item)?;
        if !path.is_file() {
            return Err(format!(
                "ltx_2_5 trainer: prepared bundle does not exist: {}",
                path.display()
            )
            .into());
        }
        let weights = Weights::from_file(path)?;
        validate_prepared_bundle_contract(&weights, path, &plan)?;
    }
    Ok(())
}

/// Every weights-free [`Trainer::validate`] floor — all of it but the single-use check.
fn validate_floors(descriptor: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    // Shared control-training floor (F-006 / F-055): the LTX trainer has no control branch, so a
    // request carrying `control_type` / per-item control images is rejected (typed `Unsupported`)
    // rather than silently training a plain LoRA and reporting success.
    gen_core::train::validate_control_request(descriptor, req)?;
    // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
    // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
    // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)?;
    if descriptor.id == MODEL_25_ID {
        validate_ltx25_training_request(req)?;
    } else {
        validate_request(req, &format!("{} trainer", descriptor.id))?;
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Single-use enforcement (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB)
        // after the embed cache, so a second `train` on the same instance can't re-encode. Fail here,
        // up front (validate runs before any progress is emitted), instead of with a late, confusing
        // "text encoder missing" mid-run. Construct a fresh trainer (via `load_trainer`) per job.
        let label = format!("{} trainer", self.descriptor.id);
        if self.text_encoder.is_none() || self.tokenizer.is_none() {
            return Err(format!(
                "{label}: single-use — the Gemma text encoder was freed after the \
                        first train() to reclaim ~24 GB; construct a fresh trainer for each job"
            )
            .into());
        }
        validate_floors(self.descriptor(), req)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        refuse_ltx25_subject_mask(self.descriptor.id, req)?;
        refuse_ltx25_depth_anchoring(self.descriptor.id, req)?;
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl LtxTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        if matches!(self.transformer, TrainingTransformer::Ltx25(_)) {
            return self.train_25_impl(req, on_progress);
        }
        self.train_23_impl(req, on_progress)
    }

    fn train_23_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let TrainingTransformer::Ltx23(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.3 path")
        };
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one pixel edge (multiple of 32) per resolution bucket (just `[resolution]` when
        // buckets are off), and its latent tokens per side. The memory guard and previews size for
        // the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let latent_edges: Vec<usize> = edges.iter().map(|&e| latent_edge_of(e)).collect();
        let latent_edge = guard_latent_edge(&edges);

        // sc-4942 — LTX trains in **f32 activations** (× the Q-packed base), NOT bf16, even though the
        // SceneWorks worker passes `train_dtype=bf16` (sc-4881). MEASURED on real weights (the
        // `bf16_grads_direction_and_memory_vs_f32` harness): a bf16 activation cast DECORRELATES the
        // gradient from the f32 (quality) path — global cosine 0.31–0.45, with the early/deep K
        // projections of BOTH attentions pointing ~opposite — because the 48-block distilled DiT is
        // chaos-sensitive (the same reason inference uses `quant_f32`, not `quant_bf16`, for quality;
        // see `transformer::Precision`/`Mode::QuantF32`). bf16 would save ~30% memory (1024: 28 vs 43 GB)
        // but the f32 working set already fits the video tier with the checkpointing levers below
        // (1024 ≈ 43 GB attn-ckpt / 27 GB block-ckpt, measured 128 GB box; the guard auto-scales), so
        // honoring bf16 would trade training quality for memory we do not need. (`LtxDiT::cast_weights`
        // stays available — the bf16 harness that produced this finding exercises it — but the
        // production trainer never invokes it; this trainer is f32-only.)
        let compute_dtype = Dtype::Float32;

        // sc-4942 — fail-fast pre-flight memory guard (the sc-4874 mechanism, ported to LTX). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`; at
        // high resolution that working set can exceed unified memory and the OS hard-kills the worker
        // with an UNCATCHABLE SIGKILL. We predict it and refuse up front with an actionable, catchable
        // error — BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not
        // enabled. (LTX is LoRA-only, so the LoRA-path condition is always met.)
        let will_checkpoint = cfg.gradient_checkpointing;
        // Epic 2123 E7 (sc-24830): the aux-loss models (TAELTX2.3 + every enabled arm)
        // count against the budget on BOTH paths — a checkpointed depth job must not skip it.
        let aux_gb = perceptual_footprint_gb(
            cfg,
            edges.iter().copied().max().unwrap_or(0),
            req.items.len() * edges.len(),
        );
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(cfg, latent_edge, aux_gb, will_checkpoint)?;
        }

        // Epic 2123 depth anchoring (sc-24830): load the frozen TAELTX2.3 decoder +
        // Depth-Anything-V2 before the caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: normalized latents + prompt embeds (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Per entry: patchified clean
        // latent, prompt embeds, and (subject-masked loss, sc-24828) that latent's patchified
        // loss-weight map — `None` when the technique is off.
        let mut cache: Vec<(Array, Array, Option<Array>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        // sc-5637 — preview-sample prompts, pre-encoded inside the `te`/`tok` scope below (the Gemma
        // encoder is freed before the train loop). LTX is distilled (no CFG) → one ctx per prompt.
        let mut sample_ctxs: Vec<(String, Array)> = Vec::new();
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_3 trainer: text encoder missing".into())
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_3 trainer: tokenizer missing".into()))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                // sc-24828: the item's mask is read and checked once, resampled per bucket below.
                let subject_mask = PreparedSubjectMask::load_if_enabled(
                    "ltx_2_3 trainer",
                    item,
                    cfg.subject_mask_loss.as_ref(),
                )?;
                let (ids, mask) = tok.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?; // (1, L, 4096)
                eval([&ctx])?;
                for (clean, mask_weight) in encode_bucket_latents(
                    &edges,
                    |edge| {
                        let prep = preprocess_conditioning_image(&img, edge, edge)?; // (1,3,1,edge,edge)
                        self.vae.encode(&prep) // (1,128,1,le,le), normalized, f32
                    },
                    |shape| still_subject_weight("ltx_2_3 trainer", subject_mask.as_ref(), shape),
                )? {
                    cache.push((clean, ctx.clone(), mask_weight));
                }
            }
            // sc-5637 — pre-encode the preview-sample prompts while the encoder is still resident.
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tok.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?;
                    eval([&ctx])?;
                    sample_ctxs.push((prompt.clone(), ctx));
                }
            }
        }
        if cache.is_empty() {
            // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
            // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
            // cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("ltx_2_3 trainer: no usable dataset items".into());
        }
        // Free the Gemma text encoder + tokenizer (~24 GB) before training — they are only needed for
        // the one-time embed cache (mirrors the reference `prepare_dataset` release).
        self.text_encoder = None;
        self.tokenizer = None;

        let sampling_enabled = !sample_ctxs.is_empty();

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAELTX2.3 decode of its
        // cached clean latent → DA2 depth) is computed exactly once per job, by the `AuxDriver`
        // before the loop.
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "ltx trainer",
                &req.items,
                cfg,
                latent_edges.len(),
                CropBox::center_square,
            )?);
        }

        // The RoPE position grid is identical across items at a fixed latent resolution (single
        // frame) — build it once per bucket (sc-2127: each step uses its sampled bucket's grid).
        // Preview-sample rendering (sc-5637) uses the largest bucket's grid at `latent_edge`.
        let bucket_positions: Vec<Array> = latent_edges
            .iter()
            .map(|&le| create_position_grid(1, 1, le, le))
            .collect();
        let preview_positions = &bucket_positions[largest_bucket(&latent_edges)];

        // --- adapter targets + trainable factors ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_3 trainer: no LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-4942 — gradient checkpointing. Group the resolved targets by their owning block so the
        // checkpointed forward can thread each block's LoRA factors as explicit recompute inputs. Every
        // target of this trainer lives in a `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping
        // covers the whole adapter surface.
        let n_layers = self.cfg.num_layers as usize;
        let block_targets = group_block_targets(&targets, n_layers);
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing" toggle),
        // never auto-forced — a run that would OOM is caught instead by the fail-fast pre-flight guard
        // above, which recommends this flag rather than silently changing the user's training dynamics.
        let use_checkpoint = cfg.gradient_checkpointing;
        let checkpoint_block: Option<&[Vec<BlockLoraRef>]> = if use_checkpoint {
            Some(&block_targets)
        } else {
            None
        };
        // sc-4942 — attention-segment checkpointing is on for the dense (non-block-checkpointed) path:
        // it is numerically identical to the retained backward (same decomposed attention, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every torch
        // trainer gets from its fused SDPA kernel. When whole-block checkpointing is on it goes OFF (the
        // block recompute already covers attention; nesting would recompute it twice for no win).
        transformer.set_sdpa_checkpoint(!use_checkpoint);

        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[F-125] resuming from step {start_step} (optimizer update {update_idx})"
                );
            }
        }

        // --- train loop ---
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: the perceptual alternation interleaves optimizer windows over the
        // schedule's epochs (sc-2124) — a pure function of the step, so a resumed run needs no
        // replay.
        let mut aux_driver = match perceptual {
            Some(path) => Some(aux_driver(
                path,
                &cache,
                &latent_edges,
                &schedule,
                accum,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (entry, bucket) = step_entry(&schedule, step);
            let (clean, ctx, mask_weight) = &cache[entry];
            let positions = &bucket_positions[bucket];
            // σ ~ U(1e-3, 1-1e-3), deterministic in seed (the reference's uniform timestep).
            let mut sigma = {
                let k = random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3f32, 1.0 - 1e-3, &[1], Some(&k))?.item::<f32>()
            };
            // Epic 2123 E8: plan the step's loss terms; an aux-only step trains at the σ remapped
            // into the loss window.
            let planned = match aux_driver.as_mut() {
                Some(d) => d.sample(step, &schedule).plan(sigma)?,
                None => None,
            };
            if let Some(p) = &planned {
                sigma = p.plan.noise_level;
            }
            let aux = planned.as_ref().map(|p| AuxStep {
                path: p.path,
                plan: &p.plan,
                image: p.entry,
                latent_edge: latent_edges[bucket],
            });
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (losses, grads) = compute_step_loss_grads(
                transformer,
                &params,
                &targets,
                alpha,
                rank,
                clean,
                ctx,
                positions,
                sigma,
                &noise,
                mae,
                mask_weight.as_ref(),
                checkpoint_block,
                compute_dtype,
                aux,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // F-017: average by the ACTUAL in-window count, not the full `accum`. The final-step
                // flush is usually a partial window (cfg.steps % accum != 0); dividing by `accum`
                // down-scaled that update (halved effective LR on the tail). Mirrors z-image/lens
                // F-069. (When step%accum==0 the window is the full `accum`.)
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx, cfg.seed)?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_lora(&params, &targets, alpha, cfg.rank, &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort preview frames from the in-progress adapter. Install the
            // current factors concretely for the forward-only render (the next step's traced `loss_fn`
            // re-installs them); a render failure logs and is skipped, never failing the training run.
            if sampling_enabled && cfg.sample_every > 0 && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                install_train_lora(transformer, &params, &targets, alpha, rank, lora_dtype)?;
                let total = sample_ctxs.len() as u32;
                for (i, (prompt, ctx)) in sample_ctxs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        transformer,
                        &self.vae,
                        ctx,
                        preview_positions,
                        sample_seed,
                        latent_edge,
                        compute_dtype,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => eprintln!(
                            "[sc-5637] {MODEL_ID} preview sample failed at step {step} \
                             (prompt {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a no-op adapter.
        // Surface the typed `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`)
        // rather than writing a valid-looking `.safetensors` and returning `Ok` — downstream tooling
        // would otherwise ship an identity LoRA as a trained artifact (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }

    fn train_25_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let plan = Ltx25TrainingPlan::from_request(req)?;
        let cfg = &req.config;
        let compute_dtype = Dtype::Float32;
        on_progress(TrainingProgress::Preparing);
        // sc-24830 (epic 2123 E7 + E8): depth anchoring on the generated video stream — the
        // memory guard (decoded frames × the prepared video grid) and the frozen models load
        // BEFORE caching, so a missing checkpoint or an over-budget run fails fast.
        let depth_frames = if mlx_gen_perceptual::any_aux_loss(cfg) {
            let k = depth_anchoring_frames(cfg)?;
            let (f, h, w) = ltx25_video_grids(&req.items)?;
            let aux_gb = ltx25_perceptual_footprint_gb(
                cfg,
                h,
                w,
                k.min(f.max(1) as usize) as u32,
                req.items.len(),
            );
            ltx25_aux_memory_guard(cfg, aux_gb, get_memory_limit())?;
            Some(k)
        } else {
            None
        };
        let perceptual = load_perceptual_path_for(cfg, "ltx_2_5 trainer", ltx25_decoder())?;
        on_progress(TrainingProgress::LoadingModel);

        let mut cached = Vec::with_capacity(req.items.len());
        let mut sample_contexts: Vec<(String, Array, Array)> = Vec::new();
        let mut negative_contexts: Option<(Array, Array)> = None;
        {
            let encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: text encoder missing".into())
            })?;
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_5 trainer: tokenizer missing".into()))?;
            for (index, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: index as u32 + 1,
                    total: req.items.len() as u32,
                });
                let (ids, mask) = tokenizer.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let (video_context, audio_context) = encoder.encode_av(&ids, &mask)?;
                let audio_context = audio_context.ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: Gemma-4 encoder has no audio connector".into(),
                    )
                })?;
                let video_context = to_dtype(&video_context, Dtype::Float32)?;
                let audio_context = to_dtype(&audio_context, Dtype::Float32)?;
                let example = load_prepared_example(
                    item,
                    &plan,
                    &video_context,
                    Some(&audio_context),
                    cfg.seed,
                    index as u64,
                    depth_frames,
                )?;
                eval([&video_context, &audio_context])?;
                cached.push(example);
            }
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let (video, audio) = encoder.encode_av(&ids, &mask)?;
                    let video = to_dtype(&video, Dtype::Float32)?;
                    let audio = to_dtype(
                        &audio.ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: validation requires audio text context".into(),
                            )
                        })?,
                        Dtype::Float32,
                    )?;
                    eval([&video, &audio])?;
                    sample_contexts.push((prompt.clone(), video, audio));
                }
                let (ids, mask) = tokenizer.encode("", MAX_PROMPT_TOKENS)?;
                let (video, audio) = encoder.encode_av(&ids, &mask)?;
                let video = to_dtype(&video, Dtype::Float32)?;
                let audio = to_dtype(
                    &audio.ok_or_else(|| {
                        mlx_gen::Error::Msg(
                            "ltx_2_5 trainer: validation requires negative audio context".into(),
                        )
                    })?,
                    Dtype::Float32,
                )?;
                eval([&video, &audio])?;
                negative_contexts = Some((video, audio));
            }
        }
        if cached.is_empty() {
            return if req.cancel.is_cancelled() {
                Err(mlx_gen::Error::Canceled)
            } else {
                Err("ltx_2_5 trainer: no usable prepared examples".into())
            };
        }
        self.text_encoder = None;
        self.tokenizer = None;

        let TrainingTransformer::Ltx25(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.5 path")
        };
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|value| value.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets_25(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_5 trainer: no AV LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        // AV whole-block checkpointing is not yet a separate path; the attention-segment checkpoint
        // still bounds all six attention score graphs while preserving the full AV forward.
        transformer.set_sdpa_checkpoint(true);

        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut optimizer =
            TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
        let accumulation = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accumulation, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("lora")
            .to_string();
        let mut update_index = 0;
        let mut start_step = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, metadata) = checkpoint::load_resume(&snapshot, &mut optimizer)?;
                params = loaded;
                start_step = metadata.step;
                update_index = metadata.update_idx;
            }
        }

        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
        // sc-24830: each example's depth reference (its clean video frames decoded → DA2) once,
        // and the alternation over the round-robin windows (a pure function of the step, so a
        // resumed run needs no replay).
        let mut aux_driver = match perceptual {
            Some(path) => Some(ltx25_aux_driver(
                path,
                &cached,
                accumulation,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut final_loss = 0.0;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let mut sigma = {
                let key =
                    random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3, 1.0 - 1e-3, &[1], Some(&key))?.item::<f32>()
            };
            let item = (step as usize - 1) % cached.len();
            let example = &cached[item];
            // sc-24830: plan the step; an aux-only step trains at σ remapped into the loss window.
            let step_plan = match aux_driver.as_mut() {
                Some(driver) => {
                    let key = driver.key(step);
                    let plan = driver.path().plan(key, item, sigma)?;
                    sigma = plan.noise_level;
                    Some(plan)
                }
                _ => None,
            };
            let prepared = prepare_cached_ltx25(
                example,
                sigma,
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2),
            )?;
            let aux = match (
                aux_driver.as_ref().map(AuxDriver::path),
                step_plan.as_ref(),
                example.depth.as_ref(),
            ) {
                (Some(path), Some(plan), Some(depth)) => Some(Ltx25AuxStep {
                    path,
                    plan,
                    image: item,
                    depth,
                }),
                (Some(_), Some(_), None) => {
                    return Err("ltx_2_5 trainer: depth anchoring has no video frames".into())
                }
                _ => None,
            };
            let (losses, grads) = compute_ltx25_step_loss_grads(
                transformer,
                &params,
                &targets,
                cfg.alpha,
                cfg.rank as f32,
                &prepared,
                mae,
                compute_dtype,
                aux,
            )?;
            final_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;
            if step % accumulation == 0 || step == cfg.steps {
                optimizer.set_lr_scaled(lr_multiplier(
                    cfg.lr_scheduler,
                    update_index,
                    total_updates,
                    warmup_updates,
                ));
                let window = if step % accumulation == 0 {
                    accumulation
                } else {
                    step % accumulation
                };
                let average = average_grads(
                    accumulated.take().expect("update requires gradients"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &average,
                    cfg,
                    update_index,
                    cfg.seed,
                )?;
                update_index += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: final_loss,
            });
            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                save_lora(
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank,
                    &req.output_dir.join(checkpoint_filename(&stem, step)),
                )?;
                checkpoint::save_resume(
                    &req.output_dir,
                    &stem,
                    step,
                    update_index,
                    &optimizer,
                    &params,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            if cfg.sample_every > 0 && step % cfg.sample_every == 0 && !sample_contexts.is_empty() {
                install_train_lora(
                    transformer,
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank as f32,
                    None,
                )?;
                let negatives = negative_contexts.as_ref().ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: negative validation context missing".into(),
                    )
                })?;
                for (index, (prompt, video_context, audio_context)) in
                    sample_contexts.iter().enumerate()
                {
                    let image = render_ltx25_validation_sample(
                        transformer,
                        &self.vae,
                        video_context,
                        audio_context,
                        &negatives.0,
                        &negatives.1,
                        &plan.validation,
                        cfg.seed
                            .wrapping_add(step as u64)
                            .wrapping_add(index as u64),
                        &req.cancel,
                    )?;
                    on_progress(TrainingProgress::Sample {
                        step,
                        index: index as u32 + 1,
                        total: sample_contexts.len() as u32,
                        prompt: prompt.clone(),
                        image,
                    });
                }
            }
        }
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, cfg.alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss,
        })
    }
}

/// Flatten a single-frame VAE latent `(1, 128, 1, le, le)` to the patchified `(1, S, 128)` the DiT
/// consumes (`S = le·le`) — the reference's `transpose(reshape(latent, (B, C, -1)), (0, 2, 1))`.
fn flatten_latent(latent: &Array) -> Result<Array> {
    let sh = latent.shape(); // [1, 128, 1, le, le]
    let (b, c) = (sh[0], sh[1]);
    let s = sh[2] * sh[3] * sh[4];
    let flat = latent.reshape(&[b, c, s])?; // (1, 128, S)
    Ok(flat.transpose_axes(&[0, 2, 1])?) // (1, S, 128)
}

/// The inverse of [`flatten_latent`] for a square single-frame latent: the patchified
/// `(1, S, 128)` with `S = latent_edge²` back to the decoder's NCHW `(1, 128, le, le)` (the
/// per-channel-normalized space `LtxVideoVae::encode` produces).
fn unflatten_latent(flat: &Array, latent_edge: usize) -> Result<Array> {
    let sh = flat.shape();
    let le = latent_edge as i32;
    if sh.len() != 3 || sh[1] != le * le {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3 trainer: expected a patchified (B, {}, C) still latent, got {sh:?}",
            le * le
        )));
    }
    Ok(flat
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[sh[0], sh[2], le, le])?)
}

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
#[derive(Clone, Copy, Debug, PartialEq)]
struct StepLosses {
    /// The differentiated step loss.
    total: f32,
    /// The diffusion (velocity-regression) term, `None` on an aux-only step.
    diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`].
struct AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// The step's (item, bucket) cache entry (selects its cached reference).
    image: usize,
    /// The entry's latent tokens per side (to unpatchify the x0 estimate).
    latent_edge: usize,
}

/// The LTX-2.5 loop's [`AuxDriver`] (sc-24830): each example's depth reference (its clean video
/// target tokens at the selected frames, decoded) once per job, and the alternation over the
/// loop's round-robin order (`(step − 1) mod n`, an `n`-step epoch) with `accumulation`
/// micro-steps per update.
fn ltx25_aux_driver(
    path: PerceptualPath,
    cached: &[CachedLtx25Example],
    accumulation: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    let n = cached.len();
    AuxDriver::prepare_keyed(
        path,
        n,
        |index| {
            let example = &cached[index];
            let (Some(video), Some(depth)) = (example.video.as_ref(), example.depth.as_ref())
            else {
                return Err(format!(
                    "ltx_2_5 trainer: depth anchoring needs a generated video clip (example {index})"
                )
                .into());
            };
            depth.nchw(&video.clean)
        },
        n,
        accumulation,
        cancel,
    )
}

/// LTX-2.3's latent family for the shared aux-loss builder (epic 2123 E8): the 128-channel
/// LTX-2.3 VAE latent normalized by its per-channel statistics (`(μ − mean)/std`,
/// `LtxVideoVae::encode`), decoded by TAELTX2.3 — upstream's TAEHV checkpoint for LTX-2.3, which
/// takes exactly that normalized latent (its demo feeds `(z − latents_mean)/latents_std`). A still
/// image is one `T = 1` clip (8 grown frames, the last kept).
fn ltx_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Taehv {
        name: "TAELTX2.3",
        config: TaehvConfig::taeltx2_3(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    load_perceptual_path_for(cfg, "ltx_2_3 trainer", ltx_decoder())
}

/// [`load_perceptual_path`] with an explicit error label and decoder (the LTX-2.5 route uses its
/// own: [`ltx25_decoder`]).
fn load_perceptual_path_for(
    cfg: &TrainingConfig,
    label: &str,
    decoder: mlx_gen_perceptual::DecoderSpec,
) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label,
            decoder,
            latent_lpips: None,
        },
    )
}

/// The LTX-2.3 loop's [`AuxDriver`] (epic 2123 E8): every (item, bucket) cache entry's perceptual
/// reference computed once, from its patchified clean latent unpatchified at its bucket's latent
/// edge (item-major: bucket = entry % n_buckets); the alternation over the schedule's epochs with
/// `accum` micro-steps per update.
fn aux_driver(
    path: PerceptualPath,
    cache: &[(Array, Array, Option<Array>)],
    latent_edges: &[usize],
    schedule: &BucketSchedule,
    accum: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    AuxDriver::prepare(
        path,
        cache.len(),
        |entry| unflatten_latent(&cache[entry].0, latent_edges[entry % latent_edges.len()]),
        schedule,
        accum,
        cancel,
    )
}

/// Extra training memory (GB) the enabled perceptual losses add at the largest bucket pixel `edge`
/// (epic 2123 E7): TAELTX2.3 + the losses' frozen models plus `entries` cached references. `0`
/// when nothing is enabled.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &ltx_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// The largest prepared video `(H, W)` latent grid and the frame count over `items` (sc-24830) —
/// read from each bundle's `videoShape` header metadata alone (no tensor data) before caching, for
/// the depth memory estimate.
fn ltx25_video_grids(items: &[mlx_gen::TrainingItem]) -> Result<(i32, i32, i32)> {
    let mut max = (0, 0, 0);
    for item in items {
        let path = prepared_bundle_path(item)?;
        let metadata =
            mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(path).map_err(|error| {
                mlx_gen::Error::Msg(format!(
                    "ltx_2_5 trainer: could not read prepared bundle header `{}`: {error}",
                    path.display()
                ))
            })?;
        let raw = metadata.get("videoShape").ok_or_else(|| {
            mlx_gen::Error::Msg(
                "ltx_2_5 trainer: depth anchoring needs the prepared bundle's `videoShape`".into(),
            )
        })?;
        let shape: Vec<i32> = serde_json::from_str(raw).map_err(|error| {
            mlx_gen::Error::Msg(format!(
                "ltx_2_5 trainer: prepared bundle `videoShape` is not a JSON shape: {error}"
            ))
        })?;
        if shape.len() != 5 {
            return Err(
                format!("ltx_2_5 trainer: videoShape must be [B,C,F,H,W], got {shape:?}").into(),
            );
        }
        max = (
            max.0.max(shape[2]),
            max.1.max(shape[3]),
            max.2.max(shape[4]),
        );
    }
    Ok(max)
}

/// Extra training memory (GB) LTX-2.5 depth anchoring adds (sc-24830, epic 2123 E7): TAELTX2.3 +
/// the losses' frozen models for `frames` selected latent frames per step at the `(h, w)` latent
/// grid (×32 pixels) — each frame's decode sized as a `T = 2` clip with its predecessor
/// ([`ltx25_decoder`]) — plus `entries` cached references. `0` when nothing is enabled.
fn ltx25_perceptual_footprint_gb(
    cfg: &TrainingConfig,
    h: i32,
    w: i32,
    frames: u32,
    entries: usize,
) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &ltx25_decoder(),
        mlx_gen_perceptual::AuxGeometry {
            height: (h as i64 * SPATIAL_SCALE) as u32,
            width: (w as i64 * SPATIAL_SCALE) as u32,
            frames,
            entries,
        },
    )
}

/// The LTX-2.5 depth memory guard (sc-24830, epic 2123 E7): refuse when the resident DiT base (the
/// calibrated resident term [`checkpointed_baseline_gb`] — a lower bound; the 2.5 trainer has no
/// fitted AV working-set curve) plus the aux models exceeds `budget_bytes × 0.85`, through the
/// shared [`mlx_gen_perceptual::check_aux_memory`] (naming `cfg`'s enabled aux losses).
fn ltx25_aux_memory_guard(cfg: &TrainingConfig, extra_gb: f64, budget_bytes: usize) -> Result<()> {
    let projected = checkpointed_baseline_gb() + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    mlx_gen_perceptual::check_aux_memory(
        "ltx_2_5 trainer",
        cfg,
        &format!("a training step decoding the selected `{DEPTH_ANCHORING_FRAMES_KEY}` frames"),
        extra_gb,
        projected,
        safe,
        &format!("{budget_gb:.0} GB MLX limit × 0.85"),
    )
}

/// One LTX-2.3 item's cached latents, one per bucket edge in `edges` order (the item-major layout
/// [`BucketSchedule::cache_index`] indexes, sc-2127): `encode(edge)` the unpatchified
/// `(1, 128, 1, le, le)` latent, `mask_weight(latent.shape())` — the item's subject mask resampled
/// onto **that** latent's grid (sc-24828) — then both patchified by the same [`flatten_latent`] →
/// `(1, S, 128)` so the weight lines up token-for-token with that bucket's clean latent; each pair
/// evaluated so it survives the encoder's drop.
fn encode_bucket_latents(
    edges: &[u32],
    mut encode: impl FnMut(u32) -> Result<Array>,
    mut mask_weight: impl FnMut(&[i32]) -> Result<Option<Array>>,
) -> Result<Vec<(Array, Option<Array>)>> {
    edges
        .iter()
        .map(|&edge| {
            let latent = encode(edge)?;
            let weight = mask_weight(latent.shape())?
                .map(|w| flatten_latent(&w))
                .transpose()?;
            let clean = flatten_latent(&latent)?;
            eval(std::iter::once(&clean).chain(weight.as_ref()))?;
            Ok((clean, weight))
        })
        .collect()
}

/// The subject-masked loss weight (sc-24828) of one unpatchified still latent
/// `(1, 128, 1, le, le)`: the image went through `center_crop_square`, so the mask's crop is
/// [`CropBox::center_square`], area-averaged onto the latent's `(le, le)` grid and broadcast over
/// every channel — the latent's exact shape. `None` when off.
fn still_subject_weight(
    label: &str,
    mask: Option<&PreparedSubjectMask>,
    latent_shape: &[i32],
) -> Result<Option<Array>> {
    prepared_subject_mask_weight(label, mask, CropBox::center_square, latent_shape)
}

fn validation_guider(plan: Ltx25GuidancePlan) -> crate::params::GuiderParams {
    crate::params::GuiderParams {
        cfg_scale: plan.cfg_scale,
        stg_scale: plan.stg_scale,
        // Block selection is executed by `AvPerturbation`; guidance combination consumes scales.
        stg_blocks: &[],
        rescale_scale: plan.rescale_scale,
        modality_scale: plan.modality_scale,
    }
}

#[allow(clippy::too_many_arguments)]
fn denoise_ltx25_validation_av(
    dit: &AvDiT,
    video: &crate::conditioning::VideoTokenState,
    audio: &Array,
    video_context: &Array,
    audio_context: &Array,
    negative_video_context: &Array,
    negative_audio_context: &Array,
    audio_positions: &Array,
    plan: &Ltx25ValidationRenderPlan,
    cancel: &gen_core::CancelFlag,
) -> Result<crate::conditioning::VideoTokenState> {
    let dtype = video.latent.dtype();
    let shape = audio.shape();
    let (batch, channels, frames, bins) = (shape[0], shape[1], shape[2], shape[3]);
    let mut video_tokens = video.latent.clone();
    let mut audio_latents = audio.clone();
    let rope_epoch = Some(dit.next_rope_epoch());
    let stg = AvPerturbation::stg_blocks(&plan.stg_blocks)?;
    let video_guider = validation_guider(plan.video_guidance);
    let audio_guider = validation_guider(plan.audio_guidance);
    for (step, sigmas) in plan.sigmas.windows(2).enumerate() {
        if cancel.is_cancelled() {
            return Err(mlx_gen::Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[0], sigmas[1]);
        let audio_tokens = audio_latents.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            frames,
            channels * bins,
        ])?;
        let video_timestep =
            crate::conditioning::token_timesteps(&video.denoise_mask, video_tokens.dtype(), sigma)?;
        let sigma_array = Array::from_slice(&[sigma], &[1]).as_dtype(dtype)?;
        let audio_timestep = broadcast_to(&sigma_array, &[batch, frames])?;
        let forward = |vctx: &Array, actx: &Array, perturbation: AvPerturbation| {
            dit.forward_controlled(
                &video_tokens,
                &video_timestep,
                vctx,
                None,
                &video.positions,
                &audio_tokens,
                &audio_timestep,
                actx,
                None,
                audio_positions,
                video.keyframes_mask.as_ref(),
                rope_epoch,
                perturbation,
            )
        };
        let (conditional_video, conditional_audio) =
            forward(video_context, audio_context, AvPerturbation::NONE)?;
        let (unconditional_video, unconditional_audio) = forward(
            negative_video_context,
            negative_audio_context,
            AvPerturbation::NONE,
        )?;
        let (stg_video, stg_audio) = forward(video_context, audio_context, stg)?;
        let (isolated_video, isolated_audio) = forward(
            video_context,
            audio_context,
            AvPerturbation::modality_isolated(),
        )?;
        let denoised = |video_velocity: Array, audio_velocity: Array| -> Result<(Array, Array)> {
            let audio_velocity = audio_velocity
                .reshape(&[batch, frames, channels, bins])?
                .transpose_axes(&[0, 2, 1, 3])?;
            Ok((
                crate::transformer::to_denoised(&video_tokens, &video_velocity, &sigma_array)?,
                crate::transformer::to_denoised(&audio_latents, &audio_velocity, &sigma_array)?,
            ))
        };
        let (conditional_video, conditional_audio) =
            denoised(conditional_video, conditional_audio)?;
        let (unconditional_video, unconditional_audio) =
            denoised(unconditional_video, unconditional_audio)?;
        let (stg_video, stg_audio) = denoised(stg_video, stg_audio)?;
        let (isolated_video, isolated_audio) = denoised(isolated_video, isolated_audio)?;
        let guided_video = crate::dev_sampler::combine_guidance(
            &conditional_video,
            &unconditional_video,
            &stg_video,
            &isolated_video,
            video_guider,
        )?;
        let guided_video = crate::conditioning::apply_denoise_mask(
            &guided_video,
            &video.clean_latent,
            &video.denoise_mask,
        )?;
        let guided_audio = crate::dev_sampler::combine_guidance(
            &conditional_audio,
            &unconditional_audio,
            &stg_audio,
            &isolated_audio,
            audio_guider,
        )?;
        video_tokens =
            crate::pipeline::euler_step(&video_tokens, &guided_video, sigma, sigma_next)?;
        audio_latents =
            crate::pipeline::euler_step(&audio_latents, &guided_audio, sigma, sigma_next)?;
        eval([&video_tokens, &audio_latents])?;
        let _ = step;
    }
    Ok(crate::conditioning::VideoTokenState {
        latent: video_tokens,
        clean_latent: video.clean_latent.clone(),
        denoise_mask: video.denoise_mask.clone(),
        positions: video.positions.clone(),
        target_tokens: video.target_tokens,
        keyframes_mask: video.keyframes_mask.clone(),
        generated_keyframe_layout: video.generated_keyframe_layout.clone(),
    })
}

fn denoise_ltx25_validation_video(
    dit: &AvDiT,
    video: &crate::conditioning::VideoTokenState,
    video_context: &Array,
    negative_video_context: &Array,
    plan: &Ltx25ValidationRenderPlan,
    cancel: &gen_core::CancelFlag,
) -> Result<crate::conditioning::VideoTokenState> {
    let mut tokens = video.latent.clone();
    let rope_epoch = Some(dit.next_rope_epoch());
    let stg = AvPerturbation::stg_blocks(&plan.stg_blocks)?;
    let guider = validation_guider(plan.video_guidance);
    for sigmas in plan.sigmas.windows(2) {
        if cancel.is_cancelled() {
            return Err(mlx_gen::Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[0], sigmas[1]);
        let timestep =
            crate::conditioning::token_timesteps(&video.denoise_mask, tokens.dtype(), sigma)?;
        let forward = |context: &Array, perturbation: AvPerturbation| {
            dit.forward_video_only_controlled(
                &tokens,
                &timestep,
                context,
                None,
                &video.positions,
                video.keyframes_mask.as_ref(),
                rope_epoch,
                perturbation,
            )
        };
        let conditional = forward(video_context, AvPerturbation::NONE)?;
        let unconditional = forward(negative_video_context, AvPerturbation::NONE)?;
        let perturbed = forward(video_context, stg)?;
        let sigma_array = Array::from_slice(&[sigma], &[1]).as_dtype(tokens.dtype())?;
        let conditional = crate::transformer::to_denoised(&tokens, &conditional, &sigma_array)?;
        let unconditional = crate::transformer::to_denoised(&tokens, &unconditional, &sigma_array)?;
        let perturbed = crate::transformer::to_denoised(&tokens, &perturbed, &sigma_array)?;
        // With no audio modality, the isolated-modality branch is the conditional branch and its
        // configured guidance term is therefore exactly zero.
        let guided = crate::dev_sampler::combine_guidance(
            &conditional,
            &unconditional,
            &perturbed,
            &conditional,
            guider,
        )?;
        let guided = crate::conditioning::apply_denoise_mask(
            &guided,
            &video.clean_latent,
            &video.denoise_mask,
        )?;
        tokens = crate::pipeline::euler_step(&tokens, &guided, sigma, sigma_next)?;
        eval([&tokens])?;
    }
    Ok(crate::conditioning::VideoTokenState {
        latent: tokens,
        clean_latent: video.clean_latent.clone(),
        denoise_mask: video.denoise_mask.clone(),
        positions: video.positions.clone(),
        target_tokens: video.target_tokens,
        keyframes_mask: video.keyframes_mask.clone(),
        generated_keyframe_layout: video.generated_keyframe_layout.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn render_ltx25_validation_sample(
    dit: &AvDiT,
    vae: &LtxVideoVae,
    video_context: &Array,
    audio_context: &Array,
    negative_video_context: &Array,
    negative_audio_context: &Array,
    config: &Ltx25ValidationConfig,
    seed: u64,
    cancel: &gen_core::CancelFlag,
) -> Result<Image> {
    let plan = build_ltx25_validation_render_plan(config)?;
    let latent_frames = ((plan.frames - 1) / crate::positions::TEMPORAL_SCALE as u32 + 1) as usize;
    let latent_height = (plan.height / SPATIAL_SCALE as u32) as usize;
    let latent_width = (plan.width / SPATIAL_SCALE as u32) as usize;
    if latent_height == 0
        || latent_width == 0
        || !plan.height.is_multiple_of(SPATIAL_SCALE as u32)
        || !plan.width.is_multiple_of(SPATIAL_SCALE as u32)
    {
        return Err("ltx_2_5 trainer: validation dimensions must be VAE aligned".into());
    }
    let video_noise = random::normal::<f32>(
        &[
            1,
            128,
            latent_frames as i32,
            latent_height as i32,
            latent_width as i32,
        ],
        None,
        None,
        Some(&random::key(seed)?),
    )?;
    let video_positions = create_position_grid_with(
        1,
        latent_frames,
        latent_height,
        latent_width,
        crate::positions::TEMPORAL_SCALE,
        SPATIAL_SCALE,
        plan.fps as f32,
        true,
    );
    let video_state = crate::conditioning::VideoTokenState::base(&video_noise, &video_positions)?;
    let video = if plan.generate_audio {
        let audio_frames =
            crate::positions::compute_audio_frames(plan.frames as usize, plan.fps as f64);
        let audio_noise = random::normal::<f32>(
            &[
                1,
                crate::positions::AUDIO_LATENT_CHANNELS as i32,
                audio_frames as i32,
                crate::positions::AUDIO_MEL_BINS as i32,
            ],
            None,
            None,
            Some(&random::key(seed.wrapping_add(1))?),
        )?;
        let audio_positions = create_audio_position_grid(1, audio_frames);
        denoise_ltx25_validation_av(
            dit,
            &video_state,
            &audio_noise,
            video_context,
            audio_context,
            negative_video_context,
            negative_audio_context,
            &audio_positions,
            &plan,
            cancel,
        )?
    } else {
        denoise_ltx25_validation_video(
            dit,
            &video_state,
            video_context,
            negative_video_context,
            &plan,
            cancel,
        )?
    };
    let latents = crate::conditioning::unpatchify_grid(
        &video.latent,
        128,
        latent_frames as i32,
        latent_height as i32,
        latent_width as i32,
    )?;
    let frames = crate::pipeline::decode_to_frames(vae, &latents, cancel)?;
    let shape = frames.shape();
    if shape.len() != 4 || shape[0] == 0 || shape[3] != 3 {
        return Err(
            format!("ltx_2_5 trainer: validation decode returned invalid shape {shape:?}").into(),
        );
    }
    let (height, width) = (shape[1] as usize, shape[2] as usize);
    let count = height * width * 3;
    let first = frames.reshape(&[-1])?;
    eval([&first])?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels: first.as_slice::<u8>()[..count].to_vec(),
    })
}

/// `to_out.0` → `to_out`, the only diffusers→checkpoint rename in the attention LoRA surface (the
/// inference loader does the same in `adapters::normalize`); other suffixes pass through.
fn resolve_segments(save_path: &str) -> Vec<String> {
    crate::adapters::normalize_ltx_key(save_path)
        .split('.')
        .map(String::from)
        .collect()
}

const LTX25_ATTENTION_MODULES: [&str; 6] = [
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
];

/// Expand the reference's suffix matching against the complete AV model.  Short attention suffixes
/// intentionally hit all six attention modules; FF patterns remain modality-specific exactly as in
/// the upstream YAML. No bias target is manufactured: `ff_bias=false` means the 2.5 video FF base
/// legitimately has no bias while its two weight projections remain trainable.
fn ltx25_target_paths(num_layers: i32, suffixes: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    let push_unique = |paths: &mut Vec<String>, path: String| {
        if !paths.contains(&path) {
            paths.push(path);
        }
    };
    for i in 0..num_layers {
        for suffix in suffixes {
            match suffix.as_str() {
                "to_q" | "to_k" | "to_v" | "to_out.0" | "to_gate_logits" => {
                    for module in LTX25_ATTENTION_MODULES {
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.{module}.{suffix}"),
                        );
                    }
                }
                "ff.net.0.proj" | "ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                    push_unique(&mut paths, format!("transformer_blocks.{i}.audio_{suffix}"));
                }
                "ff" => {
                    for projection in ["ff.net.0.proj", "ff.net.2"] {
                        push_unique(&mut paths, format!("transformer_blocks.{i}.{projection}"));
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.audio_{projection}"),
                        );
                    }
                }
                "audio_ff.net.0.proj" | "audio_ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                }
                qualified if qualified.contains('.') => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{qualified}"));
                }
                _ => {}
            }
        }
    }
    paths
}

fn build_targets_from_paths(
    dit: &mut impl LtxAdaptable,
    paths: impl IntoIterator<Item = String>,
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut targets = Vec::new();
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    for (idx, save_path) in paths.into_iter().enumerate() {
        let segs = resolve_segments(&save_path);
        let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
        let Some(lin) = dit.adaptable_mut(&seg_refs) else {
            continue;
        };
        let shape = lin.base_shape();
        let (out_f, in_f) = (shape[0], shape[1]);
        let a_key: Rc<str> = Rc::from(format!("{save_path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{save_path}.lora_b"));
        let ka = random::key(seed.wrapping_add(2 * idx as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            &small,
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LtxLoraTarget {
            save_path,
            segs,
            a_key,
            b_key,
        });
    }
    Ok((targets, params))
}

fn build_targets_25(
    dit: &mut AvDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    build_targets_from_paths(dit, ltx25_target_paths(num_layers, suffixes), rank, seed)
}

/// Enumerate the `attn1`/`attn2` × `suffixes` targets across the DiT's `num_layers` blocks, resolve
/// each on the (mutable) DiT, read its `[out,in]` base shape, and initialise the trainable factors
/// the reference `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`, `B = 0` `[out,rank]` — keyed
/// `{save_path}.lora_a` / `.lora_b`. Targets that do not resolve (a missing gated branch, a typo'd
/// suffix) are skipped.
fn build_targets(
    dit: &mut LtxDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut paths = Vec::new();
    for i in 0..num_layers {
        for attn in ["attn1", "attn2"] {
            for suf in suffixes {
                paths.push(format!("transformer_blocks.{i}.{attn}.{suf}"));
            }
        }
    }
    build_targets_from_paths(dit, paths, rank, seed)
}

/// Inject the current trainable factors as one LoRA residual per target via the LTX training seam —
/// transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold `alpha/rank` into `b` — so the residual
/// is `(x·Aᵀ·Bᵀ)·(alpha/rank)`, matching the reference `_MlxLoRALinear`. Differentiable.
fn install_train_lora(
    dit: &mut impl LtxAdaptable,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    lora_dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b = params[&t.b_key]
            .t()
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?; // [out,r] -> [r,out] · (α/r)

        // sc-4942 — under the bf16 training cast the f32 factors must join the bf16 stream, or every
        // adapted Linear re-promotes its block to f32 (defeating the activation saving). No-op in f32.
        let (a, b) = match lora_dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let seg_refs: Vec<&str> = t.segs.iter().map(String::as_str).collect();
        let lin = dit
            .adaptable_mut(&seg_refs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.save_path)))?;
        lin.set_train_lora(a, b);
    }
    Ok(())
}

/// Group resolved targets by their owning block (sc-4942) — `block_targets[i]` lists block `i`'s
/// trainable LoRA targets as the block-local path (`segs` minus the `transformer_blocks.{i}` prefix)
/// plus the factor-map keys, for the gradient-checkpoint closure. Every target lives in a
/// `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping is exhaustive.
fn group_block_targets(targets: &[LtxLoraTarget], n_layers: usize) -> Vec<Vec<BlockLoraRef>> {
    let mut out: Vec<Vec<BlockLoraRef>> = (0..n_layers).map(|_| Vec::new()).collect();
    for t in targets {
        // segs = ["transformer_blocks", "{i}", attn, suffix...]; the block-local path is segs[2..].
        if t.segs.len() < 3 || t.segs[0] != "transformer_blocks" {
            continue;
        }
        let Ok(i) = t.segs[1].parse::<usize>() else {
            continue;
        };
        if i >= n_layers {
            continue;
        }
        out[i].push(BlockLoraRef {
            local: t.segs[2..].to_vec(),
            a_key: t.a_key.to_string(),
            b_key: t.b_key.to_string(),
        });
    }
    out
}

/// One forward+backward over the trainable factors: build the rectified-flow input `x_t`, inject the
/// factors, run the video DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
///
/// `checkpoint_block`, when `Some`, lists each block's trainable targets and switches the forward to
/// the gradient-checkpointed path (sc-4942) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (attention-segment-checkpointed) forward.
/// `dtype` is the training compute dtype (sc-4942): for bf16 the DiT weights were cast once in
/// `train_impl` and `preprocess` casts the activation stream, so the LoRA factors are cast at install
/// (here and inside the checkpoint segment) to keep the whole graph bf16; the noising / loss / grads
/// stay f32.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        params,
        targets,
        alpha,
        rank,
        clean,
        context,
        positions,
        sigma,
        noise,
        mae,
        mask_weight,
        checkpoint_block,
        dtype,
        None,
    )?;
    Ok((losses.total, grads))
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed and the loss is the weighted perceptual term on the DiT's x0
/// estimate `x_t − σ·v` (the raw output regresses `noise − clean`), unpatchified to the decoder's
/// NCHW layout and decoded by TAELTX2.3; summed (`every_n == 1`) aux terms ride along on a
/// diffusion step. The dense and block-checkpointed forwards land in the same closure, so both
/// carry it. With `aux = None` (or a plan with no aux loss) the traced graph is exactly the
/// pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    // x_t = (1-σ)·clean + σ·noise; target = noise - clean (the raw-output velocity); timestep = σ.
    // x_t / context stay f32 here; `preprocess` casts the activation stream to the compute dtype.
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?;
    let x_t_f32 = x_t.clone();
    let target = subtract(noise, clean)?;
    let timestep = Array::from_slice(&[sigma], &[1, 1]); // (B, 1), broadcast over tokens
    let ctx = context.clone();
    let pos = positions.clone();
    let mask_weight = mask_weight.cloned();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        let v = match checkpoint_block {
            Some(bt) => dit
                .forward_with_main_checkpointed(&x_t, &timestep, &ctx, None, &pos, &p, bt, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => {
                install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
                // `None`: content-keyed RoPE memo (the per-stage epoch fast path is inference-only;
                // training positions are constant within a step, so the content compare hits — sc-7141).
                dit.forward(&x_t, &timestep, &ctx, None, &pos, None)
                    .map_err(|e| Exception::custom(e.to_string()))?
            }
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // requires a scalar cotangent). Dense and block-checkpointed forwards both land here.
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the raw velocity (`noise − clean`): x0 = x_t − σ·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| unflatten_latent(&x, a.latent_edge))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0_hat)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        // Only the first output is differentiated; the other two are reported terms.
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: val[0].item::<f32>(),
        diffusion: diffusion_on.then(|| val[1].item::<f32>()),
        aux: aux_on.then(|| val[2].item::<f32>()),
    };
    Ok((losses, grads))
}

fn masked_modality_loss(
    prediction: &Array,
    modality: &Ltx25PreparedModality,
    mae: bool,
) -> MlxResult<Array> {
    if modality.loss_denominator <= 0.0 {
        return Err(Exception::custom(
            "ltx_2_5 trainer: generated modality has an empty loss mask",
        ));
    }
    let diff = subtract(prediction, &modality.target)?;
    let point = if mae { diff.abs()? } else { diff.square()? };
    let masked = multiply(&point, &modality.loss_mask)?;
    divide(
        &masked.sum(None)?,
        Array::from_f32(modality.loss_denominator),
    )
}

/// Full flexible-strategy AV loss: each present modality has its own target and loss mask; frozen
/// modalities participate in cross-modal attention but contribute zero loss. Audio-only and
/// video-only workflows execute only their stream, matching upstream's optional modalities.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn compute_ltx25_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) =
        compute_ltx25_step_loss_grads(dit, params, targets, alpha, rank, batch, mae, dtype, None)?;
    Ok((losses.total, grads))
}

/// One LTX-2.5 aux-loss step's view of the perceptual path (sc-24830): the plan, the example's
/// reference key and its depth frame selection.
struct Ltx25AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// The example index (selects its cached reference).
    image: usize,
    depth: &'a Ltx25DepthFrames,
}

/// The depth x0 estimate of the generated video stream (sc-24830): over the target tokens (the
/// appended reference tokens dropped), `x0 = x_t − t·v` with the per-token timestep `t` (the
/// planned σ on generated tokens, `0` on conditioning tokens — whose x0 is then their clean
/// latent), the velocity regressing `noise − clean`; the selected frames as the decoder's NCHW.
fn ltx25_video_x0(
    prediction: &Array,
    video: &Ltx25PreparedModality,
    depth: &Ltx25DepthFrames,
) -> Result<Array> {
    use mlx_rs::ops::indexing::IndexOp;
    let tokens = video.target_tokens as i32;
    let noisy = video.noisy.index((.., ..tokens));
    let timestep = video.timestep.index((.., ..tokens)).expand_dims(2)?;
    let velocity = prediction.index((.., ..tokens)).as_dtype(Dtype::Float32)?;
    let x0 = subtract(&noisy, &multiply(&timestep, &velocity)?)?;
    depth.nchw(&x0)
}

/// [`compute_ltx25_loss_grads`] with the step's perceptual plan (sc-24830): on an aux-only step the
/// (video + audio) velocity term is not computed and the loss is the weighted depth term on the
/// generated video's x0 estimate ([`ltx25_video_x0`]); summed (`every_n == 1`) aux terms ride along
/// on a diffusion step. With `aux = None` (or a plan with no aux loss) the traced graph is exactly
/// the pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
fn compute_ltx25_step_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
    aux: Option<Ltx25AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    if batch.video.is_none() && batch.audio.is_none() {
        return Err("ltx_2_5 trainer: prepared batch has no modalities".into());
    }
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    if aux_on && batch.video.is_none() {
        return Err("ltx_2_5 trainer: depth anchoring needs the generated video stream".into());
    }
    let prepared = batch.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
        let mut video_prediction_out: Option<Array> = None;
        let diffusion = match (&prepared.video, &prepared.audio) {
            (Some(video), Some(audio)) => {
                let (video_prediction, audio_prediction) = dit
                    .forward(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = if diffusion_on {
                    let video_loss = (video.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&video_prediction, video, mae))
                        .transpose()?;
                    let audio_loss = (audio.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&audio_prediction, audio, mae))
                        .transpose()?;
                    Some(match (video_loss, audio_loss) {
                        (Some(video), Some(audio)) => add(&video, &audio)?,
                        (Some(loss), None) | (None, Some(loss)) => loss,
                        (None, None) => {
                            return Err(Exception::custom(
                                "ltx_2_5 trainer: both modality loss masks are empty",
                            ))
                        }
                    })
                } else {
                    None
                };
                video_prediction_out = Some(video_prediction);
                diffusion
            }
            (Some(video), None) => {
                let prediction = dit
                    .forward_video_only(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = diffusion_on
                    .then(|| masked_modality_loss(&prediction, video, mae))
                    .transpose()?;
                video_prediction_out = Some(prediction);
                diffusion
            }
            (None, Some(audio)) => {
                let prediction = dit
                    .forward_audio_only(
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                diffusion_on
                    .then(|| masked_modality_loss(&prediction, audio, mae))
                    .transpose()?
            }
            (None, None) => unreachable!(),
        };
        let aux_term = match (&aux, &prepared.video, &video_prediction_out) {
            (Some(a), Some(video), Some(prediction)) if aux_on => {
                let x0 = ltx25_video_x0(prediction, video, a.depth)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (value, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: value[0].item::<f32>(),
        diffusion: diffusion_on.then(|| value[1].item::<f32>()),
        aux: aux_on.then(|| value[2].item::<f32>()),
    };
    Ok((losses, grads))
}

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the LTX
/// latent token count `s` (the trainer trains single-frame still latents, so `s = (edge/32)²`). With
/// attention-segment checkpointing always on (sc-4942) the seq² attention term is demoted to a single
/// layer's backward transient, so the measured f32 curve is essentially **linear** in `s`: the
/// constant is the resident base (the Q-packed 22B DiT after the Gemma TE is freed) plus the
/// f32-activation working set, the slope the per-token retained hidden activations across the 48
/// blocks.
///
/// CALIBRATED from `first_step_attn_ckpt_sweep` (128 GB Mac, rank 8 / 384 targets / batch 1, f32):
/// s=256 → 23.3 GB, s=576 → 31.5 GB, s=1024 → 42.6 GB (fit error < 0.2 GB). Refit if that harness
/// prints materially different numbers. (LTX trains f32 only — see the `compute_dtype` note in
/// `train_impl`; there is no bf16 production path to size.)
fn projected_dense_peak_gb(s: f64) -> f64 {
    16.9 + 0.0251 * s
}

/// Latent tokens per side for a bucketed pixel `edge` (the VAE's ×32 spatial stride, floor 1).
fn latent_edge_of(edge: u32) -> usize {
    (edge / SPATIAL_SCALE as u32).max(1) as usize
}

/// The latent edge the pre-flight guard and previews size for: the largest bucket's (epic 2123 E7).
fn guard_latent_edge(edges: &[u32]) -> usize {
    edges.iter().map(|&e| latent_edge_of(e)).max().unwrap_or(1)
}

/// The item-major cache entry and its bucket (whose RoPE grid the step uses) for 1-based `step`
/// (sc-2127). With one bucket the entry is the pre-bucket `(step-1) % n_items` and the bucket is 0.
fn step_entry(schedule: &BucketSchedule, step: u32) -> (usize, usize) {
    let entry = schedule.cache_index((step - 1) as usize);
    (entry, entry % schedule.n_buckets())
}

/// The bucket index of the largest edge in `edges` (the first on a tie; `0` for an empty list) —
/// the bucket training previews render at (sc-2127).
fn largest_bucket(edges: &[usize]) -> usize {
    edges
        .iter()
        .enumerate()
        .fold(0, |best, (i, &e)| if e > edges[best] { i } else { best })
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead (sc-4942 — the sc-4874 mechanism).
/// `latent_edge` is the latent tokens per side (`edge/32`); the token count is `latent_edge²` (the
/// trainer trains single-frame still latents). The budget is MLX's reported memory limit (≈ the
/// device's recommended working set) × 0.85 for worker/host headroom. Only consulted when gradient
/// checkpointing is OFF. With resolution buckets the caller passes the LARGEST bucket's latent edge
/// (sc-2127 / epic 2123 E7): the per-step working set peaks at the biggest latent, and the cached
/// still latents (KBs–MBs per item) are not part of the projection.
///
/// Epic 2123 E7 (sc-24830): `extra_gb` is the training-time auxiliary models' footprint
/// ([`perceptual_footprint_gb`]); with it the guard also runs with gradient checkpointing on
/// (`checkpointed`), stacking on the resident base ([`checkpointed_baseline_gb`]).
fn preflight_memory_guard(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    preflight_memory_guard_with_budget(cfg, latent_edge, extra_gb, checkpointed, get_memory_limit())
}

/// The checkpointed baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation term; the base
/// stays). A lower bound.
fn checkpointed_baseline_gb() -> f64 {
    projected_dense_peak_gb(0.0)
}

/// [`preflight_memory_guard`] against an explicit budget (`budget_bytes`, the live MLX limit in
/// production) — so the guard's arithmetic is testable on any host. A checkpointed refusal goes
/// through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled aux losses.
fn preflight_memory_guard_with_budget(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
    budget_bytes: usize,
) -> Result<()> {
    let s = (latent_edge * latent_edge) as f64;
    let projected = if checkpointed {
        checkpointed_baseline_gb()
    } else {
        projected_dense_peak_gb(s)
    } + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if checkpointed {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return mlx_gen_perceptual::check_aux_memory(
            "ltx_2_3 trainer",
            cfg,
            &format!("a checkpointed training step at resolution {px}"),
            extra_gb,
            projected,
            safe,
            &format!("{budget_gb:.0} GB MLX limit × 0.85"),
        );
    }
    if projected > safe {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return Err(mlx_gen_perceptual::name_aux_losses(
            format!(
                "ltx_2_3 trainer: a dense first training step at resolution {px} needs ~{projected:.0} GB \
                 (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
                 safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
                 worker (SIGKILL) at the first step with no recoverable error (sc-4874/sc-4942). Enable Gradient \
                 Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
            )
            .into(),
            cfg,
            extra_gb,
        ));
    }
    Ok(())
}

/// Write the trained LoRA as safetensors keyed by the LTX module paths — `{module}.lora_A.weight`
/// `[rank,in]`, `{module}.lora_B.weight` `[out,rank]`, scalar `{module}.alpha` (= `alpha`) — the
/// reference `_save_lora` format, reloadable by [`crate::apply_ltx_adapters`] (which folds
/// `scale = alpha/rank`). `networkType`/`rank`/`alpha` metadata mirrors the other family trainers.
fn save_lora(
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: u32,
    path: &Path,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.save_path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.save_path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.save_path), &params[&t.b_key]));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    // The split LTX-2.5 provider consumes this explicit file-wide contract and deliberately does
    // not infer a rank from a factor.  Retain the legacy keys above so LTX-2.3/third-party loaders
    // remain byte-for-byte compatible while a newly trained adapter can be selected by either route.
    meta.insert("lora_rank".to_string(), rank.to_string());
    meta.insert("lora_alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = mlx_gen::gen_core::__model_input_result!(
        image::open(path),
        "crates/media/mlx-gen/mlx-gen-ltx/src/training.rs::decode_image/image::open",
        Reference,
        &[path]
    )
    .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

// ===========================================================================================
// sc-4942 — first-step memory + grad-parity harness (weight-gated, run as its own process).
//
// Ports the z-image sc-4874/4886/4887 `first_step_repro` harness to LTX: drives the exact inner
// training step (`compute_loss_grads` + the step-1 grad `eval`) directly, sweeping resolution with
// MLX peak-memory probes, and asserts the three levers' invariants on REAL weights:
//   * attention-segment checkpointing is bit-identical to the retained backward,
//   * block (gradient) checkpointing matches the dense path within fp tolerance,
//   * bf16 grads point the same way as f32 and materially shrink the working set.
//
//   cargo test -p mlx-gen-ltx --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
mod first_step_repro;

mod bucket_tests;

mod preflight_tests;

mod validate_request_tests;

mod ltx25_tiny_production_lifecycle;

mod load_trainer_tests;

/// sc-24830 (epic 2123 depth anchoring) — the LTX-2.3 step seam on a tiny synthetic video DiT
/// (`rung4_block_window_tests::tiny_cfg` / `tiny_weight_map`: 4 layers, 4 latent channels) with a
/// random-init tiny-width TAEHV carrying TAELTX2.3's hyperparameters (patch 4, 8× temporal) at 4
/// latent channels, and a random-init tiny Depth-Anything-V2. Drives the same
/// [`compute_step_loss_grads`] / [`aux_driver`]
/// `train_23_impl` runs; the LTX-2.5 refusal through its typed reason. CPU; no weights downloaded.
mod depth_anchoring_tests;

/// sc-24830 — LTX-2.5 depth anchoring on the tiny synthetic AV DiT
/// (`rung4_block_window_tests::tiny_cfg`: 4 latent channels, video 2 frames × 2 × 2 tokens + audio)
/// with a random-init tiny-width TAEHV carrying TAELTX2.3's hyperparameters at 4 latent channels and
/// a random-init tiny Depth-Anything-V2. Drives the same [`compute_ltx25_step_loss_grads`] /
/// [`Ltx25DepthFrames`] / [`refuse_ltx25_depth_anchoring`] `train_25_impl` runs. CPU.
mod ltx25_depth_anchoring_tests;
