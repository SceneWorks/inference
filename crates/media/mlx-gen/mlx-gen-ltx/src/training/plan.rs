use super::*;

pub(super) fn ensure_ltx25_training_variant(
    variant: crate::dev_sampler::TransformerVariant,
) -> Result<()> {
    if variant != crate::dev_sampler::TransformerVariant::Dev {
        return Err(
            "ltx_2_5 trainer requires the dev transformer (the dev/q4 training tier); a distilled \
             transformer cannot execute the canonical 30-step CFG/STG validation route"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn ensure_ltx25_training_tier(split: SplitModel) -> Result<()> {
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

pub(super) fn validate_ltx25_adapter_scale(alpha: f32) -> Result<()> {
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
    pub(super) const fn simple(kind: Ltx25ConditionKind, probability: f32) -> Self {
        Self {
            kind,
            probability,
            temporal_boundary: None,
            spatial_region: None,
            spatial_scale_factor: None,
            temporal_scale_factor: None,
        }
    }

    pub(super) const fn temporal(kind: Ltx25ConditionKind, boundary: usize) -> Self {
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

pub(super) const GENERATED: Ltx25ModalityPlan = Ltx25ModalityPlan {
    is_generated: true,
    conditions: &[],
};

pub(super) const FROZEN: Ltx25ModalityPlan = Ltx25ModalityPlan {
    is_generated: false,
    conditions: &[],
};

pub(super) const fn generated(conditions: &'static [Ltx25ConditionSpec]) -> Ltx25ModalityPlan {
    Ltx25ModalityPlan {
        is_generated: true,
        conditions,
    }
}

pub(super) const FIRST_FRAME_50: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(
    Ltx25ConditionKind::FirstFrame,
    0.5,
)];

pub(super) const PREFIX_8: &[Ltx25ConditionSpec] =
    &[Ltx25ConditionSpec::temporal(Ltx25ConditionKind::Prefix, 8)];

pub(super) const SUFFIX_8: &[Ltx25ConditionSpec] =
    &[Ltx25ConditionSpec::temporal(Ltx25ConditionKind::Suffix, 8)];

pub(super) const MASK: &[Ltx25ConditionSpec] =
    &[Ltx25ConditionSpec::simple(Ltx25ConditionKind::Mask, 1.0)];

pub(super) const SPATIAL_CROP: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec {
    kind: Ltx25ConditionKind::SpatialCrop,
    probability: 1.0,
    temporal_boundary: None,
    spatial_region: Some((0, 0, 288, 576)),
    spatial_scale_factor: None,
    temporal_scale_factor: None,
}];

pub(super) const REFERENCE: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(
    Ltx25ConditionKind::Reference,
    1.0,
)];

pub(super) const REFERENCE_FIRST_FRAME_20: &[Ltx25ConditionSpec] = &[
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

pub(super) fn condition_draw(seed: u64, sample_index: u64, ordinal: usize) -> f32 {
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

pub(super) fn build_ltx25_token_plan_from_specs(
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

pub(super) fn canonical_modality(plan: Ltx25ModalityPlan) -> Ltx25ResolvedModality {
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

pub(super) fn json_u32(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<u32>> {
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

pub(super) fn parse_resolved_condition(
    value: &serde_json::Value,
) -> Result<Ltx25ResolvedCondition> {
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

pub(super) fn resolve_modality(
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
