//! Stable Audio 3 diffusion-transformer backbone and conditioning assembly.
//!
//! This module follows frozen upstream commit
//! `124e8a799f57a1f665495ecb72e547d0a62867f1`. It intentionally remains
//! unregistered: sampler, identity, and catalog composition belong to later stories.

use candle_audio::candle_core::{bail, DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::config::{
    ConditionerConfig, DiffusionModelConfig, DiffusionObjective, FourierFeaturesType,
};
use crate::transformer::{
    AttentionMasks, MemoryTokens, RotaryEmbedding, TransformerBlock, TransformerBlockMasks,
};
use crate::weights::{SnapshotLayout, StableAudioVarBuilders};

const EXPO_MIN_FREQ: f64 = 0.5;
const EXPO_MAX_FREQ: f64 = 10_000.0;

fn expo_fourier_features(values: &Tensor, dim: usize) -> Result<Tensor> {
    if dim == 0 || !dim.is_multiple_of(2) {
        bail!("Expo Fourier feature dimension must be positive and even")
    }
    let input_dtype = values.dtype();
    let values = values.to_dtype(DType::F32)?;
    let values = match values.rank() {
        1 => values.unsqueeze(1)?,
        2 if values.dim(1)? == 1 => values,
        _ => bail!("Expo Fourier inputs must have shape [batch] or [batch,1]"),
    };
    let half = dim / 2;
    let log_min = EXPO_MIN_FREQ.ln() as f32;
    let log_range = (EXPO_MAX_FREQ.ln() - EXPO_MIN_FREQ.ln()) as f32;
    let frequencies: Vec<f32> = (0..half)
        .map(|index| {
            let ramp = if half == 1 {
                0.0
            } else {
                index as f32 / (half - 1) as f32
            };
            (ramp * log_range + log_min).exp()
        })
        .collect();
    // Keep the same fp32 operation order as upstream:
    // exp(linspace * (log(max)-log(min)) + log(min)), then multiply by 2π.
    let frequencies = Tensor::from_vec(frequencies, (1, half), values.device())?;
    let arguments = (values.broadcast_mul(&frequencies)? * std::f64::consts::TAU)?;
    Tensor::cat(&[&arguments.cos()?, &arguments.sin()?], 1)?.to_dtype(input_dtype)
}

fn sequential_two(first: &Linear, second: &Linear, input: &Tensor) -> Result<Tensor> {
    second.forward(&candle_nn::ops::silu(&first.forward(input)?)?)
}

fn assemble_raw_context(
    prompt: &Tensor,
    duration: &Tensor,
    zero_from_batch: Option<usize>,
) -> Result<Tensor> {
    let mut context = Tensor::cat(&[prompt, &duration.unsqueeze(1)?], 1)?;
    if let Some(zero_from) = zero_from_batch {
        let batch = context.dim(0)?;
        if zero_from > batch {
            bail!("CFG unconditional context offset exceeds the assembled batch")
        }
        let mut keep = vec![1f32; batch];
        keep[zero_from..].fill(0.0);
        let keep =
            Tensor::from_vec(keep, (batch, 1, 1), context.device())?.to_dtype(context.dtype())?;
        context = context.broadcast_mul(&keep)?;
    }
    Ok(context)
}

struct Conv1x1 {
    weight: Tensor,
}

impl Conv1x1 {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get((channels, channels, 1), "weight")?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.weight.squeeze(2)?.transpose(0, 1)?;
        input
            .transpose(1, 2)?
            .broadcast_matmul(&weight)?
            .transpose(1, 2)?
            .contiguous()
    }
}

/// Shipped `[0,384]` duration conditioner.
pub struct NumberConditioner {
    min: f64,
    max: f64,
    feature_dim: usize,
    projection: Linear,
}

impl NumberConditioner {
    pub fn load(model: &DiffusionModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = model
            .conditioning
            .configs
            .iter()
            .find_map(|entry| match entry {
                ConditionerConfig::Number { id, config } if id == "seconds_total" => Some(config),
                _ => None,
            })
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "missing seconds_total NumberConditioner".into(),
                )
            })?;
        if config.fourier_features_type != FourierFeaturesType::Expo {
            bail!("only the shipped Expo NumberConditioner is supported")
        }
        Ok(Self {
            min: config.min_val,
            max: config.max_val,
            feature_dim: 256,
            projection: linear(
                256,
                model.conditioning.cond_dim,
                vb.pp("conditioners.seconds_total.embedder.embedding.1"),
            )?,
        })
    }

    pub fn features(&self, seconds: &Tensor) -> Result<Tensor> {
        let normalized = ((seconds.to_dtype(DType::F32)?.clamp(self.min, self.max)? - self.min)?
            / (self.max - self.min))?;
        expo_fourier_features(&normalized, self.feature_dim)
    }

    pub fn forward(&self, seconds: &Tensor) -> Result<Tensor> {
        self.projection.forward(&self.features(seconds)?)
    }
}

/// Inputs owned by the DiT. Prompt rows are already learned-padded by T5Gemma.
pub struct DitInputs<'a> {
    pub latents: &'a Tensor,
    pub timestep: &'a Tensor,
    pub prompt: &'a Tensor,
    pub seconds_total: &'a Tensor,
    /// `[batch,257,time]`, ordered `[inpaint_mask, inpaint_masked_input]`.
    pub local_conditioning: &'a Tensor,
    /// Frozen CPU semantics: invalid keys retain K and zero V only.
    pub padding_mask: Option<&'a Tensor>,
}

/// Compact intermediate state for independent reference validation.
#[derive(Default)]
pub struct DitTrace {
    pub number_features: Option<Tensor>,
    pub number_embedding: Option<Tensor>,
    pub raw_context: Option<Tensor>,
    pub projected_context: Option<Tensor>,
    pub duration_global: Option<Tensor>,
    pub timestep_features: Option<Tensor>,
    pub timestep_embedding: Option<Tensor>,
    pub combined_global: Option<Tensor>,
    pub global_modulation: Option<Tensor>,
    pub preprocessed: Option<Tensor>,
    pub projected_input: Option<Tensor>,
    pub with_memory: Option<Tensor>,
    pub rotary_frequencies: Option<Tensor>,
    pub layer0_local: Option<Tensor>,
    pub layer0_output: Option<Tensor>,
    pub trimmed: Option<Tensor>,
    pub projected_output: Option<Tensor>,
    pub output: Option<Tensor>,
}

/// Pure guidance controls consumed here; sampler scheduling remains outside this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guidance {
    pub cfg_scale: f64,
    pub apg_scale: f64,
    pub cfg_norm_threshold: f64,
    pub scale_phi: f64,
}

impl Default for Guidance {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            apg_scale: 1.0,
            cfg_norm_threshold: 0.0,
            scale_phi: 0.0,
        }
    }
}

/// Exact unregistered Stable Audio 3 small/medium DiT.
pub struct StableAudio3Dit {
    objective: DiffusionObjective,
    io_channels: usize,
    embed_dim: usize,
    depth: usize,
    number: NumberConditioner,
    to_cond_first: Linear,
    to_cond_second: Linear,
    to_global_first: Linear,
    to_global_second: Linear,
    to_timestep_first: Linear,
    to_timestep_second: Linear,
    preprocess: Conv1x1,
    postprocess: Conv1x1,
    project_in: Linear,
    project_out: Linear,
    memory: MemoryTokens,
    rotary: RotaryEmbedding,
    global_first: Linear,
    global_second: Linear,
    blocks: Vec<TransformerBlock>,
}

impl StableAudio3Dit {
    pub fn from_layout(layout: &SnapshotLayout, device: &Device) -> candle_audio::Result<Self> {
        let model = match &layout.config.model {
            crate::config::ModelConfig::Diffusion(model) => model,
            _ => {
                return Err(candle_audio::AudioError::Msg(
                    "Stable Audio 3 DiT requires a full diffusion snapshot".into(),
                ));
            }
        };
        let builders = layout.mmap_builders(DType::F32, device)?;
        Self::load(model, builders)
            .map_err(|error| candle_audio::AudioError::Msg(format!("load SA3 DiT: {error}")))
    }

    pub fn load(
        model: &DiffusionModelConfig,
        builders: StableAudioVarBuilders<'_>,
    ) -> Result<Self> {
        // The public config loader validates the complete fail-closed shipped surface.
        let config = &model.diffusion.config;
        let root = builders
            .dit
            .ok_or_else(|| candle_audio::candle_core::Error::Msg("missing DiT builder".into()))?
            .pp("model");
        let conditioner = builders.conditioner.ok_or_else(|| {
            candle_audio::candle_core::Error::Msg("missing conditioner builder".into())
        })?;
        let transformer = root.pp("transformer");
        let dim_head = config.embed_dim / config.num_heads;
        let number = NumberConditioner::load(model, conditioner)?;
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            blocks.push(TransformerBlock::load(
                config.embed_dim,
                dim_head,
                Some(config.embed_dim),
                config.norm_type,
                &config.norm_kwargs,
                config.attn_kwargs.qk_norm,
                config.attn_kwargs.qk_norm_eps,
                config.attn_kwargs.differential,
                config.causal,
                &config.ff_kwargs,
                config.zero_init_branch_outputs,
                true,
                config.local_add_cond_dim,
                config.layer_scale,
                transformer.pp(format!("layers.{index}")),
            )?);
        }
        Ok(Self {
            objective: model.diffusion.diffusion_objective,
            io_channels: config.io_channels,
            embed_dim: config.embed_dim,
            depth: config.depth,
            number,
            to_cond_first: linear_no_bias(
                config.cond_token_dim,
                config.embed_dim,
                root.pp("to_cond_embed.0"),
            )?,
            to_cond_second: linear_no_bias(
                config.embed_dim,
                config.embed_dim,
                root.pp("to_cond_embed.2"),
            )?,
            to_global_first: linear_no_bias(
                config.global_cond_dim,
                config.embed_dim,
                root.pp("to_global_embed.0"),
            )?,
            to_global_second: linear_no_bias(
                config.embed_dim,
                config.embed_dim,
                root.pp("to_global_embed.2"),
            )?,
            to_timestep_first: linear(
                config.timestep_features_dim,
                config.embed_dim,
                root.pp("to_timestep_embed.0"),
            )?,
            to_timestep_second: linear(
                config.embed_dim,
                config.embed_dim,
                root.pp("to_timestep_embed.2"),
            )?,
            preprocess: Conv1x1::load(config.io_channels, root.pp("preprocess_conv"))?,
            postprocess: Conv1x1::load(config.io_channels, root.pp("postprocess_conv"))?,
            project_in: linear_no_bias(
                config.io_channels,
                config.embed_dim,
                transformer.pp("project_in"),
            )?,
            project_out: linear_no_bias(
                config.embed_dim,
                config.io_channels,
                transformer.pp("project_out"),
            )?,
            memory: MemoryTokens::load(
                config.num_memory_tokens,
                config.embed_dim,
                transformer.clone(),
            )?,
            // Upstream constructs RotaryEmbedding(max(dim_head / 2, 32)).
            rotary: RotaryEmbedding::load(dim_head.max(64) / 2, transformer.pp("rotary_pos_emb"))?,
            global_first: linear(
                config.embed_dim,
                config.embed_dim,
                transformer.pp("global_cond_embedder.0"),
            )?,
            global_second: linear(
                config.embed_dim,
                6 * config.embed_dim,
                transformer.pp("global_cond_embedder.2"),
            )?,
            blocks,
        })
    }

    pub fn objective(&self) -> DiffusionObjective {
        self.objective
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn forward(&self, inputs: DitInputs<'_>) -> Result<Tensor> {
        self.forward_impl(inputs, None, None, None)
    }

    pub fn forward_with_trace(&self, inputs: DitInputs<'_>) -> Result<(Tensor, DitTrace)> {
        let mut trace = DitTrace::default();
        let output = self.forward_impl(inputs, Some(&mut trace), None, None)?;
        Ok((output, trace))
    }

    fn forward_impl(
        &self,
        inputs: DitInputs<'_>,
        mut trace: Option<&mut DitTrace>,
        zero_cross_context_from_batch: Option<usize>,
        is_canceled: Option<&dyn Fn() -> bool>,
    ) -> Result<Tensor> {
        let (batch, channels, time) = inputs.latents.dims3()?;
        if channels != self.io_channels {
            bail!(
                "Stable Audio 3 DiT expected {} latent channels, got {channels}",
                self.io_channels
            )
        }
        if inputs.timestep.dims1()? != batch || inputs.seconds_total.dims1()? != batch {
            bail!("Stable Audio 3 timestep and duration batch dimensions must match latents")
        }
        if inputs.prompt.dims3()? != (batch, 256, 768) {
            bail!("Stable Audio 3 prompt must be learned-padded [batch,256,768]")
        }
        if inputs.local_conditioning.dims3()? != (batch, 257, time) {
            bail!("Stable Audio 3 local conditioning must be [batch,257,time]")
        }
        if let Some(mask) = inputs.padding_mask {
            if mask.dims2()? != (batch, time) {
                bail!("Stable Audio 3 padding mask must be [batch,time]")
            }
        }

        let number_features = self.number.features(inputs.seconds_total)?;
        let number_embedding = self.number.projection.forward(&number_features)?;
        let raw_context = assemble_raw_context(
            inputs.prompt,
            &number_embedding,
            zero_cross_context_from_batch,
        )?;
        let context = sequential_two(&self.to_cond_first, &self.to_cond_second, &raw_context)?;
        let duration_global = sequential_two(
            &self.to_global_first,
            &self.to_global_second,
            &number_embedding,
        )?;
        // Direct F32 t: no logSNR conversion.
        let timestep_features = expo_fourier_features(&inputs.timestep.to_dtype(DType::F32)?, 256)?;
        let timestep_embedding = sequential_two(
            &self.to_timestep_first,
            &self.to_timestep_second,
            &timestep_features,
        )?;
        let combined_global = (&duration_global + &timestep_embedding)?;
        let global_modulation =
            sequential_two(&self.global_first, &self.global_second, &combined_global)?;

        let preprocessed = (inputs.latents + self.preprocess.forward(inputs.latents)?)?;
        let projected_input = self
            .project_in
            .forward(&preprocessed.transpose(1, 2)?.contiguous()?)?;
        let (mut hidden, extended_mask) =
            self.memory.prepend(&projected_input, inputs.padding_mask)?;
        let rotary = self.rotary.frequencies(hidden.dim(1)?)?;
        let local = inputs.local_conditioning.transpose(1, 2)?.contiguous()?;

        if let Some(trace) = trace.as_deref_mut() {
            trace.number_features = Some(number_features.clone());
            trace.number_embedding = Some(number_embedding.clone());
            trace.raw_context = Some(raw_context.clone());
            trace.projected_context = Some(context.clone());
            trace.duration_global = Some(duration_global.clone());
            trace.timestep_features = Some(timestep_features.clone());
            trace.timestep_embedding = Some(timestep_embedding.clone());
            trace.combined_global = Some(combined_global.clone());
            trace.global_modulation = Some(global_modulation.clone());
            trace.preprocessed = Some(preprocessed.clone());
            trace.projected_input = Some(projected_input.clone());
            trace.with_memory = Some(hidden.clone());
            trace.rotary_frequencies = Some(rotary.clone());
            trace.layer0_local = self.blocks[0].project_local(&local, hidden.dim(1)?)?;
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if is_canceled.is_some_and(|probe| probe()) {
                bail!("Stable Audio 3 DiT canceled")
            }
            hidden = block.forward(
                &hidden,
                Some(&context),
                Some(&global_modulation),
                Some(&local),
                Some(&rotary),
                None,
                TransformerBlockMasks {
                    self_attention: AttentionMasks {
                        key_padding: extended_mask.as_ref(),
                        additive: None,
                    },
                    // Frozen DiT deliberately disables the T5 cross-attention mask.
                    cross_attention: AttentionMasks::default(),
                },
            )?;
            if index == 0 {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.layer0_output = Some(hidden.clone());
                }
            }
        }
        let trimmed = self.memory.trim(&hidden)?;
        let projected_output = self.project_out.forward(&trimmed)?;
        let projected_bct = projected_output.transpose(1, 2)?.contiguous()?;
        let output = (&projected_bct + self.postprocess.forward(&projected_bct)?)?;
        if let Some(trace) = trace {
            trace.trimmed = Some(trimmed);
            trace.projected_output = Some(projected_output);
            trace.output = Some(output.clone());
        }
        Ok(output)
    }

    /// Upstream classifier-free guidance with optional APG and channel-wise rescale.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_guided(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        positive_prompt: &Tensor,
        negative_prompt: Option<&Tensor>,
        negative_prompt_mask: Option<&Tensor>,
        seconds_total: &Tensor,
        local_conditioning: &Tensor,
        padding_mask: Option<&Tensor>,
        guidance: Guidance,
    ) -> Result<Tensor> {
        self.forward_guided_impl(
            latents,
            timestep,
            positive_prompt,
            negative_prompt,
            negative_prompt_mask,
            seconds_total,
            local_conditioning,
            padding_mask,
            guidance,
            None,
        )
    }

    /// Guided forward with cooperative cancellation before every transformer block.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_guided_with_cancel(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        positive_prompt: &Tensor,
        negative_prompt: Option<&Tensor>,
        negative_prompt_mask: Option<&Tensor>,
        seconds_total: &Tensor,
        local_conditioning: &Tensor,
        padding_mask: Option<&Tensor>,
        guidance: Guidance,
        is_canceled: &dyn Fn() -> bool,
    ) -> candle_audio::Result<Tensor> {
        match self.forward_guided_impl(
            latents,
            timestep,
            positive_prompt,
            negative_prompt,
            negative_prompt_mask,
            seconds_total,
            local_conditioning,
            padding_mask,
            guidance,
            Some(is_canceled),
        ) {
            Err(_) if is_canceled() => Err(candle_audio::AudioError::Canceled),
            Err(error) => Err(error.into()),
            Ok(output) => Ok(output),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_guided_impl(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        positive_prompt: &Tensor,
        negative_prompt: Option<&Tensor>,
        negative_prompt_mask: Option<&Tensor>,
        seconds_total: &Tensor,
        local_conditioning: &Tensor,
        padding_mask: Option<&Tensor>,
        guidance: Guidance,
        is_canceled: Option<&dyn Fn() -> bool>,
    ) -> Result<Tensor> {
        if guidance.cfg_scale == 1.0 {
            return self.forward_impl(
                DitInputs {
                    latents,
                    timestep,
                    prompt: positive_prompt,
                    seconds_total,
                    local_conditioning,
                    padding_mask,
                },
                None,
                None,
                is_canceled,
            );
        }
        let (batch, _, _) = latents.dims3()?;
        let zero_cross_context_from_batch = negative_prompt.is_none().then_some(batch);
        let negative = match negative_prompt {
            Some(negative) => {
                let mut negative = negative.clone();
                if let Some(mask) = negative_prompt_mask {
                    negative = negative
                        .broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(negative.dtype())?)?;
                }
                negative
            }
            None => Tensor::zeros_like(positive_prompt)?,
        };
        let batch_latents = Tensor::cat(&[latents, latents], 0)?;
        let batch_timestep = Tensor::cat(&[timestep, timestep], 0)?;
        let batch_prompt = Tensor::cat(&[positive_prompt, &negative], 0)?;
        let batch_seconds = Tensor::cat(&[seconds_total, seconds_total], 0)?;
        let batch_local = Tensor::cat(&[local_conditioning, local_conditioning], 0)?;
        let batch_mask = padding_mask
            .map(|mask| Tensor::cat(&[mask, mask], 0))
            .transpose()?;
        let predictions = self.forward_impl(
            DitInputs {
                latents: &batch_latents,
                timestep: &batch_timestep,
                prompt: &batch_prompt,
                seconds_total: &batch_seconds,
                local_conditioning: &batch_local,
                padding_mask: batch_mask.as_ref(),
            },
            None,
            zero_cross_context_from_batch,
            is_canceled,
        )?;
        let conditional = predictions.narrow(0, 0, batch)?;
        let unconditional = predictions.narrow(0, batch, batch)?;
        guided_prediction(
            self.objective,
            latents,
            timestep,
            &conditional,
            &unconditional,
            padding_mask,
            guidance,
        )
    }
}

/// Decompose `value` into components parallel and orthogonal to `reference`.
pub fn apg_project(
    value: &Tensor,
    reference: &Tensor,
    padding_mask: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    if value.dims() != reference.dims() || value.rank() != 3 {
        bail!("APG tensors must have matching [batch,channels,time] shapes")
    }
    let original = value.dtype();
    let value = value.to_dtype(DType::F32)?;
    let reference = reference.to_dtype(DType::F32)?;
    let (value_for_dot, normalized, output_mask) = match padding_mask {
        Some(mask) => {
            let mask = mask.unsqueeze(1)?.to_dtype(DType::F32)?;
            let value_masked = value.broadcast_mul(&mask)?;
            let reference_masked = reference.broadcast_mul(&mask)?;
            let norm = reference_masked
                .sqr()?
                .sum_keepdim((1, 2))?
                .sqrt()?
                .clamp(1e-8, f64::INFINITY)?;
            (
                value_masked,
                reference_masked.broadcast_div(&norm)?,
                Some(mask),
            )
        }
        None => {
            let norm = reference
                .sqr()?
                .sum_keepdim((1, 2))?
                .sqrt()?
                .clamp(1e-8, f64::INFINITY)?;
            (value.clone(), reference.broadcast_div(&norm)?, None)
        }
    };
    let coefficient = value_for_dot
        .broadcast_mul(&normalized)?
        .sum_keepdim((1, 2))?;
    let parallel = normalized.broadcast_mul(&coefficient)?;
    let orthogonal = (&value - &parallel)?;
    let orthogonal = match output_mask {
        Some(mask) => orthogonal.broadcast_mul(&mask)?,
        None => orthogonal,
    };
    Ok((parallel.to_dtype(original)?, orthogonal.to_dtype(original)?))
}

#[allow(clippy::too_many_arguments)]
fn guided_prediction(
    objective: DiffusionObjective,
    latents: &Tensor,
    timestep: &Tensor,
    conditional: &Tensor,
    unconditional: &Tensor,
    padding_mask: Option<&Tensor>,
    guidance: Guidance,
) -> Result<Tensor> {
    let sigma = match objective {
        DiffusionObjective::V => (timestep * (std::f64::consts::PI / 2.0))?.sin()?,
        DiffusionObjective::RfDenoiser | DiffusionObjective::RectifiedFlow => timestep.clone(),
    }
    .unsqueeze(1)?
    .unsqueeze(2)?;
    let conditional_denoised = match objective {
        DiffusionObjective::V => {
            let alpha = (timestep * (std::f64::consts::PI / 2.0))?
                .cos()?
                .unsqueeze(1)?
                .unsqueeze(2)?;
            latents
                .broadcast_mul(&alpha)?
                .broadcast_sub(&conditional.broadcast_mul(&sigma)?)?
        }
        _ => latents.broadcast_sub(&conditional.broadcast_mul(&sigma)?)?,
    };
    let unconditional_denoised = match objective {
        DiffusionObjective::V => {
            let alpha = (timestep * (std::f64::consts::PI / 2.0))?
                .cos()?
                .unsqueeze(1)?
                .unsqueeze(2)?;
            latents
                .broadcast_mul(&alpha)?
                .broadcast_sub(&unconditional.broadcast_mul(&sigma)?)?
        }
        _ => latents.broadcast_sub(&unconditional.broadcast_mul(&sigma)?)?,
    };
    let mut difference = (&conditional_denoised - &unconditional_denoised)?;
    if guidance.cfg_norm_threshold > 0.0 {
        let measured = match padding_mask {
            Some(mask) => difference
                .broadcast_mul(&mask.unsqueeze(1)?.to_dtype(difference.dtype())?)?
                .sqr()?
                .sum_keepdim((1, 2))?
                .sqrt()?,
            None => difference.sqr()?.sum_keepdim((1, 2))?.sqrt()?,
        };
        let denominator = measured.clamp(guidance.cfg_norm_threshold, f64::INFINITY)?;
        difference = difference.broadcast_mul(&(guidance.cfg_norm_threshold / denominator)?)?;
    }
    let guided_difference = if guidance.apg_scale == 0.0 {
        difference
    } else {
        let (_, orthogonal) = apg_project(&difference, &conditional_denoised, padding_mask)?;
        if guidance.apg_scale == 1.0 {
            orthogonal
        } else {
            ((orthogonal * guidance.apg_scale)? + (difference * (1.0 - guidance.apg_scale))?)?
        }
    };
    let guided_denoised =
        (&conditional_denoised + (guided_difference * (guidance.cfg_scale - 1.0))?)?;
    let mut output = match objective {
        DiffusionObjective::V => {
            let alpha = (timestep * (std::f64::consts::PI / 2.0))?
                .cos()?
                .unsqueeze(1)?
                .unsqueeze(2)?;
            latents
                .broadcast_mul(&alpha)?
                .broadcast_sub(&guided_denoised)?
                .broadcast_div(&sigma)?
        }
        _ => latents
            .broadcast_sub(&guided_denoised)?
            .broadcast_div(&sigma)?,
    };
    if guidance.scale_phi != 0.0 {
        let channel_std = |tensor: &Tensor| -> Result<Tensor> {
            let channels = tensor.dim(1)?;
            let centered = tensor.broadcast_sub(&tensor.mean_keepdim(1)?)?;
            (centered.sqr()?.sum_keepdim(1)? / (channels.saturating_sub(1).max(1) as f64))?.sqrt()
        };
        let ratio = channel_std(conditional)?
            .broadcast_div(&channel_std(&output)?.clamp(1e-12, f64::INFINITY)?)?;
        output = ((output.broadcast_mul(&ratio)? * guidance.scale_phi)?
            + (output * (1.0 - guidance.scale_phi))?)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expo_features_have_frozen_cos_then_sin_order_and_endpoints() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![0f32, 1.], 2, &device).unwrap();
        let features = expo_fourier_features(&t, 4)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(features[0], vec![1.0, 1.0, 0.0, 0.0]);
        let first_angle = std::f32::consts::PI;
        assert!((features[1][0] - first_angle.cos()).abs() < 1e-6);
        assert!((features[1][2] - first_angle.sin()).abs() < 1e-6);
    }

    #[test]
    fn cfg_raw_context_distinguishes_absent_and_explicit_negative_conditioning() {
        let device = Device::Cpu;
        let prompt = Tensor::from_vec(vec![1f32, 2., 0., 0.], (2, 2, 1), &device).unwrap();
        let duration = Tensor::from_vec(vec![3f32, 3.], (2, 1), &device).unwrap();

        let absent = assemble_raw_context(&prompt, &duration, Some(1))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(absent[0], vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(absent[1], vec![vec![0.0], vec![0.0], vec![0.0]]);

        // An explicit negative prompt has already had its invalid prompt rows zeroed, but its
        // independently conditioned duration row remains present exactly as frozen upstream.
        let explicit = assemble_raw_context(&prompt, &duration, None)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(explicit[1], vec![vec![0.0], vec![0.0], vec![3.0]]);
    }

    #[test]
    fn apg_zero_reference_and_mask_are_well_defined() {
        let device = Device::Cpu;
        let value = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &device).unwrap();
        let zero = Tensor::zeros_like(&value).unwrap();
        let (parallel, orthogonal) = apg_project(&value, &zero, None).unwrap();
        assert_eq!(
            parallel.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 4]
        );
        assert_eq!(
            orthogonal.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let invalid = Tensor::zeros((1, 2), DType::U8, &device).unwrap();
        let (_, masked) = apg_project(&value, &zero, Some(&invalid)).unwrap();
        assert_eq!(
            masked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 4]
        );
    }

    #[test]
    fn apg_projection_mask_dtype_and_guidance_endpoints_are_exact() {
        let device = Device::Cpu;
        let value = Tensor::from_vec(vec![3f32, 4.], (1, 1, 2), &device).unwrap();
        let reference = Tensor::from_vec(vec![1f32, 0.], (1, 1, 2), &device).unwrap();
        let (parallel, orthogonal) = apg_project(&value, &reference, None).unwrap();
        assert_eq!(
            parallel.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, 0.0]
        );
        assert_eq!(
            orthogonal.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 4.0]
        );
        let mask = Tensor::from_vec(vec![1u8, 0], (1, 2), &device).unwrap();
        let (_, masked) = apg_project(&value, &reference, Some(&mask)).unwrap();
        assert_eq!(
            masked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 0.0]
        );
        let bf16_value = value.to_dtype(DType::BF16).unwrap();
        let bf16_reference = reference.to_dtype(DType::BF16).unwrap();
        let (parallel, orthogonal) = apg_project(&bf16_value, &bf16_reference, None).unwrap();
        assert_eq!(parallel.dtype(), DType::BF16);
        assert_eq!(orthogonal.dtype(), DType::BF16);

        let latents = Tensor::ones((1, 1, 2), DType::F32, &device).unwrap();
        let timestep = Tensor::from_vec(vec![0.5f32], 1, &device).unwrap();
        let conditional = Tensor::from_vec(vec![0.2f32, -0.1], (1, 1, 2), &device).unwrap();
        let unconditional = Tensor::from_vec(vec![-0.2f32, 0.3], (1, 1, 2), &device).unwrap();
        let vanilla = guided_prediction(
            DiffusionObjective::RectifiedFlow,
            &latents,
            &timestep,
            &conditional,
            &unconditional,
            None,
            Guidance {
                cfg_scale: 3.0,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        )
        .unwrap();
        let orthogonal = guided_prediction(
            DiffusionObjective::RectifiedFlow,
            &latents,
            &timestep,
            &conditional,
            &unconditional,
            None,
            Guidance {
                cfg_scale: 3.0,
                apg_scale: 1.0,
                ..Guidance::default()
            },
        )
        .unwrap();
        let blended = guided_prediction(
            DiffusionObjective::RectifiedFlow,
            &latents,
            &timestep,
            &conditional,
            &unconditional,
            None,
            Guidance {
                cfg_scale: 3.0,
                apg_scale: 0.5,
                ..Guidance::default()
            },
        )
        .unwrap();
        let midpoint = ((&vanilla + &orthogonal).unwrap() / 2.0).unwrap();
        let error = blended
            .broadcast_sub(&midpoint)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "APG blend endpoint interpolation drifted");
    }
}
