//! Config-driven SAME patched autoencoder runtime.
//!
//! This is the shared implementation used by both SAME-S and SAME-L.  The small checkpoint uses
//! contiguous 34-token attention chunks; the later SAME-L slice extends the attention scheduling
//! seam for sliding windows without cloning the patch/resampling/SoftNorm pipeline here.

use candle_audio::candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::{linear_b, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::config::{
    AutoencoderConfig, DecoderConfig, EncoderConfig, FeedForwardConfig, NormConfig, NormType,
    QkNorm,
};
use crate::pretransform::PatchedPretransform;
use crate::softnorm::SoftNorm;
use crate::transformer::{
    is_sinusoidal_block, RotaryEmbedding, TransformerBlock, TransformerBlockMasks,
};
use crate::weight_norm::wn_conv1d;
use crate::weights::StableAudioVarBuilders;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Encode,
    Decode,
}

/// Semantic identity of one SAME stochastic draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameNoiseKind {
    SoftNormRegularization,
    EncoderTokens { stage: usize },
    DecoderTokens { stage: usize },
}

/// One captured unit-normal draw plus the scalar applied by the SAME runtime.
pub struct SameNoiseCapture {
    pub kind: SameNoiseKind,
    pub scale: f64,
    pub unit: Tensor,
}

/// Launch-portable, seedable SAME noise source.
///
/// Noise is sampled on the host through `StdRng`, transferred to the target device, and optionally
/// captured before scaling. One instance is threaded through a complete encode/decode so call order
/// is observable and stable across CPU, Metal, and CUDA.
pub struct SameNoiseRng {
    rng: StdRng,
    capture: bool,
    captures: Vec<SameNoiseCapture>,
}

impl SameNoiseRng {
    pub fn from_entropy() -> Self {
        Self {
            rng: StdRng::from_os_rng(),
            capture: false,
            captures: Vec::new(),
        }
    }

    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            capture: false,
            captures: Vec::new(),
        }
    }

    pub fn capturing(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            capture: true,
            captures: Vec::new(),
        }
    }

    pub fn captures(&self) -> &[SameNoiseCapture] {
        &self.captures
    }

    pub fn take_captures(&mut self) -> Vec<SameNoiseCapture> {
        std::mem::take(&mut self.captures)
    }

    fn scaled(
        &mut self,
        kind: SameNoiseKind,
        scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let count = shape.iter().try_fold(1usize, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| {
                candle_audio::candle_core::Error::Msg("SAME noise shape overflow".into())
            })
        })?;
        let values: Vec<f32> = (0..count)
            .map(|_| self.rng.sample(StandardNormal))
            .collect();
        let unit = Tensor::from_vec(values, shape.to_vec(), device)?.to_dtype(dtype)?;
        if self.capture {
            self.captures.push(SameNoiseCapture {
                kind,
                scale,
                unit: unit.clone(),
            });
        }
        unit.affine(scale, 0.0)
    }

    fn unit(
        &mut self,
        kind: SameNoiseKind,
        scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        self.scaled(kind, 1.0, shape, dtype, device).inspect(|_| {
            if let Some(capture) = self.captures.last_mut() {
                if capture.kind == kind {
                    capture.scale = scale;
                }
            }
        })
    }
}

enum Mapping {
    Identity,
    Conv(Conv1d),
}

impl Mapping {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity => Ok(x.clone()),
            Self::Conv(conv) => conv.forward(x),
        }
    }
}

struct SameTransformerLayer {
    block: TransformerBlock,
    rope: RotaryEmbedding,
}

/// Intermediate tensors exposed only when a caller explicitly requests parity evidence.
#[derive(Clone)]
pub struct ResamplingTrace {
    /// Sequence after channel mapping and transpose, before segment folding.
    pub mapped_sequence: Tensor,
    /// Folded 34-token chunks presented to the first transformer block.
    pub folded_input: Tensor,
    /// Learned tokens after variable-stride expansion and any configured/explicit noise.
    pub expanded_tokens: Tensor,
    /// Output of every transformer block, restored to `[B, sequence, D]` and edge-cropped.
    pub block_outputs: Vec<Tensor>,
    /// The selected learned-token portion of every final `(stride + 1)` subchunk.
    pub selected_segments: Tensor,
    /// Final channel-first block output.
    pub output: Tensor,
}

/// Frozen-input transformer evidence used to separate per-layer parity from accumulated roundoff.
#[doc(hidden)]
pub struct ControlledResamplingTrace {
    pub block_outputs: Vec<Tensor>,
    pub selected_segments: Tensor,
    pub output: Tensor,
}

struct TransformerResamplingBlock {
    direction: Direction,
    mapping: Mapping,
    new_tokens: Tensor,
    transformers: Vec<SameTransformerLayer>,
    stride: usize,
    chunk_size: usize,
    chunk_midpoint_shift: bool,
    variable_stride: bool,
    mask_noise: f64,
}

#[derive(Clone, Copy)]
struct BlockOptions {
    stride: usize,
    transformer_depth: usize,
    dim_head: usize,
    chunk_size: usize,
    chunk_midpoint_shift: bool,
    differential: bool,
    variable_stride: bool,
    mask_noise: f64,
    conv_mapping: bool,
    mapping_bias: bool,
    dyt: bool,
    ff_mult: f64,
    layer_scale: bool,
    sinusoidal_blocks: usize,
}

impl TransformerResamplingBlock {
    fn load(
        direction: Direction,
        in_channels: usize,
        out_channels: usize,
        options: BlockOptions,
        vb: VarBuilder,
    ) -> Result<Self> {
        if options.stride == 0 || options.chunk_size == 0 {
            bail!("SAME stride and chunk_size must be non-zero")
        }
        if !options.chunk_size.is_multiple_of(options.stride) {
            bail!(
                "SAME stride {} must divide chunk_size {}",
                options.stride,
                options.chunk_size
            )
        }
        let transformer_dim = match direction {
            Direction::Encode => out_channels,
            Direction::Decode => in_channels,
        };
        if transformer_dim % options.dim_head != 0 {
            bail!(
                "SAME transformer dim {transformer_dim} is not divisible by head dim {}",
                options.dim_head
            )
        }
        let mapping = if in_channels == out_channels {
            Mapping::Identity
        } else {
            let kernel = if options.conv_mapping { 3 } else { 1 };
            Mapping::Conv(wn_conv1d(
                in_channels,
                out_channels,
                kernel,
                options.mapping_bias,
                Conv1dConfig {
                    padding: kernel / 2,
                    ..Default::default()
                },
                vb.pp("mapping"),
            )?)
        };
        let norm_type = if options.dyt {
            NormType::Dyt
        } else {
            NormType::RmsNorm
        };
        let qk_norm = if options.dyt {
            QkNorm::Dyt
        } else {
            QkNorm::Rms
        };
        let norm = NormConfig {
            eps: 1e-3,
            ..Default::default()
        };
        let mut transformers = Vec::with_capacity(options.transformer_depth);
        for index in 0..options.transformer_depth {
            let block_vb = vb.pp(format!("transformers.{index}"));
            let ff = FeedForwardConfig {
                mult: options.ff_mult,
                no_bias: false,
                sinusoidal: is_sinusoidal_block(
                    index,
                    options.transformer_depth,
                    options.sinusoidal_blocks,
                ),
                ..Default::default()
            };
            transformers.push(SameTransformerLayer {
                block: TransformerBlock::load(
                    transformer_dim,
                    options.dim_head,
                    None,
                    norm_type,
                    &norm,
                    qk_norm,
                    1e-3,
                    options.differential,
                    false,
                    &ff,
                    !options.layer_scale,
                    false,
                    None,
                    options.layer_scale,
                    block_vb.clone(),
                )?,
                // Upstream constructs RotaryEmbedding(dim_heads / 2).
                rope: RotaryEmbedding::load(options.dim_head / 2, block_vb.pp("rope"))?,
            });
        }
        let output_segment = match direction {
            Direction::Encode => 1,
            Direction::Decode => options.stride,
        };
        let token_segment = if options.variable_stride {
            1
        } else {
            output_segment
        };
        Ok(Self {
            direction,
            mapping,
            new_tokens: vb.get((1, token_segment, transformer_dim), "new_tokens")?,
            transformers,
            stride: options.stride,
            chunk_size: options.chunk_size,
            chunk_midpoint_shift: options.chunk_midpoint_shift,
            variable_stride: options.variable_stride,
            mask_noise: options.mask_noise,
        })
    }

    fn active_stride(&self, override_stride: Option<usize>) -> Result<usize> {
        match override_stride {
            None => Ok(self.stride),
            Some(stride) => {
                if !self.variable_stride {
                    bail!("SAME stride override requires variable_stride=true")
                }
                if stride == 0 || !self.chunk_size.is_multiple_of(stride) {
                    bail!(
                        "SAME override stride {stride} must be non-zero and divide chunk_size {}",
                        self.chunk_size
                    )
                }
                Ok(stride)
            }
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        override_stride: Option<usize>,
        override_new_tokens: Option<&Tensor>,
        noise_rng: Option<(&mut SameNoiseRng, SameNoiseKind)>,
        trace: bool,
    ) -> Result<(Tensor, Option<ResamplingTrace>)> {
        let (batch, _, _) = x.dims3()?;
        let stride = self.active_stride(override_stride)?;
        let input_segment = match self.direction {
            Direction::Encode => stride,
            Direction::Decode => 1,
        };
        let output_segment = match self.direction {
            Direction::Encode => 1,
            Direction::Decode => stride,
        };
        let sub_chunk = stride + 1;

        let mut x = x.clone();
        if self.direction == Direction::Encode && !self.transformers.is_empty() {
            x = zero_pad_channel_first(&x, self.chunk_size)?;
        }
        if self.direction == Direction::Encode {
            x = self.mapping.forward(&x)?;
        }
        if self.transformers.is_empty() {
            if self.direction == Direction::Decode {
                x = self.mapping.forward(&x)?;
            }
            return Ok((x, None));
        }

        x = x.transpose(1, 2)?;
        let mapped_sequence = x.clone();
        if self.direction == Direction::Decode {
            x = zero_pad_sequence(&x, self.chunk_size / stride)?;
        }
        let (_, length, dim) = x.dims3()?;
        if !length.is_multiple_of(input_segment) {
            bail!("SAME input length {length} is not divisible by segment {input_segment}")
        }
        let segments = length / input_segment;
        x = x.reshape((batch, segments, input_segment, dim))?.reshape((
            batch * segments,
            input_segment,
            dim,
        ))?;
        let mut tokens = self
            .new_tokens
            .expand((batch * segments, output_segment, dim))?;
        if let Some(noise) = override_new_tokens {
            let expected = (batch, segments * output_segment, dim);
            if noise.dims3()? != expected {
                bail!(
                    "SAME override token noise shape {:?}, expected {:?}",
                    noise.shape(),
                    expected
                )
            }
            tokens = tokens.broadcast_add(
                &noise
                    .reshape((batch, segments, output_segment, dim))?
                    .reshape((batch * segments, output_segment, dim))?,
            )?;
        } else if self.mask_noise > 0.0 {
            let noise = match noise_rng {
                Some((rng, kind)) => rng.scaled(
                    kind,
                    self.mask_noise,
                    tokens.dims(),
                    tokens.dtype(),
                    tokens.device(),
                )?,
                None => Tensor::randn(
                    0f32,
                    self.mask_noise as f32,
                    tokens.shape(),
                    tokens.device(),
                )?
                .to_dtype(tokens.dtype())?,
            };
            tokens = (&tokens + noise)?;
        }
        let expanded_tokens = tokens.reshape((batch, segments * output_segment, dim))?;
        x = Tensor::cat(&[&x, &tokens], 1)?
            .reshape((batch, segments, sub_chunk, dim))?
            .reshape((batch, segments * sub_chunk, dim))?;

        let effective_chunk = self.chunk_size + self.chunk_size / stride;
        if !x.dim(1)?.is_multiple_of(effective_chunk) {
            bail!(
                "SAME folded sequence {} is not divisible by effective chunk {effective_chunk}",
                x.dim(1)?
            )
        }
        let folded_input = fold_chunks(&x, effective_chunk)?;
        let mut block_outputs = Vec::with_capacity(self.transformers.len());
        if self.chunk_midpoint_shift {
            let split = self.transformers.len() / 2;
            let shift = effective_chunk / 2;
            let mut folded = folded_input.clone();
            for layer in &self.transformers[..split] {
                let rope = layer.rope.frequencies(effective_chunk)?;
                folded = layer.block.forward(
                    &folded,
                    None,
                    None,
                    None,
                    Some(&rope),
                    None,
                    TransformerBlockMasks::default(),
                )?;
                if trace {
                    block_outputs.push(unfold_chunks(&folded, batch)?);
                }
            }
            x = unfold_chunks(&folded, batch)?;
            let first = x.narrow(1, 0, shift)?;
            let last = x.narrow(1, x.dim(1)? - shift, shift)?;
            x = Tensor::cat(&[&first, &x, &last], 1)?;
            folded = fold_chunks(&x, effective_chunk)?;
            for layer in &self.transformers[split..] {
                let rope = layer.rope.frequencies(effective_chunk)?;
                folded = layer.block.forward(
                    &folded,
                    None,
                    None,
                    None,
                    Some(&rope),
                    None,
                    TransformerBlockMasks::default(),
                )?;
                if trace {
                    let restored = unfold_chunks(&folded, batch)?;
                    block_outputs.push(restored.narrow(1, shift, restored.dim(1)? - 2 * shift)?);
                }
            }
            x = unfold_chunks(&folded, batch)?;
            x = x.narrow(1, shift, x.dim(1)? - 2 * shift)?;
        } else {
            let mut folded = folded_input.clone();
            for layer in &self.transformers {
                let rope = layer.rope.frequencies(effective_chunk)?;
                folded = layer.block.forward(
                    &folded,
                    None,
                    None,
                    None,
                    Some(&rope),
                    None,
                    TransformerBlockMasks::default(),
                )?;
                if trace {
                    block_outputs.push(unfold_chunks(&folded, batch)?);
                }
            }
            x = unfold_chunks(&folded, batch)?;
        }

        if !x.dim(1)?.is_multiple_of(sub_chunk) {
            bail!(
                "SAME transformed sequence {} is not divisible by subchunk {sub_chunk}",
                x.dim(1)?
            )
        }
        let subchunks = x.dim(1)? / sub_chunk;
        let selected = x.reshape((batch * subchunks, sub_chunk, dim))?.narrow(
            1,
            sub_chunk - output_segment,
            output_segment,
        )?;
        let selected_segments = selected.reshape((batch, subchunks * output_segment, dim))?;
        x = selected_segments.transpose(1, 2)?;
        if self.direction == Direction::Decode {
            x = self.mapping.forward(&x)?;
        }
        let result_trace = trace.then(|| ResamplingTrace {
            mapped_sequence,
            folded_input,
            expanded_tokens,
            block_outputs,
            selected_segments,
            output: x.clone(),
        });
        Ok((x, result_trace))
    }

    fn forward_controlled(
        &self,
        batch: usize,
        override_stride: Option<usize>,
        block_inputs: &[Tensor],
    ) -> Result<ControlledResamplingTrace> {
        if block_inputs.len() != self.transformers.len() {
            bail!(
                "SAME controlled trace expected {} block inputs, got {}",
                self.transformers.len(),
                block_inputs.len()
            )
        }
        let stride = self.active_stride(override_stride)?;
        let effective_chunk = self.chunk_size + self.chunk_size / stride;
        let split = self.transformers.len() / 2;
        let shift = effective_chunk / 2;
        let mut block_outputs = Vec::with_capacity(self.transformers.len());
        let mut final_raw = None;
        for (index, (layer, input)) in self
            .transformers
            .iter()
            .zip(block_inputs.iter())
            .enumerate()
        {
            if input.dim(1)? != effective_chunk {
                bail!(
                    "SAME controlled block {index} has chunk {}, expected {effective_chunk}",
                    input.dim(1)?
                )
            }
            let rope = layer.rope.frequencies(effective_chunk)?;
            let raw = layer.block.forward(
                input,
                None,
                None,
                None,
                Some(&rope),
                None,
                TransformerBlockMasks::default(),
            )?;
            let mut restored = unfold_chunks(&raw, batch)?;
            if self.chunk_midpoint_shift && index >= split {
                restored = restored.narrow(1, shift, restored.dim(1)? - 2 * shift)?;
            }
            block_outputs.push(restored);
            final_raw = Some(raw);
        }
        let Some(final_raw) = final_raw.as_ref() else {
            bail!("SAME controlled trace requires a block")
        };
        let mut sequence = unfold_chunks(final_raw, batch)?;
        if self.chunk_midpoint_shift {
            sequence = sequence.narrow(1, shift, sequence.dim(1)? - 2 * shift)?;
        }
        let output_segment = match self.direction {
            Direction::Encode => 1,
            Direction::Decode => stride,
        };
        let sub_chunk = stride + 1;
        if !sequence.dim(1)?.is_multiple_of(sub_chunk) {
            bail!(
                "SAME controlled sequence {} is not divisible by subchunk {sub_chunk}",
                sequence.dim(1)?
            )
        }
        let subchunks = sequence.dim(1)? / sub_chunk;
        let selected_segments = sequence
            .reshape((batch * subchunks, sub_chunk, sequence.dim(2)?))?
            .narrow(1, sub_chunk - output_segment, output_segment)?
            .reshape((batch, subchunks * output_segment, sequence.dim(2)?))?;
        let mut output = selected_segments.transpose(1, 2)?;
        if self.direction == Direction::Decode {
            output = self.mapping.forward(&output)?;
        }
        Ok(ControlledResamplingTrace {
            block_outputs,
            selected_segments,
            output,
        })
    }
}

fn zero_pad_channel_first(x: &Tensor, modulo: usize) -> Result<Tensor> {
    let (batch, channels, length) = x.dims3()?;
    let padded = length.div_ceil(modulo) * modulo;
    if padded == length {
        return Ok(x.clone());
    }
    let zeros = Tensor::zeros((batch, channels, padded - length), x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], 2)
}

fn zero_pad_sequence(x: &Tensor, modulo: usize) -> Result<Tensor> {
    let (batch, length, dim) = x.dims3()?;
    let padded = length.div_ceil(modulo) * modulo;
    if padded == length {
        return Ok(x.clone());
    }
    let zeros = Tensor::zeros((batch, padded - length, dim), x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], 1)
}

fn fold_chunks(x: &Tensor, chunk: usize) -> Result<Tensor> {
    let (batch, length, dim) = x.dims3()?;
    if !length.is_multiple_of(chunk) {
        bail!("SAME sequence length {length} is not divisible by chunk {chunk}")
    }
    x.reshape((batch, length / chunk, chunk, dim))?
        .reshape((batch * (length / chunk), chunk, dim))
}

fn unfold_chunks(x: &Tensor, batch: usize) -> Result<Tensor> {
    let (folded_batch, chunk, dim) = x.dims3()?;
    if !folded_batch.is_multiple_of(batch) {
        bail!("SAME folded batch {folded_batch} is not divisible by batch {batch}")
    }
    x.reshape((batch, folded_batch / batch, chunk, dim))?
        .reshape((batch, (folded_batch / batch) * chunk, dim))
}

/// Traces for each encoder or decoder resampling stage, in execution order.
pub struct SameTrace {
    pub stages: Vec<ResamplingTrace>,
    pub output: Tensor,
}

/// Config-driven SAME autoencoder. It is intentionally unregistered until the later provider slice.
pub struct SameAutoencoder {
    patch: PatchedPretransform,
    encoder: Vec<TransformerResamplingBlock>,
    encoder_out: Linear,
    decoder_in: Linear,
    decoder: Vec<TransformerResamplingBlock>,
    bottleneck: SoftNorm,
    soft_clip: bool,
    encoder_variable_stride: bool,
    decoder_variable_stride: bool,
}

impl SameAutoencoder {
    pub fn load(config: &AutoencoderConfig, builders: StableAudioVarBuilders<'_>) -> Result<Self> {
        validate_supported(config)?;
        let patch = PatchedPretransform::new(
            config.pretransform.config.channels,
            config.pretransform.config.patch_size,
        )?;
        let encoder = load_encoder(&config.encoder.config, builders.encoder.clone())?;
        let encoder_depth = config.encoder.config.c_mults.len();
        let encoder_dim =
            config.encoder.config.channels * config.encoder.config.c_mults[encoder_depth - 1];
        let encoder_out = linear_b(
            encoder_dim,
            config.latent_dim,
            true,
            builders.encoder.pp(format!("layers.{}", encoder_depth + 1)),
        )?;

        let decoder_depth = config.decoder.config.c_mults.len();
        let decoder_dim =
            config.decoder.config.channels * config.decoder.config.c_mults[decoder_depth - 1];
        let decoder_in = linear_b(
            config.latent_dim,
            decoder_dim,
            true,
            builders.decoder.pp("layers.1"),
        )?;
        let decoder = load_decoder(&config.decoder.config, builders.decoder.clone())?;
        let bottleneck_cfg = &config.bottleneck.config;
        Ok(Self {
            patch,
            encoder,
            encoder_out,
            decoder_in,
            decoder,
            bottleneck: SoftNorm::load(
                bottleneck_cfg.dim,
                bottleneck_cfg.noise_augment_dim,
                bottleneck_cfg.noise_regularize,
                bottleneck_cfg.auto_scale,
                builders.bottleneck,
            )?,
            soft_clip: config.decoder.soft_clip,
            encoder_variable_stride: config.encoder.config.variable_stride,
            decoder_variable_stride: config.decoder.config.variable_stride,
        })
    }

    /// Encode stereo waveform `[B, 2, samples]` to channel-first SAME latents.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let mut rng = SameNoiseRng::from_entropy();
        self.encode_with_rng(audio, None, &mut rng)
    }

    pub fn encode_with_strides(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
    ) -> Result<Tensor> {
        let mut rng = SameNoiseRng::from_entropy();
        self.encode_with_rng(audio, override_strides, &mut rng)
    }

    /// Seedable encoder path retained for SAME-L's configured learned-token noise.
    pub fn encode_with_rng(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
        rng: &mut SameNoiseRng,
    ) -> Result<Tensor> {
        Ok(self
            .encode_internal(audio, override_strides, None, Some(rng), false)?
            .0)
    }

    /// Explicit encoder token-noise parity seam, in encoder stage order.
    pub fn encode_with_noise(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
        mask_noises: &[Tensor],
    ) -> Result<Tensor> {
        Ok(self
            .encode_internal(audio, override_strides, Some(mask_noises), None, false)?
            .0)
    }

    pub fn encode_with_trace(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
    ) -> Result<(Tensor, SameTrace)> {
        let mut rng = SameNoiseRng::from_entropy();
        let (output, stages) =
            self.encode_internal(audio, override_strides, None, Some(&mut rng), true)?;
        Ok((output.clone(), SameTrace { stages, output }))
    }

    fn encode_internal(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
        mask_noises: Option<&[Tensor]>,
        mut rng: Option<&mut SameNoiseRng>,
        trace: bool,
    ) -> Result<(Tensor, Vec<ResamplingTrace>)> {
        validate_strides(
            override_strides,
            self.encoder.len(),
            self.encoder_variable_stride,
        )?;
        validate_noise_count("encoder", mask_noises, self.encoder.len())?;
        let mut x = self.patch.encode(audio)?;
        let mut traces = Vec::new();
        for (index, block) in self.encoder.iter().enumerate() {
            let noise_rng = rng
                .as_deref_mut()
                .map(|rng| (rng, SameNoiseKind::EncoderTokens { stage: index }));
            let (next, block_trace) = block.forward(
                &x,
                override_strides.map(|s| s[index]),
                mask_noises.map(|noises| &noises[index]),
                noise_rng,
                trace,
            )?;
            x = next;
            if let Some(block_trace) = block_trace {
                traces.push(block_trace);
            }
        }
        x = self
            .encoder_out
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        x = self.bottleneck.encode(&x)?;
        Ok((x, traces))
    }

    /// Production decode. Both upstream stochastic paths remain active in evaluation mode.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut rng = SameNoiseRng::from_entropy();
        self.decode_with_rng(latents, None, &mut rng)
    }

    /// Seedable/captureable production decode.
    ///
    /// The single RNG is called first for SoftNorm regularization and then once per decoder stage
    /// with nonzero token noise, preserving upstream evaluation order.
    pub fn decode_with_rng(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        rng: &mut SameNoiseRng,
    ) -> Result<Tensor> {
        Ok(self
            .decode_internal(
                latents,
                override_strides,
                None,
                None,
                Some(rng),
                false,
                false,
            )?
            .0)
    }

    /// Deterministic parity seam.
    ///
    /// `regularization_noise` is unit-normal and is scaled by SoftNorm's evaluation factor.
    /// Each `mask_noise` tensor is the already-scaled offset added to that decoder stage's learned
    /// tokens, shaped `[B, padded_segments * stride, D]`.
    pub fn decode_with_noise(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        regularization_noise: Option<&Tensor>,
        mask_noises: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        self.decode_with_noise_mode(
            latents,
            override_strides,
            regularization_noise,
            mask_noises,
            false,
        )
    }

    /// Explicit-noise training/evaluation mutation seam.
    pub fn decode_with_noise_mode(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        regularization_noise: Option<&Tensor>,
        mask_noises: Option<&[Tensor]>,
        training: bool,
    ) -> Result<Tensor> {
        Ok(self
            .decode_internal(
                latents,
                override_strides,
                regularization_noise,
                mask_noises,
                None,
                training,
                false,
            )?
            .0)
    }

    pub fn decode_with_trace(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        regularization_noise: Option<&Tensor>,
        mask_noises: Option<&[Tensor]>,
    ) -> Result<(Tensor, SameTrace)> {
        let (output, stages) = self.decode_internal(
            latents,
            override_strides,
            regularization_noise,
            mask_noises,
            None,
            false,
            true,
        )?;
        Ok((output.clone(), SameTrace { stages, output }))
    }

    /// Replays one decoder resampling stage from frozen upstream inputs at every block.
    #[doc(hidden)]
    pub fn decode_stage_with_controlled_block_inputs(
        &self,
        stage: usize,
        batch: usize,
        override_stride: Option<usize>,
        block_inputs: &[Tensor],
    ) -> Result<ControlledResamplingTrace> {
        let Some(block) = self.decoder.get(stage) else {
            bail!(
                "SAME decoder stage {stage} is out of range for {} stages",
                self.decoder.len()
            )
        };
        block.forward_controlled(batch, override_stride, block_inputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_internal(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        regularization_noise: Option<&Tensor>,
        mask_noises: Option<&[Tensor]>,
        mut rng: Option<&mut SameNoiseRng>,
        training: bool,
        trace: bool,
    ) -> Result<(Tensor, Vec<ResamplingTrace>)> {
        validate_strides(
            override_strides,
            self.decoder.len(),
            self.decoder_variable_stride,
        )?;
        validate_noise_count("decoder", mask_noises, self.decoder.len())?;
        let generated_regularization =
            if regularization_noise.is_none() && self.bottleneck.noise_regularize() {
                rng.as_deref_mut()
                    .map(|rng| {
                        rng.unit(
                            SameNoiseKind::SoftNormRegularization,
                            if training { 5e-2 } else { 1e-3 },
                            latents.dims(),
                            latents.dtype(),
                            latents.device(),
                        )
                    })
                    .transpose()?
            } else {
                None
            };
        let regularization_noise = regularization_noise.or(generated_regularization.as_ref());
        let mut x =
            self.bottleneck
                .decode_with_noise(latents, training, regularization_noise, None)?;
        x = self
            .decoder_in
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let mut traces = Vec::new();
        for (index, block) in self.decoder.iter().enumerate() {
            let noise_rng = rng
                .as_deref_mut()
                .map(|rng| (rng, SameNoiseKind::DecoderTokens { stage: index }));
            let (next, block_trace) = block.forward(
                &x,
                override_strides.map(|s| s[index]),
                mask_noises.map(|n| &n[index]),
                noise_rng,
                trace,
            )?;
            x = next;
            if let Some(block_trace) = block_trace {
                traces.push(block_trace);
            }
        }
        x = self.patch.decode(&x)?;
        if self.soft_clip {
            x = x.tanh()?;
        }
        Ok((x, traces))
    }

    /// Crop the padded decode to a caller-owned original length.
    pub fn crop_valid_prefix(decoded: &Tensor, original_samples: usize) -> Result<Tensor> {
        if original_samples > decoded.dim(2)? {
            bail!(
                "SAME crop length {original_samples} exceeds decoded length {}",
                decoded.dim(2)?
            )
        }
        decoded.narrow(2, 0, original_samples)
    }
}

fn validate_strides(strides: Option<&[usize]>, depth: usize, variable_stride: bool) -> Result<()> {
    if let Some(strides) = strides {
        if !variable_stride {
            bail!("SAME stride override requires variable_stride=true")
        }
        if strides.len() != depth {
            bail!(
                "SAME stride override needs {depth} values, got {}",
                strides.len()
            )
        }
    }
    Ok(())
}

fn validate_noise_count(direction: &str, noises: Option<&[Tensor]>, depth: usize) -> Result<()> {
    if let Some(noises) = noises {
        if noises.len() != depth {
            bail!(
                "SAME {direction} expected {depth} token-noise tensors, got {}",
                noises.len()
            )
        }
    }
    Ok(())
}

fn validate_supported(config: &AutoencoderConfig) -> Result<()> {
    let encoder = &config.encoder.config;
    let decoder = &config.decoder.config;
    if encoder.c_mults.is_empty() || decoder.c_mults.is_empty() {
        bail!("SAME encoder and decoder must each contain at least one resampling stage")
    }
    let patched_channels =
        config.pretransform.config.channels * config.pretransform.config.patch_size;
    if encoder.in_channels != patched_channels || decoder.out_channels != patched_channels {
        bail!(
            "SAME patched channels {patched_channels} disagree with encoder {} / decoder {}",
            encoder.in_channels,
            decoder.out_channels
        )
    }
    if config.bottleneck.config.dim != config.latent_dim {
        bail!(
            "SAME bottleneck dim {} disagrees with latent dim {}",
            config.bottleneck.config.dim,
            config.latent_dim
        )
    }
    if encoder.sliding_window.is_some() || decoder.sliding_window.is_some() {
        bail!("SAME sliding-window attention is not yet available; use the contiguous SAME runtime")
    }
    if encoder.conformer || decoder.conformer {
        bail!("SAME conformer blocks are not supported by shipped SA3 checkpoints")
    }
    if encoder.cross_attn || decoder.cross_attn {
        bail!("SAME autoencoder cross-attention is not supported")
    }
    if encoder.feat_scale || decoder.feat_scale {
        bail!("SAME feature scaling is not supported by shipped SA3 checkpoints")
    }
    if encoder.causal || decoder.causal {
        bail!("SAME autoencoder must be non-causal")
    }
    if encoder.use_snake
        || decoder.use_snake
        || encoder.use_dilated_conv
        || decoder.use_dilated_conv
    {
        bail!("SAME snake/dilated-convolution blocks are not supported")
    }
    if encoder.mapping_style != "none" || decoder.mapping_style != "none" {
        bail!("SAME mapping_style must be none")
    }
    if config.bottleneck.config.noise_augment_dim != 0 {
        bail!("SAME decoder noise augmentation channels are not supported")
    }
    Ok(())
}

fn load_encoder(config: &EncoderConfig, vb: VarBuilder) -> Result<Vec<TransformerResamplingBlock>> {
    let mut dimensions = Vec::with_capacity(config.c_mults.len() + 1);
    dimensions.push(config.in_channels);
    dimensions.extend(config.c_mults.iter().map(|m| m * config.channels));
    let mut blocks = Vec::with_capacity(config.c_mults.len());
    for index in 0..config.c_mults.len() {
        blocks.push(TransformerResamplingBlock::load(
            Direction::Encode,
            dimensions[index],
            dimensions[index + 1],
            BlockOptions {
                stride: config.strides[index],
                transformer_depth: config.transformer_depths[index],
                dim_head: config.dim_heads,
                chunk_size: config.chunk_size,
                chunk_midpoint_shift: config.chunk_midpoint_shift,
                differential: config.differential,
                variable_stride: config.variable_stride,
                mask_noise: config.mask_noise,
                conv_mapping: config.conv_mapping,
                mapping_bias: config.mapping_bias,
                dyt: config.dyt,
                ff_mult: config.ff_mult,
                layer_scale: config.layer_scale,
                sinusoidal_blocks: 0,
            },
            vb.pp(format!("layers.{index}")),
        )?);
    }
    Ok(blocks)
}

fn load_decoder(config: &DecoderConfig, vb: VarBuilder) -> Result<Vec<TransformerResamplingBlock>> {
    let depth = config.c_mults.len();
    let mut dimensions = Vec::with_capacity(depth + 1);
    dimensions.push(config.out_channels);
    dimensions.extend(config.c_mults.iter().map(|m| m * config.channels));
    let mut blocks = Vec::with_capacity(depth);
    for (execution_index, stage) in (1..=depth).rev().enumerate() {
        blocks.push(TransformerResamplingBlock::load(
            Direction::Decode,
            dimensions[stage],
            dimensions[stage - 1],
            BlockOptions {
                stride: config.strides[stage - 1],
                transformer_depth: config.transformer_depths[stage - 1],
                dim_head: config.dim_heads,
                chunk_size: config.chunk_size,
                chunk_midpoint_shift: config.chunk_midpoint_shift,
                differential: config.differential,
                variable_stride: config.variable_stride,
                mask_noise: config.mask_noise,
                conv_mapping: config.conv_mapping,
                mapping_bias: config.mapping_bias,
                dyt: config.dyt,
                ff_mult: config.ff_mult,
                layer_scale: config.layer_scale,
                sinusoidal_blocks: config.sinusoidal_blocks[stage - 1],
            },
            vb.pp(format!("layers.{}", execution_index + 3)),
        )?);
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn fold_unfold_preserves_batch_order() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..2 * 68 * 3).map(|v| v as f32).collect(),
            (2, 68, 3),
            &device,
        )
        .unwrap();
        let folded = fold_chunks(&x, 34).unwrap();
        assert_eq!(folded.dims3().unwrap(), (4, 34, 3));
        assert_eq!(
            unfold_chunks(&folded, 2).unwrap().to_vec3::<f32>().unwrap(),
            x.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn stride_validation_rejects_invalid_surface() {
        assert!(validate_strides(Some(&[16]), 1, false).is_err());
        assert!(validate_strides(Some(&[16, 8]), 1, true).is_err());
        assert!(validate_strides(Some(&[16]), 1, true).is_ok());
    }
}
