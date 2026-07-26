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
use crate::sampler::SeededNoise;
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

/// Upstream's default bounded query size for the SAME-L chunked-halo fallback.
pub const SAME_BAND_QUERY_TILE: usize = 1024;
pub const SAME_OUTER_CHUNK_SIZE: usize = 128;
pub const SAME_OUTER_CHUNK_OVERLAP: usize = 32;

/// Request-time policy for the outer autoencoder chunker.
///
/// Full Stable Audio 3 checkpoints pass their parsed `pretransform.chunked` value as
/// `model_default`; generation decode may additionally supply `request_override`. Standalone
/// autoencoder calls use `model_default=false` and an explicit override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SameChunkingPolicy {
    pub model_default: bool,
    pub request_override: Option<bool>,
}

impl SameChunkingPolicy {
    /// Full-model encode is config-driven; upstream exposes no request-time encode override.
    pub const fn full_model_encode(model_default: bool) -> Self {
        Self {
            model_default,
            request_override: None,
        }
    }

    /// Full-model generation decode accepts the public tri-state override.
    pub const fn full_model_decode(model_default: bool, request_override: Option<bool>) -> Self {
        Self {
            model_default,
            request_override,
        }
    }

    pub const fn standalone(enabled: bool) -> Self {
        Self {
            model_default: false,
            request_override: Some(enabled),
        }
    }

    pub const fn enabled(self) -> bool {
        match self.request_override {
            Some(enabled) => enabled,
            None => self.model_default,
        }
    }
}

/// Latent-space parameters for upstream's outer SAME chunker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SameChunkingParameters {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for SameChunkingParameters {
    fn default() -> Self {
        Self {
            chunk_size: SAME_OUTER_CHUNK_SIZE,
            overlap: SAME_OUTER_CHUNK_OVERLAP,
        }
    }
}

/// One disjoint final-ownership slice.
///
/// Frozen upstream writes every chunk sequentially into one output tensor. Later chunks overwrite
/// earlier chunks where a right-anchored final window creates extra overlap. These disjoint slices
/// are the equivalent immutable representation of that exact last-writer-wins result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SameChunkOwnership {
    pub chunk_index: usize,
    pub chunk_start: usize,
    pub source_start: usize,
    pub source_end: usize,
    pub output_start: usize,
    pub output_end: usize,
}

/// Pure, unit-agnostic plan for upstream's hard overlap-discard chunker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SameChunkPlan {
    pub total_units: usize,
    pub chunk_size: usize,
    pub overlap: usize,
    pub starts: Vec<usize>,
    pub ownership: Vec<SameChunkOwnership>,
    pub chunked: bool,
}

impl SameChunkPlan {
    pub fn build(
        total_units: usize,
        enabled: bool,
        parameters: SameChunkingParameters,
    ) -> Result<Self> {
        if !enabled {
            return Ok(Self::direct(total_units, parameters));
        }
        if parameters.chunk_size == 0 {
            bail!("SAME outer chunk_size must be non-zero")
        }
        if parameters.overlap >= parameters.chunk_size {
            bail!(
                "SAME outer overlap {} must be smaller than chunk_size {}",
                parameters.overlap,
                parameters.chunk_size
            )
        }
        if total_units < parameters.chunk_size {
            return Ok(Self::direct(total_units, parameters));
        }

        let hop = parameters.chunk_size - parameters.overlap;
        let final_start = total_units - parameters.chunk_size;
        let mut starts = Vec::new();
        let mut start = 0usize;
        loop {
            if start > final_start {
                break;
            }
            starts.push(start);
            let Some(next) = start.checked_add(hop) else {
                bail!("SAME outer chunk start overflow")
            };
            start = next;
        }
        if starts.last().copied() != Some(final_start) {
            starts.push(final_start);
        }

        let half_overlap = parameters.overlap / 2;
        let mut ownership = Vec::with_capacity(starts.len());
        for (index, &chunk_start) in starts.iter().enumerate() {
            let output_start = if index == 0 {
                0
            } else {
                chunk_start.checked_add(half_overlap).ok_or_else(|| {
                    candle_audio::candle_core::Error::Msg(
                        "SAME outer ownership start overflow".into(),
                    )
                })?
            };
            let output_end = if index + 1 == starts.len() {
                total_units
            } else {
                starts[index + 1].checked_add(half_overlap).ok_or_else(|| {
                    candle_audio::candle_core::Error::Msg(
                        "SAME outer ownership end overflow".into(),
                    )
                })?
            };
            if output_start > output_end || output_end > total_units {
                bail!(
                    "SAME outer ownership [{output_start},{output_end}) exceeds total {total_units}"
                )
            }
            let source_start = output_start.checked_sub(chunk_start).ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer ownership precedes its source chunk".into(),
                )
            })?;
            let source_end = output_end.checked_sub(chunk_start).ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer ownership precedes its source chunk".into(),
                )
            })?;
            if source_end > parameters.chunk_size {
                bail!(
                    "SAME outer source [{source_start},{source_end}) exceeds chunk_size {}",
                    parameters.chunk_size
                )
            }
            ownership.push(SameChunkOwnership {
                chunk_index: index,
                chunk_start,
                source_start,
                source_end,
                output_start,
                output_end,
            });
        }
        let covered: usize = ownership
            .iter()
            .map(|slice| slice.output_end - slice.output_start)
            .sum();
        if covered != total_units
            || ownership.first().map(|slice| slice.output_start) != Some(0)
            || ownership.last().map(|slice| slice.output_end) != Some(total_units)
            || ownership
                .windows(2)
                .any(|pair| pair[0].output_end != pair[1].output_start)
        {
            bail!("SAME outer ownership does not cover the output exactly")
        }
        Ok(Self {
            total_units,
            chunk_size: parameters.chunk_size,
            overlap: parameters.overlap,
            starts,
            ownership,
            chunked: true,
        })
    }

    fn direct(total_units: usize, parameters: SameChunkingParameters) -> Self {
        Self {
            total_units,
            chunk_size: parameters.chunk_size,
            overlap: parameters.overlap,
            starts: Vec::new(),
            ownership: Vec::new(),
            chunked: false,
        }
    }
}

/// Per-chunk controlled decoder noise for frozen-reference parity.
///
/// `regularization_noise` is optional only for checkpoints whose SoftNorm has regularization
/// disabled. A controlled decode fails closed instead of falling back to device RNG when the
/// loaded checkpoint requires it.
pub struct SameDecodeChunkNoise {
    pub regularization_noise: Option<Tensor>,
    pub mask_noises: Vec<Tensor>,
}

/// Config-driven transformer layout. SAME-S folds independent full-attention chunks; SAME-L keeps
/// one globally positioned sequence and evaluates a bounded attention band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameAttentionSchedule {
    Full,
    Band {
        latent_left: usize,
        latent_right: usize,
        query_tile: usize,
    },
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

trait SameNoiseSource {
    fn scaled(
        &mut self,
        kind: SameNoiseKind,
        scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor>;

    fn unit(
        &mut self,
        kind: SameNoiseKind,
        scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor>;
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
}

impl SameNoiseSource for SameNoiseRng {
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

struct RequestSameNoise<'a> {
    source: &'a mut SeededNoise,
}

impl SameNoiseSource for RequestSameNoise<'_> {
    fn scaled(
        &mut self,
        _kind: SameNoiseKind,
        scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        self.source
            .standard_normal(shape, dtype, device)
            .map_err(|error| candle_audio::candle_core::Error::Msg(error.to_string()))?
            .affine(scale, 0.0)
    }

    fn unit(
        &mut self,
        _kind: SameNoiseKind,
        _scale: f64,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        self.source
            .standard_normal(shape, dtype, device)
            .map_err(|error| candle_audio::candle_core::Error::Msg(error.to_string()))
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
    /// Folded chunks for [`SameAttentionSchedule::Full`], or the globally packed sequence for
    /// [`SameAttentionSchedule::Band`].
    pub folded_input: Tensor,
    /// Learned tokens after variable-stride expansion and any configured/explicit noise.
    pub expanded_tokens: Tensor,
    /// Output of every transformer block, restored to `[B, sequence, D]` and edge-cropped.
    pub block_outputs: Vec<Tensor>,
    /// Compact absolute-position slices for every band-attention layer. Full layer activations are
    /// deliberately never retained by the SAME-L trace path.
    pub band_layers: Vec<BandLayerTrace>,
    pub layout: SameAttentionSchedule,
    /// The selected learned-token portion of every final `(stride + 1)` subchunk.
    pub selected_segments: Tensor,
    /// Final channel-first block output.
    pub output: Tensor,
}

/// One compact globally positioned activation slice.
#[derive(Clone)]
pub struct BandBoundarySlice {
    pub start: usize,
    pub values: Tensor,
}

/// Bounded evidence for one globally packed SAME-L transformer layer.
#[derive(Clone)]
pub struct BandLayerTrace {
    pub layer: usize,
    pub sequence_len: usize,
    pub slices: Vec<BandBoundarySlice>,
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
    attention_schedule: SameAttentionSchedule,
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
    attention_schedule: SameAttentionSchedule,
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
        if options.stride == 0 {
            bail!("SAME stride must be non-zero")
        }
        match options.attention_schedule {
            SameAttentionSchedule::Full => {
                if options.chunk_size == 0 {
                    bail!("SAME full-attention chunk_size must be non-zero")
                }
                if !options.chunk_size.is_multiple_of(options.stride) {
                    bail!(
                        "SAME stride {} must divide chunk_size {}",
                        options.stride,
                        options.chunk_size
                    )
                }
            }
            SameAttentionSchedule::Band { query_tile, .. } => {
                if query_tile == 0 {
                    bail!("SAME band-attention query tile must be non-zero")
                }
                if options.chunk_midpoint_shift {
                    bail!("SAME band attention cannot use chunk_midpoint_shift")
                }
            }
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
            attention_schedule: options.attention_schedule,
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
                if stride == 0 {
                    bail!("SAME override stride must be non-zero")
                }
                if self.attention_schedule == SameAttentionSchedule::Full
                    && !self.chunk_size.is_multiple_of(stride)
                {
                    bail!(
                        "SAME full-attention override stride {stride} must divide chunk_size {}",
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
        noise_rng: Option<(&mut dyn SameNoiseSource, SameNoiseKind)>,
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
            let modulo = match self.attention_schedule {
                SameAttentionSchedule::Full => self.chunk_size,
                SameAttentionSchedule::Band { .. } => input_segment,
            };
            x = zero_pad_channel_first(&x, modulo)?;
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
        let mapped_sequence = trace.then(|| x.clone());
        if self.direction == Direction::Decode {
            let modulo = match self.attention_schedule {
                SameAttentionSchedule::Full => self.chunk_size / stride,
                SameAttentionSchedule::Band { .. } => input_segment,
            };
            x = zero_pad_sequence(&x, modulo)?;
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
        let expanded_tokens = trace
            .then(|| tokens.reshape((batch, segments * output_segment, dim)))
            .transpose()?;
        x = Tensor::cat(&[&x, &tokens], 1)?
            .reshape((batch, segments, sub_chunk, dim))?
            .reshape((batch, segments * sub_chunk, dim))?;
        drop(tokens);

        let mut block_outputs = Vec::with_capacity(self.transformers.len());
        let mut band_layers = Vec::with_capacity(self.transformers.len());
        let trace_input = match self.attention_schedule {
            SameAttentionSchedule::Full => {
                let effective_chunk = self.chunk_size + self.chunk_size / stride;
                if !x.dim(1)?.is_multiple_of(effective_chunk) {
                    bail!(
                        "SAME folded sequence {} is not divisible by effective chunk {effective_chunk}",
                        x.dim(1)?
                    )
                }
                let folded_input = fold_chunks(&x, effective_chunk)?;
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
                            block_outputs.push(restored.narrow(
                                1,
                                shift,
                                restored.dim(1)? - 2 * shift,
                            )?);
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
                trace.then(|| folded_input.clone())
            }
            SameAttentionSchedule::Band {
                latent_left,
                latent_right,
                query_tile,
            } => {
                let packed_input = trace.then(|| x.clone());
                let left = latent_left.saturating_mul(sub_chunk);
                let right = latent_right.saturating_mul(sub_chunk);
                for (index, layer) in self.transformers.iter().enumerate() {
                    x = layer
                        .block
                        .forward_band(&x, &layer.rope, left, right, query_tile)?;
                    if trace {
                        band_layers.push(BandLayerTrace {
                            layer: index,
                            sequence_len: x.dim(1)?,
                            slices: band_boundary_slices(&x, sub_chunk, left, right, query_tile)?,
                        });
                    }
                }
                packed_input
            }
        };

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
            mapped_sequence: mapped_sequence.expect("mapped sequence exists when tracing"),
            folded_input: trace_input.expect("trace input exists when tracing"),
            expanded_tokens: expanded_tokens.expect("expanded tokens exist when tracing"),
            block_outputs,
            band_layers,
            layout: self.attention_schedule,
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
        if let SameAttentionSchedule::Band {
            latent_left,
            latent_right,
            query_tile,
        } = self.attention_schedule
        {
            let sub_chunk = stride + 1;
            let left = latent_left.saturating_mul(sub_chunk);
            let right = latent_right.saturating_mul(sub_chunk);
            let mut block_outputs = Vec::with_capacity(self.transformers.len());
            let mut final_raw = None;
            for (index, (layer, input)) in self
                .transformers
                .iter()
                .zip(block_inputs.iter())
                .enumerate()
            {
                if input.dim(0)? != batch {
                    bail!(
                        "SAME controlled band block {index} has batch {}, expected {batch}",
                        input.dim(0)?
                    )
                }
                let raw = layer
                    .block
                    .forward_band(input, &layer.rope, left, right, query_tile)?;
                block_outputs.push(raw.clone());
                final_raw = Some(raw);
            }
            let Some(sequence) = final_raw else {
                bail!("SAME controlled trace requires a block")
            };
            return self.controlled_output(sequence, batch, stride, block_outputs);
        }
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
        self.controlled_output(sequence, batch, stride, block_outputs)
    }

    fn controlled_output(
        &self,
        sequence: Tensor,
        batch: usize,
        stride: usize,
        block_outputs: Vec<Tensor>,
    ) -> Result<ControlledResamplingTrace> {
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

fn band_boundary_slices(
    x: &Tensor,
    sub_chunk: usize,
    left: usize,
    right: usize,
    query_tile: usize,
) -> Result<Vec<BandBoundarySlice>> {
    let sequence_len = x.dim(1)?;
    if sequence_len == 0 {
        return Ok(Vec::new());
    }
    let width = sequence_len.min((left + right + 1).max(sub_chunk).max(1));
    let radius = width / 2;
    let mut starts = vec![0, sequence_len.saturating_sub(width)];
    for boundary in [sub_chunk, query_tile, sequence_len / 2] {
        if boundary > 0 && boundary < sequence_len {
            starts.push(boundary.saturating_sub(radius).min(sequence_len - width));
        }
    }
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .map(|start| {
            Ok(BandBoundarySlice {
                start,
                // `narrow` is a view and would otherwise keep the complete layer allocation alive
                // for every trace slice. Force a storage copy so the compact trace is compact in
                // memory as well as in logical shape.
                values: x.narrow(1, start, width)?.force_contiguous()?,
            })
        })
        .collect()
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
    downsampling_ratio: usize,
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
            downsampling_ratio: config.downsampling_ratio,
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

    /// Explicit-noise encoder parity seam with compact layout-aware layer tracing.
    pub fn encode_with_noise_and_trace(
        &self,
        audio: &Tensor,
        override_strides: Option<&[usize]>,
        mask_noises: &[Tensor],
    ) -> Result<(Tensor, SameTrace)> {
        let (output, stages) =
            self.encode_internal(audio, override_strides, Some(mask_noises), None, true)?;
        Ok((output.clone(), SameTrace { stages, output }))
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
        mut rng: Option<&mut dyn SameNoiseSource>,
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
            let kind = SameNoiseKind::EncoderTokens { stage: index };
            let (next, block_trace) = match rng.as_mut() {
                Some(source) => block.forward(
                    &x,
                    override_strides.map(|s| s[index]),
                    mask_noises.map(|noises| &noises[index]),
                    Some((&mut **source, kind)),
                    trace,
                )?,
                None => block.forward(
                    &x,
                    override_strides.map(|s| s[index]),
                    mask_noises.map(|noises| &noises[index]),
                    None,
                    trace,
                )?,
            };
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
        self.decode_with_noise_source(latents, override_strides, rng)
    }

    fn decode_with_noise_source(
        &self,
        latents: &Tensor,
        override_strides: Option<&[usize]>,
        rng: &mut dyn SameNoiseSource,
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
        mut rng: Option<&mut dyn SameNoiseSource>,
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
            let kind = SameNoiseKind::DecoderTokens { stage: index };
            let (next, block_trace) = match rng.as_mut() {
                Some(source) => block.forward(
                    &x,
                    override_strides.map(|s| s[index]),
                    mask_noises.map(|n| &n[index]),
                    Some((&mut **source, kind)),
                    trace,
                )?,
                None => block.forward(
                    &x,
                    override_strides.map(|s| s[index]),
                    mask_noises.map(|n| &n[index]),
                    None,
                    trace,
                )?,
            };
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

    /// Upstream-compatible outer encode entry point.
    ///
    /// `chunk_size` and `overlap` are latent units. Chunk recursion deliberately does not forward
    /// stride overrides or other inner kwargs, matching frozen upstream. A single request-local RNG
    /// is threaded through chunks in start order so stochastic draw order remains observable.
    pub fn encode_audio(
        &self,
        audio: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
    ) -> Result<Tensor> {
        let mut rng = SameNoiseRng::from_entropy();
        self.encode_audio_with_rng(audio, policy, parameters, &mut rng)
    }

    pub fn encode_audio_with_rng(
        &self,
        audio: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        rng: &mut SameNoiseRng,
    ) -> Result<Tensor> {
        let plan = self.encode_chunk_plan(audio, policy, parameters)?;
        if !plan.chunked {
            return self.encode_with_rng(audio, None, rng);
        }
        let chunk_samples = plan
            .chunk_size
            .checked_mul(self.downsampling_ratio)
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer encode chunk size overflow".into(),
                )
            })?;
        let mut pieces = Vec::with_capacity(plan.ownership.len());
        for ownership in &plan.ownership {
            let sample_start = ownership
                .chunk_start
                .checked_mul(self.downsampling_ratio)
                .ok_or_else(|| {
                    candle_audio::candle_core::Error::Msg("SAME outer encode start overflow".into())
                })?;
            let chunk = audio.narrow(2, sample_start, chunk_samples)?;
            let encoded = self.encode_with_rng(&chunk, None, rng)?;
            pieces.push(owned_piece(&encoded, ownership)?);
        }
        concatenate_owned(pieces, plan.total_units, "encode")
    }

    /// Controlled-noise outer encode seam. One stage-noise vector is required per executed chunk;
    /// the direct/bypass path is represented by one vector.
    pub fn encode_audio_with_chunk_noises(
        &self,
        audio: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        chunk_noises: &[Vec<Tensor>],
    ) -> Result<Tensor> {
        let plan = self.encode_chunk_plan(audio, policy, parameters)?;
        if !plan.chunked {
            if chunk_noises.len() != 1 {
                bail!(
                    "SAME outer direct encode needs one chunk-noise entry, got {}",
                    chunk_noises.len()
                )
            }
            return self.encode_with_noise(audio, None, &chunk_noises[0]);
        }
        validate_outer_noise_count("encode", chunk_noises.len(), plan.starts.len())?;
        let chunk_samples = plan
            .chunk_size
            .checked_mul(self.downsampling_ratio)
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer encode chunk size overflow".into(),
                )
            })?;
        let mut pieces = Vec::with_capacity(plan.ownership.len());
        for ownership in &plan.ownership {
            let sample_start = ownership
                .chunk_start
                .checked_mul(self.downsampling_ratio)
                .ok_or_else(|| {
                    candle_audio::candle_core::Error::Msg("SAME outer encode start overflow".into())
                })?;
            let chunk = audio.narrow(2, sample_start, chunk_samples)?;
            let encoded =
                self.encode_with_noise(&chunk, None, &chunk_noises[ownership.chunk_index])?;
            pieces.push(owned_piece(&encoded, ownership)?);
        }
        concatenate_owned(pieces, plan.total_units, "encode")
    }

    /// Upstream-compatible outer decode entry point.
    pub fn decode_audio(
        &self,
        latents: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
    ) -> Result<Tensor> {
        let mut rng = SameNoiseRng::from_entropy();
        self.decode_audio_with_rng(latents, policy, parameters, &mut rng)
    }

    pub fn decode_audio_with_rng(
        &self,
        latents: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        rng: &mut SameNoiseRng,
    ) -> Result<Tensor> {
        self.decode_audio_with_source(latents, policy, parameters, rng)
    }

    /// Decode with the same request-local stream used for initial latents and Pingpong.
    ///
    /// Cancellation is checked before every direct or chunked SAME dispatch.
    pub fn decode_audio_with_request_rng(
        &self,
        latents: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        rng: &mut SeededNoise,
        is_canceled: &dyn Fn() -> bool,
    ) -> candle_audio::Result<Tensor> {
        let mut adapter = RequestSameNoise { source: rng };
        let total_latents = latents.dim(2)?;
        let plan = SameChunkPlan::build(total_latents, policy.enabled(), parameters)?;
        if !plan.chunked {
            if is_canceled() {
                return Err(candle_audio::AudioError::Canceled);
            }
            return Ok(self.decode_with_noise_source(latents, None, &mut adapter)?);
        }
        let mut pieces = Vec::with_capacity(plan.ownership.len());
        for ownership in &plan.ownership {
            if is_canceled() {
                return Err(candle_audio::AudioError::Canceled);
            }
            let chunk = latents.narrow(2, ownership.chunk_start, plan.chunk_size)?;
            let decoded = self.decode_with_noise_source(&chunk, None, &mut adapter)?;
            let sample_ownership = scale_ownership(ownership, self.downsampling_ratio)?;
            pieces.push(owned_piece(&decoded, &sample_ownership)?);
        }
        let total_samples = total_latents
            .checked_mul(self.downsampling_ratio)
            .ok_or_else(|| {
                candle_audio::AudioError::Msg("SAME outer decode output size overflow".into())
            })?;
        Ok(concatenate_owned(pieces, total_samples, "decode")?)
    }

    fn decode_audio_with_source(
        &self,
        latents: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        rng: &mut dyn SameNoiseSource,
    ) -> Result<Tensor> {
        let total_latents = latents.dim(2)?;
        let plan = SameChunkPlan::build(total_latents, policy.enabled(), parameters)?;
        if !plan.chunked {
            return self.decode_with_noise_source(latents, None, rng);
        }
        let mut pieces = Vec::with_capacity(plan.ownership.len());
        for ownership in &plan.ownership {
            let chunk = latents.narrow(2, ownership.chunk_start, plan.chunk_size)?;
            let decoded = self.decode_with_noise_source(&chunk, None, rng)?;
            let sample_ownership = scale_ownership(ownership, self.downsampling_ratio)?;
            pieces.push(owned_piece(&decoded, &sample_ownership)?);
        }
        let total_samples = total_latents
            .checked_mul(self.downsampling_ratio)
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer decode output size overflow".into(),
                )
            })?;
        concatenate_owned(pieces, total_samples, "decode")
    }

    /// Controlled-noise outer decode seam. One entry is required per executed chunk; the
    /// direct/bypass path is represented by one entry.
    pub fn decode_audio_with_chunk_noises(
        &self,
        latents: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
        chunk_noises: &[SameDecodeChunkNoise],
    ) -> Result<Tensor> {
        let total_latents = latents.dim(2)?;
        let plan = SameChunkPlan::build(total_latents, policy.enabled(), parameters)?;
        if !plan.chunked {
            if chunk_noises.len() != 1 {
                bail!(
                    "SAME outer direct decode needs one chunk-noise entry, got {}",
                    chunk_noises.len()
                )
            }
            let noise = &chunk_noises[0];
            self.validate_controlled_decode_noise(noise)?;
            return self.decode_with_noise(
                latents,
                None,
                noise.regularization_noise.as_ref(),
                Some(&noise.mask_noises),
            );
        }
        validate_outer_noise_count("decode", chunk_noises.len(), plan.starts.len())?;
        let mut pieces = Vec::with_capacity(plan.ownership.len());
        for ownership in &plan.ownership {
            let chunk = latents.narrow(2, ownership.chunk_start, plan.chunk_size)?;
            let noise = &chunk_noises[ownership.chunk_index];
            self.validate_controlled_decode_noise(noise)?;
            let decoded = self.decode_with_noise(
                &chunk,
                None,
                noise.regularization_noise.as_ref(),
                Some(&noise.mask_noises),
            )?;
            let sample_ownership = scale_ownership(ownership, self.downsampling_ratio)?;
            pieces.push(owned_piece(&decoded, &sample_ownership)?);
        }
        let total_samples = total_latents
            .checked_mul(self.downsampling_ratio)
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "SAME outer decode output size overflow".into(),
                )
            })?;
        concatenate_owned(pieces, total_samples, "decode")
    }

    fn validate_controlled_decode_noise(&self, noise: &SameDecodeChunkNoise) -> Result<()> {
        if self.bottleneck.noise_regularize() && noise.regularization_noise.is_none() {
            bail!("SAME controlled decode requires explicit SoftNorm regularization noise")
        }
        Ok(())
    }

    fn encode_chunk_plan(
        &self,
        audio: &Tensor,
        policy: SameChunkingPolicy,
        parameters: SameChunkingParameters,
    ) -> Result<SameChunkPlan> {
        let samples = audio.dim(2)?;
        let latent_units = samples / self.downsampling_ratio;
        let plan = SameChunkPlan::build(latent_units, policy.enabled(), parameters)?;
        if plan.chunked && !samples.is_multiple_of(self.downsampling_ratio) {
            bail!(
                "SAME outer chunked encode needs audio aligned to downsampling ratio {}, got {samples}",
                self.downsampling_ratio
            )
        }
        Ok(plan)
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

fn owned_piece(chunk: &Tensor, ownership: &SameChunkOwnership) -> Result<Tensor> {
    let available = chunk.dim(2)?;
    if ownership.source_end > available {
        bail!(
            "SAME outer chunk {} owns source [{},{}) but produced only {available} units",
            ownership.chunk_index,
            ownership.source_start,
            ownership.source_end
        )
    }
    chunk
        .narrow(
            2,
            ownership.source_start,
            ownership.source_end - ownership.source_start,
        )?
        .contiguous()
}

fn concatenate_owned(pieces: Vec<Tensor>, expected: usize, direction: &str) -> Result<Tensor> {
    let refs: Vec<&Tensor> = pieces.iter().collect();
    let output = Tensor::cat(&refs, 2)?;
    if output.dim(2)? != expected {
        bail!(
            "SAME outer {direction} stitched {} units, expected {expected}",
            output.dim(2)?
        )
    }
    Ok(output)
}

fn scale_ownership(ownership: &SameChunkOwnership, scale: usize) -> Result<SameChunkOwnership> {
    let multiply = |value: usize| {
        value.checked_mul(scale).ok_or_else(|| {
            candle_audio::candle_core::Error::Msg("SAME outer ownership scale overflow".into())
        })
    };
    Ok(SameChunkOwnership {
        chunk_index: ownership.chunk_index,
        chunk_start: multiply(ownership.chunk_start)?,
        source_start: multiply(ownership.source_start)?,
        source_end: multiply(ownership.source_end)?,
        output_start: multiply(ownership.output_start)?,
        output_end: multiply(ownership.output_end)?,
    })
}

fn validate_outer_noise_count(direction: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!("SAME outer chunked {direction} needs {expected} chunk-noise entries, got {actual}")
    }
    Ok(())
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
    validate_attention_schedule(
        "encoder",
        &encoder.sliding_window,
        encoder.chunk_midpoint_shift,
    )?;
    validate_attention_schedule(
        "decoder",
        &decoder.sliding_window,
        decoder.chunk_midpoint_shift,
    )?;
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

fn validate_attention_schedule(
    direction: &str,
    sliding_window: &Option<Vec<usize>>,
    chunk_midpoint_shift: bool,
) -> Result<()> {
    if let Some(window) = sliding_window {
        if window.len() != 2 {
            bail!(
                "SAME {direction} sliding_window must contain exactly [left,right], got {} values",
                window.len()
            )
        }
        if chunk_midpoint_shift {
            bail!("SAME {direction} sliding_window cannot use chunk_midpoint_shift")
        }
    }
    Ok(())
}

fn attention_schedule(window: &Option<Vec<usize>>) -> SameAttentionSchedule {
    match window {
        Some(window) => SameAttentionSchedule::Band {
            latent_left: window[0],
            latent_right: window[1],
            query_tile: SAME_BAND_QUERY_TILE,
        },
        None => SameAttentionSchedule::Full,
    }
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
                attention_schedule: attention_schedule(&config.sliding_window),
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
                attention_schedule: attention_schedule(&config.sliding_window),
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

    fn parameters(chunk_size: usize, overlap: usize) -> SameChunkingParameters {
        SameChunkingParameters {
            chunk_size,
            overlap,
        }
    }

    #[test]
    fn outer_chunk_policy_resolves_model_default_and_request_override() {
        assert!(SameChunkingPolicy::full_model_encode(true).enabled());
        assert!(!SameChunkingPolicy::full_model_encode(false).enabled());
        assert!(SameChunkingPolicy::full_model_decode(true, None).enabled());
        assert!(!SameChunkingPolicy::full_model_decode(false, None).enabled());
        assert!(!SameChunkingPolicy::full_model_decode(true, Some(false)).enabled());
        assert!(SameChunkingPolicy::full_model_decode(false, Some(true)).enabled());
        assert!(!SameChunkingPolicy::standalone(false).enabled());
        assert!(SameChunkingPolicy::standalone(true).enabled());
    }

    #[test]
    fn outer_chunk_plan_matches_frozen_boundary_and_final_anchor_cases() {
        let default = SameChunkingParameters::default();
        for total in [0, 1, 127] {
            let plan = SameChunkPlan::build(total, true, default).unwrap();
            assert!(!plan.chunked, "{total}");
            assert!(plan.starts.is_empty(), "{total}");
        }

        let cases = [
            (128, vec![0]),
            (129, vec![0, 1]),
            (224, vec![0, 96]),
            (225, vec![0, 96, 97]),
            (256, vec![0, 96, 128]),
            (257, vec![0, 96, 129]),
            (
                1292,
                vec![
                    0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1164,
                ],
            ),
            (
                1300,
                vec![
                    0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1172,
                ],
            ),
            (
                1358,
                vec![
                    0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1230,
                ],
            ),
        ];
        for (total, expected) in cases {
            let plan = SameChunkPlan::build(total, true, default).unwrap();
            assert!(plan.chunked, "{total}");
            assert_eq!(plan.starts, expected, "{total}");
            assert_eq!(plan.ownership.first().unwrap().output_start, 0, "{total}");
            assert_eq!(plan.ownership.last().unwrap().output_end, total, "{total}");
            for pair in plan.ownership.windows(2) {
                assert_eq!(pair[0].output_end, pair[1].output_start, "{total}");
            }
        }

        let plan = SameChunkPlan::build(225, true, default).unwrap();
        assert_eq!(
            plan.ownership
                .iter()
                .map(|slice| (slice.output_start, slice.output_end))
                .collect::<Vec<_>>(),
            vec![(0, 112), (112, 113), (113, 225)]
        );
        let max = SameChunkPlan::build(4096, true, default).unwrap();
        assert_eq!(max.starts.len(), 43);
        assert_eq!(max.starts.last(), Some(&3968));
        assert_eq!(max.starts.len() * max.chunk_size, 5504);
    }

    #[test]
    fn outer_chunk_plan_preserves_floor_half_and_later_writer_ownership() {
        for (chunk_size, overlap, total) in
            [(7, 0, 19), (7, 1, 19), (7, 2, 19), (9, 3, 23), (9, 8, 23)]
        {
            let plan = SameChunkPlan::build(total, true, parameters(chunk_size, overlap)).unwrap();
            let mut reconstructed = Vec::new();
            for ownership in &plan.ownership {
                for local in ownership.source_start..ownership.source_end {
                    reconstructed.push((ownership.chunk_index, local));
                }
            }

            let mut sequential = vec![(usize::MAX, usize::MAX); total];
            let half = overlap / 2;
            for (index, &start) in plan.starts.iter().enumerate() {
                let left = if index == 0 { 0 } else { half };
                let right = if index + 1 == plan.starts.len() {
                    chunk_size
                } else {
                    chunk_size - half
                };
                for local in left..right {
                    sequential[start + local] = (index, local);
                }
            }
            assert_eq!(reconstructed, sequential, "{chunk_size}/{overlap}/{total}");
            assert!(
                sequential.iter().all(|owner| owner.0 != usize::MAX),
                "{chunk_size}/{overlap}/{total}"
            );
        }
    }

    #[test]
    fn outer_tensor_stitch_preserves_batch_channels_and_rejects_mutations() {
        let device = Device::Cpu;
        let parameters = parameters(7, 3);
        let plan = SameChunkPlan::build(19, true, parameters).unwrap();
        let chunks = plan
            .starts
            .iter()
            .enumerate()
            .map(|(chunk_index, _)| {
                Tensor::from_vec(
                    (0..2)
                        .flat_map(|batch| {
                            (0..2).flat_map(move |channel| {
                                (0..parameters.chunk_size).map(move |local| {
                                    (chunk_index * 1_000 + batch * 100 + channel * 10 + local)
                                        as f32
                                })
                            })
                        })
                        .collect::<Vec<_>>(),
                    (2, 2, parameters.chunk_size),
                    &device,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pieces = plan
            .ownership
            .iter()
            .map(|ownership| owned_piece(&chunks[ownership.chunk_index], ownership).unwrap())
            .collect();
        let stitched = concatenate_owned(pieces, plan.total_units, "test")
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        let mut sequential = vec![vec![vec![-1f32; plan.total_units]; 2]; 2];
        let half = parameters.overlap / 2;
        for (chunk_index, &start) in plan.starts.iter().enumerate() {
            let left = if chunk_index == 0 { 0 } else { half };
            let right = if chunk_index + 1 == plan.starts.len() {
                parameters.chunk_size
            } else {
                parameters.chunk_size - half
            };
            for (batch, channels) in sequential.iter_mut().enumerate() {
                for (channel, output) in channels.iter_mut().enumerate() {
                    for local in left..right {
                        output[start + local] =
                            (chunk_index * 1_000 + batch * 100 + channel * 10 + local) as f32;
                    }
                }
            }
        }
        assert_eq!(stitched, sequential);

        let earlier_writer_at_final_anchor = chunks[2].to_vec3::<f32>().unwrap()[0][0][5];
        assert_ne!(stitched[0][0][13], earlier_writer_at_final_anchor);
        assert_eq!(
            stitched[0][0][13],
            chunks[3].to_vec3::<f32>().unwrap()[0][0][1]
        );

        let cropped = SameAutoencoder::crop_valid_prefix(
            &Tensor::from_vec(
                (0..40).map(|value| value as f32).collect(),
                (2, 2, 10),
                &device,
            )
            .unwrap(),
            7,
        )
        .unwrap();
        assert_eq!(cropped.dims3().unwrap(), (2, 2, 7));
        assert!(SameAutoencoder::crop_valid_prefix(&cropped, 8).is_err());
    }

    #[test]
    fn outer_chunk_validation_fails_closed_only_when_requested() {
        assert!(SameChunkPlan::build(1, true, parameters(0, 0)).is_err());
        assert!(SameChunkPlan::build(1, true, parameters(8, 8)).is_err());
        assert!(SameChunkPlan::build(1, true, parameters(8, 9)).is_err());
        assert!(SameChunkPlan::build(1, false, parameters(0, usize::MAX)).is_ok());
        let ownership = SameChunkOwnership {
            chunk_index: 0,
            chunk_start: usize::MAX,
            source_start: 0,
            source_end: 1,
            output_start: 0,
            output_end: 1,
        };
        assert!(scale_ownership(&ownership, 2).is_err());
    }
}
