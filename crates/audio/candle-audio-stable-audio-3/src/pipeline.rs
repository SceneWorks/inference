//! Connected Stable Audio 3 inference pipeline, shared by every registered variant.

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};

use crate::config::{ConditionerConfig, DiffusionObjective, ModelConfig};
use crate::dit::{Guidance, StableAudio3Dit};
use crate::model::VariantShape;
use crate::same::{
    SameAutoencoder, SameChunkingParameters, SameChunkingPolicy, SameDecodeChunkNoise,
};
use crate::sampler::{
    build_schedule, default_sample_geometry, inference_shift, padding_mask,
    sample_dit_with_interval_and_cancel, GuidanceInterval, InjectedNoise, NoiseSource,
    ProgressCallback, SampleGeometry, SamplerKind, SeededNoise,
};
use crate::t5gemma::T5GemmaConditioner;
use crate::weights::{SnapshotKind, SnapshotLayout};

pub const SAMPLE_RATE: u32 = 44_100;
pub const CHANNELS: usize = 2;
pub const LATENT_CHANNELS: usize = 256;
pub const DEFAULT_STEPS: usize = 8;
pub const DEFAULT_GUIDANCE: f64 = 1.0;

/// What the strict wrapper authenticates a snapshot against: one variant's architecture bound to
/// the repository that published it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantGeometry {
    pub shape: VariantShape,
    pub expected_repo: &'static str,
}

/// Compute dtypes for one loaded graph.
///
/// # Why this is F32 everywhere, and what was measured to decide that
///
/// Upstream runs `model_half=True` when a CUDA device is selected and forces fp32 otherwise
/// (`stable_audio_3/model.py`). sc-14545 built the machinery to mirror that — the graph now runs
/// end to end at F16, which it could not before — and then measured it, because "mirror upstream"
/// is not evidence that the output survives.
///
/// Measured on Metal at 30 s / 8 steps on `stable_audio_3_medium`, three seeds at each dtype
/// (`tests/dtype_policy.rs`):
///
/// | statistic | F32 range | F16 range |
/// |---|---|---|
/// | rms | 0.057639 … 0.067437 | 0.069209 … 0.075448 |
/// | peak | 0.519312 … 0.619903 | 0.525879 … 0.603027 |
/// | hf emphasis | 0.121821 … 0.150911 | 0.095454 … 0.123614 |
/// | side ratio | 0.502560 … 0.650129 | 0.373487 … 0.951624 |
///
/// F16 is louder on all three seeds, duller on all three, and its stereo image spreads past twice
/// the fp32 envelope. Those are the three signatures of a decoder losing precision. They are *not*
/// conclusive, because F16 does not perturb the sample — at a fixed seed the F16 and F32 waveforms
/// sit at cosine `0.222`, against `0.005` for two F32 renders at adjacent seeds — so half precision
/// selects a different draw and a different draw legitimately has a different brightness. Three
/// seeds cannot separate those two explanations.
///
/// What is not ambiguous is the decision that follows. The only backend fp16 would apply to is
/// CUDA, no CUDA hardware was available to measure the policy on the backend it would ship to, and
/// the one measurement that could be taken points at degradation rather than away from it. Adopting
/// it would also re-open the two already-merged small providers on that backend. So the shipped
/// policy is F32 on every backend, the seam stays typed and tested so the fp16 path cannot rot, and
/// the split policy worth trying next — half the 1.45B DiT, keep the 852M SAME autoencoder at F32,
/// which would isolate whether the dullness comes from the decoder — is filed as its own story with
/// these numbers attached.
///
/// The text side was never part of that decision. sc-14537 pinned the canonical cross-runtime text
/// policy — BF16 weights on disk, F32 compute, one BF16 rounding at the raw-embedding boundary —
/// and `tests/text_oracle.rs` gates it against the frozen Transformers 5.8.0 oracle on CPU and
/// Metal. Switching T5Gemma to BF16 *compute* would move a surface with a numeric parity gate behind
/// it for no memory win worth having: the encoder is 281 MB of medium's 10.4 GB resident set.
///
/// # The F16 path did not load on CUDA at all until sc-14545's second fix cycle
///
/// The first real-weight CUDA run of this seam failed at `load SA3 SAME` with
/// `DriverError(CUDA_ERROR_INVALID_VALUE)` while F32 loaded cleanly. The cause was not the dtype
/// threading: SAME's `bottleneck.noise_scaling_factor` is persisted as an empty `[1, 0, 1]` buffer,
/// and casting a zero-element tensor to F16 launches a kernel with a zero grid dimension, which the
/// driver rejects. The same zero-element buffer broke the Metal embedded-SAME-L lane by a different
/// mechanism. Both are fixed in [`crate::weights`], where the reasoning is recorded in full. Worth
/// stating here because it bounds what "the graph runs end to end at F16" meant before that fix: it
/// was demonstrated on Metal, and on CUDA it was not demonstrated at all.
///
/// # What is pinned F32 independently of `root`, and what is not
///
/// Three things do not follow `root`:
///
/// * The **DiT's block normalizations**. `config.rs` rejects any Stable Audio 3 DiT config whose
///   `norm_kwargs.force_fp32` is not `true`, and [`crate::transformer`]'s `Norm::forward` upcasts
///   to F32 when that flag is set, so the DiT's RMS statistics are F32 at any `root`.
/// * **RoPE**. `inv_freq` is requested at `DType::F32` explicitly rather than inheriting the
///   `VarBuilder` dtype, and the position outer product is evaluated against it in F32.
/// * **The sigma schedule**. `sampler.rs` materializes it in F32 and keeps it F32 for every value
///   handed to the model; only the solver arithmetic runs at the latents' dtype.
///
/// **`same.rs` is not on that list, and that is load-bearing.** The SAME autoencoder builds its
/// blocks with `NormConfig { eps: 1e-3, ..Default::default() }`, whose `force_fp32` is `false`, and
/// the QK norms inside its attention are constructed with `force_fp32: false` as well. Medium's
/// SAME-L config sets `dyt: true`, and `Norm::forward` returns through the `DynamicTanh` branch
/// *before* the `force_fp32` upcast is reached, so DyT would not honour the flag even if it were
/// set. At `root = F16` the whole autoencoder — normalization statistics included — runs in half
/// precision.
///
/// That is the concrete mechanism behind the split policy filed as the follow-up (sc-15151). "Half
/// the DiT, keep the SAME autoencoder at F32" is not merely a memory trade: it is the only
/// arrangement in which SAME's statistics stay F32. Adopting F16 wholesale would mean either
/// accepting half-precision DyT statistics through an 852M-parameter decoder or threading a second
/// dtype into `same.rs`, and neither was measured here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeDTypes {
    /// DiT + SAME + learned conditioner.
    pub root: DType,
    /// T5Gemma compute dtype.
    pub text: DType,
}

impl ComputeDTypes {
    /// Resolve the compute policy for a selected device.
    ///
    /// Keyed on the *selected* device rather than on host CUDA availability. Upstream keys its own
    /// half-cast on whether a CUDA device exists anywhere on the host, which silently half-casts a
    /// CPU run on a CUDA box; that bug is deliberately not reproduced, so this stays a `match` on
    /// the device even while every arm currently agrees.
    pub fn for_device(device: &Device) -> Self {
        Self {
            root: match device {
                Device::Cpu | Device::Metal(_) | Device::Cuda(_) => DType::F32,
            },
            text: DType::F32,
        }
    }
}

/// Runtime choices mapped from the backend-neutral generation request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthesisParameters {
    pub duration_secs: f32,
    pub steps: usize,
    pub sampler: SamplerKind,
    pub guidance: Guidance,
    pub seed: u64,
}

/// A loaded, connected text-to-stereo-audio graph.
pub struct StableAudio3Pipeline {
    config: crate::config::StableAudioConfig,
    conditioner: T5GemmaConditioner,
    dit: StableAudio3Dit,
    same: SameAutoencoder,
    device: Device,
    dtypes: ComputeDTypes,
}

impl StableAudio3Pipeline {
    /// Build the connected graph, refusing any snapshot that is not `geometry`'s exact checkpoint.
    pub fn from_layout(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
    ) -> Result<Self> {
        Self::from_layout_with_dtypes(layout, geometry, device, ComputeDTypes::for_device(device))
    }

    /// Build the connected graph with an explicit compute policy.
    ///
    /// Production loads go through [`Self::from_layout`], which derives the policy from the selected
    /// device. This form exists so a backend-dtype oracle can hold the device fixed and vary the
    /// dtype, which is the only way to measure an F16-vs-F32 bound rather than assert one.
    pub fn from_layout_with_dtypes(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
        dtypes: ComputeDTypes,
    ) -> Result<Self> {
        validate_layout(layout, geometry)?;
        let diffusion = match &layout.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let builders = layout.full_pipeline_builders(dtypes.root, dtypes.text, device)?;
        let conditioner =
            T5GemmaConditioner::from_layout_with_builders(layout, device, builders.clone())?;
        let dit = StableAudio3Dit::load(diffusion, builders.clone())
            .map_err(|error| AudioError::Msg(format!("load SA3 DiT: {error}")))?;
        let same = SameAutoencoder::load(&diffusion.pretransform.config, builders)
            .map_err(|error| AudioError::Msg(format!("load SA3 SAME: {error}")))?;
        Ok(Self {
            config: layout.config.clone(),
            conditioner,
            dit,
            same,
            device: device.clone(),
            dtypes,
        })
    }

    /// The compute policy this graph was loaded with.
    pub fn dtypes(&self) -> ComputeDTypes {
        self.dtypes
    }

    pub fn synthesize(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        canceled(is_canceled)?;
        let geometry = self.geometry(parameters.duration_secs)?;
        let template = Tensor::zeros(
            (1, LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        let mut noise = SeededNoise::new(parameters.seed);
        let initial = noise.standard_normal_like(&template)?;
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            on_progress,
            &mut noise,
            is_canceled,
        )?;
        canceled(is_canceled)?;
        on_decoding();

        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        // The full small-variant config owns the default-on outer chunking decision. The same
        // request-local stream has already produced initial latents and every Pingpong draw.
        let decoded = self.same.decode_audio_with_request_rng(
            &sampled,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            &mut noise,
            is_canceled,
        )?;
        self.finish(decoded, &padding, parameters.duration_secs, is_canceled)
    }

    /// Replay the complete frozen-upstream path with explicit stochastic inputs.
    ///
    /// This exists for numerical-oracle tests. Production requests always use one request-local
    /// [`SeededNoise`] stream instead.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_controlled(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        initial: &Tensor,
        pingpong_noises: Vec<Tensor>,
        decode_noises: &[SameDecodeChunkNoise],
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Tensor, Vec<f32>)> {
        canceled(is_canceled)?;
        let geometry = self.geometry(parameters.duration_secs)?;
        if initial.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "controlled initial noise has shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                initial.dims(),
                geometry.latent_length
            )));
        }
        let initial = initial
            .to_device(&self.device)?
            .to_dtype(self.dtypes.root)?;
        let mut noise = InjectedNoise::new(pingpong_noises);
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            on_progress,
            &mut noise,
            is_canceled,
        )?;
        if noise.draws() != parameters.steps {
            return Err(AudioError::Msg(format!(
                "controlled Pingpong replay consumed {} draws, expected {}",
                noise.draws(),
                parameters.steps
            )));
        }
        canceled(is_canceled)?;
        on_decoding();
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let decoded = self.same.decode_audio_with_chunk_noises(
            &sampled,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            decode_noises,
        )?;
        let audio = self.finish(decoded, &padding, parameters.duration_secs, is_canceled)?;
        Ok((sampled, audio))
    }

    /// Replay only the decode boundary with explicit latents and SAME noise.
    #[doc(hidden)]
    pub fn decode_controlled(
        &self,
        latents: &Tensor,
        duration_secs: f32,
        decode_noises: &[SameDecodeChunkNoise],
    ) -> Result<Vec<f32>> {
        let geometry = self.geometry(duration_secs)?;
        if latents.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "controlled latents have shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                latents.dims(),
                geometry.latent_length
            )));
        }
        let padding = padding_mask(
            &geometry.valid_lengths,
            geometry.latent_length,
            &self.device,
        )?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let decoded = self.same.decode_audio_with_chunk_noises(
            &latents
                .to_device(&self.device)?
                .to_dtype(self.dtypes.root)?,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            decode_noises,
        )?;
        self.finish(decoded, &padding, duration_secs, &|| false)
    }

    /// Replay one unchunked SAME dispatch for oracle boundary diagnosis.
    #[doc(hidden)]
    pub fn decode_chunk_controlled(
        &self,
        latents: &Tensor,
        noise: &SameDecodeChunkNoise,
    ) -> Result<Tensor> {
        Ok(self.same.decode_with_noise(
            &latents
                .to_device(&self.device)?
                .to_dtype(self.dtypes.root)?,
            None,
            noise.regularization_noise.as_ref(),
            Some(&noise.mask_noises),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample<N: NoiseSource>(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        geometry: &SampleGeometry,
        initial: &Tensor,
        on_progress: &mut dyn FnMut(usize, usize),
        noise: &mut N,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Tensor, Tensor)> {
        let positive = self
            .conditioner
            .encode_with_cancel(&[prompt.to_owned()], is_canceled)?;
        canceled(is_canceled)?;
        let negative = negative_prompt
            .map(|prompt| {
                self.conditioner
                    .encode_with_cancel(&[prompt.to_owned()], is_canceled)
            })
            .transpose()?;
        canceled(is_canceled)?;

        let seconds = Tensor::new(&[parameters.duration_secs], &self.device)?;
        let local = Tensor::zeros(
            (1, 1 + LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        let padding = padding_mask(
            &geometry.valid_lengths,
            geometry.latent_length,
            &self.device,
        )?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let schedule = build_schedule(
            parameters.steps,
            1.0,
            &inference_shift(&diffusion.diffusion),
            geometry.effective_lengths.as_deref(),
            geometry.latent_length,
        )?;
        let total = parameters.steps;
        let mut progress = |step: &crate::sampler::SampleStep| -> Result<()> {
            canceled(is_canceled)?;
            on_progress(step.index + 1, total);
            Ok(())
        };
        let sampled = sample_dit_with_interval_and_cancel(
            &self.dit,
            parameters.sampler,
            initial,
            &schedule,
            &positive.embeddings,
            negative.as_ref().map(|value| &value.embeddings),
            negative.as_ref().map(|value| &value.attention_mask),
            &seconds,
            &local,
            Some(&padding),
            parameters.guidance,
            GuidanceInterval::FULL,
            Some(&mut progress as &mut ProgressCallback<'_>),
            noise,
            false,
            is_canceled,
        )?;
        Ok((sampled.latents, padding))
    }

    fn geometry(&self, duration_secs: f32) -> Result<SampleGeometry> {
        default_sample_geometry(&self.config, &[Some(duration_secs as f64)])
    }

    fn finish(
        &self,
        decoded: Tensor,
        padding: &Tensor,
        duration_secs: f32,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        canceled(is_canceled)?;
        let decoded = apply_padding_mask(&decoded, padding, 4_096)?;
        let frames = ((duration_secs as f64) * SAMPLE_RATE as f64) as usize;
        let decoded = SameAutoencoder::crop_valid_prefix(&decoded, frames)?.clamp(-1.0, 1.0)?;
        planar_to_interleaved(&decoded)
    }
}

/// The conditioner `repo_id` declared by a full Stable Audio 3 snapshot's `model_config.json`.
///
/// This is the only field in the shipped configs that identifies which post-trained checkpoint the
/// snapshot is: `small-music` and `small-sfx` are otherwise architecturally identical, differing
/// only in training-only ARC fields and demo prompts.
pub fn conditioner_repo_id(layout: &SnapshotLayout) -> Option<&str> {
    let diffusion = match &layout.config.model {
        ModelConfig::Diffusion(model) => model,
        ModelConfig::Autoencoder(_) => return None,
    };
    diffusion
        .conditioning
        .configs
        .iter()
        .find_map(|config| match config {
            ConditionerConfig::T5gemma { config, .. } => config.repo_id.as_deref(),
            ConditionerConfig::Number { .. } => None,
        })
}

/// Validate that `layout` is the exact post-trained SA3 checkpoint `geometry` describes.
///
/// The architecture record is per registered variant, so a small snapshot cannot authenticate as
/// `stable_audio_3_medium` on shape alone and vice versa; the `repo_id` check is what separates the
/// two architecturally identical smalls from each other. `ModelRegistration::load` carries no
/// provider id, so without this the registry would happily serve one checkpoint's weights from
/// another's registration.
pub fn validate_layout(layout: &SnapshotLayout, geometry: VariantGeometry) -> Result<()> {
    let expected_repo = geometry.expected_repo;
    let shape = geometry.shape;
    if layout.kind != SnapshotKind::Full
        || layout.config.sample_rate != SAMPLE_RATE
        || layout.config.sample_size != shape.sample_size
        || layout.config.audio_channels != CHANNELS
        || layout.keys.total != shape.total_keys
        || layout.keys.dit != shape.dit_keys
        || layout.keys.encoder + layout.keys.decoder + layout.keys.bottleneck
            != shape.autoencoder_keys
        || layout.keys.conditioner != 3
    {
        return Err(AudioError::Msg(format!(
            "snapshot is not the exact {expected_repo} full checkpoint"
        )));
    }
    let diffusion = match &layout.config.model {
        ModelConfig::Diffusion(model) => model,
        ModelConfig::Autoencoder(_) => {
            return Err(AudioError::Msg(format!(
                "{expected_repo} provider rejects standalone SAME snapshots"
            )));
        }
    };
    let dit = &diffusion.diffusion;
    let cfg = &dit.config;
    if dit.diffusion_objective != DiffusionObjective::RfDenoiser
        || diffusion.io_channels != LATENT_CHANNELS
        || cfg.io_channels != LATENT_CHANNELS
        || cfg.embed_dim != shape.embed_dim
        || cfg.depth != shape.depth
        || cfg.num_heads != shape.num_heads
        || cfg.attn_kwargs.differential != shape.differential
        || cfg.cond_token_dim != 768
        || cfg.global_cond_dim != 768
        || cfg.local_add_cond_dim != Some(257)
        || diffusion.pretransform.config.latent_dim != LATENT_CHANNELS
        || diffusion.pretransform.config.downsampling_ratio != 4_096
        || !diffusion.pretransform.chunked
    {
        return Err(AudioError::Msg(format!(
            "snapshot architecture is not {expected_repo}"
        )));
    }
    match conditioner_repo_id(layout) {
        Some(repo) if repo == expected_repo => {}
        Some(repo) => {
            return Err(AudioError::Msg(format!(
                "snapshot declares conditioner repo_id {repo}, which is not {expected_repo}; \
                 refusing to serve one Stable Audio 3 checkpoint under another's provider id"
            )));
        }
        None => {
            return Err(AudioError::Msg(format!(
                "snapshot declares no conditioner repo_id and cannot be authenticated as \
                 {expected_repo}"
            )));
        }
    }
    let text = layout.text_keys.as_ref().ok_or_else(|| {
        AudioError::Msg(format!("{expected_repo} snapshot has no bundled T5Gemma"))
    })?;
    if text.total != crate::weights::TextWeightSummary::TOTAL
        || text.encoder != crate::weights::TextWeightSummary::ENCODER
        || text.decoder != crate::weights::TextWeightSummary::DECODER
    {
        return Err(AudioError::Msg(format!(
            "{expected_repo} bundled T5Gemma inventory mismatch"
        )));
    }
    Ok(())
}

fn canceled(is_canceled: &dyn Fn() -> bool) -> Result<()> {
    if is_canceled() {
        Err(AudioError::Canceled)
    } else {
        Ok(())
    }
}

fn apply_padding_mask(audio: &Tensor, padding: &Tensor, ratio: usize) -> Result<Tensor> {
    let (batch, channels, samples) = audio.dims3()?;
    let mask = padding
        .unsqueeze(1)?
        .unsqueeze(3)?
        .repeat((1, channels, 1, ratio))?
        .reshape((batch, channels, padding.dim(1)? * ratio))?;
    let mask = if mask.dim(2)? >= samples {
        mask.narrow(2, 0, samples)?
    } else {
        mask.pad_with_zeros(2, 0, samples - mask.dim(2)?)?
    };
    Ok(audio.broadcast_mul(&mask.to_dtype(audio.dtype())?)?)
}

fn planar_to_interleaved(audio: &Tensor) -> Result<Vec<f32>> {
    let (batch, channels, frames) = audio.dims3()?;
    if batch != 1 || channels != CHANNELS {
        return Err(AudioError::Msg(format!(
            "SA3 small decode expected [1,{CHANNELS},frames], got [{batch},{channels},{frames}]"
        )));
    }
    let planar = audio.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output = Vec::with_capacity(frames * channels);
    for (&left, &right) in planar[0][0].iter().zip(&planar[0][1]) {
        output.push(left);
        output.push(right);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_mask_repeats_each_latent_over_its_complete_audio_span() {
        let device = Device::Cpu;
        let audio = Tensor::ones((1, 2, 12), DType::F32, &device).unwrap();
        let padding = Tensor::new(&[[1u8, 1, 0]], &device).unwrap();
        let masked = apply_padding_mask(&audio, &padding, 4)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(
            masked,
            vec![vec![
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]]
        );
    }
}
