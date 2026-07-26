//! Connected Stable Audio 3 small (music / SFX) inference pipeline.

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};

use crate::config::{ConditionerConfig, DiffusionObjective, ModelConfig};
use crate::dit::{Guidance, StableAudio3Dit};
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
pub const DEFAULT_DURATION_SECS: f32 = 120.0;
pub const DEFAULT_STEPS: usize = 8;
pub const DEFAULT_GUIDANCE: f64 = 1.0;

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
pub struct StableAudio3SmallPipeline {
    config: crate::config::StableAudioConfig,
    conditioner: T5GemmaConditioner,
    dit: StableAudio3Dit,
    same: SameAutoencoder,
    device: Device,
}

impl StableAudio3SmallPipeline {
    /// Build the connected graph, refusing any snapshot that is not `expected_repo`'s checkpoint.
    pub fn from_layout(
        layout: &SnapshotLayout,
        expected_repo: &str,
        device: &Device,
    ) -> Result<Self> {
        validate_small_layout(layout, expected_repo)?;
        let diffusion = match &layout.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let builders = layout.full_pipeline_builders(DType::F32, DType::F32, device)?;
        let conditioner =
            T5GemmaConditioner::from_layout_with_builders(layout, device, builders.clone())?;
        let dit = StableAudio3Dit::load(diffusion, builders.clone())
            .map_err(|error| AudioError::Msg(format!("load SA3 DiT: {error}")))?;
        let same = SameAutoencoder::load(&diffusion.pretransform.config, builders)
            .map_err(|error| AudioError::Msg(format!("load SA3 SAME-S: {error}")))?;
        Ok(Self {
            config: layout.config.clone(),
            conditioner,
            dit,
            same,
            device: device.clone(),
        })
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
            DType::F32,
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
        let initial = initial.to_device(&self.device)?.to_dtype(DType::F32)?;
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
            &latents.to_device(&self.device)?.to_dtype(DType::F32)?,
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
            &latents.to_device(&self.device)?.to_dtype(DType::F32)?,
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
            DType::F32,
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

/// Validate that `layout` is the exact post-trained SA3 small checkpoint published by
/// `expected_repo`.
///
/// The architecture checks are shared by both registered variants; the `repo_id` check is what
/// stops one variant's snapshot loading under the other's provider id. `ModelRegistration::load`
/// carries no provider id, so without this the registry would happily serve music weights from the
/// `stable_audio_3_small_sfx` registration.
pub fn validate_small_layout(layout: &SnapshotLayout, expected_repo: &str) -> Result<()> {
    if layout.kind != SnapshotKind::Full
        || layout.config.sample_rate != SAMPLE_RATE
        || layout.config.sample_size != 5_292_032
        || layout.config.audio_channels != CHANNELS
        || layout.keys.total != 685
        || layout.keys.dit != 438
        || layout.keys.encoder + layout.keys.decoder + layout.keys.bottleneck != 244
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
        || cfg.embed_dim != 1_024
        || cfg.depth != 20
        || cfg.num_heads != 16
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
