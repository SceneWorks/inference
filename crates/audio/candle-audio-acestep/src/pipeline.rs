//! The assembled ACE-Step 1.5 text-to-music synthesis pipeline (sc-12842): Qwen prompt/lyric
//! encode → condition-encoder context → flow-matching DiT denoise (turbo, no CFG) → Oobleck VAE
//! decode → peak-normalized stereo waveform — the reference `AceStepPipeline.__call__`
//! text-to-music flow.
//!
//! ## Determinism
//!
//! The only stochastic input is the initial noise, drawn host-side from a seeded `StdRng`
//! (standard-normal), so the same request + seed re-synthesizes byte-identically on the same
//! backend (the gen-core seed law). Cross-framework noise parity with torch's Philox is not a goal.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use tokenizers::Tokenizer;

use crate::condition::ConditionEncoder;
use crate::config::SnapshotConfig;
use crate::dit::DiT;
use crate::qwen::Qwen3Encoder;
use crate::scheduler::{FlowMatchSchedule, DEFAULT_SHIFT};
use crate::text::{build_prompt, tokenize_lyrics, tokenize_prompt, Metadata};
use crate::vae::{OobleckDecoder, VAE_FILE};

/// Sampling knobs of one synthesis request (defaults are the reference turbo call defaults).
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Requested output duration in seconds (reference default 60.0; clamped to the model cap).
    pub seconds: f32,
    /// Flow-matching solver steps (turbo default 8).
    pub steps: usize,
    /// Flow-match sigma shift (turbo default 3.0).
    pub shift: f64,
    /// Lyrics (structured with `[verse]`/`[chorus]`/…), empty for instrumental.
    pub lyrics: String,
    /// Musical metadata woven into the prompt.
    pub metadata: Metadata,
    /// Noise seed.
    pub seed: u64,
}

pub const DEFAULT_SECONDS: f32 = 60.0;
pub const DEFAULT_STEPS: usize = 8;

/// Pipeline-level progress events, mapped by the provider onto `gen_core::Progress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineProgress {
    Step(usize),
    Decoding,
}

/// The loaded pipeline (all components resident, f32).
pub struct AceStepPipeline {
    pub config: SnapshotConfig,
    tokenizer: Tokenizer,
    text_encoder: Qwen3Encoder,
    condition: ConditionEncoder,
    dit: DiT,
    vae: OobleckDecoder,
    device: Device,
}

/// Resolve the safetensors shard paths for a component whose files are `{stem}.safetensors` (single
/// file) or `{stem}-NNNNN-of-MMMMM.safetensors` enumerated by `{stem}.safetensors.index.json`.
fn safetensors_shards(dir: &Path, stem: &str) -> Result<Vec<PathBuf>> {
    let index_path = dir.join(format!("{stem}.safetensors.index.json"));
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", index_path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| AudioError::Msg(format!("parse {}: {e}", index_path.display())))?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| AudioError::Msg(format!("{}: no weight_map", index_path.display())))?;
        let mut shards: Vec<String> = map
            .values()
            .filter_map(|s| s.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        return Ok(shards.into_iter().map(|s| dir.join(s)).collect());
    }
    let single = dir.join(format!("{stem}.safetensors"));
    if single.is_file() {
        return Ok(vec![single]);
    }
    Err(AudioError::Msg(format!(
        "{}: neither {stem}.safetensors.index.json nor {stem}.safetensors present",
        dir.display()
    )))
}

fn mmap_vb(paths: &[PathBuf], device: &Device) -> Result<VarBuilder<'static>> {
    // Safety: mmap of pinned-SHA snapshot files the contract guarantees are not mutated.
    unsafe {
        VarBuilder::from_mmaped_safetensors(paths, DType::F32, device)
            .map_err(|e| AudioError::Msg(format!("mmap safetensors: {e}")))
    }
}

impl AceStepPipeline {
    pub fn from_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let config = SnapshotConfig::from_snapshot(root)?;

        let tokenizer_path = root.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AudioError::Msg(format!("load {}: {e}", tokenizer_path.display())))?;

        let te_shards = safetensors_shards(&root.join("text_encoder"), "model")?;
        let text_encoder = Qwen3Encoder::new(&config.text_encoder, mmap_vb(&te_shards, device)?)
            .map_err(|e| AudioError::Msg(format!("build qwen text encoder: {e}")))?;

        let ce_path = root.join("condition_encoder/diffusion_pytorch_model.safetensors");
        let ce_path = if ce_path.is_file() {
            ce_path
        } else {
            root.join("condition_encoder/model.safetensors")
        };
        let condition = ConditionEncoder::new(&config.condition, mmap_vb(&[ce_path], device)?)
            .map_err(|e| AudioError::Msg(format!("build condition encoder: {e}")))?;

        let dit_shards = safetensors_shards(&root.join("transformer"), "diffusion_pytorch_model")?;
        let dit = DiT::new(&config.transformer, mmap_vb(&dit_shards, device)?)
            .map_err(|e| AudioError::Msg(format!("build acestep DiT: {e}")))?;

        let vae = OobleckDecoder::load(&root.join("vae").join(VAE_FILE), &config.vae, device)?;

        Ok(Self {
            config,
            tokenizer,
            text_encoder,
            condition,
            dit,
            vae,
            device: device.clone(),
        })
    }

    /// Latent frames for a requested duration (`ceil(seconds · latents_per_second)`).
    pub fn latent_length(&self, seconds: f32) -> usize {
        (seconds as f64 * self.config.vae.latents_per_second()).ceil() as usize
    }

    fn encode_context(&self, prompt: &str, lyrics: &str, meta: &Metadata) -> Result<Tensor> {
        let full_prompt = build_prompt(prompt, meta);
        let prompt_ids = tokenize_prompt(&self.tokenizer, &full_prompt)?;
        let text_hidden = if prompt_ids.is_empty() {
            Tensor::zeros(
                (1, 1, self.text_encoder.hidden_size()),
                DType::F32,
                &self.device,
            )?
        } else {
            self.text_encoder
                .encode(&prompt_ids)
                .map_err(AudioError::from)?
        };
        let lyric_ids = tokenize_lyrics(&self.tokenizer, lyrics)?;
        let lyric_embeds = if lyric_ids.is_empty() {
            None
        } else {
            Some(
                self.text_encoder
                    .embed(&lyric_ids)
                    .map_err(AudioError::from)?,
            )
        };
        // Text-to-music: the condition encoder supplies its own timbre special token.
        self.condition
            .encode(&text_hidden, lyric_embeds.as_ref())
            .map_err(AudioError::from)
    }

    /// Seeded standard-normal initial latents `[1, latent_len, acoustic]`.
    fn seeded_noise_with(&self, latent_len: usize, seed: u64) -> Result<Tensor> {
        let acoustic = self.config.transformer.audio_acoustic_hidden_dim;
        let n = latent_len * acoustic;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        Ok(Tensor::from_vec(
            noise,
            (1, latent_len, acoustic),
            &self.device,
        )?)
    }

    /// Synthesize one clip → interleaved stereo PCM in `[-1, 1]`.
    pub fn synthesize(
        &self,
        prompt: &str,
        params: &SynthesisParams,
        on_progress: &mut dyn FnMut(PipelineProgress),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let acoustic = self.config.transformer.audio_acoustic_hidden_dim;
        let seconds = params.seconds.max(0.1);
        let latent_len = self.latent_length(seconds).max(1);

        if cancel() {
            return Err(AudioError::Canceled);
        }

        let ctx_raw = self.encode_context(prompt, &params.lyrics, &params.metadata)?;
        let ctx = self.dit.embed_context(&ctx_raw)?;

        // Text-to-music context latents: [src_latents(silence) | chunk_mask(ones)].
        let src_latents = self
            .condition
            .src_latents(latent_len, &self.device)
            .map_err(AudioError::from)?;
        let chunk_mask = Tensor::ones((1, latent_len, acoustic), DType::F32, &self.device)?;
        let context_latents = Tensor::cat(
            &[&src_latents, &chunk_mask],
            candle_audio::candle_core::D::Minus1,
        )?;

        let shift = if params.shift > 0.0 {
            params.shift
        } else {
            DEFAULT_SHIFT
        };
        let schedule = FlowMatchSchedule::new(params.steps, shift);
        let mut latents = self.seeded_noise_with(latent_len, params.seed)?;

        for k in 0..schedule.num_steps() {
            if cancel() {
                return Err(AudioError::Canceled);
            }
            let t = schedule.timestep(k);
            let v = self
                .dit
                .forward(&latents, &context_latents, t, &ctx, cancel)?
                .ok_or(AudioError::Canceled)?;
            latents = schedule.step(&v, &latents, k)?;
            on_progress(PipelineProgress::Step(k + 1));
        }

        on_progress(PipelineProgress::Decoding);
        if cancel() {
            return Err(AudioError::Canceled);
        }
        // Decode: [1, T, acoustic] → [1, acoustic, T] → Oobleck → [1, channels, samples].
        let audio_latents = latents.transpose(1, 2)?.contiguous()?;
        let audio = self
            .vae
            .decode(&audio_latents, cancel)?
            .ok_or(AudioError::Canceled)?;
        let channels = self.vae.audio_channels();
        let interleaved = interleave(&audio, channels)?;

        // Crop to the requested duration and peak-normalize to −1 dBFS (reference two-stage norm).
        let sample_rate = self.config.vae.sampling_rate as f64;
        let want =
            ((sample_rate * seconds as f64).round() as usize * channels).min(interleaved.len());
        let mut out = interleaved[..want].to_vec();
        peak_normalize(&mut out);
        Ok(out)
    }
}

/// `[1, C, T]` → interleaved `[t0c0, t0c1, t1c0, …]`.
fn interleave(audio: &Tensor, channels: usize) -> Result<Vec<f32>> {
    let planar: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
    let t = planar.len() / channels;
    let mut out = Vec::with_capacity(planar.len());
    for i in 0..t {
        for c in 0..channels {
            out.push(planar[c * t + i]);
        }
    }
    Ok(out)
}

/// The reference two-stage loudness normalization: clip peaks over 1.0, then rescale so the peak
/// sits at −1 dBFS (`10^(−1/20) ≈ 0.891`).
fn peak_normalize(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak > 1.0 {
        let d = peak.max(1.0);
        for s in samples.iter_mut() {
            *s /= d;
        }
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs())).max(1e-6);
    let target = 10f32.powf(-1.0 / 20.0);
    let g = target / peak;
    for s in samples.iter_mut() {
        *s *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_normalize_hits_minus_one_dbfs() {
        let mut s = vec![0.5, -0.25, 0.1];
        peak_normalize(&mut s);
        let peak = s.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!((peak - 10f32.powf(-1.0 / 20.0)).abs() < 1e-5, "peak {peak}");
    }

    #[test]
    fn interleave_zips_channels() {
        let dev = Device::Cpu;
        // [1, 2, 3]: ch0 = [0,1,2], ch1 = [10,11,12].
        let a = Tensor::from_vec(vec![0f32, 1., 2., 10., 11., 12.], (1, 2, 3), &dev).unwrap();
        assert_eq!(interleave(&a, 2).unwrap(), vec![0., 10., 1., 11., 2., 12.]);
    }
}
