//! The assembled MOSS-SoundEffect synthesis pipeline (sc-12841): Qwen3 text encode →
//! flow-matching DiT denoise (CFG) → continuous DAC VAE decode — the reference
//! `MossSoundEffectPipeline.__call__` / `WanAudioPipeline.__call__` flow.
//!
//! ## Duration and the denoise window
//!
//! The reference always denoises a fixed window of `max_inference_seconds` (30 s) latents and
//! crops the decoded waveform to the requested `seconds`; the same call exposes
//! `max_inference_seconds` as a per-call override. This port sets that window to
//! `ceil(seconds)` (clamped to the model's 30 s cap) — the exact computation the reference
//! performs when handed `max_inference_seconds=ceil(seconds)` — so a 4-second clip costs a
//! 4-second denoise, not a 30-second one. The duration conditioning itself is textual (the
//! `" duration: {seconds:.1}s"` prompt suffix), unchanged from the reference.
//!
//! ## Determinism
//!
//! The only stochastic input is the initial noise, drawn host-side from a seeded `StdRng`
//! (standard-normal), so the same request + seed re-synthesizes byte-identically on the same
//! backend (the gen-core seed law). Cross-framework noise parity with torch's Philox is not a
//! goal.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use tokenizers::Tokenizer;

use crate::config::SnapshotConfig;
use crate::dit::DiT;
use crate::qwen3::Qwen3Encoder;
use crate::sampler::FlowMatchSchedule;
use crate::text::{clean_prompt, tokenize, with_duration_suffix, TEXT_LEN};
use crate::vae::{DacDecoder, HOP_LENGTH, VAE_FILE};

/// Sampling knobs of one synthesis request (defaults are the reference call defaults).
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Requested output duration in seconds (reference default 10.0), rounded to 0.1 s.
    pub seconds: f32,
    /// Flow-matching solver steps (reference default 100).
    pub steps: usize,
    /// Classifier-free guidance scale (reference default 4.0; 1.0 disables the negative pass).
    pub cfg_scale: f32,
    /// Flow-match sigma shift (reference default 5.0 — the scheduler config value).
    pub sigma_shift: Option<f64>,
    /// Negative prompt ("" — the reference default — encodes to the all-zero context).
    pub negative_prompt: String,
    /// Noise seed.
    pub seed: u64,
}

pub const DEFAULT_SECONDS: f32 = 10.0;
pub const DEFAULT_STEPS: usize = 100;
pub const DEFAULT_CFG_SCALE: f32 = 4.0;

/// Pipeline-level progress events, mapped by the provider onto
/// [`gen_core::Progress`](candle_audio::gen_core::Progress) (one callback so the caller's
/// progress sink is borrowed exactly once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineProgress {
    /// Solver step `k` of the run just completed (`k = 1..=steps`).
    Step(usize),
    /// The terminal VAE decode is about to run (fires exactly once).
    Decoding,
}

/// The loaded pipeline (all components resident, f32).
pub struct MossSfxPipeline {
    pub config: SnapshotConfig,
    tokenizer: Tokenizer,
    text_encoder: Qwen3Encoder,
    dit: DiT,
    vae: DacDecoder,
    device: Device,
}

/// Enumerate the text-encoder safetensors shards via `model.safetensors.index.json` (falling
/// back to the single-file layout when no index exists).
fn text_encoder_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
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
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    Err(AudioError::Msg(format!(
        "{}: neither model.safetensors.index.json nor model.safetensors present",
        dir.display()
    )))
}

impl MossSfxPipeline {
    /// Load every component from a pinned snapshot directory. All weights are converted to
    /// f32 (the compute dtype of the CPU-first audio lane; the bf16 text encoder and f32 DiT
    /// both land on the same dtype).
    pub fn from_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let config = SnapshotConfig::from_snapshot(root)?;

        let tokenizer_path = root.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AudioError::Msg(format!("load {}: {e}", tokenizer_path.display())))?;

        let shards = text_encoder_shards(&root.join("text_encoder"))?;
        // Safety: mmap of files that the pinned-SHA snapshot contract guarantees are not
        // mutated concurrently — the same invariant every provider family relies on.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, device)
                .map_err(|e| AudioError::Msg(format!("mmap text encoder shards: {e}")))?
        };
        let text_encoder = Qwen3Encoder::new(&config.text_encoder, vb)
            .map_err(|e| AudioError::Msg(format!("build qwen3 text encoder: {e}")))?;

        let dit_path = root.join("transformer/diffusion_pytorch_model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&dit_path), DType::F32, device)
                .map_err(|e| AudioError::Msg(format!("mmap {}: {e}", dit_path.display())))?
        };
        let dit = DiT::new(&config.dit, vb)
            .map_err(|e| AudioError::Msg(format!("build audio DiT: {e}")))?;
        if config.dit.in_dim != crate::vae::LATENT_DIM {
            return Err(AudioError::Msg(format!(
                "moss-sfx: DiT in_dim {} != VAE latent dim {}",
                config.dit.in_dim,
                crate::vae::LATENT_DIM
            )));
        }

        let vae = DacDecoder::load(&root.join("vae").join(VAE_FILE), device)?;

        Ok(Self {
            config,
            tokenizer,
            text_encoder,
            dit,
            vae,
            device: device.clone(),
        })
    }

    /// Encode one prompt to the DiT text context `[1, TEXT_LEN, text_dim]`: cleaned +
    /// tokenized (truncated to [`TEXT_LEN`]), Qwen3 last-hidden-states for the valid rows,
    /// zero rows after — the reference's padded-then-zeroed context, computed unpadded (causal
    /// attention makes the valid rows identical; see `crate::text`).
    fn encode_context(&self, prompt: &str) -> Result<Tensor> {
        let cleaned = clean_prompt(prompt);
        let ids = tokenize(&self.tokenizer, &cleaned)?;
        let text_dim = self.config.text_encoder.hidden_size;
        if ids.is_empty() {
            return Ok(Tensor::zeros(
                (1, TEXT_LEN, text_dim),
                DType::F32,
                &self.device,
            )?);
        }
        let valid = self.text_encoder.encode(&ids)?; // [1, n, text_dim]
        let n = valid.dims3().map_err(AudioError::from)?.1;
        if n >= TEXT_LEN {
            return Ok(valid.narrow(1, 0, TEXT_LEN)?);
        }
        let pad = Tensor::zeros((1, TEXT_LEN - n, text_dim), DType::F32, &self.device)?;
        Ok(Tensor::cat(&[&valid, &pad], 1)?)
    }

    /// Seeded standard-normal initial latents `[1, latent_dim, latent_len]`.
    fn seeded_noise(&self, latent_len: usize, seed: u64) -> Result<Tensor> {
        let n = self.config.dit.in_dim * latent_len;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        Ok(Tensor::from_vec(
            noise,
            (1, self.config.dit.in_dim, latent_len),
            &self.device,
        )?)
    }

    /// The denoise window in whole seconds for a requested duration (see module docs).
    pub fn window_seconds(&self, seconds: f32) -> u32 {
        (seconds.ceil() as u32)
            .max(1)
            .min(self.config.index.max_inference_seconds)
    }

    /// Synthesize one clip. `on_progress` receives [`PipelineProgress::Step`] after each
    /// completed solver step (`k = 1..=steps`) and [`PipelineProgress::Decoding`] once before
    /// the VAE decode; `cancel` is polled before every solver step, between DiT blocks, and
    /// inside the VAE decode stages, returning the typed [`AudioError::Canceled`].
    pub fn synthesize(
        &self,
        prompt: &str,
        params: &SynthesisParams,
        on_progress: &mut dyn FnMut(PipelineProgress),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        let sample_rate = self.config.index.sample_rate;
        let seconds = crate::text::round_seconds(params.seconds);
        if seconds <= 0.0 {
            return Err(AudioError::Msg(format!(
                "moss-sfx: seconds must be > 0 after 0.1 s rounding (got {seconds})"
            )));
        }
        let window = self.window_seconds(seconds);
        let latent_len = window as usize * sample_rate as usize / HOP_LENGTH;

        if cancel() {
            return Err(AudioError::Canceled);
        }

        // Text conditioning: positive prompt carries the duration suffix; the negative prompt
        // is passed through as-is (reference behavior).
        let positive = with_duration_suffix(prompt, seconds);
        let ctx_pos = self.dit.embed_context(&self.encode_context(&positive)?)?;
        let use_cfg = params.cfg_scale != 1.0;
        let ctx_neg = if use_cfg {
            Some(
                self.dit
                    .embed_context(&self.encode_context(&params.negative_prompt)?)?,
            )
        } else {
            None
        };

        let schedule =
            FlowMatchSchedule::new(&self.config.scheduler, params.steps, params.sigma_shift);
        let (cos, sin) = self.dit.rope(latent_len, &self.device)?;
        let mut latents = self.seeded_noise(latent_len, params.seed)?;

        for k in 0..schedule.num_steps() {
            if cancel() {
                return Err(AudioError::Canceled);
            }
            let t = schedule.timestep(k);
            let v_pos = self
                .dit
                .forward(&latents, t, &ctx_pos, &cos, &sin, cancel)?
                .ok_or(AudioError::Canceled)?;
            let v = if let Some(ctx_neg) = &ctx_neg {
                let v_neg = self
                    .dit
                    .forward(&latents, t, ctx_neg, &cos, &sin, cancel)?
                    .ok_or(AudioError::Canceled)?;
                // v = v_neg + s·(v_pos − v_neg)
                (&v_neg + (v_pos - &v_neg)?.affine(params.cfg_scale as f64, 0.0)?)?
            } else {
                v_pos
            };
            latents = schedule.step(&v, &latents, k)?;
            on_progress(PipelineProgress::Step(k + 1));
        }

        on_progress(PipelineProgress::Decoding);
        if cancel() {
            return Err(AudioError::Canceled);
        }
        let audio = self
            .vae
            .decode(&latents, cancel)?
            .ok_or(AudioError::Canceled)?; // [1, 1, window·sr]
        let full: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        let out_len = ((sample_rate as f64) * seconds as f64).round() as usize;
        let out_len = out_len.min(full.len());
        Ok(full[..out_len].to_vec())
    }
}
