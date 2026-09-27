//! The assembled MOSS-SoundEffect synthesis pipeline (sc-12841): Qwen3 text encode →
//! flow-matching DiT denoise (CFG) → continuous DAC VAE decode — the reference
//! `MossSoundEffectPipeline.__call__` / `WanAudioPipeline.__call__` flow.
//!
//! ## Duration and the denoise window
//!
//! The reference **always** denoises a fixed window of `max_inference_seconds` (30 s) latents and
//! crops the decoded waveform to the requested `seconds`:
//!
//! ```text
//!   full_seconds     = int(max_inference_seconds or self.max_inference_seconds)   # → 30
//!   num_samples_full = sample_rate * full_seconds                                 # denoised
//!   audio            = audio[:, :, :int(sample_rate * seconds)]                   # cropped
//! ```
//!
//! `max_inference_seconds` is a per-call *override* that defaults to `None`, so the shipped
//! default window is the model's full 30 s regardless of the requested duration. The duration
//! conditioning is purely textual (the `" duration: {seconds:.1}s"` prompt suffix).
//!
//! This port previously shortened the window to `ceil(seconds)` so a 4-second clip cost a
//! 4-second denoise. That is **not** the reference default, and it is not a safe optimization:
//! the DiT is trained on the full-length latent sequence, so a short window puts it far out of
//! distribution. The velocity field it then predicts is wrong, CFG multiplies that error by
//! `cfg_scale` (`v = v_neg + s·(v_pos − v_neg)`, so 4.0 triples the unconditional term's
//! contribution), and an accurate solve converges onto a degenerate solution that the VAE decodes
//! to a −74 dBFS residual floor. A *coarse* solve accidentally stepped over the degeneracy, which
//! is why the 30-step conformance test passed while the shipped default of 100 steps produced
//! silence. Measured, 3 s request: crest 11.2 dB at a 3 s window vs 26.7 dB at the full window.
//!
//! The window is therefore always the model's `max_inference_seconds`. This costs a full-length
//! denoise for every clip — the price the reference pays for correctness.
//!
//! ## Determinism
//!
//! The only stochastic input is the initial noise, drawn host-side from a seeded `StdRng`
//! (standard-normal), so the same request + seed re-synthesizes byte-identically on the same
//! backend (the gen-core seed law). Cross-framework noise parity with torch's Philox is not a
//! goal.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::safetensors_shards::{
    resolve_safetensors_shards, snapshot_shard_roots,
};
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

fn configured_full_window_seconds(max_inference_seconds: u32) -> u32 {
    max_inference_seconds.max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SynthesisGeometry {
    denoise_seconds: u32,
    requested_samples: usize,
}

fn synthesis_geometry(
    denoise_seconds: u32,
    sample_rate: u32,
    requested_seconds: f32,
) -> SynthesisGeometry {
    SynthesisGeometry {
        denoise_seconds,
        requested_samples: ((sample_rate as f64) * requested_seconds as f64).round() as usize,
    }
}

fn crop_decoded_audio(mut samples: Vec<f32>, requested_samples: usize) -> Vec<f32> {
    samples.truncate(requested_samples);
    samples
}

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
    let roots = snapshot_shard_roots(dir).map_err(|e| AudioError::Msg(e.to_string()))?;
    resolve_safetensors_shards(dir, "model", &roots).map_err(|e| AudioError::Msg(e.to_string()))
}

fn mmap_text_encoder_shards(shards: &[PathBuf], device: &Device) -> Result<VarBuilder<'static>> {
    // Safety: mmap of files that the pinned-SHA snapshot contract guarantees are not
    // mutated concurrently — the same invariant every provider family relies on.
    unsafe {
        VarBuilder::from_mmaped_safetensors(shards, DType::F32, device)
            .map_err(|e| AudioError::Msg(format!("mmap text encoder shards: {e}")))
    }
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
        let vb = mmap_text_encoder_shards(&shards, device)?;
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

    /// The configured full denoise window in whole seconds. Requested output duration does not
    /// enter this seam; it controls only the post-decode crop.
    pub fn full_window_seconds(&self) -> u32 {
        configured_full_window_seconds(self.config.index.max_inference_seconds)
    }

    /// Compatibility wrapper for the original public API. The argument was always deliberately
    /// ignored: MOSS-SFX denoises the configured full window and crops only after decoding.
    pub fn window_seconds(&self, _requested_seconds: f32) -> u32 {
        self.full_window_seconds()
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
        let geometry = synthesis_geometry(self.full_window_seconds(), sample_rate, seconds);
        let latent_len = geometry.denoise_seconds as usize * sample_rate as usize / HOP_LENGTH;

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
        Ok(crop_decoded_audio(full, geometry.requested_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        configured_full_window_seconds, crop_decoded_audio, synthesis_geometry, SynthesisGeometry,
    };
    use super::{mmap_text_encoder_shards, text_encoder_shards};
    use candle_audio::candle_core::Device;
    use std::path::Path;

    fn shard_fixture(path: &Path) {
        let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    fn shard_index(dir: &Path, name: &str) {
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": {"x": name}}).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn text_encoder_shard_loader_rejects_invalid_entries_before_mmap() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("text_encoder");
        std::fs::create_dir(&dir).unwrap();
        shard_fixture(&dir.join("model.safetensors"));
        shard_index(&dir, "model.safetensors");
        let shards = text_encoder_shards(&dir).unwrap();
        assert_eq!(
            mmap_text_encoder_shards(&shards, &Device::Cpu)
                .unwrap()
                .get((1,), "x")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );

        std::fs::create_dir(dir.join("nested")).unwrap();
        shard_fixture(&dir.join("nested/part.bin"));
        shard_index(&dir, "nested/part.bin");
        let nested = text_encoder_shards(&dir).unwrap();
        assert_eq!(
            mmap_text_encoder_shards(&nested, &Device::Cpu)
                .unwrap()
                .get((1,), "x")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );

        for name in [
            "../outside.safetensors",
            "/tmp/outside.safetensors",
            "missing.safetensors",
        ] {
            shard_index(&dir, name);
            assert!(text_encoder_shards(&dir).is_err(), "{name}");
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = temp.path().join("outside.safetensors");
            shard_fixture(&outside);
            symlink(&outside, dir.join("external.safetensors")).unwrap();
            shard_index(&dir, "external.safetensors");
            assert!(text_encoder_shards(&dir)
                .unwrap_err()
                .to_string()
                .contains("outside authorized"));

            let fifo = dir.join("pipe.safetensors");
            assert!(std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success());
            shard_index(&dir, "pipe.safetensors");
            assert!(text_encoder_shards(&dir)
                .unwrap_err()
                .to_string()
                .contains("not a regular file"));

            let repository = temp.path().join("models--org--moss");
            let cache_dir = repository.join("snapshots/revision/text_encoder");
            let blobs = repository.join("blobs");
            std::fs::create_dir_all(&cache_dir).unwrap();
            std::fs::create_dir(&blobs).unwrap();
            shard_fixture(&blobs.join("digest"));
            symlink("../../../blobs/digest", cache_dir.join("model.safetensors")).unwrap();
            shard_index(&cache_dir, "model.safetensors");
            let shards = text_encoder_shards(&cache_dir).unwrap();
            assert_eq!(
                mmap_text_encoder_shards(&shards, &Device::Cpu)
                    .unwrap()
                    .get((1,), "x")
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                vec![1.0]
            );
        }
    }

    #[test]
    fn denoise_window_is_configured_full_window_not_requested_crop() {
        assert_eq!(
            synthesis_geometry(30, 48_000, 4.0),
            SynthesisGeometry {
                denoise_seconds: 30,
                requested_samples: 4 * 48_000,
            }
        );
        assert_eq!(configured_full_window_seconds(0), 1);

        let decoded = vec![0.0_f32; 30 * 48_000];
        let allocation = decoded.as_ptr();
        let cropped = crop_decoded_audio(decoded, 4 * 48_000);
        assert_eq!(cropped.len(), 4 * 48_000);
        assert_eq!(cropped.as_ptr(), allocation, "crop must truncate in place");
    }
}
