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
use crate::vae::{OobleckDecoder, OobleckEncoder, VAE_FILE};

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

/// Which ACE-Step audio-to-audio task a prompted edit drives (sc-12847). The gen-core
/// [`AudioEditMode`](candle_audio::gen_core::AudioEditMode) maps onto this at the provider boundary;
/// `Cover` is intentionally absent (the pinned diffusers checkpoint ships no audio quantizer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditTask {
    /// Regenerate a bounded window fresh — the window is silence-seeded so the model fills it anew,
    /// conditioning on the surrounding source (ACE-Step `repaint`).
    Inpaint,
    /// Regenerate a bounded window conditioned on the surrounding source (ACE-Step `repaint`).
    Repaint,
    /// Generate an appended tail beyond the source length (ACE-Step `repaint` with the generate
    /// window at the tail).
    Extend,
}

/// One prompted-edit request's parameters (sc-12847). The `base` sampling knobs are shared with
/// [`SynthesisParams`]; the source waveform is passed separately to [`AceStepPipeline::edit`].
#[derive(Debug, Clone)]
pub struct EditParams {
    pub task: EditTask,
    /// Region start (seconds). For `Extend` this is where generation begins (defaults applied by
    /// the caller to the source length).
    pub region_start_secs: f32,
    /// Region end (seconds). For region tasks `None` ⇒ the source end; for `Extend` it is the new
    /// total length (required — the caller validates).
    pub region_end_secs: Option<f32>,
    /// Shared sampling knobs (steps / shift / lyrics / metadata / seed). `base.seconds` is unused
    /// for edits (durations come from the source length / region).
    pub base: SynthesisParams,
}

/// The loaded pipeline (all components resident, f32). The VAE encoder is resident too so a source
/// clip can be latent-encoded for prompted editing (sc-12847).
pub struct AceStepPipeline {
    pub config: SnapshotConfig,
    tokenizer: Tokenizer,
    text_encoder: Qwen3Encoder,
    condition: ConditionEncoder,
    dit: DiT,
    vae: OobleckDecoder,
    vae_encoder: OobleckEncoder,
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

        let vae_path = root.join("vae").join(VAE_FILE);
        let vae = OobleckDecoder::load(&vae_path, &config.vae, device)?;
        let vae_encoder = OobleckEncoder::load(&vae_path, &config.vae, device)?;

        Ok(Self {
            config,
            tokenizer,
            text_encoder,
            condition,
            dit,
            vae,
            vae_encoder,
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

        let latents = self
            .denoise(
                &context_latents,
                &ctx,
                latent_len,
                params,
                on_progress,
                cancel,
            )?
            .ok_or(AudioError::Canceled)?;

        on_progress(PipelineProgress::Decoding);
        let (interleaved, channels) = self
            .decode_to_interleaved(&latents, cancel)?
            .ok_or(AudioError::Canceled)?;

        // Crop to the requested duration and peak-normalize to −1 dBFS (reference two-stage norm).
        let sample_rate = self.config.vae.sampling_rate as f64;
        let want =
            ((sample_rate * seconds as f64).round() as usize * channels).min(interleaved.len());
        let mut out = interleaved[..want].to_vec();
        peak_normalize(&mut out);
        Ok(out)
    }

    /// The shared flow-matching denoise loop: seeded noise → `steps` Euler updates driven by
    /// `context_latents` (`[1, T, 2·acoustic]`) → final latents `[1, T, acoustic]`. Returns `None`
    /// on cancellation (checked before generate, at every step, and between DiT blocks). Both the
    /// text-to-music path and the edit path share this — the only difference between them is how
    /// `context_latents` is built.
    fn denoise(
        &self,
        context_latents: &Tensor,
        ctx: &Tensor,
        latent_len: usize,
        params: &SynthesisParams,
        on_progress: &mut dyn FnMut(PipelineProgress),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Tensor>> {
        let shift = if params.shift > 0.0 {
            params.shift
        } else {
            DEFAULT_SHIFT
        };
        let schedule = FlowMatchSchedule::new(params.steps, shift);
        let mut latents = self.seeded_noise_with(latent_len, params.seed)?;
        for k in 0..schedule.num_steps() {
            if cancel() {
                return Ok(None);
            }
            let t = schedule.timestep(k);
            let v = match self
                .dit
                .forward(&latents, context_latents, t, ctx, cancel)?
            {
                Some(v) => v,
                None => return Ok(None),
            };
            latents = schedule.step(&v, &latents, k)?;
            on_progress(PipelineProgress::Step(k + 1));
        }
        Ok(Some(latents))
    }

    /// Decode final latents `[1, T, acoustic]` to interleaved PCM, returning `(samples, channels)`.
    /// Returns `None` on cancellation.
    fn decode_to_interleaved(
        &self,
        latents: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<(Vec<f32>, usize)>> {
        if cancel() {
            return Ok(None);
        }
        // Decode: [1, T, acoustic] → [1, acoustic, T] → Oobleck → [1, channels, samples].
        let audio_latents = latents.transpose(1, 2)?.contiguous()?;
        let audio = match self.vae.decode(&audio_latents, cancel)? {
            Some(a) => a,
            None => return Ok(None),
        };
        let channels = self.vae.audio_channels();
        Ok(Some((interleave(&audio, channels)?, channels)))
    }

    /// Prompted source-audio editing (sc-12847): latent-encode `source` (interleaved PCM at the
    /// model's native rate, `source_channels` channels), build the ACE-Step mask-conditioned
    /// `context_latents` for the task (source outside the window, silence inside, the per-frame
    /// chunk mask marking the generate region), denoise, decode, and **stitch** the regenerated
    /// window back into the original waveform with an equal-power crossfade.
    ///
    /// The generation mechanism is the faithful ACE-Step `repaint` recipe (mask + reference latents
    /// fed as `context_latents`; no in-loop latent blending — the model is natively mask-conditioned,
    /// diffusers `pipeline_ace_step.py`). The waveform stitch is the preservation guarantee layered
    /// on top: samples strictly outside the region are the *original* source (bit-preserved, no VAE
    /// round-trip), only the region carries new audio, and the crossfade — placed entirely inside
    /// the window — hides the seam without touching the untouched span.
    pub fn edit(
        &self,
        prompt: &str,
        source: &[f32],
        source_channels: usize,
        params: &EditParams,
        on_progress: &mut dyn FnMut(PipelineProgress),
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        if cancel() {
            return Err(AudioError::Canceled);
        }
        let hop = self.config.vae.hop_length();
        let out_channels = self.vae.audio_channels();
        let src_ch = source_channels.max(1);
        let src_frames = source.len() / src_ch;
        if src_frames == 0 {
            return Err(AudioError::Msg("acestep edit: empty source audio".into()));
        }

        // 1. Source waveform → planar [1, out_channels, N_pad] (padded to a hop multiple) → latents.
        let planar = self.source_to_planar(source, src_ch, out_channels, hop)?;
        let src_lat = self
            .vae_encoder
            .encode(&planar, cancel)?
            .ok_or(AudioError::Canceled)?; // [1, L_src, acoustic]
        let l_src = src_lat.dim(1)?;

        // 2. Target latent length + generate window [w0, w1) in latent frames.
        let (target_len, w0, w1) = self.edit_window(params, src_frames, l_src)?;

        // 3. src_latents: original source outside the window, silence inside; silence tail past the
        //    source length (extend). Mirrors the reference `torch.where(chunk_mask>0.5, silence, src)`.
        let silence = self
            .condition
            .src_latents(target_len, &self.device)
            .map_err(AudioError::from)?; // [1, target_len, acoustic]
        let base_src = if l_src >= target_len {
            src_lat.narrow(1, 0, target_len)?
        } else {
            Tensor::cat(
                &[&src_lat, &silence.narrow(1, l_src, target_len - l_src)?],
                1,
            )?
        };
        let mut parts: Vec<Tensor> = Vec::new();
        if w0 > 0 {
            parts.push(base_src.narrow(1, 0, w0)?);
        }
        parts.push(silence.narrow(1, w0, w1 - w0)?);
        if w1 < target_len {
            parts.push(base_src.narrow(1, w1, target_len - w1)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let src_full = Tensor::cat(&refs, 1)?;

        // 4. chunk_mask: 1 inside the window (generate), 0 outside (keep source).
        let acoustic = self.config.transformer.audio_acoustic_hidden_dim;
        let mask = self.window_mask(target_len, w0, w1, acoustic)?;
        let context_latents =
            Tensor::cat(&[&src_full, &mask], candle_audio::candle_core::D::Minus1)?;

        // 5. Condition context (prompt + lyrics + metadata) — identical to text-to-music.
        let ctx_raw = self.encode_context(prompt, &params.base.lyrics, &params.base.metadata)?;
        let ctx = self.dit.embed_context(&ctx_raw)?;

        // 6. Denoise + decode.
        let latents = self
            .denoise(
                &context_latents,
                &ctx,
                target_len,
                &params.base,
                on_progress,
                cancel,
            )?
            .ok_or(AudioError::Canceled)?;
        on_progress(PipelineProgress::Decoding);
        let (mut generated, gen_ch) = self
            .decode_to_interleaved(&latents, cancel)?
            .ok_or(AudioError::Canceled)?;
        // -1 dBFS, matching the loudness a source clip typically carries so the stitched region sits
        // at the same level as the preserved surroundings.
        peak_normalize(&mut generated);

        // 7. Stitch the regenerated window back into the original waveform.
        let gen_frames = generated.len() / gen_ch;
        let out_frames = match params.task {
            EditTask::Extend => (target_len * hop).min(gen_frames),
            EditTask::Inpaint | EditTask::Repaint => src_frames.min(gen_frames),
        };
        let f0 = w0 * hop;
        let f1 = (w1 * hop).min(out_frames);
        let xfade = crossfade_frames(f0, f1);
        Ok(stitch(
            source, src_ch, &generated, gen_ch, out_frames, f0, f1, xfade,
        ))
    }

    /// Source interleaved PCM → planar `[1, out_channels, N_pad]`, padded up to a whole hop so the
    /// strided encoder produces exactly `N_pad / hop` latent frames. A mono source is replicated to
    /// every output channel; extra source channels beyond `out_channels` are dropped.
    fn source_to_planar(
        &self,
        source: &[f32],
        src_ch: usize,
        out_ch: usize,
        hop: usize,
    ) -> Result<Tensor> {
        let frames = source.len() / src_ch;
        let padded = frames.div_ceil(hop).max(1) * hop;
        let mut planar = vec![0f32; out_ch * padded];
        for f in 0..frames {
            for c in 0..out_ch {
                let sc = if c < src_ch { c } else { src_ch - 1 };
                planar[c * padded + f] = source[f * src_ch + sc];
            }
        }
        Ok(Tensor::from_vec(planar, (1, out_ch, padded), &self.device)?)
    }

    /// Resolve `(target_latent_len, window_start, window_end)` in latent frames for an edit task.
    fn edit_window(
        &self,
        params: &EditParams,
        src_frames: usize,
        l_src: usize,
    ) -> Result<(usize, usize, usize)> {
        let lps = self.config.vae.latents_per_second();
        let sample_rate = self.config.vae.sampling_rate as f64;
        let src_secs = src_frames as f64 / sample_rate;
        match params.task {
            EditTask::Extend => {
                let total_secs = params.region_end_secs.ok_or_else(|| {
                    AudioError::Msg(
                        "acestep extend: region end (the new total length in seconds) is required"
                            .into(),
                    )
                })? as f64;
                if total_secs <= src_secs {
                    return Err(AudioError::Msg(format!(
                        "acestep extend: new length {total_secs:.3}s must exceed the source length \
                         {src_secs:.3}s"
                    )));
                }
                let target_len = (total_secs * lps).ceil() as usize;
                let start = (params.region_start_secs.max(0.0) as f64 * lps).floor() as usize;
                let w0 = start.min(l_src).min(target_len - 1);
                Ok((target_len, w0, target_len))
            }
            EditTask::Inpaint | EditTask::Repaint => {
                let target_len = l_src.max(1);
                let w0 = (params.region_start_secs.max(0.0) as f64 * lps).floor() as usize;
                let w1 = match params.region_end_secs {
                    Some(e) => (e as f64 * lps).floor() as usize,
                    None => target_len,
                };
                let w0 = w0.min(target_len - 1);
                let w1 = w1.clamp(w0 + 1, target_len);
                Ok((target_len, w0, w1))
            }
        }
    }

    /// The `[1, target_len, acoustic]` chunk mask: `1` inside `[w0, w1)`, `0` elsewhere.
    fn window_mask(
        &self,
        target_len: usize,
        w0: usize,
        w1: usize,
        acoustic: usize,
    ) -> Result<Tensor> {
        let mut m = vec![0f32; target_len * acoustic];
        for f in w0..w1 {
            for slot in m[f * acoustic..(f + 1) * acoustic].iter_mut() {
                *slot = 1.0;
            }
        }
        Ok(Tensor::from_vec(
            m,
            (1, target_len, acoustic),
            &self.device,
        )?)
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

/// Equal-power crossfade half-width (frames) for an edit window: 20 ms at 48 kHz (960 frames),
/// capped at a quarter of the window so a short region still leaves a pure-generated centre.
fn crossfade_frames(f0: usize, f1: usize) -> usize {
    let win = f1.saturating_sub(f0);
    960.min(win / 4)
}

/// Generated-signal weight `∈ [0, 1]` at output frame `f` inside the edit window `[f0, f1)`: a
/// raised-cosine ramp `0 → 1` over the leading `xfade` frames (only when there is preceding source
/// to blend with, `f0 > 0`) and `1 → 0` over the trailing `xfade` frames (only when there is
/// following source, `f1 < out_frames`). The crossfade lives **entirely inside** the window, so
/// samples outside `[f0, f1)` stay pure source.
fn window_alpha(f: usize, f0: usize, f1: usize, out_frames: usize, xfade: usize) -> f32 {
    if xfade == 0 {
        return 1.0;
    }
    let mut a = 1.0f32;
    if f0 > 0 {
        let k = f - f0;
        if k < xfade {
            let t = (k as f32 + 0.5) / xfade as f32;
            a = a.min(0.5 - 0.5 * (std::f32::consts::PI * t).cos());
        }
    }
    if f1 < out_frames {
        let k = f1 - 1 - f;
        if k < xfade {
            let t = (k as f32 + 0.5) / xfade as f32;
            a = a.min(0.5 - 0.5 * (std::f32::consts::PI * t).cos());
        }
    }
    a
}

/// Stitch the regenerated window `[f0, f1)` (from `generated`) into the original `source`, keeping
/// every sample outside the window identical to the source (bit-preserved) and crossfading the
/// seam. `out_frames` per-channel frames of `gen_ch`-channel interleaved PCM are produced; a mono
/// source is replicated to every channel, and beyond the source length (extend tail) the source
/// contribution is silence.
#[allow(clippy::too_many_arguments)]
fn stitch(
    source: &[f32],
    src_ch: usize,
    generated: &[f32],
    gen_ch: usize,
    out_frames: usize,
    f0: usize,
    f1: usize,
    xfade: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; out_frames * gen_ch];
    let src_frames = source.len() / src_ch.max(1);
    let gen_frames = generated.len() / gen_ch.max(1);
    for f in 0..out_frames {
        let alpha = if f < f0 || f >= f1 {
            0.0 // pure source outside the window
        } else {
            window_alpha(f, f0, f1, out_frames, xfade)
        };
        for c in 0..gen_ch {
            let s = if f < src_frames {
                let sc = if c < src_ch { c } else { src_ch - 1 };
                source[f * src_ch + sc]
            } else {
                0.0
            };
            let g = if f < gen_frames {
                generated[f * gen_ch + c]
            } else {
                0.0
            };
            out[f * gen_ch + c] = alpha * g + (1.0 - alpha) * s;
        }
    }
    out
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

    // ---- Prompted-edit region + stitch math (sc-12847) --------------------------------------

    /// A stereo ramp source: ch0 = frame index, ch1 = −index, so any altered sample is obvious.
    fn ramp_source(frames: usize) -> Vec<f32> {
        let mut s = Vec::with_capacity(frames * 2);
        for f in 0..frames {
            s.push(f as f32);
            s.push(-(f as f32));
        }
        s
    }

    #[test]
    fn stitch_preserves_untouched_span_byte_exact() {
        // xfade = 0 isolates the pure keep/replace behaviour: outside [f0,f1) must equal the source
        // sample-for-sample; inside must equal the generated signal.
        let src = ramp_source(100);
        let gen = vec![9.0f32; 200]; // 100 stereo frames of a constant
        let out = stitch(&src, 2, &gen, 2, 100, 40, 60, 0);
        assert_eq!(out.len(), 200);
        for f in 0..100 {
            let (o0, o1) = (out[f * 2], out[f * 2 + 1]);
            if (40..60).contains(&f) {
                assert_eq!((o0, o1), (9.0, 9.0), "frame {f} inside window is generated");
            } else {
                assert_eq!(
                    (o0, o1),
                    (f as f32, -(f as f32)),
                    "frame {f} outside window is the untouched source"
                );
            }
        }
    }

    #[test]
    fn stitch_crossfade_stays_inside_the_window() {
        // With a crossfade, samples strictly OUTSIDE [f0,f1) are still exact source; the seam blend
        // lives only inside the window, and the window centre is pure generated.
        let src = ramp_source(100);
        let gen = vec![9.0f32; 200];
        let (f0, f1) = (40usize, 60usize);
        let xfade = crossfade_frames(f0, f1); // 960.min(20/4) = 5
        assert_eq!(xfade, 5);
        let out = stitch(&src, 2, &gen, 2, 100, f0, f1, xfade);
        for f in 0..100 {
            if f < f0 || f >= f1 {
                assert_eq!(
                    (out[f * 2], out[f * 2 + 1]),
                    (f as f32, -(f as f32)),
                    "frame {f} outside the window must be pristine source"
                );
            }
        }
        // Window centre (beyond both crossfade ramps) is pure generated.
        for f in (f0 + xfade)..(f1 - xfade) {
            assert_eq!((out[f * 2], out[f * 2 + 1]), (9.0, 9.0), "centre frame {f}");
        }
        // The very first window frame is mostly source (alpha near 0), not a hard jump to generated.
        let a0 = window_alpha(f0, f0, f1, 100, xfade);
        assert!(
            a0 < 0.25,
            "leading crossfade starts near source (alpha {a0})"
        );
    }

    #[test]
    fn stitch_extend_tail_is_generated_and_head_preserved() {
        // Extend: source covers [0,50); output is 80 frames; the head is preserved, the tail (the
        // window [50,80)) is generated. No source after the window ⇒ no trailing crossfade.
        let src = ramp_source(50);
        let gen = vec![7.0f32; 160]; // 80 stereo frames
        let out = stitch(&src, 2, &gen, 2, 80, 50, 80, 0);
        assert_eq!(out.len(), 160);
        for f in 0..50 {
            assert_eq!((out[f * 2], out[f * 2 + 1]), (f as f32, -(f as f32)));
        }
        for f in 50..80 {
            assert_eq!((out[f * 2], out[f * 2 + 1]), (7.0, 7.0));
        }
    }

    #[test]
    fn crossfade_width_is_capped_at_a_quarter_window() {
        assert_eq!(crossfade_frames(0, 100), 25); // 100/4
        assert_eq!(crossfade_frames(1000, 9000), 960); // 20 ms cap
        assert_eq!(crossfade_frames(10, 10), 0); // empty window
    }

    #[test]
    fn mono_source_replicates_to_stereo_untouched() {
        // A mono source (channels=1) fills both output channels outside the window.
        let src: Vec<f32> = (0..10).map(|f| f as f32).collect();
        let gen = vec![3.0f32; 20];
        let out = stitch(&src, 1, &gen, 2, 10, 4, 6, 0);
        for f in 0..10 {
            if (4..6).contains(&f) {
                assert_eq!((out[f * 2], out[f * 2 + 1]), (3.0, 3.0));
            } else {
                assert_eq!((out[f * 2], out[f * 2 + 1]), (f as f32, f as f32));
            }
        }
    }
}
