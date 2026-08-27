//! SANA text-to-image sampling pipeline (epic 11776, story sc-11780 — **the candle-gen half**).
//!
//! Composes the three already-merged native SANA components into one end-to-end prompt→image path,
//! the Windows/CUDA + Linux sibling of `mlx-gen-sana::pipeline` (mlx sc-8489):
//!
//! ```text
//!  prompt ─▶ SanaTextEncoder (sc-11779: CHI → gemma-2-2b-it last-hidden) ─▶ [1, 300, 2304]
//!         ─▶ SanaTransformer  (sc-11778: Linear-DiT trunk, velocity prediction) ─▶ [1, 32, h, w]
//!         ─▶ DcAeDecoder      (sc-11777: DC-AE f32c32 decode)                   ─▶ [1, 3, 1024, 1024]
//! ```
//!
//! driven by the **unified flow-matching scheduler** (epic 7114): the σ schedule is built by
//! `gen_core::sampling::build_flow_sigmas` and integrated by [`candle_gen::run_flow_sampler`] — the
//! SAME machinery the sibling candle flow-match families use (`candle-gen-z-image`, `candle-gen-sd3`).
//! No bespoke scheduler.
//!
//! ## Sampler / shift / timestep convention (mirrored from `mlx-gen-sana::pipeline`)
//!
//! * **Flow-match Euler, static shift 3.0.** `Sana_1600M_1024px_diffusers` ships a
//!   `FlowMatchEulerDiscreteScheduler` with `shift = 3.0` and `use_dynamic_shifting = false`, so the
//!   native schedule is `build_flow_sigmas(steps, ln(3))` (resolution-independent, `exp(mu) = shift`).
//!   An unset `scheduler` keeps that byte-exact; a curated epic-7114 name re-shapes σ over the same
//!   `mu = ln(3)` via [`candle_gen::resolve_flow_schedule`].
//! * **Timestep convention.** The unified sampler hands the predict closure `ms.timestep(σ) = σ`
//!   ([`TimestepConvention::Sigma`]); the SANA trunk embeds the diffusers-scale timestep `σ · 1000`
//!   (`num_train_timesteps`), so the closure scales it before the forward (identical to SD3's MMDiT).
//!   The Euler update itself stays in σ-space (`x += (σ_{t+1} − σ_t) · v`).
//!
//! ## CFG
//!
//! Base SANA is a **true-CFG** model. Each step runs the trunk TWICE — cond (prompt) + uncond
//! (negative/empty prompt) — and combines `pred = uncond + scale · (cond − uncond)` (diffusers
//! `SanaPipeline.__call__` default `guidance_scale = 4.5`). When `guidance_scale <= 1.0` the uncond
//! forward is skipped (CFG off), matching diffusers' `do_classifier_free_guidance = guidance_scale > 1.0`.
//!
//! ## DC-AE latent scaling
//!
//! diffusers `SanaPipeline` decodes `latents / vae.config.scaling_factor` (the DC-AE
//! `scaling_factor = 0.41407`, [`DcAeConfig::scaling_factor`]); [`DcAeDecoder::decode`] expects the
//! **already-unscaled** latent, so the division is applied here before decode. The decoder emits NCHW
//! `[1, 3, H, W]` in `[-1, 1]`, mapped to RGB8 (`clip(x·0.5 + 0.5)·255`).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::sampling::{build_flow_sigmas, TimestepConvention};
use candle_gen::gen_core::{CancelFlag, Image, PreviewSink, Progress};
use candle_gen::{
    resolve_flow_schedule, run_flow_sampler, run_scm_sampler, run_scm_sampler_from, CandleError,
    Result, ScmScheduler, Weights,
};
use candle_gen_pid::{Gemma2, Gemma2Config};

use crate::config::{DcAeConfig, SanaTransformerConfig};
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// DC-AE f32c32 latent channel count (the SANA trunk's `out_channels`).
pub const LATENT_CHANNELS: usize = 32;
/// DC-AE deep-compression spatial downsample (latent edge is image/32).
pub const SPATIAL_SCALE: u32 = 32;
/// diffusers `num_train_timesteps` — the SANA trunk embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SANA-1.6B static flow-match shift (`scheduler_config.json` `shift = 3.0`, no dynamic shifting).
pub const SCHEDULE_SHIFT: f32 = 3.0;
/// diffusers `SanaPipeline` default `num_inference_steps`.
pub const DEFAULT_STEPS: usize = 20;
/// diffusers `SanaPipeline` default `guidance_scale`.
pub const DEFAULT_GUIDANCE: f32 = 4.5;
/// Shared SANA img2img strength when neither the reference nor request supplies one.
pub const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.5;

/// Resolve the product img2img strength precedence. Explicit zero is preserved as txt2img.
pub fn resolve_strength(reference: Option<f32>, request: Option<f32>) -> f32 {
    reference.or(request).unwrap_or(DEFAULT_IMG2IMG_STRENGTH)
}

/// Resolve the product img2img start-step convention: positive strength selects
/// `max(1, floor(steps * strength))`, clamped to the schedule; non-positive is txt2img.
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => ((num_steps as f32 * s.clamp(0.0, 1.0)) as usize).max(1),
        _ => 0,
    }
}

fn resolve_init_start(init_image: Option<&Image>, steps: usize, strength: Option<f32>) -> usize {
    init_image
        .map(|_| init_time_step(steps, strength))
        .unwrap_or(0)
}

fn blend_flow_init(clean: &Tensor, noise: &Tensor, sigmas: &[f32], start: usize) -> Result<Tensor> {
    let sigma = *sigmas.get(start).ok_or_else(|| {
        CandleError::Msg(format!(
            "sana img2img: start step {start} out of range for {}-element schedule",
            sigmas.len()
        ))
    })? as f64;
    Ok((clean.affine(1.0 - sigma, 0.0)? + noise.affine(sigma, 0.0)?)?)
}

fn renoise_sprint_init(
    clean: &Tensor,
    noise: &Tensor,
    scheduler: &ScmScheduler,
    start: usize,
) -> Result<Tensor> {
    let t = *scheduler.timesteps.get(start).ok_or_else(|| {
        CandleError::Msg(format!(
            "sana sprint img2img: start step {start} out of range for {}-element angle schedule",
            scheduler.timesteps.len()
        ))
    })?;
    let sd = scheduler.sigma_data as f64;
    Ok(clean
        .affine(sd * t.cos() as f64, 0.0)?
        .add(&noise.affine(sd * t.sin() as f64, 0.0)?)?)
}

/// Seeded txt2img latent noise — shape `[1, 32, height/32, width/32]`, f32. diffusers
/// `randn_tensor([B, 32, H/32, W/32])`; we draw f32 on CPU (launch-portable, sc-3673) then move to
/// `device`. (`init_noise_sigma = 1.0` for flow-match, so the latent is the raw normal draw.)
pub fn create_noise(device: &Device, seed: u64, width: u32, height: u32) -> Result<Tensor> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (lh, lw) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    Ok(candle_gen::seeded_noise_nchw(
        &mut rng,
        LATENT_CHANNELS,
        lh,
        lw,
        device,
    )?)
}

/// Build the descending flow-match σ schedule for SANA (static shift 3.0), honoring a curated
/// epic-7114 `scheduler` name (which re-shapes σ over the same `mu = ln(shift)`). An unset / unknown /
/// native-aliased name returns the byte-exact `build_flow_sigmas(steps, ln(3))` schedule.
pub fn sana_sigmas(scheduler_name: Option<&str>, steps: usize) -> Vec<f32> {
    let mu = SCHEDULE_SHIFT.ln();
    let native = build_flow_sigmas(steps, mu);
    resolve_flow_schedule(scheduler_name, mu, steps, &native)
}

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the SANA trunk twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The trunk timestep is `σ·1000`. When `guidance_scale`
/// is `<= 1.0` the uncond branch is skipped (CFG off, one forward per step; diffusers parity).
///
/// `preview` is the base route's per-step latent preview hook (epic 16948, sc-16959) — the **single**
/// [`run_flow_sampler`] site in this crate, and the whole of the `sana_1600m` lane's wiring. It is
/// taken by **reference, not as an `Option`**, so a caller cannot take this lane dark by editing one
/// argument, which is invisible to `candle-gen-catalog`'s route inventory because that classifies the
/// driver argument one hop further in. Handing a hook over an inert [`PreviewSink`] is the
/// way to run this without previews, and it is byte-identical to a run without the seam.
///
/// The preview projects the **combined** running latent, never a fused unconditional half: the CFG
/// pair is two separate trunk forwards inside `predict` and is blended before the solver ever sees it,
/// so no `[2, …]` batch is ever the tensor handed to the hook.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &SanaTransformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    guidance_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> Result<Tensor> {
    denoise_cfg_from(
        transformer,
        sigmas,
        sampler_name,
        0,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        device,
        cancel,
        on_progress,
        preview,
    )
}

/// Base SANA flow-match denoise starting at `start_step` in the supplied sigma schedule.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_from(
    transformer: &SanaTransformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    guidance_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> Result<Tensor> {
    denoise_cfg_from_memory(
        transformer,
        sigmas,
        sampler_name,
        start_step,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        device,
        cancel,
        on_progress,
        preview,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_from_memory(
    transformer: &SanaTransformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    guidance_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
    memory: Option<candle_gen::gen_core::GenerationMemory>,
) -> Result<Tensor> {
    let attention_budget = memory
        .filter(|memory| memory.chunk_attention)
        .and_then(|memory| memory.attention_chunk_size)
        .map(|value| value as usize);
    let predict = |x: &Tensor, timestep: f32| -> Result<Tensor> {
        // The unified flow sampler hands `timestep = σ`; the SANA trunk embeds `σ·1000`.
        let t = Tensor::from_vec(vec![timestep * NUM_TRAIN_TIMESTEPS], (1,), device)?;
        let pred_cond =
            transformer.forward_with_guidance_memory(x, cond, &t, None, attention_budget)?;
        match uncond {
            Some(uc) if guidance_scale > 1.0 => {
                let pred_uncond =
                    transformer.forward_with_guidance_memory(x, uc, &t, None, attention_budget)?;
                // pred = uncond + scale·(cond − uncond).
                let delta = (&pred_cond - &pred_uncond)?;
                Ok((&pred_uncond + (delta * guidance_scale as f64)?)?)
            }
            _ => Ok(pred_cond),
        }
    };
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        &sigmas[start_step.min(sigmas.len().saturating_sub(1))..],
        latents,
        seed,
        cancel,
        on_progress,
        Some(preview),
        predict,
    )
}

/// DC-AE-decode the final `[1, 32, H/32, W/32]` latent → an RGB8 [`Image`]. diffusers `SanaPipeline`
/// divides by `vae.config.scaling_factor` before decode; the decoder emits NCHW `[1, 3, H, W]` in
/// `[-1, 1]`, mapped to `[0, 255]` u8.
pub fn decode_to_image(decoder: &DcAeDecoder, cfg: &DcAeConfig, latents: &Tensor) -> Result<Image> {
    decode_to_image_memory(decoder, cfg, latents, None)
}

pub fn decode_to_image_memory(
    decoder: &DcAeDecoder,
    cfg: &DcAeConfig,
    latents: &Tensor,
    memory: Option<candle_gen::gen_core::GenerationMemory>,
) -> Result<Image> {
    // diffusers: latents / scaling_factor.
    let unscaled = (latents / cfg.scaling_factor as f64)?;
    // VRAM-fit gate (sc-11804): single-pass on a card with headroom (the Blackwell target), tiled tail
    // on a small card whose f32 decode peak (~17.7 GB at 1024²) would OOM. Byte-identical to `decode`
    // when it fits; seam-free when it tiles.
    let decoded = match memory {
        Some(memory) => decoder.decode_with(&unscaled, memory.tile_vae_decode)?,
        None => decoder.decode_fit(&unscaled)?,
    }; // [1, 3, H, W] NCHW, f32 in [-1, 1]
    let rgb = (((decoded * 0.5)? + 0.5)?.clamp(0f32, 1f32)? * 255.0)?;
    let rgb = candle_gen::round_rgb8(&rgb)?
        .i(0)?
        .to_device(&Device::Cpu)?; // [3, H, W]
    let (c, h, w) = rgb.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = rgb.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// The composed SANA text-to-image pipeline: text encoder + trunk + DC-AE decoder, with the DC-AE
/// config (for the latent `scaling_factor`). A clean `generate` entrypoint mirroring the sibling candle
/// flow-match pipelines. Base SANA-1.6B only (true-CFG flow-match Euler); the CFG-free SCM/Sprint
/// distilled variant is [`SanaSprintPipeline`], a SEPARATE entrypoint (sc-11781) — the base flow here
/// is byte-unchanged.
pub struct SanaPipeline {
    text_encoder: SanaTextEncoder,
    transformer: SanaTransformer,
    encoder: DcAeEncoder,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
}

/// One text-to-image request for [`SanaPipeline::generate`]. `None` fields fall back to the diffusers
/// `SanaPipeline` defaults (`steps = 20`, `guidance = 4.5`, `seed = 0`, empty negative prompt).
#[derive(Clone, Debug)]
pub struct SanaGenerateRequest<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub height: u32,
    pub width: u32,
    pub steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    /// Optional curated epic-7114 sampler name (e.g. `"euler"`, `"dpmpp_2m"`); `None` = native Euler.
    pub sampler: Option<&'a str>,
    /// Optional curated epic-7114 scheduler name re-shaping σ over the same `mu = ln(shift)`.
    pub scheduler: Option<&'a str>,
    /// Optional img2img source. A positive `strength` encodes and renoises it; zero is txt2img.
    pub init_image: Option<&'a Image>,
    pub strength: Option<f32>,
}

/// Seed-independent base-SANA prompt conditioning, prepared once for a whole image batch.
pub(crate) struct SanaConditioning {
    cond: Tensor,
    uncond: Option<Tensor>,
}

impl<'a> SanaGenerateRequest<'a> {
    /// A 1024px request for `prompt` with all diffusers defaults.
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            negative_prompt: None,
            height: 1024,
            width: 1024,
            steps: None,
            guidance_scale: None,
            seed: None,
            sampler: None,
            scheduler: None,
            init_image: None,
            strength: None,
        }
    }
}

impl SanaPipeline {
    /// Compose the base SANA-1.6B pipeline from its three already-constructed components plus the
    /// DC-AE config (used for the latent `scaling_factor`).
    pub fn new(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            encoder,
            decoder,
            dc_ae_cfg,
        }
    }

    /// Assemble the pipeline from an `Efficient-Large-Model/Sana_1600M_1024px_diffusers`-shaped
    /// snapshot directory (the whole-repo HF snapshot: `transformer/ vae/ text_encoder/ tokenizer/`).
    ///
    /// Everything runs **f32** (the parity precision + the dense-GEMM-safe path, matching the DC-AE
    /// decoder and the trunk's f32 forward): the transformer, DC-AE, and gemma-2-2b-it caption encoder
    /// are all coerced to f32 on load. The component file selection ([`resolve_component_files`]) picks
    /// the fp32 (non-`fp16`) safetensors and tolerates both single-file and sharded checkpoints, so the
    /// raw diffusers tree loads without a curated allow-list.
    pub fn from_diffusers_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let trunk_files = resolve_component_files(&root.join("transformer"))?;
        let trunk_w = Weights::from_files(&trunk_files, device, DType::F32)?;
        let trunk = SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_1600m())?;

        let dcfg = DcAeConfig::sana_f32c32();
        let vae_files = resolve_component_files(&root.join("vae"))?;
        let vae_w = Weights::from_files(&vae_files, device, DType::F32)?;
        let encoder = DcAeEncoder::from_weights(&vae_w, &dcfg)?;
        let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;

        let te = load_text_encoder(root, device)?;

        Ok(Self::new(te, trunk, encoder, decoder, dcfg))
    }

    /// Run the full prompt→image pipeline with caller-supplied cancellation + progress (the seam the
    /// gen-core `Generator` adapter wires into the contract). Encodes the prompt (and the negative
    /// prompt when CFG is active) ONCE, seeds the DC-AE latent, runs the flow-match Euler denoise over
    /// the SANA trunk with true CFG, then DC-AE-decodes to an RGB8 [`Image`].
    pub fn generate_with(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
    ) -> Result<Image> {
        let guidance = req.guidance_scale.unwrap_or(DEFAULT_GUIDANCE);
        let conditioning = self.encode_conditioning(req, guidance)?;
        self.generate_with_conditioning(req, &conditioning, device, cancel, on_progress, preview)
    }

    /// Encode the seed-independent prompt inputs once for a whole `count` batch.
    pub(crate) fn encode_conditioning(
        &self,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
    ) -> Result<SanaConditioning> {
        let cond = self.text_encoder.encode(req.prompt)?;
        let uncond = if guidance > 1.0 {
            Some(
                self.text_encoder
                    .encode(req.negative_prompt.unwrap_or(""))?,
            )
        } else {
            None
        };
        Ok(SanaConditioning { cond, uncond })
    }

    /// Encode the seed-independent base-SANA img2img reference once for a whole `count` batch.
    pub(crate) fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &CancelFlag,
    ) -> Result<Option<Tensor>> {
        let start_step = resolve_init_start(
            req.init_image,
            req.steps.unwrap_or(DEFAULT_STEPS),
            req.strength,
        );
        if start_step == 0 {
            return Ok(None);
        }
        let image = req.init_image.ok_or_else(|| {
            CandleError::Msg("SANA positive img2img start requires an init image".into())
        })?;
        Ok(Some(encode_init_latents(
            &self.encoder,
            &self.dc_ae_cfg,
            image,
            req.width,
            req.height,
            device,
            cancel,
        )?))
    }

    /// Run the seed-dependent sampling and decode tail with precomputed conditioning.
    pub(crate) fn generate_with_conditioning(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &SanaConditioning,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
    ) -> Result<Image> {
        self.generate_with_conditioning_memory(
            req,
            conditioning,
            device,
            cancel,
            on_progress,
            preview,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_conditioning_memory(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &SanaConditioning,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<candle_gen::gen_core::GenerationMemory>,
    ) -> Result<Image> {
        let prepared_reference = self.prepare_reference(req, device, cancel)?;
        self.generate_with_conditioning_and_reference_memory(
            req,
            conditioning,
            prepared_reference.as_ref(),
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_conditioning_and_reference_memory(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &SanaConditioning,
        prepared_reference: Option<&Tensor>,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<candle_gen::gen_core::GenerationMemory>,
    ) -> Result<Image> {
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        let guidance = req.guidance_scale.unwrap_or(DEFAULT_GUIDANCE);
        let seed = req.seed.unwrap_or(0);

        // Static shift=3.0 schedule (scheduler_config.json), resolution-independent. An unset scheduler
        // keeps it byte-exact; a curated name re-shapes σ over the same mu=ln(3).
        let sigmas = sana_sigmas(req.scheduler, steps);

        let noise = create_noise(device, seed, req.width, req.height)?;
        let start_step = resolve_init_start(req.init_image, steps, req.strength);
        let latents = if start_step > 0 {
            let clean = prepared_reference.ok_or_else(|| {
                CandleError::Msg("SANA img2img denoise requires a prepared reference latent".into())
            })?;
            blend_flow_init(clean, &noise, &sigmas, start_step)?
        } else {
            noise
        };
        let latents = denoise_cfg_from_memory(
            &self.transformer,
            &sigmas,
            req.sampler,
            start_step,
            seed,
            latents,
            &conditioning.cond,
            conditioning.uncond.as_ref(),
            guidance,
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )?;
        on_progress(Progress::Decoding);
        decode_to_image_memory(&self.decoder, &self.dc_ae_cfg, &latents, memory)
    }

    /// Convenience [`SanaPipeline::generate_with`] with a no-op cancel + progress (examples / tests).
    ///
    /// Deliberately preview-inert: it takes no [`PreviewSink`], so it hands the denoise a hook over a
    /// default (inert) one. That costs one `is_active()` check per evaluation and is byte-identical to
    /// a run without the seam. The **registered** `Generator` lane is [`crate::model`], which builds
    /// its hook over the request's real sink.
    pub fn generate(&self, req: &SanaGenerateRequest<'_>, device: &Device) -> Result<Image> {
        let cancel = CancelFlag::default();
        let mut noop = |_: Progress| {};
        let inert = PreviewSink::default();
        let preview = crate::preview::base_hook(&inert);
        self.generate_with(req, device, &cancel, &mut noop, &preview)
    }
}

fn revalidate_before_load(
    check: &mut impl FnMut() -> candle_gen::gen_core::Result<()>,
) -> Result<()> {
    check().map_err(|error| CandleError::Msg(error.to_string()))
}

fn load_vae_encoder(root: &Path, device: &Device, cfg: &DcAeConfig) -> Result<DcAeEncoder> {
    let files = resolve_component_files(&root.join("vae"))?;
    let weights = Weights::from_files_filtered(&files, device, DType::F32, &["encoder."])?;
    DcAeEncoder::from_weights(&weights, cfg)
}

fn load_vae_decoder(root: &Path, device: &Device, cfg: &DcAeConfig) -> Result<DcAeDecoder> {
    let files = resolve_component_files(&root.join("vae"))?;
    let weights = Weights::from_files_filtered(&files, device, DType::F32, &["decoder."])?;
    DcAeDecoder::from_weights(&weights, cfg.clone())
}

fn load_staged_transformer(
    root: &Path,
    device: &Device,
    cfg: SanaTransformerConfig,
    memory: candle_gen::gen_core::GenerationMemory,
) -> Result<SanaTransformer> {
    let files = resolve_component_files(&root.join("transformer"))?;
    if memory.stream_transformer_blocks {
        let window = memory.transformer_window_size.unwrap_or(1) as usize;
        SanaTransformer::from_files_windowed(&files, cfg, window, device)
    } else {
        let weights = Weights::from_files(&files, device, DType::F32)?;
        SanaTransformer::from_weights(&weights, cfg)
    }
}

/// Execute true per-request Base phase residency: Gemma conditioning, optional DC-AE encode,
/// Linear-DiT denoise, then DC-AE decode. Each load is preceded by immutable seal revalidation and
/// each component is synchronized and dropped before the next is opened.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_base_staged(
    root: &Path,
    req: &SanaGenerateRequest<'_>,
    seeds: &[u64],
    memory: candle_gen::gen_core::GenerationMemory,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
    mut check: impl FnMut() -> candle_gen::gen_core::Result<()>,
) -> Result<Vec<Image>> {
    revalidate_before_load(&mut check)?;
    let text = load_text_encoder(root, device)?;
    let guidance = req.guidance_scale.unwrap_or(DEFAULT_GUIDANCE);
    let conditioning = SanaConditioning {
        cond: text.encode(req.prompt)?,
        uncond: if guidance > 1.0 {
            Some(text.encode(req.negative_prompt.unwrap_or(""))?)
        } else {
            None
        },
    };
    device.synchronize()?;
    drop(text);
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }

    let cfg = DcAeConfig::sana_f32c32();
    let steps = req.steps.unwrap_or(DEFAULT_STEPS);
    let start_step = resolve_init_start(req.init_image, steps, req.strength);
    let clean = if start_step > 0 {
        revalidate_before_load(&mut check)?;
        let encoder = load_vae_encoder(root, device, &cfg)?;
        let clean = encode_init_latents(
            &encoder,
            &cfg,
            req.init_image.expect("positive start requires init image"),
            req.width,
            req.height,
            device,
            cancel,
        )?;
        device.synchronize()?;
        drop(encoder);
        Some(clean)
    } else {
        None
    };

    revalidate_before_load(&mut check)?;
    let transformer =
        load_staged_transformer(root, device, SanaTransformerConfig::sana_1600m(), memory)?;
    let sigmas = sana_sigmas(req.scheduler, steps);
    let mut latents = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let noise = create_noise(device, *seed, req.width, req.height)?;
        let initial = match &clean {
            Some(clean) => blend_flow_init(clean, &noise, &sigmas, start_step)?,
            None => noise,
        };
        latents.push(denoise_cfg_from_memory(
            &transformer,
            &sigmas,
            req.sampler,
            start_step,
            *seed,
            initial,
            &conditioning.cond,
            conditioning.uncond.as_ref(),
            guidance,
            device,
            cancel,
            on_progress,
            preview,
            Some(memory),
        )?);
    }
    device.synchronize()?;
    drop(transformer);
    drop(conditioning);

    revalidate_before_load(&mut check)?;
    let decoder = load_vae_decoder(root, device, &cfg)?;
    let mut images = Vec::with_capacity(latents.len());
    for latent in latents {
        on_progress(Progress::Decoding);
        images.push(decode_to_image_memory(
            &decoder,
            &cfg,
            &latent,
            Some(memory),
        )?);
    }
    device.synchronize()?;
    Ok(images)
}

/// Load the gemma-2-2b-it caption encoder from a diffusers SANA snapshot. The gemma **weights** live in
/// `text_encoder/` (fp32 shards) and the gemma **tokenizer** in `tokenizer/tokenizer.json` (the
/// `Sana_1600M_1024px_diffusers` layout), so we build [`SanaTextEncoder`] directly rather than via
/// [`SanaTextEncoder::from_snapshot`] (which expects the tokenizer co-located under the weights dir).
///
/// Public so a harness can encode a prompt and **drop the ~10 GB f32 encoder** before materializing the
/// trunk — the sc-11045 NVFP4 validation builds several trunk variants against one set of conditioning
/// embeddings and cannot afford to hold the encoder resident alongside them.
pub fn load_text_encoder(root: &Path, device: &Device) -> Result<SanaTextEncoder> {
    let te_files = resolve_component_files(&root.join("text_encoder"))?;
    let gw = Weights::from_files(&te_files, device, DType::F32)?;
    // The diffusers SANA `text_encoder/` saves the Gemma2Model UN-prefixed (`embed_tokens.weight`,
    // `layers.0.…`); PiD's `SceneWorks/gemma-2-2b-it` mirror wraps it under `model.`. Pick whichever
    // this snapshot uses so both layouts load.
    let prefix = if gw.contains("embed_tokens.weight") {
        ""
    } else {
        "model."
    };
    let gemma = Gemma2::from_weights(&gw, prefix, &Gemma2Config::gemma_2_2b())?;
    // Prefer the sibling `tokenizer/` dir (the diffusers layout); fall back to a co-located file.
    let tok = {
        let t1 = root.join("tokenizer").join("tokenizer.json");
        if t1.is_file() {
            t1
        } else {
            root.join("text_encoder").join("tokenizer.json")
        }
    };
    SanaTextEncoder::new(gemma, tok)
}

/// Whether a safetensors filename is a shard of a multi-file checkpoint (`…-00001-of-00002.safetensors`).
fn is_shard(name: &str) -> bool {
    name.contains("-of-")
}

/// Select the usable `.safetensors` files in a diffusers component dir. The raw
/// `Sana_1600M_1024px_diffusers` tree ships BOTH fp32 and `fp16` copies, and the transformer ships a
/// single-file AND a sharded fp32 copy — loading all of them would collide on duplicate keys. Policy:
///
/// - drop any `fp16` copy (we run f32 everywhere; `Weights::from_files` coerces on load anyway);
/// - if the dir holds a sharded checkpoint (`…-of-…`), use the shard set (the diffusers-native fp32
///   split); otherwise use the remaining single file(s).
///
/// Deterministically sorted, and a hard error if nothing usable is found.
pub fn resolve_component_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let rd = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("read component dir {dir:?}: {e}")))?;
    let mut candidates = Vec::new();
    for entry in rd {
        let path = entry
            .map_err(|e| CandleError::Msg(format!("read entry in {dir:?}: {e}")))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        // Drop fp16 copies (we load everything f32; keeps single-vs-sharded selection unambiguous).
        if name.contains("fp16") {
            continue;
        }
        candidates.push(path);
    }
    let sharded: Vec<PathBuf> = candidates
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(is_shard)
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    let mut chosen = if sharded.is_empty() {
        candidates
    } else {
        sharded
    };
    if chosen.is_empty() {
        return Err(CandleError::Msg(format!(
            "no usable (non-fp16) .safetensors found in {dir:?}"
        )));
    }
    chosen.sort();
    Ok(chosen)
}

// =================================================================================================
// SANA-Sprint — continuous-time-consistency (SCM/TrigFlow), CFG-free, 1–4 step (sc-11781, epic 11776;
// the candle sibling of `mlx-gen-sana`'s sc-8490). A SEPARATE pipeline + entrypoint: the base
// [`SanaPipeline`] flow above is byte-unchanged; only the trunk-call (embedded guidance, no CFG uncond
// pass) and the sampler (the SCM loop in `candle_gen::run_scm_sampler`) differ. TE + DC-AE are reused.
// =================================================================================================

/// diffusers `SanaSprintPipeline` default `num_inference_steps` (the Sprint operating band is 1–4).
pub const SPRINT_DEFAULT_STEPS: usize = 2;
/// diffusers `SanaSprintPipeline` default `guidance_scale` (embedded, NOT classifier-free).
pub const SPRINT_DEFAULT_GUIDANCE: f32 = 4.5;

/// One SCM (TrigFlow continuous-time-consistency) denoise — the **CFG-free, few-step** SANA-Sprint
/// loop. Builds the embedded guidance scalar (`guidance_scale · guidance_embeds_scale`, a `[1]` tensor
/// fed to the trunk's guidance embedder — NOT classifier-free guidance) and runs
/// [`candle_gen::run_scm_sampler`] with a single-trunk-forward-per-step `predict` closure. The SCM
/// scheduler math (angle schedule, trigflow recombination, renoise) lives in the shared sampler; this
/// only wires the trunk call.
///
/// `preview` is the Sprint route's per-step latent preview hook (epic 16948, sc-16959) — the **single**
/// [`run_scm_sampler`] site in this crate, and the whole of the `sana_sprint_1600m` lane's wiring.
/// Taken by **reference, not as an `Option`**, for the reason spelled out on [`denoise_cfg`].
///
/// Two things differ from the base lane and both are the hook's concern rather than this function's.
/// The SCM loop has **no σ schedule** — it walks `ScmScheduler` angle timesteps — so the driver keys
/// frames on the step index, and a 1-step schedule is a real request shape rather than an edge case.
/// And the loop hands the hook a latent **pre-scaled by `σ_data`**, which the Sprint projector
/// ([`crate::preview::project_sprint_latents`], through the crate-internal `sprint_hook`) divides back
/// out. There is no unconditional half here at all: Sprint's guidance is an embedded scalar, not a
/// cond/uncond pair.
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> Result<Tensor> {
    denoise_sprint_memory(
        transformer,
        scheduler,
        seed,
        latents,
        cond,
        guidance_scale,
        guidance_embeds_scale,
        device,
        cancel,
        on_progress,
        preview,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_memory(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
    memory: Option<candle_gen::gen_core::GenerationMemory>,
) -> Result<Tensor> {
    // The embedded guidance scalar (CFG-free): guidance_scale · guidance_embeds_scale, a [1] tensor
    // fed to the trunk's guidance embedder. Constant across steps.
    let guidance = Tensor::from_vec(vec![guidance_scale * guidance_embeds_scale], (1,), device)?;
    let predict = |lat_in: &Tensor, scm_t: f32| -> Result<Tensor> {
        // The trunk embeds `scm_t` as its timestep (NOT the raw angle) + the embedded guidance scalar;
        // ONE forward per step (Sprint is CFG-free — no uncond branch).
        let t = Tensor::from_vec(vec![scm_t], (1,), device)?;
        let budget = memory
            .filter(|memory| memory.chunk_attention)
            .and_then(|memory| memory.attention_chunk_size)
            .map(|value| value as usize);
        transformer
            .forward_with_guidance_memory(lat_in, cond, &t, Some(&guidance), budget)
            .map_err(CandleError::from)
    };
    run_scm_sampler(
        scheduler,
        latents,
        seed,
        cancel,
        on_progress,
        Some(preview),
        predict,
    )
}

/// The composed **SANA-Sprint** text-to-image pipeline (CFG-free SCM/TrigFlow few-step, sc-11781) — a
/// SEPARATE type from the base [`SanaPipeline`] so the base flow stays byte-unchanged. Same three
/// components (gemma-2-2b-it TE + Linear-DiT trunk + DC-AE decoder), but the trunk is loaded with
/// [`SanaTransformerConfig::sana_sprint_1600m`] (its guidance embedder + rms-norm-across-heads are
/// config-gated) and driven by the CFG-free SCM few-step loop.
pub struct SanaSprintPipeline {
    text_encoder: SanaTextEncoder,
    transformer: SanaTransformer,
    encoder: DcAeEncoder,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
    /// The trunk config's `guidance_embeds_scale` (`0.1`), pre-multiplied into the guidance scalar.
    guidance_embeds_scale: f32,
}

impl SanaSprintPipeline {
    /// Compose the Sprint pipeline from its already-constructed components + the DC-AE config and the
    /// trunk's `guidance_embeds_scale`.
    pub fn new(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
        guidance_embeds_scale: f32,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            encoder,
            decoder,
            dc_ae_cfg,
            guidance_embeds_scale,
        }
    }

    /// Assemble the Sprint pipeline from an `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`-
    /// shaped snapshot directory. Identical layout to the base
    /// ([`SanaPipeline::from_diffusers_snapshot`]) — `transformer/ vae/ text_encoder/ tokenizer/` — but
    /// the transformer loads the Sprint config (so the guidance embedder + qk-norm weights are
    /// required). Everything runs f32.
    pub fn from_diffusers_snapshot(root: &Path, device: &Device) -> Result<Self> {
        let trunk_cfg = SanaTransformerConfig::sana_sprint_1600m();
        let guidance_embeds_scale = trunk_cfg.guidance_embeds_scale;
        let trunk_files = resolve_component_files(&root.join("transformer"))?;
        let trunk_w = Weights::from_files(&trunk_files, device, DType::F32)?;
        let trunk = SanaTransformer::from_weights(&trunk_w, trunk_cfg)?;

        let dcfg = DcAeConfig::sana_f32c32();
        let vae_files = resolve_component_files(&root.join("vae"))?;
        let vae_w = Weights::from_files(&vae_files, device, DType::F32)?;
        let encoder = DcAeEncoder::from_weights(&vae_w, &dcfg)?;
        let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;

        let te = load_text_encoder(root, device)?;

        Ok(Self::new(
            te,
            trunk,
            encoder,
            decoder,
            dcfg,
            guidance_embeds_scale,
        ))
    }

    /// Run the full Sprint prompt→image path. Encodes the prompt ONCE (no uncond — Sprint is
    /// CFG-free), seeds the DC-AE latent, runs [`denoise_sprint`] over an [`ScmScheduler`] (default 2
    /// steps, embedded guidance 4.5), then DC-AE-decodes. The negative prompt / curated sampler +
    /// scheduler knobs are inapplicable to the SCM loop and ignored.
    pub fn generate_with(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
    ) -> Result<Image> {
        let cond = self.encode_conditioning(req.prompt)?;
        self.generate_with_conditioning(req, &cond, device, cancel, on_progress, preview)
    }

    /// Encode the seed-independent Sprint prompt once for a whole `count` batch.
    pub(crate) fn encode_conditioning(&self, prompt: &str) -> Result<Tensor> {
        self.text_encoder.encode(prompt)
    }

    /// Encode the seed-independent Sprint img2img reference once for a whole `count` batch.
    pub(crate) fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &CancelFlag,
    ) -> Result<Option<Tensor>> {
        let start_step = resolve_init_start(
            req.init_image,
            req.steps.unwrap_or(SPRINT_DEFAULT_STEPS),
            req.strength,
        );
        if start_step == 0 {
            return Ok(None);
        }
        let image = req.init_image.ok_or_else(|| {
            CandleError::Msg("SANA-Sprint positive img2img start requires an init image".into())
        })?;
        Ok(Some(encode_init_latents(
            &self.encoder,
            &self.dc_ae_cfg,
            image,
            req.width,
            req.height,
            device,
            cancel,
        )?))
    }

    /// Run the seed-dependent Sprint sampling and decode tail with precomputed conditioning.
    pub(crate) fn generate_with_conditioning(
        &self,
        req: &SanaGenerateRequest<'_>,
        cond: &Tensor,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
    ) -> Result<Image> {
        self.generate_with_conditioning_memory(
            req,
            cond,
            device,
            cancel,
            on_progress,
            preview,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_conditioning_memory(
        &self,
        req: &SanaGenerateRequest<'_>,
        cond: &Tensor,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<candle_gen::gen_core::GenerationMemory>,
    ) -> Result<Image> {
        let prepared_reference = self.prepare_reference(req, device, cancel)?;
        self.generate_with_conditioning_and_reference_memory(
            req,
            cond,
            prepared_reference.as_ref(),
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_conditioning_and_reference_memory(
        &self,
        req: &SanaGenerateRequest<'_>,
        cond: &Tensor,
        prepared_reference: Option<&Tensor>,
        device: &Device,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<candle_gen::gen_core::GenerationMemory>,
    ) -> Result<Image> {
        let steps = req.steps.unwrap_or(SPRINT_DEFAULT_STEPS);
        let guidance = req.guidance_scale.unwrap_or(SPRINT_DEFAULT_GUIDANCE);
        let seed = req.seed.unwrap_or(0);

        let scheduler = ScmScheduler::new(steps);
        let noise = create_noise(device, seed, req.width, req.height)?;
        let start_step = resolve_init_start(req.init_image, steps, req.strength);
        let latents = if start_step > 0 {
            let clean = prepared_reference.ok_or_else(|| {
                CandleError::Msg(
                    "SANA-Sprint img2img denoise requires a prepared reference latent".into(),
                )
            })?;
            renoise_sprint_init(clean, &noise, &scheduler, start_step)?
        } else {
            noise.affine(scheduler.sigma_data as f64, 0.0)?
        };
        let latents = denoise_sprint_from_memory(
            &self.transformer,
            &scheduler,
            start_step,
            seed,
            latents,
            cond,
            guidance,
            self.guidance_embeds_scale,
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )?;
        on_progress(Progress::Decoding);
        decode_to_image_memory(&self.decoder, &self.dc_ae_cfg, &latents, memory)
    }

    /// Convenience [`SanaSprintPipeline::generate_with`] with a no-op cancel + progress.
    ///
    /// Preview-inert for the same reason [`SanaPipeline::generate`] is, and over the **Sprint** hook —
    /// the two are not interchangeable, because the two routes carry different fits.
    pub fn generate(&self, req: &SanaGenerateRequest<'_>, device: &Device) -> Result<Image> {
        let cancel = CancelFlag::default();
        let mut noop = |_: Progress| {};
        let inert = PreviewSink::default();
        let preview = crate::preview::sprint_hook(&inert);
        self.generate_with(req, device, &cancel, &mut noop, &preview)
    }
}

/// Sprint's phase-separated twin. It preserves the CFG-free single-forward SCM identity and keeps
/// the embedded guidance scalar in the denoise phase; no unconditional caption is created.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_sprint_staged(
    root: &Path,
    req: &SanaGenerateRequest<'_>,
    seeds: &[u64],
    memory: candle_gen::gen_core::GenerationMemory,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
    mut check: impl FnMut() -> candle_gen::gen_core::Result<()>,
) -> Result<Vec<Image>> {
    revalidate_before_load(&mut check)?;
    let text = load_text_encoder(root, device)?;
    let conditioning = text.encode(req.prompt)?;
    device.synchronize()?;
    drop(text);
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }

    let cfg = DcAeConfig::sana_f32c32();
    let steps = req.steps.unwrap_or(SPRINT_DEFAULT_STEPS);
    let scheduler = ScmScheduler::new(steps);
    let start_step = resolve_init_start(req.init_image, steps, req.strength);
    let clean = if start_step > 0 {
        revalidate_before_load(&mut check)?;
        let encoder = load_vae_encoder(root, device, &cfg)?;
        let clean = encode_init_latents(
            &encoder,
            &cfg,
            req.init_image.expect("positive start requires init image"),
            req.width,
            req.height,
            device,
            cancel,
        )?;
        device.synchronize()?;
        drop(encoder);
        Some(clean)
    } else {
        None
    };

    revalidate_before_load(&mut check)?;
    let trunk_cfg = SanaTransformerConfig::sana_sprint_1600m();
    let embedded_scale = trunk_cfg.guidance_embeds_scale;
    let transformer = load_staged_transformer(root, device, trunk_cfg, memory)?;
    let guidance = req.guidance_scale.unwrap_or(SPRINT_DEFAULT_GUIDANCE);
    let mut latents = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let noise = create_noise(device, *seed, req.width, req.height)?;
        let initial = match &clean {
            Some(clean) => renoise_sprint_init(clean, &noise, &scheduler, start_step)?,
            None => noise.affine(scheduler.sigma_data as f64, 0.0)?,
        };
        latents.push(denoise_sprint_from_memory(
            &transformer,
            &scheduler,
            start_step,
            *seed,
            initial,
            &conditioning,
            guidance,
            embedded_scale,
            device,
            cancel,
            on_progress,
            preview,
            Some(memory),
        )?);
    }
    device.synchronize()?;
    drop(transformer);
    drop(conditioning);

    revalidate_before_load(&mut check)?;
    let decoder = load_vae_decoder(root, device, &cfg)?;
    let mut images = Vec::with_capacity(latents.len());
    for latent in latents {
        on_progress(Progress::Decoding);
        images.push(decode_to_image_memory(
            &decoder,
            &cfg,
            &latent,
            Some(memory),
        )?);
    }
    device.synchronize()?;
    Ok(images)
}

/// RGB8 init image -> denoise-space DC-AE latent, matching the MLX SANA contract.
pub fn encode_init_latents(
    encoder: &DcAeEncoder,
    cfg: &DcAeConfig,
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    cancel: &CancelFlag,
) -> Result<Tensor> {
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    let image = preprocess_init_image(image, width, height, device)?;
    let latent = encoder
        .encode(&image)?
        .affine(cfg.scaling_factor as f64, 0.0)?;
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    Ok(latent)
}

/// LANCZOS fit and `[0,255] -> [-1,1]` HWC-to-NCHW preprocessing used by MLX SANA.
pub fn preprocess_init_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let expected =
        candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX);
    if image.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "sana: reference pixel buffer {} != {}x{}x3 ({expected})",
            image.pixels.len(),
            image.width,
            image.height
        )));
    }
    let (tw, th) = (width as usize, height as usize);
    let resized = if (iw, ih) == (tw, th) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)?
    };
    let mut nchw = vec![0.0f32; 3 * th * tw];
    for y in 0..th {
        for x in 0..tw {
            for c in 0..3 {
                nchw[c * th * tw + y * tw + x] = resized[(y * tw + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(nchw, (1, 3, th, tw), device)?)
}

/// Sprint SCM/TrigFlow tail over an already sigma-data-scaled img2img or txt2img latent.
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_from(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> Result<Tensor> {
    denoise_sprint_from_memory(
        transformer,
        scheduler,
        start_step,
        seed,
        latents,
        cond,
        guidance_scale,
        guidance_embeds_scale,
        device,
        cancel,
        on_progress,
        preview,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_from_memory(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Tensor,
    cond: &Tensor,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    device: &Device,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
    memory: Option<candle_gen::gen_core::GenerationMemory>,
) -> Result<Tensor> {
    let guidance = Tensor::from_vec(vec![guidance_scale * guidance_embeds_scale], (1,), device)?;
    let predict = |lat_in: &Tensor, scm_t: f32| -> Result<Tensor> {
        let t = Tensor::from_vec(vec![scm_t], (1,), device)?;
        let budget = memory
            .filter(|memory| memory.chunk_attention)
            .and_then(|memory| memory.attention_chunk_size)
            .map(|value| value as usize);
        transformer
            .forward_with_guidance_memory(lat_in, cond, &t, Some(&guidance), budget)
            .map_err(CandleError::from)
    };
    run_scm_sampler_from(
        scheduler,
        start_step,
        latents,
        seed,
        cancel,
        on_progress,
        Some(preview),
        predict,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn noise_shape_is_batch1_32ch() {
        let dev = Device::Cpu;
        let n = create_noise(&dev, 0, 1024, 1024).unwrap();
        assert_eq!(n.dims(), &[1, 32, 32, 32]);
        let n = create_noise(&dev, 0, 512, 1024).unwrap();
        // width 512 → latent w = 16; height 1024 → latent h = 32.
        assert_eq!(n.dims(), &[1, 32, 32, 16]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let dev = Device::Cpu;
        let a = create_noise(&dev, 7, 256, 256).unwrap();
        let b = create_noise(&dev, 7, 256, 256).unwrap();
        let c = create_noise(&dev, 8, 256, 256).unwrap();
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v(&a), v(&b), "same seed reproduces");
        assert_ne!(v(&a), v(&c), "diff seed differs");
    }

    #[test]
    fn img2img_strength_law_matches_mlx() {
        assert_eq!(resolve_strength(Some(0.7), Some(0.3)), 0.7);
        assert_eq!(resolve_strength(None, Some(0.3)), 0.3);
        assert_eq!(resolve_strength(None, None), DEFAULT_IMG2IMG_STRENGTH);
        assert_eq!(init_time_step(20, None), 0);
        assert_eq!(init_time_step(20, Some(0.0)), 0);
        assert_eq!(init_time_step(20, Some(0.5)), 10);
        assert_eq!(init_time_step(20, Some(0.01)), 1);
        assert_eq!(init_time_step(20, Some(1.0)), 20);
        assert_eq!(init_time_step(20, Some(2.0)), 20);

        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        };
        assert_eq!(resolve_init_start(Some(&image), 20, Some(0.0)), 0);
        assert_eq!(resolve_init_start(None, 20, Some(0.8)), 0);
        assert_eq!(resolve_init_start(Some(&image), 20, Some(0.5)), 10);
    }

    #[test]
    fn base_and_sprint_img2img_init_math_matches_the_contract() {
        let clean = Tensor::from_vec(vec![2.0f32, -2.0], (2,), &Device::Cpu).unwrap();
        let noise = Tensor::from_vec(vec![10.0f32, 6.0], (2,), &Device::Cpu).unwrap();
        let flow = blend_flow_init(&clean, &noise, &[1.0, 0.5, 0.0], 1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(flow, vec![6.0, 2.0]);
        assert!(blend_flow_init(&clean, &noise, &[1.0, 0.0], 2).is_err());

        let scheduler = ScmScheduler::new(2);
        let sprint = renoise_sprint_init(&clean, &noise, &scheduler, 1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let t = scheduler.timesteps[1];
        let sd = scheduler.sigma_data;
        for ((got, x0), eps) in sprint.iter().zip([2.0f32, -2.0]).zip([10.0f32, 6.0]) {
            let want = sd * (t.cos() * x0 + t.sin() * eps);
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        assert!(renoise_sprint_init(&clean, &noise, &scheduler, 3).is_err());
    }

    #[test]
    fn img2img_preprocess_resizes_normalizes_and_rejects_bad_shape() {
        let white = Image {
            width: 2,
            height: 3,
            pixels: vec![255; 2 * 3 * 3],
        };
        let tensor = preprocess_init_image(&white, 4, 4, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 4, 4]);
        assert!(tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-6));

        let bad = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 11],
        };
        assert!(preprocess_init_image(&bad, 4, 4, &Device::Cpu).is_err());
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SANA-1.6B: FlowMatchEulerDiscreteScheduler shift=3.0, no dynamic shifting. The native (unset
        // scheduler) path must reproduce the diffusers static-shift σ table exactly.
        let s = sana_sigmas(None, 4);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.len(), 5);
        for (got, want) in s.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    #[test]
    fn curated_scheduler_reshapes_but_stays_descending_to_zero() {
        // A curated epic-7114 scheduler name re-shapes σ over the same mu=ln(3): still descending,
        // trailing 0, and distinct from the native ramp (so the knob has an effect).
        let native = sana_sigmas(None, 12);
        let karras = sana_sigmas(Some("karras"), 12);
        assert_eq!(*karras.last().unwrap(), 0.0);
        assert!(karras.windows(2).all(|w| w[0] >= w[1]));
        assert_ne!(karras, native);
    }

    #[test]
    fn resolve_component_files_prefers_shards_and_drops_fp16() {
        use std::fs::File;
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // Mimic the diffusers transformer dir: single bf16 + fp32 shards + an fp16 copy + non-weights.
        for f in [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ] {
            File::create(dir.join(f)).unwrap();
        }
        let chosen = resolve_component_files(&dir).unwrap();
        let names: Vec<String> = chosen
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "diffusion_pytorch_model-00001-of-00002.safetensors".to_string(),
                "diffusion_pytorch_model-00002-of-00002.safetensors".to_string(),
            ],
            "shards win, single + fp16 dropped"
        );

        // A single-file component dir (the vae layout: one fp32 + one fp16) → the single fp32 file.
        let vdir = dir.join("vae");
        std::fs::create_dir_all(&vdir).unwrap();
        for f in [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ] {
            File::create(vdir.join(f)).unwrap();
        }
        let chosen = resolve_component_files(&vdir).unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(
            chosen[0].file_name().unwrap().to_str().unwrap(),
            "diffusion_pytorch_model.safetensors"
        );
    }
}
