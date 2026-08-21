//! SD3.5 text-to-image sampling pipeline (E5, sc-7864): tokenization → triple-TE conditioning →
//! seeded latent noise → flow-match Euler denoise (with true-CFG) → VAE decode → RGB8.
//!
//! ## Sampler / shift / CFG
//!
//! * **Flow-match Euler with static shift 3.0.** SD3.5-Large's `scheduler/scheduler_config.json`
//!   pins `FlowMatchEulerDiscreteScheduler { shift: 3.0 }` with no dynamic shifting, so the schedule
//!   is [`FlowMatchEuler::for_static_shift(steps, 3.0)`] — identical to the Z-Image-Turbo path. An
//!   unset `req.scheduler` keeps that native schedule byte-exact; a curated name re-shapes σ over the
//!   same `mu = ln(3)` (epic 7114).
//! * **Timestep convention.** The MMDiT embeds the diffusers-scale timestep `sigma * 1000` (the
//!   scheduler's `num_train_timesteps`). The unified flow sampler hands the predict closure
//!   `ms.timestep(σ) = σ` (the `Sigma` convention); the closure scales it to `σ·1000` before the
//!   forward. The Euler update itself stays in σ-space (`x += (σ_{t+1}-σ_t)·v`).
//! * **True CFG.** SD3.5-Large is a true-CFG model: each step runs TWO forwards (cond + uncond) and
//!   combines `pred = uncond + scale·(cond − uncond)`. The uncond branch conditions on the
//!   (empty/negative) prompt's triple-TE embedding. `guidance_scale` defaults to 3.5.

use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::{
    run_flow_sampler_with_latent_hook, CancelFlag, FlowMatchEuler, Image, PreviewSink, Progress,
    Result, TimestepConvention,
};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::loader::{Sd3ClipPad, CLIP_MAX_LENGTH};
use crate::text::{Sd3Conditioning, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;

/// SD3.5 latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downsample (latent edge is image/8).
pub const SPATIAL_SCALE: u32 = 8;
/// diffusers `num_train_timesteps` — the MMDiT embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SD3.5-Large static flow-match shift (`scheduler_config.json` `shift = 3.0`, no dynamic shifting).
pub const SCHEDULE_SHIFT: f32 = 3.0;

/// Seeded txt2img latent noise — shape `[1, 16, height/8, width/8]`, f32. diffusers
/// `randn_tensor([B, 16, H/8, W/8])`; we draw f32 via `mx.random.normal` keyed on `seed`.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        1,
        LATENT_CHANNELS,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Tokenize one prompt for CLIP into the raw (unpadded, capped-at-77) int32 id sequence. The
/// **empty** prompt is NOT special-cased: `ClipBpeTokenizer::tokenize("")` returns `[BOS, EOS]` (BOS
/// is always prepended, EOS always appended), which after padding is exactly diffusers
/// `tokenizer("", padding="max_length")`. This is load-bearing for the true-CFG uncond branch of
/// every default (unset-negative) render — an earlier `is_empty() → Vec::new()` shortcut produced
/// 77×EOS with NO BOS, changing every hidden state and shifting the pooled-at-argmax EOS selection
/// from index 1 to 0 (F-004; same bug family as z-image sc-8958).
///
/// An over-long prompt is **truncated** here, deliberately: SD3.5 carries its long-form prompt on
/// the 256-token T5 lane, and diffusers' `_get_clip_prompt_embeds` likewise tokenizes CLIP with
/// `truncation=True, max_length=77`. The two CLIP lanes (L and bigG) share this one id sequence.
fn clip_token_ids(tokenizer: &ClipBpeTokenizer, prompt: &str) -> Result<Vec<i32>> {
    let mut ids = tokenizer.tokenize(prompt)?;
    if ids.len() > CLIP_MAX_LENGTH {
        ids.truncate(CLIP_MAX_LENGTH);
        // sc-20528: restore the EOS the truncation just cut off. Until sc-20528,
        // `ClipBpeTokenizer::tokenize` capped at 77 itself and wrote eos into the last slot; it now
        // returns the full encoding (SDXL windows it instead), so a bare `truncate` would hand the
        // encoder `[BOS, 76 content]` with NO end-of-text token. `ClipTextEncoder::forward` pools
        // at `argmax(row)` — EOS is the highest CLIP id, so with no EOS present the pooled vector
        // (SD3's adaLN conditioning, and half of the 2048-wide pooled projection) would be gathered
        // at whichever content token happened to hold the largest id: silent quality loss on every
        // >77-token SD3.5 render and on SD3 LoRA training captions. Terminating the window keeps
        // the pre-sc-20528 ids byte-for-byte.
        ids[CLIP_MAX_LENGTH - 1] = tokenizer.eos_id();
    }
    Ok(ids)
}

/// Right-pad a raw CLIP id sequence to a fixed `[1, 77]` int32 row with `pad_id`
/// (diffusers `padding="max_length", max_length=77`). The pad token DIFFERS per encoder — CLIP-L
/// pads with eos (49407), OpenCLIP-bigG with `!` (0) — see [`Sd3ClipPad`] (sc-9581).
fn pad_clip_row(ids: &[i32], pad_id: i32) -> Array {
    let mut row = ids.to_vec();
    row.resize(CLIP_MAX_LENGTH, pad_id);
    Array::from_slice(&row, &[1, CLIP_MAX_LENGTH as i32])
}

/// Encode one prompt into SD3.5 conditioning (`pooled [1,2048]`, `context [1,333,4096]`) via the
/// triple-TE aggregator. CLIP ids are padded to 77; T5 ids to 256 (the gen-core T5 tokenizer's
/// `pad_to_max_length`). T5 runs unmasked (diffusers default).
///
/// CLIP-L and bigG share ONE BPE tokenizer (identical token sequence), but SD3.5 pads them with
/// DIFFERENT pad tokens: L with eos (49407), bigG with `!` (0). Tokenize once, then pad each row with
/// its encoder's pad id (`clip_pad`) — padding bigG with eos corrupts its penultimate hidden on every
/// pad slot and thus the joint context for any sub-77-token prompt (sc-9581, mirrors candle-gen-sd3).
pub fn encode_prompt(
    encoders: &Sd3TextEncoders,
    clip_tokenizer: &ClipBpeTokenizer,
    clip_pad: Sd3ClipPad,
    t5_tokenizer: &mlx_gen::tokenizer::TextTokenizer,
    prompt: &str,
) -> Result<Sd3Conditioning> {
    let clip_ids = clip_token_ids(clip_tokenizer, prompt)?;
    let clip_l_row = pad_clip_row(&clip_ids, clip_pad.pad_l);
    let clip_g_row = pad_clip_row(&clip_ids, clip_pad.pad_g);
    let t5 = t5_tokenizer.tokenize(prompt)?;
    let (t5_ids, _t5_mask) = mlx_gen::tokenizer::to_arrays(&t5);
    encoders.encode(&clip_l_row, &clip_g_row, &t5_ids, None)
}

/// The shared flow-match Euler denoise core over an explicit `sigmas` slice — the true-CFG predict
/// closure + the unified sampler. Runs the MMDiT once (cond) or twice (cond + uncond → `uncond +
/// scale·(cond − uncond)`) per step; the Euler step advances the latents in σ-space; the MMDiT
/// timestep is `σ·1000`. txt2img passes the full schedule ([`denoise_cfg`]); img2img passes the tail
/// `sigmas[start..]` from a noised init latent ([`denoise_img2img_cfg`]).
#[allow(clippy::too_many_arguments)]
fn denoise_over_sigmas(
    transformer: &Sd3Transformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    attention: mlx_gen::attention::AttentionPlan<'_>,
    transformer_window: Option<usize>,
) -> Result<Array> {
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        // The unified flow sampler hands `timestep = σ`; the MMDiT embeds `σ·1000`.
        let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
        let window = transformer_window.map(|size| (size, cancel));
        let pred_cond =
            transformer.forward_inference(x, &cond.context, &cond.pooled, &t, attention, window)?;
        match uncond {
            Some(uc) if guidance_scale != 1.0 => {
                let pred_uncond = transformer.forward_inference(
                    x,
                    &uc.context,
                    &uc.pooled,
                    &t,
                    attention,
                    window,
                )?;
                // pred = uncond + scale·(cond − uncond).
                let delta = subtract(&pred_cond, &pred_uncond)?;
                Ok(add(
                    &pred_uncond,
                    &multiply(&delta, Array::from_slice(&[guidance_scale], &[1]))?,
                )?)
            }
            _ => Ok(pred_cond),
        }
    };
    let previews = mlx_gen::preview::PreviewCounter::new(sigmas);
    run_flow_sampler_with_latent_hook(
        sampler_name,
        TimestepConvention::Sigma,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        |latents, sigma| {
            crate::preview::emit_preview(preview, &previews, sigmas, sigma, latents);
        },
        predict,
    )
}

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the MMDiT twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The MMDiT timestep is `σ·1000`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_cfg_with_preview(
        transformer,
        scheduler,
        sampler_name,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

/// [`denoise_cfg`] with an optional best-effort per-outer-step preview sink.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_with_preview(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_cfg_with_memory(
        transformer,
        scheduler,
        sampler_name,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        preview,
        mlx_gen::attention::AttentionPlan::UNBOUNDED,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_cfg_with_memory(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    attention: mlx_gen::attention::AttentionPlan<'_>,
    transformer_window: Option<usize>,
) -> Result<Array> {
    denoise_over_sigmas(
        transformer,
        &scheduler.sigmas,
        sampler_name,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        preview,
        attention,
        transformer_window,
    )
}

/// **img2img latent-init** (epic 8588 slice A4, sc-10189) — reference-guided generation on SD3.5.
/// VAE-encode `init` into the same normalized 16-ch latent space as [`create_noise`] (SD3.5's VAE
/// `encode` returns `(mean − shift)·scale`, matching diffusers' `StableDiffusion3Img2ImgPipeline`),
/// blend `(1 − σ_k)·clean + σ_k·noise` at the start sigma `σ_k = sigmas[k]`, and run the true-CFG
/// flow-match Euler sampler over the tail `sigmas[k..]`. `strength` is reference fidelity in the fork's
/// [`init_time_step`] convention (`k = max(1, ⌊num_steps·strength⌋)`): higher strength → later start →
/// fewer denoise steps → the output stays closer to the reference; `strength ≤ 0` degenerates to a full
/// txt2img (`k = 0`, identical to [`denoise_cfg`]). Unlike the packed Qwen-Image / Z-Image path, the
/// SD3.5 MMDiT patchifies internally, so the clean latent is used **unpacked** `[1, 16, H/8, W/8]`
/// (matching `create_noise`) — no pre-pack. Shares the true-CFG predict core with [`denoise_cfg`], so
/// Large/Medium run two forwards/step and the distilled Large-Turbo runs one (`guidance == 1.0`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_img2img_cfg(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    vae: &Vae,
    init: &Image,
    strength: f32,
    width: u32,
    height: u32,
    steps: usize,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_img2img_cfg_with_preview(
        transformer,
        scheduler,
        sampler_name,
        seed,
        vae,
        init,
        strength,
        width,
        height,
        steps,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

/// [`denoise_img2img_cfg`] with an optional best-effort per-outer-step preview sink.
#[allow(clippy::too_many_arguments)]
pub fn denoise_img2img_cfg_with_preview(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    vae: &Vae,
    init: &Image,
    strength: f32,
    width: u32,
    height: u32,
    steps: usize,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_img2img_cfg_with_memory(
        transformer,
        scheduler,
        sampler_name,
        seed,
        vae,
        init,
        strength,
        width,
        height,
        steps,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        preview,
        mlx_gen::attention::AttentionPlan::UNBOUNDED,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_img2img_cfg_with_memory(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    vae: &Vae,
    init: &Image,
    strength: f32,
    width: u32,
    height: u32,
    steps: usize,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    attention: mlx_gen::attention::AttentionPlan<'_>,
    transformer_window: Option<usize>,
) -> Result<Array> {
    // Reference → clean latent [1, 16, H/8, W/8]. `Vae::encode` returns the normalized `(mean−shift)·
    // scale` latent (the same space as `create_noise`); SD3.5's MMDiT patchifies internally, so keep it
    // unpacked.
    let image_nchw = preprocess_init_image(init, width, height)?;
    let clean = vae.encode(&image_nchw)?;
    let noise = create_noise(seed, width, height)?;

    // Start step from strength; blend clean⊕noise at σ_k, then denoise sigmas[k..]. The schedule has
    // `steps + 1` sigmas, so clamp the start index inside it (strength ≥ 1 → the last usable step).
    let start = init_time_step(steps, Some(strength)).min(scheduler.sigmas.len().saturating_sub(1));
    let x_start = add_noise_by_interpolation(&clean, &noise, scheduler.sigmas[start])?;
    denoise_over_sigmas(
        transformer,
        &scheduler.sigmas[start..],
        sampler_name,
        seed,
        x_start,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
        preview,
        attention,
        transformer_window,
    )
}

/// VAE-decode the final `[1,16,H/8,W/8]` latent → an RGB8 [`Image`]. The de-norm (`z/scale + shift`)
/// is applied inside [`Vae::decode`] (the reused Z-Image VAE with SD3.5's factors), so the raw latent
/// is handed straight through.
pub fn decode_to_image(vae: &Vae, latents: &Array) -> Result<Image> {
    let decoded = vae.decode(latents)?.as_dtype(Dtype::Float32)?;
    mlx_gen::image::decoded_to_image(&decoded)
}

pub(crate) fn decode_to_image_tiled(
    vae: &Vae,
    latents: &Array,
    tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cancel: &CancelFlag,
) -> Result<Image> {
    let decoded = match tiling {
        Some(config) => vae.decode_tiled(latents, config, Some(cancel))?,
        None => vae.decode(latents)?,
    }
    .as_dtype(Dtype::Float32)?;
    mlx_gen::image::decoded_to_image(&decoded)
}

// =================================================================================================
// Chained denoise passes (epic 20414, sc-20425)
// =================================================================================================

/// The scheduler ids SD3.5 honors on a chained pass **beyond** the curated registry.
///
/// `flow_match` names the family's own static-shift schedule ([`FlowMatchEuler::for_static_shift`])
/// — the byte-exact default an unset `req.scheduler` already resolves to through
/// `resolve_flow_schedule`'s N3 fallback. It is advertised (`Sd3Variant::descriptor`) and declared
/// (`denoise_pass_surface`) so a resolved plan can *name* the model default and replay through
/// validation.
///
/// `linear` is deliberately NOT here. This crate advertises it on the flat request, but no curated
/// `Scheduler` implements it and neither does this family, so on a chain it would have taken the
/// same native fallback under a different name — the wrong schedule, reported as success. Leaving it
/// undeclared makes a pass naming it a typed rejection (sc-20425 item 3).
pub const NATIVE_SCHEDULERS: &[&str] = &["flow_match"];

/// One chained pass's **fresh** σ schedule (the `gen_core::sampling::DenoisePassHost` seam).
///
/// Deliberately the same two lines `model.rs`'s Phase B runs — the native static-shift schedule for
/// `schedule_steps`, then the curated scheduler axis over the same `mu = ln(3)` — so a one-pass
/// chain is the legacy render rather than a second numerical path. The addition is
/// `gen_core::resolve_pass_scheduler`, which turns an id this family cannot honor into a typed,
/// pass-indexed rejection before `resolve_flow_schedule` would quietly hand back the native
/// schedule under someone else's name.
pub fn pass_schedule(
    pass: &mlx_gen::gen_core::ResolvedDenoisePass,
    schedule_steps: usize,
) -> mlx_gen::gen_core::Result<Vec<f32>> {
    mlx_gen::gen_core::resolve_pass_scheduler(pass, NATIVE_SCHEDULERS)?;
    let native = FlowMatchEuler::for_static_shift(schedule_steps, SCHEDULE_SHIFT);
    Ok(mlx_gen::resolve_flow_schedule(
        Some(pass.scheduler.as_str()),
        SCHEDULE_SHIFT.ln(),
        schedule_steps,
        &native.sigmas,
    ))
}

/// Chain-global denoise-pass preview (sc-20425), the MLX twin of `candle_gen::preview::PassPreview`
/// and a copy of the shape `mlx-gen-krea` established in sc-20418.
///
/// Frames are numbered by the executor's `chain_step` (0-based outer step across the WHOLE chain),
/// so a multi-pass job reads as one continuous `1..=total` trajectory instead of restarting per
/// pass. There is no single σ array to key the shared sigma-position counter on — each pass owns a
/// fresh schedule — so this dedups on the step index directly; multi-eval solvers repeat a
/// `chain_step` and only its first evaluation emits. One per image. Projection failures are
/// swallowed (previews are decorative and never fail a render).
struct PassPreview<'a> {
    sink: &'a PreviewSink,
    emitted: std::cell::Cell<u32>,
    /// Frames whose position was consumed and whose projection then failed (sc-20425 review
    /// MINOR 6). Swallowing is the contract — previews are decorative and a lost frame must never
    /// fail a render — but *silence* was an accident, and it hides a 100 % loss rate behind
    /// something indistinguishable from "this model does not preview". The candle twin
    /// (`candle_gen::preview::PreviewCounter::dropped_frames`) counts and logs; so does this.
    dropped: std::cell::Cell<u32>,
}

impl<'a> PassPreview<'a> {
    fn new(sink: &'a PreviewSink) -> Self {
        Self {
            sink,
            emitted: std::cell::Cell::new(0),
            dropped: std::cell::Cell::new(0),
        }
    }

    /// Frames this chain consumed a position for and then failed to project. `0` on a healthy
    /// chain; equal to the frame count on the σ-less-into-σ-required failure the shared seam exists
    /// to prevent.
    fn dropped_frames(&self) -> u32 {
        self.dropped.get()
    }

    fn emit(&self, chain_step: usize, chain_total_steps: usize, latents: &Array) {
        if !self.sink.is_active() {
            return;
        }
        let total = chain_total_steps.max(1) as u32;
        let candidate = (chain_step as u32 + 1).min(total);
        if candidate <= self.emitted.get() {
            return;
        }
        // Consume the position before projecting (the shared emit_preview contract): a failed
        // projection loses only this decorative frame and is never retried as a duplicate.
        self.emitted.set(candidate);
        match crate::preview::project_pass_latents(latents) {
            Ok(image) => self.sink.emit(mlx_gen::gen_core::PreviewFrame {
                current: candidate,
                total,
                image,
            }),
            Err(err) => {
                let first = self.dropped_frames() == 0;
                self.dropped.set(self.dropped.get().saturating_add(1));
                if first {
                    eprintln!(
                        "preview: dropping frame {candidate}/{total} — projection failed: {err}. \
                         Previews are decorative, so the render continues; further drops on this \
                         trajectory are counted but not printed."
                    );
                }
            }
        }
    }
}

/// Compile-time witness that this crate really implements the chained-denoise host it advertises
/// (sc-20425 review MINOR 9).
///
/// The descriptor conformance sweep closes the *derived-descriptor* half of "advertises without a
/// host" — a control route, an unusable menu, a half-inherited surface — but it cannot see whether a
/// `DenoisePassHost` exists at all, because a descriptor is data and the host is a trait impl in
/// another crate. This is the missing half, in the one place that can check it: if the impl below is
/// ever deleted or renamed while `supports_denoise_passes` stays `true`, THIS crate stops compiling
/// rather than shipping a capability nothing serves.
const _: fn(&mut Sd3PassHost<'_>) = |host| {
    let _: &mut dyn mlx_gen::gen_core::sampling::DenoisePassHost<mlx_gen::MlxLatentOps> = host;
};

/// SD3.5's `gen_core::sampling::DenoisePassHost`: the family schedule seam, the per-pass forward
/// (per-pass guidance combined inside, exactly as `denoise_over_sigmas`'s closure does it), and the
/// chain-global preview.
///
/// Everything it holds is a shared borrow or `Copy` — the same set the single-pass predict closure
/// captures — because SD3's conditioning is hoisted once per batch and the transformer is immutable
/// during a render. There is no per-pass adapter state to apply or revert (SD3 folds adapters at
/// load), so `begin_pass`/`end_pass` stay the trait defaults and the descriptor's
/// `per_pass_adapters` is `false`.
struct Sd3PassHost<'a> {
    transformer: &'a Sd3Transformer,
    cond: &'a Sd3Conditioning,
    uncond: Option<&'a Sd3Conditioning>,
    cfg_enabled: bool,
    attention: mlx_gen::attention::AttentionPlan<'a>,
    transformer_window: Option<(usize, &'a CancelFlag)>,
    preview: PassPreview<'a>,
}

impl mlx_gen::gen_core::sampling::DenoisePassHost<mlx_gen::MlxLatentOps> for Sd3PassHost<'_> {
    fn build_schedule(
        &mut self,
        pass: &mlx_gen::gen_core::ResolvedDenoisePass,
        schedule_steps: usize,
    ) -> mlx_gen::gen_core::Result<Vec<f32>> {
        pass_schedule(pass, schedule_steps)
    }

    fn predict(
        &mut self,
        pass: &mlx_gen::gen_core::ResolvedDenoisePass,
        x: &Array,
        timestep: f32,
    ) -> mlx_gen::gen_core::Result<Array> {
        let run = || -> Result<Array> {
            // Per-eval compute boundary (the shared driver contract): force the prior step's lazy
            // graph so the chain stays cancellable per evaluation instead of becoming one
            // un-cancellable graph that only runs at decode.
            mlx_rs::transforms::eval([x])?;
            // `timestep` is `FlowModelSampling::timestep(σ) == σ` (the Sigma convention); the MMDiT
            // embeds `σ·1000`, exactly as the single-pass closure does.
            let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
            let pred_cond = self.transformer.forward_inference(
                x,
                &self.cond.context,
                &self.cond.pooled,
                &t,
                self.attention,
                self.transformer_window,
            )?;
            // Per-pass CFG: the distilled Turbo has no guidance axis at all (`cfg_enabled` false,
            // and the shared floor rejects a per-pass `guidance` on it), and a CFG variant collapses
            // to the conditional branch at scale 1.0 exactly as the single-pass lane does.
            let guidance = if self.cfg_enabled {
                pass.guidance.unwrap_or(1.0)
            } else {
                1.0
            };
            match self.uncond {
                Some(uc) if guidance != 1.0 => {
                    let pred_uncond = self.transformer.forward_inference(
                        x,
                        &uc.context,
                        &uc.pooled,
                        &t,
                        self.attention,
                        self.transformer_window,
                    )?;
                    let delta = subtract(&pred_cond, &pred_uncond)?;
                    Ok(add(
                        &pred_uncond,
                        &multiply(&delta, Array::from_slice(&[guidance], &[1]))?,
                    )?)
                }
                _ => Ok(pred_cond),
            }
        };
        run().map_err(Into::into)
    }

    fn observe(&mut self, obs: mlx_gen::gen_core::sampling::PassObservation<'_, Array>) {
        self.preview
            .emit(obs.chain_step, obs.chain_total_steps, obs.latent);
    }
}

/// Run one resolved chained-denoise plan over SD3.5 and return the final latent — the chained twin
/// of `denoise_cfg_with_memory`.
///
/// The caller owns the single VAE decode after the chain, exactly as it does on the single-pass
/// lane; the executor never decodes.
#[allow(clippy::too_many_arguments)] // mirrors the sibling denoise entry points in this module
pub fn render_denoise_passes(
    transformer: &Sd3Transformer,
    plan: &mlx_gen::gen_core::ResolvedDenoisePlan,
    initial: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    cfg_enabled: bool,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    attention: mlx_gen::attention::AttentionPlan<'_>,
    transformer_window: Option<usize>,
) -> Result<(Array, mlx_gen::gen_core::DenoisePlanExecution)> {
    use mlx_gen::gen_core::sampling::{
        execute_denoise_plan, FlowModelSampling, TimestepConvention as Conv,
    };

    let ms = FlowModelSampling::new(Conv::Sigma);
    let mut host = Sd3PassHost {
        transformer,
        cond,
        uncond,
        cfg_enabled,
        attention,
        transformer_window: transformer_window.map(|size| (size, cancel)),
        preview: PassPreview::new(preview),
    };
    let run = execute_denoise_plan(
        &mlx_gen::MlxLatentOps,
        &ms,
        plan,
        initial,
        &mut host,
        cancel,
        on_progress,
    )?;
    // Force the final step's lazy graph before decode (the shared driver's tail contract).
    mlx_rs::transforms::eval([&run.latent])?;
    Ok((run.latent, run.execution))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    use crate::loader::{resolve_clip_pad_id, CLIP_EOS_ID};

    // ---- Chained denoise passes (epic 20414, sc-20425) ------------------------------------------

    fn dp_resolved(
        sampler: &str,
        scheduler: &str,
        steps: u32,
    ) -> mlx_gen::gen_core::ResolvedDenoisePass {
        mlx_gen::gen_core::ResolvedDenoisePass {
            index: 0,
            steps,
            sampler: sampler.to_owned(),
            scheduler: scheduler.to_owned(),
            denoise: 1.0,
            guidance: None,
            seed: 1234,
            adapters: Vec::new(),
        }
    }

    /// Adoption: the backend-neutral executor conformance suite
    /// (`gen_core_testkit::denoise_passes`) runs over SD3.5's REAL per-pass schedule seam and over
    /// the advertised native alias the resolution ladder's model default names. Pure host math,
    /// weights-free: the executor runs over `CpuLatentOps` with a stub model (no `Array` anywhere),
    /// so what varies per family is exactly the schedule math under test — the same invocation shape
    /// as the candle twin.
    #[test]
    fn shared_pass_executor_conformance_over_the_sd3_schedule_seam() {
        gen_core_testkit::denoise_passes::denoise_pass_conformance("mlx sd3_5", &|pass, steps| {
            pass_schedule(pass, steps).expect("a curated id always resolves")
        });
        gen_core_testkit::denoise_passes::denoise_pass_conformance(
            "mlx sd3_5 flow_match alias",
            &|_pass, steps| {
                pass_schedule(&dp_resolved("euler", "flow_match", steps as u32), steps)
                    .expect("the declared native alias is honored")
            },
        );
    }

    /// **The sc-20425 review's MAJOR 1, on this family.** The generator binds the executor's
    /// `DenoisePlanExecution` and publishes it through `GenerationRequest::emit_denoise_pass_report`
    /// before returning any image; without that the plan a render actually ran is unrecoverable and
    /// the epic's replay path has nothing to replay from. This drives the REAL schedule seam,
    /// model defaults and descriptor context through the shared adopter check, so the record's
    /// requested-vs-resolved contents and eval accounting are pinned against this family's own
    /// resolution ladder.
    #[test]
    fn the_generator_publishes_one_execution_record_for_a_chain() {
        let requested = vec![
            gen_core::DenoisePass {
                steps: Some(4),
                ..Default::default()
            },
            gen_core::DenoisePass {
                steps: Some(3),
                sampler: Some("euler".to_owned()),
                denoise: Some(0.5),
                ..Default::default()
            },
        ];
        let req = GenerationRequest {
            denoise_passes: Some(requested.clone()),
            ..Default::default()
        };
        let ms = mlx_gen::gen_core::sampling::FlowModelSampling::new(
            mlx_gen::gen_core::sampling::TimestepConvention::Sigma,
        );
        let caps = crate::config::Sd3Variant::Large.descriptor().capabilities;
        let ctx = caps.denoise_pass_context(None);
        let defaults = crate::model::sd3_denoise_defaults(crate::config::Sd3Variant::Large);
        let record = gen_core_testkit::denoise_passes::check_execution_record(
            &|pass: &mlx_gen::gen_core::ResolvedDenoisePass, steps: usize| {
                pass_schedule(pass, steps).expect("a curated id always resolves")
            },
            &ms,
            &req,
            0x5eed,
            &defaults,
            &ctx,
        )
        .expect("the execution record must satisfy the shared adopter contract");

        // The ladder's own answers, published: pass 0 named no sampler/scheduler, so both come
        // from this family's model defaults; pass 1 named its sampler and denoise.
        assert_eq!(record.passes.len(), 2);
        assert_eq!(
            record.passes[0].resolved.sampler, defaults.sampler,
            "an unnamed per-pass sampler must resolve to this family's default"
        );
        assert_eq!(record.passes[0].resolved.scheduler, defaults.scheduler);
        assert_eq!(record.passes[0].resolved.steps, 4);
        assert_eq!(record.passes[1].resolved.sampler, "euler");
        assert_eq!(record.passes[1].resolved.denoise, 0.5);
        // And the requested values ride alongside, so a consumer can tell the two apart.
        assert_eq!(record.passes[0].requested.as_ref(), Some(&requested[0]));
        assert_eq!(record.passes[1].requested.as_ref(), Some(&requested[1]));
        // SD3.5 Large has a guidance axis, so the ladder fills it from the model default.
        assert_eq!(record.passes[0].resolved.guidance, defaults.guidance);
    }

    /// The per-pass schedule seam is the single-pass one: the native static-shift schedule under the
    /// advertised alias, and the curated axis over the same `mu = ln(3)` otherwise.
    #[test]
    fn a_pass_schedule_is_the_single_pass_schedule() {
        let native = FlowMatchEuler::for_static_shift(8, SCHEDULE_SHIFT);
        assert_eq!(
            pass_schedule(&dp_resolved("euler", "flow_match", 8), 8).unwrap(),
            native.sigmas,
            "the native alias must not re-shape the family's own schedule"
        );
        assert_eq!(
            pass_schedule(&dp_resolved("euler", "karras", 8), 8).unwrap(),
            mlx_gen::resolve_flow_schedule(Some("karras"), SCHEDULE_SHIFT.ln(), 8, &native.sigmas),
            "a curated id must resolve through the same seam the render core uses"
        );
    }

    /// **The sc-20425 item-3 trap, on this family.** `linear` is advertised in this crate's
    /// scheduler menu and no curated `Scheduler` implements it; undeclared, `resolve_flow_schedule`
    /// would hand back the native schedule under that name — the wrong algorithm, reported as
    /// success. `flow_match` is advertised in the SAMPLER menu and is not a curated `Solver`, so it
    /// is rejected on that axis too rather than integrating as Euler under another name.
    #[test]
    fn unhonored_pass_algorithms_are_rejected_not_silently_substituted() {
        let err = pass_schedule(&dp_resolved("euler", "linear", 8), 8)
            .expect_err("an undeclared native scheduler must be rejected");
        assert!(
            matches!(err, mlx_gen::gen_core::Error::Unsupported(_)),
            "a capability gap must stay typed: {err:?}"
        );
        assert!(
            format!("{err}").contains("denoisePasses[0].scheduler"),
            "{err}"
        );

        let caps = crate::config::Sd3Variant::Large.descriptor().capabilities;
        assert!(
            caps.samplers.contains(&"flow_match"),
            "the menu still advertises it for the single-pass lane"
        );
        let ctx = caps.denoise_pass_context(None);
        let err = mlx_gen::gen_core::validate_denoise_passes(
            &[mlx_gen::gen_core::DenoisePass {
                sampler: Some("flow_match".to_owned()),
                ..Default::default()
            }],
            false,
            &ctx,
        )
        .expect_err("an uncurated sampler must be rejected");
        assert_eq!(err.field(), mlx_gen::gen_core::DenoisePassField::Sampler);
        assert!(err.is_capability_gap());
    }

    /// The chained denoise-pass preview numbers frames by the executor's chain-global outer step —
    /// one continuous `1..=total` run across pass boundaries — and dedups the multi-eval solver
    /// repeats of a step, exactly like the σ-keyed counters do per schedule.
    #[test]
    fn pass_preview_numbers_chain_steps_once_across_passes() {
        use std::sync::{Arc, Mutex};
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let preview = PassPreview::new(&sink);
        let latents = Array::zeros::<f32>(&[1, 16, 2, 2]).unwrap();
        // Two passes of 3 outer steps each; the solver repeats step 1 and step 4.
        for step in [0usize, 1, 1, 2, 3, 4, 4, 5] {
            preview.emit(step, 6, &latents);
        }
        let frames = frames.lock().unwrap();
        assert_eq!(
            frames.iter().map(|f| f.current).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert!(frames.iter().all(|f| f.total == 6));
        assert_eq!(
            preview.dropped_frames(),
            0,
            "a healthy chain must lose no frames"
        );
    }

    /// A projection failure stays swallowed — previews are decorative and never fail a render — but
    /// is now COUNTED and logged once, so a chain that silently delivers nothing is distinguishable
    /// from a model that does not preview at all (sc-20425 review MINOR 6). The candle twin counts
    /// the same way, in `candle_gen::preview::PreviewCounter::dropped_frames`.
    #[test]
    fn a_dropped_chain_frame_is_swallowed_but_counted() {
        use std::sync::{Arc, Mutex};
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| captured.lock().unwrap().push(frame));
        let preview = PassPreview::new(&sink);
        // A latent that is not the LAYOUT contract: the projector rejects it.
        let malformed = Array::zeros::<f32>(&[1, 15, 2, 2]).unwrap();
        let valid = Array::zeros::<f32>(&[1, 16, 2, 2]).unwrap();
        preview.emit(0, 3, &malformed);
        preview.emit(1, 3, &valid);
        preview.emit(2, 3, &malformed);

        let frames = frames.lock().unwrap();
        assert_eq!(
            frames.iter().map(|f| f.current).collect::<Vec<_>>(),
            vec![2],
            "only the well-formed frame is delivered, and its position is preserved"
        );
        assert_eq!(preview.dropped_frames(), 2);
    }

    /// Build a tiny synthetic CLIP BPE tokenizer (no real weights) whose special tokens match the
    /// real CLIP vocab ids so [`CLIP_EOS_ID`] (= EOS = 49407) behaves identically. Enough vocab to
    /// tokenize a short ASCII prompt; the empty prompt needs only BOS/EOS. Also writes a `!` (0)
    /// entry + a `tokenizer_config.json` (`pad_token`) so [`resolve_clip_pad_id`] can be exercised.
    /// Written to a unique temp dir and loaded through the real [`ClipBpeTokenizer::from_dir`] path so
    /// this exercises production code (matching the crate's `std::env::temp_dir()` test convention).
    /// `pad_token` selects the config's pad string (`"!"` = bigG, `"<|endoftext|>"` = L).
    fn synthetic_clip_tokenizer_dir(
        tmp: &tempfile::TempDir,
        tag: &str,
        pad_token: &str,
    ) -> std::path::PathBuf {
        let dir = tmp.path().join(format!("mlx_gen_sd3_clip_tok_{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        // Vocab: the two specials at their real CLIP ids, `!` at 0 (bigG's pad), plus a few
        // SINGLE-character `</w>` word tokens. The synthetic `merges.txt` has NO merges, so the
        // char-level BPE leaves each word as its per-character sub-tokens; only single-char words
        // (which become one `<char></w>` unigram) map to a vocab entry. A multi-char word like
        // `"fox"` would BPE to `["f", "o", "x</w>"]` — none in this vocab — and error. So the
        // non-empty test prompt below uses only single-char words (`"a b"`).
        let vocab = serde_json::json!({
            "!": 0,
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a</w>": 320,
            "b</w>": 321,
        });
        std::fs::write(dir.join("vocab.json"), vocab.to_string()).unwrap();
        // merges.txt: a header line + no merges (single-token words need none).
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({ "pad_token": pad_token }).to_string(),
        )
        .unwrap();
        dir
    }

    fn synthetic_clip_tokenizer(tmp: &tempfile::TempDir) -> ClipBpeTokenizer {
        ClipBpeTokenizer::from_dir(synthetic_clip_tokenizer_dir(tmp, "l", "<|endoftext|>")).unwrap()
    }

    #[test]
    fn empty_prompt_clip_ids_keep_bos_and_match_tokenize_path() {
        let tmp = tempfile::tempdir().unwrap();
        // F-004 (default-run, no real weights): the empty (uncond) prompt must NOT be special-cased.
        // The padded row must equal the padded `tokenize("")` path and begin with BOS (49406) then
        // EOS (49407) — NOT 77×EOS-with-no-BOS as the removed `is_empty() → Vec::new()` shortcut did.
        let tok = synthetic_clip_tokenizer(&tmp);

        // tokenize("") is [BOS, EOS].
        assert_eq!(tok.tokenize("").unwrap(), vec![49406, 49407]);

        // The L path pads with eos (49407).
        let ids = pad_clip_row(&clip_token_ids(&tok, "").unwrap(), CLIP_EOS_ID);
        eval([&ids]).unwrap();
        let row = ids.as_slice::<i32>();
        assert_eq!(row.len(), CLIP_MAX_LENGTH);
        // First slot is BOS, second is EOS, remainder padded with EOS/pad.
        assert_eq!(row[0], 49406, "empty-prompt uncond row must START with BOS");
        assert_eq!(row[1], CLIP_EOS_ID, "second id is EOS");
        assert!(
            row[2..].iter().all(|&x| x == CLIP_EOS_ID),
            "the tail is EOS/pad"
        );
        // The buggy path would have produced row[0] == EOS (no BOS) — assert we are NOT that.
        assert_ne!(
            row[0], CLIP_EOS_ID,
            "regression: empty-prompt row must not be 77×EOS (missing BOS)"
        );

        // Equivalence with the general tokenize(...) → pad path applied by hand.
        let mut expected = tok.tokenize("").unwrap();
        expected.resize(CLIP_MAX_LENGTH, CLIP_EOS_ID);
        assert_eq!(
            row,
            expected.as_slice(),
            "padded tokenize(\"\") matches by-hand"
        );
    }

    /// sc-20528: a prompt past CLIP's window must still END in EOS. `ClipBpeTokenizer::tokenize`
    /// used to cap at 77 and write eos into the last slot; it now returns the full encoding, so
    /// `clip_token_ids`' truncation has to re-terminate the window itself. Without that, the row is
    /// `[BOS, 76 content]` and `ClipTextEncoder::forward`'s `argmax` EOS-pooling gathers at an
    /// arbitrary content token — silently wrong adaLN conditioning on long SD3.5 prompts and on SD3
    /// LoRA training captions (`training.rs` → `encode_prompt`).
    #[test]
    fn over_long_prompt_truncates_to_an_eos_terminated_window() {
        let tmp = tempfile::tempdir().unwrap();
        let tok = synthetic_clip_tokenizer(&tmp);
        // The synthetic vocab holds only single-char words, so "a " x100 is a legal ~100-word prompt.
        let prompt = "a ".repeat(100);

        // The tokenizer itself no longer caps — that is what makes the re-termination load-bearing.
        assert_eq!(tok.tokenize(&prompt).unwrap().len(), 102, "BOS + 100 + EOS");
        assert_eq!(
            tok.eos_id(),
            CLIP_EOS_ID,
            "synthetic eos is the real CLIP eos"
        );

        let ids = clip_token_ids(&tok, &prompt).unwrap();
        assert_eq!(ids.len(), CLIP_MAX_LENGTH, "exactly one CLIP window");
        assert_eq!(ids[0], 49406, "the window still opens with BOS");
        assert_eq!(
            ids[CLIP_MAX_LENGTH - 1],
            CLIP_EOS_ID,
            "the truncated window must be EOS-terminated"
        );
        assert!(
            ids[1..CLIP_MAX_LENGTH - 1].iter().all(|&t| t == 320),
            "the surviving 75 slots are content"
        );
        // EOS appears exactly once, at the end — so `argmax` pools at the true end of text.
        assert_eq!(
            ids.iter().position(|&t| t == CLIP_EOS_ID),
            Some(CLIP_MAX_LENGTH - 1),
            "argmax EOS-pooling must land on the final slot"
        );
    }

    /// The other half of the contract: a prompt inside the window is passed through untouched, so
    /// every ≤77-token SD3.5 render keeps its pre-sc-20528 ids byte for byte.
    #[test]
    fn short_prompt_clip_ids_are_the_tokenizer_encoding_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let tok = synthetic_clip_tokenizer(&tmp);
        for prompt in ["", "a", "a b", "a b a b a"] {
            assert_eq!(
                clip_token_ids(&tok, prompt).unwrap(),
                tok.tokenize(prompt).unwrap(),
                "{prompt:?} fits the window and must not be rewritten"
            );
        }
        assert_eq!(
            clip_token_ids(&tok, "a b").unwrap(),
            vec![49406, 320, 321, 49407]
        );
    }

    #[test]
    fn resolve_clip_pad_reads_per_encoder_pad_token() {
        let tmp = tempfile::tempdir().unwrap();
        // sc-9581: L resolves `<|endoftext|>` (49407); bigG resolves `!` (0). A `tokenizer_config.json`
        // with no `pad_token` (or an unknown token) falls back to eos.
        let l_dir = synthetic_clip_tokenizer_dir(&tmp, "padl", "<|endoftext|>");
        let g_dir = synthetic_clip_tokenizer_dir(&tmp, "padg", "!");
        assert_eq!(
            resolve_clip_pad_id(&l_dir).unwrap(),
            49407,
            "CLIP-L pad = eos"
        );
        assert_eq!(
            resolve_clip_pad_id(&g_dir).unwrap(),
            0,
            "CLIP-bigG pad = `!` = 0"
        );

        // Fallback: a dir whose config lacks `pad_token` -> eos.
        let f_dir_tmp = tempfile::tempdir().unwrap();
        let f_dir = f_dir_tmp.path().to_path_buf();
        std::fs::write(f_dir.join("tokenizer_config.json"), "{}").unwrap();
        assert_eq!(
            resolve_clip_pad_id(&f_dir).unwrap(),
            CLIP_EOS_ID,
            "missing pad_token -> eos fallback"
        );

        std::fs::remove_file(f_dir.join("tokenizer_config.json")).unwrap();
        assert_eq!(
            resolve_clip_pad_id(&f_dir).unwrap(),
            CLIP_EOS_ID,
            "absent config -> eos fallback"
        );

        std::fs::write(f_dir.join("tokenizer_config.json"), "{").unwrap();
        let error = resolve_clip_pad_id(&f_dir).unwrap_err();
        assert!(
            matches!(error, mlx_gen::Error::Msg(ref message) if
            message.contains("parse") && message.contains("tokenizer_config.json")),
            "corrupt config must return a contextual parse error, got: {error}"
        );

        std::fs::remove_file(f_dir.join("tokenizer_config.json")).unwrap();
        std::fs::create_dir(f_dir.join("tokenizer_config.json")).unwrap();
        assert!(
            matches!(resolve_clip_pad_id(&f_dir), Err(mlx_gen::Error::Io(_))),
            "a present unreadable config must preserve the typed IO error"
        );
    }

    #[test]
    fn bigg_pads_with_bang_not_eos() {
        let tmp = tempfile::tempdir().unwrap();
        // sc-9581 core regression: with a sub-77-token prompt, the bigG row must be padded with `!`
        // (0), NOT eos (49407). The pre-fix code shared one eos-padded row for both encoders.
        let tok = synthetic_clip_tokenizer(&tmp);
        // Single-char words only (`"a b"` -> [BOS, 320, 321, EOS], len 4) so the no-merges synthetic
        // BPE tokenizes without an OOV error; still a sub-77 prompt with a real pad region.
        let ids = clip_token_ids(&tok, "a b").unwrap();
        assert_eq!(
            ids,
            vec![49406, 320, 321, 49407],
            "synthetic tokenize(\"a b\")"
        );
        let l_row = pad_clip_row(&ids, 49407);
        let g_row = pad_clip_row(&ids, 0);
        eval([&l_row, &g_row]).unwrap();
        let (l, g) = (l_row.as_slice::<i32>(), g_row.as_slice::<i32>());
        // Both share the leading content + BOS/EOS.
        assert_eq!(l[0], 49406);
        assert_eq!(g[0], 49406);
        // The pad region (after the real tokens) differs: L=eos, bigG=`!`(0).
        let pad_start = ids.len();
        assert!(
            pad_start < CLIP_MAX_LENGTH,
            "prompt must be shorter than 77"
        );
        assert!(
            l[pad_start..].iter().all(|&x| x == 49407),
            "L pads with eos"
        );
        assert!(g[pad_start..].iter().all(|&x| x == 0), "bigG pads with `!`");
        assert_ne!(l, g, "the two encoder rows must differ on the pad region");
    }

    #[test]
    fn noise_shape_is_batch1_16ch() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 16, 128, 128]);
        let n = create_noise(0, 512, 768).unwrap();
        assert_eq!(n.shape(), &[1, 16, 96, 64]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let a = create_noise(42, 256, 256).unwrap();
        let b = create_noise(42, 256, 256).unwrap();
        let c = create_noise(43, 256, 256).unwrap();
        eval([&a, &b, &c]).unwrap();
        let av = a.as_slice::<f32>();
        let bv = b.as_slice::<f32>();
        let cv = c.as_slice::<f32>();
        assert_eq!(av, bv, "same seed must reproduce the same noise");
        assert_ne!(av, cv, "a different seed must differ");
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SD3.5-Large: FlowMatchEulerDiscreteScheduler shift=3.0, no dynamic shifting.
        let s = FlowMatchEuler::for_static_shift(4, SCHEDULE_SHIFT);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }
}
