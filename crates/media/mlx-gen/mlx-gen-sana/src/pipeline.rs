//! SANA text-to-image sampling pipeline (epic 8485, story sc-8489 — **Phase A: the mlx-gen side**).
//!
//! Composes the three already-merged native SANA components into one end-to-end prompt→image path:
//!
//! ```text
//!  prompt ─▶ SanaTextEncoder (sc-8488: CHI → gemma-2-2b-it last-hidden) ─▶ [1, 300, 2304]
//!         ─▶ SanaTransformer  (sc-8487: Linear-DiT trunk, velocity prediction) ─▶ [1, 32, h, w]
//!         ─▶ DcAeDecoder      (sc-8486: DC-AE f32c32 decode)                   ─▶ [1, 1024, 1024, 3]
//! ```
//!
//! driven by the **unified flow-matching scheduler** (epic 7114): the schedule is built by
//! [`mlx_gen::FlowMatchEuler`] and integrated by [`mlx_gen::run_flow_sampler`] — the SAME machinery
//! the sibling flow-match families use (`mlx-gen-sd3`, `mlx-gen-z-image`). No bespoke scheduler.
//!
//! ## Sampler / shift / timestep convention
//!
//! * **Flow-match Euler, static shift 3.0 (a deliberate divergence from the repo default).**
//!   `Sana_1600M_1024px_diffusers` actually ships a `DPMSolverMultistepScheduler` (`solver_order = 2`,
//!   `prediction_type = flow_prediction`, `use_flow_sigmas = true`, `flow_shift = 3.0`) — NOT a
//!   `FlowMatchEulerDiscreteScheduler`. We deliberately run flow-match Euler instead: on the good
//!   `_BF16` checkpoint the 2nd-order DPM solver produces a garish / over-saturated /
//!   chromatic-aberration artifact, while Euler renders clean (verified in sc-11760). Do NOT "restore"
//!   a DPM-Solver default to "match the reference" — that reintroduces the artifact. Only `flow_shift`
//!   carries over: the native schedule is [`FlowMatchEuler::for_static_shift(steps, 3.0)`]
//!   (resolution-independent, `exp(mu) = shift`). An unset `scheduler` keeps that byte-exact; a curated
//!   epic-7114 name re-shapes σ over the same `mu = ln(3)` via [`mlx_gen::resolve_flow_schedule`].
//! * **Timestep convention.** The unified sampler hands the predict closure `ms.timestep(σ) = σ`
//!   ([`TimestepConvention::Sigma`]); the SANA trunk embeds the diffusers-scale timestep `σ · 1000`
//!   (`num_train_timesteps`), so the closure scales it before the forward (identical to SD3's MMDiT).
//!   The Euler update itself stays in σ-space (`x += (σ_{t+1} − σ_t) · v`).
//!
//! ## CFG
//!
//! Base SANA is a **true-CFG** model (the Sprint CFG-free distilled variant is the LATER story
//! sc-8490). Each step runs the trunk TWICE — cond (prompt) + uncond (negative/empty prompt) — and
//! combines `pred = uncond + scale · (cond − uncond)` (diffusers `SanaPipeline.__call__` default
//! `guidance_scale = 4.5`). When `guidance_scale <= 1.0` the uncond forward is skipped (CFG off),
//! matching diffusers' `do_classifier_free_guidance = guidance_scale > 1.0`.
//!
//! ## DC-AE latent scaling
//!
//! diffusers `SanaPipeline` decodes `latents / vae.config.scaling_factor` (the DC-AE
//! `scaling_factor = 0.41407`, [`DcAeConfig::scaling_factor`]); [`DcAeDecoder::decode`] expects the
//! **already-unscaled** latent, so the division is applied here before decode. The decoder emits NHWC
//! `[1, H, W, 3]`; [`mlx_gen::image::decoded_to_image`] expects NCHW, so the output is transposed back
//! to NCHW before the `clip(x·0.5 + 0.5)` → RGB8 conversion.

use mlx_gen::attention::AttentionBudget;
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::image::decoded_to_image;
use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::{
    run_flow_sampler_with_latent_hook, CancelFlag, Error, FlowMatchEuler, Image, PreviewSink,
    Progress, Result, StagedHeavy, TimestepConvention,
};
use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::{random, Array};

use crate::config::DcAeConfig;
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::scm::ScmScheduler;
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// DC-AE f32c32 latent channel count (the SANA trunk's `out_channels`).
pub const LATENT_CHANNELS: i32 = 32;
/// DC-AE deep-compression spatial downsample (latent edge is image/32).
pub const SPATIAL_SCALE: u32 = 32;
/// diffusers `num_train_timesteps` — the SANA trunk embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SANA-1.6B static flow-match shift (`scheduler_config.json` `flow_shift = 3.0`, no dynamic
/// shifting). The repo default solver is DPM-Solver; we run flow-match Euler over this shift by
/// design — see the module doc's Sampler/shift section for why (sc-11760).
pub const SCHEDULE_SHIFT: f32 = 3.0;
/// diffusers `SanaPipeline` default `num_inference_steps`.
pub const DEFAULT_STEPS: usize = 20;
/// diffusers `SanaPipeline` default `guidance_scale`.
pub const DEFAULT_GUIDANCE: f32 = 4.5;

/// Seeded txt2img latent noise — shape `[1, 32, height/32, width/32]`, f32. diffusers
/// `randn_tensor([B, 32, H/32, W/32])`; we draw f32 via `mx.random.normal` keyed on `seed`.
/// (`init_noise_sigma = 1.0` for flow-match, so the latent is the raw normal draw.)
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

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the SANA trunk twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The trunk timestep is `σ·1000`. When `guidance_scale`
/// is `<= 1.0` the uncond branch is skipped (CFG off, one forward per step; diffusers parity).
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &SanaTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    uncond: Option<&Array>,
    uncond_mask: Option<&Array>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_cfg_with_preview(
        transformer,
        scheduler,
        sampler_name,
        start_step,
        seed,
        latents,
        cond,
        cond_mask,
        uncond,
        uncond_mask,
        guidance_scale,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

/// [`denoise_cfg`] with an optional best-effort preview of each actual outer solver step.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_with_preview(
    transformer: &SanaTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    uncond: Option<&Array>,
    uncond_mask: Option<&Array>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_cfg_with_memory(
        transformer,
        scheduler,
        sampler_name,
        start_step,
        seed,
        latents,
        cond,
        cond_mask,
        uncond,
        uncond_mask,
        guidance_scale,
        cancel,
        on_progress,
        preview,
        crate::transformer::SanaForwardPlan::RESIDENT,
    )
}

/// [`denoise_cfg_with_preview`] under an explicit memory plan (SC-15523 rungs 3 and 4).
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_cfg_with_memory(
    transformer: &SanaTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    uncond: Option<&Array>,
    uncond_mask: Option<&Array>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    plan: crate::transformer::SanaForwardPlan,
) -> Result<Array> {
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        // The unified flow sampler hands `timestep = σ`; the SANA trunk embeds `σ·1000`.
        let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
        let pred_cond =
            transformer.forward_with_memory(x, cond, &t, None, cond_mask, plan, cancel)?;
        match uncond {
            Some(uc) if guidance_scale > 1.0 => {
                let pred_uncond =
                    transformer.forward_with_memory(x, uc, &t, None, uncond_mask, plan, cancel)?;
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
    // img2img runs the tail of the schedule (`sigmas[start_step..]`); txt2img passes `start_step = 0`
    // → the full schedule, byte-identical to the pre-img2img path. The pre-noised init latent (blended
    // at `sigmas[start_step]` by the caller) is the loop's starting point.
    let sigmas = &scheduler.sigmas[start_step.min(scheduler.sigmas.len().saturating_sub(1))..];
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
            crate::preview::emit_base_preview(preview, &previews, sigmas, sigma, latents);
        },
        predict,
    )
}

/// DC-AE-decode the final `[1, 32, H/32, W/32]` latent → an RGB8 [`Image`]. diffusers
/// `SanaPipeline` divides by `vae.config.scaling_factor` before decode; the decoder emits NHWC and
/// [`decoded_to_image`] expects NCHW, so the result is transposed back before the RGB8 conversion.
pub fn decode_to_image(
    decoder: &DcAeDecoder,
    cfg: &DcAeConfig,
    latents: &Array,
    cancel: &CancelFlag,
) -> Result<Image> {
    decode_to_image_with_tiling(decoder, cfg, latents, cancel, None)
}

fn decode_to_image_with_tiling(
    decoder: &DcAeDecoder,
    cfg: &DcAeConfig,
    latents: &Array,
    cancel: &CancelFlag,
    tiling: Option<&TilingConfig>,
) -> Result<Image> {
    let scale = Array::from_slice(&[cfg.scaling_factor], &[1]);
    let unscaled = divide(latents, &scale)?; // diffusers: latents / scaling_factor
    let decoded_nhwc = match tiling {
        Some(tiling) => decode_tiled(decoder, &unscaled, tiling, cancel)?,
        None => decoder.decode(&unscaled, cancel)?,
    }; // [1, H, W, 3] NHWC, f32
    let decoded_nchw = decoded_nhwc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for decoded_to_image
    decoded_to_image(&decoded_nchw)
}

/// Measured production DC-AE tile domain. All edges use one fixed 48-pixel overlap; edges below
/// 192 reached the same 3294-MiB request floor while degrading output and remain a rejection set.
///
/// The overlap is quantized by the 32x DC-AE scale — the shared tiling plan computes
/// `overlap_px / 32` **latent cells**, so 48 px is ONE blended latent cell and anything below
/// 32 px is no blend at all. Both ends of that lever are measured (sc-17863, `sana_1600m` q4 at
/// 1024², tiled vs whole-image decode): dropping the blend (24 px = 0 cells) produces visible
/// blocky patches and meanD 6.31..6.43 — past the declared 6.0 ceiling — while widening it to
/// 96 px (3 cells) cuts meanD to 3.04..3.28 at a request peak IDENTICAL to four decimals
/// (3.2172 GiB; overlap adds tiles, not bigger tiles) but costs +1..4 s of decode wall per
/// render, which Sprint's 2-step schedule cannot absorb. 48 px is the adjudicated shipping
/// point: no visible artifact at 1:1, the memory floor intact, and the sweep table at
/// `DECODE_TILING_MEAN_ABS_U8` (tests/memory_ladder_real_weights.rs) records the measured menu
/// for any future quality-first move.
///
/// **sc-19753 — that sweep measured a superseded mechanism.** Every row above was captured while
/// [`decode_tiled`] tiled the *whole* decoder, so each tile's nine `EfficientVit` blocks aggregated
/// their ReLU-linear attention over that tile's tokens instead of the image. That is why widening
/// the overlap to 96 px only floored at meanD ~3.0 instead of converging: overlap width cannot fix
/// a per-tile global reduction. The attention now runs once in the dense head and only the
/// attention-free tail tiles, so the residual these numbers describe is expected to be
/// substantially smaller. **The recorded values are accepted ceilings, not targets** — they are
/// left untouched here (a better result still passes) and re-capturing them is a measurement
/// campaign, not part of this fix.
pub const DECODE_TILE_EDGE: i32 = 192;
pub const DECODE_OVERLAP: i32 = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeTilingSource {
    EnvOverride,
    Request,
    SequentialDefault,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeTilingPlan {
    pub edge: i32,
    pub overlap: i32,
    pub source: DecodeTilingSource,
}

/// Resolve the actual DC-AE geometry with total precedence: request-scoped shared-contract signal,
/// admitted measurement override, then the Sequential shipping default. The calibration harness
/// cannot supersede an admitted request or run unpublished geometry. Resident stays whole-image
/// unless the caller explicitly selects bounded decode.
pub fn resolved_decode_plan(
    memory: Option<GenerationMemory>,
    is_sequential: bool,
) -> Option<DecodeTilingPlan> {
    if let Some(memory) = memory {
        if !memory.tile_vae_decode {
            return None;
        }
        return Some(DecodeTilingPlan {
            edge: memory
                .decode_tile_edge
                .map_or(DECODE_TILE_EDGE, |edge| edge as i32),
            overlap: memory
                .decode_overlap
                .map_or(DECODE_OVERLAP, |overlap| overlap as i32),
            source: DecodeTilingSource::Request,
        });
    }
    if let Some(edge) = std::env::var("MLX_GEN_SANA_DECODE_TILE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
    {
        if edge == 0 {
            return None;
        }
        if edge > 0 && crate::memory_strategy::DECODE_TILE_EDGES.contains(&(edge as u32)) {
            return Some(DecodeTilingPlan {
                edge,
                overlap: DECODE_OVERLAP,
                source: DecodeTilingSource::EnvOverride,
            });
        }
    }
    is_sequential.then_some(DecodeTilingPlan {
        edge: DECODE_TILE_EDGE,
        overlap: DECODE_OVERLAP,
        source: DecodeTilingSource::SequentialDefault,
    })
}

pub fn resolve_decode_tiling(
    memory: Option<GenerationMemory>,
    is_sequential: bool,
) -> Option<TilingConfig> {
    resolved_decode_plan(memory, is_sequential)
        .map(|plan| TilingConfig::spatial_only(plan.edge, plan.overlap))
}

/// Resolve the two **denoise-phase** constrained rungs from the request-scoped shared-contract
/// signal (SC-15523). Rung 2 has its own resolver above; this is rungs 3 and 4.
///
/// Nothing is selected by default: a request that sets neither flag gets
/// [`SanaForwardPlan::RESIDENT`], which is byte-for-byte the pre-SC-15523 trunk forward. `n_blocks`
/// comes from the loaded trunk rather than from the config constant, so a plan can never describe a
/// different stack than the one that will run it.
///
/// Parameter *values* are validated against the published domain before this runs — see
/// [`crate::memory_strategy::validate_request_memory`], which the production `generate` path calls
/// and which refuses an out-of-domain selection rather than silently executing an unmeasured one.
pub(crate) fn resolved_rung_plan(
    memory: Option<GenerationMemory>,
    n_blocks: usize,
) -> Result<crate::transformer::SanaForwardPlan> {
    let Some(memory) = memory else {
        return Ok(crate::transformer::SanaForwardPlan::RESIDENT);
    };
    let attention = if memory.chunk_attention {
        AttentionBudget::from_score_elements(
            u64::from(
                memory
                    .attention_chunk_size
                    .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
            ),
            // The per-chunk graph cut is where MLX's saving comes from; a lazily-chunked budget
            // measured as no better than unbounded on the sibling families (SC-15615).
            true,
        )
    } else {
        AttentionBudget::UNBOUNDED
    };
    let window = memory
        .stream_transformer_blocks
        .then(|| {
            BlockPlan::new(
                n_blocks,
                memory
                    .transformer_window_size
                    .unwrap_or(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
                    as usize,
            )
        })
        .transpose()?;
    Ok(crate::transformer::SanaForwardPlan { attention, window })
}

/// Decode one NCHW DC-AE latent through the shared 5-D tiled-VAE seam. SANA's decoder emits NHWC,
/// so a dummy temporal axis keeps the head-feature and decoded tile spatial axes aligned.
///
/// **Normalization semantics (sc-19753).** Only the decoder's shallow, attention-free
/// [`DcAeDecoder::decode_tail`] is tiled. The deep `EfficientVit` stages run once on the whole
/// latent via [`DcAeDecoder::decode_head`], because their ReLU-linear attention contracts over
/// **every** `H·W` token — evaluating one on a crop aggregates over that crop instead of the image,
/// the same defect class as a per-tile GroupNorm. This route previously tiled the *entire* decoder,
/// putting nine such attention blocks inside the per-tile closure.
///
/// The tile plan is therefore keyed on the **tail's** ×8 upsample rather than the whole decoder's
/// ×32: the tiles now partition the head feature map, which is already ×4 the latent.
///
/// The candle sibling (`candle_gen_sana::DcAeDecoder::decode_with`) has had this split since
/// sc-11804; the MLX lane is the one that had not been converted.
fn decode_tiled(
    decoder: &DcAeDecoder,
    latent: &Array,
    tiling: &TilingConfig,
    cancel: &CancelFlag,
) -> Result<Array> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // A config with no attention-free shallow run has no tileable tail at all.
    if decoder.num_tail_stages() == 0 {
        return decoder.decode(latent, cancel);
    }
    let head = decoder.decode_head(latent)?; // NHWC, at the tail's input resolution
    let shape = head.shape();
    let (height, width, channels) = (shape[1], shape[2], shape[3]);
    let vae = VaeTiling {
        spatial_scale: decoder.tail_scale(),
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 3,
    };
    if !tiling.needs_tiling(vae, 1, height, width) {
        return decoder.decode_tail(&head);
    }
    let plan = tiling.plan(vae, 1, height, width);
    let lifted = head.reshape(&[1, 1, height, width, channels])?;
    let out = mlx_gen::vae_tiling::tiled_decode(&lifted, &plan, [1, 2, 3], Some(cancel), |tile| {
        let shape = tile.shape();
        let tile = tile.reshape(&[1, shape[2], shape[3], shape[4]])?;
        let decoded = decoder.decode_tail(&tile)?;
        let shape = decoded.shape();
        Ok(decoded.reshape(&[1, 1, shape[1], shape[2], shape[3]])?)
    })?;
    let shape = out.shape();
    Ok(out.reshape(&[1, shape[2], shape[3], shape[4]])?)
}

// =================================================================================================
// SANA-Sprint: continuous-time-consistency (SCM/TrigFlow), CFG-free, 1–4 step (sc-8490).
// =================================================================================================

/// diffusers `SanaSprintPipeline` default `num_inference_steps`.
pub const SPRINT_DEFAULT_STEPS: usize = 2;
/// diffusers `SanaSprintPipeline` default `guidance_scale` (embedded, NOT classifier-free).
pub const SPRINT_DEFAULT_GUIDANCE: f32 = 4.5;

fn arr1(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// One SCM (TrigFlow continuous-time consistency) denoise — the **CFG-free, few-step** SANA-Sprint
/// loop. A faithful port of the diffusers `SanaSprintPipeline` denoise + `SCMScheduler.step`:
///
/// 1. seed the latent and pre-scale by `sigma_data` (the diffusers `latents = latents * sigma_data`);
/// 2. per step `i` over the angle schedule `t = scheduler.timesteps[i]`:
///    * `scm_t = sin(t)/(cos(t)+sin(t))`; model input = `(latents / sigma_data) · sqrt(scm_t² + (1−scm_t)²)`;
///    * ONE trunk forward with the **embedded guidance scalar** (`guidance · guidance_embeds_scale`)
///      and `timestep = scm_t` (no uncond branch — Sprint is CFG-free);
///    * recombine the raw output trigonometrically, `· sigma_data`;
///    * `SCMScheduler.step`: `x0 = cos(s)·x − sin(s)·output`; renoise `x = cos(t')·x0 + sin(t')·noise·sigma_data`
///      (skipped on the final step / single-step schedule);
/// 3. return `denoised / sigma_data` (the diffusers `latents = denoised / sigma_data`).
///
/// The per-step `eval` boundary + cooperative cancel + monotone progress mirror the unified
/// [`mlx_gen::run_flow_sampler`] run-loop contract (the epic-7114 seam SCM reuses; its trigflow step
/// is the consistency parameterization the flow-match `Solver` menu cannot represent).
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_sprint_with_preview(
        transformer,
        scheduler,
        seed,
        latents,
        cond,
        cond_mask,
        guidance_scale,
        guidance_embeds_scale,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

/// [`denoise_sprint`] with an optional best-effort preview of each SCM step.
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_with_preview(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    // txt2img: seed the SCM prior (`latents * sigma_data`) and run the whole angle schedule (start 0).
    let latents = multiply(&latents, arr1(scheduler.sigma_data))?;
    denoise_sprint_from_with_preview(
        transformer,
        scheduler,
        0,
        seed,
        latents,
        cond,
        cond_mask,
        guidance_scale,
        guidance_embeds_scale,
        cancel,
        on_progress,
        preview,
    )
}

/// The SCM (TrigFlow) few-step denoise loop starting at angle index `start_step`, over an **already
/// `sigma_data`-scaled** `latents` (the caller seeds it: txt2img = `noise · σ_data`; img2img =
/// `cos(t)·x0 + sin(t)·noise·σ_data` renoised to `t = timesteps[start_step]`). `start_step = 0` runs
/// the full schedule (the txt2img path, via [`denoise_sprint`]). Progress is reported over the steps
/// actually run (`n - start_step`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_from(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_sprint_from_with_preview(
        transformer,
        scheduler,
        start_step,
        seed,
        latents,
        cond,
        cond_mask,
        guidance_scale,
        guidance_embeds_scale,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

/// [`denoise_sprint_from`] with an optional best-effort preview of each SCM step.
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_from_with_preview(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_sprint_from_with_memory(
        transformer,
        scheduler,
        start_step,
        seed,
        latents,
        cond,
        cond_mask,
        guidance_scale,
        guidance_embeds_scale,
        cancel,
        on_progress,
        preview,
        crate::transformer::SanaForwardPlan::RESIDENT,
    )
}

/// [`denoise_sprint_from_with_preview`] under an explicit memory plan (SC-15523 rungs 3 and 4).
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_sprint_from_with_memory(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    plan: crate::transformer::SanaForwardPlan,
) -> Result<Array> {
    use mlx_rs::transforms::eval;

    let sd = scheduler.sigma_data;
    let mut latents = latents;

    // The embedded guidance scalar (CFG-free): guidance_scale * guidance_embeds_scale, a [1] tensor
    // fed to the trunk's guidance embedder. Constant across steps.
    let guidance = arr1(guidance_scale * guidance_embeds_scale);

    let n = scheduler.num_steps();
    let start = start_step.min(n);
    let total = (n - start).max(1) as u32;
    let preview_schedule = &scheduler.timesteps[start..];
    let previews = mlx_gen::preview::PreviewCounter::new(preview_schedule);
    let mut denoised = latents.clone();
    // Per-step renoise key — a distinct subkey per step so the between-step noise is decorrelated and
    // deterministic for a given request seed (mirrors the unified sampler's `StepRng` derivation).
    let step_key = |step: usize| -> Result<Array> {
        let sub = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        Ok(random::key(sub)?)
    };

    for i in start..n {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Per-eval compute boundary (MLX is lazy): force the prior step's graph so cancel/progress are
        // responsive rather than deferred to decode.
        eval([&latents])?;
        on_progress(Progress::Step {
            current: (i - start) as u32 + 1,
            total,
        });

        let s = scheduler.timesteps[i];
        crate::preview::emit_sprint_preview(
            preview,
            &previews,
            preview_schedule,
            s,
            &latents,
            1.0 / sd,
        );
        let t_next = scheduler.timesteps[i + 1];
        let scm_t = scheduler.scm_timestep(i);
        let in_scale = scheduler.input_scale(i);

        // model input = (latents / sigma_data) * sqrt(scm_t² + (1-scm_t)²).
        let lat_in = multiply(&divide(&latents, arr1(sd))?, arr1(in_scale))?;
        let scm_t_arr = arr1(scm_t);
        let raw = transformer.forward_with_memory(
            &lat_in,
            cond,
            &scm_t_arr,
            Some(&guidance),
            cond_mask,
            plan,
            cancel,
        )?;

        // diffusers trigflow recombination of the raw output (uses `latent_model_input` = the SCALED
        // `lat_in`, NOT the un-scaled latent):
        //   noise_pred = ((1-2·scm_t)·lat_in + (1-2·scm_t+2·scm_t²)·raw) / sqrt(scm_t²+(1-scm_t)²)
        //   noise_pred = noise_pred * sigma_data
        let a = 1.0 - 2.0 * scm_t;
        let b = 1.0 - 2.0 * scm_t + 2.0 * scm_t * scm_t;
        let model_output = multiply(
            &divide(
                &add(&multiply(&lat_in, arr1(a))?, &multiply(&raw, arr1(b))?)?,
                arr1(in_scale),
            )?,
            arr1(sd),
        )?;

        // SCMScheduler.step (trigflow x0-pred + renoise). `s` = current angle, `t_next` = next angle.
        // pred_x0 = cos(s)·latents − sin(s)·model_output.
        let pred_x0 = subtract(
            &multiply(&latents, arr1(s.cos()))?,
            &multiply(&model_output, arr1(s.sin()))?,
        )?;
        denoised = pred_x0.clone();
        // Renoise to the next angle (skipped on the final / single-step transition, matching diffusers
        // `if len(self.timesteps) > 1`). On the last step `t_next == 0` ⇒ `cos(0)=1`, `sin(0)=0`, so the
        // renoise reduces to exactly `pred_x0` — gate the noise DRAW on a non-terminal step (`i+1 < n`)
        // so the final step doesn't burn a wasted `random::normal` + key derivation (F-092; bit-exact,
        // the drawn noise was multiplied by `sin(0)=0` anyway).
        latents = if scheduler.is_single_step() || i + 1 >= n {
            pred_x0
        } else {
            let noise = multiply(
                &random::normal::<f32>(latents.shape(), None, None, Some(&step_key(i)?))?,
                arr1(sd),
            )?;
            add(
                &multiply(&pred_x0, arr1(t_next.cos()))?,
                &multiply(&noise, arr1(t_next.sin()))?,
            )?
        };
    }

    // diffusers: latents = denoised / sigma_data (the decode input).
    let out = divide(&denoised, arr1(sd))?;
    eval([&out])?;
    Ok(out)
}

/// SANA prompt conditioning materialized by [`encode_conditioning`] — the Gemma-2 CHI last-hidden
/// caption embedding + its pad mask, plus the optional uncond twin (base SANA true-CFG only; `None`
/// for Sprint / CFG-off). The phase-A output of the shared [`mlx_gen::Residency`] seam: it is
/// `eval`ed and the text encoder dropped before the trunk/VAE load under `Sequential`.
pub struct SanaConditioning {
    /// Positive-prompt CHI embedding `[1, 300, 2304]`.
    pub cond: Array,
    /// Positive-prompt caption pad mask (`attn2` cross-attention key mask).
    pub cond_mask: Array,
    /// `(uncond, uncond_mask)` — Some only for base SANA with CFG active (`guidance > 1.0`); `None`
    /// for Sprint (CFG-free) and for a CFG-off base request.
    pub uncond: Option<(Array, Array)>,
}

/// Encode the prompt (and, for base SANA with CFG active, the negative prompt) into [`SanaConditioning`]
/// — the phase-A step of the shared residency seam. Encodes WITH the caption pad mask (SANA's `attn2`
/// masks PAD keys; dropping it lets padding swamp short-prompt conditioning). `sprint` and the resolved
/// `guidance` gate the uncond forward: `!sprint && guidance > 1.0` (diffusers'
/// `do_classifier_free_guidance = guidance_scale > 1.0`); Sprint is CFG-free (embedded scalar), so it
/// never encodes an uncond. Seed-independent and RNG-free, so hoisting it above the per-image render
/// loop is byte-identical to the pre-seam per-image encode.
pub fn encode_conditioning(
    text_encoder: &SanaTextEncoder,
    sprint: bool,
    prompt: &str,
    negative_prompt: Option<&str>,
    guidance: f32,
) -> Result<SanaConditioning> {
    let (cond, cond_mask) = text_encoder.encode_with_mask(prompt)?;
    let uncond = if !sprint && guidance > 1.0 {
        let (u, um) = text_encoder.encode_with_mask(negative_prompt.unwrap_or(""))?;
        Some((u, um))
    } else {
        None
    };
    Ok(SanaConditioning {
        cond,
        cond_mask,
        uncond,
    })
}

/// The heavy render-phase components (everything but the Gemma text encoder): the Linear-DiT trunk, the
/// DC-AE encoder (img2img) + decoder, the DC-AE config, and the variant flags. Owned by the `Resident`
/// components (held for the whole job) or by a `Sequential` generate (loaded after the text encoder is
/// dropped, freed when the job ends). `sprint` selects the denoise path: `false` = base SANA-1.6B
/// (true-CFG flow-match Euler); `true` = SANA-Sprint (CFG-free SCM/TrigFlow few-step, sc-8490). SANA has
/// no PiD/control overlay, so the heavy bundle is just trunk + VAE.
pub struct SanaHeavy {
    transformer: SanaTransformer,
    /// DC-AE **encoder** — the img2img reference→latent path (sc-10190). Loaded from the SAME
    /// `vae/` snapshot as the decoder (the checkpoint ships both `encoder.*` and `decoder.*` keys).
    encoder: Option<DcAeEncoder>,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
    sprint: bool,
    guidance_embeds_scale: f32,
}

/// The composed SANA text-to-image pipeline: text encoder + heavy render bundle (trunk + DC-AE), with
/// the DC-AE config (for the latent `scaling_factor`). A clean `generate` entrypoint mirroring the
/// sibling flow-match pipelines (`mlx-gen-sd3`). The gen-core `Generator` adapter drives the
/// component-split ([`SanaTextEncoder`] + [`SanaHeavy`]) directly through the shared
/// [`mlx_gen::Residency`] seam; this composed type is the standalone / real-weight-contract entrypoint
/// and shares the exact [`encode_conditioning`] + [`SanaHeavy::render_one`] bodies, so both are
/// byte-identical.
///
/// `sprint` selects the variant: `false` = base SANA-1.6B (true-CFG flow-match Euler); `true` =
/// SANA-Sprint (CFG-free SCM/TrigFlow few-step, sc-8490). The trunk must be loaded with the matching
/// config (`SanaTransformerConfig::sana_sprint_1600m()` for Sprint — its guidance embedder +
/// rms-norm-across-heads are config-gated).
pub struct SanaPipeline {
    text_encoder: SanaTextEncoder,
    heavy: SanaHeavy,
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
    /// **img2img** reference image (sc-10190): when present with a positive [`Self::strength`], the
    /// DC-AE-encoded init latent seeds the denoise instead of pure noise. `None` = plain txt2img.
    pub init_image: Option<&'a Image>,
    /// img2img strength ∈ `(0, 1]` (the fork's `init_time_step` convention: higher → start later →
    /// output stays closer to the init image). `None` (or with no `init_image`) = txt2img.
    pub strength: Option<f32>,
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

/// VAE-encode an img2img reference image into a **denoise-space** DC-AE latent
/// `[1, latent_channels, H/32, W/32]` (sc-10190): LANCZOS-resize + `[-1,1]` NCHW preprocess →
/// [`DcAeEncoder::encode`] → multiply by the DC-AE `scaling_factor`. The `scaling_factor` places the
/// latent in the same space the denoise loop + [`decode_to_image`] (which divides it back) operate in.
pub fn encode_init_latents(
    encoder: &DcAeEncoder,
    cfg: &DcAeConfig,
    image: &Image,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, width, height)?;
    let raw = encoder.encode(&image_nchw, cancel)?; // [1, 32, H/32, W/32], raw (pre-scale)
    Ok(multiply(&raw, arr1(cfg.scaling_factor))?)
}

impl SanaHeavy {
    /// Compose the **base SANA-1.6B** heavy bundle (true-CFG flow-match) from its render-phase
    /// components plus the DC-AE config (used for the latent `scaling_factor`).
    pub fn new(
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            transformer,
            encoder: Some(encoder),
            decoder,
            dc_ae_cfg,
            sprint: false,
            guidance_embeds_scale: 0.0,
        }
    }

    pub fn new_text_to_image(
        transformer: SanaTransformer,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            transformer,
            encoder: None,
            decoder,
            dc_ae_cfg,
            sprint: false,
            guidance_embeds_scale: 0.0,
        }
    }

    /// Compose the **SANA-Sprint** heavy bundle (CFG-free SCM/TrigFlow few-step, sc-8490). The
    /// `transformer` MUST be loaded with [`crate::SanaTransformerConfig::sana_sprint_1600m`] (its
    /// guidance embedder + rms-norm-across-heads are required for the embedded-guidance forward).
    /// `guidance_embeds_scale` is the trunk config's `guidance_embeds_scale` (`0.1`), pre-multiplied
    /// into the guidance scalar before the embedder.
    pub fn new_sprint(
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
        guidance_embeds_scale: f32,
    ) -> Self {
        Self {
            transformer,
            encoder: Some(encoder),
            decoder,
            dc_ae_cfg,
            sprint: true,
            guidance_embeds_scale,
        }
    }

    pub fn new_sprint_text_to_image(
        transformer: SanaTransformer,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
        guidance_embeds_scale: f32,
    ) -> Self {
        Self {
            transformer,
            encoder: None,
            decoder,
            dc_ae_cfg,
            sprint: true,
            guidance_embeds_scale,
        }
    }

    /// Whether this heavy bundle drives the SANA-Sprint (CFG-free few-step) path.
    pub fn is_sprint(&self) -> bool {
        self.sprint
    }

    /// The resolved default guidance for this variant (base 4.5, Sprint's embedded 4.5) — used by the
    /// caller to resolve `req.guidance_scale` consistently for BOTH the encode's uncond decision and
    /// the render's denoise, so the two never disagree.
    pub fn default_guidance(&self) -> f32 {
        if self.sprint {
            SPRINT_DEFAULT_GUIDANCE
        } else {
            DEFAULT_GUIDANCE
        }
    }

    /// Render ONE image from pre-encoded [`SanaConditioning`] + a per-image request. `guidance` is the
    /// already-resolved guidance scale (the caller resolved it against [`Self::default_guidance`] so the
    /// encode's uncond decision and this render agree). Branches on `sprint`. Byte-identical to the tail
    /// of the pre-seam `generate_with` / `generate_sprint` (the encode is hoisted out; everything below
    /// is unchanged). Seed-carrying via `req.seed`.
    pub fn render_one(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.render_one_with_preview(
            cond,
            req,
            guidance,
            cancel,
            on_progress,
            &PreviewSink::default(),
        )
    }

    /// [`Self::render_one`] with an optional best-effort native-latent preview sink.
    pub fn render_one_with_preview(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
    ) -> Result<Image> {
        self.render_one_with_preview_and_tiling(
            cond,
            req,
            guidance,
            cancel,
            on_progress,
            preview,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_one_with_preview_and_tiling(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
        tiling: Option<&TilingConfig>,
    ) -> Result<Image> {
        let latents = self.denoise_one_with_preview(
            cond,
            req,
            guidance,
            cancel,
            on_progress,
            preview,
            crate::transformer::SanaForwardPlan::RESIDENT,
        )?;
        mlx_rs::transforms::eval([&latents])?;
        on_progress(Progress::Decoding);
        self.decode_view().decode_one(&latents, cancel, tiling)
    }

    /// The trunk's block count — the rung-4 plan's `n_blocks`, read from the loaded trunk rather
    /// than from a config constant.
    pub(crate) fn transformer_blocks(&self) -> usize {
        self.transformer.n_blocks()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise_one_with_preview(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
        plan: crate::transformer::SanaForwardPlan,
    ) -> Result<Array> {
        if self.sprint {
            self.denoise_sprint(cond, req, guidance, cancel, on_progress, preview, plan)
        } else {
            self.denoise_cfg(cond, req, guidance, cancel, on_progress, preview, plan)
        }
    }

    /// The base SANA-1.6B true-CFG flow-match denoise for one image.
    #[allow(clippy::too_many_arguments)]
    fn denoise_cfg(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
        plan: crate::transformer::SanaForwardPlan,
    ) -> Result<Array> {
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        let seed = req.seed.unwrap_or(0);

        // Static shift=3.0 schedule (scheduler_config.json), resolution-independent — build once. An
        // unset scheduler keeps it byte-exact; a curated name re-shapes σ over the same mu=ln(3).
        let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
        let scheduler = FlowMatchEuler::from_sigmas(mlx_gen::resolve_flow_schedule(
            req.scheduler,
            SCHEDULE_SHIFT.ln(),
            steps,
            &native.sigmas,
        ))?;

        // img2img (sc-10190): a reference image + positive strength starts the denoise at
        // `sigmas[start_step]` over the DC-AE-encoded init latent blended with noise; else start 0
        // (pure-noise txt2img). `init_time_step` returns 0 when strength is None/≤0 (→ txt2img).
        let start_step = match req.init_image {
            Some(_) => init_time_step(steps, req.strength),
            None => 0,
        };
        let clean = if start_step > 0 {
            let image = req
                .init_image
                .expect("start_step > 0 implies an init image");
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg("SANA text-to-image bundle cannot encode an init image".into())
            })?;
            Some(encode_init_latents(
                encoder,
                &self.dc_ae_cfg,
                image,
                req.width,
                req.height,
                cancel,
            )?)
        } else {
            None
        };

        let noise = create_noise(seed, req.width, req.height)?;
        let latents = match &clean {
            // Blend the pre-encoded clean latents with the noise at `sigma = sigmas[start_step]`.
            Some(clean) => {
                let sigma = *scheduler.sigmas.get(start_step).ok_or_else(|| {
                    Error::Msg(format!(
                        "sana img2img: start step {start_step} out of range for {}-element schedule",
                        scheduler.sigmas.len()
                    ))
                })?;
                add_noise_by_interpolation(clean, &noise, sigma)?
            }
            None => noise,
        };
        // The uncond twin is present only for base SANA with CFG active (`encode_conditioning`).
        let uncond = cond.uncond.as_ref().map(|(u, _)| u);
        let uncond_mask = cond.uncond.as_ref().map(|(_, um)| um);
        let latents = denoise_cfg_with_memory(
            &self.transformer,
            &scheduler,
            req.sampler,
            start_step,
            seed,
            latents,
            &cond.cond,
            Some(&cond.cond_mask),
            uncond,
            uncond_mask,
            guidance,
            cancel,
            on_progress,
            preview,
            plan,
        )?;
        Ok(latents)
    }

    /// The **SANA-Sprint** (CFG-free SCM/TrigFlow few-step) render for one image (the pre-seam
    /// `generate_sprint` tail). The negative prompt / curated sampler+scheduler knobs are inapplicable
    /// to the SCM loop and ignored; `cond.uncond` is always `None` for Sprint.
    ///
    /// **img2img (sc-10190):** a reference image + positive strength starts the SCM loop at angle
    /// index `start = init_time_step(n, strength)`, seeding `latents` by TrigFlow-renoising the
    /// DC-AE-encoded init to that angle: `x_t = cos(t)·x0 + sin(t)·noise·σ_data` with `x0 =
    /// encode·scaling_factor·σ_data` and `t = timesteps[start]`. Distilled/consistency, so the strength
    /// window is narrow — validate the band on-device. `start = 0` is the byte-identical txt2img path.
    #[allow(clippy::too_many_arguments)]
    fn denoise_sprint(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &PreviewSink,
        plan: crate::transformer::SanaForwardPlan,
    ) -> Result<Array> {
        let steps = req.steps.unwrap_or(SPRINT_DEFAULT_STEPS);
        let seed = req.seed.unwrap_or(0);

        let scheduler = ScmScheduler::new(steps);
        let n = scheduler.num_steps();
        let sd = scheduler.sigma_data;

        let start_step = match req.init_image {
            Some(_) => init_time_step(n, req.strength),
            None => 0,
        };
        let noise = create_noise(seed, req.width, req.height)?;
        let latents = if start_step > 0 {
            // img2img: renoise the encoded init to the start angle `timesteps[start_step]`.
            let image = req
                .init_image
                .expect("start_step > 0 implies an init image");
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg("SANA text-to-image bundle cannot encode an init image".into())
            })?;
            let clean = encode_init_latents(
                encoder,
                &self.dc_ae_cfg,
                image,
                req.width,
                req.height,
                cancel,
            )?;
            // x0 in the SCM prior space (σ_data-scaled); noise likewise. TrigFlow renoise to angle t.
            let x0 = multiply(&clean, arr1(sd))?;
            let noise_sd = multiply(&noise, arr1(sd))?;
            let t = scheduler.timesteps[start_step];
            add(
                &multiply(&x0, arr1(t.cos()))?,
                &multiply(&noise_sd, arr1(t.sin()))?,
            )?
        } else {
            // txt2img: the SCM prior is `noise · σ_data`.
            multiply(&noise, arr1(sd))?
        };
        let latents = denoise_sprint_from_with_memory(
            &self.transformer,
            &scheduler,
            start_step,
            seed,
            latents,
            &cond.cond,
            Some(&cond.cond_mask),
            guidance,
            self.guidance_embeds_scale,
            cancel,
            on_progress,
            preview,
            plan,
        )?;
        Ok(latents)
    }
}

pub struct SanaLight {
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
}

pub struct SanaDecodeView<'a> {
    decoder: &'a DcAeDecoder,
    dc_ae_cfg: &'a DcAeConfig,
}

impl SanaDecodeView<'_> {
    pub fn decode_one(
        &self,
        latents: &Array,
        cancel: &CancelFlag,
        tiling: Option<&TilingConfig>,
    ) -> Result<Image> {
        decode_to_image_with_tiling(self.decoder, self.dc_ae_cfg, latents, cancel, tiling)
    }
}

impl mlx_gen::StagedHeavy for SanaHeavy {
    type Light = SanaLight;
    type DecodeView<'a> = SanaDecodeView<'a>;

    fn shed_dit(self) -> SanaLight {
        SanaLight {
            decoder: self.decoder,
            dc_ae_cfg: self.dc_ae_cfg,
        }
    }

    fn decode_view(&self) -> SanaDecodeView<'_> {
        SanaDecodeView {
            decoder: &self.decoder,
            dc_ae_cfg: &self.dc_ae_cfg,
        }
    }

    fn light_view(light: &SanaLight) -> SanaDecodeView<'_> {
        SanaDecodeView {
            decoder: &light.decoder,
            dc_ae_cfg: &light.dc_ae_cfg,
        }
    }
}

impl SanaPipeline {
    /// Compose the **base SANA-1.6B** pipeline (true-CFG flow-match) from its four already-constructed
    /// components plus the DC-AE config (used for the latent `scaling_factor`).
    pub fn new(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            text_encoder,
            heavy: SanaHeavy::new(transformer, encoder, decoder, dc_ae_cfg),
        }
    }

    /// Compose the **SANA-Sprint** pipeline (CFG-free SCM/TrigFlow few-step, sc-8490). The
    /// `transformer` MUST be loaded with [`crate::SanaTransformerConfig::sana_sprint_1600m`] (its
    /// guidance embedder + rms-norm-across-heads are required for the embedded-guidance forward).
    /// `guidance_embeds_scale` is the trunk config's `guidance_embeds_scale` (`0.1`), pre-multiplied
    /// into the guidance scalar before the embedder.
    pub fn new_sprint(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
        guidance_embeds_scale: f32,
    ) -> Self {
        Self {
            text_encoder,
            heavy: SanaHeavy::new_sprint(
                transformer,
                encoder,
                decoder,
                dc_ae_cfg,
                guidance_embeds_scale,
            ),
        }
    }

    /// Whether this is a SANA-Sprint (CFG-free few-step) pipeline.
    pub fn is_sprint(&self) -> bool {
        self.heavy.is_sprint()
    }

    /// Run the full prompt→image pipeline. Encodes the prompt (and the negative prompt when CFG is
    /// active) ONCE, seeds the DC-AE latent, runs the flow-match Euler denoise over the SANA trunk
    /// with true CFG, then DC-AE-decodes to an RGB8 [`Image`].
    pub fn generate(&self, req: &SanaGenerateRequest<'_>) -> Result<Image> {
        let cancel = CancelFlag::default();
        let mut noop = |_: Progress| {};
        self.generate_with(req, &cancel, &mut noop)
    }

    /// [`SanaPipeline::generate`] with caller-supplied cancellation + progress (the seam Phase B's
    /// worker `Generator` adapter wires into the gen-core contract). Shares the exact
    /// [`encode_conditioning`] + [`SanaHeavy::render_one`] bodies the gen-core adapter's residency seam
    /// uses, so the composed and split paths are byte-identical.
    pub fn generate_with(
        &self,
        req: &SanaGenerateRequest<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        // F-091: `create_noise` derives the latent grid via `dim / SPATIAL_SCALE` integer division, so
        // a width/height not a multiple of 32 silently truncates the latent (and the output image) to
        // the floor multiple instead of honoring the request. Reject it up front (both the CFG and the
        // Sprint path funnel through here) rather than returning a quietly-smaller image.
        if !req.width.is_multiple_of(SPATIAL_SCALE) || !req.height.is_multiple_of(SPATIAL_SCALE) {
            return Err(Error::Msg(format!(
                "sana: width and height must be multiples of {SPATIAL_SCALE}, got {}x{}",
                req.width, req.height
            )));
        }
        // Resolve guidance against the variant default ONCE so the encode's uncond decision and the
        // render's denoise agree.
        let guidance = req.guidance_scale.unwrap_or(self.heavy.default_guidance());
        let cond = encode_conditioning(
            &self.text_encoder,
            self.heavy.is_sprint(),
            req.prompt,
            req.negative_prompt,
            guidance,
        )?;
        self.heavy
            .render_one(&cond, req, guidance, cancel, on_progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn decode_default_is_the_measured_192_at_fixed_48_overlap() {
        let _lock = env_lock();
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        assert!(resolved_decode_plan(None, false).is_none());
        assert_eq!(
            resolved_decode_plan(None, true),
            Some(DecodeTilingPlan {
                edge: 192,
                overlap: 48,
                source: DecodeTilingSource::SequentialDefault,
            })
        );
        assert!(resolved_decode_plan(Some(GenerationMemory::default()), true).is_none());
    }

    #[test]
    fn request_geometry_uses_the_single_published_overlap() {
        let _lock = env_lock();
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        let plan = resolved_decode_plan(
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                ..Default::default()
            }),
            false,
        )
        .unwrap();
        assert_eq!((plan.edge, plan.overlap), (512, 48));
        assert_eq!(plan.source, DecodeTilingSource::Request);
    }

    #[test]
    fn admitted_measurement_override_never_supersedes_a_contract_request() {
        let _lock = env_lock();
        let request = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(256),
            ..Default::default()
        });
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "512");
        let plan = resolved_decode_plan(request, true).unwrap();
        assert_eq!(plan.source, DecodeTilingSource::Request);
        assert_eq!((plan.edge, plan.overlap), (256, 48));
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
        assert_eq!(
            resolved_decode_plan(request, true).unwrap().source,
            DecodeTilingSource::Request
        );

        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "512");
        let plan = resolved_decode_plan(None, true).unwrap();
        assert_eq!(plan.source, DecodeTilingSource::EnvOverride);
        assert_eq!((plan.edge, plan.overlap), (512, 48));
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "128");
        assert_eq!(
            resolved_decode_plan(None, true).unwrap().source,
            DecodeTilingSource::SequentialDefault
        );
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
        assert!(resolved_decode_plan(None, true).is_none());
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
    }

    #[test]
    fn noise_shape_is_batch1_32ch() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 32, 32, 32]);
        let n = create_noise(0, 512, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 32, 32, 16]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let a = create_noise(7, 256, 256).unwrap();
        let b = create_noise(7, 256, 256).unwrap();
        let c = create_noise(8, 256, 256).unwrap();
        eval([&a, &b, &c]).unwrap();
        assert_eq!(
            a.as_slice::<f32>(),
            b.as_slice::<f32>(),
            "same seed reproduces"
        );
        assert_ne!(
            a.as_slice::<f32>(),
            c.as_slice::<f32>(),
            "diff seed differs"
        );
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SANA-1.6B: flow-match Euler over flow_shift=3.0, no dynamic shifting (our deliberate
        // divergence from the repo's DPM-Solver default; see module doc).
        let s = FlowMatchEuler::for_static_shift(4, SCHEDULE_SHIFT);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }
}
