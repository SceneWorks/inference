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

use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::image::decoded_to_image;
use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::{
    run_flow_sampler, CancelFlag, Error, FlowMatchEuler, Image, Progress, Result,
    TimestepConvention,
};
use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::{random, Array};

use crate::config::DcAeConfig;
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::scm::ScmScheduler;
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::StagedHeavy;

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
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        // The unified flow sampler hands `timestep = σ`; the SANA trunk embeds `σ·1000`.
        let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
        let pred_cond = transformer.forward_with_guidance(x, cond, &t, None, cond_mask)?;
        match uncond {
            Some(uc) if guidance_scale > 1.0 => {
                let pred_uncond =
                    transformer.forward_with_guidance(x, uc, &t, None, uncond_mask)?;
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
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        &scheduler.sigmas[start_step.min(scheduler.sigmas.len().saturating_sub(1))..],
        latents,
        seed,
        cancel,
        on_progress,
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
    // `Some` decodes in overlapping spatial tiles (see `decode_tiled`); `None` is the whole-image
    // path, byte-identical to before tiling existed.
    tiling: Option<&TilingConfig>,
) -> Result<Image> {
    let scale = Array::from_slice(&[cfg.scaling_factor], &[1]);
    let unscaled = divide(latents, &scale)?; // diffusers: latents / scaling_factor
    let decoded_nhwc = match tiling {
        Some(tiling) => decode_tiled(decoder, &unscaled, tiling, cancel)?,
        None => decoder.decode(&unscaled, cancel)?, // [1, H, W, 3] NHWC, f32
    };
    let decoded_nchw = decoded_nhwc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for decoded_to_image
    decoded_to_image(&decoded_nchw)
}

/// Default tile edge in **output** pixels, and its overlap. Measured: at 1024² this holds the
/// sequential peak to 3294 MiB on the host and 2733 MiB on device, against 9177 MiB untiled.
/// A quarter-tile overlap is enough for the trapezoidal blend without paying for large redundant
/// borders.
///
/// # Why 192, and not the 128 this shipped with
///
/// **192 is the largest edge that still reaches the minimum possible request peak.** Swept on real
/// weights at 1024², `Sequential`:
///
/// | edge | 512 | 384 | 256 | **192** | 128 | 96 | 64 |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | host peak (MiB) | 5146 | 4496 | 3465 | **3294** | 3294 | 3294 | 3294 |
/// | mean \|Δ\| vs whole-image | 2.454 | 3.036 | 3.558 | **5.158** | 6.192 | 9.212 | 11.140 |
///
/// The peak **floors at 3294 MiB from 192 downward** — below it the denoise phase binds and a
/// smaller tile buys nothing — while fidelity keeps degrading. So every edge under 192 pays image
/// quality for no admission win, which is a strictly bad trade. 128 was exactly that trade, unknowingly.
///
/// The device agrees and then improves on the argument: 192 is not merely equal-cost, it is cheaper
/// and faster, because a coarser grid is fewer per-tile dispatches.
///
/// | iPhone 17 Pro Max, 1024² | time | MLX peak | footprint | min headroom |
/// |---|---:|---:|---:|---:|
/// | **edge 192** | **29.2 s** | **2733 MiB** | **2860 MiB** | **3218 MiB** |
/// | edge 128 | 34.9 s | 2839 MiB | 2904 MiB | 3147 MiB |
///
/// Going the other way costs real memory: 256 adds 171 MiB of peak and 347 MiB of footprint for a
/// better image (3.558). That is a legitimate trade a caller may now make through
/// `GenerationMemory::decode_tile_edge` — but not the default, because 4096 MiB is the budget an
/// 8 GB device is assumed to have and edge 256 lands its footprint exactly on that line.
pub const DECODE_TILE_EDGE: i32 = 192;
/// Overlap for [`DECODE_TILE_EDGE`], in output pixels.
pub const DECODE_OVERLAP: i32 = DECODE_TILE_EDGE / 4;

/// The decode tiling for this render, or `None` to decode whole-image.
///
/// # Why `Sequential` is the trigger
///
/// `Sequential` is the memory-constrained signal — it is the policy a phone loads under, and the
/// one `runtime-ios` uses. Under it, **tiling is not optional**: an untiled DC-AE decode was
/// measured at 9177 MiB on the host and **killed the app on device**, while the tiled path
/// completed at 2751 MiB (`docs/ios-epics.md`, E5).
///
/// This default is the fix for a real shipping defect, not a preference. Tiling used to be
/// reachable *only* through `MLX_GEN_SANA_DECODE_TILE`, which nothing in `runtime-ios` or
/// `mlx-gen-ios-catalog` sets — so a product building the iOS bundle and calling SANA got exactly
/// the configuration that dies. The proven-good path must be the default one.
///
/// `Resident` keeps the whole-image decode. That is deliberate and not merely conservative: tiling
/// changes output pixels by construction (DC-AE's attention normalizer is global — see
/// [`decode_tiled`]), so a Mac with memory to spare should pay nothing for a bound it does not need.
///
/// The env var survives as a **measurement override**, which is what it was built for
/// (`mlx-gen-ios-catalog`'s `image_budget` and `tiling_fidelity` sweep it). `0` forces whole-image
/// so the untiled path stays reachable for A/Bs.
///
/// SC-15449: this is rung 2 (`BoundedDecode`). The **request-scoped** half of the contract is
/// honoured here; the calibrated-ladder half (a published edge domain with minted evidence, tiling
/// only when the predicted peak exceeds the budget) is still the adoption story — see
/// [`DecodeTilingSource`].
///
/// # Precedence, stated because it has bitten
///
/// `env override > request > load-time default`. Three sources can ask for a decode geometry and
/// they must not be able to disagree silently:
///
/// 1. **`MLX_GEN_SANA_DECODE_TILE`** wins outright, including `0` for whole-image. It exists to be a
///    *measurement* override — `image_budget` and `tiling_fidelity` sweep it, and
///    `runtime-macos`'s `sana_canonical` example refuses to run at all when it is set, which is only
///    coherent if the env var beats everything. An A/B knob that a request could quietly override
///    would make every sweep it appears in untrustworthy.
/// 2. **[`GenerationMemory::tile_vae_decode`]** — the contract's rung-2 signal. Honoured **even
///    under `Resident`**: a caller asking for a bound gets one, and accepts that DC-AE tiling is not
///    output-preserving (see [`decode_tiled`]). This is the half that was missing, and its absence
///    was a defect rather than an omission — the field existed, a caller could set it, and the
///    decode ran whole-image at 9177 MiB with no error and no diagnostic.
/// 3. **`Sequential`** — the load-time default. Untiled under this policy was measured at 9177 MiB
///    and killed the app on device while the tiled path completed at 2751 MiB, so the proven-good
///    path is the default one.
///
/// `Resident` with nothing requested stays whole-image, deliberately: tiling changes output pixels,
/// so a Mac with memory to spare should pay nothing for a bound it does not need. That asymmetry
/// with z-image (whose `Resident` case tiles) is real and intentional — DC-AE's attention normalizer
/// is global, z-image's GroupNorm VAE is not.
pub(crate) fn decode_tiling(
    memory: Option<GenerationMemory>,
    is_sequential: bool,
) -> Option<TilingConfig> {
    resolved_decode_plan(memory, is_sequential)
        .map(|plan| TilingConfig::spatial_only(plan.edge, plan.overlap))
}

/// Which of the three inputs decided a decode geometry.
///
/// Reported rather than inferred. Three separate times this session a knob that read correctly in
/// source was not the knob the run used — an env var that stopped being forwarded, one that changed
/// meaning when a default moved, and a summary line that asserted a configuration instead of reading
/// it back. Naming the winning source is what turns "the code says it should tile" into a fact the
/// harness can print.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeTilingSource {
    /// `MLX_GEN_SANA_DECODE_TILE`.
    EnvOverride,
    /// [`GenerationMemory::tile_vae_decode`] on the request.
    Request,
    /// The `OffloadPolicy::Sequential` load-time default.
    SequentialDefault,
}

/// A resolved decode geometry and the source that chose it. `edge`/`overlap` are output pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeTilingPlan {
    pub edge: i32,
    pub overlap: i32,
    pub source: DecodeTilingSource,
}

/// What the decode will **actually** do: `Some(plan)` when it tiles, `None` for whole-image.
///
/// Pure, and consulted by [`decode_tiling`] itself rather than duplicating its rules, so it cannot
/// drift from the decision it reports on. Public because the difference between a tiled and an
/// untiled DC-AE decode at 1024² is 2751 MiB against 9177 MiB — the difference between a render and
/// a jetsam kill — and reading the *request* back does not establish which way it went.
pub fn resolved_decode_plan(
    memory: Option<GenerationMemory>,
    is_sequential: bool,
) -> Option<DecodeTilingPlan> {
    if let Some(px) = std::env::var("MLX_GEN_SANA_DECODE_TILE")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        // `0` is the whole-image A/B control, and must stay expressible.
        return (px > 0).then_some(DecodeTilingPlan {
            edge: px,
            overlap: px / 4,
            source: DecodeTilingSource::EnvOverride,
        });
    }
    if memory.is_some_and(|m| m.tile_vae_decode) {
        let edge = memory
            .and_then(|m| m.decode_tile_edge)
            .map_or(DECODE_TILE_EDGE, |e| e as i32);
        return Some(DecodeTilingPlan {
            edge,
            overlap: memory
                .and_then(|m| m.decode_overlap)
                .map_or(edge / 4, |o| o as i32),
            source: DecodeTilingSource::Request,
        });
    }
    is_sequential.then_some(DecodeTilingPlan {
        edge: DECODE_TILE_EDGE,
        overlap: DECODE_OVERLAP,
        source: DecodeTilingSource::SequentialDefault,
    })
}

/// DC-AE tiling parameters: the ×32 spatial compression, and the single stage that runs at full
/// output resolution.
///
/// `full_res_channels: 3` — DC-AE's last stage emits `[1, H, W, 128]` at H/1, i.e. the widest
/// full-resolution write is the 3-channel RGB output; the 128-channel stage is what tiling bounds.
const DC_AE_TILING: VaeTiling = VaeTiling {
    spatial_scale: 32,
    temporal_scale: 1,
    causal_temporal: false,
    full_res_channels: 3,
};

/// Decode `latent` (`[1, 32, h, w]` NCHW, already unscaled) in overlapping spatial tiles.
///
/// # Why
///
/// The DC-AE decode's memory is dominated by *transients inside its late stages*, not by weights
/// or by the tensors it hands between stages. Measured on SANA at 512px: the last two stages add
/// ~3.5 GB while their outputs are 64 MiB and 128 MiB — roughly 18× the tensor produced. Nothing
/// outside the decoder can release that, which is why weight reduction moved the peak 0 MiB and
/// stage-wise `eval` moved it 2 MiB (`docs/ios-epics.md`, E5).
///
/// Tiling reaches inside: each stage's internals are then sized by the *tile*, not the image. The
/// shared [`mlx_gen::vae_tiling::tiled_decode`] does the slicing, trapezoidal seam blending, and
/// per-tile `eval` (the eval is essential — without it the whole tiled graph materializes at once
/// and tiling achieves nothing).
///
/// # Layout
///
/// [`mlx_gen::vae_tiling::tiled_decode`] is **5-D** (its blend masks and pad specs are `[_; 5]`) and
/// slices `denorm` and shapes the decoded tile through the *same* `[t, h, w]` axis indices — so the
/// latent and the decoder's output must agree on where h and w live. SANA's do not: the latent is
/// NCHW `[1, 32, h, w]` and [`DcAeDecoder::decode`] emits NHWC `[1, H, W, 3]`.
///
/// The reconciling layout is channels-last **NTHWC** with axes `[1, 2, 3]`: the latent is transposed
/// to `[1, 1, h, w, 32]` and each decoded tile is lifted to `[1, 1, TH, TW, 3]`, which puts h and w at
/// axes 2 and 3 on both sides. `T` is a 1-length dummy — a still image has no temporal extent, and
/// [`DC_AE_TILING`]'s `temporal_scale: 1` plus a `spatial_only` config means the plan emits exactly
/// one temporal tile that is never split.
fn decode_tiled(
    decoder: &DcAeDecoder,
    latent: &Array,
    tiling: &TilingConfig,
    cancel: &CancelFlag,
) -> Result<Array> {
    let sh = latent.shape();
    let (h, w) = (sh[2], sh[3]);
    let plan = tiling.plan(DC_AE_TILING, 1, h, w);

    // NCHW [1, 32, h, w] → NTHWC [1, 1, h, w, 32].
    let denorm = latent
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[1, 1, h, w, LATENT_CHANNELS])?;

    let out = mlx_gen::vae_tiling::tiled_decode(&denorm, &plan, [1, 2, 3], Some(cancel), |tile| {
        let ts = tile.shape();
        let (th, tw) = (ts[2], ts[3]);
        // NTHWC tile → NCHW for the decoder, then its NHWC output back up to NTHWC.
        let tile_nchw = tile
            .reshape(&[1, th, tw, LATENT_CHANNELS])?
            .transpose_axes(&[0, 3, 1, 2])?;
        let dec = decoder.decode(&tile_nchw, cancel)?; // [1, TH, TW, 3] NHWC
        let ds = dec.shape();
        Ok(dec.reshape(&[1, 1, ds[1], ds[2], ds[3]])?)
    })?;

    // NTHWC [1, 1, H, W, 3] → NHWC [1, H, W, 3], the layout `decode_to_image` transposes from.
    let os = out.shape();
    Ok(out.reshape(&[1, os[2], os[3], os[4]])?)
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
    // txt2img: seed the SCM prior (`latents * sigma_data`) and run the whole angle schedule (start 0).
    let latents = multiply(&latents, arr1(scheduler.sigma_data))?;
    denoise_sprint_from(
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
    use mlx_rs::transforms::eval;

    let sd = scheduler.sigma_data;
    let mut latents = latents;

    // The embedded guidance scalar (CFG-free): guidance_scale * guidance_embeds_scale, a [1] tensor
    // fed to the trunk's guidance embedder. Constant across steps.
    let guidance = arr1(guidance_scale * guidance_embeds_scale);

    let n = scheduler.num_steps();
    let start = start_step.min(n);
    let total = (n - start).max(1) as u32;
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
        let t_next = scheduler.timesteps[i + 1];
        let scm_t = scheduler.scm_timestep(i);
        let in_scale = scheduler.input_scale(i);

        // model input = (latents / sigma_data) * sqrt(scm_t² + (1-scm_t)²).
        let lat_in = multiply(&divide(&latents, arr1(sd))?, arr1(in_scale))?;
        let scm_t_arr = arr1(scm_t);
        let raw = transformer.forward_with_guidance(
            &lat_in,
            cond,
            &scm_t_arr,
            Some(&guidance),
            cond_mask,
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
    ///
    /// `None` for a **text-to-image-only** bundle. Text-to-image never encodes a reference image,
    /// so eagerly building this held ~0.61 GB of weights through the render phase — the phase that
    /// sets peak memory — to serve a path the request had already declined. That is affordable on a
    /// Mac and decisive on a phone, where SANA is otherwise over the per-app cap
    /// (`docs/ios-epics.md`, E5). See [`new_text_to_image`](Self::new_text_to_image).
    encoder: Option<DcAeEncoder>,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
    sprint: bool,
    guidance_embeds_scale: f32,
}

/// Materialize `latents` and release everything the denoise phase was still holding, before the
/// DC-AE decode allocates.
///
/// MLX is lazy: until an array is evaluated it holds a reference to the graph that produced it, so
/// at the decode boundary the *entire denoise history* — every step's intermediates, and through
/// them the trunk's weights — is still live. Measured on SANA at 512px, decode therefore began
/// with **~3.5 GB already resident**, against a first-stage output of 1 MiB
/// (`docs/ios-epics.md`, E5).
///
/// Evaluating collapses the latents to a concrete buffer and drops that graph; `clear_cache` then
/// returns the freed blocks rather than letting MLX hold them for reuse. Decode starts near the
/// latents' own size instead of near the denoise peak.
///
/// This is the same load→use→drop discipline the residency seam already applies to the text
/// encoder, applied at the phase boundary *inside* the render bundle.
///
/// # Why this is still called under `Resident`, where nothing is shed
///
/// [`Residency::run_staged`] runs its `materialize_mid` hook on the `Sequential` arm only — correctly,
/// since its job there is to make [`StagedHeavy::shed_dit`] actually free the trunk, and `Resident`
/// sheds nothing. But the eval buys something *independent* of shedding: it drops the denoise
/// **activation** graph, which is worth ~1.4 GB on SANA at 512² whichever policy is in force. So the
/// staged call site invokes this at the tail of its denoise phase (both arms) *and* passes it as
/// `materialize_mid` (the seam's contract hook, so the shed's correctness does not depend on the
/// denoise closure's internals). The second call finds evaluated arrays and is a no-op.
pub(crate) fn release_denoise_graph(latents: &[Array]) -> Result<()> {
    mlx_rs::transforms::eval(latents.iter())?;
    mlx_rs::memory::clear_cache();
    Ok(())
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

    /// Compose a **text-to-image-only** base bundle: identical to [`new`](Self::new) but without the
    /// DC-AE encoder, which that path never uses.
    ///
    /// A `generate` carrying an `init_image` against this bundle is a caller error and returns
    /// [`Error::Msg`] rather than silently ignoring the reference image.
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

    /// The [`new_text_to_image`](Self::new_text_to_image) counterpart for SANA-Sprint.
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
        let latents = self.denoise_one(cond, req, guidance, cancel, on_progress)?;
        on_progress(Progress::Decoding);
        release_denoise_graph(std::slice::from_ref(&latents))?;
        // Resident semantics: this standalone entrypoint holds every component itself and has no
        // memory pressure to bound, so it does not tile by DEFAULT. It still honours an explicit
        // request (`tile_vae_decode`) and the env override — `false` here is the residency, not a
        // refusal. The `Sequential` default lives at the residency-driven call site in `model.rs`.
        let tiling = resolve_decode_tiling(None, false);
        self.decode_view()
            .decode_one(&latents, cancel, tiling.as_ref())
    }

    /// **Denoise** one image from pre-encoded conditioning, stopping at the latents — the phase-B half
    /// of [`render_one`](Self::render_one), split out so [`Residency::run_staged`] can shed the DiT
    /// between denoise and decode (see the [`StagedHeavy`] impl below).
    ///
    /// The caller owns the `eval` of the returned latents: under the staged seam that is
    /// `materialize_mid`, which must run while the DiT is still alive.
    pub fn denoise_one(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        if self.sprint {
            self.denoise_one_sprint(cond, req, guidance, cancel, on_progress)
        } else {
            self.denoise_one_cfg(cond, req, guidance, cancel, on_progress)
        }
    }

    /// The base SANA-1.6B true-CFG flow-match denoise for one image (the pre-seam `generate_with` tail,
    /// less the decode).
    fn denoise_one_cfg(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
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
                Error::Msg(
                    "SANA: this bundle is text-to-image-only (no DC-AE encoder), so an init_image \
                     cannot be encoded -- build it with `SanaHeavy::new` rather than \
                     `new_text_to_image`"
                        .to_string(),
                )
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
        denoise_cfg(
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
        )
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
    fn denoise_one_sprint(
        &self,
        cond: &SanaConditioning,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
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
                Error::Msg(
                    "SANA: this bundle is text-to-image-only (no DC-AE encoder), so an init_image \
                     cannot be encoded -- build it with `SanaHeavy::new` rather than \
                     `new_text_to_image`"
                        .to_string(),
                )
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
        denoise_sprint_from(
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
        )
    }
}

/// The **light** (decode-only) SANA bundle that survives the DiT drop under `Sequential` staged
/// decode: the DC-AE decoder and its config. [`StagedHeavy::shed_dit`] drops the ~2.0 GB (Q4)
/// transformer — and the DC-AE *encoder*, when an img2img load carried one — so the decode-phase peak
/// excludes both.
///
/// This is the lever `release_denoise_graph` alone could not reach. Releasing the denoise graph frees
/// the *activations* the DiT produced; only shedding frees the DiT's **weights**, and on iOS those
/// weights are the difference between fitting the per-app cap and not (`docs/ios-epics.md`, E5).
pub struct SanaLight {
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
}

/// A borrowed decode view — from the owned [`SanaLight`] under `Sequential` (post-shed) or from the
/// still-warm [`SanaHeavy`] under `Resident` — so the decode body is written once for both policies.
pub struct SanaDecodeView<'a> {
    decoder: &'a DcAeDecoder,
    dc_ae_cfg: &'a DcAeConfig,
}

impl SanaDecodeView<'_> {
    /// DC-AE-decode one already-denoised, already-evaluated latent to an [`Image`].
    ///
    /// `tiling` is the caller's decision, from [`resolve_decode_tiling`] — the same shape as
    /// `mlx-gen-z-image`'s `decode_batch`, and passed rather than derived because only the caller
    /// knows the load's residency.
    pub fn decode_one(
        &self,
        latents: &Array,
        cancel: &CancelFlag,
        tiling: Option<&TilingConfig>,
    ) -> Result<Image> {
        decode_to_image(self.decoder, self.dc_ae_cfg, latents, cancel, tiling)
    }
}

/// The decode tiling for a request under a load with the given residency. See [`decode_tiling`] for
/// the policy and its precedence; this is the public entry point the model layer calls.
///
/// Takes the request's `memory` block, not just the residency: `tile_vae_decode` is a
/// request-scoped signal, so a per-load answer cannot honour it. `None` is "no request-scoped
/// signal" — the standalone [`SanaPipeline`] entrypoint, which has no contract request.
pub fn resolve_decode_tiling(
    memory: Option<GenerationMemory>,
    is_sequential: bool,
) -> Option<TilingConfig> {
    decode_tiling(memory, is_sequential)
}

impl StagedHeavy for SanaHeavy {
    type Light = SanaLight;
    type DecodeView<'a> = SanaDecodeView<'a>;

    fn shed_dit(self) -> SanaLight {
        // `self.transformer` (and `self.encoder`, when present) drop here; only the DC-AE decoder and
        // its config move into the light bundle.
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

    /// The default under `Sequential` must TILE, with no environment help.
    ///
    /// This is a regression gate on a real shipping defect, not a style check. Tiling used to be
    /// reachable only through `MLX_GEN_SANA_DECODE_TILE`, which nothing in `runtime-ios` or
    /// `mlx-gen-ios-catalog` sets — so a product building the iOS bundle got the whole-image decode,
    /// which was measured at 9177 MiB and **killed the app on device**. Everything the iOS work
    /// proved was reachable only through a knob the product does not turn.
    ///
    /// Deliberately asserted against the env var being ABSENT, because its presence is exactly what
    /// masked the bug.
    #[test]
    fn sequential_tiles_by_default_without_any_env_var() {
        // The suite runs single-threaded (`.cargo/config.toml` forces RUST_TEST_THREADS=1), so
        // mutating process env here cannot race a sibling test.
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");

        let sequential = decode_tiling(None, true).expect(
            "Sequential must tile by default: untiled DC-AE decode is the configuration that was \
             jetsam-killed on device",
        );
        let spatial = sequential.spatial.expect("spatial tiling");
        assert_eq!(spatial.tile_px, DECODE_TILE_EDGE);
        assert_eq!(spatial.overlap_px, DECODE_OVERLAP);

        // Resident keeps the exact whole-image decode: tiling changes pixels by construction
        // (DC-AE's attention normalizer is global), so a host with memory to spare pays nothing.
        assert!(
            decode_tiling(None, false).is_none(),
            "Resident must keep the exact untiled decode"
        );
    }

    /// The env var stays an override for measurement, `0` meaning whole-image.
    #[test]
    fn env_override_wins_over_the_residency_default() {
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "256");
        let forced =
            decode_tiling(None, false).expect("an explicit edge tiles even under Resident");
        assert_eq!(forced.spatial.unwrap().tile_px, 256);

        // `0` is the A/B control: it must force whole-image even under Sequential, or the untiled
        // path becomes unreachable for the comparison that established it is fatal.
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
        assert!(
            decode_tiling(None, true).is_none(),
            "0 must force whole-image, so the untiled control stays measurable"
        );

        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
    }

    /// A request asking for rung 2 must GET rung 2 — including under `Resident`.
    ///
    /// This is the defect the adoption closes, and it was silent: `GenerationMemory::tile_vae_decode`
    /// is a documented contract field, a caller could set it, and SANA ignored it entirely. No error,
    /// no diagnostic — the decode simply ran whole-image at a peak measured at 9177 MiB, the exact
    /// configuration that was jetsam-killed on device.
    ///
    /// `Resident` is the interesting half. SANA does not tile there by DEFAULT (tiling changes output
    /// pixels, so a host with memory to spare should pay nothing), but a caller that explicitly asks
    /// for a bound has accepted that trade and must receive it.
    #[test]
    fn an_explicit_request_tiles_under_both_residencies() {
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        let asked = Some(GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        });

        for is_sequential in [false, true] {
            let plan = resolved_decode_plan(asked, is_sequential).unwrap_or_else(|| {
                panic!("tile_vae_decode must be honoured (is_sequential={is_sequential})")
            });
            assert_eq!(plan.source, DecodeTilingSource::Request);
            assert_eq!(
                plan.edge, DECODE_TILE_EDGE,
                "unspecified edge falls back to the default"
            );
            assert_eq!(plan.overlap, DECODE_TILE_EDGE / 4);
        }
    }

    /// The request's geometry is used when it names one, and `decode_overlap` is independent of edge.
    #[test]
    fn the_request_geometry_is_honoured_field_by_field() {
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        let plan = resolved_decode_plan(
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(32),
                ..Default::default()
            }),
            false,
        )
        .expect("an explicit geometry tiles");
        assert_eq!((plan.edge, plan.overlap), (256, 32));

        // An edge with no overlap derives edge/4 rather than inheriting the DEFAULT overlap — a
        // 128-derived overlap on a 256 tile would silently change the blend ratio.
        let derived = resolved_decode_plan(
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                ..Default::default()
            }),
            false,
        )
        .expect("an explicit edge tiles");
        assert_eq!(derived.overlap, 64);
    }

    /// Precedence is total and ordered: env beats request beats residency.
    ///
    /// Asserted rather than documented because three separate times this session a knob that read
    /// correctly in source was not the knob the run used. The env var must win even against an
    /// explicit request, or every A/B sweep that sets it (`image_budget`, `tiling_fidelity`) becomes
    /// untrustworthy the moment a caller also asks for tiling.
    #[test]
    fn precedence_is_env_then_request_then_residency() {
        let asked = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(256),
            ..Default::default()
        });

        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "512");
        let plan = resolved_decode_plan(asked, true).expect("env tiles");
        assert_eq!(plan.source, DecodeTilingSource::EnvOverride);
        assert_eq!(plan.edge, 512, "env must beat an explicit request geometry");

        // And `0` must beat a request too, or the untiled control cannot be measured against a
        // caller that asks for tiling.
        std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
        assert!(
            resolved_decode_plan(asked, true).is_none(),
            "env 0 must force whole-image even when the request asks for a bound"
        );

        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        let plan = resolved_decode_plan(asked, true).expect("request tiles");
        assert_eq!(plan.source, DecodeTilingSource::Request);
        assert_eq!(plan.edge, 256, "request must beat the residency default");

        let plan = resolved_decode_plan(None, true).expect("sequential tiles");
        assert_eq!(plan.source, DecodeTilingSource::SequentialDefault);
    }

    /// The default edge is a measured choice, not a round number — pin it.
    ///
    /// 192 is the **largest** edge that still reaches SANA's minimum request peak. The peak floors
    /// at 3294 MiB from 192 downward (below it the denoise phase binds and a smaller tile buys
    /// nothing) while fidelity keeps degrading, so every smaller edge pays image quality for no
    /// admission win. On device 192 is also 16% faster and 106 MiB cheaper than the 128 this
    /// shipped with, because a coarser grid is fewer per-tile dispatches.
    ///
    /// Moving it DOWN re-introduces that bad trade. Moving it UP costs real memory — 256 adds
    /// 347 MiB of footprint, which lands exactly on the 4096 MiB budget an 8 GB device is assumed to
    /// have. Either direction needs the sweep re-run, not a judgement call, which is what this test
    /// is here to force.
    #[test]
    fn the_default_edge_is_the_largest_one_that_reaches_the_memory_floor() {
        assert_eq!(
            DECODE_TILE_EDGE, 192,
            "changing the default decode edge requires re-running the memory and fidelity sweeps \
             (mlx-gen-ios-catalog's image_budget and tiling_fidelity) — see the constant's docs"
        );
        assert_eq!(DECODE_OVERLAP, 48, "a quarter-tile overlap");
    }

    /// `tile_vae_decode: false` is not a request to tile — a `memory` block that says nothing about
    /// the decode must leave the residency default alone in both directions.
    #[test]
    fn a_memory_block_that_does_not_ask_changes_nothing() {
        std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        // Set a rung that is NOT rung 2, to prove the check reads the right field.
        let other = Some(GenerationMemory {
            chunk_attention: true,
            ..Default::default()
        });
        assert!(
            resolved_decode_plan(other, false).is_none(),
            "Resident stays whole-image"
        );
        assert_eq!(
            resolved_decode_plan(other, true).map(|p| p.source),
            Some(DecodeTilingSource::SequentialDefault),
            "Sequential keeps its default rather than being upgraded to a Request source"
        );
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
