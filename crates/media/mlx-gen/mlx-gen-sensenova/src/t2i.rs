//! Text-to-image generation (sc-3188) — the `t2i_generate` spine end to end.
//!
//! Ports `modeling_neo_chat.py::t2i_generate` for the dense 8B-MoT checkpoint. The flow:
//!
//! 1. Build the `neo1_0` query ([`build_neo1_query`] + [`SYSTEM_MESSAGE_FOR_GEN`] + the think
//!    sentinel), tokenize, and **prefill** it into a KV cache on the understanding path
//!    ([`Qwen3Backbone::forward_cached`] append). With CFG (`cfg_scale > 1`) a second, *uncondition*
//!    prefix (`<img>` after an empty prompt) is prefilled into its own cache.
//! 2. (think-mode) run the [`Qwen3Backbone::generate_think`] rollout, extending the cache and
//!    placing the image block after the appended `\n\n<img>`.
//! 3. **Denoise** for `num_steps` over the standard flow-matching schedule
//!    ([`apply_time_schedule`]): each step embeds the current noisy image through the gen-path
//!    [`NeoVisionEmbedder`] (channel-first patches) + the timestep (and noise-scale) embedding, runs
//!    the **generation** path over `[cached prefix ++ image block]` via `forward_cached`
//!    **use-only** (`update_cache=False`), maps the image hidden states through the [`FmHead`] to a
//!    patch latent `x_pred`, forms the [`velocity`], and takes an [`euler_step`]. CFG blends the
//!    condition/uncondition velocities ([`CfgNorm`] variants).
//! 4. [`unpatchify`] the final latent → RGB `[1, 3, H, W]`.
//!
//! The pixel path is the `fm_head` → unpatchify (`use_pixel_head = false`); the conv decoders and the
//! dynamic-μ schedule are dead code for this checkpoint.

use std::cell::RefCell;

use mlx_llm::core_llm::token_execution as execution;
use mlx_rs::ops::{add, divide, matmul, minimum, multiply, subtract, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec,
    SliceTransition, StepHooks,
};
use mlx_gen::gen_core::render::conditional::{
    run_conditional, ConditionalCeilings, ConditionalExecutor, ConditionalInput, ConditionalPlan,
    ConditionalReceipt, PrefillModelShape, RolloutStop, SourceImageGeometry, TextRollout,
};
use mlx_gen::gen_core::{BoundaryState, GenerationOutput, LoadPhase};
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Progress, Quant, Result};

use crate::config::NeoChatConfig;
use crate::fm::{
    apply_time_schedule, euler_step, patchify, patchify_channel_first, unpatchify, velocity,
    FmHead, TimestepEmbedder,
};
use crate::qwen3::{KvCache, Path, Qwen3Backbone, RopeMask};
use crate::runtime::Sampler;
use crate::text::{build_neo1_query, image_indexes, text_indexes, tokens, SYSTEM_MESSAGE_FOR_GEN};
use crate::vision::NeoVisionEmbedder;
// The shared deterministic PRNG backs the Box–Muller noise sampler; the `KvCache` trait is brought
// in anonymously so the cached-length reads (`cache.offset()`) resolve on the shared cache (sc-7159).
use mlx_llm::primitives::{KvCache as _, SplitMix64};

/// Per-step cancellation + progress for the denoise loops, matching the SDXL family's `&CancelFlag` +
/// `on_progress` threading. `None` ⇒ uncancellable with no progress (diagnostic/parity callers); the
/// production [`SenseNova`](crate::SenseNova) path passes the request's flag and the worker's progress
/// callback so a multi-minute 8B run is cancellable and reports **denoise steps**, not the image index
/// (F-128).
///
/// ## Lifecycle boundaries (sc-22738)
///
/// The reporter also owns the two **phase boundaries** a consumer segments a run by — the
/// SceneWorks memory-measurement adapter cuts its conditioning/denoise/decode windows on exactly
/// these, the same events `mlx-gen-bernini` and `mlx-gen-mage` emit:
///
/// * [`Progress::Loading`]`(`[`LoadPhase::Renderer`]`)` opens the **denoise** phase. It is emitted
///   by the denoise loops ([`T2iModel::denoise`], [`T2iModel::it2i_denoise`]) after every prefill —
///   the conditioning work — and before the first step. On the deferred load shape this is
///   literally a load: the generation half of every dual-path block is materialized in bounded
///   windows only from here on. On the resident shape it is the phase marker.
/// * [`Progress::Decoding`] opens the **decode** phase: emitted after the last step, before the
///   final RGB patches are converted to an image (`unpatchify` + the host copy).
///
/// Both are emitted **once per request**, never once per image: a multi-image request (`count >
/// 1`, or the model-driven interleave loop) restarts neither the bar nor the phases. That is what
/// the `opens_denoise` / `opens_decode` flags carry — the route that owns the request decides
/// which of its per-image reporters opens which phase ([`Self::with_phase_bounds`]).
pub struct StepReporter<'a> {
    cancel: &'a CancelFlag,
    on_progress: &'a mut dyn FnMut(Progress),
    /// Steps completed by prior denoises to fold onto this one's `current` (F-136). `0` for a
    /// single-image render; `images_so_far × num_steps` for the interleave path.
    offset: usize,
    /// When set, overrides the per-denoise `total` with the aggregate grand total so the interleave
    /// bar is one monotone `1..=(max_images × num_steps)` sweep instead of restarting per image.
    total_override: Option<usize>,
    /// Whether this reporter's denoise loop opens the request's denoise phase
    /// (`Loading(Renderer)` before its first step). See the type docs.
    opens_denoise: bool,
    /// Whether this reporter's denoise loop opens the request's decode phase (`Decoding` after its
    /// last step). See the type docs.
    opens_decode: bool,
    prompt_encoding: bool,
}

impl<'a> StepReporter<'a> {
    /// A reporter for a single-image request: opens both phases around its one denoise loop.
    pub fn new(cancel: &'a CancelFlag, on_progress: &'a mut dyn FnMut(Progress)) -> Self {
        Self {
            cancel,
            on_progress,
            offset: 0,
            total_override: None,
            opens_denoise: true,
            opens_decode: true,
            prompt_encoding: false,
        }
    }

    /// Bracket the Generator's prompt prefills; cached denoise-only callers leave this unarmed.
    pub(crate) fn with_prompt_encoding(mut self) -> Self {
        (self.on_progress)(Progress::Loading(LoadPhase::EncodingPrompt));
        self.prompt_encoding = true;
        self
    }

    fn finish_prompt_encoding(&mut self) {
        if self.prompt_encoding {
            self.prompt_encoding = false;
            (self.on_progress)(Progress::PromptEncodingComplete);
        }
    }

    /// Choose which of the request's phase boundaries this reporter's denoise loop opens
    /// (sc-22738). A route that runs several denoise loops per request hands the first loop
    /// `opens_denoise` and the last loop `opens_decode`, so the request reports each boundary
    /// exactly once — see [`request_phase_bounds`] and the interleave loop.
    pub fn with_phase_bounds(mut self, opens_denoise: bool, opens_decode: bool) -> Self {
        self.opens_denoise = opens_denoise;
        self.opens_decode = opens_decode;
        self
    }

    /// A reporter for one image of a multi-image aggregate (F-136, sc-11133): its `current` is
    /// offset by the steps already completed and its `total` is pinned to the grand total, so a
    /// sequence of these across the interleave loop yields one monotone, non-restarting bar.
    pub fn new_folded(
        cancel: &'a CancelFlag,
        on_progress: &'a mut dyn FnMut(Progress),
        offset: usize,
        total: usize,
    ) -> Self {
        Self {
            cancel,
            on_progress,
            offset,
            total_override: Some(total),
            opens_denoise: true,
            opens_decode: true,
            prompt_encoding: false,
        }
    }

    /// Abort with a typed error if the request was cancelled (checked before each denoise step).
    #[cfg(test)]
    fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        Ok(())
    }

    /// The bound cancellation flag, so AR text loops that run alongside the denoise (F-037) can share
    /// the same request cancel without a separate parameter.
    pub fn cancel_flag(&self) -> &'a CancelFlag {
        self.cancel
    }

    /// Report one completed denoise step (`current` is 1-based). Folds in the aggregate `offset` /
    /// `total_override` (F-136) so a multi-image run reports one monotone bar rather than restarting.
    fn step(&mut self, current: usize, total: usize) {
        let total = self.total_override.unwrap_or(total);
        (self.on_progress)(Progress::Step {
            current: (self.offset + current).min(total) as u32,
            total: total as u32,
        });
    }

    /// Open the denoise phase — `Progress::Loading(LoadPhase::Renderer)` — if this reporter owns
    /// that boundary (sc-22738). Called by the denoise loops after every prefill and before their
    /// first step.
    fn open_denoise_phase(&mut self) {
        self.finish_prompt_encoding();
        if self.opens_denoise {
            (self.on_progress)(Progress::Loading(LoadPhase::Renderer));
        }
    }

    /// Open the decode phase — `Progress::Decoding` — if this reporter owns that boundary
    /// (sc-22738). Called by the denoise loops after their last step.
    fn open_decode_phase(&mut self) {
        if self.opens_decode {
            open_decode_phase(self.on_progress);
        }
    }
}

impl Drop for StepReporter<'_> {
    fn drop(&mut self) {
        self.finish_prompt_encoding();
    }
}

/// The one place the decode boundary is spelled: [`Progress::Decoding`], emitted after the last
/// denoise step of a request and before its RGB patches become an image (sc-22738).
fn open_decode_phase(on_progress: &mut dyn FnMut(Progress)) {
    on_progress(Progress::Decoding);
}

/// Which of a request's phase boundaries the denoise loop for image `index` of `count` opens
/// (sc-22738): the first loop opens the denoise phase, the last opens the decode phase, and a
/// single-image request opens both. Each boundary is reported exactly once per request whatever the
/// count, so a multi-image request does not restart the phases the way it would if every image
/// opened its own.
pub fn request_phase_bounds(index: u32, count: u32) -> (bool, bool) {
    (index == 0, index + 1 >= count)
}

/// F-136 residual (sc-11133): `interleave_gen`'s loop is model-driven and can stop before
/// `max_images` (an `IM_END` / `hit_max` / non-`img_start` next token ends it early). The per-image
/// folded [`StepReporter`]s pin `total = max_images × num_steps`, so a short run leaves `current`
/// frozen at `realized_images × num_steps < total` — a bar that never fills. When the realized image
/// count lands strictly between 1 and `max_images`, emit one terminal `Step { current: total, total
/// }` so the aggregate bar reaches its (unchanged) total. This only ever raises `current`, keeping
/// the stream monotone. A 0-image run drew no bar and a full `max_images` run already reached
/// `total`, so neither emits (a duplicate terminal would violate the strict-increase contract).
#[cfg(test)]
fn fill_interleave_bar(
    on_progress: &mut dyn FnMut(Progress),
    realized_images: usize,
    max_images: usize,
    num_steps: usize,
) {
    if realized_images == 0 || realized_images >= max_images {
        return;
    }
    let grand_total = (max_images * num_steps) as u32;
    on_progress(Progress::Step {
        current: grand_total,
        total: grand_total,
    });
}

#[cfg(test)]
fn finish_interleave_progress(
    on_progress: &mut dyn FnMut(Progress),
    realized_images: usize,
    max_images: usize,
    num_steps: usize,
) {
    fill_interleave_bar(on_progress, realized_images, max_images, num_steps);
    if realized_images > 0 {
        open_decode_phase(on_progress);
    }
}

/// Classifier-free-guidance velocity-blend normalisation (`t2i_generate`'s `cfg_norm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CfgNorm {
    /// Plain blend `v_uncond + cfg·(v_cond − v_uncond)`.
    #[default]
    None,
    /// Rescale the blended velocity to the condition velocity's global norm.
    Global,
    /// Per-token rescale to the condition velocity's per-token norm.
    Channel,
    /// CFG-Zero* (optimised-scale uncondition + step-0 zeroing).
    CfgZeroStar,
}

/// Knobs for [`T2iModel::generate`] (the `t2i_generate` arguments).
#[derive(Clone, Copy, Debug)]
pub struct T2iOptions {
    pub cfg_scale: f32,
    /// Image-guidance scale for it2i (`img_cfg_scale`): edit ≈ 1.0, character ≈ 1.5. Unused by T2I.
    pub img_cfg_scale: f32,
    pub cfg_norm: CfgNorm,
    pub cfg_interval: (f32, f32),
    pub num_steps: usize,
    pub timestep_shift: f32,
    pub enable_timestep_shift: bool,
    pub t_eps: f32,
    pub seed: u64,
    pub think_mode: bool,
    pub max_think_tokens: usize,
    /// Maximum score-domain elements per shared bounded-attention call. `None` preserves the
    /// historical single-call path. Production request paths consume it for both understanding and
    /// generation forwards, including VQA, interleave, and think-token attention.
    pub attention_score_budget: Option<u32>,
    /// Generation-path Qwen block window selected by rung 4. `None` is the resident/default
    /// execution shape (or one all-covering window on a deferred load).
    pub transformer_window_size: Option<u32>,
    /// Calibration-only deterministic failure after the first streamed Gen block materializes.
    #[doc(hidden)]
    pub calibration_stream_fault: bool,
}

impl Default for T2iOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            img_cfg_scale: 1.0,
            cfg_norm: CfgNorm::None,
            cfg_interval: (0.0, 1.0),
            num_steps: 30,
            timestep_shift: 1.0,
            enable_timestep_shift: true,
            t_eps: 0.02,
            seed: 0,
            think_mode: false,
            max_think_tokens: 1024,
            attention_score_budget: None,
            transformer_window_size: None,
            calibration_stream_fault: false,
        }
    }
}

fn attention_plan<'a>(opts: &T2iOptions, cancel: Option<&'a CancelFlag>) -> AttentionPlan<'a> {
    let Some(score_elements) = opts.attention_score_budget else {
        return AttentionPlan::UNBOUNDED;
    };
    let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
        u64::from(score_elements),
        true,
    ));
    match cancel {
        Some(cancel) => plan.with_cancel(cancel),
        None => plan,
    }
}

/// The result of a [`T2iModel::interleave_gen`] run: the composed text (with `<image>`
/// placeholders where images were generated) and the generated images in order.
pub struct InterleaveOutput {
    pub text: String,
    pub images: Vec<Array>,
}

impl InterleaveOutput {
    /// Convert composed text and model-space `[1, 3, H, W]` images into typed
    /// [`GenerationOutput::Interleaved`].
    pub fn into_generation_output(self) -> Result<GenerationOutput> {
        let mut decoded = Vec::with_capacity(self.images.len());
        for image in &self.images {
            decoded.push(decoded_to_image(image)?);
        }
        Ok(GenerationOutput::Interleaved {
            text: self.text,
            images: decoded,
        })
    }
}

impl TryFrom<InterleaveOutput> for GenerationOutput {
    type Error = Error;

    fn try_from(output: InterleaveOutput) -> Result<Self> {
        output.into_generation_output()
    }
}

/// Guidance regime for [`SenseNovaFmDriver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SenseNovaCfgMode {
    /// Text-to-image: 2-cache CFG (condition / uncondition) with inclusive `[i0, i1]` gate
    /// and optional `CfgZeroStar`.
    T2i,
    /// Image-conditioned / interleave: 3-cache dual CFG (condition / text-uncond / image-uncond)
    /// with exclusive `(i0, i1) || i0 == 0` gate and post-blend `Global`/`Channel` norm.
    It2i,
}

/// Weights-free velocity forward seam for [`SenseNovaFmDriver`] (`PIPE-7`, `SW-6974`).
///
/// Generic over step preparation, single/dual-CFG velocity evaluation, and unpatchification so
/// Tier 0 [`conformance::verify_driver_conformance`] tests verify every [`DriverSupport`] claim
/// without model weights or real forwards.
pub trait SenseNovaVelocityForward {
    /// Prepare step latent `z` and conditioned token embedding from model-space `image` at `t`.
    fn prepare_step(&self, image: &Array, t: f32) -> Result<(Array, Array)>;
    /// Conditional velocity prediction.
    fn predict_cond(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array>;
    /// Image-conditioned / text-unconditioned velocity prediction (`cache_img` / `cache_tu`).
    fn predict_img(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array>;
    /// Unconditional velocity prediction (`cache_uncond` / `cache_iu`).
    fn predict_uncond(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array>;
    /// Unpatchify stepped latent `next_z` back to `[1, 3, H, W]` model space.
    fn unpatchify_step(&self, next_z: &Array) -> Result<Array>;
}

/// Typed [`FamilyDriver`] for MLX SenseNova-U1 flow-matching Euler denoising across both T2I and
/// IT2I / interleave passes (`PIPE-7`, `PIPE-14`, `B3`, `SW-6974`).
pub struct SenseNovaFmDriver<F> {
    timesteps: Vec<f32>,
    mode: SenseNovaCfgMode,
    cfg_scale: f32,
    img_cfg_scale: f32,
    has_uncond: bool,
    cfg_norm: CfgNorm,
    cfg_interval: (f32, f32),
    t_eps: f32,
    forward: F,
    support: DriverSupport,
}

impl<F: SenseNovaVelocityForward> SenseNovaFmDriver<F> {
    fn build_support(driver_id: &'static str) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            // MLX SenseNova seeds explicit SplitMix64 streams rather than global `mlx_rs::random`.
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native MLX SenseNova FM driver does not re-noise from clean boundary estimates",
                geometry_change:
                    "native MLX SenseNova FM driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native MLX SenseNova FM driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        timesteps: Vec<f32>,
        mode: SenseNovaCfgMode,
        opts: &T2iOptions,
        has_img: bool,
        has_uncond: bool,
        forward: F,
    ) -> Result<Self> {
        if opts.num_steps == 0 || timesteps.len() != opts.num_steps + 1 {
            return Err(Error::Msg(format!(
                "{driver_id}: invalid SenseNova schedule len {} for num_steps={}",
                timesteps.len(),
                opts.num_steps
            )));
        }
        if mode == SenseNovaCfgMode::It2i {
            let (needs_img, needs_uncond) =
                it2i_cache_requirements(opts.cfg_scale, opts.img_cfg_scale);
            if needs_img && !has_img {
                return Err(img_cache_err());
            }
            if needs_uncond && !has_uncond {
                return Err(uncond_cache_err());
            }
            if opts.cfg_norm == CfgNorm::CfgZeroStar {
                return Err(cfg_zero_star_it2i_err());
            }
        }
        Ok(Self {
            timesteps,
            mode,
            cfg_scale: opts.cfg_scale,
            img_cfg_scale: opts.img_cfg_scale,
            has_uncond,
            cfg_norm: opts.cfg_norm,
            cfg_interval: opts.cfg_interval,
            t_eps: opts.t_eps,
            forward,
            support: Self::build_support(driver_id),
        })
    }

    pub fn total_steps(&self) -> usize {
        self.timesteps.len().saturating_sub(1)
    }

    fn guided_velocity(
        &self,
        z: &Array,
        cond_emb: &Array,
        t: f32,
        outer_step: usize,
    ) -> Result<Array> {
        let out_cond = self.forward.predict_cond(z, cond_emb, t)?;
        match self.mode {
            SenseNovaCfgMode::T2i => {
                let needs_cfg = self.cfg_scale > 1.0 && self.has_uncond;
                if needs_cfg && t >= self.cfg_interval.0 && t <= self.cfg_interval.1 {
                    let v_uncond = self.forward.predict_uncond(z, cond_emb, t)?;
                    cfg_blend(
                        &out_cond,
                        &v_uncond,
                        self.cfg_scale,
                        self.cfg_norm,
                        outer_step,
                    )
                } else {
                    Ok(out_cond)
                }
            }
            SenseNovaCfgMode::It2i => {
                let (cfg, img_cfg) = (self.cfg_scale, self.img_cfg_scale);
                let (i0, i1) = self.cfg_interval;
                let use_cfg = (t > i0 && t < i1) || i0 == 0.0;
                let mut v_pred = if !use_cfg || (cfg == 1.0 && img_cfg == 1.0) {
                    out_cond.clone()
                } else if img_cfg == 1.0 {
                    let oi = self.forward.predict_img(z, cond_emb, t)?;
                    add(
                        &oi,
                        &multiply(&subtract(&out_cond, &oi)?, Array::from_f32(cfg))?,
                    )?
                } else if cfg == img_cfg {
                    let ou = self.forward.predict_uncond(z, cond_emb, t)?;
                    add(
                        &ou,
                        &multiply(&subtract(&out_cond, &ou)?, Array::from_f32(cfg))?,
                    )?
                } else {
                    let oi = self.forward.predict_img(z, cond_emb, t)?;
                    let ou = self.forward.predict_uncond(z, cond_emb, t)?;
                    let a = multiply(&subtract(&out_cond, &oi)?, Array::from_f32(cfg))?;
                    let b = multiply(&subtract(&oi, &ou)?, Array::from_f32(img_cfg))?;
                    add(&add(&ou, &a)?, &b)?
                };
                if (cfg > 1.0 || img_cfg > 1.0) && use_cfg {
                    v_pred = apply_cfg_norm(v_pred, &out_cond, self.cfg_norm)?;
                }
                Ok(v_pred)
            }
        }
    }
}

impl<F: SenseNovaVelocityForward> FamilyDriver<Array> for SenseNovaFmDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        let _ = self.t_eps;
        self.support.validate_slice(&input, state)?;
        state.enter_slice(&input)?;
        let total = self.total_steps();
        if input.span.end > total {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice span {:?} exceeds {total} steps",
                self.support.driver_id, input.span
            )));
        }

        let mut image = match input.state {
            BoundaryState::XSigma(img)
            | BoundaryState::CleanEstimate { latent: img, .. }
            | BoundaryState::RouteSpecific { state: img, .. } => img,
        };

        for outer_step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let t = self.timesteps[outer_step];
            let t_next = self.timesteps[outer_step + 1];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: t_next,
            };
            hooks.observe(&info, &image)?;

            let (z, cond_emb) = self
                .forward
                .prepare_step(&image, t)
                .map_err(mlx_gen::gen_core::Error::from)?;
            let v_pred = self
                .guided_velocity(&z, &cond_emb, t, outer_step)
                .map_err(mlx_gen::gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let next_z =
                euler_step(&v_pred, &z, t, t_next).map_err(mlx_gen::gen_core::Error::from)?;
            image = self
                .forward
                .unpatchify_step(&next_z)
                .map_err(mlx_gen::gen_core::Error::from)?;
            image = hooks.post_step(&info, image)?;
            hooks.materialize(&info, &image)?;
            state.record_outer_step(outer_step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(image))
    }
}

struct ProductionSenseNovaForward<'a> {
    model: &'a T2iModel,
    grid_h: i32,
    grid_w: i32,
    l: i32,
    cell: i32,
    token_h: i32,
    token_w: i32,
    noise_embed: &'a Option<Array>,
    t_eps: f32,
    attention: AttentionPlan<'a>,
    transformer_window: Option<usize>,
    calibration_stream_fault: bool,
    cache_cond: RefCell<&'a mut KvCache>,
    rm_cond: &'a RopeMask,
    cache_img: Option<(RefCell<&'a mut KvCache>, &'a RopeMask)>,
    cache_uncond: Option<(RefCell<&'a mut KvCache>, &'a RopeMask)>,
}

impl SenseNovaVelocityForward for ProductionSenseNovaForward<'_> {
    fn prepare_step(&self, image: &Array, t: f32) -> Result<(Array, Array)> {
        self.model
            .step_cond_embeds(image, self.grid_h, self.grid_w, self.l, t, self.noise_embed)
    }

    fn predict_cond(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
        let mut cache = self.cache_cond.borrow_mut();
        self.model.predict_v(
            cond_emb,
            self.rm_cond,
            *cache,
            z,
            t,
            self.t_eps,
            self.attention,
            self.transformer_window,
            self.calibration_stream_fault,
        )
    }

    fn predict_img(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
        let (cache_cell, rm) = self.cache_img.as_ref().ok_or_else(img_cache_err)?;
        let mut cache = cache_cell.borrow_mut();
        self.model.predict_v(
            cond_emb,
            rm,
            *cache,
            z,
            t,
            self.t_eps,
            self.attention,
            self.transformer_window,
            self.calibration_stream_fault,
        )
    }

    fn predict_uncond(&self, z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
        let (cache_cell, rm) = self.cache_uncond.as_ref().ok_or_else(|| {
            Error::Msg("sensenova: guidance needs an uncond cache, but none was supplied".into())
        })?;
        let mut cache = cache_cell.borrow_mut();
        self.model.predict_v(
            cond_emb,
            rm,
            *cache,
            z,
            t,
            self.t_eps,
            self.attention,
            self.transformer_window,
            self.calibration_stream_fault,
        )
    }

    fn unpatchify_step(&self, next_z: &Array) -> Result<Array> {
        unpatchify(next_z, self.cell, Some(self.token_h), Some(self.token_w))
    }
}

/// The interleave resolution buckets (`examples/interleave/inference.py::SUPPORTED_RESOLUTIONS`) —
/// `(width, height)` per aspect ratio. Document Studio picks one of these. Default `"16:9"`.
pub const INTERLEAVE_RESOLUTIONS: &[(&str, (i32, i32))] = &[
    ("1:1", (1536, 1536)),
    ("16:9", (2048, 1152)),
    ("9:16", (1152, 2048)),
    ("3:2", (1888, 1248)),
    ("2:3", (1248, 1888)),
    ("4:3", (1760, 1312)),
    ("3:4", (1312, 1760)),
    ("1:2", (1088, 2144)),
    ("2:1", (2144, 1088)),
    ("1:3", (864, 2592)),
    ("3:1", (2592, 864)),
];

/// Look up an interleave resolution bucket by aspect-ratio key (e.g. `"16:9"`).
pub fn interleave_resolution_for(ratio: &str) -> Option<(i32, i32)> {
    INTERLEAVE_RESOLUTIONS
        .iter()
        .find(|(r, _)| *r == ratio)
        .map(|(_, wh)| *wh)
}

/// The result of a [`T2iModel::generate`] run.
pub struct T2iOutput {
    /// The generated image `[1, 3, H, W]` (model space, roughly `[-1, 1]`).
    pub image: Array,
    /// The decoded think-block text, when `think_mode` was set.
    pub think_text: Option<String>,
}

/// The T2I model: the backbone plus the flow-matching generation modules.
pub struct T2iModel {
    backbone: Qwen3Backbone,
    gen_vision: NeoVisionEmbedder,
    /// The **understanding**-path vision embedder (`vision_model.embeddings`) used to embed source /
    /// reference images for it2i (sc-3189). `None` for fixtures that omit `vision_model.*`.
    und_vision: Option<NeoVisionEmbedder>,
    fm_head: FmHead,
    timestep_embedder: TimestepEmbedder,
    noise_scale_embedder: Option<TimestepEmbedder>,
    patch_size: i32,
    merge_size: i32,
    noise_scale: f32,
    noise_scale_mode: String,
    noise_scale_base_image_seq_len: f32,
    noise_scale_max_value: f32,
    /// `<IMG_CONTEXT>` / `<img>` / `</img>` ids (the checkpoint constants; overridable for tiny test
    /// fixtures whose vocab can't hold the real ids).
    img_context_id: i32,
    img_start_id: i32,
    img_end_id: i32,
    prefill_shape: PrefillModelShape,
    cache_token_bytes: u64,
    resident_bytes_estimate: u64,
}

impl T2iModel {
    /// Build from a loaded checkpoint (`language_model.*` + `fm_modules.*`).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig) -> Result<Self> {
        Self::validate_config(cfg)?;
        let backbone = Qwen3Backbone::from_weights(w, cfg, "language_model")?;
        Self::from_weights_with_backbone(w, cfg, backbone)
    }

    pub fn from_weights_deferred(
        w: &Weights,
        cfg: &NeoChatConfig,
        artifact: crate::memory_strategy::PinnedArtifact,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Self::validate_config(cfg)?;
        let backbone =
            Qwen3Backbone::from_weights_deferred(w, cfg, "language_model", artifact, quant)?;
        Self::from_weights_with_backbone(w, cfg, backbone)
    }

    fn validate_config(cfg: &NeoChatConfig) -> Result<()> {
        // `noise_scale_embed` divides each denoise step's conditioning by `noise_scale_max_value`
        // (`scale.min(max)` then `/max`); a zero/negative value from a misconfigured config.json
        // would inject NaN/Inf conditioning silently. Reject it at load (F-012).
        if cfg.noise_scale_max_value <= 0.0 || cfg.noise_scale_max_value.is_nan() {
            return Err(Error::Msg(format!(
                "sensenova: noise_scale_max_value must be > 0 (got {})",
                cfg.noise_scale_max_value
            )));
        }
        Ok(())
    }

    fn from_weights_with_backbone(
        w: &Weights,
        cfg: &NeoChatConfig,
        backbone: Qwen3Backbone,
    ) -> Result<Self> {
        let noise_scale_embedder = if cfg.add_noise_scale_embedding {
            Some(TimestepEmbedder::from_weights(
                w,
                "fm_modules.noise_scale_embedder",
            )?)
        } else {
            None
        };
        // The understanding-path vision embedder is only needed for it2i; gate on its presence so
        // T2I-only fixtures (no `vision_model.*`) still load.
        let und_vision = if w
            .get("vision_model.embeddings.patch_embedding.weight")
            .is_some()
        {
            Some(NeoVisionEmbedder::from_weights(
                w,
                cfg,
                "vision_model.embeddings",
            )?)
        } else {
            None
        };
        Ok(Self {
            backbone,
            gen_vision: NeoVisionEmbedder::from_weights(
                w,
                cfg,
                "fm_modules.vision_model_mot_gen.embeddings",
            )?,
            und_vision,
            fm_head: FmHead::from_weights(w, "fm_modules.fm_head")?,
            timestep_embedder: TimestepEmbedder::from_weights(w, "fm_modules.timestep_embedder")?,
            noise_scale_embedder,
            patch_size: cfg.patch_size as i32,
            merge_size: (1.0 / cfg.downsample_ratio).round() as i32,
            noise_scale: cfg.noise_scale,
            noise_scale_mode: cfg.noise_scale_mode.clone(),
            noise_scale_base_image_seq_len: cfg.noise_scale_base_image_seq_len as f32,
            noise_scale_max_value: cfg.noise_scale_max_value,
            img_context_id: tokens::IMG_CONTEXT,
            img_start_id: tokens::IMG_START,
            img_end_id: tokens::IMG_END,
            prefill_shape: cfg.prefill_model_shape(),
            cache_token_bytes: cfg.cache_token_bytes_bound(),
            resident_bytes_estimate: cfg.estimated_resident_bytes(),
        })
    }

    /// Spatial cell stride (`patch_size * merge_size`).
    pub fn cell(&self) -> i32 {
        self.patch_size * self.merge_size
    }

    /// Pre-load / loaded architectural shape used by [`ConditionalPlan`].
    pub fn prefill_model_shape(&self) -> PrefillModelShape {
        self.prefill_shape
    }

    /// Quantize the backbone decoder stack to Q4/Q8 (sc-3193) — the bulk of the 8B params
    /// (attention projections + SwiGLU on both paths). The vision embedders, FM head, and
    /// timestep/noise embedders stay dense (small; precision-sensitive flow-matching head).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.backbone.quantize(bits)
    }

    /// Merge the 8-step distill LoRA (sc-3192) for the `sensenova_u1_8b_fast` variant: the backbone
    /// generation-path projections (`7 · layers`) + the two FM-head Linears. The understanding path,
    /// vision embedders, and timestep/noise embedders are untouched (the LoRA carries no targets for
    /// them). Must run on the dense model **before** [`T2iModel::quantize`] — the merge seam errors
    /// on a quantized base, matching the reference which merges into the dense weight pre-quant.
    /// Returns the total number of Linears merged (so the loader can assert full coverage).
    pub fn merge_distill_lora(&mut self, lora: &Weights) -> Result<usize> {
        let n = self.backbone.merge_distill_lora(lora, "language_model")?;
        Ok(n + self
            .fm_head
            .merge_distill_lora(lora, "fm_modules.fm_head")?)
    }

    /// Override the `<IMG_CONTEXT>` / `<img>` / `</img>` token ids (for tiny test fixtures whose
    /// vocab cannot hold the real checkpoint ids). Production callers never need this.
    pub fn with_image_token_ids(
        mut self,
        img_context_id: i32,
        img_start_id: i32,
        img_end_id: i32,
    ) -> Self {
        self.img_context_id = img_context_id;
        self.img_start_id = img_start_id;
        self.img_end_id = img_end_id;
        self
    }

    /// The resolution-mode noise scale for a `grid_h × grid_w` patch grid (the `t2i_generate`
    /// formula). For non-resolution modes the bare `noise_scale` is used; both are clamped to
    /// `noise_scale_max_value`.
    fn noise_scale_for(&self, grid_h: i32, grid_w: i32) -> f32 {
        let mut scale = self.noise_scale;
        if matches!(
            self.noise_scale_mode.as_str(),
            "resolution" | "dynamic" | "dynamic_sqrt"
        ) {
            let seq = (grid_h * grid_w) as f32 / (self.merge_size * self.merge_size) as f32;
            scale = (seq / self.noise_scale_base_image_seq_len).sqrt() * self.noise_scale;
            if self.noise_scale_mode == "dynamic_sqrt" {
                scale = scale.sqrt();
            }
        }
        scale.min(self.noise_scale_max_value)
    }

    /// Build the gen-path [`RopeMask`] for the image block: the `image_indexes` positions for a
    /// `token_h × token_w` grid sitting after `text_len` prefix tokens, with the block mask at cache
    /// prefix length `past`. Invariant across denoise steps, so the loops build it once per cache and
    /// reuse it for every step / guidance branch (F-139).
    fn prepare_gen(
        &self,
        token_h: i32,
        token_w: i32,
        text_len: usize,
        past: i32,
    ) -> Result<RopeMask> {
        let (it, ih, iw) = image_indexes(token_h as usize, token_w as usize, text_len);
        self.backbone.prepare_rope_mask(&it, &ih, &iw, past)
    }

    /// Run the gen-path velocity prediction for one diffusion step against a prefilled cache
    /// (`_t2i_predict_v`): `forward_prepared` (Gen path, use-only) over the image block with the
    /// per-cache [`RopeMask`] from [`Self::prepare_gen`], `fm_head` → `x_pred`, then the
    /// flow-matching velocity. `image_embeds` is the vision+timestep conditioned image block
    /// `[1, L, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn predict_v(
        &self,
        image_embeds: &Array,
        rm: &RopeMask,
        cache: &mut KvCache,
        z: &Array,
        t: f32,
        t_eps: f32,
        attention: AttentionPlan<'_>,
        transformer_window: Option<usize>,
        calibration_stream_fault: bool,
    ) -> Result<Array> {
        let hidden = self.backbone.forward_prepared_memory(
            image_embeds,
            rm,
            Path::Gen,
            cache,
            false,
            attention,
            transformer_window,
            calibration_stream_fault,
        )?;
        let x_pred = self.fm_head.forward(&hidden)?;
        velocity(&x_pred, z, t, t_eps)
    }

    /// Prefill a text query into a fresh cache on the understanding path. Returns the cache, the
    /// last-position logits (for think-mode), and the prefix token length.
    fn prefill(
        &self,
        ids: &[i32],
        opts: &T2iOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<(KvCache, Array, usize)> {
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.backbone.embed(&ids_arr)?;
        let (t, h, wid) = text_indexes(ids.len());
        let mut cache = self.backbone.new_cache();
        let hidden = self.backbone.forward_cached_budgeted(
            &embeds,
            &t,
            &h,
            &wid,
            Path::Und,
            &mut cache,
            true,
            attention_plan(opts, cancel),
        )?;
        // Only the last position's logits are kept, so slice the hidden state to `[1, 1, 4096]`
        // *before* `lm_head` — applying it over the whole `[1, S, 4096]` prefix would materialize an
        // `[1, S, vocab]` (~GB) tensor and an `S×4096×vocab` matmul just to drop all but one row (F-129).
        let last_hidden = hidden.take_axis(Array::from_slice(&[n - 1], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        // F-144: explicit f32 so the `as_slice::<f32>()` readback in the think-mode caller can't
        // mis-read a bf16/f16 lm_head buffer (safe-by-accident via f32 RoPE today). No-op when f32.
        let last = logits.reshape(&[vocab])?.as_dtype(Dtype::Float32)?;
        Ok((cache, last, ids.len()))
    }

    /// Generate an image for `prompt` at `width × height` (both multiples of `patch·merge = 32`).
    /// `init_noise`, when supplied, is a standard-normal tensor `[1, 3, H, W]` used in place of
    /// fresh sampling (for cross-build parity); it is scaled by the resolution-mode `noise_scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<T2iOutput> {
        let (traj, think_text) =
            self.t2i_run(tokenizer, prompt, width, height, opts, init_noise, reporter)?;
        let image = traj.into_iter().last().expect("at least one step");
        Ok(T2iOutput { image, think_text })
    }

    /// Diagnostic (sc-3192 parity): the full per-step denoise **trajectory** for a T2I run — every
    /// step's decoded image, not just the final. Identical setup to [`Self::generate`]; lets a test compare
    /// the port's trajectory to the reference step by step (e.g. to show that a distilled few-step
    /// run agrees early and diverges only on the big decisive final steps, i.e. compounding precision
    /// chaos rather than a per-step bug).
    pub fn t2i_trajectory(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
    ) -> Result<Vec<Array>> {
        Ok(self
            .t2i_run(tokenizer, prompt, width, height, opts, init_noise, None)?
            .0)
    }

    /// Shared T2I body: condition/uncondition prefixes (+ optional think rollout) and the denoise
    /// loop. Returns the per-step trajectory and any think text. [`generate`] keeps the last frame;
    /// [`t2i_trajectory`](Self::t2i_trajectory) returns them all.
    #[allow(clippy::too_many_arguments)]
    fn t2i_run(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<(Vec<Array>, Option<String>)> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova t2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }

        // ---- Condition prefix ----
        let think_sentinel = if opts.think_mode {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n<img>"
        };
        let query_cond = format!(
            "{}{}",
            build_neo1_query(prompt, SYSTEM_MESSAGE_FOR_GEN),
            think_sentinel
        );
        let ids_cond = tokenizer.encode_ids(&query_cond, true)?;
        let (mut cache_cond, last_logits, prefix_len) = self.prefill(
            &ids_cond,
            opts,
            reporter.as_ref().map(StepReporter::cancel_flag),
        )?;

        // think-mode: roll out the reasoning block, then append `\n\n<img>`.
        let mut think_text = None;
        let mut text_len = prefix_len;
        if opts.think_mode {
            let append_ids = tokenizer.encode_ids("\n\n<img>", false)?;
            let roll = self.backbone.generate_think_budgeted(
                last_logits.as_slice::<f32>(),
                &mut cache_cond,
                (prefix_len - 1) as i32,
                tokens::THINK_END,
                tokens::IM_END,
                &append_ids,
                opts.max_think_tokens,
                reporter.as_ref().map(StepReporter::cancel_flag),
                attention_plan(opts, reporter.as_ref().map(StepReporter::cancel_flag)),
            )?;
            let ids_u32: Vec<u32> = roll.think_token_ids.iter().map(|&i| i as u32).collect();
            think_text = Some(tokenizer.decode(&ids_u32, false)?);
            text_len = (roll.t_idx + 1) as usize;
        }

        // ---- Uncondition prefix (CFG) ----
        let needs_cfg = opts.cfg_scale > 1.0;
        let mut cache_uncond = None;
        if needs_cfg {
            let query_uncond = format!("{}<img>", build_neo1_query("", ""));
            let ids_uncond = tokenizer.encode_ids(&query_uncond, true)?;
            let (cache, _, plen) = self.prefill(
                &ids_uncond,
                opts,
                reporter.as_ref().map(StepReporter::cancel_flag),
            )?;
            cache_uncond = Some((cache, plen));
        }

        let base_noise = match init_noise {
            Some(n) => n.as_dtype(Dtype::Float32)?,
            None => gaussian(&[1, 3, height, width], opts.seed)?,
        };
        let cond_u = cache_uncond.as_mut().map(|(c, l)| (c, *l));
        let traj = self.denoise(
            &mut cache_cond,
            text_len,
            cond_u,
            width,
            height,
            &base_noise,
            opts,
            reporter,
        )?;
        Ok((traj, think_text))
    }

    /// Prefill `ids` into a fresh understanding-path cache; returns the cache and prefix length.
    /// Exposed for tests/callers that drive [`T2iModel::denoise`] with an explicit prefix.
    pub fn prefill_ids(&self, ids: &[i32]) -> Result<(KvCache, usize)> {
        let (cache, _, len) = self.prefill(ids, &T2iOptions::default(), None)?;
        Ok((cache, len))
    }

    /// The flow-matching denoise loop. `cache_cond` (and the optional `(cache_uncond, text_len)` for
    /// CFG) are prefilled understanding-path caches; `base_noise` is a standard-normal `[1,3,H,W]`
    /// tensor (scaled here by the resolution-mode noise scale). Returns the per-step image
    /// trajectory `[1,3,H,W]` (the last entry is the final image).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        cache_cond: &mut KvCache,
        text_len: usize,
        mut cache_uncond: Option<(&mut KvCache, usize)>,
        width: i32,
        height: i32,
        base_noise: &Array,
        opts: &T2iOptions,
        mut reporter: Option<StepReporter>,
    ) -> Result<Vec<Array>> {
        let cell = self.patch_size * self.merge_size;
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let image = multiply(
            &base_noise.as_dtype(Dtype::Float32)?,
            Array::from_f32(noise_scale),
        )?;

        let steps = opts.num_steps;
        let timesteps = step_schedule(opts)?;
        execution::observe_image_schedule(&timesteps);
        execution::observe_document_policy(
            serde_json::json!({"denoise_options":format!("{opts:?}"),"noise_scale_bits":noise_scale.to_bits(),"width":width,"height":height}),
        );
        // Constant noise-scale conditioning token (added to every step's timestep embedding).
        let noise_embed = self.noise_scale_embed(noise_scale, l)?;

        let needs_cfg = opts.cfg_scale > 1.0 && cache_uncond.is_some();
        // The RoPE tables + block mask are invariant across steps for a given cache, so build them
        // once per cache instead of inside every `predict_v` call (F-139).
        let rm_cond = self.prepare_gen(token_h, token_w, text_len, cache_cond.offset())?;
        let rm_uncond = match cache_uncond.as_ref() {
            Some((cu, tlu)) => Some(self.prepare_gen(token_h, token_w, *tlu, cu.offset())?),
            None => None,
        };
        let fallback_cancel = CancelFlag::default();
        let cancel: &CancelFlag = match reporter.as_ref() {
            Some(r) => r.cancel_flag(),
            None => &fallback_cancel,
        };
        let attention = attention_plan(opts, Some(cancel));
        let forward = ProductionSenseNovaForward {
            model: self,
            grid_h,
            grid_w,
            l,
            cell,
            token_h,
            token_w,
            noise_embed: &noise_embed,
            t_eps: opts.t_eps,
            attention,
            transformer_window: opts.transformer_window_size.map(|window| window as usize),
            calibration_stream_fault: opts.calibration_stream_fault,
            cache_cond: RefCell::new(cache_cond),
            rm_cond: &rm_cond,
            cache_img: None,
            cache_uncond: cache_uncond
                .as_mut()
                .zip(rm_uncond.as_ref())
                .map(|((cu, _), rm)| (RefCell::new(&mut **cu), rm)),
        };
        let driver = SenseNovaFmDriver::new(
            "sensenova_t2i_euler",
            timesteps.clone(),
            SenseNovaCfgMode::T2i,
            opts,
            false,
            needs_cfg,
            forward,
        )?;
        // Conditioning (the prefills) is complete: open the denoise phase before the first step
        // (sc-22738).
        if let Some(r) = reporter.as_mut() {
            r.open_denoise_phase();
        }
        let mut state = DriverRunState::new();
        let mut progress_cb = |event: Progress| {
            if let Progress::Step { current, total } = event {
                if let Some(r) = reporter.as_mut() {
                    r.step(current as usize, total as usize);
                }
            }
        };
        let mut hooks = StepHooks::new(&mut progress_cb);
        let mut on_materialize = |_info: &DriverStepInfo, img: &Array| {
            // Per-step eval so the cancel check interrupts mid-render (sc-5522 / sc-5399).
            img.eval()
                .map_err(|e| mlx_gen::gen_core::Error::from(Error::from(e)))
        };
        hooks.on_materialize = Some(&mut on_materialize);
        let out_state = run_mlx_driver_slices(
            &driver,
            [SliceSpec {
                geometry: Some((width as u32, height as u32)),
                ..SliceSpec::full(steps)
            }],
            BoundaryState::XSigma(image),
            steps,
            opts.seed,
            cancel,
            None,
            &mut state,
            &mut hooks,
            |_, out, _| Ok(out),
        )?;
        let image = match out_state {
            BoundaryState::XSigma(img)
            | BoundaryState::CleanEstimate { latent: img, .. }
            | BoundaryState::RouteSpecific { state: img, .. } => img,
        };
        // F-036: keep only the final evaluated frame (1-element Vec preserves the signature).
        let traj = vec![image.clone()];
        // The last step is done: open the decode phase before the RGB patches become an image
        // (sc-22738).
        if let Some(r) = reporter.as_mut() {
            r.open_decode_phase();
        }
        mlx_gen::image_execution::policy_debug(
            "sensenova.attention_score_budget",
            &opts.attention_score_budget,
        );
        mlx_gen::image_execution::policy_debug(
            "sensenova.transformer_window",
            &opts.transformer_window_size,
        );
        mlx_gen::image_execution::policy_debug("sensenova.cfg_norm", &opts.cfg_norm);
        mlx_gen::image_execution::policy_debug(
            "sensenova.cfg_interval_bits",
            &(opts.cfg_interval.0.to_bits(), opts.cfg_interval.1.to_bits()),
        );
        mlx_gen::image_execution::scalar("sensenova.cfg_scale", opts.cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.img_cfg_scale", opts.img_cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.noise_scale", noise_scale);
        mlx_gen::image_execution::scalar("sensenova.t_eps", opts.t_eps);
        mlx_gen::image_execution::sampled_native_from_primary_seed(
            "sensenova-denoise-euler-v1",
            "noise-fraction",
            image.dtype(),
            &timesteps,
            "splitmix64-offset-one-box-muller-f64-to-f32-v1;initial=output-seed;no-step-draws",
        );
        Ok(traj)
    }

    /// Build one denoise step's latent `z` (channel-last patchify at `cell`) and the conditioned
    /// image block `cond = gen_vision(image) + timestep_embed(t) [+ noise_scale_embed]` `[1, L,
    /// hidden]`. Shared by the T2I and it2i denoise loops.
    fn step_cond_embeds(
        &self,
        image: &Array,
        grid_h: i32,
        grid_w: i32,
        l: i32,
        t: f32,
        noise_embed: &Option<Array>,
    ) -> Result<(Array, Array)> {
        let cell = self.patch_size * self.merge_size;
        let z = patchify(image, cell)?;
        let image_input =
            patchify_channel_first(image, self.patch_size)?.reshape(&[grid_h * grid_w, -1])?;
        let vis = self
            .gen_vision
            .forward(&image_input, &[(grid_h as usize, grid_w as usize)])?
            .reshape(&[1, l, -1])?;
        let t_tok = self
            .timestep_embedder
            .forward(&Array::from_slice(&vec![t; l as usize], &[l]))?
            .reshape(&[1, l, -1])?;
        let mut cond = add(&vis, &t_tok)?;
        if let Some(ne) = noise_embed {
            cond = add(&cond, ne)?;
        }
        Ok((z, cond))
    }

    /// The constant noise-scale conditioning token `[1, L, hidden]` added to every step's timestep
    /// embedding — present only when the model has a `noise_scale_embedder`. Shared by both denoise
    /// loops (F-134).
    fn noise_scale_embed(&self, noise_scale: f32, l: i32) -> Result<Option<Array>> {
        let Some(emb) = &self.noise_scale_embedder else {
            return Ok(None);
        };
        let ns = vec![noise_scale / self.noise_scale_max_value; l as usize];
        Ok(Some(
            emb.forward(&Array::from_slice(&ns, &[l]))?
                .reshape(&[1, l, -1])?,
        ))
    }

    // ===================== it2i (instruction edit + Character Studio) =====================

    /// ImageNet-normalise a source RGB image `[3, H, W]` (f32 in `[0, 1]`) and channel-first
    /// patchify to `[grid_h·grid_w, 3·ps²]` for the understanding vision embedder. Returns the
    /// patches and the `(grid_h, grid_w)` patch grid. `H`/`W` must be multiples of `patch·merge`
    /// (use [`smart_resize`] upstream). Mirrors the reference `load_image_native` transform
    /// (ToTensor + Normalize) + `preprocess_pixel_values`.
    pub fn preprocess_image(&self, rgb: &Array) -> Result<(Array, (i32, i32))> {
        let sh = rgb.shape();
        if sh.len() != 3 || sh[0] != 3 {
            return Err(Error::Msg(format!(
                "expected source image [3,H,W], got {sh:?}"
            )));
        }
        let (h, w) = (sh[1], sh[2]);
        let cell = self.patch_size * self.merge_size;
        if h % cell != 0 || w % cell != 0 {
            return Err(Error::Msg(format!(
                "source image H/W must be multiples of {cell}, got {h}x{w}"
            )));
        }
        let mean = Array::from_slice(&[0.485f32, 0.456, 0.406], &[3, 1, 1]);
        let std = Array::from_slice(&[0.229f32, 0.224, 0.225], &[3, 1, 1]);
        let norm = divide(&subtract(&rgb.as_dtype(Dtype::Float32)?, &mean)?, &std)?;
        let (gh, gw) = (h / self.patch_size, w / self.patch_size);
        let patches = patchify_channel_first(&norm.reshape(&[1, 3, h, w])?, self.patch_size)?
            .reshape(&[gh * gw, -1])?;
        Ok((patches, (gh, gw)))
    }

    /// Understanding-path vision features for source-image patches (diagnostic / it2i internals).
    pub fn und_vision_features(&self, pixel_values: &Array, grids: &[(i32, i32)]) -> Result<Array> {
        let und = self
            .und_vision
            .as_ref()
            .ok_or_else(|| Error::Msg("vision_model not loaded".into()))?;
        let g: Vec<(usize, usize)> = grids
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect();
        und.forward(pixel_values, &g)
    }

    /// (t, h, w) position rows for a prefix containing source-image blocks (the reference
    /// `get_thw_indexes`): text tokens advance temporal by one; an image-context block shares one
    /// temporal index and carries its **merged-grid** `(row, col)` as `(h, w)`. `grids` are the full
    /// patch grids `(grid_h, grid_w)` per image, in order.
    fn get_thw_indexes(&self, ids: &[i32], grids: &[(i32, i32)]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let n = ids.len();
        let mut t = Vec::with_capacity(n);
        let mut acc = 0i32;
        for i in 0..n {
            let shift = i32::from(i > 0 && ids[i - 1] == self.img_start_id);
            let not_img = i32::from(ids[i] != self.img_context_id);
            acc += shift + not_img;
            t.push(acc - 1);
        }
        // Merged-grid (row=y, col=x) coordinates, concatenated across images in order.
        let merge = self.merge_size;
        let mut abs = Vec::new();
        for &(gh, gw) in grids {
            let (mh, mw) = (gh / merge, gw / merge);
            for idx in 0..(mh * mw) {
                abs.push((idx / mw, idx % mw));
            }
        }
        let mut h = vec![0i32; n];
        let mut w = vec![0i32; n];
        let mut k = 0usize;
        for i in 0..n {
            if ids[i] == self.img_context_id {
                let (y, x) = abs[k];
                h[i] = y;
                w[i] = x;
                k += 1;
            }
        }
        (t, h, w)
    }

    /// Embed `ids` and splice the understanding vision features into the `<IMG_CONTEXT>` positions
    /// (the reference `_build_it2i_inputs`). Returns the prefix embeds `[1, S, hidden]` and its
    /// `(t, h, w)` rows. `pixel_values` is the concatenated per-image patch list; `grids` the full
    /// patch grids. Scatter is done with a one-hot selection matmul (no in-place index assignment).
    #[allow(clippy::type_complexity)]
    fn build_it2i_prefix(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(Array, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let s = ids.len();
        let ids_arr = Array::from_slice(ids, &[1, s as i32]);
        let mut embeds = self.backbone.embed(&ids_arr)?; // [1, S, H]
        let (t, h, w) = self.get_thw_indexes(ids, grids);

        if let Some(pv) = pixel_values {
            let und = self.und_vision.as_ref().ok_or_else(|| {
                Error::Msg("it2i needs the understanding vision embedder (vision_model.*)".into())
            })?;
            let grids_us: Vec<(usize, usize)> = grids
                .iter()
                .map(|&(a, b)| (a as usize, b as usize))
                .collect();
            let vit = und.forward(pv, &grids_us)?; // [n_ctx, H]
            let hidden = embeds.shape()[2];
            let dt = embeds.dtype();
            let ctx: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|(_, &id)| id == self.img_context_id)
                .map(|(i, _)| i)
                .collect();
            let n_ctx = vit.shape()[0];
            if ctx.len() as i32 != n_ctx {
                return Err(Error::Msg(format!(
                    "it2i: {} <IMG_CONTEXT> tokens but {n_ctx} vision tokens",
                    ctx.len()
                )));
            }
            // P [S, n_ctx] one-hot: row = sequence position, col = vision-token index.
            let mut p = vec![0f32; s * n_ctx as usize];
            let mut mask = vec![0f32; s];
            for (k, &pos) in ctx.iter().enumerate() {
                p[pos * n_ctx as usize + k] = 1.0;
                mask[pos] = 1.0;
            }
            let p_arr = Array::from_slice(&p, &[s as i32, n_ctx]).as_dtype(dt)?;
            let vit_full = matmul(&p_arr, &vit.as_dtype(dt)?)?; // [S, H], 0 off-context
            let keep_mask = Array::from_slice(
                &mask.iter().map(|m| 1.0 - m).collect::<Vec<_>>(),
                &[s as i32, 1],
            )
            .as_dtype(dt)?;
            let e2d = embeds.reshape(&[s as i32, hidden])?;
            embeds =
                add(&multiply(&e2d, &keep_mask)?, &vit_full)?.reshape(&[1, s as i32, hidden])?;
        }
        Ok((embeds, t, h, w))
    }

    /// Prefill a prepared prefix (embeds + positions) on the understanding path. Returns the cache,
    /// the last-position logits (for think-mode), and the image-block temporal index (`max(t) + 1`).
    fn prefill_prefix(
        &self,
        embeds: &Array,
        t: &[i32],
        h: &[i32],
        w: &[i32],
    ) -> Result<(KvCache, Array, usize)> {
        self.prefill_prefix_budgeted(embeds, t, h, w, AttentionPlan::UNBOUNDED)
    }

    fn prefill_prefix_budgeted(
        &self,
        embeds: &Array,
        t: &[i32],
        h: &[i32],
        w: &[i32],
        attention: AttentionPlan<'_>,
    ) -> Result<(KvCache, Array, usize)> {
        let mut cache = self.backbone.new_cache();
        let hidden = self.backbone.forward_cached_budgeted(
            embeds,
            t,
            h,
            w,
            Path::Und,
            &mut cache,
            true,
            attention,
        )?;
        // Slice the last hidden row before `lm_head` — only its logits are used, and the prefix here
        // includes image-context blocks, so the full `[1, S, vocab]` projection is the worst case of
        // F-129 (an `S×4096×vocab` matmul + ~GB tensor) repeated per CFG cache.
        let s = t.len() as i32;
        let last_hidden = hidden.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?;
        let vocab = logits.shape()[2];
        let last = logits.reshape(&[vocab])?;
        let img_temporal = (*t.iter().max().unwrap_or(&0) + 1) as usize;
        Ok((cache, last, img_temporal))
    }

    /// The understanding-path prefill hidden states `[1, S, hidden]` for an it2i prefix (diagnostic).
    pub fn prefill_it2i_hidden(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<Array> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let mut cache = self.backbone.new_cache();
        self.backbone
            .forward_cached(&embeds, &t, &h, &w, Path::Und, &mut cache, true)
    }

    /// Build + prefill an it2i prefix from explicit token ids, source-image patches, and per-image
    /// patch grids (the tokenizer-free entry the parity test drives). Returns the cache and the
    /// image-block temporal index. `pixel_values` is the concatenated per-image patch list.
    pub fn prefill_it2i(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(KvCache, usize)> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let (cache, _, img_temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
        Ok((cache, img_temporal))
    }

    /// Build + prefill an it2i/VQA prefix (image-conditioned) and return the cache, the
    /// last-position logits, and the next-token temporal index (`max(t)` — decode starts at `+1`).
    /// The tokenizer-free entry the VQA path and its parity test share.
    pub fn prefill_it2i_logits(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(KvCache, Vec<f32>, usize)> {
        self.prefill_it2i_logits_budgeted(ids, pixel_values, grids, AttentionPlan::UNBOUNDED)
    }

    pub fn prefill_it2i_logits_budgeted(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
        attention: AttentionPlan<'_>,
    ) -> Result<(KvCache, Vec<f32>, usize)> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let (cache, last, img_temporal) =
            self.prefill_prefix_budgeted(&embeds, &t, &h, &w, attention)?;
        Ok((cache, last.as_slice::<f32>().to_vec(), img_temporal - 1))
    }

    /// Greedy/sampled understanding-path text decode from a prefilled cache (wraps
    /// [`Qwen3Backbone::generate`]). `first_logits` are the prefix's last-position logits; `t_idx`
    /// the prefix's max temporal index. Returns the generated token ids (stop ids excluded).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_text(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: usize,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        self.decode_text_budgeted(
            first_logits,
            cache,
            t_idx,
            eos,
            max_new_tokens,
            sampler,
            cancel,
            AttentionPlan::UNBOUNDED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_text_budgeted(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: usize,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<Vec<i32>> {
        self.backbone.generate_budgeted(
            first_logits,
            cache,
            t_idx as i32,
            eos,
            max_new_tokens,
            sampler,
            cancel,
            attention,
        )
    }

    /// Build the it2i condition prefix from a prompt + source images (replacing `<image>` markers
    /// with `<img><IMG_CONTEXT>×n</img>` per image, auto-prepending markers like `it2i_generate`).
    /// Returns the cache, last logits, image-block temporal index, and the per-image patch grids.
    fn build_it2i_query_ids(
        &self,
        tokenizer: &TextTokenizer,
        base_query: &str,
        grids: &[(i32, i32)],
    ) -> Result<Vec<i32>> {
        let mut query = base_query.to_string();
        for &(gh, gw) in grids {
            let n = (gh / self.merge_size) * (gw / self.merge_size);
            let block = format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n as usize));
            query = query.replacen("<image>", &block, 1);
        }
        Ok(tokenizer.encode_ids(&query, true)?)
    }

    /// Image-conditioned generation (`it2i_generate`): edit / Character-Studio reference. `images`
    /// are decoded source RGB tensors `[3, H, W]` in `[0, 1]` (smart-resized to multiples of
    /// `patch·merge`); `prompt` may contain `<image>` markers (auto-prepended otherwise). Dual
    /// guidance: `opts.cfg_scale` (text) + `opts.img_cfg_scale` (image; edit ≈ 1.0, character ≈
    /// 1.5). `init_noise` (optional) is a standard-normal `[1,3,H,W]` for cross-build parity.
    #[allow(clippy::too_many_arguments)]
    pub fn it2i_generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<T2iOutput> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova it2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }
        if images.is_empty() {
            return Err(Error::Msg("it2i requires at least one source image".into()));
        }

        // Preprocess source images → concatenated patches + per-image grids.
        let mut pv_parts = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for img in images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pv_refs: Vec<&Array> = pv_parts.iter().collect();
        let pixel_values = mlx_rs::ops::concatenate_axis(&pv_refs, 0)?;

        // Auto-prepend `<image>` markers when the prompt has fewer than the image count.
        let count = prompt.matches("<image>").count();
        let mut question = prompt.to_string();
        if images.len() > count {
            let extra = images.len() - count;
            let pre = if count == 0 && images.len() > 1 {
                (0..images.len())
                    .map(|i| format!("Image-{}:<image>\n", i + 1))
                    .collect::<String>()
            } else {
                "<image>\n".repeat(extra)
            };
            question = format!("{pre}{question}");
        }

        let think_sentinel = if opts.think_mode {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n<img>"
        };

        // Guidance plan (matches it2i_generate's needs_* flags).
        let needs_cfg = !(opts.cfg_scale == 1.0 && opts.img_cfg_scale == 1.0);
        let needs_img =
            needs_cfg && (opts.img_cfg_scale == 1.0 || opts.cfg_scale != opts.img_cfg_scale);
        let needs_uncond = needs_cfg && opts.img_cfg_scale != 1.0;

        // Condition: prompt + images.
        let cond_query = format!(
            "{}{}",
            build_neo1_query(&question, SYSTEM_MESSAGE_FOR_GEN),
            think_sentinel
        );
        let cond_ids = self.build_it2i_query_ids(tokenizer, &cond_query, &grids)?;
        let (cond_embeds, ct, ch, cw) =
            self.build_it2i_prefix(&cond_ids, Some(&pixel_values), &grids)?;
        let (mut cache_cond, last_logits, mut cond_temporal) = self.prefill_prefix_budgeted(
            &cond_embeds,
            &ct,
            &ch,
            &cw,
            attention_plan(opts, reporter.as_ref().map(StepReporter::cancel_flag)),
        )?;

        // think-mode rollout (extends cache + advances the image-block temporal index).
        let mut think_text = None;
        if opts.think_mode {
            let append_ids = tokenizer.encode_ids("\n\n<img>", false)?;
            let roll = self.backbone.generate_think_budgeted(
                last_logits.as_slice::<f32>(),
                &mut cache_cond,
                (cond_temporal - 1) as i32,
                tokens::THINK_END,
                tokens::IM_END,
                &append_ids,
                opts.max_think_tokens,
                reporter.as_ref().map(StepReporter::cancel_flag),
                attention_plan(opts, reporter.as_ref().map(StepReporter::cancel_flag)),
            )?;
            let u32s: Vec<u32> = roll.think_token_ids.iter().map(|&i| i as u32).collect();
            think_text = Some(tokenizer.decode(&u32s, false)?);
            cond_temporal = (roll.t_idx + 1) as usize;
        }

        // Image-condition: images only, empty prompt.
        let mut cache_img = None;
        if needs_img {
            let q = format!(
                "{}<img>",
                build_neo1_query(&"<image>".repeat(images.len()), "")
            );
            let ids = self.build_it2i_query_ids(tokenizer, &q, &grids)?;
            let (embeds, t, h, w) = self.build_it2i_prefix(&ids, Some(&pixel_values), &grids)?;
            let (cache, _, temporal) = self.prefill_prefix_budgeted(
                &embeds,
                &t,
                &h,
                &w,
                attention_plan(opts, reporter.as_ref().map(StepReporter::cancel_flag)),
            )?;
            cache_img = Some((cache, temporal));
        }

        // Uncondition: empty prompt, no images.
        let mut cache_uncond = None;
        if needs_uncond {
            let q = format!("{}<img>", build_neo1_query("", ""));
            let ids = tokenizer.encode_ids(&q, true)?;
            let (embeds, t, h, w) = self.build_it2i_prefix(&ids, None, &[])?;
            let (cache, _, temporal) = self.prefill_prefix_budgeted(
                &embeds,
                &t,
                &h,
                &w,
                attention_plan(opts, reporter.as_ref().map(StepReporter::cancel_flag)),
            )?;
            cache_uncond = Some((cache, temporal));
        }

        let base_noise = match init_noise {
            Some(n) => n.as_dtype(Dtype::Float32)?,
            None => gaussian(&[1, 3, height, width], opts.seed)?,
        };
        let img_ref = cache_img.as_mut().map(|(c, l)| (c, *l));
        let un_ref = cache_uncond.as_mut().map(|(c, l)| (c, *l));
        let traj = self.it2i_denoise(
            (&mut cache_cond, cond_temporal),
            img_ref,
            un_ref,
            width,
            height,
            &base_noise,
            opts,
            reporter,
        )?;
        let image = traj.into_iter().last().expect("at least one step");
        Ok(T2iOutput { image, think_text })
    }

    /// VQA / understanding (`chat` / `answer_question`): image(s) + question → understanding-path
    /// AR text generation → answer. The question prefix (empty system message, `<image>` markers
    /// auto-prepended) is built and prefilled exactly like the it2i condition prefix (vision features
    /// spliced into `<IMG_CONTEXT>`), then [`Qwen3Backbone::generate`] decodes to the `<|im_end|>`
    /// stop. `images` are decoded RGB `[3,H,W]` in `[0,1]` (sized to multiples of `patch·merge`);
    /// pass an empty slice for a text-only question. Returns the decoded answer (special tokens
    /// stripped, trimmed).
    /// `cancel` is the request's cooperative cancellation handle (F-037): threaded into the AR decode
    /// so a long VQA / understanding rollout consumed directly by the worker is cancellable per token.
    #[allow(clippy::too_many_arguments)]
    pub fn vqa(
        &self,
        tokenizer: &TextTokenizer,
        question: &str,
        images: &[Array],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<String> {
        self.vqa_with_options(
            tokenizer,
            question,
            images,
            max_new_tokens,
            sampler,
            &T2iOptions::default(),
            cancel,
        )
    }

    /// Request-memory-aware VQA. Bounded attention applies to image/text prefill and every AR
    /// decode token; transformer block residency remains structurally N/A because VQA never enters
    /// the generation path.
    #[allow(clippy::too_many_arguments)]
    pub fn vqa_with_options(
        &self,
        tokenizer: &TextTokenizer,
        question: &str,
        images: &[Array],
        max_new_tokens: usize,
        sampler: Sampler,
        opts: &T2iOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<String> {
        self.vqa_with_options_and_progress(
            tokenizer,
            question,
            images,
            max_new_tokens,
            sampler,
            opts,
            cancel,
            &mut |_| {},
        )
    }

    /// Progress-capable sibling preserving the original entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn vqa_with_options_and_progress(
        &self,
        tokenizer: &TextTokenizer,
        question: &str,
        images: &[Array],
        max_new_tokens: usize,
        sampler: Sampler,
        opts: &T2iOptions,
        cancel: Option<&CancelFlag>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<String> {
        execution::observe_document_policy(
            serde_json::json!({"effective_options":format!("{opts:?}"),"native_route":"sensenova_understanding"}),
        );
        // Preprocess any source images.
        let mut pv_parts = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for img in images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Array> = pv_parts.iter().collect();
            Some(mlx_rs::ops::concatenate_axis(&refs, 0)?)
        };

        // Auto-prepend `<image>` markers (the reference `chat` prepends one per missing image).
        let count = question.matches("<image>").count();
        let mut q = question.to_string();
        if images.len() > count {
            let pre = "<image>\n".repeat(images.len() - count);
            q = format!("{pre}{q}");
        }
        // VQA uses the neo1_0 default (empty) system message, not SYSTEM_MESSAGE_FOR_GEN. Prime an
        // empty `<think></think>` block so the model answers directly without a chain-of-thought —
        // matching the reference `chat(think=False)` (`modeling_neo_chat.py`:
        // `get_prompt() + '<think>\n\n</think>\n\n'`) and the engine's own no-think interleave path.
        // Without it the greedy decode spends its whole budget reasoning and never emits the answer.
        let base = format!("{}<think>\n\n</think>\n\n", build_neo1_query(&q, ""));
        let ids = if images.is_empty() {
            tokenizer.encode_ids(&base, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &base, &grids)?
        };

        let (mut cache, last_logits, t_idx) = mlx_gen::gen_core::with_prompt_encoding(
            on_progress,
            || -> std::result::Result<_, mlx_gen::Error> {
                let (cache, last_logits, t_idx) = self.prefill_it2i_logits_budgeted(
                    &ids,
                    pixel_values.as_ref(),
                    &grids,
                    attention_plan(opts, cancel),
                )?;
                Ok((cache, last_logits, t_idx))
            },
        )?;
        on_progress(Progress::ConditioningComplete);
        let tokens = self.decode_text_budgeted(
            &last_logits,
            &mut cache,
            t_idx,
            &[tokens::IM_END],
            max_new_tokens,
            sampler,
            cancel,
            attention_plan(opts, cancel),
        )?;
        let u32s: Vec<u32> = tokens.iter().map(|&i| i as u32).collect();
        Ok(tokenizer.decode(&u32s, true)?.trim().to_string())
    }

    /// Re-encode a generated image through the **understanding** vision embedder and append it (plus
    /// the `</img>` token) to a text cache, so subsequent text generation attends to it (the
    /// reference `interleave_gen`'s inner `append_image_to_cache`). The generated `image` `[1,3,H,W]`
    /// is mapped model-space→`[0,1]` (`·0.5+0.5`) then ImageNet-normalised. Image tokens take temporal
    /// `t_idx+1` with their merged-grid `(h,w)`; `</img>` takes `t_idx+2`. The block-causal mask makes
    /// image tokens attend bidirectionally to each other + all past but **not** to the `</img>`
    /// position, while `</img>` sees them — exactly what [`cached_block_mask`](crate::qwen3) yields
    /// for that temporal layout. Returns the next-token logits and the advanced `t_idx` (`+2`).
    /// Public so callers can compose custom interleave loops and so the parity test can drive it.
    pub fn append_generated_image(
        &self,
        image: &Array,
        token_h: i32,
        token_w: i32,
        t_idx: usize,
        cache: &mut KvCache,
    ) -> Result<(Vec<f32>, usize)> {
        self.append_generated_image_budgeted(
            image,
            token_h,
            token_w,
            t_idx,
            cache,
            AttentionPlan::UNBOUNDED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_generated_image_budgeted(
        &self,
        image: &Array,
        token_h: i32,
        token_w: i32,
        t_idx: usize,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<(Vec<f32>, usize)> {
        let sh = image.shape();
        let (h, w) = (sh[2], sh[3]);
        let raw = add(
            &multiply(image, Array::from_f32(0.5))?,
            Array::from_f32(0.5),
        )?
        .reshape(&[3, h, w])?;
        let (patches, (gh, gw)) = self.preprocess_image(&raw)?;
        let vit = self.und_vision_features(&patches, &[(gh, gw)])?; // [n_img, hidden]
        let n_img = vit.shape()[0];
        let hidden = vit.shape()[1];
        let end = self
            .backbone
            .embed(&Array::from_slice(&[self.img_end_id], &[1, 1]))?
            .reshape(&[1, hidden])?;
        let embeds =
            mlx_rs::ops::concatenate_axis(&[&vit, &end], 0)?.reshape(&[1, n_img + 1, hidden])?;

        let ti = t_idx as i32;
        let mut t = vec![ti + 1; n_img as usize];
        t.push(ti + 2);
        // Row-major 2D position ids for the merged image grid (+ trailing `img_end` token), with the
        // `n_img == token_h * token_w` cross-check that the old `let _ = (token_h, token_w);` hid (F-135).
        let (hh, ww) = merged_grid_position_ids(n_img, token_h, token_w)?;
        let hs = self.backbone.forward_cached_budgeted(
            &embeds,
            &t,
            &hh,
            &ww,
            Path::Und,
            cache,
            true,
            attention,
        )?;
        // Same one-row-kept pattern as the prefill paths (F-129): slice the kept hidden row (the `end`
        // token at index `n_img`) before `lm_head` instead of projecting all `n_img + 1` rows.
        let last_hidden = hs.take_axis(Array::from_slice(&[n_img], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?;
        let vocab = logits.shape()[2];
        let last = logits.reshape(&[vocab])?.as_slice::<f32>().to_vec();
        Ok((last, t_idx + 2))
    }

    /// Interleaved text-image generation (`interleave_gen`) — the **Document Studio** deliverable. A
    /// single rollout that alternates understanding-path text generation and gen-path flow-matching
    /// image generation: text streams until the model emits `<img>`, an image is generated (3-cache
    /// CFG: condition / text-uncondition / image-uncondition) and re-encoded back into the text
    /// caches, then text resumes — all as **single-path** forwards over growing caches (the upstream
    /// mixed-token attention is never issued). `input_images` are optional source images;
    /// `system_message` is normally [`crate::INTERLEAVE_SYSTEM_MESSAGE`]; `init_noises`, when supplied, are
    /// per-image standard-normal `[1,3,H,W]` tensors for cross-build parity. Returns the composed text
    /// (with `<image>` placeholders) and the generated images in order.
    /// `cancel` is the request's cooperative cancellation handle (F-037): the AR text rollout checks it
    /// per decoded token and each per-image denoise runs under a [`StepReporter`] built from `cancel` +
    /// `on_progress`, so a worker-consumed Document Studio interleave is cancellable and reports denoise
    /// progress (the prior F-128 fix covered only the registry T2I path). Returns [`Error::Canceled`] on
    /// trip.
    #[allow(clippy::too_many_arguments)]
    pub fn interleave_gen(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        input_images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        init_noises: Option<&[Array]>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<InterleaveOutput> {
        let (output, _receipt) = self.interleave_gen_with_conditional_receipt(
            tokenizer,
            prompt,
            input_images,
            width,
            height,
            opts,
            system_message,
            max_new_tokens,
            max_images,
            init_noises,
            None,
            None,
            cancel,
            on_progress,
        )?;
        Ok(output)
    }

    /// Interleaved generation executed on an explicit pre-load [`ConditionalPlan`] and returning the
    /// realized [`ConditionalReceipt`] alongside [`InterleaveOutput`] (`§6.10`, `B2`, `SW-6974`).
    #[allow(clippy::too_many_arguments)]
    pub fn interleave_gen_with_conditional_receipt(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        input_images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        init_noises: Option<&[Array]>,
        precompiled_plan: Option<(&ConditionalInput, &ConditionalPlan)>,
        ceilings: Option<ConditionalCeilings>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(InterleaveOutput, ConditionalReceipt)> {
        self.interleave_gen_for_provider_with_conditional_receipt(
            crate::MODEL_ID,
            tokenizer,
            prompt,
            input_images,
            width,
            height,
            opts,
            system_message,
            max_new_tokens,
            max_images,
            init_noises,
            precompiled_plan,
            ceilings,
            cancel,
            on_progress,
        )
    }

    /// Provider-id parameterized variant of [`Self::interleave_gen_with_conditional_receipt`] so
    /// both `sensenova_u1_8b` and `sensenova_u1_8b_fast` compile and verify their exact `provider_id`.
    #[allow(clippy::too_many_arguments)]
    pub fn interleave_gen_for_provider_with_conditional_receipt(
        &self,
        provider_id: &str,
        tokenizer: &TextTokenizer,
        prompt: &str,
        input_images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        init_noises: Option<&[Array]>,
        precompiled_plan: Option<(&ConditionalInput, &ConditionalPlan)>,
        ceilings: Option<ConditionalCeilings>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(InterleaveOutput, ConditionalReceipt)> {
        execution::observe_document_policy(
            serde_json::json!({"effective_options":format!("{opts:?}"),"native_route":"sensenova_understanding"}),
        );
        if opts.cfg_norm == CfgNorm::CfgZeroStar {
            return Err(cfg_zero_star_it2i_err());
        }
        let cell = self.cell();
        if width <= 0 || height <= 0 || width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova interleave: width/height must be positive multiples of {cell}, got {width}x{height}"
            )));
        }
        let (width_u32, height_u32) = (width as u32, height as u32);
        if let Some(precompiled) = precompiled_plan {
            validate_precompiled_interleave_request(
                provider_id,
                width_u32,
                height_u32,
                opts.seed,
                input_images.len(),
                max_new_tokens,
                max_images,
                opts.num_steps,
                precompiled,
                ceilings.as_ref(),
            )?;
        }
        let cell_u32 = u32::try_from(cell)
            .map_err(|_| Error::Msg("sensenova interleave: negative cell".into()))?;
        let mut source_geometries = Vec::with_capacity(input_images.len());
        for img in input_images {
            source_geometries.push(source_image_geometry(img, cell_u32)?);
        }

        let token_h = height / cell;
        let token_w = width / cell;

        let grids_from_sources: Vec<(i32, i32)> = source_geometries
            .iter()
            .map(|g| {
                (
                    (g.height as i32) / self.patch_size,
                    (g.width as i32) / self.patch_size,
                )
            })
            .collect();
        let ([cond_ids, tu_ids, iu_ids], grids) = build_interleave_prefix_ids(
            tokenizer,
            prompt,
            system_message,
            opts.think_mode,
            &grids_from_sources,
            self.merge_size,
        )?;
        let prefill_tokens = [
            u32::try_from(cond_ids.len()).map_err(|_| Error::Msg("cond prefix overflow".into()))?,
            u32::try_from(tu_ids.len()).map_err(|_| Error::Msg("tu prefix overflow".into()))?,
            u32::try_from(iu_ids.len()).map_err(|_| Error::Msg("iu prefix overflow".into()))?,
        ];
        let (compiled_input, compiled_plan) = resolve_interleave_conditional_plan(
            provider_id,
            prompt,
            opts.seed,
            width_u32,
            height_u32,
            &source_geometries,
            prefill_tokens,
            self.prefill_shape,
            max_new_tokens,
            max_images,
            opts.num_steps,
            self.resident_bytes_estimate,
            self.cache_token_bytes,
            precompiled_plan,
            ceilings,
        )?;

        // Source images (optional). Only preprocessed after the pre-load plan admits geometry/tokens.
        let mut pv_parts = Vec::with_capacity(input_images.len());
        let mut actual_grids = Vec::with_capacity(input_images.len());
        for img in input_images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            actual_grids.push(g);
        }
        debug_assert_eq!(actual_grids, grids);
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Array> = pv_parts.iter().collect();
            Some(mlx_rs::ops::concatenate_axis(&refs, 0)?)
        };

        let ops = ProductionInterleaveOps {
            model: self,
            tokenizer,
            opts,
            width,
            height,
            token_h,
            token_w,
            init_noises,
            source_geometries,
            prefill_tokens,
            cond_ids,
            tu_ids,
            iu_ids,
            pixel_values,
            grids,
        };
        let mut executor = SenseNovaInterleaveExecutor::new(ops, cancel);
        let output = run_conditional(
            &compiled_plan,
            &compiled_input,
            cancel,
            &mut executor,
            on_progress,
        )
        .map_err(Error::from)?;
        Ok((
            InterleaveOutput {
                text: output.text,
                images: output.images,
            },
            output.receipt,
        ))
    }

    /// Build + run a single slice of the IT2I / interleave [`SenseNovaFmDriver`]. Shared between
    /// [`Self::it2i_denoise`] and [`ProductionInterleaveOps`] (`AP9`, `SW-6974`).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run_it2i_driver_slice(
        &self,
        cache_cond: &mut KvCache,
        rm_cond: &RopeMask,
        mut cache_img: Option<(&mut KvCache, &RopeMask)>,
        mut cache_uncond: Option<(&mut KvCache, &RopeMask)>,
        noise_embed: &Option<Array>,
        timesteps: Vec<f32>,
        grid_h: i32,
        grid_w: i32,
        l: i32,
        token_h: i32,
        token_w: i32,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        slice: SliceSpec,
        total_steps: usize,
        image: Array,
        seed: u64,
        cancel: &CancelFlag,
        driver_state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> Result<Array> {
        let attention = attention_plan(opts, Some(cancel));
        let has_img = cache_img.is_some();
        let has_uncond = cache_uncond.is_some();
        let forward = ProductionSenseNovaForward {
            model: self,
            grid_h,
            grid_w,
            l,
            cell: self.cell(),
            token_h,
            token_w,
            noise_embed,
            t_eps: opts.t_eps,
            attention,
            transformer_window: opts.transformer_window_size.map(|window| window as usize),
            calibration_stream_fault: opts.calibration_stream_fault,
            cache_cond: RefCell::new(cache_cond),
            rm_cond,
            cache_img: cache_img
                .as_mut()
                .map(|(c, rm)| (RefCell::new(&mut **c), *rm)),
            cache_uncond: cache_uncond
                .as_mut()
                .map(|(c, rm)| (RefCell::new(&mut **c), *rm)),
        };
        let driver = SenseNovaFmDriver::new(
            "sensenova_it2i_euler",
            timesteps,
            SenseNovaCfgMode::It2i,
            opts,
            has_img,
            has_uncond,
            forward,
        )?;
        let out_state = run_mlx_driver_slices(
            &driver,
            [SliceSpec {
                geometry: Some((width as u32, height as u32)),
                ..slice
            }],
            BoundaryState::XSigma(image),
            total_steps,
            seed,
            cancel,
            None,
            driver_state,
            hooks,
            |_, out, _| Ok(out),
        )?;
        Ok(match out_state {
            BoundaryState::XSigma(img)
            | BoundaryState::CleanEstimate { latent: img, .. }
            | BoundaryState::RouteSpecific { state: img, .. } => img,
        })
    }

    /// The dual-guidance denoise loop (`it2i_generate`'s body). `cond`/`img`/`uncond` are
    /// `(cache, image-block temporal index)`; the per-step blend follows the reference's
    /// `cfg_scale`/`img_cfg_scale` cases, then optional `cfg_norm`. Returns the image trajectory.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn it2i_denoise(
        &self,
        cond: (&mut KvCache, usize),
        mut img: Option<(&mut KvCache, usize)>,
        mut uncond: Option<(&mut KvCache, usize)>,
        width: i32,
        height: i32,
        base_noise: &Array,
        opts: &T2iOptions,
        mut reporter: Option<StepReporter>,
    ) -> Result<Vec<Array>> {
        let cell = self.cell();
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;
        let (cache_cond, cond_t) = cond;

        // Validate guidance cache requirements & CfgNorm before building RoPE/masks (F-126, F-131).
        let (needs_img, needs_uncond) = it2i_cache_requirements(opts.cfg_scale, opts.img_cfg_scale);
        if needs_img && img.is_none() {
            return Err(img_cache_err());
        }
        if needs_uncond && uncond.is_none() {
            return Err(uncond_cache_err());
        }
        if opts.cfg_norm == CfgNorm::CfgZeroStar {
            return Err(cfg_zero_star_it2i_err());
        }

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let image = multiply(
            &base_noise.as_dtype(Dtype::Float32)?,
            Array::from_f32(noise_scale),
        )?;

        let steps = opts.num_steps;
        let timesteps = step_schedule(opts)?;
        execution::observe_image_schedule(&timesteps);
        execution::observe_document_policy(
            serde_json::json!({"denoise_options":format!("{opts:?}"),"noise_scale_bits":noise_scale.to_bits(),"width":width,"height":height}),
        );
        let noise_embed = self.noise_scale_embed(noise_scale, l)?;

        // RoPE tables + block mask are invariant across steps for a given cache (F-139).
        let rm_cond = self.prepare_gen(token_h, token_w, cond_t, cache_cond.offset())?;
        let rm_img = match img.as_ref() {
            Some((c, tl)) => Some(self.prepare_gen(token_h, token_w, *tl, c.offset())?),
            None => None,
        };
        let rm_uncond = match uncond.as_ref() {
            Some((c, tl)) => Some(self.prepare_gen(token_h, token_w, *tl, c.offset())?),
            None => None,
        };

        let fallback_cancel = CancelFlag::default();
        let cancel: &CancelFlag = match reporter.as_ref() {
            Some(r) => r.cancel_flag(),
            None => &fallback_cancel,
        };
        // Conditioning (the prefills) is complete: open the denoise phase before the first step
        // (sc-22738).
        if let Some(r) = reporter.as_mut() {
            r.open_denoise_phase();
        }
        let mut state = DriverRunState::new();
        let mut progress_cb = |event: Progress| {
            if let Progress::Step { current, total } = event {
                if let Some(r) = reporter.as_mut() {
                    r.step(current as usize, total as usize);
                }
            }
        };
        let mut hooks = StepHooks::new(&mut progress_cb);
        let mut on_materialize = |_info: &DriverStepInfo, img_arr: &Array| {
            // Per-step eval so the cancel check interrupts mid-render (sc-5522 / sc-5399).
            img_arr
                .eval()
                .map_err(|e| mlx_gen::gen_core::Error::from(Error::from(e)))
        };
        hooks.on_materialize = Some(&mut on_materialize);
        let image = self.run_it2i_driver_slice(
            cache_cond,
            &rm_cond,
            img.as_mut()
                .zip(rm_img.as_ref())
                .map(|((c, _), rm)| (&mut **c, rm)),
            uncond
                .as_mut()
                .zip(rm_uncond.as_ref())
                .map(|((c, _), rm)| (&mut **c, rm)),
            &noise_embed,
            timesteps.clone(),
            grid_h,
            grid_w,
            l,
            token_h,
            token_w,
            width,
            height,
            opts,
            SliceSpec::full(steps),
            steps,
            image,
            opts.seed,
            cancel,
            &mut state,
            &mut hooks,
        )?;
        // F-036: keep only the final evaluated frame (1-element Vec preserves signature).
        let traj = vec![image.clone()];
        // The last step is done: open the decode phase before the RGB patches become an image
        // (sc-22738).
        if let Some(r) = reporter.as_mut() {
            r.open_decode_phase();
        }
        mlx_gen::image_execution::policy_debug(
            "sensenova.attention_score_budget",
            &opts.attention_score_budget,
        );
        mlx_gen::image_execution::policy_debug(
            "sensenova.transformer_window",
            &opts.transformer_window_size,
        );
        mlx_gen::image_execution::policy_debug("sensenova.cfg_norm", &opts.cfg_norm);
        mlx_gen::image_execution::policy_debug(
            "sensenova.cfg_interval_bits",
            &(opts.cfg_interval.0.to_bits(), opts.cfg_interval.1.to_bits()),
        );
        mlx_gen::image_execution::scalar("sensenova.cfg_scale", opts.cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.img_cfg_scale", opts.img_cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.noise_scale", noise_scale);
        mlx_gen::image_execution::scalar("sensenova.t_eps", opts.t_eps);
        mlx_gen::image_execution::sampled_native_from_primary_seed(
            "sensenova-it2i_denoise-euler-v1",
            "noise-fraction",
            image.dtype(),
            &timesteps,
            "splitmix64-offset-one-box-muller-f64-to-f32-v1;initial=output-seed;no-step-draws",
        );
        Ok(traj)
    }
}

/// Default conservative ceilings for unbounded standalone SenseNova interleave calls.
pub fn default_interleave_ceilings() -> ConditionalCeilings {
    ConditionalCeilings::sensenova_default()
}

/// Pre-load / admission source geometry extractor for `[3, H, W]` MLX image tensors.
pub fn source_image_geometry(rgb: &Array, cell: u32) -> Result<SourceImageGeometry> {
    let sh = rgb.shape();
    if sh.len() != 3 || sh[0] != 3 {
        return Err(Error::Msg(format!(
            "expected source image [3,H,W], got {sh:?}"
        )));
    }
    let (h, w) = (sh[1], sh[2]);
    if h <= 0 || w <= 0 {
        return Err(Error::Msg(format!(
            "source image must have positive dimensions, got {w}x{h}"
        )));
    }
    let h_u32 = u32::try_from(h).map_err(|_| Error::Msg("source image height overflow".into()))?;
    let w_u32 = u32::try_from(w).map_err(|_| Error::Msg("source image width overflow".into()))?;
    if cell == 0 || !h_u32.is_multiple_of(cell) || !w_u32.is_multiple_of(cell) {
        return Err(Error::Msg(format!(
            "source image H/W must be multiples of {cell}, got {h_u32}x{w_u32}"
        )));
    }
    let storage_bytes = rgb.nbytes() as u64;
    Ok(SourceImageGeometry {
        width: w_u32,
        height: h_u32,
        storage_bytes,
    })
}

fn build_interleave_query_ids(
    tokenizer: &TextTokenizer,
    base_query: &str,
    grids: &[(i32, i32)],
    merge_size: i32,
) -> Result<Vec<i32>> {
    if merge_size <= 0 {
        return Err(Error::Msg(format!(
            "sensenova interleave: merge_size must be > 0, got {merge_size}"
        )));
    }
    let mut query = base_query.to_string();
    for &(gh, gw) in grids {
        let n = (gh / merge_size) * (gw / merge_size);
        let block = format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n as usize));
        query = query.replacen("<image>", &block, 1);
    }
    Ok(tokenizer.encode_ids(&query, true)?)
}

#[allow(clippy::type_complexity)]
fn build_interleave_prefix_ids(
    tokenizer: &TextTokenizer,
    prompt: &str,
    system_message: &str,
    think_mode: bool,
    grids: &[(i32, i32)],
    merge_size: i32,
) -> Result<([Vec<i32>; 3], Vec<(i32, i32)>)> {
    let mut cond_query = build_neo1_query(prompt, system_message);
    if !think_mode {
        cond_query.push_str("<think>\n\n</think>\n\n");
    }
    let cond_ids = if grids.is_empty() {
        tokenizer.encode_ids(&cond_query, true)?
    } else {
        build_interleave_query_ids(tokenizer, &cond_query, grids, merge_size)?
    };
    let tu_query = build_neo1_query(&"<image>".repeat(grids.len()), "");
    let tu_ids = if grids.is_empty() {
        tokenizer.encode_ids(&tu_query, true)?
    } else {
        build_interleave_query_ids(tokenizer, &tu_query, grids, merge_size)?
    };
    let iu_query = format!("{}<img>", build_neo1_query("", ""));
    let iu_ids = tokenizer.encode_ids(&iu_query, true)?;
    Ok(([cond_ids, tu_ids, iu_ids], grids.to_vec()))
}

/// Validate live request arguments that do not require tokenization against an explicit
/// precompiled `(input, plan)` before entering a runtime scope or preprocessing source images.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_precompiled_interleave_request(
    provider_id: &str,
    width: u32,
    height: u32,
    seed: u64,
    input_image_count: usize,
    max_new_tokens: usize,
    max_images: usize,
    steps_per_image: usize,
    precompiled_plan: (&ConditionalInput, &ConditionalPlan),
    ceilings: Option<&ConditionalCeilings>,
) -> Result<()> {
    let (compiled_input, plan) = precompiled_plan;
    plan.validate().map_err(Error::from)?;
    plan.check_input(compiled_input).map_err(Error::from)?;
    if let Some(expected_ceilings) = ceilings {
        if plan.ceilings() != expected_ceilings {
            return Err(Error::Msg(
                "sensenova interleave: precompiled conditional plan ceilings differ from requested ceilings"
                    .into(),
            ));
        }
    }
    let expected = plan.input();
    let input_image_count = u32::try_from(input_image_count)
        .map_err(|_| Error::Msg("sensenova interleave: input_images count overflow".into()))?;
    let max_new_tokens = u32::try_from(max_new_tokens)
        .map_err(|_| Error::Msg("sensenova interleave: max_new_tokens overflow".into()))?;
    let max_images = u32::try_from(max_images)
        .map_err(|_| Error::Msg("sensenova interleave: max_images overflow".into()))?;
    let steps_per_image = u32::try_from(steps_per_image)
        .map_err(|_| Error::Msg("sensenova interleave: num_steps overflow".into()))?;
    if expected.provider_id != provider_id {
        return Err(Error::Msg(format!(
            "sensenova: precompiled conditional plan targets {:?}, loaded generator is {:?}",
            expected.provider_id, provider_id
        )));
    }
    if expected.width != width || expected.height != height {
        return Err(Error::Msg(format!(
            "sensenova interleave: live geometry {width}x{height} differs from precompiled plan {}x{}",
            expected.width, expected.height
        )));
    }
    if expected.seed != seed {
        return Err(Error::Msg(format!(
            "sensenova interleave: live seed {seed} differs from precompiled plan {}",
            expected.seed
        )));
    }
    if expected.input_images != input_image_count
        || expected.source_images.len() != input_image_count as usize
    {
        return Err(Error::Msg(format!(
            "sensenova interleave: live input_images count {input_image_count} differs from precompiled plan {}",
            expected.input_images
        )));
    }
    if expected.max_new_tokens != max_new_tokens {
        return Err(Error::Msg(format!(
            "sensenova interleave: live max_new_tokens {max_new_tokens} differs from precompiled plan {}",
            expected.max_new_tokens
        )));
    }
    if expected.max_images != max_images {
        return Err(Error::Msg(format!(
            "sensenova interleave: live max_images {max_images} differs from precompiled plan {}",
            expected.max_images
        )));
    }
    if expected.steps_per_image != steps_per_image {
        return Err(Error::Msg(format!(
            "sensenova interleave: live num_steps {steps_per_image} differs from precompiled plan {}",
            expected.steps_per_image
        )));
    }
    Ok(())
}

/// Resolve and admit the active `(ConditionalInput, ConditionalPlan)` for an interleave run
/// before source image tensor preprocessing or backbone execution.
#[allow(clippy::too_many_arguments)]
fn resolve_interleave_conditional_plan(
    provider_id: &str,
    prompt: &str,
    seed: u64,
    width: u32,
    height: u32,
    source_geometries: &[SourceImageGeometry],
    prefill_tokens: [u32; 3],
    shape: PrefillModelShape,
    max_new_tokens: usize,
    max_images: usize,
    steps_per_image: usize,
    model_resident_bytes: u64,
    cache_token_bytes: u64,
    precompiled_plan: Option<(&ConditionalInput, &ConditionalPlan)>,
    ceilings: Option<ConditionalCeilings>,
) -> Result<(ConditionalInput, ConditionalPlan)> {
    let (compiled_input, compiled_plan) = match precompiled_plan {
        Some((compiled_input, plan)) => {
            validate_precompiled_interleave_request(
                provider_id,
                width,
                height,
                seed,
                source_geometries.len(),
                max_new_tokens,
                max_images,
                steps_per_image,
                (compiled_input, plan),
                ceilings.as_ref(),
            )?;
            let (expected_input, _) = compile_interleave_conditional_plan_from_shape(
                provider_id,
                prompt,
                seed,
                width,
                height,
                source_geometries.to_vec(),
                prefill_tokens,
                shape,
                max_new_tokens,
                max_images,
                steps_per_image,
                model_resident_bytes,
                cache_token_bytes,
                Some(plan.ceilings().clone()),
            )?;
            plan.check_input(&expected_input).map_err(Error::from)?;
            if compiled_input != &expected_input {
                return Err(Error::Msg(
                    "sensenova interleave: precompiled conditional input differs from live execution parameters"
                        .into(),
                ));
            }
            (expected_input, plan.clone())
        }
        None => compile_interleave_conditional_plan_from_shape(
            provider_id,
            prompt,
            seed,
            width,
            height,
            source_geometries.to_vec(),
            prefill_tokens,
            shape,
            max_new_tokens,
            max_images,
            steps_per_image,
            model_resident_bytes,
            cache_token_bytes,
            ceilings,
        )?,
    };
    compiled_plan
        .check_source_images(source_geometries)
        .map_err(Error::from)?;
    compiled_plan
        .check_prefill_shape(shape)
        .map_err(Error::from)?;
    compiled_plan
        .check_prefill_tokens(prefill_tokens)
        .map_err(Error::from)?;
    Ok((compiled_input, compiled_plan))
}

/// Compile a pre-load [`ConditionalPlan`] from pure geometry, token counts, and architectural shape
/// without loading model weights (`§6.10`, `B2`, `SW-6974`).
#[allow(clippy::too_many_arguments)]
pub fn compile_interleave_conditional_plan_from_shape(
    provider_id: &str,
    prompt: &str,
    seed: u64,
    width: u32,
    height: u32,
    source_images: Vec<SourceImageGeometry>,
    prefill_tokens: [u32; 3],
    shape: PrefillModelShape,
    max_new_tokens: usize,
    max_images: usize,
    steps_per_image: usize,
    model_resident_bytes: u64,
    cache_token_bytes: u64,
    ceilings: Option<ConditionalCeilings>,
) -> Result<(ConditionalInput, ConditionalPlan)> {
    ConditionalPlan::compile_sensenova_interleave(
        provider_id,
        prompt,
        seed,
        width,
        height,
        source_images,
        prefill_tokens,
        shape,
        max_new_tokens,
        max_images,
        steps_per_image,
        model_resident_bytes,
        cache_token_bytes,
        ceilings,
    )
    .map_err(Error::from)
}

/// Compile a pre-load [`ConditionalPlan`] from `NeoChatConfig` + `TextTokenizer` (zero weights
/// loaded).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn compile_interleave_conditional_plan(
    provider_id: &str,
    cfg: &NeoChatConfig,
    tokenizer: &TextTokenizer,
    prompt: &str,
    source_images: Vec<SourceImageGeometry>,
    width: u32,
    height: u32,
    opts: &T2iOptions,
    system_message: &str,
    max_new_tokens: usize,
    max_images: usize,
    model_resident_bytes: u64,
    ceilings: Option<ConditionalCeilings>,
) -> Result<(
    ConditionalInput,
    ConditionalPlan,
    [Vec<i32>; 3],
    Vec<(i32, i32)>,
)> {
    let cell = u32::try_from(cfg.cell())
        .map_err(|_| Error::Msg("sensenova interleave: config cell overflow".into()))?;
    for source in &source_images {
        if source.width == 0
            || source.height == 0
            || cell == 0
            || !source.width.is_multiple_of(cell)
            || !source.height.is_multiple_of(cell)
        {
            return Err(Error::Msg(format!(
                "sensenova interleave: source image geometry {}x{} must be multiples of {cell}",
                source.width, source.height
            )));
        }
    }
    let patch_i32 = i32::try_from(cfg.patch_size)
        .map_err(|_| Error::Msg("sensenova interleave: patch_size overflow".into()))?;
    let merge_i32 = i32::try_from(cfg.merge_size())
        .map_err(|_| Error::Msg("sensenova interleave: merge_size overflow".into()))?;
    let grids_in: Vec<(i32, i32)> = source_images
        .iter()
        .map(|s| ((s.height as i32) / patch_i32, (s.width as i32) / patch_i32))
        .collect();
    let ([cond_ids, tu_ids, iu_ids], grids) = build_interleave_prefix_ids(
        tokenizer,
        prompt,
        system_message,
        opts.think_mode,
        &grids_in,
        merge_i32,
    )?;
    let prefill_tokens = [
        u32::try_from(cond_ids.len()).map_err(|_| Error::Msg("cond prefix overflow".into()))?,
        u32::try_from(tu_ids.len()).map_err(|_| Error::Msg("tu prefix overflow".into()))?,
        u32::try_from(iu_ids.len()).map_err(|_| Error::Msg("iu prefix overflow".into()))?,
    ];
    let resident = if model_resident_bytes > 0 {
        model_resident_bytes
    } else {
        cfg.estimated_resident_bytes()
    };
    let (input, plan) = compile_interleave_conditional_plan_from_shape(
        provider_id,
        prompt,
        opts.seed,
        width,
        height,
        source_images,
        prefill_tokens,
        cfg.prefill_model_shape(),
        max_new_tokens,
        max_images,
        opts.num_steps,
        resident,
        cfg.cache_token_bytes_bound(),
        ceilings,
    )?;
    Ok((input, plan, [cond_ids, tu_ids, iu_ids], grids))
}

/// Compile a pre-load [`ConditionalPlan`] directly from a SenseNova snapshot directory before
/// deserializing any `.safetensors` weights (`§6.10`). Reads only `config.json`, `tokenizer.json`,
/// and snapshot byte metadata.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn pre_load_interleave_conditional_plan(
    provider_id: &str,
    snapshot_dir: &std::path::Path,
    prompt: &str,
    source_images: Vec<SourceImageGeometry>,
    width: u32,
    height: u32,
    opts: &T2iOptions,
    system_message: &str,
    max_new_tokens: usize,
    max_images: usize,
    ceilings: Option<ConditionalCeilings>,
) -> Result<(
    ConditionalInput,
    ConditionalPlan,
    [Vec<i32>; 3],
    Vec<(i32, i32)>,
)> {
    let cfg = NeoChatConfig::from_dir(snapshot_dir)?;
    let tokenizer = crate::text::load_tokenizer(snapshot_dir)?;
    let resident = mlx_gen::safetensors_dir_bytes(snapshot_dir);
    compile_interleave_conditional_plan(
        provider_id,
        &cfg,
        &tokenizer,
        prompt,
        source_images,
        width,
        height,
        opts,
        system_message,
        max_new_tokens,
        max_images,
        resident,
        ceilings,
    )
}

/// Prefilled 3-cache interleave state returned by [`SenseNovaInterleaveOps::prefill_all`].
pub struct InterleavePrefill<C> {
    pub caches: [C; 3],
    pub segment_logits: Vec<f32>,
    pub t_cond: usize,
    pub t_tu: usize,
    pub iu_max: usize,
}

/// Backend operation seam for [`SenseNovaInterleaveExecutor`] (`ConditionalExecutor` impl),
/// enabling both production [`T2iModel`] execution and weights-free Tier 0 runner verification
/// over the exact same state machine and [`SenseNovaFmDriver`] slice contract (`§6.10`, `B2`).
pub trait SenseNovaInterleaveOps {
    type Cache;
    type ImageState;

    fn source_images(&self) -> Result<Vec<SourceImageGeometry>>;
    fn prefill_model_shape(&self) -> Result<PrefillModelShape>;
    fn prefill_token_counts(&self) -> Result<[u32; 3]>;

    fn prefill_all(&mut self, cancel: &CancelFlag) -> Result<InterleavePrefill<Self::Cache>>;

    fn rollout_segment(
        &mut self,
        segment_logits: &[f32],
        cond_cache: &mut Self::Cache,
        t_cond: usize,
        granted_draws: u32,
        cancel: &CancelFlag,
    ) -> Result<(TextRollout, usize)>;

    #[allow(clippy::too_many_arguments)]
    fn begin_image(
        &mut self,
        caches: &mut [Self::Cache; 3],
        t_cond: &mut usize,
        t_tu: &mut usize,
        iu_max: usize,
        image_idx: u32,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Self::ImageState>;

    #[allow(clippy::too_many_arguments)]
    fn denoise_step(
        &mut self,
        caches: &mut [Self::Cache; 3],
        state: &mut Self::ImageState,
        image_idx: u32,
        step: u32,
        cancel: &CancelFlag,
    ) -> Result<()>;

    fn finish_image(&mut self, state: Self::ImageState) -> Result<Array>;

    fn append_feedback(
        &mut self,
        caches: &mut [Self::Cache; 3],
        t_cond: &mut usize,
        t_tu: &mut usize,
        output: &Array,
        cancel: &CancelFlag,
    ) -> Result<Vec<f32>>;
}

pub struct InterleaveContext<ImageState> {
    segment_logits: Vec<f32>,
    t_cond: usize,
    t_tu: usize,
    iu_max: usize,
    active_image: Option<ImageState>,
}

pub struct SenseNovaInterleaveExecutor<'c, O> {
    ops: O,
    cancel: &'c CancelFlag,
}

impl<'c, O> SenseNovaInterleaveExecutor<'c, O> {
    pub fn new(ops: O, cancel: &'c CancelFlag) -> Self {
        Self { ops, cancel }
    }
}

impl<O: SenseNovaInterleaveOps> ConditionalExecutor for SenseNovaInterleaveExecutor<'_, O> {
    type Cache = O::Cache;
    type Context = InterleaveContext<O::ImageState>;
    type Image = Array;

    fn source_images(&self) -> mlx_gen::gen_core::Result<Vec<SourceImageGeometry>> {
        self.ops.source_images().map_err(Into::into)
    }

    fn prefill_model_shape(&self) -> mlx_gen::gen_core::Result<PrefillModelShape> {
        self.ops.prefill_model_shape().map_err(Into::into)
    }

    fn prefill_token_counts(&self) -> mlx_gen::gen_core::Result<[u32; 3]> {
        self.ops.prefill_token_counts().map_err(Into::into)
    }

    fn image_placeholder(&self) -> &'static str {
        "<image>"
    }

    fn prefill(
        &mut self,
        plan: &ConditionalPlan,
    ) -> mlx_gen::gen_core::Result<(Self::Context, [Self::Cache; 3])> {
        self.prefill_with_progress(plan, &mut |_| {})
    }

    fn prefill_with_progress(
        &mut self,
        _plan: &ConditionalPlan,
        progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<(Self::Context, [Self::Cache; 3])> {
        mlx_gen::gen_core::with_prompt_encoding(progress, || {
            let prefill = self
                .ops
                .prefill_all(self.cancel)
                .map_err(mlx_gen::gen_core::Error::from)?;
            Ok((
                InterleaveContext {
                    segment_logits: prefill.segment_logits,
                    t_cond: prefill.t_cond,
                    t_tu: prefill.t_tu,
                    iu_max: prefill.iu_max,
                    active_image: None,
                },
                prefill.caches,
            ))
        })
    }

    fn rollout(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; 3],
        granted_draws: u32,
    ) -> mlx_gen::gen_core::Result<TextRollout> {
        let (segment, next_t_cond) = self
            .ops
            .rollout_segment(
                &context.segment_logits,
                &mut caches[0],
                context.t_cond,
                granted_draws,
                self.cancel,
            )
            .map_err(mlx_gen::gen_core::Error::from)?;
        context.t_cond = next_t_cond;
        Ok(segment)
    }

    fn denoise_step(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; 3],
        image: u32,
        step: u32,
    ) -> mlx_gen::gen_core::Result<()> {
        self.denoise_step_with_progress(context, caches, image, step, &mut |_| {})
    }

    fn denoise_step_with_progress(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; 3],
        image: u32,
        step: u32,
        progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<()> {
        if step == 0 {
            let state = self
                .ops
                .begin_image(
                    caches,
                    &mut context.t_cond,
                    &mut context.t_tu,
                    context.iu_max,
                    image,
                    self.cancel,
                    progress,
                )
                .map_err(mlx_gen::gen_core::Error::from)?;
            context.active_image = Some(state);
        }
        let active = context.active_image.as_mut().ok_or_else(|| {
            mlx_gen::gen_core::Error::Msg("sensenova interleave: missing active image state".into())
        })?;
        self.ops
            .denoise_step(caches, active, image, step, self.cancel)
            .map_err(mlx_gen::gen_core::Error::from)
    }

    fn finish_image(
        &mut self,
        context: &mut Self::Context,
        _image: u32,
    ) -> mlx_gen::gen_core::Result<Self::Image> {
        let active = context.active_image.take().ok_or_else(|| {
            mlx_gen::gen_core::Error::Msg(
                "sensenova interleave: finish_image called before denoise".into(),
            )
        })?;
        self.ops
            .finish_image(active)
            .map_err(mlx_gen::gen_core::Error::from)
    }

    fn feedback(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; 3],
        image: u32,
        output: &Self::Image,
    ) -> mlx_gen::gen_core::Result<()> {
        self.feedback_with_progress(context, caches, image, output, &mut |_| {})
    }

    fn feedback_with_progress(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; 3],
        _image: u32,
        output: &Self::Image,
        progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<()> {
        mlx_gen::gen_core::with_prompt_encoding(progress, || {
            context.segment_logits = self
                .ops
                .append_feedback(
                    caches,
                    &mut context.t_cond,
                    &mut context.t_tu,
                    output,
                    self.cancel,
                )
                .map_err(mlx_gen::gen_core::Error::from)?;
            Ok(())
        })
    }
}

struct ProductionDenoiseState {
    image: Array,
    grid_h: i32,
    grid_w: i32,
    l: i32,
    noise_scale: f32,
    timesteps: Vec<f32>,
    noise_embed: Option<Array>,
    rm_cond: RopeMask,
    rm_tu: RopeMask,
    rm_iu: RopeMask,
    driver_state: DriverRunState,
}

struct ProductionInterleaveOps<'a> {
    model: &'a T2iModel,
    tokenizer: &'a TextTokenizer,
    opts: &'a T2iOptions,
    width: i32,
    height: i32,
    token_h: i32,
    token_w: i32,
    init_noises: Option<&'a [Array]>,
    source_geometries: Vec<SourceImageGeometry>,
    prefill_tokens: [u32; 3],
    cond_ids: Vec<i32>,
    tu_ids: Vec<i32>,
    iu_ids: Vec<i32>,
    pixel_values: Option<Array>,
    grids: Vec<(i32, i32)>,
}

impl SenseNovaInterleaveOps for ProductionInterleaveOps<'_> {
    type Cache = KvCache;
    type ImageState = ProductionDenoiseState;

    fn source_images(&self) -> Result<Vec<SourceImageGeometry>> {
        Ok(self.source_geometries.clone())
    }

    fn prefill_model_shape(&self) -> Result<PrefillModelShape> {
        Ok(self.model.prefill_model_shape())
    }

    fn prefill_token_counts(&self) -> Result<[u32; 3]> {
        Ok(self.prefill_tokens)
    }

    fn prefill_all(&mut self, cancel: &CancelFlag) -> Result<InterleavePrefill<KvCache>> {
        let attention = attention_plan(self.opts, Some(cancel));
        let (cache_cond, cond_logits, t_cond) = self.model.prefill_it2i_logits_budgeted(
            &self.cond_ids,
            self.pixel_values.as_ref(),
            &self.grids,
            attention,
        )?;
        let (cache_tu, _, t_tu) = self.model.prefill_it2i_logits_budgeted(
            &self.tu_ids,
            self.pixel_values.as_ref(),
            &self.grids,
            attention,
        )?;
        let (cache_iu, _, iu_max) =
            self.model
                .prefill_it2i_logits_budgeted(&self.iu_ids, None, &[], attention)?;
        Ok(InterleavePrefill {
            caches: [cache_cond, cache_tu, cache_iu],
            segment_logits: cond_logits,
            t_cond,
            t_tu,
            iu_max,
        })
    }

    fn rollout_segment(
        &mut self,
        segment_logits: &[f32],
        cond_cache: &mut KvCache,
        t_cond: usize,
        granted_draws: u32,
        cancel: &CancelFlag,
    ) -> Result<(TextRollout, usize)> {
        let rollout = self.model.backbone.rollout_budgeted(
            segment_logits,
            cond_cache,
            t_cond as i32,
            &[tokens::IM_END, self.model.img_start_id],
            granted_draws as usize,
            Sampler::Greedy,
            Some(cancel),
            attention_plan(self.opts, Some(cancel)),
        )?;
        let gen_tokens = rollout.tokens;
        let emitted = u32::try_from(gen_tokens.len())
            .map_err(|_| Error::Msg("sensenova rollout token count overflow".into()))?;
        let text = if gen_tokens.is_empty() {
            String::new()
        } else {
            let u32s: Vec<u32> = gen_tokens.iter().map(|&i| i as u32).collect();
            self.tokenizer.decode(&u32s, true)?
        };
        let stop = if rollout.stop == Some(self.model.img_start_id) {
            RolloutStop::Image
        } else if rollout.stop.is_some() {
            RolloutStop::End
        } else {
            RolloutStop::Exhausted
        };
        Ok((
            TextRollout {
                text,
                emitted,
                stop,
            },
            t_cond + gen_tokens.len(),
        ))
    }

    fn begin_image(
        &mut self,
        caches: &mut [KvCache; 3],
        t_cond: &mut usize,
        t_tu: &mut usize,
        iu_max: usize,
        image_idx: u32,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<ProductionDenoiseState> {
        let attention = attention_plan(self.opts, Some(cancel));
        *t_cond = self.model.backbone.append_tokens_budgeted(
            &[self.model.img_start_id],
            *t_cond as i32,
            &mut caches[0],
            attention,
        )? as usize;
        *t_tu = self.model.backbone.append_tokens_budgeted(
            &[self.model.img_start_id],
            *t_tu as i32,
            &mut caches[1],
            attention,
        )? as usize;

        let idx = image_idx as usize;
        let base_noise = match self.init_noises {
            Some(ns) if idx < ns.len() => ns[idx].as_dtype(Dtype::Float32)?,
            _ => gaussian(
                &[1, 3, self.height, self.width],
                self.opts.seed.wrapping_add(u64::from(image_idx)),
            )?,
        };
        if execution::observing() {
            if self.init_noises.is_some_and(|noises| idx < noises.len()) {
                execution::observe_document_image(
                    idx,
                    "explicit-f32-noise-sha256",
                    None,
                    Some(base_noise.as_slice::<f32>()),
                );
            } else {
                execution::observe_document_image(
                    idx,
                    "splitmix64-box-muller-v1",
                    Some(self.opts.seed.wrapping_add(u64::from(image_idx))),
                    None,
                );
            }
        }
        let grid_h = self.height / self.model.patch_size;
        let grid_w = self.width / self.model.patch_size;
        let l = self.token_h * self.token_w;
        let noise_scale = self.model.noise_scale_for(grid_h, grid_w);
        let image = multiply(&base_noise, Array::from_f32(noise_scale))?;
        let timesteps = step_schedule(self.opts)?;
        execution::observe_image_schedule(&timesteps);
        execution::observe_document_policy(
            serde_json::json!({"denoise_options":format!("{:?}", self.opts),"noise_scale_bits":noise_scale.to_bits(),"width":self.width,"height":self.height}),
        );
        let noise_embed = self.model.noise_scale_embed(noise_scale, l)?;
        let rm_cond =
            self.model
                .prepare_gen(self.token_h, self.token_w, *t_cond + 1, caches[0].offset())?;
        let rm_tu =
            self.model
                .prepare_gen(self.token_h, self.token_w, *t_tu + 1, caches[1].offset())?;
        let rm_iu =
            self.model
                .prepare_gen(self.token_h, self.token_w, iu_max + 1, caches[2].offset())?;
        if image_idx == 0 {
            progress(Progress::Loading(LoadPhase::Renderer));
        }
        Ok(ProductionDenoiseState {
            image,
            grid_h,
            grid_w,
            l,
            noise_scale,
            timesteps,
            noise_embed,
            rm_cond,
            rm_tu,
            rm_iu,
            driver_state: DriverRunState::new(),
        })
    }

    fn denoise_step(
        &mut self,
        caches: &mut [KvCache; 3],
        state: &mut ProductionDenoiseState,
        image_idx: u32,
        step: u32,
        cancel: &CancelFlag,
    ) -> Result<()> {
        let [cache_cond, cache_tu, cache_iu] = caches;
        let current = std::mem::replace(&mut state.image, Array::from_f32(0.0));
        let step_usize = step as usize;
        let slice = SliceSpec {
            pass: image_idx as usize,
            phase: 0,
            span: step_usize..step_usize + 1,
            transition: if step == 0 {
                SliceTransition::Start
            } else {
                SliceTransition::Continue
            },
            history: DriverHistoryPolicy::Carry,
            geometry: Some((self.width as u32, self.height as u32)),
            request_boundary_estimate: false,
        };
        let mut noop_progress = |_: Progress| {};
        let mut hooks = StepHooks::new(&mut noop_progress);
        let mut on_materialize = |_info: &DriverStepInfo, img_arr: &Array| {
            img_arr
                .eval()
                .map_err(|e| mlx_gen::gen_core::Error::from(Error::from(e)))
        };
        hooks.on_materialize = Some(&mut on_materialize);
        state.image = self.model.run_it2i_driver_slice(
            cache_cond,
            &state.rm_cond,
            Some((cache_tu, &state.rm_tu)),
            Some((cache_iu, &state.rm_iu)),
            &state.noise_embed,
            state.timesteps.clone(),
            state.grid_h,
            state.grid_w,
            state.l,
            self.token_h,
            self.token_w,
            self.width,
            self.height,
            self.opts,
            slice,
            self.opts.num_steps,
            current,
            self.opts.seed.wrapping_add(u64::from(image_idx)),
            cancel,
            &mut state.driver_state,
            &mut hooks,
        )?;
        Ok(())
    }

    fn finish_image(&mut self, state: ProductionDenoiseState) -> Result<Array> {
        mlx_gen::image_execution::policy_debug(
            "sensenova.attention_score_budget",
            &self.opts.attention_score_budget,
        );
        mlx_gen::image_execution::policy_debug(
            "sensenova.transformer_window",
            &self.opts.transformer_window_size,
        );
        mlx_gen::image_execution::policy_debug("sensenova.cfg_norm", &self.opts.cfg_norm);
        mlx_gen::image_execution::policy_debug(
            "sensenova.cfg_interval_bits",
            &(
                self.opts.cfg_interval.0.to_bits(),
                self.opts.cfg_interval.1.to_bits(),
            ),
        );
        mlx_gen::image_execution::scalar("sensenova.cfg_scale", self.opts.cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.img_cfg_scale", self.opts.img_cfg_scale);
        mlx_gen::image_execution::scalar("sensenova.noise_scale", state.noise_scale);
        mlx_gen::image_execution::scalar("sensenova.t_eps", self.opts.t_eps);
        mlx_gen::image_execution::sampled_native_from_primary_seed(
            "sensenova-it2i_denoise-euler-v1",
            "noise-fraction",
            state.image.dtype(),
            &state.timesteps,
            "splitmix64-offset-one-box-muller-f64-to-f32-v1;initial=output-seed;no-step-draws",
        );
        Ok(state.image)
    }

    fn append_feedback(
        &mut self,
        caches: &mut [KvCache; 3],
        t_cond: &mut usize,
        t_tu: &mut usize,
        output: &Array,
        cancel: &CancelFlag,
    ) -> Result<Vec<f32>> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let attention = attention_plan(self.opts, Some(cancel));
        let (cond_next, nt_cond) = self.model.append_generated_image_budgeted(
            output,
            self.token_h,
            self.token_w,
            *t_cond,
            &mut caches[0],
            attention,
        )?;
        *t_cond = nt_cond;
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (_, nt_tu) = self.model.append_generated_image_budgeted(
            output,
            self.token_h,
            self.token_w,
            *t_tu,
            &mut caches[1],
            attention,
        )?;
        *t_tu = nt_tu;
        Ok(cond_next)
    }
}

/// Drive any [`SenseNovaInterleaveOps`] implementation through the bounded `gen-core`
/// [`run_conditional`] contract (`§6.10`, `B2`, `SW-6974`).
pub fn run_sensenova_interleave_ops<O: SenseNovaInterleaveOps>(
    plan: &ConditionalPlan,
    input: &ConditionalInput,
    cancel: &CancelFlag,
    ops: O,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(InterleaveOutput, ConditionalReceipt)> {
    let mut executor = SenseNovaInterleaveExecutor::new(ops, cancel);
    let output =
        run_conditional(plan, input, cancel, &mut executor, on_progress).map_err(Error::from)?;
    Ok((
        InterleaveOutput {
            text: output.text,
            images: output.images,
        },
        output.receipt,
    ))
}

/// `(needs_img, needs_uncond)`: which extra caches `it2i_denoise` requires for the given guidance
/// scales. Mirrors the `needs_*` flags `it2i_generate` uses to decide which caches to build, so the
/// denoise guard and the cache construction agree (F-126).
fn it2i_cache_requirements(cfg_scale: f32, img_cfg_scale: f32) -> (bool, bool) {
    let needs_cfg = !(cfg_scale == 1.0 && img_cfg_scale == 1.0);
    let needs_img = needs_cfg && (img_cfg_scale == 1.0 || cfg_scale != img_cfg_scale);
    let needs_uncond = needs_cfg && img_cfg_scale != 1.0;
    (needs_img, needs_uncond)
}

fn img_cache_err() -> Error {
    Error::Msg(
        "it2i_denoise: image-CFG guidance needs an image-conditioned cache (`img`), but none was \
         supplied"
            .into(),
    )
}

fn uncond_cache_err() -> Error {
    Error::Msg(
        "it2i_denoise: guidance needs an uncond cache (`uncond`), but none was supplied".into(),
    )
}

/// Blend condition/uncondition velocities under the chosen [`CfgNorm`].
/// The flow-matching timestep schedule: linear `0..=1` over `opts.num_steps`, optionally exp-shifted
/// by [`apply_time_schedule`]. Returns the `num_steps + 1` boundary timesteps. Errors on
/// `num_steps == 0` (a 0/0 NaN schedule whose callers' `.last()` would panic; F-125). Shared by both
/// denoise loops (F-134).
fn step_schedule(opts: &T2iOptions) -> Result<Vec<f32>> {
    let steps = opts.num_steps;
    if steps == 0 {
        return Err(Error::Msg("sensenova: num_steps must be >= 1".into()));
    }
    let lin: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
    let lin_arr = Array::from_slice(&lin, &[(steps + 1) as i32]);
    let ts_arr = if opts.enable_timestep_shift {
        apply_time_schedule(&lin_arr, opts.timestep_shift)?
    } else {
        lin_arr
    };
    Ok(ts_arr.as_slice::<f32>().to_vec())
}

/// 2D position ids for an appended image's `n_img` merged-grid tokens followed by the trailing
/// `img_end` token: row-major `(i / token_w, i % token_w)` for the image rows, then `(0, 0)` for the
/// end token. Cross-checks the merged-grid token count against the caller's expected
/// `token_h × token_w` grid so an `n_img != token_h * token_w` mismatch surfaces instead of silently
/// mis-placing the position ids (the loop assumes exactly `token_w` columns per row; F-135).
fn merged_grid_position_ids(
    n_img: i32,
    token_h: i32,
    token_w: i32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    if n_img != token_h * token_w {
        return Err(Error::Msg(format!(
            "sensenova append_generated_image: re-encoded image has {n_img} merged tokens but the \
             caller's grid is {token_h}×{token_w} ({} expected)",
            token_h * token_w
        )));
    }
    let mut hh = Vec::with_capacity(n_img as usize + 1);
    let mut ww = Vec::with_capacity(n_img as usize + 1);
    for i in 0..n_img {
        hh.push(i / token_w);
        ww.push(i % token_w);
    }
    hh.push(0);
    ww.push(0);
    Ok((hh, ww))
}

fn cfg_blend(
    v_cond: &Array,
    v_uncond: &Array,
    scale: f32,
    norm: CfgNorm,
    step: usize,
) -> Result<Array> {
    if norm == CfgNorm::CfgZeroStar {
        // CFG-Zero*: project uncond onto cond (optimised scale), zero step 0.
        if step == 0 {
            return multiply(v_cond, Array::from_f32(0.0)).map_err(Error::from);
        }
        let alpha = optimized_scale(v_cond, v_uncond)?;
        let scaled_u = multiply(v_uncond, Array::from_f32(alpha))?;
        let guided = multiply(&subtract(v_cond, &scaled_u)?, Array::from_f32(scale))?;
        return add(&scaled_u, &guided).map_err(Error::from);
    }

    let diff = subtract(v_cond, v_uncond)?;
    let blended = add(v_uncond, &multiply(&diff, Array::from_f32(scale))?)?;
    // The Global/Channel norm-clamp (clamp the blended velocity's norm to the condition velocity's)
    // is identical to the it2i post-blend rescale; delegate to the single implementation (F-134).
    // `CfgZeroStar` returned above, so `norm` here is always None/Global/Channel and never reaches
    // `apply_cfg_norm`'s `CfgZeroStar => Err` arm.
    apply_cfg_norm(blended, v_cond, norm)
}

/// `cfg_norm=cfg_zero_star` is a T2I-only blend mode (optimized-scale projection + step-0 zeroing,
/// done inside [`cfg_blend`]), NOT a post-rescale — so it has no place on the it2i/edit path. The
/// reference `it2i_generate` asserts `cfg_norm in ['none','global','channel']`; we mirror that with a
/// typed error rather than letting it silently degrade to plain CFG (F-131).
fn cfg_zero_star_it2i_err() -> Error {
    Error::Msg(
        "sensenova it2i: cfg_norm=cfg_zero_star is T2I-only — the it2i/edit path supports only \
         none/global/channel (matching the reference it2i_generate assert)"
            .into(),
    )
}

/// Apply the post-blend `cfg_norm` rescale (`it2i_generate`): clamp the guided velocity's norm to
/// the condition velocity's (global = whole-tensor, channel = per-token). `None` is a no-op;
/// `CfgZeroStar` is rejected (it is a T2I-only blend mode, not a post-rescale — see
/// [`cfg_zero_star_it2i_err`]).
fn apply_cfg_norm(v: Array, out_cond: &Array, norm: CfgNorm) -> Result<Array> {
    match norm {
        CfgNorm::None => Ok(v),
        CfgNorm::Global => {
            let nc = frobenius(out_cond)?;
            let nv = frobenius(&v)?;
            let s = (nc / (nv + 1e-8)).clamp(0.0, 1.0);
            multiply(&v, Array::from_f32(s)).map_err(Error::from)
        }
        CfgNorm::Channel => {
            let nc = l2_last(out_cond)?;
            let nv = l2_last(&v)?;
            let ratio = divide(&nc, &add(&nv, Array::from_f32(1e-8))?)?;
            let s = minimum(&ratio, Array::from_f32(1.0))?;
            multiply(&v, &s).map_err(Error::from)
        }
        CfgNorm::CfgZeroStar => Err(cfg_zero_star_it2i_err()),
    }
}

/// `‖x‖₂` over the whole tensor (the reference `torch.norm(v, dim=(1,2))` for batch 1).
fn frobenius(x: &Array) -> Result<f32> {
    Ok(sum_all(&multiply(x, x)?)?.sqrt())
}

/// Per-token L2 norm over the last axis, keeping dims: `[1,L,D] → [1,L,1]`.
fn l2_last(x: &Array) -> Result<Array> {
    let rank = x.shape().len() as i32;
    sum_axes(&multiply(x, x)?, &[rank - 1], true)?
        .sqrt()
        .map_err(Error::from)
}

/// Sum every element to a scalar.
fn sum_all(x: &Array) -> Result<f32> {
    let axes: Vec<i32> = (0..x.shape().len() as i32).collect();
    Ok(sum_axes(x, &axes, false)?.item::<f32>())
}

/// CFG-Zero* optimised scale `⟨cond,uncond⟩ / ‖uncond‖²` (computed in f32).
fn optimized_scale(v_cond: &Array, v_uncond: &Array) -> Result<f32> {
    let dot = sum_all(&multiply(v_cond, v_uncond)?)?;
    let nrm = sum_all(&multiply(v_uncond, v_uncond)?)?;
    Ok(dot / (nrm + 1e-8))
}

/// `smart_resize` (Qwen2.5-VL, the vendored `utils.smart_resize`): round `height`/`width` to
/// multiples of `factor` (use `patch·merge = 32`) with total pixels held in `[min_pixels,
/// max_pixels]`. Returns `(height, width)`.
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i64,
    max_pixels: i64,
) -> (i32, i32) {
    let round_by = |n: f64| ((n / factor as f64).round() as i32) * factor;
    let floor_by = |n: f64| ((n / factor as f64).floor() as i32) * factor;
    let ceil_by = |n: f64| ((n / factor as f64).ceil() as i32) * factor;
    let (hf, wf) = (height as f64, width as f64);
    let mut h_bar = factor.max(round_by(hf));
    let mut w_bar = factor.max(round_by(wf));
    let area = (h_bar as i64) * (w_bar as i64);
    if area > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max(floor_by(hf / beta));
        w_bar = factor.max(floor_by(wf / beta));
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_by(hf * beta);
        w_bar = ceil_by(wf * beta);
    }
    (h_bar, w_bar)
}

/// Standard-normal `[shape]` via Box–Muller over a SplitMix64 stream (deterministic per `seed`).
fn gaussian(shape: &[i32], seed: u64) -> Result<Array> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    // Reuse the shared `mlx_llm` `SplitMix64` so the scramble constants live in one place (F-133;
    // sc-7159). The original inline form pre-incremented `seed` once before producing any value, so
    // seed the RNG offset by one increment to keep the produced stream byte-identical. The f64 (0, 1]
    // mapping below is gaussian-specific (53-bit mantissa, biased off zero to keep `ln()` finite) and
    // differs from `SplitMix64::next_f32`, so it stays here.
    let mut rng = SplitMix64::new(seed.wrapping_add(SplitMix64::INCREMENT));
    let mut next_f = || ((rng.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_f();
        let u2 = next_f();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    Ok(Array::from_slice(&out, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(a: &Array) -> Vec<f32> {
        let n = a.shape().iter().product::<i32>();
        a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
    }

    #[test]
    fn gaussian_matches_inline_reference() {
        // Guard the F-133 dedup: the refactored `gaussian` (now reusing `mlx_llm::SplitMix64`) must
        // produce a byte-identical stream to the original inline SplitMix64 Box–Muller. Reconstruct
        // the original algorithm here independently so future drift in either copy is caught.
        fn reference(shape: &[i32], seed: u64) -> Vec<f32> {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut next_f = || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64)
            };
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let u1 = next_f();
                let u2 = next_f();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                out.push((r * theta.cos()) as f32);
                if out.len() < n {
                    out.push((r * theta.sin()) as f32);
                }
            }
            out
        }
        // Cover both even and odd element counts (the trailing-sin branch).
        for shape in [[3, 5].as_slice(), [1, 7].as_slice()] {
            for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
                let got = slice(&gaussian(shape, seed).unwrap());
                assert_eq!(got, reference(shape, seed), "shape={shape:?} seed={seed}");
            }
        }
    }

    #[test]
    fn armed_prompt_reporter_closes_at_denoise_or_early_return_and_cached_denoise_stays_silent() {
        for denoise in [false, true] {
            let cancel = CancelFlag::new();
            let mut events = Vec::new();
            {
                let mut sink = |event| events.push(event);
                let mut reporter = StepReporter::new(&cancel, &mut sink)
                    .with_phase_bounds(false, false)
                    .with_prompt_encoding();
                if denoise {
                    reporter.open_denoise_phase();
                    reporter.open_denoise_phase();
                }
            }
            assert_eq!(
                events,
                [
                    Progress::Loading(LoadPhase::EncodingPrompt),
                    Progress::PromptEncodingComplete
                ]
            );
            events.clear();
            {
                let mut sink = |event| events.push(event);
                let mut cached =
                    StepReporter::new(&cancel, &mut sink).with_phase_bounds(false, false);
                cached.open_denoise_phase();
            }
            assert!(
                events.is_empty(),
                "precomputed conditioning must not claim encoding"
            );
        }
    }

    #[test]
    fn step_reporter_cancels_and_reports_steps() {
        use mlx_gen::CancelFlag;

        // A cancelled flag makes the pre-step check error (the denoise loop aborts here).
        let cancelled = CancelFlag::new();
        cancelled.cancel();
        let mut sink = |_p: Progress| {};
        let r = StepReporter::new(&cancelled, &mut sink);
        assert!(r.check_cancel().is_err());

        // A live flag passes the check; step() forwards 1-based denoise progress (not the image index).
        let live = CancelFlag::new();
        let seen = std::cell::RefCell::new(Vec::new());
        let mut rec = |p: Progress| {
            if let Progress::Step { current, total } = p {
                seen.borrow_mut().push((current, total));
            }
        };
        let mut r = StepReporter::new(&live, &mut rec);
        assert!(r.check_cancel().is_ok());
        r.step(1, 4);
        r.step(4, 4);
        assert_eq!(*seen.borrow(), vec![(1u32, 4u32), (4, 4)]);
    }

    /// sc-22738: a single-image reporter brackets its denoise loop with the two lifecycle
    /// boundaries — `Loading(Renderer)` before the first step, `Decoding` after the last — and
    /// [`request_phase_bounds`] hands a multi-image request each boundary exactly once.
    #[test]
    fn step_reporter_opens_each_phase_once_per_request() {
        let live = CancelFlag::new();
        let mut events: Vec<Progress> = Vec::new();
        {
            let mut sink = |p: Progress| events.push(p);
            let mut r = StepReporter::new(&live, &mut sink);
            r.open_denoise_phase();
            r.step(1, 2);
            r.step(2, 2);
            r.open_decode_phase();
        }
        assert_eq!(
            events,
            vec![
                Progress::Loading(LoadPhase::Renderer),
                Progress::Step {
                    current: 1,
                    total: 2
                },
                Progress::Step {
                    current: 2,
                    total: 2
                },
                Progress::Decoding,
            ]
        );

        // A reporter that owns neither boundary reports only its steps.
        let mut events: Vec<Progress> = Vec::new();
        {
            let mut sink = |p: Progress| events.push(p);
            let mut r = StepReporter::new(&live, &mut sink).with_phase_bounds(false, false);
            r.open_denoise_phase();
            r.step(1, 1);
            r.open_decode_phase();
        }
        assert_eq!(
            events,
            vec![Progress::Step {
                current: 1,
                total: 1
            }]
        );

        // The first image opens denoise, the last opens decode, a single image opens both.
        assert_eq!(request_phase_bounds(0, 1), (true, true));
        assert_eq!(request_phase_bounds(0, 3), (true, false));
        assert_eq!(request_phase_bounds(1, 3), (false, false));
        assert_eq!(request_phase_bounds(2, 3), (false, true));
        // Every boundary opens exactly once across any count.
        for count in 1..=8u32 {
            let opened: Vec<(bool, bool)> =
                (0..count).map(|i| request_phase_bounds(i, count)).collect();
            assert_eq!(opened.iter().filter(|(d, _)| *d).count(), 1, "{count}");
            assert_eq!(opened.iter().filter(|(_, d)| *d).count(), 1, "{count}");
        }
    }

    /// sc-22738: the interleave loop's per-image reporters open the denoise phase on the first
    /// image only and never the decode phase; [`finish_interleave_progress`] closes the run with
    /// exactly one `Decoding`, after the bar fill, and only when an image was denoised.
    #[test]
    fn interleave_run_opens_denoise_once_and_closes_with_one_decode_boundary() {
        let cancel = CancelFlag::default();
        let num_steps = 2usize;
        let max_images = 3usize;
        let grand_total = max_images * num_steps;

        // Two of three images realized, driven the way `interleave_gen` drives its reporters.
        let mut events: Vec<Progress> = Vec::new();
        {
            let mut sink = |p: Progress| events.push(p);
            for image_idx in 0..2usize {
                let mut r = StepReporter::new_folded(
                    &cancel,
                    &mut sink,
                    image_idx * num_steps,
                    grand_total,
                )
                .with_phase_bounds(image_idx == 0, false);
                r.open_denoise_phase();
                for i in 0..num_steps {
                    r.step(i + 1, num_steps);
                }
                r.open_decode_phase();
            }
            finish_interleave_progress(&mut sink, 2, max_images, num_steps);
        }
        let step = |current: u32| Progress::Step {
            current,
            total: grand_total as u32,
        };
        assert_eq!(
            events,
            vec![
                Progress::Loading(LoadPhase::Renderer),
                step(1),
                step(2),
                step(3),
                step(4),
                step(6),
                Progress::Decoding,
            ],
            "one denoise boundary before the first step, one decode boundary after the fill"
        );

        // A full run: no fill, still exactly one terminal `Decoding`.
        let mut full: Vec<Progress> = Vec::new();
        finish_interleave_progress(&mut |p| full.push(p), max_images, max_images, num_steps);
        assert_eq!(full, vec![Progress::Decoding]);

        // A text-only run denoised nothing and opens no decode phase.
        let mut none: Vec<Progress> = Vec::new();
        finish_interleave_progress(&mut |p| none.push(p), 0, max_images, num_steps);
        assert!(none.is_empty(), "{none:?}");
    }

    #[test]
    fn it2i_cache_requirements_match_guidance_scales() {
        // F-126: the denoise guard's cache requirements must match it2i_generate's needs_* flags.
        assert_eq!(it2i_cache_requirements(1.0, 1.0), (false, false)); // no guidance
        assert_eq!(it2i_cache_requirements(4.0, 1.0), (true, false)); // image-CFG only
        assert_eq!(it2i_cache_requirements(4.0, 4.0), (false, true)); // uncond only
        assert_eq!(it2i_cache_requirements(4.0, 2.0), (true, true)); // dual guidance
                                                                     // The error builders name the missing cache so a caller knows what to pass.
        assert!(img_cache_err().to_string().contains("img"));
        assert!(uncond_cache_err().to_string().contains("uncond"));
    }

    #[test]
    fn cfg_blend_none_is_linear_extrapolation() {
        // v_uncond + scale·(v_cond − v_uncond): [1,1,2] tensors.
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::None, 1).unwrap();
        // 1 + 3·(2−1) = 4 ; 1 + 3·(4−1) = 10
        assert_eq!(slice(&out), vec![4.0, 10.0]);
    }

    #[test]
    fn cfg_zero_star_zeroes_first_step() {
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::CfgZeroStar, 0).unwrap();
        assert_eq!(slice(&out), vec![0.0, 0.0]);
    }

    /// F-131: `apply_cfg_norm` (the it2i/edit post-blend rescale) rejects `CfgZeroStar` rather than
    /// silently no-op'ing it (the old `_ => Ok(v)` arm); None/Global/Channel still apply.
    #[test]
    fn it2i_cfg_norm_rejects_cfg_zero_star() {
        let v = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let cond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        assert!(apply_cfg_norm(v.clone(), &cond, CfgNorm::CfgZeroStar).is_err());
        assert!(apply_cfg_norm(v.clone(), &cond, CfgNorm::None).is_ok());
        assert!(apply_cfg_norm(v, &cond, CfgNorm::Global).is_ok());
    }

    #[test]
    fn global_norm_never_amplifies() {
        // Blended norm > cond norm → scale clamps to keep ‖blended‖ ≤ ‖cond‖.
        let v_cond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[-2.0f32, -2.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 4.0, CfgNorm::Global, 1).unwrap();
        let on = (slice(&out).iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cn = (2.0f32).sqrt();
        assert!(
            on <= cn + 1e-4,
            "global-norm output {on} exceeds cond norm {cn}"
        );
    }

    #[test]
    fn merged_grid_position_ids_builds_rows_and_guards_mismatch() {
        // F-135: row-major (i/token_w, i%token_w) for the image grid, then (0,0) for the trailing
        // `img_end` token; the count is cross-checked against token_h × token_w.
        let (hh, ww) = merged_grid_position_ids(6, 2, 3).unwrap();
        assert_eq!(hh, vec![0, 0, 0, 1, 1, 1, 0]);
        assert_eq!(ww, vec![0, 1, 2, 0, 1, 2, 0]);

        // A merged-grid token count that doesn't match the caller's grid is rejected, not silently
        // mis-placed (the bug the old blanket suppression hid).
        assert!(merged_grid_position_ids(5, 2, 3).is_err());
        assert!(merged_grid_position_ids(6, 3, 3).is_err());
    }

    #[test]
    fn step_schedule_guards_zero_and_applies_shift() {
        // The extracted F-134 helper shared by both denoise loops: `num_steps == 0` errors (F-125),
        // the no-shift schedule is the plain linear `0..=1` boundary grid, and the shifted schedule
        // matches `apply_time_schedule` while staying a descending `1 → 0` walk.
        let base = T2iOptions::default();

        let zero = T2iOptions {
            num_steps: 0,
            ..base
        };
        assert!(step_schedule(&zero).is_err());

        let linear = T2iOptions {
            num_steps: 4,
            enable_timestep_shift: false,
            ..base
        };
        assert_eq!(
            step_schedule(&linear).unwrap(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );

        let shifted = T2iOptions {
            num_steps: 4,
            enable_timestep_shift: true,
            timestep_shift: 3.0,
            ..base
        };
        let ts = step_schedule(&shifted).unwrap();
        let lin = Array::from_slice(&[0.0f32, 0.25, 0.5, 0.75, 1.0], &[5]);
        let expected = slice(&apply_time_schedule(&lin, 3.0).unwrap());
        assert_eq!(ts, expected);
        assert_eq!(ts.first(), Some(&0.0));
        assert_eq!(ts.last(), Some(&1.0));
    }

    /// F-136 (sc-11133): a folded `StepReporter` per image produces one monotone, non-restarting bar
    /// across the interleave loop — `current` offset by prior steps, `total` pinned to the grand total.
    #[test]
    fn folded_step_reporter_aggregates_across_images() {
        let cancel = CancelFlag::default();
        let mut events: Vec<(u32, u32)> = Vec::new();
        let num_steps = 3usize;
        let max_images = 2usize;
        let grand_total = max_images * num_steps;
        {
            let mut sink = |p: Progress| {
                if let Progress::Step { current, total } = p {
                    events.push((current, total));
                }
            };
            // Simulate the interleave loop: one folded reporter per image, driving `num_steps` steps.
            for image_idx in 0..max_images {
                let offset = image_idx * num_steps;
                let mut r = StepReporter::new_folded(&cancel, &mut sink, offset, grand_total);
                for i in 0..num_steps {
                    r.step(i + 1, num_steps);
                }
            }
        }
        assert_eq!(
            events,
            vec![(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)],
            "the aggregate bar must be monotone 1..=6 with a constant total, not two restarting 1..=3 runs"
        );
    }

    /// F-136 residual (sc-11133): when `interleave_gen`'s model-driven loop stops early (fewer than
    /// `max_images` images), the folded bar freezes at `realized × num_steps` below the pinned grand
    /// total. `fill_interleave_bar` must emit a terminal `Step` so `current` reaches `total`.
    #[test]
    fn short_interleave_run_fills_bar_to_total() {
        let num_steps = 3usize;
        let max_images = 3usize; // grand total = 9
        let grand_total = (max_images * num_steps) as u32;

        // Case 1: the model produced only 1 of 3 images — the folded bar reached 3/9 and stalled.
        // Seed the prior folded steps for the single realized image (1..=3 of the grand total of 9).
        let mut events: Vec<(u32, u32)> =
            (1..=num_steps).map(|c| (c as u32, grand_total)).collect();
        {
            let mut sink = |p: Progress| {
                if let Progress::Step { current, total } = p {
                    events.push((current, total));
                }
            };
            fill_interleave_bar(&mut sink, 1, max_images, num_steps);
        }
        assert_eq!(
            events.last(),
            Some(&(grand_total, grand_total)),
            "a short run must fill the bar to its total so it does not freeze below total"
        );
        // Monotone, constant total, strictly increasing — matches the testkit progress contract.
        let mut prev = 0u32;
        for &(current, total) in &events {
            assert_eq!(total, grand_total, "total must stay constant");
            assert!(current > prev, "current must strictly increase");
            assert!(current <= total, "current must not overrun total");
            prev = current;
        }

        // Case 2: a full `max_images` run already reached total — no duplicate terminal (which would
        // break strict-increase).
        let mut full_events: Vec<(u32, u32)> = Vec::new();
        {
            let mut sink = |p: Progress| {
                if let Progress::Step { current, total } = p {
                    full_events.push((current, total));
                }
            };
            fill_interleave_bar(&mut sink, max_images, max_images, num_steps);
        }
        assert!(
            full_events.is_empty(),
            "a full run already reached total; no terminal fill should be emitted"
        );

        // Case 3: a 0-image run drew no bar — nothing to fill.
        let mut zero_events: Vec<(u32, u32)> = Vec::new();
        {
            let mut sink = |p: Progress| {
                if let Progress::Step { current, total } = p {
                    zero_events.push((current, total));
                }
            };
            fill_interleave_bar(&mut sink, 0, max_images, num_steps);
        }
        assert!(
            zero_events.is_empty(),
            "a 0-image run drew no bar; no terminal fill should be emitted"
        );
    }

    #[test]
    fn interleave_resolution_lookup() {
        assert_eq!(interleave_resolution_for("16:9"), Some((2048, 1152)));
        assert_eq!(interleave_resolution_for("1:1"), Some((1536, 1536)));
        assert_eq!(interleave_resolution_for("nope"), None);
        for (_, (w, h)) in INTERLEAVE_RESOLUTIONS {
            assert_eq!(w % 32, 0, "interleave bucket width not 32-aligned");
            assert_eq!(h % 32, 0, "interleave bucket height not 32-aligned");
        }
    }

    #[test]
    fn smart_resize_upscales_to_min_pixels() {
        // 100×100 rounds to 96×96 (< 65536 px) → upscaled to the 256×256 bucket.
        assert_eq!(smart_resize(100, 100, 32, 65536, 4_194_304), (256, 256));
    }

    #[test]
    fn smart_resize_keeps_in_range_multiple() {
        assert_eq!(smart_resize(512, 512, 32, 65536, 4_194_304), (512, 512));
        // Non-multiples round to the nearest factor.
        assert_eq!(smart_resize(500, 500, 32, 65536, 4_194_304), (512, 512));
    }

    struct MockVelocityForward {
        cell: i32,
        token_h: i32,
        token_w: i32,
    }

    impl SenseNovaVelocityForward for MockVelocityForward {
        fn prepare_step(&self, image: &Array, t: f32) -> Result<(Array, Array)> {
            let z = patchify(image, self.cell)?;
            let scaled = multiply(&z, Array::from_f32(t + 1.0))?;
            Ok((z, scaled))
        }

        fn predict_cond(&self, _z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
            add(cond_emb, Array::from_f32(0.1 * (1.0 + t))).map_err(Error::from)
        }

        fn predict_img(&self, _z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
            add(cond_emb, Array::from_f32(0.05 * (1.0 + t))).map_err(Error::from)
        }

        fn predict_uncond(&self, _z: &Array, cond_emb: &Array, t: f32) -> Result<Array> {
            subtract(cond_emb, Array::from_f32(0.02 * (1.0 + t))).map_err(Error::from)
        }

        fn unpatchify_step(&self, next_z: &Array) -> Result<Array> {
            unpatchify(next_z, self.cell, Some(self.token_h), Some(self.token_w))
        }
    }

    fn mlx_arrays_bit_equal(a: &Array, b: &Array) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        let va = slice(a);
        let vb = slice(b);
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn sensenova_fm_driver_t2i_and_it2i_pass_tier0_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let opts_t2i = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 1.0,
            cfg_norm: CfgNorm::None,
            num_steps: 4,
            enable_timestep_shift: false,
            ..T2iOptions::default()
        };
        let ts_t2i = step_schedule(&opts_t2i).unwrap();
        let report_t2i = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 19,
            base_geometry: (16, 16),
            alt_geometry: (32, 32),
            make_driver: || {
                Box::new(
                    SenseNovaFmDriver::new(
                        "sensenova_t2i_euler",
                        ts_t2i.clone(),
                        SenseNovaCfgMode::T2i,
                        &opts_t2i,
                        false,
                        true,
                        MockVelocityForward {
                            cell: 4,
                            token_h: 4,
                            token_w: 4,
                        },
                    )
                    .expect("valid T2I driver"),
                )
            },
            make_initial: || gaussian(&[1, 3, 16, 16], 19).unwrap(),
            latents_equal: mlx_arrays_bit_equal,
        })
        .expect("MLX SenseNova T2I driver passes Tier 0 conformance");
        assert!(report_t2i.verified_inert_split);
        assert!(report_t2i.verified_cancellation);
        assert!(report_t2i.verified_renoise_and_resample_guards);

        let opts_it2i = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 2.0,
            cfg_norm: CfgNorm::Global,
            num_steps: 4,
            enable_timestep_shift: false,
            ..T2iOptions::default()
        };
        let ts_it2i = step_schedule(&opts_it2i).unwrap();
        let report_it2i = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 23,
            base_geometry: (16, 16),
            alt_geometry: (32, 32),
            make_driver: || {
                Box::new(
                    SenseNovaFmDriver::new(
                        "sensenova_it2i_euler",
                        ts_it2i.clone(),
                        SenseNovaCfgMode::It2i,
                        &opts_it2i,
                        true,
                        true,
                        MockVelocityForward {
                            cell: 4,
                            token_h: 4,
                            token_w: 4,
                        },
                    )
                    .expect("valid IT2I driver"),
                )
            },
            make_initial: || gaussian(&[1, 3, 16, 16], 23).unwrap(),
            latents_equal: mlx_arrays_bit_equal,
        })
        .expect("MLX SenseNova IT2I driver passes Tier 0 conformance");
        assert!(report_it2i.verified_inert_split);
        assert!(report_it2i.verified_cancellation);

        // Missing guidance caches and T2I-only CfgZeroStar must be refused on IT2I driver construction.
        assert!(SenseNovaFmDriver::new(
            "sensenova_it2i_euler",
            ts_it2i.clone(),
            SenseNovaCfgMode::It2i,
            &opts_it2i,
            false,
            true,
            MockVelocityForward {
                cell: 4,
                token_h: 4,
                token_w: 4,
            },
        )
        .is_err());
        let opts_zero = T2iOptions {
            cfg_norm: CfgNorm::CfgZeroStar,
            ..opts_it2i
        };
        assert!(SenseNovaFmDriver::new(
            "sensenova_it2i_euler",
            ts_it2i,
            SenseNovaCfgMode::It2i,
            &opts_zero,
            true,
            true,
            MockVelocityForward {
                cell: 4,
                token_h: 4,
                token_w: 4,
            },
        )
        .is_err());
    }

    #[derive(Default)]
    struct MockInterleaveTrace {
        drops: std::rc::Rc<std::cell::Cell<usize>>,
        prefill_count: usize,
        rollout_calls: Vec<u32>,
        denoise_steps: Vec<(u32, u32)>,
        feedback_calls: Vec<(u32, [bool; 3])>,
        cache_tokens: [usize; 3],
    }

    struct TrackingCache {
        slot: usize,
        tokens: usize,
        drops: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl Drop for TrackingCache {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    struct MockDenoiseImageState {
        image: Array,
        timesteps: Vec<f32>,
        driver_state: DriverRunState,
    }

    struct MockInterleaveOps {
        segments: std::collections::VecDeque<TextRollout>,
        prefill_tokens: [u32; 3],
        source_geometries: Vec<SourceImageGeometry>,
        shape: PrefillModelShape,
        width: i32,
        height: i32,
        cell: i32,
        opts: T2iOptions,
        cancel_after_denoise_step: Option<(u32, u32)>,
        cancel_during_feedback: Option<u32>,
        trace: std::rc::Rc<RefCell<MockInterleaveTrace>>,
        feedback_idx: u32,
    }

    impl MockInterleaveOps {
        fn new(
            segments: Vec<TextRollout>,
            steps_per_image: usize,
            cancel_after_denoise_step: Option<(u32, u32)>,
            cancel_during_feedback: Option<u32>,
        ) -> (Self, std::rc::Rc<RefCell<MockInterleaveTrace>>) {
            let trace = std::rc::Rc::new(RefCell::new(MockInterleaveTrace::default()));
            let ops = Self {
                segments: segments.into(),
                prefill_tokens: [12, 6, 4],
                source_geometries: Vec::new(),
                shape: PrefillModelShape {
                    hidden_width: 128,
                    intermediate_width: 256,
                    vision_hidden_width: 64,
                    source_patch_size: 16,
                    source_merge_size: 2,
                    attention_heads: 4,
                    kv_heads: 2,
                    head_dim: 32,
                    layers: 2,
                    vocab_size: 64,
                },
                width: 64,
                height: 64,
                cell: 32,
                opts: T2iOptions {
                    cfg_scale: 3.0,
                    img_cfg_scale: 1.5,
                    cfg_norm: CfgNorm::Global,
                    num_steps: steps_per_image,
                    enable_timestep_shift: false,
                    ..T2iOptions::default()
                },
                cancel_after_denoise_step,
                cancel_during_feedback,
                trace: std::rc::Rc::clone(&trace),
                feedback_idx: 0,
            };
            (ops, trace)
        }
    }

    impl SenseNovaInterleaveOps for MockInterleaveOps {
        type Cache = TrackingCache;
        type ImageState = MockDenoiseImageState;

        fn source_images(&self) -> Result<Vec<SourceImageGeometry>> {
            Ok(self.source_geometries.clone())
        }

        fn prefill_model_shape(&self) -> Result<PrefillModelShape> {
            Ok(self.shape)
        }

        fn prefill_token_counts(&self) -> Result<[u32; 3]> {
            Ok(self.prefill_tokens)
        }

        fn prefill_all(&mut self, _cancel: &CancelFlag) -> Result<InterleavePrefill<Self::Cache>> {
            let mut tr = self.trace.borrow_mut();
            tr.prefill_count += 1;
            let drops = std::rc::Rc::clone(&tr.drops);
            tr.cache_tokens = [
                self.prefill_tokens[0] as usize,
                self.prefill_tokens[1] as usize,
                self.prefill_tokens[2] as usize,
            ];
            let caches = [
                TrackingCache {
                    slot: 0,
                    tokens: self.prefill_tokens[0] as usize,
                    drops: std::rc::Rc::clone(&drops),
                },
                TrackingCache {
                    slot: 1,
                    tokens: self.prefill_tokens[1] as usize,
                    drops: std::rc::Rc::clone(&drops),
                },
                TrackingCache {
                    slot: 2,
                    tokens: self.prefill_tokens[2] as usize,
                    drops,
                },
            ];
            Ok(InterleavePrefill {
                caches,
                segment_logits: vec![0.1, 0.2],
                t_cond: self.prefill_tokens[0] as usize,
                t_tu: self.prefill_tokens[1] as usize,
                iu_max: self.prefill_tokens[2] as usize,
            })
        }

        fn rollout_segment(
            &mut self,
            _segment_logits: &[f32],
            cond_cache: &mut Self::Cache,
            t_cond: usize,
            granted_draws: u32,
            _cancel: &CancelFlag,
        ) -> Result<(TextRollout, usize)> {
            let seg = self
                .segments
                .pop_front()
                .ok_or_else(|| Error::Msg("mock ran out of segments".into()))?;
            let draws = seg.emitted + u32::from(seg.stop != RolloutStop::Exhausted);
            cond_cache.tokens += draws as usize;
            let mut tr = self.trace.borrow_mut();
            tr.rollout_calls.push(granted_draws);
            tr.cache_tokens[cond_cache.slot] = cond_cache.tokens;
            Ok((seg, t_cond + draws as usize))
        }

        fn begin_image(
            &mut self,
            caches: &mut [Self::Cache; 3],
            t_cond: &mut usize,
            t_tu: &mut usize,
            _iu_max: usize,
            image_idx: u32,
            _cancel: &CancelFlag,
            progress: &mut dyn FnMut(Progress),
        ) -> Result<Self::ImageState> {
            caches[0].tokens += 1;
            caches[1].tokens += 1;
            *t_cond += 1;
            *t_tu += 1;
            if image_idx == 0 {
                progress(Progress::Loading(LoadPhase::Renderer));
            }
            let image = gaussian(
                &[1, 3, self.height, self.width],
                self.opts.seed.wrapping_add(u64::from(image_idx)),
            )?;
            let timesteps = step_schedule(&self.opts)?;
            Ok(MockDenoiseImageState {
                image,
                timesteps,
                driver_state: DriverRunState::new(),
            })
        }

        fn denoise_step(
            &mut self,
            _caches: &mut [Self::Cache; 3],
            state: &mut Self::ImageState,
            image_idx: u32,
            step: u32,
            cancel: &CancelFlag,
        ) -> Result<()> {
            self.trace
                .borrow_mut()
                .denoise_steps
                .push((image_idx, step));
            let token_h = self.height / self.cell;
            let token_w = self.width / self.cell;
            let driver = SenseNovaFmDriver::new(
                "sensenova_it2i_euler",
                state.timesteps.clone(),
                SenseNovaCfgMode::It2i,
                &self.opts,
                true,
                true,
                MockVelocityForward {
                    cell: self.cell,
                    token_h,
                    token_w,
                },
            )?;
            let current = std::mem::replace(&mut state.image, Array::from_f32(0.0));
            let step_us = step as usize;
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let out_state = run_mlx_driver_slices(
                &driver,
                [SliceSpec {
                    pass: image_idx as usize,
                    phase: 0,
                    span: step_us..step_us + 1,
                    transition: if step == 0 {
                        SliceTransition::Start
                    } else {
                        SliceTransition::Continue
                    },
                    history: DriverHistoryPolicy::Carry,
                    geometry: Some((self.width as u32, self.height as u32)),
                    request_boundary_estimate: false,
                }],
                BoundaryState::XSigma(current),
                self.opts.num_steps,
                self.opts.seed,
                cancel,
                None,
                &mut state.driver_state,
                &mut hooks,
                |_, out, _| Ok(out),
            )?;
            state.image = match out_state {
                BoundaryState::XSigma(img)
                | BoundaryState::CleanEstimate { latent: img, .. }
                | BoundaryState::RouteSpecific { state: img, .. } => img,
            };
            if self.cancel_after_denoise_step == Some((image_idx, step)) {
                cancel.cancel();
            }
            Ok(())
        }

        fn finish_image(&mut self, state: Self::ImageState) -> Result<Array> {
            Ok(state.image)
        }

        fn append_feedback(
            &mut self,
            caches: &mut [Self::Cache; 3],
            t_cond: &mut usize,
            t_tu: &mut usize,
            _output: &Array,
            cancel: &CancelFlag,
        ) -> Result<Vec<f32>> {
            let idx = self.feedback_idx;
            self.feedback_idx += 1;
            if self.cancel_during_feedback == Some(idx) {
                cancel.cancel();
                return Err(Error::Canceled);
            }
            let n_img = ((self.height / self.cell) * (self.width / self.cell)) as usize;
            caches[0].tokens += n_img + 1;
            caches[1].tokens += n_img + 1;
            *t_cond += 2;
            *t_tu += 2;
            let mut tr = self.trace.borrow_mut();
            tr.cache_tokens = [caches[0].tokens, caches[1].tokens, caches[2].tokens];
            tr.feedback_calls.push((idx, [true, true, false]));
            Ok(vec![idx as f32, 0.5])
        }
    }

    fn mock_plan(
        ops: &MockInterleaveOps,
        max_new_tokens: usize,
        max_images: usize,
    ) -> (ConditionalInput, ConditionalPlan) {
        let mut c = default_interleave_ceilings();
        c.max_new_tokens = max_new_tokens as u32;
        c.max_images = max_images as u32;
        c.max_steps_per_image = ops.opts.num_steps as u32;
        compile_interleave_conditional_plan_from_shape(
            crate::MODEL_ID,
            "interleave test",
            ops.opts.seed,
            ops.width as u32,
            ops.height as u32,
            ops.source_geometries.clone(),
            ops.prefill_tokens,
            ops.shape,
            max_new_tokens,
            max_images,
            ops.opts.num_steps,
            1_024 * 1_024,
            2 * 2 * 2 * 32 * 4,
            Some(c),
        )
        .expect("mock plan compiles")
    }

    #[test]
    fn sensenova_interleave_executor_zero_images_no_bar_no_decode_and_drops_caches() {
        let (ops, trace) = MockInterleaveOps::new(
            vec![TextRollout {
                text: "text only".into(),
                emitted: 4,
                stop: RolloutStop::End,
            }],
            4,
            None,
            None,
        );
        let (input, plan) = mock_plan(&ops, 32, 3);
        let cancel = CancelFlag::default();
        let mut events = Vec::new();
        let (out, receipt) =
            run_sensenova_interleave_ops(&plan, &input, &cancel, ops, &mut |e| events.push(e))
                .expect("zero-image interleave succeeds");
        assert_eq!(out.text, "text only");
        assert!(out.images.is_empty());
        assert_eq!(receipt.images, 0);
        assert_eq!(receipt.feedback_calls, 0);
        assert_eq!(receipt.actual_outer_steps, 0);
        assert_eq!(receipt.reported_final_steps, None);
        assert_eq!(
            receipt.terminal,
            mlx_gen::gen_core::render::conditional::TerminalReason::EndToken
        );
        assert!(!events.iter().any(|e| matches!(
            e,
            Progress::Step { .. } | Progress::Decoding | Progress::Loading(LoadPhase::Renderer)
        )));
        assert!(events.contains(&Progress::Loading(LoadPhase::EncodingPrompt)));
        assert!(events.contains(&Progress::PromptEncodingComplete));
        assert!(events.contains(&Progress::ConditioningComplete));
        let tr = trace.borrow();
        assert_eq!(tr.drops.get(), 3, "all 3 caches must drop cleanly");
        assert!(tr.feedback_calls.is_empty());
    }

    #[test]
    fn sensenova_interleave_executor_two_of_three_images_fills_progress_and_caches_feedback() {
        let (ops, trace) = MockInterleaveOps::new(
            vec![
                TextRollout {
                    text: "A".into(),
                    emitted: 2,
                    stop: RolloutStop::Image,
                },
                TextRollout {
                    text: "B".into(),
                    emitted: 2,
                    stop: RolloutStop::Image,
                },
                TextRollout {
                    text: "C".into(),
                    emitted: 2,
                    stop: RolloutStop::End,
                },
            ],
            4,
            None,
            None,
        );
        let (input, plan) = mock_plan(&ops, 32, 3);
        let cancel = CancelFlag::default();
        let mut events = Vec::new();
        let (out, receipt) =
            run_sensenova_interleave_ops(&plan, &input, &cancel, ops, &mut |e| events.push(e))
                .expect("two-of-three interleave succeeds");

        assert_eq!(out.text, "A<image>B<image>C");
        assert_eq!(out.images.len(), 2);
        assert_eq!(receipt.images, 2);
        assert_eq!(receipt.feedback_calls, 2);
        assert_eq!(receipt.actual_outer_steps, 8);
        assert_eq!(receipt.reported_final_steps, Some(12));
        assert_eq!(
            receipt.terminal,
            mlx_gen::gen_core::render::conditional::TerminalReason::EndToken
        );

        let steps: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|e| match *e {
                Progress::Step { current, total } => Some((current, total)),
                _ => None,
            })
            .collect();
        assert_eq!(
            steps,
            vec![
                (1, 12),
                (2, 12),
                (3, 12),
                (4, 12),
                (5, 12),
                (6, 12),
                (7, 12),
                (8, 12),
                (12, 12),
            ]
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Progress::Decoding))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Progress::Loading(LoadPhase::Renderer)))
                .count(),
            1
        );
        assert!(matches!(events.last(), Some(Progress::Decoding)));

        let tr = trace.borrow();
        assert_eq!(tr.drops.get(), 3, "all 3 caches must drop cleanly");
        assert_eq!(
            tr.feedback_calls,
            vec![(0, [true, true, false]), (1, [true, true, false])]
        );
        // Prefill(12,6,4) + rollouts(3+3+3=9) + 2*img_start(2,2) + 2*(4+1)(10,10).
        assert_eq!(tr.cache_tokens, [12 + 9 + 2 + 10, 6 + 2 + 10, 4]);
    }

    #[test]
    fn sensenova_interleave_executor_cancel_mid_second_image_and_no_cache_bleed() {
        let (ops, trace) = MockInterleaveOps::new(
            vec![
                TextRollout {
                    text: "A".into(),
                    emitted: 1,
                    stop: RolloutStop::Image,
                },
                TextRollout {
                    text: "B".into(),
                    emitted: 1,
                    stop: RolloutStop::Image,
                },
                TextRollout {
                    text: "SHOULD_NOT_RUN".into(),
                    emitted: 1,
                    stop: RolloutStop::End,
                },
            ],
            4,
            Some((1, 1)),
            None,
        );
        let (input, plan) = mock_plan(&ops, 32, 3);
        let cancel = CancelFlag::default();
        let mut events = Vec::new();
        let err =
            run_sensenova_interleave_ops(&plan, &input, &cancel, ops, &mut |e| events.push(e))
                .err()
                .expect("cancel mid second image must fail closed");
        assert!(matches!(err, Error::Canceled));
        assert!(!events.iter().any(|e| matches!(e, Progress::Decoding)));
        {
            let tr = trace.borrow();
            assert_eq!(
                tr.drops.get(),
                3,
                "cancelled attempt must drop all 3 caches"
            );
            assert_eq!(
                tr.denoise_steps,
                vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
                "cancellation during 2nd image must stop remaining steps"
            );
            assert_eq!(
                tr.feedback_calls.len(),
                1,
                "2nd image feedback must be suppressed after cancellation"
            );
        }

        // Subsequent clean run must see fresh caches with zero bleed from cancelled attempt.
        let (fresh_ops, fresh_trace) = MockInterleaveOps::new(
            vec![TextRollout {
                text: "clean".into(),
                emitted: 1,
                stop: RolloutStop::End,
            }],
            4,
            None,
            None,
        );
        let (fresh_input, fresh_plan) = mock_plan(&fresh_ops, 32, 3);
        let fresh_cancel = CancelFlag::default();
        let (out, receipt) = run_sensenova_interleave_ops(
            &fresh_plan,
            &fresh_input,
            &fresh_cancel,
            fresh_ops,
            &mut |_| {},
        )
        .expect("next request runs cleanly after canceled attempt");
        assert_eq!(out.text, "clean");
        assert_eq!(receipt.images, 0);
        assert_eq!(fresh_trace.borrow().cache_tokens, [12 + 2, 6, 4]);
    }

    #[test]
    fn sensenova_interleave_executor_cancel_mid_feedback_drops_caches() {
        let (ops, trace) = MockInterleaveOps::new(
            vec![
                TextRollout {
                    text: "A".into(),
                    emitted: 1,
                    stop: RolloutStop::Image,
                },
                TextRollout {
                    text: "NEVER".into(),
                    emitted: 1,
                    stop: RolloutStop::End,
                },
            ],
            4,
            None,
            Some(0),
        );
        let (input, plan) = mock_plan(&ops, 32, 3);
        let cancel = CancelFlag::default();
        let err = run_sensenova_interleave_ops(&plan, &input, &cancel, ops, &mut |_| {})
            .err()
            .expect("cancel during feedback must fail closed");
        assert!(matches!(err, Error::Canceled));
        let tr = trace.borrow();
        assert_eq!(tr.drops.get(), 3);
        assert_eq!(tr.rollout_calls.len(), 1);
    }

    #[test]
    fn preload_conditional_plan_from_dir_without_weights_and_invalid_bounds() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"llm_config":{},"vision_config":{}}"#,
        )
        .unwrap();
        let tok_json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151644, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151645, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151667, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151668, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151669, "content": "<IMG_CONTEXT>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151670, "content": "<img>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151671, "content": "</img>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2},
                "unk_token": "<unk>"
            }
        });
        std::fs::write(dir.path().join("tokenizer.json"), tok_json.to_string()).unwrap();

        let opts = T2iOptions {
            num_steps: 4,
            ..T2iOptions::default()
        };
        let (input, plan, _, _) = pre_load_interleave_conditional_plan(
            crate::MODEL_ID,
            dir.path(),
            "<image> hello world",
            vec![SourceImageGeometry {
                width: 64,
                height: 64,
                storage_bytes: 64 * 64 * 3 * 4,
            }],
            64,
            64,
            &opts,
            crate::INTERLEAVE_SYSTEM_MESSAGE,
            64,
            2,
            None,
        )
        .expect("pre-load conditional plan must compile with zero .safetensors weights");
        assert_eq!(input.provider_id, crate::MODEL_ID);
        assert!(plan.budget().peak_live_bytes > 0);

        let (_fast_in, fast_plan, _, _) = pre_load_interleave_conditional_plan(
            crate::MODEL_ID_FAST,
            dir.path(),
            "hello world",
            vec![],
            64,
            64,
            &opts,
            crate::INTERLEAVE_SYSTEM_MESSAGE,
            64,
            2,
            None,
        )
        .expect("fast variant pre-load conditional plan compiles");
        assert_eq!(fast_plan.input().provider_id, crate::MODEL_ID_FAST);

        // Unaligned dimensions and tight ceilings must be refused cleanly.
        assert!(pre_load_interleave_conditional_plan(
            crate::MODEL_ID,
            dir.path(),
            "hello",
            vec![],
            500,
            64,
            &opts,
            crate::INTERLEAVE_SYSTEM_MESSAGE,
            64,
            2,
            None,
        )
        .is_err());

        let mut tight = default_interleave_ceilings();
        tight.max_peak_bytes = fast_plan.budget().peak_live_bytes - 1;
        assert!(pre_load_interleave_conditional_plan(
            crate::MODEL_ID,
            dir.path(),
            "hello world",
            vec![],
            64,
            64,
            &opts,
            crate::INTERLEAVE_SYSTEM_MESSAGE,
            64,
            2,
            Some(tight),
        )
        .is_err());
    }

    #[test]
    fn precompiled_interleave_plan_rejects_live_parameter_and_ceiling_mismatches() {
        let shape = PrefillModelShape {
            hidden_width: 128,
            intermediate_width: 256,
            vision_hidden_width: 64,
            source_patch_size: 16,
            source_merge_size: 2,
            attention_heads: 4,
            kv_heads: 2,
            head_dim: 32,
            layers: 2,
            vocab_size: 64,
        };
        let source = vec![SourceImageGeometry {
            width: 64,
            height: 64,
            storage_bytes: 64 * 64 * 3 * 4,
        }];
        let prefill_tokens = [24, 12, 4];
        let resident_bytes = 1_024 * 1_024;
        let cache_token_bytes = shape.cache_token_bytes_bound();
        let mut ceilings = default_interleave_ceilings();
        ceilings.max_width = 256;
        ceilings.max_height = 256;
        ceilings.max_new_tokens = 128;
        ceilings.max_images = 4;
        ceilings.max_steps_per_image = 8;

        let (compiled_input, compiled_plan) = compile_interleave_conditional_plan_from_shape(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            source.clone(),
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some(ceilings.clone()),
        )
        .expect("precompiled plan compiles");

        // Matching live arguments pass with either `None` or matching `Some(ceilings)`.
        let (resolved_input, resolved_plan) = resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .expect("matching precompiled plan resolves");
        assert_eq!(resolved_input, compiled_input);
        assert_eq!(resolved_plan, compiled_plan);

        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            Some(ceilings.clone()),
        )
        .is_ok());

        // Mismatched `steps_per_image` vs `opts.num_steps` (still under max_steps_per_image=8).
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            6,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());

        // Mismatched width, height, and max_images (still within ceilings).
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            96,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            96,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            3,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());

        // Mismatched ceilings.
        let mut other_ceilings = ceilings.clone();
        other_ceilings.max_images = 2;
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            Some(other_ceilings.clone()),
        )
        .is_err());

        // Mismatched prompt, seed, provider_id, max_new_tokens, source geometry, prefill tokens, and shape.
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "other prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            99,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID_FAST,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            48,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &[SourceImageGeometry {
                width: 32,
                height: 32,
                storage_bytes: 32 * 32 * 3 * 4,
            }],
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            [25, 12, 4],
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        let mut wrong_shape = shape;
        wrong_shape.hidden_width = 256;
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            wrong_shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes + 1024,
            cache_token_bytes,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());
        assert!(resolve_interleave_conditional_plan(
            crate::MODEL_ID,
            "interleave prompt",
            42,
            64,
            64,
            &source,
            prefill_tokens,
            shape,
            32,
            2,
            4,
            resident_bytes,
            cache_token_bytes + 64,
            Some((&compiled_input, &compiled_plan)),
            None,
        )
        .is_err());

        // Mismatched `ConditionalInput` paired with `plan` is rejected by both request & full guards.
        let mut wrong_input = compiled_input.clone();
        wrong_input.steps_per_image = 6;
        assert!(validate_precompiled_interleave_request(
            crate::MODEL_ID,
            64,
            64,
            42,
            1,
            32,
            2,
            4,
            (&wrong_input, &compiled_plan),
            Some(&ceilings),
        )
        .is_err());
        assert!(validate_precompiled_interleave_request(
            crate::MODEL_ID,
            64,
            64,
            42,
            1,
            32,
            2,
            6,
            (&compiled_input, &compiled_plan),
            Some(&ceilings),
        )
        .is_err());
        assert!(validate_precompiled_interleave_request(
            crate::MODEL_ID,
            96,
            64,
            42,
            1,
            32,
            2,
            4,
            (&compiled_input, &compiled_plan),
            Some(&ceilings),
        )
        .is_err());
        assert!(validate_precompiled_interleave_request(
            crate::MODEL_ID,
            64,
            64,
            42,
            1,
            32,
            3,
            4,
            (&compiled_input, &compiled_plan),
            Some(&ceilings),
        )
        .is_err());
        assert!(validate_precompiled_interleave_request(
            crate::MODEL_ID,
            64,
            64,
            42,
            1,
            32,
            2,
            4,
            (&compiled_input, &compiled_plan),
            Some(&other_ceilings),
        )
        .is_err());
    }

    #[test]
    fn interleave_output_converts_to_generation_output_interleaved() {
        let out = InterleaveOutput {
            text: "one<image>two".into(),
            images: vec![mlx_rs::ops::zeros::<f32>(&[1, 3, 2, 4]).unwrap()],
        };
        let gen_out = GenerationOutput::try_from(out).expect("convert to GenerationOutput");
        match gen_out {
            GenerationOutput::Interleaved { text, images } => {
                assert_eq!(text, "one<image>two");
                assert_eq!(images.len(), 1);
                assert_eq!((images[0].width, images[0].height), (4, 2));
                assert_eq!(images[0].pixels.len(), 4 * 2 * 3);
            }
            other => panic!("expected Interleaved variant, got {other:?}"),
        }
    }
}
