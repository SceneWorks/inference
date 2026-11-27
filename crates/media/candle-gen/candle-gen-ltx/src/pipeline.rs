//! Pipeline glue for LTX-2.3 video generation: latent geometry, deterministic CPU-seeded noise,
//! conditioned/native joint denoise, latent token flatten/unflatten, and frames → `gen_core::Image`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::driver::{DriverRunState, SliceSpec, StepHooks, StreamId};
use candle_gen::gen_core::multistream::{
    LearnedUpsamplerComponent, ResolvedSchedule, ScheduleSource, StreamTransition,
    StreamTransitionKind,
};
use candle_gen::gen_core::{phase::ScaleContext, AudioTrack, BoundaryState, Image, Progress};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::audio_vae::AudioDecoder;
use crate::config::{
    AUDIO_LATENT_CHANNELS, AUDIO_MEL_BINS, LATENT_CHANNELS, SPATIAL_SCALE, STAGE2_SIGMAS,
    TEMPORAL_SCALE,
};
use crate::drivers::{
    CandleAvState, CandleDevAvBranches, CandleStdRngNoise, CandleVideoTokenMask,
    LtxCandleAvDevDriver, LtxCandleDistilledAvDriver, LtxCandleRfAncestralDriver,
    LtxCandleVideoDevDriver,
};
use crate::vocoder::LtxVocoder;
use crate::{
    conditioning,
    params::GuiderParams,
    transformer::{AvDiT, AvPerturbation},
};

pub(crate) fn resolve_candle_schedule(
    table_id: &str,
    sigmas: &[f32],
) -> candle_gen::Result<ResolvedSchedule> {
    ScheduleSource::ExplicitTable {
        table_id: table_id.to_owned(),
        sigmas: sigmas.to_vec(),
    }
    .resolve()
    .map_err(Into::into)
}

fn unwrap_xsigma<L>(state: BoundaryState<L>) -> L {
    match state {
        BoundaryState::XSigma(latent)
        | BoundaryState::CleanEstimate { latent, .. }
        | BoundaryState::RouteSpecific { state: latent, .. } => latent,
    }
}

pub(crate) fn stage2_video_upsample_transition(
    spatial_workspace_bytes: u64,
) -> candle_gen::Result<StreamTransition> {
    StreamTransition::new(
        "stage1_to_stage2_video_spatial_upsample",
        vec![StreamId::Video],
        StreamTransitionKind::LatentResample {
            upsampler: Some(LearnedUpsamplerComponent {
                name: gen_core_ltx::LTX_SPATIAL_UPSAMPLER_COMPONENT_NAME.into(),
                normalization: gen_core_ltx::LTX_UPSAMPLER_NORMALIZATION.into(),
                workspace_bytes: spatial_workspace_bytes.max(1),
            }),
            spatial_factor: (2, 2),
            temporal_factor: 1,
        },
    )
    .map_err(Into::into)
}

pub(crate) fn stage2_stream_renoise_transition(
    stream: StreamId,
    label: &str,
) -> candle_gen::Result<StreamTransition> {
    StreamTransition::new(
        label,
        vec![stream],
        StreamTransitionKind::Renoise {
            sigma: STAGE2_SIGMAS[0],
        },
    )
    .map_err(Into::into)
}

pub(crate) const BASE_DENOISE_SCALES: ScaleContext<'static> = ScaleContext::new();
pub(crate) const REFINEMENT_DENOISE_SCALES: ScaleContext<'static> =
    ScaleContext::new().with_pass(1);

/// Latent dims `(t_lat, h_lat, w_lat)` for `frames × height × width`: temporal `(F-1)/8 + 1`, spatial
/// `/32`.
pub fn latent_dims(frames: u32, width: u32, height: u32) -> (usize, usize, usize) {
    let t_lat = (frames as usize - 1) / TEMPORAL_SCALE + 1;
    let h_lat = height as usize / SPATIAL_SCALE;
    let w_lat = width as usize / SPATIAL_SCALE;
    (t_lat, h_lat, w_lat)
}

/// Geometry for the truthful two-stage LTX path. Stage one is exactly half of
/// the requested spatial resolution; the learned upsampler produces stage two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStageGeometry {
    pub t: usize,
    pub h1: usize,
    pub w1: usize,
    pub h2: usize,
    pub w2: usize,
}

/// Seed split consumed by the two-stage A/V route. Keeping it named rather
/// than scattering `wrapping_add`s makes the four independent streams auditable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TwoStageSeeds {
    pub video_stage1: u64,
    pub video_stage2: u64,
    pub audio_stage1: u64,
    pub audio_stage2: u64,
}

/// Production orchestration trace. The renderer calls this small stateful seam
/// at its real stage boundaries; CPU tests can assert the model-independent
/// order without loading the 22B weights.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum StageOperation {
    AdapterPass(usize),
    Stage1Noise(TwoStageSeeds),
    Keyframes { stage: u8 },
    ClipsStage1,
    LearnedUpsample,
    Stage2Renoise { sigma: f32 },
    Stage2Forward,
}

#[derive(Debug)]
pub(crate) struct TwoStageOrchestration {
    seeds: TwoStageSeeds,
    operations: Vec<StageOperation>,
    phase: StagePhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagePhase {
    Start,
    Stage1,
    Upsampled,
    Stage2,
}

impl TwoStageOrchestration {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            seeds: TwoStageSeeds {
                video_stage1: seed,
                video_stage2: seed.wrapping_add(1),
                audio_stage1: seed.wrapping_add(2),
                audio_stage2: seed.wrapping_add(3),
            },
            operations: Vec::new(),
            phase: StagePhase::Start,
        }
    }

    pub(crate) fn stage1_setup(
        &mut self,
        has_stage1_keyframes: bool,
        has_clips: bool,
    ) -> (TwoStageSeeds, ScaleContext<'static>) {
        assert_eq!(
            self.phase,
            StagePhase::Start,
            "stage one may begin only once"
        );
        self.phase = StagePhase::Stage1;
        let scales = BASE_DENOISE_SCALES;
        self.operations
            .push(StageOperation::AdapterPass(scales.pass));
        self.operations
            .push(StageOperation::Stage1Noise(self.seeds));
        if has_stage1_keyframes {
            self.operations.push(StageOperation::Keyframes { stage: 1 });
        }
        if has_clips {
            self.operations.push(StageOperation::ClipsStage1);
        }
        (self.seeds, scales)
    }

    /// Execute the learned component as the stage boundary. This is a real
    /// production wrapper, so test traces and render ordering cannot diverge.
    pub(crate) fn learned_upsample<T>(
        &mut self,
        op: impl FnOnce() -> candle_gen::Result<T>,
    ) -> candle_gen::Result<T> {
        if self.phase != StagePhase::Stage1 {
            return Err(candle_gen::CandleError::Msg(
                "ltx two-stage: learned upsample must follow stage one".into(),
            ));
        }
        let result = op()?;
        self.operations.push(StageOperation::LearnedUpsample);
        self.phase = StagePhase::Upsampled;
        Ok(result)
    }

    /// Execute fresh re-noise only after the learned component completed.
    pub(crate) fn stage2_renoise<T>(
        &mut self,
        sigma: f32,
        op: impl FnOnce() -> candle_gen::Result<T>,
    ) -> candle_gen::Result<(T, ScaleContext<'static>)> {
        if self.phase != StagePhase::Upsampled {
            return Err(candle_gen::CandleError::Msg(
                "ltx two-stage: stage-two re-noise requires learned upsample output".into(),
            ));
        }
        let result = op()?;
        let scales = REFINEMENT_DENOISE_SCALES;
        self.operations
            .push(StageOperation::Stage2Renoise { sigma });
        self.operations
            .push(StageOperation::AdapterPass(scales.pass));
        self.phase = StagePhase::Stage2;
        Ok((result, scales))
    }

    /// Apply full-resolution keyframes only after learned upsample and fresh
    /// re-noise have made the stage-two latent. This deliberately cannot run in
    /// stage-one setup.
    pub(crate) fn stage2_keyframes<T>(
        &mut self,
        op: impl FnOnce() -> candle_gen::Result<T>,
    ) -> candle_gen::Result<T> {
        if self.phase != StagePhase::Stage2 {
            return Err(candle_gen::CandleError::Msg(
                "ltx two-stage: full-resolution keyframes require stage-two re-noise".into(),
            ));
        }
        let result = op()?;
        self.operations.push(StageOperation::Keyframes { stage: 2 });
        Ok(result)
    }

    /// Execute one stage-two DiT evaluation. The wrapper protects against a
    /// refinement forward before re-noise while recording the real invocation.
    pub(crate) fn stage2_forward<T>(
        &mut self,
        op: impl FnOnce() -> candle_gen::Result<T>,
    ) -> candle_gen::Result<T> {
        if self.phase != StagePhase::Stage2 {
            return Err(candle_gen::CandleError::Msg(
                "ltx two-stage: stage-two forward requires fresh re-noise".into(),
            ));
        }
        let result = op()?;
        self.operations.push(StageOperation::Stage2Forward);
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) fn operations(&self) -> &[StageOperation] {
        &self.operations
    }
}

pub fn two_stage_geometry(frames: u32, width: u32, height: u32) -> TwoStageGeometry {
    let (t, h2, w2) = latent_dims(frames, width, height);
    TwoStageGeometry {
        t,
        h1: h2 / 2,
        w1: w2 / 2,
        h2,
        w2,
    }
}

/// Deterministic N(0,1) latent noise `[1, 128, t_lat, h_lat, w_lat]` (f32) — CPU `StdRng` (ChaCha),
/// launch-portable per seed.
pub fn create_noise(
    seed: u64,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = LATENT_CHANNELS * t_lat * h_lat * w_lat;
    candle_gen::execution::draw("ltx_video_stage_latents", seed, 0);
    candle_gen::execution::runtime(device, candle_gen::candle_core::DType::F32);
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(data, (1, LATENT_CHANNELS, t_lat, h_lat, w_lat), device)
}

/// `[B, 128, F, H, W]` → `[B, S, 128]` packed tokens (C-major over F,H,W).
pub fn flatten_latent(latent: &Tensor) -> Result<Tensor> {
    let (b, c, f, h, w) = latent.dims5()?;
    latent
        .reshape((b, c, f * h * w))?
        .transpose(1, 2)?
        .contiguous()
}

/// `[B, S, 128]` velocity → `[B, 128, F, H, W]`.
pub fn unflatten_latent(tokens: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
    let (b, _s, c) = tokens.dims3()?;
    tokens
        .transpose(1, 2)?
        .reshape((b, c, f, h, w))?
        .contiguous()
}

// --- Synchronized audio (sc-5495) ----------------------------------------------------------------

/// Deterministic N(0,1) audio latent noise `[1, 8, audio_frames, 16]` (f32) — seed offset +2 keeps it
/// distinct from the video noise stream (callers pass the explicit stream seed).
pub fn create_audio_noise(seed: u64, audio_frames: usize, device: &Device) -> Result<Tensor> {
    let ch = AUDIO_LATENT_CHANNELS as usize;
    let mel = AUDIO_MEL_BINS as usize;
    let n = ch * audio_frames * mel;
    candle_gen::execution::draw("ltx_audio_stage_latents", seed, 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(data, (1, ch, audio_frames, mel), device)
}

/// Re-noise a generated clean latent for the next distilled stage.
pub fn renoise(latent: &Tensor, fresh_noise: &Tensor, sigma: f32) -> Result<Tensor> {
    (latent * (1.0 - sigma) as f64)? + (fresh_noise * sigma as f64)?
}

/// Fold sampler-specific progress into LTX's fixed two-stage schedule. Curated
/// Heun/DPM++ drivers can report a scheduled position more than once for their
/// extra model evaluations; the product progress contract counts positions, not
/// evaluations, so each source position is emitted once.
#[derive(Debug)]
pub(crate) struct StageProgressFold {
    offset: u32,
    stage_steps: u32,
    total_steps: u32,
    last_source_position: Option<u32>,
    emitted: u32,
}

impl StageProgressFold {
    pub(crate) fn new(offset: u32, stage_steps: u32, total_steps: u32) -> Self {
        Self {
            offset,
            stage_steps,
            total_steps,
            last_source_position: None,
            emitted: 0,
        }
    }

    pub(crate) fn fold(&mut self, event: Progress) -> Option<Progress> {
        match event {
            Progress::Step { current, .. } => {
                if self.emitted == self.stage_steps
                    || self
                        .last_source_position
                        .is_some_and(|last| current <= last)
                {
                    return None;
                }
                self.last_source_position = Some(current);
                self.emitted += 1;
                Some(Progress::Step {
                    current: self.offset + self.emitted,
                    total: self.total_steps,
                })
            }
            other => Some(other),
        }
    }
}

/// Audio latent `[1, 8, T, 16]` → tokens `[1, T, 128]` (per time-frame flatten of `(ch, mel)`,
/// channel-major — matches the reference `(B,C,T,F)→(B,T,C·F)` patchify).
pub fn flatten_audio_latent(latent: &Tensor) -> Result<Tensor> {
    let (b, c, t, f) = latent.dims4()?;
    latent
        .permute((0, 2, 1, 3))?
        .reshape((b, t, c * f))?
        .contiguous()
}

/// Audio velocity tokens `[1, T, 128]` → latent `[1, 8, T, 16]`.
pub fn unflatten_audio_latent(tokens: &Tensor, t: usize) -> Result<Tensor> {
    let (b, _t, _) = tokens.dims3()?;
    let c = AUDIO_LATENT_CHANNELS as usize;
    let f = AUDIO_MEL_BINS as usize;
    tokens
        .reshape((b, t, c, f))?
        .permute((0, 2, 1, 3))?
        .contiguous()
}

/// Native distilled Euler for image/keyframe and IC-LoRA clip conditioning. Unlike the generic
/// sampler driver, this path preserves LTX's per-token timesteps and post-prediction clean-latent
/// blend. The audio stream remains uniform-sigma, matching the MLX/reference A/V implementation.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av_conditioned(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    audio: &Tensor,
    video_ctx: &Tensor,
    audio_ctx: &Tensor,
    audio_frames: usize,
    audio_grid: &Tensor,
    sigmas: &[f32],
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<(conditioning::VideoTokenState, Tensor)> {
    denoise_av_conditioned_with_scales(
        dit,
        video,
        audio,
        video_ctx,
        audio_ctx,
        audio_frames,
        audio_grid,
        sigmas,
        &BASE_DENOISE_SCALES,
        cancel,
        on_model_forward,
        on_progress,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_av_conditioned_with_scales(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    audio: &Tensor,
    video_ctx: &Tensor,
    audio_ctx: &Tensor,
    audio_frames: usize,
    audio_grid: &Tensor,
    sigmas: &[f32],
    scales: &ScaleContext<'_>,
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<(conditioning::VideoTokenState, Tensor)> {
    candle_gen::execution::schedule("ltx_conditioned_euler_v1", sigmas, "flow_velocity");
    let schedule = resolve_candle_schedule("ltx_conditioned_euler_v1", sigmas)?;
    let total_steps = schedule.num_steps();
    // The conditioned state owns its final geometry (including appended clip tokens), so prepare
    // RoPE only after that state is complete and retain it for every denoise step.
    let audio_request = flatten_audio_latent(audio)?;
    let prepared_rope =
        dit.prepare_rope(&video.latent, &audio_request, &video.positions, audio_grid)?;
    let forward_cell = std::cell::RefCell::new(on_model_forward);
    let driver = LtxCandleDistilledAvDriver::new(
        "candle_ltx_av_conditioned",
        schedule,
        CandleVideoTokenMask {
            clean_latent: &video.clean_latent,
            denoise_mask: &video.denoise_mask,
        },
        audio_frames,
        |vtok, timesteps, aflat, sigma, _kind| {
            (forward_cell.borrow_mut())()?;
            Ok(dit.forward_conditioned_prepared(
                vtok,
                aflat,
                timesteps,
                sigma as f64,
                video_ctx,
                audio_ctx,
                &video.positions,
                audio_grid,
                video.keyframes_mask.as_ref(),
                &prepared_rope,
                scales,
            )?)
        },
    );
    let mut run_state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(CandleAvState::av(video.latent.clone(), audio.clone())),
        total_steps,
        0,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, s, _| Ok(s),
    )?;
    let av = unwrap_xsigma(out);
    let mut state = video.clone();
    state.latent = av.video;
    Ok((state, av.audio.expect("audio preserved")))
}

/// Non-distilled LTX-2.5 single-video Dev denoise.  This is the actual
/// `ltxValidation.generateAudio=false` route: no audio latent is allocated, patchified, or
/// forwarded.  With no cross-modal stream the isolated-modality prediction is the conditional
/// prediction, so its configured guidance contribution is exactly zero.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_dev_conditioned(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    video_ctx: &Tensor,
    negative_video_ctx: &Tensor,
    sigmas: &[f32],
    stg_blocks: &[usize],
    video_guider: GuiderParams,
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<conditioning::VideoTokenState> {
    candle_gen::execution::schedule("ltx_dev_euler_v1", sigmas, "flow_velocity");
    candle_gen::execution::policy(
        "ltx_dev_guidance",
        serde_json::json!({"video": format!("{video_guider:?}"), "stg_blocks": stg_blocks}),
    );
    if stg_blocks.is_empty() || stg_blocks.len() > 8 || stg_blocks.iter().any(|&block| block >= 48)
    {
        return Err(candle_gen::CandleError::Msg(format!(
            "ltx_2_5: dev STG blocks must contain 1..=8 unique indices in 0..48, got {stg_blocks:?}"
        )));
    }
    let mut unique = stg_blocks.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != stg_blocks.len() {
        return Err(candle_gen::CandleError::Msg(
            "ltx_2_5: dev STG blocks must not contain duplicates".into(),
        ));
    }
    let schedule = resolve_candle_schedule("ltx_dev_euler_v1", sigmas)?;
    let total_steps = schedule.num_steps();
    let forward_cell = std::cell::RefCell::new(on_model_forward);
    let driver = LtxCandleVideoDevDriver::new(
        "candle_ltx_video_dev_conditioned",
        schedule,
        CandleVideoTokenMask {
            clean_latent: &video.clean_latent,
            denoise_mask: &video.denoise_mask,
        },
        video_guider,
        |vtok, timesteps, _sigma, _kind| {
            let forward = |context: &Tensor, perturbation: AvPerturbation| -> Result<Tensor> {
                (forward_cell.borrow_mut())()?;
                dit.forward_video_only_conditioned_controlled(
                    vtok,
                    timesteps,
                    context,
                    &video.positions,
                    video.keyframes_mask.as_ref(),
                    &perturbation,
                    &BASE_DENOISE_SCALES,
                )
            };
            let conditional = forward(video_ctx, AvPerturbation::none())?;
            let negative = forward(negative_video_ctx, AvPerturbation::none())?;
            let perturbed = forward(video_ctx, AvPerturbation::stg(stg_blocks))?;
            Ok((conditional, negative, perturbed))
        },
    );
    let mut run_state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(video.latent.clone()),
        total_steps,
        0,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, s, _| Ok(s),
    )?;
    let mut state = video.clone();
    state.latent = unwrap_xsigma(out);
    Ok(state)
}

/// Non-distilled LTX-2.5 stage-one denoise.  Each of the thirty official transitions makes four
/// real joint-AvDiT evaluations: positive, negative-text, STG (self attention skipped at block
/// 28 only), and modality-isolated (A2V/V2A skipped at every block).  Guidance combines denoised
/// predictions, not velocities; stage two deliberately remains the existing distilled refinement.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av_dev_conditioned(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    audio: &Tensor,
    video_ctx: &Tensor,
    audio_ctx: &Tensor,
    negative_video_ctx: &Tensor,
    negative_audio_ctx: &Tensor,
    audio_frames: usize,
    audio_grid: &Tensor,
    sigmas: &[f32],
    stg_blocks: &[usize],
    video_guider: GuiderParams,
    audio_guider: GuiderParams,
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<(conditioning::VideoTokenState, Tensor)> {
    denoise_av_dev_conditioned_with_scales(
        dit,
        video,
        audio,
        video_ctx,
        audio_ctx,
        negative_video_ctx,
        negative_audio_ctx,
        audio_frames,
        audio_grid,
        sigmas,
        stg_blocks,
        video_guider,
        audio_guider,
        &BASE_DENOISE_SCALES,
        cancel,
        on_model_forward,
        on_progress,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_av_dev_conditioned_with_scales(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    audio: &Tensor,
    video_ctx: &Tensor,
    audio_ctx: &Tensor,
    negative_video_ctx: &Tensor,
    negative_audio_ctx: &Tensor,
    audio_frames: usize,
    audio_grid: &Tensor,
    sigmas: &[f32],
    stg_blocks: &[usize],
    video_guider: GuiderParams,
    audio_guider: GuiderParams,
    scales: &ScaleContext<'_>,
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<(conditioning::VideoTokenState, Tensor)> {
    candle_gen::execution::schedule("ltx_dev_euler_v1", sigmas, "flow_velocity");
    candle_gen::execution::policy(
        "ltx_dev_guidance",
        serde_json::json!({"video": format!("{video_guider:?}"), "stg_blocks": stg_blocks, "audio": format!("{audio_guider:?}")}),
    );
    if stg_blocks.is_empty() || stg_blocks.len() > 8 || stg_blocks.iter().any(|&block| block >= 48)
    {
        return Err(candle_gen::CandleError::Msg(format!(
            "ltx_2_5: dev STG blocks must contain 1..=8 unique indices in 0..48, got {stg_blocks:?}"
        )));
    }
    let schedule = resolve_candle_schedule("ltx_dev_euler_v1", sigmas)?;
    let total_steps = schedule.num_steps();
    let audio_request = flatten_audio_latent(audio)?;
    let prepared_rope =
        dit.prepare_rope(&video.latent, &audio_request, &video.positions, audio_grid)?;
    let forward_cell = std::cell::RefCell::new(on_model_forward);
    let driver = LtxCandleAvDevDriver::new(
        "candle_ltx_av_dev_conditioned",
        schedule,
        CandleVideoTokenMask {
            clean_latent: &video.clean_latent,
            denoise_mask: &video.denoise_mask,
        },
        audio_frames,
        video_guider,
        audio_guider,
        |vtok, timesteps, aflat, sigma, _kind| {
            let forward = |video_context: &Tensor,
                           audio_context: &Tensor,
                           perturbation: AvPerturbation|
             -> Result<(Tensor, Tensor)> {
                (forward_cell.borrow_mut())()?;
                dit.forward_conditioned_prepared_controlled(
                    vtok,
                    aflat,
                    timesteps,
                    sigma as f64,
                    video_context,
                    audio_context,
                    &video.positions,
                    audio_grid,
                    video.keyframes_mask.as_ref(),
                    &prepared_rope,
                    &perturbation,
                    scales,
                )
            };
            let conditional = forward(video_ctx, audio_ctx, AvPerturbation::none())?;
            let negative = forward(
                negative_video_ctx,
                negative_audio_ctx,
                AvPerturbation::none(),
            )?;
            let perturbed = forward(video_ctx, audio_ctx, AvPerturbation::stg(stg_blocks))?;
            let isolated = forward(video_ctx, audio_ctx, AvPerturbation::modality_isolated())?;
            Ok(CandleDevAvBranches {
                conditional,
                negative,
                perturbed,
                isolated,
            })
        },
    );
    let mut run_state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(CandleAvState::av(video.latent.clone(), audio.clone())),
        total_steps,
        0,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, s, _| Ok(s),
    )?;
    let av = unwrap_xsigma(out);
    let mut state = video.clone();
    state.latent = av.video;
    Ok((state, av.audio.expect("audio preserved")))
}

/// The audio side of an ancestral token denoise, when the audio modality is present.
/// Deliberately mask-free — see the mlx twin's note: the reference's post-noise audio re-pin is
/// the identity because audio always carries an all-ones denoise mask on this engine's surface.
pub struct AncestralAudio<'a> {
    /// `(B, 8, T, 16)` audio latent grid.
    pub latent: &'a Tensor,
    pub ctx: &'a Tensor,
    pub grid: &'a Tensor,
    pub audio_frames: usize,
}

/// Token-native **rectified-flow ancestral** denoise (sc-18789) — the twin of
/// `mlx-gen-ltx::pipeline::denoise_tokens_rf_ancestral`; see that doc comment for the step
/// semantics (mask-corrected x0, terminal short-circuit, RF-ancestral update with the
/// post-noise conditioning re-pin, video-only forward when `audio` is `None`, and
/// per-(seed, step, modality) noise keys).
#[allow(clippy::too_many_arguments)]
pub fn denoise_tokens_rf_ancestral(
    dit: &AvDiT,
    video: &conditioning::VideoTokenState,
    video_ctx: &Tensor,
    audio: Option<AncestralAudio<'_>>,
    sigmas: &[f32],
    eta: f32,
    s_noise: f32,
    noise_seed: u64,
    cancel: &candle_gen::gen_core::CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<(conditioning::VideoTokenState, Option<Tensor>)> {
    candle_gen::execution::schedule("ltx_dfr_rf_ancestral_v1", sigmas, "flow_velocity");
    candle_gen::execution::policy(
        "ltx_rf_ancestral",
        serde_json::json!({"eta_bits": eta.to_bits(), "noise_bits": s_noise.to_bits()}),
    );
    let schedule = resolve_candle_schedule("ltx_dfr_rf_ancestral_v1", sigmas)?;
    let total_steps = schedule.num_steps();
    let audio_frames = audio.as_ref().map(|a| a.audio_frames);
    let forward_cell = std::cell::RefCell::new(on_model_forward);
    let driver = LtxCandleRfAncestralDriver::new(
        "candle_ltx_denoise_tokens_rf_ancestral",
        schedule,
        CandleVideoTokenMask {
            clean_latent: &video.clean_latent,
            denoise_mask: &video.denoise_mask,
        },
        audio_frames,
        eta,
        s_noise,
        |vtok, timesteps, aflat, sigma, _kind| {
            (forward_cell.borrow_mut())()?;
            match (aflat, &audio) {
                (Some(aflat), Some(a)) => {
                    // Temporal tiles use the base pass, independently of the detailing pass.
                    let (vv, av) = dit.forward_conditioned_with_scales(
                        vtok,
                        aflat,
                        timesteps,
                        sigma as f64,
                        video_ctx,
                        a.ctx,
                        &video.positions,
                        a.grid,
                        video.keyframes_mask.as_ref(),
                        &BASE_DENOISE_SCALES,
                    )?;
                    Ok((vv, Some(av)))
                }
                _ => Ok((
                    dit.forward_video_only_conditioned_controlled(
                        vtok,
                        timesteps,
                        video_ctx,
                        &video.positions,
                        video.keyframes_mask.as_ref(),
                        &AvPerturbation::none(),
                        &BASE_DENOISE_SCALES,
                    )?,
                    None,
                )),
            }
        },
        CandleStdRngNoise { noise_seed },
    );
    let initial = match &audio {
        Some(a) => CandleAvState::av(video.latent.clone(), a.latent.clone()),
        None => CandleAvState::video_only(video.latent.clone()),
    };
    let mut run_state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(initial),
        total_steps,
        noise_seed,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, s, _| Ok(s),
    )?;
    let av = unwrap_xsigma(out);
    let mut state = video.clone();
    state.latent = av.video;
    Ok((state, av.audio))
}

/// Decode audio latents → an interleaved-PCM [`AudioTrack`]: `AudioDecoder` → mel `(1,2,T',64)` →
/// `LtxVocoder` → waveform `(1,2,samples)` → interleaved stereo `f32`.
pub fn decode_audio_track(
    decoder: &AudioDecoder,
    vocoder: &LtxVocoder,
    audio_latents: &Tensor,
    sample_rate: u32,
) -> Result<AudioTrack> {
    let mel = decoder.decode(audio_latents)?;
    let wav = vocoder.forward(&mel)?; // (1, channels, samples)
    let (_b, channels, samples) = wav.dims3()?;
    // (1, C, S) → (S, C) → interleaved.
    let interleaved = wav
        .reshape((channels, samples))?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    Ok(AudioTrack {
        samples: interleaved.flatten_all()?.to_vec1::<f32>()?,
        sample_rate,
        channels: channels as u16,
        ..Default::default()
    })
}

/// Decoded video `[1, 3, T, H, W]` in `[-1, 1]` → one RGB8 [`Image`] per frame.
pub fn frames_to_images(decoded: &Tensor) -> Result<Vec<Image>> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let u8s = candle_gen::round_rgb8(&scaled)?.to_device(&Device::Cpu)?;
    let (_b, c, t, h, w) = u8s.dims5()?;
    let frames = u8s.squeeze(0)?; // [3,T,H,W]
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let frame = frames.narrow(1, ti, 1)?.squeeze(1)?; // [3,H,W]
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        debug_assert_eq!(c, 3);
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_stage_geometry_requires_a_64_pixel_final_grid() {
        let g = two_stage_geometry(49, 704, 512);
        assert_eq!((g.t, g.h1, g.w1, g.h2, g.w2), (7, 8, 11, 16, 22));
    }

    #[test]
    fn stage_streams_have_distinct_seed_offsets() -> Result<()> {
        let device = Device::Cpu;
        let video1 = create_noise(9, 1, 1, 1, &device)?;
        let video2 = create_noise(10, 1, 1, 1, &device)?;
        let audio1 = create_audio_noise(11, 1, &device)?;
        let audio2 = create_audio_noise(12, 1, &device)?;
        assert_ne!(
            video1.flatten_all()?.to_vec1::<f32>()?,
            video2.flatten_all()?.to_vec1::<f32>()?
        );
        assert_ne!(
            audio1.flatten_all()?.to_vec1::<f32>()?,
            audio2.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn renoise_uses_the_fresh_stage_noise_at_sigma_one() -> Result<()> {
        let d = Device::Cpu;
        let latent = Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &d)?;
        let noise = Tensor::ones((1, 1, 1, 1, 1), DType::F32, &d)?;
        assert_eq!(
            renoise(&latent, &noise, 1.0)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1.0]
        );
        Ok(())
    }

    #[test]
    fn progress_fold_deduplicates_curated_positions() {
        let mut fold = StageProgressFold::new(0, 8, 11);
        let events = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]
            .into_iter()
            .filter_map(|current| {
                fold.fold(Progress::Step { current, total: 8 })
                    .and_then(|event| match event {
                        Progress::Step { current, .. } => Some(current),
                        _ => None,
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(events, (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn progress_fold_preserves_native_stage_positions_and_global_total() {
        let mut fold = StageProgressFold::new(8, 3, 11);
        let events = (1..=3)
            .map(|current| fold.fold(Progress::Step { current, total: 3 }).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            vec![
                Progress::Step {
                    current: 9,
                    total: 11
                },
                Progress::Step {
                    current: 10,
                    total: 11
                },
                Progress::Step {
                    current: 11,
                    total: 11
                },
            ]
        );
    }

    #[test]
    fn production_two_stage_orchestration_cannot_skip_the_learned_refinement_path() {
        let mut flow = TwoStageOrchestration::new(41);
        let (seeds, stage1_scales) = flow.stage1_setup(true, true);
        assert_eq!(
            stage1_scales.pass, 0,
            "stage one selected the real adapter pass"
        );
        assert_eq!(
            seeds,
            TwoStageSeeds {
                video_stage1: 41,
                video_stage2: 42,
                audio_stage1: 43,
                audio_stage2: 44,
            }
        );
        let upsample_calls = std::cell::Cell::new(0);
        flow.learned_upsample(|| {
            upsample_calls.set(upsample_calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(upsample_calls.get(), 1, "the learned component was invoked");
        let device = Device::Cpu;
        let latent = Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &device).unwrap();
        let fresh_noise = Tensor::ones((1, 1, 1, 1, 1), DType::F32, &device).unwrap();
        let (stage2, stage2_scales) = flow
            .stage2_renoise(0.909375, || Ok(renoise(&latent, &fresh_noise, 0.909375)?))
            .unwrap();
        assert_eq!(
            stage2_scales.pass, 1,
            "stage two selected the real adapter pass"
        );
        assert_eq!(
            stage1_scales.pass, 0,
            "refinement leaves base arguments unchanged"
        );
        assert!(
            stage2.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0] > 0.9,
            "the tested callback must execute renoise at the stage-two sigma"
        );
        let keyframe = Tensor::ones((1, 1, 1, 1, 1), DType::F32, &device).unwrap();
        let keys = [conditioning::Keyframe {
            latent: &keyframe,
            frame_idx: 0,
            strength: 1.0,
        }];
        let keyed = flow
            .stage2_keyframes(|| Ok(conditioning::apply_keyframes(&stage2, &keys)?))
            .unwrap();
        assert_eq!(
            keyed
                .latent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0],
            "the post-upsample stage-two callback applies the full-resolution keyframe"
        );
        let stage2_forwards = std::cell::Cell::new(0);
        for _ in 0..3 {
            flow.stage2_forward(|| {
                stage2_forwards.set(stage2_forwards.get() + 1);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(stage2_forwards.get(), 3);
        let mut progress = StageProgressFold::new(0, 8, 11);
        let mut observed_progress = (1..=8)
            .filter_map(|current| progress.fold(Progress::Step { current, total: 8 }))
            .collect::<Vec<_>>();
        let mut stage2_progress = StageProgressFold::new(8, 3, 11);
        observed_progress.extend(
            [1, 2, 2, 3]
                .into_iter()
                .filter_map(|current| stage2_progress.fold(Progress::Step { current, total: 3 })),
        );
        assert_eq!(
            observed_progress,
            (1..=11)
                .map(|current| Progress::Step { current, total: 11 })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            flow.operations(),
            [
                StageOperation::AdapterPass(0),
                StageOperation::Stage1Noise(seeds),
                StageOperation::Keyframes { stage: 1 },
                StageOperation::ClipsStage1,
                StageOperation::LearnedUpsample,
                StageOperation::Stage2Renoise { sigma: 0.909375 },
                StageOperation::AdapterPass(1),
                StageOperation::Keyframes { stage: 2 },
                StageOperation::Stage2Forward,
                StageOperation::Stage2Forward,
                StageOperation::Stage2Forward,
            ]
        );
    }
}

#[cfg(test)]
mod execution_identity_tests {
    use super::*;
    fn output() -> candle_gen::gen_core::GenerationOutput {
        candle_gen::gen_core::GenerationOutput::Video {
            frames: vec![candle_gen::gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            }],
            fps: 24,
            audio: None,
        }
    }

    #[test]
    fn reports_all_stage_and_modality_draws_without_collapsing_equal_shapes() {
        let attempt = candle_gen::execution::Attempt::begin();
        let device = Device::Cpu;
        let _ = create_noise(40, 1, 1, 1, &device).unwrap();
        let _ = create_audio_noise(42, 1, &device).unwrap();
        let _ = create_noise(41, 1, 1, 1, &device).unwrap();
        let _ = create_audio_noise(43, 1, &device).unwrap();
        candle_gen::execution::schedule("stage_one_fixture", &[1.0, 0.0], "flow");
        candle_gen::execution::schedule("stage_two_fixture", &[0.5, 0.0], "flow");
        candle_gen::execution::policy("fixture", serde_json::json!("no_model"));
        let identity = attempt.finish_image(&crate::descriptor(), &output());
        let resolved = identity.resolved().unwrap();
        let candle_gen::gen_core::generator::ExecutionRandomness::Draws { draws, .. } =
            &resolved.randomness
        else {
            panic!("draws")
        };
        assert_eq!(
            draws.iter().map(|d| d.seed.unwrap()).collect::<Vec<_>>(),
            [40, 42, 41, 43]
        );
        assert_eq!(draws[0].purpose, "ltx_video_stage_latents");
        assert_eq!(draws[1].purpose, "ltx_audio_stage_latents");
        let candle_gen::gen_core::generator::ExecutionFact::Resolved { value } = &resolved.schedule
        else {
            panic!("schedule")
        };
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_ne!(
            value[0]["schedule"]["sha256"],
            value[1]["schedule"]["sha256"]
        );
    }
}
