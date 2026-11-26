//! Krea Realtime 14B **autoregressive chunk driver** (sc-8437 S4; clean-context recompute sc-8438 S5).
//!
//! The AR loop that turns the S3 causal forward + persistent KV cache
//! ([`CausalKreaTransformer`]) into a latent video sequence, **adapted from**
//! the reference `causal_inference.py:177-245` (`krea-ai/realtime-video`). For each of `ceil(num_frames / num_frames_per_block)`
//! chunks (a chunk = `num_frames_per_block` latent frames) it runs the Self-Forcing few-step denoise
//! ([`FewStepSchedule`]) — one batch-1 forward per step, **CFG off** — threading the persistent cache
//! and advancing the global token offset so chunk *k* attends chunk *k−1*'s committed context.
//!
//! **Cache commit — exactly one per chunk.** With [`KreaArConfig::do_kv_recomp`](crate::KreaArConfig::do_kv_recomp)
//! **off** (the S4 baseline / A-B) the final near-clean denoise step runs
//! [`CausalKreaTransformer::forward_chunk`] (commits), every earlier step
//! [`CausalKreaTransformer::forward_chunk_readonly`] (reads, appends nothing). With it **on** (the
//! shipped default, **S5**) *every* denoise step is read-only and one extra forward reruns on the
//! chunk's clean `x0` at [`KreaArConfig::context_noise`](crate::KreaArConfig::context_noise) and commits
//! *that* clean-context K/V — the reference's `causal_inference.py:227-236` "rerun with timestep zero to
//! update KV cache using clean context". The output is the **latent sequence**
//! `[out_dim, num_frames, H, W]` (f32) — UMT5 text encoding and VAE decode / clip assembly (incl. the
//! first-frame VAE re-anchor) are wired at the pipeline level in **S6**; the i2v/v2v conditioning
//! surface ([`generate_i2v_latents`] / [`generate_v2v_latents`] over [`RefConditioning`]) is added in
//! **S7** (below); long-clip coherence with real weights is S13.
//! `context` (the UMT5 text embedding) is taken as an input parameter; the DiT-side text embedding +
//! cross-attention K/V are built here once per prompt.

use std::cell::RefCell;

use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec,
    SliceTransition, StepHooks,
};
use mlx_gen::gen_core::{BoundaryState, Error as GenError, Result as GenResult};
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array};

use crate::causal::{CausalKreaTransformer, CausalKvCache};
use crate::config::KreaRealtimeConfig;
use crate::scheduler::{euler_x0, renoise_step, FewStepSchedule};

/// A text-to-video autoregressive generation request (t2v only — latents start from seeded noise).
#[derive(Clone, Debug)]
pub struct ArGenParams {
    /// Deterministic seed: fixes the per-clip init noise and the per-step renoise noise.
    pub seed: u64,
    /// Denoising-steps override — `None` uses the config list's length; `Some(n)` uses `n`. Product
    /// generation follows the pinned release server and derives a strength-1 float schedule from that
    /// count rather than executing the config's integer values.
    pub steps: Option<usize>,
    /// Number of **latent** frames to generate (the caller derives this from the requested duration ×
    /// [`fps`](Self::fps) via the VAE's temporal compression, which S6 owns).
    pub num_latent_frames: usize,
    /// Latent spatial height. Must satisfy `(latent_height / patch_h) · (latent_width / patch_w) =
    /// frame_seq_length` (the model's canonical per-frame token count) — see [`generate_latents`].
    pub latent_height: usize,
    /// Latent spatial width. See [`latent_height`](Self::latent_height).
    pub latent_width: usize,
    /// Output frames-per-second — carried onto the assembled clip at the pipeline level (S6); it does
    /// not affect the latent sequence produced here.
    pub fps: u32,
    /// Request-scoped shared-ladder levers. Carried here so the denoise and decode boundaries can
    /// honour an authorized calibration fault (sc-22738); the default arms nothing.
    pub memory: mlx_gen::gen_core::GenerationMemory,
}

fn t2v_schedule(
    cfg: &KreaRealtimeConfig,
    steps_override: Option<usize>,
) -> Result<FewStepSchedule> {
    let steps = steps_override.unwrap_or(cfg.ar.denoising_step_list.len());
    FewStepSchedule::for_strength(cfg.ar.timestep_shift as f64, 1.0, steps)
}

/// Per-chunk latent-frame counts for `num_frames` at `frames_per_block`: full blocks then a
/// (possibly partial) trailing block. Length `ceil(num_frames / frames_per_block)`.
pub(crate) fn chunk_frame_counts(num_frames: usize, frames_per_block: usize) -> Vec<usize> {
    if frames_per_block == 0 || num_frames == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut remaining = num_frames;
    while remaining > 0 {
        let take = remaining.min(frames_per_block);
        out.push(take);
        remaining -= take;
    }
    out
}

/// Run the full t2v AR generation, owning a fresh KV cache. See [`generate_latents_into`].
///
/// `context` is the UMT5 text embedding `[text_len, text_dim]` (f32) — this builds the DiT text
/// embedding + per-prompt cross-attention K/V once, then drives the chunk loop. Returns the latent
/// sequence `[out_dim, num_latent_frames, latent_height, latent_width]` (f32).
///
/// `cancel` is polled per autoregressive step inside the loop and `on_progress` streams a
/// [`Progress::Step`] per denoise step (sc-8441 S8) — a mid-clip cancel bails within ~one step with
/// the typed [`Error::Canceled`], rather than only at stage boundaries.
///
/// Errors if the latent geometry does not yield exactly `frame_seq_length` tokens per frame (the S3
/// causal forward bakes `frame_seq_length` into its cache windowing and RoPE frame offset, so the
/// caller must size latents to the model's canonical per-frame token count).
pub fn generate_latents(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut cache = transformer.new_cache();
    generate_latents_into(
        transformer,
        cfg,
        context,
        params,
        &mut cache,
        cancel,
        on_progress,
    )
}

/// [`generate_latents`] against a caller-owned KV cache — the seam S5 (clean-context recompute) and S6
/// (the `Generator`) drive, and the seam tests inspect to assert the cache grows by exactly one chunk
/// per chunk (KV threading). The `cache` must be empty on entry; it holds the full clip's committed
/// self-attention K/V on return (`stored_tokens() == num_latent_frames · frame_seq_length`).
pub fn generate_latents_into(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    cache: &mut CausalKvCache,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let (ph, pw) = (cfg.wan.patch_size.1, cfg.wan.patch_size.2);
    let frame_seq_length = cfg.ar.frame_seq_length;
    if ph == 0
        || pw == 0
        || !params.latent_height.is_multiple_of(ph)
        || !params.latent_width.is_multiple_of(pw)
    {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} is not divisible by the patch size {ph}x{pw}",
            params.latent_height, params.latent_width
        )));
    }
    let per_frame_tokens = (params.latent_height / ph) * (params.latent_width / pw);
    if per_frame_tokens != frame_seq_length {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} yields {per_frame_tokens} tokens/frame but the model's \
             frame_seq_length is {frame_seq_length}; size the latents to the canonical per-frame \
             token count",
            params.latent_height, params.latent_width
        )));
    }
    if !cache.is_empty() {
        return Err(Error::Msg(
            "krea AR: generate_latents_into requires an empty KV cache".into(),
        ));
    }

    // The online release server ignores the YAML's integer values, uses only their count, and selects
    // float timesteps from the shifted table.
    let schedule = t2v_schedule(cfg, params.steps)?;

    // DiT text embedding + per-prompt cross-attention K/V (position-independent; built once).
    let ctx = transformer.inner().embed_text(context)?;
    let cross_kv = transformer.prepare_cross_kv(&ctx)?;

    // The per-chunk forward: `commit = true` appends this forward's self-attention K/V to the
    // persistent cache; `commit = false` reads the committed history and appends nothing. The AR loop
    // decides which forward commits (the S4 final denoise step, or the S5 clean-context recompute).
    // Exactly one forward per call (CFG off — no uncond branch).
    let denoise = |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
        if commit {
            transformer.forward_chunk(chunk, t, &cross_kv, start, cache)
        } else {
            transformer.forward_chunk_readonly(chunk, t, &cross_kv, start, cache)
        }
    };

    run_ar_loop(
        &schedule,
        cfg.wan.in_dim,
        frame_seq_length,
        cfg.ar.num_frames_per_block,
        params,
        cfg.ar.do_kv_recomp,
        cfg.ar.context_noise,
        cancel,
        on_progress,
        denoise,
    )
}

/// Reference conditioning for autoregressive **i2v / v2v** generation (sc-8440 S7), mirroring the two
/// reference mechanisms in `krea-ai/realtime-video`:
///
///   * **`context_latents`** — clean VAE-encoded reference/first-frame latents `[C, F_ctx, H, W]` (f32)
///     that **warm the KV cache** before generation and are **prepended** verbatim to the output. Each
///     context block is committed by one forward at the clean timestep `0` (the reference's
///     `timestep * 0`, `causal_inference.py:147,165` — the S5 clean-context commit applied to the
///     reference), and the latents are copied into the leading output frames
///     (`output[:, :num_input] = initial_latent`). Used for i2v (a single still, `F_ctx = 1`) and video
///     extension (`F_ctx > 1`). `None` ⇒ no warm (t2v / pure v2v restyle).
///   * **`source`** — a v2v `(source_latents [C, num_latent_frames, H, W], strength)`. Each generated
///     block starts from the VAE-encoded source clip **renoised to the strength level**, and the denoise
///     schedule starts at `strength·1000` ([`FewStepSchedule::for_strength`] /
///     `v2v.py::get_denoising_schedule` + `release_server.py:426,658`), so a lower `strength` preserves
///     more of the source. `None` ⇒ pure-noise init on the config's few-step schedule (t2v / i2v).
pub struct RefConditioning {
    /// Clean VAE-encoded context latents that warm the cache + prepend to the output. See the type doc.
    pub context_latents: Option<Array>,
    /// v2v source clip + denoise strength. See the type doc.
    pub source: Option<(Array, f32)>,
}

/// Autoregressive **i2v**: warm the KV cache from a clean VAE-encoded reference still (or first-frame
/// clip) and generate `params.num_latent_frames` frames conditioned on it. Returns
/// `[out_dim, F_ref + num_latent_frames, H, W]` (f32) — the reference frames prepended verbatim to the
/// generated continuation, mirroring the reference `causal_inference.py` (`initial_latent`,
/// `num_input_frames == 1` ⇒ image-to-video). `reference_latents` is `[in_dim, F_ref, latent_height,
/// latent_width]` (the z16 Wan VAE `encode` of the still); the pipeline's
/// [`generate_i2v_from_components`](crate::generate_i2v_from_components) owns the VAE encode + decode.
pub fn generate_i2v_latents(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    reference_latents: &Array,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut cache = transformer.new_cache();
    let cond = RefConditioning {
        context_latents: Some(reference_latents.clone()),
        source: None,
    };
    generate_latents_conditioned_into(
        transformer,
        cfg,
        context,
        params,
        &cond,
        &mut cache,
        cancel,
        on_progress,
    )
}

/// Autoregressive **v2v**: generate `params.num_latent_frames` frames conditioned on a clean
/// VAE-encoded source clip, honoring a denoise `strength` (`0 ..= 1`). Each block starts from the source
/// renoised to the strength schedule's max sigma and denoises down the strength-scaled schedule (a lower
/// `strength` preserves more of the source); the AR loop's per-block clean-context recompute threads the
/// rolling generated frames as context (`release_server.py` + `v2v.py`). Returns
/// `[out_dim, num_latent_frames, H, W]` (f32). `source_latents` is `[in_dim, num_latent_frames,
/// latent_height, latent_width]` (the z16 Wan VAE `encode_sample` of the source, one frame per generated
/// frame); the pipeline's [`generate_v2v_from_components`](crate::generate_v2v_from_components) owns the
/// VAE encode + decode.
#[allow(clippy::too_many_arguments)]
pub fn generate_v2v_latents(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    source_latents: &Array,
    strength: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut cache = transformer.new_cache();
    let cond = RefConditioning {
        context_latents: None,
        source: Some((source_latents.clone(), strength)),
    };
    generate_latents_conditioned_into(
        transformer,
        cfg,
        context,
        params,
        &cond,
        &mut cache,
        cancel,
        on_progress,
    )
}

/// The shared i2v/v2v conditioned AR generation against a caller-owned KV cache (the seam the S7
/// verification inspects to assert the reference frames' clean-context K/V populate the cache before
/// generation). Validates the latent geometry, builds the schedule (strength-scaled for v2v), warms the
/// cache from `cond.context_latents` (committing each context block at timestep `0`), then runs the AR
/// loop for `params.num_latent_frames` frames — pure-noise init (i2v) or source-renoised init (v2v) —
/// and prepends the context frames to the output. The `cache` must be empty on entry.
#[allow(clippy::too_many_arguments)]
pub fn generate_latents_conditioned_into(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    cond: &RefConditioning,
    cache: &mut CausalKvCache,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let (ph, pw) = (cfg.wan.patch_size.1, cfg.wan.patch_size.2);
    let frame_seq_length = cfg.ar.frame_seq_length;
    if ph == 0
        || pw == 0
        || !params.latent_height.is_multiple_of(ph)
        || !params.latent_width.is_multiple_of(pw)
    {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} is not divisible by the patch size {ph}x{pw}",
            params.latent_height, params.latent_width
        )));
    }
    let per_frame_tokens = (params.latent_height / ph) * (params.latent_width / pw);
    if per_frame_tokens != frame_seq_length {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} yields {per_frame_tokens} tokens/frame but the model's \
             frame_seq_length is {frame_seq_length}; size the latents to the canonical per-frame \
             token count",
            params.latent_height, params.latent_width
        )));
    }
    if !cache.is_empty() {
        return Err(Error::Msg(
            "krea AR: generate_latents_conditioned_into requires an empty KV cache".into(),
        ));
    }

    // v2v uses the strength-scaled release schedule (max timetable index = strength·1000); i2v/t2v
    // use its strength-1 form.
    let schedule = match &cond.source {
        Some((_, strength)) => {
            let steps = params.steps.unwrap_or(cfg.ar.denoising_step_list.len());
            FewStepSchedule::for_strength(cfg.ar.timestep_shift as f64, *strength as f64, steps)?
        }
        None => t2v_schedule(cfg, params.steps)?,
    };

    // DiT text embedding + per-prompt cross-attention K/V (position-independent; built once).
    let ctx = transformer.inner().embed_text(context)?;
    let cross_kv = transformer.prepare_cross_kv(&ctx)?;

    // Step 2: warm the KV cache from the clean context latents (i2v / first-frame / extension). Each
    // context block is committed by ONE forward at the clean timestep 0 (the reference `timestep * 0`,
    // `causal_inference.py:147,165` — the S5 clean-context commit applied to the reference latents), so
    // the first generated chunk attends the warmed reference. `start_token` advances past the warmed
    // context so the generated chunks' causal RoPE offset + read window line up
    // (`current_start_frame · frame_seq_length`). The context latents are prepended to the output.
    let c = cfg.wan.in_dim as i32;
    let (h, w) = (params.latent_height as i32, params.latent_width as i32);
    let mut start_token = 0usize;
    let mut prefix: Option<Array> = None;
    if let Some(ctx_lat) = &cond.context_latents {
        let s = ctx_lat.shape();
        if s.len() != 4 || s[0] != c || s[1] < 1 || s[2] != h || s[3] != w {
            return Err(Error::Msg(format!(
                "krea i2v: context latents shape {s:?} must be [{c}, F_ctx>=1, {h}, {w}]"
            )));
        }
        let f_ctx = s[1] as usize;
        let mut cursor = 0usize;
        for chunk_frames in chunk_frame_counts(f_ctx, cfg.ar.num_frames_per_block) {
            // Cancellation checkpoint while warming the KV cache from clean context (bail before the
            // next reference-block commit). The bulk cancel target is the per-step AR loop below.
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let idx: Vec<i32> = (cursor as i32..(cursor + chunk_frames) as i32).collect();
            let chunk = ctx_lat.take_axis(Array::from_slice(&idx, &[idx.len() as i32]), 1)?;
            // Commit (not read-only): populate the clean-context K/V for this reference block.
            let _ = transformer.forward_chunk(&chunk, 0.0, &cross_kv, start_token, cache)?;
            start_token += chunk_frames * frame_seq_length;
            cursor += chunk_frames;
        }
        prefix = Some(ctx_lat.clone());
    }

    // v2v init noise level = the strength schedule's max-timestep sigma (source·(1−σ) + ε·σ).
    let init_source: Option<(&Array, f64)> = cond.source.as_ref().map(|(src, _)| {
        let sigma_init = schedule
            .step_timesteps()
            .first()
            .map(|&t| schedule.sigma_at_timestep(t))
            .unwrap_or(1.0);
        (src, sigma_init)
    });

    // The per-chunk denoise forward, threading the (possibly warmed) persistent cache. Exactly one
    // forward per call (CFG off).
    let denoise = |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
        if commit {
            transformer.forward_chunk(chunk, t, &cross_kv, start, cache)
        } else {
            transformer.forward_chunk_readonly(chunk, t, &cross_kv, start, cache)
        }
    };

    let generated = run_ar_loop_conditioned(
        &schedule,
        cfg.wan.in_dim,
        frame_seq_length,
        cfg.ar.num_frames_per_block,
        params,
        cfg.ar.do_kv_recomp,
        cfg.ar.context_noise,
        start_token,
        init_source,
        cancel,
        on_progress,
        denoise,
    )?;

    // Prepend the clean context frames (copied verbatim) to the generated continuation.
    match prefix {
        Some(ctx_lat) => Ok(concatenate_axis(&[&ctx_lat, &generated], 1)?),
        None => Ok(generated),
    }
}

/// The AR chunk-loop core (t2v), a thin wrapper over [`run_ar_loop_conditioned`] with no cache-warm
/// offset and pure-noise init — the S4/S6 t2v path. See [`run_ar_loop_conditioned`].
#[allow(clippy::too_many_arguments)]
fn run_ar_loop(
    schedule: &FewStepSchedule,
    channels: usize,
    frame_seq_length: usize,
    frames_per_block: usize,
    params: &ArGenParams,
    do_kv_recomp: bool,
    context_noise: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    denoise: impl FnMut(&Array, f32, usize, bool) -> Result<Array>,
) -> Result<Array> {
    run_ar_loop_conditioned(
        schedule,
        channels,
        frame_seq_length,
        frames_per_block,
        params,
        do_kv_recomp,
        context_noise,
        0,
        None,
        cancel,
        on_progress,
        denoise,
    )
}

/// Per-block forward evaluation seam for [`KreaRealtimeArDriver`], generic so weights-free Tier 0
/// conformance tests can mock the causal transformer forward without loading weights (`PIPE-7`).
pub trait KreaArBlockForward {
    /// Evaluate one autoregressive chunk at timestep `t` and global token offset `start_token`.
    /// When `commit` is `true`, the implementation commits this block's self-attention K/V into the
    /// persistent [`CausalKvCache`].
    fn forward_block(
        &mut self,
        chunk: &Array,
        timestep: f32,
        start_token: usize,
        commit: bool,
    ) -> Result<Array>;
}

impl<F> KreaArBlockForward for F
where
    F: FnMut(&Array, f32, usize, bool) -> Result<Array>,
{
    fn forward_block(
        &mut self,
        chunk: &Array,
        timestep: f32,
        start_token: usize,
        commit: bool,
    ) -> Result<Array> {
        self(chunk, timestep, start_token, commit)
    }
}

/// Owner-local state carried in [`DriverRunState`] across [`KreaRealtimeArDriver`] slices.
#[derive(Clone)]
struct KreaArDriverState {
    seed: u64,
    total_steps: usize,
    n_steps: usize,
    chunk_counts: Vec<usize>,
    key: Array,
    full_noise: Array,
    active_chunk: Option<Array>,
    outputs: Vec<Array>,
    frame_cursor: usize,
    start_token: usize,
    expected_next_step: usize,
}

/// Typed [`FamilyDriver<Array>`] executing Krea Realtime's autoregressive Self-Forcing chunk loop
/// (`PIPE-7`, `PIPE-14`, `SW-6976`).
pub struct KreaRealtimeArDriver<F> {
    support: DriverSupport,
    schedule: FewStepSchedule,
    channels: usize,
    frame_seq_length: usize,
    frames_per_block: usize,
    num_latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    do_kv_recomp: bool,
    context_noise: f32,
    start_token_offset: usize,
    init_source: Option<(Array, f64)>,
    forward: RefCell<F>,
}

impl<F: KreaArBlockForward> KreaRealtimeArDriver<F> {
    /// Construct a typed Krea Realtime AR driver over `forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        schedule: FewStepSchedule,
        channels: usize,
        frame_seq_length: usize,
        frames_per_block: usize,
        num_latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        do_kv_recomp: bool,
        context_noise: f32,
        start_token_offset: usize,
        init_source: Option<(Array, f64)>,
        forward: F,
    ) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "krea_realtime_ar",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: false,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: false,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    reset_history:
                        "Krea Realtime AR causal KV trajectory cannot reset block/cache state mid-trajectory",
                    renoise_from_clean:
                        "Krea Realtime AR driver owns its per-step Self-Forcing renoise chain and refuses external Renoise transitions",
                    geometry_change:
                        "Krea Realtime AR driver does not support mid-trajectory spatial resampling",
                    read_only_boundary_estimate:
                        "Krea Realtime AR chunk driver does not expose a separate read-only boundary clean estimate",
                    ..DriverRefusalReasons::default()
                },
            },
            schedule,
            channels,
            frame_seq_length,
            frames_per_block,
            num_latent_frames,
            latent_height,
            latent_width,
            do_kv_recomp,
            context_noise,
            start_token_offset,
            init_source,
            forward: RefCell::new(forward),
        }
    }

    fn validate_source_shape(&self) -> Result<()> {
        if let Some((source, _)) = &self.init_source {
            let c = self.channels as i32;
            let num_frames = self.num_latent_frames;
            let h = self.latent_height as i32;
            let w = self.latent_width as i32;
            let s = source.shape();
            if s.len() != 4 || s[0] != c || s[1] as usize != num_frames || s[2] != h || s[3] != w {
                return Err(Error::Msg(format!(
                    "krea v2v: source latents shape {s:?} must be [{c}, {num_frames}, {h}, {w}] \
                     (one source frame per generated frame)"
                )));
            }
        }
        Ok(())
    }
}

impl<F: KreaArBlockForward> FamilyDriver<Array> for KreaRealtimeArDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> GenResult<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        let step_ts = self.schedule.step_timesteps();
        let n_steps = step_ts.len();
        if n_steps == 0 {
            return Err(GenError::Msg("krea AR: empty denoising schedule".into()));
        }
        let chunk_counts = chunk_frame_counts(self.num_latent_frames, self.frames_per_block);
        if chunk_counts.is_empty() {
            return Err(GenError::Msg(
                "krea AR: num_latent_frames and frames_per_block must be >= 1".into(),
            ));
        }
        let expected_total_steps = chunk_counts.len() * n_steps;
        if input.total_steps != expected_total_steps || input.span.end > expected_total_steps {
            return Err(GenError::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Krea AR schedule steps={expected_total_steps}",
                self.support.driver_id, input.span, input.total_steps
            )));
        }
        self.validate_source_shape().map_err(GenError::from)?;
        if input.cancel.is_cancelled() {
            return Err(GenError::Canceled);
        }

        // Validate Carry compatibility BEFORE mutating `state` (`PIPE-3`).
        let carrying = input.transition == SliceTransition::Continue
            && input.history == DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<KreaArDriverState>() else {
                return Err(GenError::Unsupported(format!(
                    "{}: Carry requested without prior Krea AR state in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.seed != input.seed
                || existing.total_steps != input.total_steps
                || existing.n_steps != n_steps
                || existing.chunk_counts != chunk_counts
                || existing.expected_next_step != input.span.start
            {
                return Err(GenError::Unsupported(format!(
                    "{}: Carry refused across incompatible Krea AR trajectory state",
                    self.support.driver_id
                )));
            }
        }

        state.enter_slice(&input)?;

        let mut ar_state = if carrying {
            state
                .take_history::<KreaArDriverState>()
                .expect("validated Krea AR state exists")
        } else {
            // Init the whole clip's noise once from the seed (then slice per chunk, matching the
            // reference's `noise[:, block]`), and carry a split PRNG key for per-step renoise draws.
            let key = random::key(input.seed)
                .map_err(Error::from)
                .map_err(GenError::from)?;
            let (noise_key, next_key) = random::split(&key, 2)
                .map_err(Error::from)
                .map_err(GenError::from)?;
            mlx_gen::video_execution::draw(
                0,
                "krea_split_chain_initial_left_then_right",
                input.seed,
            );
            let _ = state.next_draw_index();
            let full_noise = random::normal::<f32>(
                &[
                    self.channels as i32,
                    self.num_latent_frames as i32,
                    self.latent_height as i32,
                    self.latent_width as i32,
                ],
                None,
                None,
                Some(&noise_key),
            )
            .map_err(Error::from)
            .map_err(GenError::from)?;
            KreaArDriverState {
                seed: input.seed,
                total_steps: input.total_steps,
                n_steps,
                chunk_counts,
                key: next_key,
                full_noise,
                active_chunk: None,
                outputs: Vec::with_capacity(self.num_latent_frames.div_ceil(self.frames_per_block)),
                frame_cursor: 0,
                start_token: self.start_token_offset,
                expected_next_step: input.span.start,
            }
        };

        for outer_step in input.span.clone() {
            // Per-step / per-chunk cancellation checkpoint (F-003): bail before committing to a
            // chunk or running the next denoise forward.
            if input.cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let chunk_idx = outer_step / n_steps;
            let step_in_chunk = outer_step % n_steps;
            let chunk_frames = ar_state.chunk_counts[chunk_idx];

            let mut cur = match ar_state.active_chunk.take() {
                Some(existing) => existing,
                None => {
                    let idx: Vec<i32> = (ar_state.frame_cursor as i32
                        ..(ar_state.frame_cursor + chunk_frames) as i32)
                        .collect();
                    let idx = Array::from_slice(&idx, &[idx.len() as i32]);
                    let noise_chunk = ar_state
                        .full_noise
                        .take_axis(&idx, 1)
                        .map_err(Error::from)
                        .map_err(GenError::from)?;
                    match &self.init_source {
                        None => noise_chunk,
                        Some((source, sigma_init)) => {
                            let source_chunk = source
                                .take_axis(&idx, 1)
                                .map_err(Error::from)
                                .map_err(GenError::from)?;
                            renoise_step(&source_chunk, &noise_chunk, *sigma_init)
                                .map_err(GenError::from)?
                        }
                    }
                }
            };

            let t = step_ts[step_in_chunk];
            let sigma_f64 = self.schedule.sigma_at_timestep(t);
            let next_sigma_f64 = if step_in_chunk + 1 < n_steps {
                self.schedule.sigma_at_timestep(step_ts[step_in_chunk + 1])
            } else {
                0.0
            };
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: sigma_f64 as f32,
                next_sigma: next_sigma_f64 as f32,
            };

            hooks.observe(&info, &cur)?;

            let is_final = step_in_chunk + 1 == n_steps;
            let commit = is_final && !self.do_kv_recomp;
            let velocity = self
                .forward
                .borrow_mut()
                .forward_block(&cur, t as f32, ar_state.start_token, commit)
                .map_err(GenError::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let x0_i = euler_x0(&cur, &velocity, sigma_f64).map_err(GenError::from)?;
            if is_final {
                cur = hooks.post_step(&info, x0_i)?;
                mlx_rs::transforms::eval([&cur]).map_err(|e| GenError::Msg(e.to_string()))?;
                hooks.materialize(&info, &cur)?;
                state.record_outer_step(outer_step);
                hooks.emit_progress(&info);

                // Step 3.3: clean-context KV recompute — rerun one forward on the chunk's clean x0
                // at context_noise and commit *that* K/V (the read-only denoise steps left the
                // cache untouched, so this is the chunk's one and only commit).
                if self.do_kv_recomp {
                    if input.cancel.is_cancelled() {
                        return Err(GenError::Canceled);
                    }
                    let _ = self
                        .forward
                        .borrow_mut()
                        .forward_block(&cur, self.context_noise, ar_state.start_token, true)
                        .map_err(GenError::from)?;
                    state.record_evaluation(DriverEvaluationKind::Trajectory);
                }

                ar_state.outputs.push(cur);
                ar_state.frame_cursor += chunk_frames;
                ar_state.start_token += chunk_frames * self.frame_seq_length;
            } else {
                let (eps_key, next_key) = random::split(&ar_state.key, 2)
                    .map_err(Error::from)
                    .map_err(GenError::from)?;
                mlx_gen::video_execution::draw(
                    0,
                    &format!(
                        "krea_split_chain_renoise_left_then_right:frame_{}:step_{step_in_chunk}",
                        ar_state.frame_cursor
                    ),
                    input.seed,
                );
                let _ = state.next_draw_index();
                ar_state.key = next_key;
                let eps = random::normal::<f32>(x0_i.shape(), None, None, Some(&eps_key))
                    .map_err(Error::from)
                    .map_err(GenError::from)?;
                let next_cur = renoise_step(&x0_i, &eps, next_sigma_f64).map_err(GenError::from)?;
                let next_cur = hooks.post_step(&info, next_cur)?;
                mlx_rs::transforms::eval([&next_cur]).map_err(|e| GenError::Msg(e.to_string()))?;
                hooks.materialize(&info, &next_cur)?;
                ar_state.active_chunk = Some(next_cur);
                state.record_outer_step(outer_step);
                hooks.emit_progress(&info);
            }
            ar_state.expected_next_step = outer_step + 1;
        }

        state.finish_slice(input.span.end);
        if input.span.end == expected_total_steps {
            let refs: Vec<&Array> = ar_state.outputs.iter().collect();
            let full_latents = concatenate_axis(&refs, 1)
                .map_err(Error::from)
                .map_err(GenError::from)?;
            state.set_history(ar_state);
            Ok(BoundaryState::XSigma(full_latents))
        } else {
            let boundary_latent = ar_state
                .active_chunk
                .clone()
                .or_else(|| ar_state.outputs.last().cloned())
                .unwrap_or_else(|| Array::from_f32(0.0));
            state.set_history(ar_state);
            Ok(BoundaryState::XSigma(boundary_latent))
        }
    }
}

/// The AR chunk-loop core, driven through [`KreaRealtimeArDriver`] + [`run_mlx_driver_slices`].
///
/// `denoise(chunk, t, start_token, commit) -> velocity` runs one batch-1 forward: it reads the
/// committed history at `start_token` and returns the model velocity `[out_dim, F_chunk, H, W]`;
/// `commit = true` appends that forward's self-attention K/V to the persistent cache.
#[allow(clippy::too_many_arguments)]
fn run_ar_loop_conditioned(
    schedule: &FewStepSchedule,
    channels: usize,
    frame_seq_length: usize,
    frames_per_block: usize,
    params: &ArGenParams,
    do_kv_recomp: bool,
    context_noise: f32,
    start_token_offset: usize,
    init_source: Option<(&Array, f64)>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    denoise: impl FnMut(&Array, f32, usize, bool) -> Result<Array>,
) -> Result<Array> {
    let step_ts = schedule.step_timesteps();
    let execution_sigmas: Vec<f64> = step_ts
        .iter()
        .map(|&t| schedule.sigma_at_timestep(t))
        .chain(std::iter::once(0.0))
        .collect();
    mlx_gen::video_execution::schedule_f64("krea_realtime_sigmas", &execution_sigmas);
    mlx_gen::video_execution::sampling(
        "krea_realtime",
        "self_forcing_euler_renoise",
        "flow_x0",
        serde_json::json!({"context_start":start_token_offset}),
    );
    mlx_gen::video_execution::schedule_f64("krea_realtime_timesteps", step_ts);
    mlx_gen::video_execution::policy(
        "krea_realtime_causal",
        serde_json::json!({"frames_per_block":frames_per_block,"init_source":init_source.is_some(),"do_kv_recomp":do_kv_recomp,"context_noise_bits":context_noise.to_bits()}),
    );
    if step_ts.is_empty() {
        return Err(Error::Msg("krea AR: empty denoising schedule".into()));
    }
    let chunk_counts = chunk_frame_counts(params.num_latent_frames, frames_per_block);
    if chunk_counts.is_empty() {
        return Err(Error::Msg(
            "krea AR: num_latent_frames and frames_per_block must be >= 1".into(),
        ));
    }
    let total_steps = chunk_counts.len() * step_ts.len();
    let driver = KreaRealtimeArDriver::new(
        schedule.clone(),
        channels,
        frame_seq_length,
        frames_per_block,
        params.num_latent_frames,
        params.latent_height,
        params.latent_width,
        do_kv_recomp,
        context_noise,
        start_token_offset,
        init_source.map(|(source, sigma_init)| (source.clone(), sigma_init)),
        denoise,
    );
    driver.validate_source_shape()?;
    let mut run_state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = run_mlx_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(Array::from_f32(0.0)),
        total_steps,
        params.seed,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, boundary, _| Ok(boundary),
    )?;
    match out {
        BoundaryState::XSigma(latents)
        | BoundaryState::CleanEstimate {
            latent: latents, ..
        }
        | BoundaryState::RouteSpecific { state: latents, .. } => Ok(latents),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A never-cancelled flag for the loop-structure/determinism tests (cancel is exercised by the
    /// dedicated `ar_loop_cancels_*` tests below).
    fn no_cancel() -> CancelFlag {
        CancelFlag::new()
    }

    /// A no-op progress sink for the tests that do not assert on progress.
    fn sink() -> impl FnMut(Progress) {
        |_| {}
    }

    fn params(seed: u64, frames: usize) -> ArGenParams {
        ArGenParams {
            seed,
            steps: None,
            num_latent_frames: frames,
            latent_height: 4,
            latent_width: 4,
            fps: 16,
            memory: Default::default(),
        }
    }

    /// Product-path activation oracle: generation uses the config list's count while ignoring its
    /// integer values, matching the pinned online release server.
    #[test]
    fn product_schedule_uses_count_and_ignores_config_values() {
        let cfg = KreaRealtimeConfig::krea_realtime_14b();
        let release = t2v_schedule(&cfg, None).unwrap();
        let want = [1000.0, 937.5, 833.333_312_988_281_2, 625.0, 0.0];
        for (got, want) in release.step_timesteps().iter().zip(want) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
        let model_timesteps: Vec<f32> =
            release.step_timesteps().iter().map(|&t| t as f32).collect();
        assert_eq!(
            model_timesteps,
            [1000.0, 937.5, 833.333_3, 625.0, 0.0],
            "product paths must pass the release values into the DiT"
        );

        let three = t2v_schedule(&cfg, Some(3)).unwrap();
        assert_eq!(three.num_steps(), 3);
        assert_eq!(three.step_timesteps()[1], 833.333_312_988_281_2);

        let mut custom = cfg;
        custom.ar.denoising_step_list = vec![900, 700, 0];
        let from_custom_count = t2v_schedule(&custom, None).unwrap();
        assert_eq!(from_custom_count.num_steps(), 3);
        assert_eq!(from_custom_count.step_timesteps()[1], 833.333_312_988_281_2);
        assert_ne!(
            from_custom_count.step_timesteps(),
            &[900.0, 700.0, 0.0],
            "product paths must not execute YAML timestep values"
        );
    }

    #[test]
    fn chunk_frame_counts_ceils_and_partials() {
        assert_eq!(chunk_frame_counts(6, 3), vec![3, 3]); // exact
        assert_eq!(chunk_frame_counts(7, 3), vec![3, 3, 1]); // ceil, partial tail
        assert_eq!(chunk_frame_counts(5, 2), vec![2, 2, 1]);
        assert_eq!(chunk_frame_counts(1, 3), vec![1]);
        assert_eq!(chunk_frame_counts(0, 3), Vec::<usize>::new());
        assert_eq!(chunk_frame_counts(3, 0), Vec::<usize>::new());
        // Count == ceil(frames / fpb).
        assert_eq!(chunk_frame_counts(7, 3).len(), 7usize.div_ceil(3));
    }

    /// The loop structure with a fake denoiser (no weights), **recompute off** (the S4 baseline):
    /// forward count = chunks · steps, `commit` fires exactly once per chunk (the last step),
    /// `start_token` advances by `chunk_frames · frame_seq_length` and equals a simulated cache position
    /// on every call (KV threading at the loop level — chunk *k* is driven against the committed history,
    /// never a reset), and the output has the correct `[C, num_frames, H, W]` shape.
    #[test]
    fn ar_loop_structure_and_kv_threading() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let fsl = 4; // 4×4 latent, patch 2×2 → 2·2 tokens/frame
        let fpb = 3;
        let p = params(1234, 7); // 3 chunks: [3, 3, 1]

        // Simulated cache position: advances only when a chunk commits, like the real cache.
        let sim_stored = RefCell::new(0usize);
        let calls = RefCell::new(Vec::<(f32, usize, bool)>::new());

        // Recompute OFF ⇒ the final denoise step commits (S4 behaviour).
        let out = run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            |chunk, t, start, commit| {
                // Every step of a chunk sees the committed history (== the simulated cache position).
                assert_eq!(
                    start,
                    *sim_stored.borrow(),
                    "chunk forward must run against the committed history"
                );
                calls.borrow_mut().push((t, start, commit));
                if commit {
                    *sim_stored.borrow_mut() += chunk.shape()[1] as usize * fsl;
                }
                // Fake velocity: zeros ⇒ x0 = cur (the loop's RNG still drives the output).
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        )
        .unwrap();

        // 3 chunks × 5 steps = 15 forwards; commit true exactly 3 times (the final step of each chunk).
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 15);
        assert_eq!(calls.iter().filter(|(_, _, f)| *f).count(), 3);
        // The commit fires on the terminal timestep 0 (the S4 final near-clean step).
        assert!(calls
            .iter()
            .filter(|(_, _, f)| *f)
            .all(|(t, _, _)| *t == 0.0));
        // start_token sequence: chunk 0 @ 0 (×5), chunk 1 @ 3·fsl (×5), chunk 2 @ 6·fsl (×5).
        assert!(calls[0..5].iter().all(|(_, s, _)| *s == 0));
        assert!(calls[5..10].iter().all(|(_, s, _)| *s == 3 * fsl));
        assert!(calls[10..15].iter().all(|(_, s, _)| *s == 6 * fsl));
        // The committed history ends at the full clip's token count.
        assert_eq!(*sim_stored.borrow(), 7 * fsl);

        // Output shape [C, num_frames, H, W].
        assert_eq!(out.shape(), &[16, 7, 4, 4]);
    }

    /// S5 clean-context recompute wiring + toggle (weight-free). With `do_kv_recomp = true` every
    /// denoise step is read-only and a single extra forward commits per chunk **at `context_noise`, on
    /// the chunk's clean `x0`** — not the noisy final denoise input. A fake non-zero velocity makes the
    /// clean `x0` differ from the final step's input so the commit input is discriminated; a single
    /// chunk makes the loop output equal that `x0`.
    #[test]
    fn ar_loop_recompute_commits_clean_context_when_on() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 2); // exactly ONE chunk ⇒ loop output == that chunk's clean x0
        let ctx_noise = 7.0f32; // sentinel: the recompute must run at exactly this timestep

        // (t, start, commit, input_sum). A non-zero constant velocity ⇒ x0 = cur − σ·1 ≠ cur, so the
        // recompute's clean-x0 input is clearly discriminable from the noisy final denoise input.
        let calls = RefCell::new(Vec::<(f32, usize, bool, f32)>::new());
        let denoise = |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
            let s = mlx_rs::ops::sum(chunk, None).unwrap().item::<f32>();
            calls.borrow_mut().push((t, start, commit, s));
            Ok(Array::ones::<f32>(chunk.shape())?)
        };
        let out = run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            true,
            ctx_noise,
            &no_cancel(),
            &mut sink(),
            denoise,
        )
        .unwrap();
        let calls = calls.into_inner();

        // n_steps read-only denoise forwards + exactly one commit (the recompute).
        assert_eq!(
            calls.len(),
            n_steps + 1,
            "denoise steps (read-only) + one recompute"
        );
        assert_eq!(
            calls.iter().filter(|(_, _, c, _)| *c).count(),
            1,
            "exactly one commit per chunk (the recompute), not the final denoise step"
        );
        // The denoise steps are all read-only; the single commit is the recompute at context_noise.
        assert!(calls[..n_steps].iter().all(|(_, _, c, _)| !*c));
        let (rt, _rs, rc, r_sum) = *calls.last().unwrap();
        assert!(rc, "the last forward is the commit");
        assert_eq!(rt, ctx_noise, "the recompute runs at context_noise");
        // The recompute ran on the clean x0 (== the single-chunk output), NOT the final denoise input.
        let out_sum = mlx_rs::ops::sum(&out, None).unwrap().item::<f32>();
        let final_denoise_input_sum = calls[n_steps - 1].3;
        assert!(
            (r_sum - out_sum).abs() < 1e-3,
            "recompute input must be the clean x0 (loop output): got {r_sum}, want {out_sum}"
        );
        assert!(
            (r_sum - final_denoise_input_sum).abs() > 1e-4,
            "recompute input must differ from the noisy final denoise input ({final_denoise_input_sum})"
        );
    }

    /// Recompute toggle: `do_kv_recomp = false` reproduces the S4 commit pattern (no extra forward; the
    /// final denoise step is the single commit) — proving the two modes are a clean A/B.
    #[test]
    fn ar_loop_recompute_off_matches_s4_commit_pattern() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let p = params(1234, 2); // one chunk
        let calls = RefCell::new(Vec::<(f32, bool)>::new());
        let denoise = |chunk: &Array, t: f32, _s: usize, commit: bool| -> Result<Array> {
            calls.borrow_mut().push((t, commit));
            Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
        };
        run_ar_loop(
            &schedule,
            16,
            4,
            2,
            &p,
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            denoise,
        )
        .unwrap();
        let calls = calls.into_inner();
        assert_eq!(calls.len(), n_steps, "no recompute forward when off");
        assert_eq!(
            calls.iter().filter(|(_, c)| *c).count(),
            1,
            "one commit: the final step"
        );
        assert!(calls.last().unwrap().1 && calls.last().unwrap().0 == 0.0);
    }

    /// Determinism: identical seed ⇒ identical latents; a different seed ⇒ different latents.
    #[test]
    fn ar_loop_is_seed_deterministic() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let zeros = |chunk: &Array, _t: f32, _s: usize, _f: bool| -> Result<Array> {
            Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
        };

        let a = run_ar_loop(
            &schedule,
            16,
            4,
            3,
            &params(7, 6),
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            zeros,
        )
        .unwrap();
        let b = run_ar_loop(
            &schedule,
            16,
            4,
            3,
            &params(7, 6),
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            zeros,
        )
        .unwrap();
        let c = run_ar_loop(
            &schedule,
            16,
            4,
            3,
            &params(8, 6),
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            zeros,
        )
        .unwrap();

        let diff_same = mlx_rs::ops::max(
            mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &b).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        assert_eq!(diff_same, 0.0, "same seed must be bit-identical");

        let diff_seed = mlx_rs::ops::max(
            mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &c).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        assert!(diff_seed > 0.0, "a different seed must change the latents");
    }

    /// S7 cache-warm offset: `start_token_offset` makes the generated chunks run against a pre-warmed
    /// cache — every forward's `start` begins at the offset and advances by `chunk_frames · fsl`, and the
    /// output shape is the generated frame count (the warm/prefix is added by the caller, not the loop).
    #[test]
    fn run_ar_loop_conditioned_threads_start_token_offset() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 4); // 2 generated chunks: [2, 2]
        let offset = 3 * fsl; // pretend 3 context frames were warmed

        let starts = RefCell::new(Vec::<usize>::new());
        let out = run_ar_loop_conditioned(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            offset,
            None,
            &no_cancel(),
            &mut sink(),
            |chunk, _t, start, _commit| {
                starts.borrow_mut().push(start);
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        )
        .unwrap();

        let starts = starts.into_inner();
        // Chunk 0 begins at the offset; chunk 1 at offset + 2·fsl. 5 steps each.
        assert!(starts[0..5].iter().all(|&s| s == offset));
        assert!(starts[5..10].iter().all(|&s| s == offset + 2 * fsl));
        // The loop returns only the generated frames; the caller prepends the context.
        assert_eq!(out.shape(), &[16, 4, 4, 4]);
    }

    /// S7 v2v init: `init_source` seeds each chunk from the source renoised to `sigma_init`, so the
    /// first denoise input **depends on the source and the strength** — a different source or a different
    /// `sigma_init` yields a different init (the discriminating v2v levers), while pure-noise init
    /// (`None`) ignores the source entirely.
    #[test]
    fn run_ar_loop_conditioned_v2v_init_uses_source_and_strength() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let fsl = 4;
        let fpb = 2;
        let p = params(7, 2); // one chunk

        // Capture the first denoise input's sum (the init the source feeds).
        let first_input_sum = |src: Option<&Array>, sigma: f64| -> f32 {
            let seen = RefCell::new(None::<f32>);
            let init_source = src.map(|s| (s, sigma));
            run_ar_loop_conditioned(
                &schedule,
                16,
                fsl,
                fpb,
                &p,
                false,
                0.0,
                0,
                init_source,
                &no_cancel(),
                &mut sink(),
                |chunk, _t, _start, _commit| {
                    if seen.borrow().is_none() {
                        *seen.borrow_mut() =
                            Some(mlx_rs::ops::sum(chunk, None).unwrap().item::<f32>());
                    }
                    Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
                },
            )
            .unwrap();
            seen.into_inner().unwrap()
        };

        let source_a = Array::full::<f32>(&[16, 2, 4, 4], Array::from_f32(1.0)).unwrap();
        let source_b = Array::full::<f32>(&[16, 2, 4, 4], Array::from_f32(-1.0)).unwrap();

        // At a partial strength (sigma 0.5) the init carries (1−σ)·source, so A and B differ.
        let a = first_input_sum(Some(&source_a), 0.5);
        let b = first_input_sum(Some(&source_b), 0.5);
        assert!(
            (a - b).abs() > 1e-3,
            "a different source must change the v2v init: {a} vs {b}"
        );

        // A different strength (sigma) with the same source also changes the init.
        let a_low = first_input_sum(Some(&source_a), 0.05);
        assert!(
            (a - a_low).abs() > 1e-3,
            "a different strength must change the v2v init"
        );

        // Pure-noise init (None) ignores the source: same seed ⇒ same init regardless of source.
        let n1 = first_input_sum(None, 1.0);
        let n2 = first_input_sum(None, 1.0);
        assert_eq!(
            n1, n2,
            "pure-noise init is source-independent and seed-deterministic"
        );
    }

    /// sc-8441 S8 — **cancel at a chunk boundary**: a `CancelFlag` tripped after the first chunk commits
    /// makes the loop bail at the next chunk's per-chunk poll with [`Error::Canceled`], running
    /// **strictly fewer** transformer forwards and committing **fewer chunks** than a full run (the
    /// discriminating assertion: a mid-generation cancel must not complete all chunks).
    #[test]
    fn ar_loop_cancels_at_chunk_boundary_and_commits_fewer_chunks() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 6); // 3 chunks: [2, 2, 2]

        // Baseline: a full (never-cancelled) run does 3·n_steps forwards and 3 commits.
        let full_calls = RefCell::new(0usize);
        let full_commits = RefCell::new(0usize);
        run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            &no_cancel(),
            &mut sink(),
            |chunk, _t, _s, commit| {
                *full_calls.borrow_mut() += 1;
                if commit {
                    *full_commits.borrow_mut() += 1;
                }
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        )
        .unwrap();
        assert_eq!(
            *full_calls.borrow(),
            3 * n_steps,
            "full run = chunks · steps"
        );
        assert_eq!(*full_commits.borrow(), 3, "full run commits every chunk");

        // Cancel after the first chunk commits (recompute off ⇒ the final step commits).
        let cancel = CancelFlag::new();
        let calls = RefCell::new(0usize);
        let commits = RefCell::new(0usize);
        let res = run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            &cancel,
            &mut sink(),
            |chunk, _t, _s, commit| {
                *calls.borrow_mut() += 1;
                if commit {
                    *commits.borrow_mut() += 1;
                    cancel.cancel(); // trip cancellation the instant the first chunk commits
                }
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        );

        assert!(
            matches!(res, Err(Error::Canceled)),
            "a set flag must bail with the typed Error::Canceled"
        );
        // DISCRIMINATING: fewer forwards AND fewer committed chunks than the full run.
        assert_eq!(
            *calls.borrow(),
            n_steps,
            "only the first chunk's steps ran before the boundary poll bailed"
        );
        assert!(
            *calls.borrow() < 3 * n_steps,
            "a mid-clip cancel must run fewer forwards than the whole clip"
        );
        assert_eq!(*commits.borrow(), 1, "only the first chunk committed");
        assert!(
            *commits.borrow() < 3,
            "a mid-clip cancel must commit fewer chunks than the whole clip"
        );
    }

    /// sc-8441 S8 — **cancel mid-chunk** (per-denoise-step polling): a flag tripped partway through the
    /// first chunk's denoise steps bails at the *next* per-step poll, before the chunk finishes its
    /// steps or commits — proving the poll is per-step, not only per-chunk.
    #[test]
    fn ar_loop_cancels_mid_chunk_before_completing_the_chunk() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps(); // 5
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 6); // 3 chunks, but we bail during chunk 0

        let cancel = CancelFlag::new();
        let calls = RefCell::new(0usize);
        let commits = RefCell::new(0usize);
        let cancel_at = 2usize; // trip during the 2nd forward — well before the chunk's final (5th) step
        let res = run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            &cancel,
            &mut sink(),
            |chunk, _t, _s, commit| {
                let n = {
                    let mut c = calls.borrow_mut();
                    *c += 1;
                    *c
                };
                if commit {
                    *commits.borrow_mut() += 1;
                }
                if n == cancel_at {
                    cancel.cancel();
                }
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        );

        assert!(matches!(res, Err(Error::Canceled)));
        // The per-step poll bails at the next step, before the chunk's remaining steps run.
        assert_eq!(
            *calls.borrow(),
            cancel_at,
            "the per-step poll bails before the next forward"
        );
        assert!(
            *calls.borrow() < n_steps,
            "a mid-chunk cancel must bail before completing the first chunk"
        );
        assert_eq!(
            *commits.borrow(),
            0,
            "no chunk commits on a mid-chunk cancel (recompute off ⇒ commit is the final step)"
        );
    }

    /// sc-8441 S8 — **per-step progress**: the loop emits exactly one [`Progress::Step`] per denoise
    /// step across all chunks, with a strictly monotonic 1-based `current` over a constant
    /// `total = num_chunks · n_steps` (the clean-context recompute forward is not counted).
    #[test]
    fn ar_loop_emits_monotonic_per_step_progress() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 6); // 3 chunks: [2, 2, 2] ⇒ 15 denoise steps

        let events = RefCell::new(Vec::<(u32, u32)>::new());
        run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            &no_cancel(),
            &mut |pr: Progress| {
                if let Progress::Step { current, total } = pr {
                    events.borrow_mut().push((current, total));
                }
            },
            |chunk, _t, _s, _c| Ok(Array::zeros::<f32>(chunk.shape()).unwrap()),
        )
        .unwrap();

        let events = events.into_inner();
        let expected_total = 3 * n_steps as u32;
        assert_eq!(
            events.len(),
            expected_total as usize,
            "one Progress::Step per denoise step across all chunks"
        );
        assert!(
            events.iter().all(|&(_, total)| total == expected_total),
            "total is a constant num_chunks · n_steps"
        );
        for (i, &(current, _)) in events.iter().enumerate() {
            assert_eq!(
                current,
                i as u32 + 1,
                "1-based, strictly monotonic step count"
            );
        }
        assert_eq!(
            events.last().unwrap().0,
            expected_total,
            "the final step reaches total"
        );
    }

    /// sc-8441 S8 — **recompute-on** (the shipped default) also polls + reports per denoise step: with
    /// `do_kv_recomp = true` every denoise step is read-only and the single per-chunk commit is the
    /// clean-context recompute, which must NOT emit a progress step (only the n_steps denoise steps do).
    #[test]
    fn ar_loop_progress_excludes_the_recompute_forward() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 2); // exactly one chunk

        let count = RefCell::new(0usize);
        run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            true, // recompute on ⇒ one extra (commit) forward per chunk
            7.0,
            &no_cancel(),
            &mut |pr: Progress| {
                if matches!(pr, Progress::Step { .. }) {
                    *count.borrow_mut() += 1;
                }
            },
            |chunk, _t, _s, _c| Ok(Array::ones::<f32>(chunk.shape()).unwrap()),
        )
        .unwrap();
        assert_eq!(
            count.into_inner(),
            n_steps,
            "progress counts denoise steps only, not the clean-context recompute forward"
        );
    }

    /// sc-17894's safety gate: the optimized cache and the exact pre-change eager-retention policy
    /// must produce bit-identical real-weight latents. Three chunks are sufficient to cross the
    /// shipped six-frame window: before chunk three the old cache holds six frames while the new one
    /// keeps only the three cached frames that chunk reads.
    #[test]
    #[ignore = "real Krea snapshot; run on the rw-krea Metal lane"]
    fn next_read_eviction_is_bit_identical_to_eager_max_window_retention() {
        use crate::load_krea_realtime_transformer_with_quant;
        use mlx_gen::weights::Weights;
        use std::collections::HashMap;
        use std::path::PathBuf;

        let root = PathBuf::from(
            std::env::var("KREA_REALTIME_SNAPSHOT_DIR")
                .expect("KREA_REALTIME_SNAPSHOT_DIR must name the q4 tier"),
        );
        assert!(root.join("dit.safetensors").is_file(), "missing real DiT");

        let (width, height, latent_frames) = (832usize, 480usize, 9usize);
        let (latent_h, latent_w) = (height / 8, width / 8);
        let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
        cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
        cfg.ar.frame_seq_length =
            (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
        cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

        let weights = Weights::from_file(root.join("dit.safetensors")).expect("open the real DiT");
        let raw: HashMap<String, Array> = weights
            .keys()
            .map(|key| {
                (
                    key.to_string(),
                    weights.get(key).expect("listed DiT key").clone(),
                )
            })
            .collect();
        let (dit, _) =
            load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the real DiT");
        let transformer = CausalKreaTransformer::new(dit, &cfg);
        let context = Array::zeros::<f32>(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
            .expect("zero text context");
        let params = ArGenParams {
            seed: 7,
            steps: Some(2),
            num_latent_frames: latent_frames,
            latent_height: latent_h,
            latent_width: latent_w,
            fps: 24,
            memory: Default::default(),
        };

        let mut optimized = transformer.new_cache();
        let optimized_latents = generate_latents_into(
            &transformer,
            &cfg,
            &context,
            &params,
            &mut optimized,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .expect("optimized real-weight generation");
        mlx_rs::transforms::eval([&optimized_latents]).expect("materialize optimized latents");
        let optimized_values = optimized_latents.as_slice::<f32>().to_vec();
        mlx_rs::memory::clear_cache();

        let mut eager = CausalKvCache::new_eager_reference(
            cfg.wan.num_layers,
            cfg.ar.max_attention_size(),
            cfg.ar.sink_tokens(),
            cfg.ar.kv_cache_quant,
        );
        let eager_latents = generate_latents_into(
            &transformer,
            &cfg,
            &context,
            &params,
            &mut eager,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .expect("eager-reference real-weight generation");
        mlx_rs::transforms::eval([&eager_latents]).expect("materialize eager-reference latents");

        assert_eq!(optimized_latents.shape(), eager_latents.shape());
        assert_eq!(
            optimized_values,
            eager_latents.as_slice::<f32>(),
            "evicting only never-read KV must not change one latent bit"
        );

        let old_window = cfg.ar.max_attention_size();
        assert_eq!(optimized.retained_tokens(), old_window);
        assert_eq!(eager.retained_tokens(), old_window);
        optimized
            .window_prev(cfg.ar.block_size())
            .expect("trim optimized cache to the next read");
        eager
            .window_prev(cfg.ar.block_size())
            .expect("read eager-reference cache");
        assert_eq!(optimized.retained_tokens() * 2, old_window);
        assert_eq!(eager.retained_tokens(), old_window);
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED generate::tests::next_read_eviction_is_bit_identical_to_eager_max_window_retention");
    }

    #[test]
    fn krea_realtime_ar_driver_passes_tier0_conformance_across_split_boundaries() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let schedule = FewStepSchedule::for_strength(5.0, 1.0, 5).unwrap();
        let latents_equal = |a: &Array, b: &Array| {
            mlx_rs::transforms::eval([a, b]).unwrap();
            a.shape() == b.shape()
                && a.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(b.as_slice::<f32>().iter().map(|v| v.to_bits()))
        };

        for do_kv_recomp in [true, false] {
            for split_at in [3usize, 5] {
                let schedule_for_case = schedule.clone();
                let report = verify_driver_conformance(ConformanceCase {
                    total_steps: 10,
                    split_at,
                    seed: 42,
                    base_geometry: (32, 32),
                    alt_geometry: (64, 64),
                    make_driver: move || {
                        let forward =
                            |chunk: &Array, t: f32, start: usize, _commit: bool| -> Result<Array> {
                                let bias = Array::from_f32(t * 0.0005 + start as f32 * 0.01);
                                let scaled = mlx_rs::ops::multiply(chunk, Array::from_f32(0.25))?;
                                Ok(mlx_rs::ops::add(&scaled, &bias)?)
                            };
                        Box::new(KreaRealtimeArDriver::new(
                            schedule_for_case.clone(),
                            4,
                            4,
                            3,
                            5,
                            4,
                            4,
                            do_kv_recomp,
                            0.0,
                            0,
                            None,
                            forward,
                        ))
                    },
                    make_initial: || Array::zeros::<f32>(&[4, 5, 4, 4]).unwrap(),
                    latents_equal,
                })
                .unwrap();

                assert_eq!(report.driver_id, "krea_realtime_ar");
                assert_eq!(report.total_steps, 10);
                assert!(report.verified_inert_split);
                assert!(report.verified_carry_refusal_or_support);
                assert!(report.verified_renoise_and_resample_guards);
                assert!(report.verified_cancellation);
                let expected_evals = if do_kv_recomp { 12 } else { 10 };
                assert_eq!(report.unsplit_evaluations, expected_evals);
                assert_eq!(report.unsplit_draws, 9);
            }
        }
    }

    #[test]
    fn krea_realtime_ar_driver_enforces_kv_commit_cadence_and_i2v_offset() {
        use std::rc::Rc;

        let schedule = FewStepSchedule::for_strength(5.0, 1.0, 5).unwrap();
        let ctx_noise = 7.0f32;
        let fsl = 4usize;
        let start_token_offset = 3 * fsl;
        let source = Array::full::<f32>(&[4, 5, 4, 4], Array::from_f32(0.5)).unwrap();
        let calls = Rc::new(RefCell::new(Vec::<(f32, usize, bool, i32)>::new()));
        let calls_ref = Rc::clone(&calls);

        let forward = move |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
            let chunk_frames = chunk.shape()[1];
            calls_ref
                .borrow_mut()
                .push((t, start, commit, chunk_frames));
            Ok(Array::full::<f32>(chunk.shape(), Array::from_f32(0.1))?)
        };

        let driver = KreaRealtimeArDriver::new(
            schedule,
            4,
            fsl,
            3,
            5,
            4,
            4,
            true,
            ctx_noise,
            start_token_offset,
            Some((source, 0.5)),
            forward,
        );
        driver.validate_source_shape().unwrap();

        let mut progress = Vec::new();
        let mut run_state = DriverRunState::new();
        let cancel = CancelFlag::new();
        let mut on_progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                progress.push((current, total));
            }
        };
        let mut hooks = StepHooks::new(&mut on_progress);
        let boundary = run_mlx_driver_slices(
            &driver,
            [
                SliceSpec {
                    pass: 0,
                    phase: 0,
                    span: 0..3,
                    transition: SliceTransition::Start,
                    history: DriverHistoryPolicy::Reset,
                    geometry: Some((32, 32)),
                    request_boundary_estimate: false,
                },
                SliceSpec {
                    pass: 0,
                    phase: 1,
                    span: 3..10,
                    transition: SliceTransition::Continue,
                    history: DriverHistoryPolicy::Carry,
                    geometry: Some((32, 32)),
                    request_boundary_estimate: false,
                },
            ],
            BoundaryState::XSigma(Array::from_f32(0.0)),
            10,
            99,
            &cancel,
            None,
            &mut run_state,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();

        let out = match boundary {
            BoundaryState::XSigma(latents) => latents,
            _ => panic!("expected XSigma terminal latents"),
        };
        assert_eq!(out.shape(), &[4, 5, 4, 4]);
        assert_eq!(run_state.completed_outer_steps, 10);
        assert_eq!(run_state.internal_evaluations, 12);
        assert_eq!(run_state.stochastic_draw_index, 9);
        assert_eq!(
            progress,
            (1u32..=10).map(|i| (i, 10u32)).collect::<Vec<_>>(),
            "clean-context KV commits must not emit extra Progress::Step events"
        );

        let log = calls.borrow();
        assert_eq!(log.len(), 12);
        // Chunk 0 (3 frames @ start_token=12): 5 readonly denoise steps + 1 context_noise commit.
        assert!(log[0..5]
            .iter()
            .all(|&(_, s, commit, f)| s == 12 && !commit && f == 3));
        assert_eq!(log[5], (ctx_noise, 12, true, 3));
        // Chunk 1 (2 frames @ start_token=24): 5 readonly denoise steps + 1 context_noise commit.
        assert!(log[6..11]
            .iter()
            .all(|&(_, s, commit, f)| s == 24 && !commit && f == 2));
        assert_eq!(log[11], (ctx_noise, 24, true, 2));

        // Mismatched V2V source latent frame count fails cleanly in validate_source_shape.
        let bad_driver = KreaRealtimeArDriver::new(
            FewStepSchedule::for_strength(5.0, 1.0, 5).unwrap(),
            4,
            fsl,
            3,
            5,
            4,
            4,
            true,
            0.0,
            0,
            Some((Array::zeros::<f32>(&[4, 4, 4, 4]).unwrap(), 0.5)),
            |chunk: &Array, _t: f32, _s: usize, _c: bool| Ok(chunk.clone()),
        );
        assert!(bad_driver.validate_source_shape().is_err());
    }
}
