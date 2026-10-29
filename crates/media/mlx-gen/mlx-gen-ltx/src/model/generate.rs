//! The generation body: DFR planning, the noise-to-latent AV route and the calibration-phase
//! bookkeeping behind `Generator::generate`.

use super::*;

/// Apply the shared frame-count resolver to the actual provider request.  The predictor is injected
/// solely so the provider-path test can prove that its result changes the allocated-frame plan;
/// production supplies [`DurationHead::predict_seconds`] above.
pub(super) fn apply_predicted_frames(
    req: &GenerationRequest,
    fps: f32,
    predict_seconds: &mut dyn FnMut() -> Result<f32>,
) -> Result<GenerationRequest> {
    let mut core_predict = || predict_seconds().map_err(mlx_gen::gen_core::Error::from);
    let frames = gen_core_ltx::duration_head::resolve_request_num_frames(
        req.frames,
        req.auto_duration,
        fps,
        TEMPORAL_SCALE,
        &mut core_predict,
    )?;
    let mut resolved = req.clone();
    if let Some(frames) = frames {
        resolved.frames = Some(frames);
    }
    Ok(resolved)
}

/// Map the two declared LTX-2.5 DFR controls onto the concrete latent pipeline shape.  A positive
/// generated-keyframe count inserts exactly that many evenly spaced slots; a temporal request
/// additionally pads to the shared DFR segment canvas so each temporal round has its real tile
/// anchors.  The same plan drives noise allocation and the actual `generate_dfr_av_latents` call.
pub(super) fn plan_dfr_request(req: &GenerationRequest) -> Result<Option<DfrPlan>> {
    let requested_frames = req.frames.unwrap_or(1);
    let generated_count = req.num_generated_keyframes.unwrap_or(0);
    let temporal_upsample_rounds = req.temporal_upsample_rounds.unwrap_or(0);
    if generated_count == 0 && temporal_upsample_rounds == 0 {
        return Ok(None);
    }

    let (canvas_frames, automatic_positions) = if temporal_upsample_rounds > 0 {
        let (canvas, _, positions) = gen_core_ltx::ltx_dfr::resolve_canvas(
            i64::from(requested_frames),
            i64::from(TEMPORAL_SCALE),
        )?;
        (canvas as u32, positions)
    } else {
        (requested_frames, Vec::new())
    };
    let keyframe_positions = if generated_count > 0 {
        gen_core_ltx::ltx_dfr::evenly_spaced_keyframe_positions(
            generated_count,
            i64::from(canvas_frames),
        )
    } else {
        automatic_positions
    };
    if keyframe_positions.is_empty() {
        return Err(Error::Msg(
            "ltx_2_5: DFR needs at least two requested frames to place a generated-keyframe slot"
                .into(),
        ));
    }
    Ok(Some(DfrPlan {
        canvas_frames,
        requested_frames,
        keyframe_positions,
        temporal_upsample_rounds,
    }))
}

/// Choose the provider's real DFR denoise branch or its established two-stage branch.  Progress is
/// passed into the selected callback so the branch decision stays observable in a synthetic test
/// without constructing multi-gigabyte Gemma/DiT fixtures.
pub(super) fn dispatch_dfr<T>(
    plan: Option<&DfrPlan>,
    on_step: &mut dyn FnMut(usize),
    dfr: impl FnOnce(&DfrPlan, &mut dyn FnMut(usize)) -> Result<T>,
    plain: impl FnOnce(&mut dyn FnMut(usize)) -> Result<T>,
) -> Result<T> {
    match plan {
        Some(plan) => dfr(plan, on_step),
        None => plain(on_step),
    }
}

/// Invoke the decoder's declared budget/mode route.  This tiny dispatch is intentionally shared
/// by the provider and its synthetic test, so replacing a DiffVAE decode with the conv path (or
/// dropping the budgeted call) cannot leave its selected mode unobserved.
pub(super) fn decode_diffvae_budgeted<T>(
    mode: DiffVaeMode,
    decode: impl FnOnce(DiffVaeMode) -> Result<T>,
) -> Result<T> {
    decode(mode)
}

impl Ltx {
    /// The A/V path from **staged** text embeddings + injected stage noise (the deterministic seam
    /// `generate` calls with RNG-drawn noise). The Gemma encode has already run and its ~24 GB weights
    /// dropped (sc-10976 — see [`stage_text_phase`](Self::stage_text_phase)); this resolves any optional
    /// I2V conditioning image (VAE-encoded at both stage resolutions — the video is image-conditioned,
    /// the audio stays pure-noise, matching `generate_av.py`'s I2V+Audio), then defers to
    /// [`generate_av_from_embeddings`](Self::generate_av_from_embeddings).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_noise(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        negative_video_ctx: Option<&Array>,
        negative_audio_ctx: Option<&Array>,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // F-051: honor a cancel before the (unbounded) per-keyframe/per-clip VAE conditioning encodes,
        // ahead of the first denoise-loop check. (The Gemma TE encode already ran + dropped in the
        // staged text phase.)
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Materialize every conditioned latent inside one encoder-residency scope. The VAE encoder
        // is request-only: once these arrays are forced, no denoise or decode operation needs its
        // weights. Release it on success, cancellation, and error before the 48-block AvDiT loads.
        let conditioned = (|| {
            let keyframes = self.build_keyframes(req)?;
            let clips = self.build_clips(req)?;
            Ok::<_, Error>((keyframes, clips))
        })();
        if self.vae.release_encoder() {
            mlx_rs::memory::clear_cache();
        }
        let (kf_owned, clip_owned) = conditioned?;
        // Replace-latent conditioning: VAE-encode each keyframe at both stage resolutions (half/full).
        // I2V = a single `Reference` at frame 0; first_last_frame / multi-keyframe = `Keyframe`s.
        let keyframes: Vec<StageKeyframe> = kf_owned
            .iter()
            .map(|(s1, s2, idx, strength)| StageKeyframe {
                stage1: s1,
                stage2: s2,
                frame_idx: *idx,
                strength: *strength,
            })
            .collect();
        // In-context clips (extend_clip / video_bridge) — VAE-encoded at stage-1 resolution, appended
        // as IC-LoRA conditioning tokens in stage 1 only.
        let clips: Vec<StageClip> = clip_owned
            .iter()
            .map(|(s1, idx, strength)| StageClip {
                stage1: s1,
                frame_idx: *idx,
                strength: *strength,
            })
            .collect();
        self.generate_av_from_embeddings(
            req,
            video_ctx,
            audio_ctx,
            negative_video_ctx,
            negative_audio_ctx,
            video_s1,
            video_s2,
            audio_s1,
            audio_s2,
            &keyframes,
            &clips,
            on_progress,
        )
    }

    /// The A/V path from **injected** text embeddings + noise — the pipeline-only seam (no Gemma), so
    /// the parity test can gate the joint 2-stage pipeline + video/audio decode against the reference
    /// conditioning. `video_ctx` `(1, ctx, 4096)`, `audio_ctx` `(1, ctx, 2048)`; video noise
    /// `(1,128,F,h,w)` per stage; audio noise `(1,8,T,16)` per stage (`T = audio_frames`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_av_from_embeddings(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        negative_video_ctx: Option<&Array>,
        negative_audio_ctx: Option<&Array>,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        video_keyframes: &[StageKeyframe],
        video_clips: &[StageClip],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let dfr_plan = self.dfr_plan(req)?;
        let fps = Self::fps(req);
        // A temporal DFR request pads the canvas before the request's public trim point, so its
        // injected noises carry the authoritative latent geometry.  Ordinary LTX keeps the
        // historical request-derived dimensions exactly.
        let (lf, h1, w1, h2, w2) = match dfr_plan.as_ref() {
            Some(_) => (
                video_s1.shape()[2] as usize,
                video_s1.shape()[3] as usize,
                video_s1.shape()[4] as usize,
                video_s2.shape()[3] as usize,
                video_s2.shape()[4] as usize,
            ),
            None => Self::latent_dims(req),
        };
        let audio_frames = match dfr_plan.as_ref() {
            Some(_) => audio_s1.shape()[2] as usize,
            None => Self::audio_frames(req),
        };
        let (pos1, pos2, audio_pos) = if dfr_plan.is_some() {
            (
                create_position_grid_with(
                    1,
                    lf,
                    h1,
                    w1,
                    TEMPORAL_SCALE as i64,
                    SPATIAL_SCALE as i64,
                    fps,
                    true,
                ),
                create_position_grid_with(
                    1,
                    lf,
                    h2,
                    w2,
                    TEMPORAL_SCALE as i64,
                    SPATIAL_SCALE as i64,
                    fps,
                    true,
                ),
                create_audio_position_grid(1, audio_frames),
            )
        } else {
            Self::position_grids(req)
        };

        // Curated unified solver (epic 7114, sc-7122): a curated solver name routes the joint two-stream
        // T2V+A denoise through `generate_av_latents`' `denoise_av_curated` branch (LTX keeps its baked
        // distilled σ schedule — sampler-only, no scheduler). `validate` already rejected any name not in
        // the advertised menu; an unset sampler → the native distilled Euler (the byte-exact default).
        let seed = req.seed.unwrap_or_else(default_seed);
        let curated = req
            .sampler
            .as_deref()
            .filter(|s| mlx_gen::Solver::from_name(s).is_some());
        // The image/keyframe (I2V) + in-context-clip paths condition with per-token `σ·mask` and a
        // post-step `apply_denoise_mask` blend, which have no single-eval curated-sampler hook — they
        // stay on the native distilled Euler. Reject a curated request there rather than silently ignore.
        if curated.is_some() && (!video_clips.is_empty() || !video_keyframes.is_empty()) {
            return Err(Error::Msg(
                "ltx: curated samplers apply to text-to-video only; the image/keyframe (I2V) and \
                 in-context-clip paths use per-token-σ conditioning that stays on the native distilled \
                 Euler"
                    .into(),
            ));
        }
        if dfr_plan.is_some() && (!video_clips.is_empty() || curated.is_some()) {
            return Err(Error::Msg(
                "ltx_2_5: DFR generated-keyframe and temporal requests do not accept \
                 in-context clips or curated samplers"
                    .into(),
            ));
        }

        // F-050 (sc-11133): fold the two-stage (8 + 3 step) distilled schedule into one monotone
        // `1..=11` bar from the σ-DERIVED `current` each stage forwards — NOT a blind per-call counter.
        // The curated 2nd-order solvers LTX advertises (`heun`, `dpmpp_sde`) evaluate the model twice
        // per step, so `on_step` fires twice per step; the old counter reached 22 against `total: 11`
        // (>100% overrun). `StageProgressFold` detects the per-stage restart, folds prior stages in,
        // dedupes the multi-eval repeats, and clamps — so the bar stays monotone non-decreasing,
        // reaches `total`, and never overruns.
        let total_steps = match self.transformer_variant() {
            TransformerVariant::Distilled => TOTAL_STEPS,
            TransformerVariant::Dev => {
                crate::params::LTX_2_5_PARAMS.num_inference_steps + (STAGE2_SIGMAS.len() as u32 - 1)
            }
        };
        let mut fold = StageProgressFold::new(total_steps);
        let mut on_step = |cur: usize| {
            if let Some(current) = fold.observe(cur as u32) {
                on_progress(Progress::Step {
                    current,
                    total: total_steps,
                });
            }
        };
        // extend_clip / video_bridge ride the IC-LoRA keyframe-append path (stage-1 in-context tokens);
        // everything else (T2V / I2V / first_last_frame) is the replace-latent path.
        // sc-10976: build the AvDiT now — AFTER the staged text phase freed the ~24 GB Gemma — so the
        // TE and the DiT never co-reside. Built past the curated-sampler validation above so a rejected
        // request doesn't pay the DiT load. Dropped + `clear_cache()`d below before the VAE decode.
        let transformer = self.build_transformer(req)?;
        let run_dfr = |plan: &DfrPlan, on_step: &mut dyn FnMut(usize)| {
            let execution = self.ltx25_execution()?;
            let dfr = generate_dfr_av_latents(
                &DfrComponents {
                    dit: &transformer,
                    spatial_upsampler: &self.upsampler,
                    temporal_upsampler: Some(&execution.temporal_upsampler),
                    latent_mean: &self.latent_mean,
                    latent_std: &self.latent_std,
                    video_ctx,
                    audio_ctx,
                    negative_video_ctx,
                    negative_audio_ctx,
                    variant: self.transformer_variant(),
                    audio_pos: &audio_pos,
                },
                &DfrRequest {
                    canvas_frames: i64::from(plan.canvas_frames),
                    requested_frames: i64::from(plan.requested_frames),
                    keyframe_positions: &plan.keyframe_positions,
                    fps,
                    seed,
                    temporal_upsample_rounds: plan.temporal_upsample_rounds,
                    detailing_downscale: None,
                    video_keyframes,
                },
                video_s1,
                &pos1,
                video_s2,
                &pos2,
                audio_s1,
                audio_s2,
                &req.cancel,
                on_step,
            )?;
            Ok((dfr.video_latent, dfr.audio_latent, dfr.playback_fps as u32))
        };
        let run_plain = |on_step: &mut dyn FnMut(usize)| {
            if !video_clips.is_empty() {
                generate_av_latents_iclora(
                    &transformer,
                    &self.upsampler,
                    video_s1,
                    &pos1,
                    video_s2,
                    &pos2,
                    audio_s1,
                    audio_s2,
                    &audio_pos,
                    video_ctx,
                    audio_ctx,
                    negative_video_ctx,
                    negative_audio_ctx,
                    self.transformer_variant(),
                    &self.latent_mean,
                    &self.latent_std,
                    video_clips,
                    (LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32),
                    fps,
                    &req.cancel,
                    on_step,
                )
                .map(|(video, audio)| (video, audio, fps as u32))
            } else {
                generate_av_latents_for_variant(
                    &transformer,
                    &self.upsampler,
                    video_s1,
                    &pos1,
                    video_s2,
                    &pos2,
                    audio_s1,
                    audio_s2,
                    &audio_pos,
                    video_ctx,
                    audio_ctx,
                    negative_video_ctx,
                    negative_audio_ctx,
                    self.transformer_variant(),
                    &self.latent_mean,
                    &self.latent_std,
                    video_keyframes,
                    curated,
                    seed,
                    &req.cancel,
                    on_step,
                )
                .map(|(video, audio)| (video, audio, fps as u32))
            }
        };
        let (video_latents, audio_latents, playback_fps) =
            dispatch_dfr(dfr_plan.as_ref(), &mut on_step, run_dfr, run_plain)?;

        // sc-10976: force the denoise output to materialize, then drop the DiT + free the allocator
        // cache so the VAE + audio decode run in the freed footprint (the AvDiT is the denoise peak and
        // nothing downstream needs it). Mirrors Wan's scope-block release before the VAE decode.
        mlx_rs::transforms::eval([&video_latents, &audio_latents])?;
        finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Denoise, || Ok(()))?;
        drop(transformer);
        mlx_rs::memory::clear_cache();

        on_progress(Progress::Decoding);
        let frames = self.decode_video(req, &video_latents, seed)?;
        let images = mlx_gen::image::u8_frames_to_images(&frames)?;
        // Audio always denoised (it conditions the video); decode it unless `--no-audio`.
        let audio = if Self::no_audio(req) {
            None
        } else {
            Some(decode_audio_track(
                &self.audio_decoder,
                &self.vocoder,
                &audio_latents,
                self.audio_sample_rate,
                &req.cancel,
            )?)
        };
        finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Decode, || Ok(()))?;
        Ok(GenerationOutput::Video {
            frames: images,
            fps: playback_fps,
            audio,
        })
    }
}

/// Request-local calibration fault injection at a completed physical phase boundary.
///
/// The shared request floor accepts the hidden fault carrier only when a conformance harness pairs
/// it with explicit authorization. Production requests leave it unset, so the hook is inert.
/// Returning the completed value through
/// this helper is deliberate: when the named fault fires, phase-local resources are dropped before
/// the error reaches the caller, which is the cleanup behavior the harness measures with a warm
/// follow-up request.
pub(super) fn finish_calibration_phase<T>(
    req: &GenerationRequest,
    phase: mlx_gen::gen_core::MemoryPhase,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let value = work()?;
    if req.memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    }) {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: injected memory-strategy calibration error at {phase:?}"
        )));
    }
    Ok(value)
}

impl Ltx {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    pub(super) fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
        warnings: &mut Vec<GenerationWarning>,
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        // sc-10976 staged text phase: build the Gemma TE, run the optional prompt enhancer (sc-2845 —
        // rewrites `req.prompt` before encoding; default off, falls back to the original prompt on any
        // failure, so the `enhance_prompt = false` parity seams are untouched) + the video/audio encode
        // inside one TE scope, then drop the ~24 GB Gemma before the DiT loads. Honor a cancel before
        // this (unbounded) text stage.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let conditioning =
            finish_calibration_phase(req, mlx_gen::gen_core::MemoryPhase::Conditioning, || {
                self.stage_text_phase(req)
            })?;
        // The TE is dropped; free the allocator cache so the DiT loads into the low-water footprint.
        mlx_rs::memory::clear_cache();
        if let Some(reason) = &conditioning.enhancer_fallback {
            warnings.push(GenerationWarning {
                code: PROMPT_ENHANCER_FALLBACK.to_owned(),
                message: format!(
                    "prompt enhancement failed, the original prompt was used: {reason}"
                ),
            });
        }
        let prompted = request_with_effective_prompt(req, conditioning.enhanced_prompt.clone());
        let owned = self.apply_auto_frames(&prompted, &conditioning.video, &conditioning.audio)?;
        // Auto-duration supplies `frames` only after the text phase.  Re-run the ordinary provider
        // floor on that concrete request so its maximum/stride and DFR geometry bounds apply to the
        // prediction before any noise allocation.
        self.validate(&owned)?;
        let req = &owned;
        // DFR may pad the latent canvas before its temporal rounds, while the public request's
        // frame count remains the trim contract.  Allocate the four ordinary provider noises for
        // that execution canvas; `generate_av_from_embeddings` consumes the same plan below.
        let dfr_plan = self.dfr_plan(req)?;
        let noise_request = match dfr_plan.as_ref() {
            Some(plan) if plan.canvas_frames != req.frames.unwrap_or(1) => GenerationRequest {
                frames: Some(plan.canvas_frames),
                ..req.clone()
            },
            _ => req.clone(),
        };
        let (lf, h1, w1, h2, w2) = Self::latent_dims(&noise_request);
        let af = Self::audio_frames(&noise_request) as i32;
        let seed = req.seed.unwrap_or_else(default_seed);
        // Seeded noise at the path dtype (the reference seeds `normal(...).astype(model_dtype)`). RNG
        // is not portable to mlx-python, so the pixel/waveform parity gate injects the reference
        // samples via `generate_with_noise`. Distinct keys per stage/modality. I2V conditioning (when
        // a `Reference` is supplied) + the audio decode are handled inside `generate_with_noise`.
        let normal = |key: u64, shape: &[i32]| -> Result<Array> {
            let k = random::key(key)?;
            Ok(random::normal::<f32>(shape, None, None, Some(&k))?.as_dtype(self.stat_dt)?)
        };
        let video_s1 = normal(seed, &[1, LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32])?;
        let video_s2 = normal(
            seed.wrapping_add(1),
            &[1, LATENT_CHANNELS, lf as i32, h2 as i32, w2 as i32],
        )?;
        let audio_s1 = normal(
            seed.wrapping_add(2),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        let audio_s2 = normal(
            seed.wrapping_add(3),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        self.generate_with_noise(
            req,
            &conditioning.video,
            &conditioning.audio,
            conditioning.negative_video.as_ref(),
            conditioning.negative_audio.as_ref(),
            &video_s1,
            &video_s2,
            &audio_s1,
            &audio_s2,
            on_progress,
        )
    }
}
