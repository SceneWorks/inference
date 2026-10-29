//! The T2VA / FL2VA generation body and the video / audio decode.

use super::*;

impl MiniMaxH3 {
    pub(super) fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        *self.veda_report.borrow_mut() = mlx_gen::gen_core::VedaExecutionReport {
            requested: req.veda.as_ref().is_some_and(|v| v.enabled),
            ..Default::default()
        };
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let geometry = request_geometry(req)?;
        let evaluations = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        // `num_inference_steps` counts the terminal σ = 0, at which the model is never evaluated.
        //
        // **The video shift is a per-request knob** (sc-18729). The base checkpoint's published
        // 12.0 is the default, but the distilled turbo variants are trained against their own
        // shift and are simply wrong on 12.0 — `lightx2v/Minimax-h3-Turbo`'s own
        // `DIFFUSERS_SETUP_AND_INFERENCE.md` passes `--video-shift 6` for the 4-step 768p file,
        // and its model-specs table lists the training shifts per variant (12/3 for the 544p
        // 4-step and 8-step files, 6/3 for the 768p one). `scheduler_shift` is the shared
        // request-level home for exactly this (`mlx-gen-wan`, `mlx-gen-scail2` and
        // `mlx-gen-sensenova` already read it), so a turbo render needs no bespoke surface.
        //
        // Only the **video** shift is overridable, because only the video shift moves across the
        // published set: every documented variant keeps audio at 3.0, and the reference's
        // `--audio-shift` is a separate flag this knob is not. `AUDIO_SIGMA_SHIFT` is therefore
        // passed explicitly rather than defaulted, so the pair this call builds is visible at the
        // call site instead of hiding behind `JointSchedule::new`'s "both at their own published
        // shifts".
        let video_shift = req.scheduler_shift.unwrap_or(VIDEO_SIGMA_SHIFT);
        let schedule = JointSchedule::with_shifts(evaluations + 1, video_shift, AUDIO_SIGMA_SHIFT)?;
        if schedule.num_evals() == 0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a {evaluations}-step schedule collapsed to no model evaluations \
                 (num_inference_steps must be >= {MIN_INFERENCE_STEPS})"
            )));
        }
        let seed = req.seed.unwrap_or_else(default_seed);

        // --- 1. conditioning ----------------------------------------------------------------
        // `t2va` and `fl2va` diverge here and stay diverged through the latent prep — deliberately,
        // as the reference does. Zero keyframes is `t2va` at the WEIGHTS level (one `transformer`
        // partition) and a different block path at every other level.
        let keyframes = req.keyframes();
        let references = request_references(req)?;
        // **The checkpoint decision.** Resolved once, here, and carried to the single
        // `MiniMaxH3Dit::load` call site — the two partitions are byte-different and structurally
        // identical, so this is the only thing standing between a `ref2va` request and a plausible
        // render off the wrong 66 GB. See [`MiniMaxH3Task`].
        let task = MiniMaxH3Task::resolve(!keyframes.is_empty(), references.is_some())?;
        let patch = self.task_dit_config(task)?.patch_size;
        // The rung-4 admission (sc-18662), resolved once and carried to every heavy phase — the
        // conditioning and denoise residency must agree, or half the request streams while the
        // other half quietly holds a resident stack.
        let window = self.requested_transformer_window(req)?;
        if let Some(refs) = &references {
            return self.generate_ref2va(
                req,
                refs,
                task,
                &geometry,
                &schedule,
                seed,
                window,
                on_progress,
            );
        }
        let anchors = keyframe_anchors(&keyframes, geometry.joint.num_frames)?;
        // Fitted ONCE, here, and shared by both keyframe paths. The vision tower and the VAE
        // encode must see the same pixels — resizing separately per path would let a resampling
        // difference put the two conditioning signals fractionally out of register, which is
        // exactly the class of divergence nothing downstream can detect.
        let fitted = if anchors.is_empty() {
            Vec::new()
        } else {
            let images = keyframe_images(&keyframes, geometry.joint.num_frames)?;
            crate::keyframe::fit_keyframes(&images, geometry.width, geometry.height)?
        };

        // The two stage boundaries this model's residency seam actually has (sc-19120). H3 is a
        // load→use→drop seam through and through, so `Progress::Loading` is the event its phases
        // were specified for — `mlx-gen-krea-realtime` emits the same pair. Two things depend on
        // it: a UI that would otherwise sit silent through a multi-GB text-encoder map with no
        // `Step` to show, and `tests/te_tier_generate_stages.rs`, which resets the MLX peak at each
        // boundary and so can attribute a high-water to the stage that set it. That attribution is
        // the whole reason this story exists: for three measurements the conditioning stage's peak
        // was read as the DiT's.
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt, window, &req.cancel)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.shape()[1] as usize];
            (context, tags)
        } else {
            let refs: Vec<&mlx_gen::media::Image> = fitted.iter().collect();
            in_fl2va_phase(
                FL2VA_GROUNDED_QWEN3_VL_PHASE,
                self.encode_prompt_grounded(&req.prompt, &refs, window, &req.cancel),
            )?
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 1b. keyframe conditioning latents (fl2va) ----------------------------------------
        // The VAE encode, the second of the keyframe's two paths. Done before the DiT is mapped so
        // the ~10 GB VAE and the ~62 GB DiT are never both resident.
        let condition_rows = if anchors.is_empty() {
            None
        } else {
            in_fl2va_phase(
                FL2VA_KEYFRAME_VAE_PHASE,
                (|| {
                    let pixels: Vec<mlx_rs::Array> = fitted
                        .iter()
                        .map(crate::keyframe::keyframe_to_vae_pixels)
                        .collect::<Result<_>>()?;
                    let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
                    let rows = crate::conditioning::build_condition_rows(
                        &vae,
                        &pixels,
                        &anchors,
                        patch,
                        &crate::conditioning::KeyframeNoise::Seeded,
                    )?;
                    if let Some(r) = &rows {
                        // Force before the VAE is dropped, for the same lazy-evaluation reason the
                        // context is forced above.
                        mlx_rs::transforms::eval([r])?;
                    }
                    release((vae, pixels));
                    Ok(rows)
                })(),
            )?
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise ---------------------------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let dit = match window {
            None => self.load_task_dit(task)?,
            Some(_) => self.load_task_dit_deferred(task)?,
        };
        debug_assert_eq!(patch, dit.config().patch_size);
        let layout = if anchors.is_empty() {
            t2va_layout(&geometry, context.shape()[1], patch)?
        } else {
            fl2va_layout(&geometry, &text_tags, &anchors, patch)?
        };
        let (video_rows, audio_rows) = initial_latents(&geometry, patch, seed)?;
        // The anchors LEAD the video row stream; the scheduler then writes only the tail.
        let video_rows = prepend_condition_rows(&layout, condition_rows.as_ref(), &video_rows)?;
        let adaln = crate::denoise::adaln_schedule(&schedule)?;
        // The 26.02 GB lever: project the whole schedule's modulation and release `adaln_proj`.
        // Every timestep this run evaluates at is enumerated in `adaln`, so the eviction is safe.
        // Under rung 4 the same projection pass runs windowed instead — a deferred load never held
        // the projections, so there is nothing to evict (`JointDit::new_windowed`).
        let mut model = match window {
            None => JointDit::new(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
            )?,
            Some(w) => JointDit::new_windowed(
                dit,
                layout.clone(),
                &context,
                adaln,
                AdaLnResidency::PrecomputeAndEvict,
                w as usize,
                req.cancel.clone(),
            )?,
        };
        if let Some(request) = req.veda.as_ref().filter(|v| v.enabled) {
            #[cfg(test)]
            if let Some(fixture) = &self.fixture {
                if fixture.predictor {
                    model.prepare_veda_fixture(
                        request,
                        self.veda_report.clone(),
                        fixture.fail_self_test,
                    );
                } else {
                    model.prepare_veda(
                        request,
                        self.veda_component.as_deref(),
                        self.veda_report.clone(),
                    );
                }
            } else {
                model.prepare_veda(
                    request,
                    self.veda_component.as_deref(),
                    self.veda_report.clone(),
                );
            }
            #[cfg(not(test))]
            model.prepare_veda(
                request,
                self.veda_component.as_deref(),
                self.veda_report.clone(),
            );
        }
        release(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let mut terminal_probe = req
            .h3_terminal_telemetry
            .then(gen_core_telemetry::Collector::default);
        let mut observe_step =
            |index, elapsed| {
                if let Some(probe) = terminal_probe.as_mut() {
                    probe.record(index, elapsed, serde_json::json!({
                    "method": "mlx_process_allocator_cumulative_highwater_no_reset",
                    "allocatorActiveSampleBytes": mlx_rs::memory::get_active_memory() as u64,
                    "allocatorCacheSampleBytes": mlx_rs::memory::get_cache_memory() as u64,
                    "allocatorPeakBytesCumulative": mlx_rs::memory::get_peak_memory() as u64,
                }));
                }
            };
        let rendered = render_latents_with_observer(
            &mut model,
            &layout,
            &schedule,
            &video_rows,
            &audio_rows,
            patch,
            &req.cancel,
            &mut on_step,
            req.h3_terminal_telemetry
                .then_some(&mut observe_step as &mut dyn FnMut(usize, std::time::Duration)),
        )?;
        if let Some(probe) = terminal_probe {
            self.veda_report.borrow_mut().terminal_telemetry = Some(probe.finish("mlx"));
        }

        release((model, video_rows, audio_rows));

        // --- 3. decode ----------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let frames = self.decode_video(req, &rendered.video)?;
        let audio = self.decode_audio(&rendered.audio)?;
        let audio = fit_audio_to_video(audio, &geometry)?;
        if audio.sample_rate != AUDIO_SAMPLE_RATE || audio.channels != AUDIO_OUTPUT_CHANNELS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the decoded soundtrack is {} Hz / {} channels, expected \
                 {AUDIO_SAMPLE_RATE} / {AUDIO_OUTPUT_CHANNELS}",
                audio.sample_rate, audio.channels
            )));
        }
        if frames.len() != geometry.joint.num_frames as usize {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: crate::denoise::MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }
}

impl MiniMaxH3 {
    /// Decode the video latent, at the tiling the request's rung-2 selection admits.
    ///
    /// **The tiling is resolved BEFORE the 10.42 GB VAE is mapped**, deliberately. An unadmitted
    /// geometry is a caller error, and answering it with a file-load error (or worse, after paying
    /// the load) would hide which of the two actually failed. It is also what lets the reachability
    /// test prove this call site exists without a snapshot: the refusal fires on a root that has no
    /// weights at all.
    pub(super) fn decode_video(
        &self,
        req: &GenerationRequest,
        latents: &mlx_rs::Array,
    ) -> Result<Vec<mlx_gen::media::Image>> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::decode_video(req, latents);
        }
        let tiling = crate::pipeline::decode_tiling_for(req)?;
        let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?.with_tiling(tiling);
        let decoded = vae.decode(latents)?;
        let pixels = revert_pixel_normalization(&decoded)?;
        mlx_rs::transforms::eval([&pixels])?;
        release(vae);
        frames_to_images(&pixels)
    }

    pub(super) fn decode_audio(
        &self,
        latents: &mlx_rs::Array,
    ) -> Result<mlx_gen::media::AudioTrack> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::decode_audio(latents);
        }
        // The geometry comes from the three FL2VA source documents the reference's own
        // `from_pretrained` reads — parsed, not defaulted, so a variant snapshot cannot silently
        // run at this one's numbers. The weights come from the repackaged `audio_vae/`, which ships
        // the same tensors under the same names.
        let source = self.root.join("FL2VA").join("audio_vae");
        let read = |name: &str| -> Result<String> {
            let path = source.join(name);
            mlx_gen::gen_core::__model_input_choose!(
                std::fs::read_to_string(&path),
                mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&path)
            )
            .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        let vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)?;
        // sc-23402: force + GPU-verify the read set (sc-22414) before the decode consumes it.
        w.materialize_accessed()?;
        let track = vae.decode_audio_track(latents)?;
        release((vae, w));
        Ok(track)
    }
}
