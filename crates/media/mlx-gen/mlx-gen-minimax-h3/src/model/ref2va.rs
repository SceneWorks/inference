//! The Ref2VA generation body and its audio-reference encoder.

use super::*;

impl MiniMaxH3 {
    /// The **`ref2va` render** — ordered multi-modal references on the `transformer_ref`
    /// checkpoint.
    ///
    /// The phase order is the same staged one `t2va` / `fl2va` use, for the same memory reason, but
    /// it has **three** heavy phases rather than two: the 66.7 GB conditioner, then the VAEs, then
    /// the 66 GB `transformer_ref`. Each is released before the next is mapped, and the DiT is
    /// mapped last so it is never resident alongside the conditioner. sc-17151 owns making that
    /// residency contract enforced rather than merely observed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_ref2va(
        &self,
        req: &GenerationRequest,
        references: &Ref2VaReferences,
        task: MiniMaxH3Task,
        geometry: &RequestGeometry,
        schedule: &JointSchedule,
        seed: u64,
        window: Option<u32>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        use crate::conditioning::{
            encode_reference_condition, keyframe_condition_rows, reference_audio_rows,
            reference_clip_to_vae_pixels, KeyframeNoise,
        };
        use crate::dit::positions::ReferenceLatentGeometry;
        use crate::pipeline::{prepend_condition_audio_rows, ref2va_layout_with_audio_channels};

        let patch = self.task_dit_config(task)?.patch_size;

        // --- 0. normalize every reference onto the model's own rates and resolutions ------------
        // References do NOT bind the canvas: the geometry was already resolved (16:9 by default)
        // and each reference is put on its own resolution here.
        //
        // The image short edge is the request's effective one (sc-23402) — validated in
        // `validate_request` before any weight was read, so this resolve cannot admit a value the
        // gate refused. It sizes the reference and its latent rows only; the canvas above is
        // untouched by it.
        let reference_short_edge = effective_reference_image_short_edge(req) as i32;
        // sc-26691: the image resize mode (default `match`) and the upstream soundtrack bound.
        let resize_mode = mlx_gen::gen_core::effective_reference_image_resize_mode(req);
        let num_frames = geometry.joint.num_frames as usize;
        let truncate = |a: &mlx_gen::media::AudioTrack| {
            gen_core_h3::truncate_reference_audio(a, num_frames, crate::denoise::MINIMAX_H3_FPS)
        };
        let mut normalized: Vec<Ref2VaReference> = Vec::with_capacity(references.len());
        for r in references.as_slice() {
            normalized.push(match r {
                Ref2VaReference::Image(img) => {
                    Ref2VaReference::Image(crate::reference::resize_reference_image(
                        img,
                        SPATIAL_STRIDE as i32,
                        geometry.width,
                        geometry.height,
                        resize_mode,
                        reference_short_edge,
                    )?)
                }
                Ref2VaReference::Video(v) => Ref2VaReference::Video(VideoReference {
                    frames: crate::reference::normalize_reference_clip(
                        &v.frames,
                        v.fps,
                        geometry.joint.num_frames as usize,
                        SPATIAL_STRIDE as i32,
                        crate::pipeline::CANVAS_SHORT_EDGE as i32,
                        i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
                        crate::denoise::MINIMAX_H3_FPS,
                    )?,
                    fps: crate::denoise::MINIMAX_H3_FPS,
                    audio: v.audio.as_ref().map(truncate),
                }),
                Ref2VaReference::Audio(a) => {
                    Ref2VaReference::Audio(crate::reference::AudioReference {
                        audio: truncate(&a.audio),
                    })
                }
            });
        }
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 1. the conditioner: vision tower over the visual references, then the 66.7 GB LM ---
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let (context, text_tags) =
            self.encode_prompt_ref2va(&req.prompt, &normalized, window, &req.cancel, on_progress)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. the VAEs: one conditioning latent per visual reference, one soundtrack per -------
        //        audio-bearing reference. Done before the DiT is mapped.
        let mut condition_rows: Vec<mlx_rs::Array> = Vec::new();
        let mut audio_rows_per_ref: Vec<mlx_rs::Array> = Vec::new();
        let mut geometries: Vec<ReferenceLatentGeometry> = Vec::new();
        let mut reference_audio_channels = Vec::new();
        {
            let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
            let audio_enc = self.load_audio_encoder()?;
            for (i, r) in normalized.iter().enumerate() {
                // Audio FIRST, mirroring the packed row order.
                let num_audio_latents = if let Some(track) = r.audio() {
                    let wave = crate::pipeline::audio_track_to_encoder_input(track)?;
                    let posterior = audio_enc.encode(&wave)?;
                    // The MODE, never a sample — a soundtrack is deterministic conditioning.
                    let normed = audio_enc.normalize(posterior.mode())?;
                    // `[B, latent_channels, latents]` -> `[channels, latents, features]`.
                    let t = normed.transpose_axes(&[0, 2, 1])?;
                    reference_audio_channels.push(t.shape()[0]);
                    let latents = t.shape()[1];
                    audio_rows_per_ref.push(reference_audio_rows(&t)?);
                    latents
                } else {
                    reference_audio_channels.push(0);
                    0
                };

                let (frames, height, width) = match r {
                    Ref2VaReference::Audio(_) => (0, 0, 0),
                    Ref2VaReference::Image(img) => {
                        let pixels = reference_clip_to_vae_pixels(std::slice::from_ref(img))?;
                        let c =
                            encode_reference_condition(&vae, &pixels, &KeyframeNoise::Seeded, i)?;
                        let s = c.shape().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            patch,
                            &KeyframeNoise::Seeded,
                            i,
                        )?);
                        (s[2], s[3], s[4])
                    }
                    Ref2VaReference::Video(v) => {
                        // Snap DOWN to the VAE's `17n + 5` lattice so nothing is padded.
                        // The `.min` keeps a clip from being over-read, and is safe only because
                        // `normalize_reference_clip` refuses a normalized clip shorter than
                        // `MIN_REFERENCE_CLIP_FRAMES`; a shorter one would reach the VAE off the
                        // `17n + 5` lattice. A hard refusal, not a `debug_assert!`, so release
                        // builds keep it.
                        let keep = crate::conditioning::snap_reference_frames_down(v.frames.len())
                            .min(v.frames.len());
                        if keep < crate::reference::MIN_REFERENCE_CLIP_FRAMES {
                            return Err(Error::Msg(format!(
                                "{MODEL_ID}: reference {i} snapped to {keep} frames, below the \
                                 {}-frame `17n + 5` floor — the normalized clip reached the \
                                 encoder off the lattice",
                                crate::reference::MIN_REFERENCE_CLIP_FRAMES
                            )));
                        }
                        let pixels = reference_clip_to_vae_pixels(&v.frames[..keep])?;
                        let c =
                            encode_reference_condition(&vae, &pixels, &KeyframeNoise::Seeded, i)?;
                        let s = c.shape().to_vec();
                        condition_rows.push(keyframe_condition_rows(
                            &c,
                            patch,
                            &KeyframeNoise::Seeded,
                            i,
                        )?);
                        (s[2], s[3], s[4])
                    }
                };
                geometries.push(ReferenceLatentGeometry {
                    kind: r.kind(),
                    num_latent_frames: frames,
                    latent_height: height,
                    latent_width: width,
                    num_audio_latents,
                });
            }
            // Force BEFORE the VAEs are dropped: under lazy evaluation these rows are graph nodes
            // holding every weight they were computed from, so dropping first frees nothing.
            let mut forced: Vec<&mlx_rs::Array> = condition_rows.iter().collect();
            forced.extend(audio_rows_per_ref.iter());
            if !forced.is_empty() {
                mlx_rs::transforms::eval(forced)?;
            }
            release((vae, audio_enc));
        }
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let condition_video = (!condition_rows.is_empty())
            .then(|| mlx_rs::ops::concatenate_axis(&condition_rows, 1))
            .transpose()?;
        let condition_audio = (!audio_rows_per_ref.is_empty())
            .then(|| mlx_rs::ops::concatenate_axis(&audio_rows_per_ref, 1))
            .transpose()?;

        // --- 3. denoise on `transformer_ref` ----------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let dit = match window {
            None => self.load_task_dit(task)?,
            Some(_) => self.load_task_dit_deferred(task)?,
        };
        debug_assert_eq!(patch, dit.config().patch_size);
        let layout = ref2va_layout_with_audio_channels(
            geometry,
            &text_tags,
            &geometries,
            &reference_audio_channels,
            patch,
        )?;
        let (video_rows, audio_rows) = initial_latents(geometry, patch, seed)?;
        let video_rows = prepend_condition_rows(&layout, condition_video.as_ref(), &video_rows)?;
        let audio_rows =
            prepend_condition_audio_rows(&layout, condition_audio.as_ref(), &audio_rows)?;
        let adaln = crate::denoise::adaln_schedule(schedule)?;
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
            schedule,
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

        // --- 4. decode --------------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let frames = self.decode_video(req, &rendered.video)?;
        let audio = self.decode_audio(&rendered.audio)?;
        let audio = fit_audio_to_video(audio, geometry)?;
        if frames.len() != geometry.joint.num_frames as usize {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame ref2va request",
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

    /// The audio VAE **encoder**, built from the same three FL2VA source documents the decoder
    /// reads its constructor arguments from.
    pub(super) fn load_audio_encoder(
        &self,
    ) -> Result<crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder> {
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
        let encoder = crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::from_weights(
            &mut w,
            &cfg,
            Dtype::Float32,
        )?;
        // sc-23402: force + GPU-verify the read set (sc-22414) before the first encode.
        w.materialize_accessed()?;
        Ok(encoder)
    }
}
