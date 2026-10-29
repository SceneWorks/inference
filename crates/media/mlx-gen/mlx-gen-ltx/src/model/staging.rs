//! Per-generate staging: the deferred text encoder and transformer builds, the latent /
//! audio geometry and the video decode.

use super::*;

impl Ltx {
    /// Build the AudioVideo Gemma-3-12B text encoder from the resolved `gemma_dir` + the snapshot's
    /// `connector.safetensors` (sc-10976). Called per-generate and dropped (+ `clear_cache()`) before
    /// [`build_transformer`](Self::build_transformer) — mirroring Wan's `encode_text_staged` — so the
    /// ~24 GB TE and the DiT never co-reside. bf16 activations (the reference TE dtype); the backbone is
    /// selectively quantized iff `gemma_quant` is set (`None` ⇒ dense bf16, the default `…-bf16`
    /// snapshot). Identical construction to the pre-sc-10976 `load()`, only deferred.
    pub(super) fn build_text_encoder(&self) -> Result<StagedTextEncoder> {
        match &self.text_assets {
            TextAssets::Gemma3 { dir, quant } => {
                let gemma_w = Weights::from_dir(dir)?;
                let connector_w = Weights::from_file(
                    self.transformer_path
                        .parent()
                        .expect("transformer path has a parent")
                        .join("connector.safetensors"),
                )?;
                let encoder = LtxTextEncoder::from_weights_av(
                    &gemma_w,
                    &connector_w,
                    GemmaConfig::gemma_3_12b(),
                    *quant,
                    &self.config,
                    self.dit_prec.with_compute_dtype(Dtype::Bfloat16),
                )?;
                // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
                gemma_w.materialize_accessed()?;
                connector_w.materialize_accessed()?;
                Ok(StagedTextEncoder::Gemma3(Box::new(encoder)))
            }
            TextAssets::Gemma4 {
                bundle,
                connector,
                offload_policy,
                ..
            } => {
                let connector_w = Weights::from_file(connector)?;
                connector_w.materialize()?;
                Ok(StagedTextEncoder::Gemma4(Box::new(
                    Ltx25TextEncoder::from_bundle_av(
                        bundle,
                        &connector_w,
                        &self.config,
                        self.dit_prec.with_compute_dtype(Dtype::Bfloat16),
                        *offload_policy,
                    )?,
                )))
            }
        }
    }

    /// Build the AvDiT from the snapshot's `transformer.safetensors` at the load-time precision,
    /// applying any LoRA/LoKr adapters (sc-2687/sc-2393). Called per-generate AFTER the text encoder is
    /// freed (sc-10976), so the 24 GB TE and the DiT never co-reside. LoRA is a forward-time residual
    /// over the (quantized/dense) base; `pass_scales` carries one strength per distilled denoise pass.
    pub(super) fn build_transformer(&self, req: &GenerationRequest) -> Result<AvDiT> {
        let transformer_w = Weights::from_file(&self.transformer_path)?;
        let memory = req.memory.unwrap_or_default();
        let execution = if self.descriptor.id == MODEL_25_ID {
            crate::memory_strategy_2_5::validate_request_memory(
                self.memory_strategy
                    .as_ref()
                    .expect("ltx_2_5 route must retain its loaded memory contract"),
                &memory,
            )
            .map_err(Error::from)?
        } else {
            crate::memory_strategy_2_5::TransformerExecution::default()
        };
        let mut transformer = if memory.stream_transformer_blocks {
            if self.descriptor.id != MODEL_25_ID {
                return Err(Error::Unsupported(
                    "ltx: bounded transformer residency is implemented only for ltx_2_5".into(),
                ));
            }
            let window = execution
                .window_size
                .expect("validated streamed execution has a window");
            let stream = crate::block_stream::LtxBlockStream::new(
                &self.transformer_path,
                self.config.clone(),
                self.dit_prec,
                &self.adapters,
            )?;
            let transformer =
                AvDiT::from_weights_streamed(&transformer_w, &self.config, self.dit_prec, stream)?;
            transformer.set_transformer_window(window as usize)?;
            transformer
        } else {
            AvDiT::from_weights(&transformer_w, &self.config, self.dit_prec)?
        };
        // Exactly the tensors the DiT read cross the GPU-view guard (sc-22414): the whole file when
        // resident, only the non-block set when streamed — the stream guards each window itself in
        // `LtxBlockStream::materialize`. `materialize()` here would force a streamed DiT resident.
        transformer_w.materialize_accessed()?;
        if let Some(chunk) = execution.attention_chunk_size {
            transformer.set_attention_budget(
                mlx_gen::attention::AttentionBudget::from_score_elements(u64::from(chunk), true),
            );
        }
        if !memory.stream_transformer_blocks && !self.adapters.is_empty() {
            if self.descriptor.id == MODEL_25_ID {
                crate::adapters::apply_ltx25_adapters(
                    &mut transformer,
                    &self.adapters,
                    crate::pipeline::NUM_DENOISE_PASSES,
                )?;
            } else {
                crate::adapters::apply_ltx_adapters(
                    &mut transformer,
                    &self.adapters,
                    crate::pipeline::NUM_DENOISE_PASSES,
                )?;
            }
        }
        Ok(transformer)
    }

    /// Stage the text phase (sc-10976): build the Gemma TE, run the optional prompt enhancer + the
    /// video/audio encode inside its scope, `eval` so the encode completes, then let the TE drop at
    /// end of scope. Returns the (optional) enhanced prompt + the `(video_ctx, audio_ctx)` embeddings.
    /// The caller `clear_cache()`s and proceeds to the DiT, which now loads into the freed footprint.
    pub(super) fn stage_text_phase(
        &self,
        req: &GenerationRequest,
    ) -> Result<StagedTextConditioning> {
        let te = self.build_text_encoder()?;
        // Prompt enhancement (sc-2845) reuses the just-built Gemma backbone (the censored path); the
        // uncensored path loads its own. Running it here keeps BOTH TE uses inside one staged load.
        let (enhanced, enhancer_fallback) = match &te {
            StagedTextEncoder::Gemma3(te) => self.maybe_enhance(te, req),
            // Unlike the legacy 2.3 best-effort enhancer, the declared 2.5 stock-snapshot route
            // is fail-closed: an asset, version, template, or decode mismatch must not silently
            // turn an enhancement request into ordinary conditioning.
            StagedTextEncoder::Gemma4(_) => (self.maybe_enhance_gemma4(req)?, None),
        };
        let prompt = enhanced.as_deref().unwrap_or(req.prompt.as_str());
        let (ids, mask) = self.tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
        let (video_ctx, audio_ctx) = te.encode_av(&ids, &mask)?;
        let negative = if self.transformer_variant().is_dev() {
            let prompt = req
                .negative_prompt
                .as_deref()
                .unwrap_or(DEFAULT_NEGATIVE_PROMPT);
            let (ids, mask) = self.tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
            let (video, audio) = te.encode_av(&ids, &mask)?;
            Some((video, audio))
        } else {
            None
        };
        // Force the encode to complete before `te` (the ~24 GB Gemma) drops at end of scope — otherwise
        // the lazy graph would keep the TE weights alive into the DiT phase, defeating the staging.
        match &negative {
            Some((negative_video, negative_audio)) => {
                mlx_rs::transforms::eval([&video_ctx, &audio_ctx, negative_video, negative_audio])?
            }
            None => mlx_rs::transforms::eval([&video_ctx, &audio_ctx])?,
        }
        let (negative_video, negative_audio) = match negative {
            Some((video, audio)) => (Some(video), Some(audio)),
            None => (None, None),
        };
        Ok(StagedTextConditioning {
            enhanced_prompt: enhanced,
            enhancer_fallback,
            video: video_ctx,
            audio: audio_ctx,
            negative_video,
            negative_audio,
        })
    }

    pub(super) fn ltx25_execution(&self) -> Result<&Ltx25Execution> {
        match &self.execution {
            LtxExecution::Ltx25(execution) => Ok(execution),
            LtxExecution::Ltx23 => Err(Error::Unsupported(
                "ltx_2_3: the LTX-2.5 duration/DFR/DiffVAE execution components are unavailable"
                    .into(),
            )),
        }
    }

    pub(super) fn transformer_variant(&self) -> TransformerVariant {
        match &self.execution {
            LtxExecution::Ltx23 => TransformerVariant::Distilled,
            LtxExecution::Ltx25(execution) => execution.variant,
        }
    }

    /// Apply an opt-in automatic frame prediction **after** the ordinary staged text encode, where the
    /// real DurationHead receives the connector outputs it was trained on.  The returned request
    /// carries the concrete frame count, so noise allocation and every downstream plan see the
    /// prediction rather than the original empty `frames` field.
    pub(super) fn apply_auto_frames(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
    ) -> Result<GenerationRequest> {
        if req.auto_duration.is_none() {
            return Ok(req.clone());
        }
        let duration_head = &self.ltx25_execution()?.duration_head;
        apply_predicted_frames(req, req.fps.unwrap_or(24) as f32, &mut || {
            duration_head.predict_seconds(Some(video_ctx), Some(audio_ctx))
        })
    }

    pub(super) fn dfr_plan(&self, req: &GenerationRequest) -> Result<Option<DfrPlan>> {
        match &self.execution {
            LtxExecution::Ltx23 => Ok(None),
            LtxExecution::Ltx25(_) => plan_dfr_request(req),
        }
    }

    /// Latent dims `(frames, stage1_h, stage1_w, stage2_h, stage2_w)` for a request.
    pub(crate) fn latent_dims(req: &GenerationRequest) -> (usize, usize, usize, usize, usize) {
        // Precondition: `validate_request`'s `SIZE_MULTIPLE`-divisibility check runs before any generate, so the
        // integer divisions below (`h/2/SPATIAL_SCALE`, `h/SPATIAL_SCALE`) are exact and lose no rows.
        // Make that implicit dependency explicit so a future direct caller that skips validation trips
        // here in debug/test instead of silently truncating the latent grid (Info/L-E).
        debug_assert!(
            req.height.is_multiple_of(SIZE_MULTIPLE) && req.width.is_multiple_of(SIZE_MULTIPLE),
            "ltx latent_dims: {}×{} is not {SIZE_MULTIPLE}-aligned — validate_request must run first",
            req.width,
            req.height
        );
        let frames = req.frames.unwrap_or(1).max(1);
        let latent_frames = 1 + (frames as usize - 1) / TEMPORAL_SCALE as usize;
        let (h, w) = (req.height, req.width);
        (
            latent_frames,
            (h / 2 / SPATIAL_SCALE) as usize,
            (w / 2 / SPATIAL_SCALE) as usize,
            (h / SPATIAL_SCALE) as usize,
            (w / SPATIAL_SCALE) as usize,
        )
    }

    /// Audio latent-frame count for the request (`compute_audio_frames(num_frames, fps)`).
    pub(crate) fn audio_frames(req: &GenerationRequest) -> usize {
        compute_audio_frames(
            req.frames.unwrap_or(1).max(1) as usize,
            Self::fps(req) as f64,
        )
    }

    /// Output cadence used by every time-based LTX input. The request owns this value: the video
    /// RoPE grids, appended-clip RoPE grid, audio-frame count, and returned video all share it.
    pub(crate) fn fps(req: &GenerationRequest) -> f32 {
        req.fps.unwrap_or(DEFAULT_FPS as u32) as f32
    }

    /// Build the request-scoped video and audio position grids. Keeping these together makes the
    /// shared `frames / fps` time base explicit and prevents a secondary grid from falling back to
    /// LTX's 24-fps fixture default.
    pub(crate) fn position_grids(req: &GenerationRequest) -> (Array, Array, Array) {
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let fps = Self::fps(req);
        let pos1 = create_position_grid_with(
            1,
            lf,
            h1,
            w1,
            TEMPORAL_SCALE as i64,
            SPATIAL_SCALE as i64,
            fps,
            true,
        );
        let pos2 = create_position_grid_with(
            1,
            lf,
            h2,
            w2,
            TEMPORAL_SCALE as i64,
            SPATIAL_SCALE as i64,
            fps,
            true,
        );
        let audio_pos = create_audio_position_grid(1, Self::audio_frames(req));
        (pos1, pos2, audio_pos)
    }

    /// `--no-audio` toggle: `req.video_mode == "no_audio"` runs the full A/V denoise but skips the
    /// audio decode + returns `audio: None` (the reference `--no-audio`).
    pub(super) fn no_audio(req: &GenerationRequest) -> bool {
        matches!(
            req.video_mode.as_deref(),
            Some("no_audio") | Some("video_only")
        )
    }

    /// Decode through the component selected at provider load.  The DiffVAE arm deliberately
    /// calls its memory-budgeted decoder with the declared mode rather than borrowing the conv
    /// VAE's tiling selector: these are different planners over different intermediate shapes.
    pub(super) fn decode_video(
        &self,
        req: &GenerationRequest,
        latents: &Array,
        seed: u64,
    ) -> Result<Array> {
        let decode_conv = || {
            let selected_tiling = crate::memory_strategy::decode_tiling(req)?;
            decode_to_frames_with_tiling(&self.vae, latents, &req.cancel, selected_tiling.as_ref())
        };
        match &self.execution {
            LtxExecution::Ltx23 => decode_conv(),
            LtxExecution::Ltx25(execution) => match &execution.decoder {
                Ltx25Decoder::Conv => decode_conv(),
                Ltx25Decoder::Diffusion { decoder, mode } => {
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let shape = latents.shape();
                    let noise_shape = decoder.config().noise_shape(shape[2], shape[3], shape[4]);
                    let key = random::key(seed.wrapping_add(4))?;
                    let noise = random::normal::<f32>(
                        &[
                            shape[0],
                            decoder.config().out_channels,
                            noise_shape[0],
                            noise_shape[1],
                            noise_shape[2],
                        ],
                        None,
                        None,
                        Some(&key),
                    )?;
                    let pixels = decode_diffvae_budgeted(*mode, |mode| {
                        decoder.decode_budgeted(latents, &noise, mode)
                    })?;
                    to_uint8_frames(&pixels)
                }
            },
        }
    }
}
