//! The `Generator` surfaces of both models and the shared request validation.

use super::*;

impl Generator for Ltx25 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        // Reuse the full conditioning/index validation that owns the LTX request surface.  The
        // descriptor supplied here is the 2.5 descriptor, so shared capability gates stay closed
        // for any axis not yet consumed by the assembled execution route.
        validate_request_for(MODEL_25_ID, &self.descriptor.capabilities, req)
            .map_err(mlx_gen::gen_core::Error::from)?;
        if req.enhance_prompt && req.use_uncensored_enhancer {
            return Err(mlx_gen::gen_core::Error::Unsupported(
                "ltx_2_5: use_uncensored_enhancer is unsupported; only the staged stock \
                 google/gemma-4-12B-it enhancer is selectable"
                    .into(),
            ));
        }
        if req.enhance_prompt && req.enhance_temperature.is_some() {
            return Err(mlx_gen::gen_core::Error::Unsupported(
                "ltx_2_5: enhance_temperature is not selectable; Gemma-4 enhancement uses the \
                 v1.2.0 greedy generation policy with no_repeat_ngram_size=5"
                    .into(),
            ));
        }
        if let Some(memory) = req.memory.as_ref() {
            crate::memory_strategy_2_5::validate_request_memory(&self.memory_strategy, memory)?;
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // The load-time metadata checks above prove the provider is a split 2.5 route; this is the
        // matching runtime reachability seam, where its Gemma-4, split DiT, component VAE/audio and
        // spatial-upscaler are actually materialised and driven through the ordinary LTX pipeline.
        let route = build_ltx25(&self.spec, Rc::clone(&self.enhancer_cache))
            .map_err(mlx_gen::gen_core::Error::from)?;
        debug_assert_eq!(route.transformer_variant(), self.variant);
        route.generate(req, on_progress)
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationReport> {
        self.validate(req)?;
        let route = build_ltx25(&self.spec, Rc::clone(&self.enhancer_cache))
            .map_err(mlx_gen::gen_core::Error::from)?;
        debug_assert_eq!(route.transformer_variant(), self.variant);
        route.generate_with_report(req, on_progress)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy_2_5::safety_check(&self.memory_strategy, self.memory_tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy_2_5::begin_request(&self.memory_strategy, self.memory_tier, context)
    }
}

/// The request passed to every deterministic downstream stage after the text encoder has returned.
/// Replacing only `prompt` means a seed-pinned enhanced T2V or I2V request is exactly equivalent to
/// an ordinary request that supplied that enhanced string explicitly: noise keys, conditioning,
/// geometry, and every mode bit remain byte-for-byte unchanged.
pub(super) fn request_with_effective_prompt(
    req: &GenerationRequest,
    enhanced: Option<String>,
) -> GenerationRequest {
    match enhanced {
        Some(prompt) => GenerationRequest {
            prompt,
            ..req.clone()
        },
        None => req.clone(),
    }
}

/// Capability-driven request validation (weight-free, so it's unit-testable without a load): the
/// shared capability floor ([`Capabilities::validate_request`] — size range, count, unsupported
/// guidance/negative/true_cfg/sampler/scheduler, only advertised conditioning kinds) plus the LTX
/// model-specific constraints: non-empty prompt, 64-aligned width/height (stage-1 runs at //2//32),
/// `num_frames = 1 + 8·k`, and all weight-free conditioning cardinality/shape/index constraints.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    validate_request_for(MODEL_ID, caps, req)
}

pub(super) fn validate_request_for(
    model_id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{model_id}: prompt must not be empty")));
    }
    // DFR knobs (sc-18789): the 2.3 checkpoint has no learned keyframe-slot marker
    // (`use_keyframes_abs_pos_embedding: false`), so generated keyframe slots would be denoised as
    // unmarked tokens — wasted compute with no conditioning effect. Refuse up front like the
    // reference's `assert_generated_keyframes_supported`, typed so the worker distinguishes the
    // capability gap from a generic failure.
    //
    // These run BEFORE the shared floor on purpose (sc-18778). The floor now refuses the same two
    // knobs from `supports_generated_keyframes` / `max_temporal_upsample_rounds`, which this
    // descriptor leaves at their refusing defaults — so the floor would catch them anyway. Going
    // first is what keeps the more actionable message: the floor can only say "this engine does
    // not support it", while these name the checkpoint generation that does, which is what a
    // caller needs to act. Deleting them would not un-refuse the knobs, only blur the reason.
    if model_id == MODEL_ID && req.num_generated_keyframes.is_some_and(|n| n > 0) {
        return Err(Error::Unsupported(
            "ltx_2_3: num_generated_keyframes requires a generated-keyframe checkpoint \
             (use_keyframes_abs_pos_embedding, LTX >= 2.5)"
                .into(),
        ));
    }
    if model_id == MODEL_ID && req.temporal_upsample_rounds.is_some_and(|r| r > 0) {
        return Err(Error::Unsupported(
            "ltx_2_3: temporal_upsample_rounds requires the LTX-2.5 DFR pipeline (generated \
             keyframe slots + the temporal latent upsampler)"
                .into(),
        ));
    }
    caps.validate_request(model_id, req)?;
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "ltx_2_3: width/height must be divisible by {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames % 8 != 1 {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames must be 1 + 8·k (got {frames})"
            )));
        }
        if frames > MAX_FRAMES {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames {frames} exceeds the maximum {MAX_FRAMES}"
            )));
        }
    }
    // The 2.5 provider's DFR plan is the execution geometry consumed after the staged duration
    // prediction.  Validate the explicit-frame shape now so a malformed keyframe/temporal request
    // never pays for Gemma; auto-duration requests are resolved and planned at generation time.
    if model_id == MODEL_25_ID && req.frames.is_some() {
        let _ = plan_dfr_request(req)?;
    }
    let latent_frames = Ltx::latent_dims(req).0 as i32;
    let resolve_latent_index = |label: &str, idx: i32| -> Result<()> {
        let resolved = if idx < 0 { latent_frames + idx } else { idx };
        if resolved < 0 || resolved >= latent_frames {
            return Err(Error::Msg(format!(
                "ltx_2_3: {label} latent frame index {idx} out of bounds for {latent_frames} latent frames"
            )));
        }
        Ok(())
    };

    // Apply-or-reject at the weight-free boundary. Plain I2V consumes one `Reference`; the closed
    // replace_person composite is exactly one ControlClip + one 1–4 image MultiReference carrier.
    // Anything crossed is refused before the ~24 GB staged text phase, not silently discarded.
    let reference_count = req
        .conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::Reference { .. }))
        .count();
    if reference_count > 1 {
        return Err(Error::Msg(
            "ltx_2_3: multiple reference images are not supported (single-image I2V only)".into(),
        ));
    }
    let multi_references: Vec<&[Image]> = req
        .conditioning
        .iter()
        .filter_map(|c| match c {
            Conditioning::MultiReference { images } => Some(images.as_slice()),
            _ => None,
        })
        .collect();
    if multi_references.len() > 1 {
        return Err(Error::Msg(
            "ltx_2_3: replace_person accepts exactly one ordered MultiReference carrier".into(),
        ));
    }
    let control_clip_count = req
        .conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::ControlClip { .. }))
        .count();
    if control_clip_count > 1 {
        return Err(Error::Msg(
            "ltx_2_3: at most one ControlClip can be applied per request".into(),
        ));
    }
    let replace_person = control_clip_count == 1 || !multi_references.is_empty();
    if replace_person {
        if control_clip_count != 1 {
            return Err(Error::Msg(
                "ltx_2_3: replace_person requires exactly one ControlClip".into(),
            ));
        }
        let Some(images) = multi_references.first().copied() else {
            return Err(Error::Msg(
                "ltx_2_3: replace_person requires exactly one ordered MultiReference carrier"
                    .into(),
            ));
        };
        if reference_count != 0
            || req.conditioning.iter().any(|c| {
                matches!(
                    c,
                    Conditioning::Keyframe { .. } | Conditioning::VideoClip { .. }
                )
            })
        {
            return Err(Error::Msg(
                "ltx_2_3: replace_person cannot be mixed with Reference, Keyframe, or VideoClip conditioning"
                    .into(),
            ));
        }
        // This also verifies every RGB buffer and dimensions before model construction. The render
        // recomputes the same deterministic contact sheet for the VAE input.
        crate::conditioning::compose_ordered_character_references(images, req.width, req.height)?;
    }
    // F-054: range-validate every conditioning strength to [0, 1]. `strength > 1` → a negative denoise
    // mask (`1 − strength`) → negative per-token σ timesteps and extrapolating blends (silent garbage,
    // every stage "succeeds"); `< 0` (or NaN — `contains` is false for NaN) is likewise degenerate.
    // `apply_replacement_mask` clamps a local copy but the same value flows unclamped as the clip
    // strength, so reject it at the request boundary. The top-level img2img `strength` is also covered
    // by the shared floor's finiteness guard; the range check here is LTX-specific.
    let check_strength = |label: &str, s: f32| -> Result<()> {
        if !(0.0..=1.0).contains(&s) {
            return Err(Error::Msg(format!(
                "ltx_2_3: {label} strength must be in [0, 1] (got {s})"
            )));
        }
        Ok(())
    };
    if let Some(s) = req.strength {
        check_strength("img2img", s)?;
    }
    for c in &req.conditioning {
        match c {
            Conditioning::Reference {
                strength: Some(s), ..
            } => check_strength("reference", *s)?,
            Conditioning::Keyframe {
                frame_idx,
                strength,
                ..
            } => {
                resolve_latent_index("keyframe", *frame_idx)?;
                check_strength("keyframe", *strength)?;
            }
            Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            } => {
                if frames.is_empty() {
                    return Err(Error::Msg(
                        "ltx_2_3: video conditioning clip is empty".into(),
                    ));
                }
                resolve_latent_index("clip", *frame_idx)?;
                check_strength("video clip", *strength)?;
            }
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                ..
            } => {
                if frames.is_empty() {
                    return Err(Error::Msg(
                        "ltx_2_3: replace_person control clip is empty".into(),
                    ));
                }
                if frames.len() != mask.len() {
                    return Err(Error::Msg(format!(
                        "ltx_2_3: replace_person frame count {} != mask count {}",
                        frames.len(),
                        mask.len()
                    )));
                }
                resolve_latent_index("replace_person", *start_frame)?;
                if *start_frame != 0 {
                    return Err(Error::Msg(format!(
                        "ltx_2_3: replace_person ControlClip must start at latent frame 0 (got {start_frame})"
                    )));
                }
                check_strength("control clip masking", *masking_strength)?;
            }
            _ => {}
        }
    }
    Ok(())
}

// Hand-written rather than `impl_generator!`: SC-19109 makes the memory contract a property of the
// loaded provider, not only a static registry row. The ordinary descriptor/validate/generate arms
// are the same delegation the macro emitted.
impl Generator for Ltx {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        let result = if self.descriptor.id == MODEL_ID {
            validate_request(&self.descriptor.capabilities, req)
                .map_err(mlx_gen::gen_core::Error::from)
        } else {
            validate_request_for(self.descriptor.id, &self.descriptor.capabilities, req)
                .map_err(mlx_gen::gen_core::Error::from)
        };
        result?;
        if self.descriptor.id == MODEL_25_ID {
            if let (Some(contract), Some(memory)) =
                (self.memory_strategy.as_ref(), req.memory.as_ref())
            {
                crate::memory_strategy_2_5::validate_request_memory(contract, memory)?;
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress, &mut Vec::new())
            .map_err(Into::into)
    }

    /// The render plus its non-fatal warnings (sc-27451): a best-effort 2.3 prompt enhancement that
    /// fell back to the original prompt is a [`PROMPT_ENHANCER_FALLBACK`] warning, not only a
    /// stderr line.
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationReport> {
        let mut warnings = Vec::new();
        let output = self.generate_impl(req, on_progress, &mut warnings)?;
        Ok(GenerationReport {
            warnings,
            ..GenerationReport::from_output(output)
        })
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                mlx_gen::gen_core::MemorySafetyDecision::Accept
            } else {
                mlx_gen::gen_core::MemorySafetyDecision::Reject {
                    reason: "ltx_2_3 loaded route has no calibrated memory-strategy contract"
                        .into(),
                }
            };
        };
        crate::memory_strategy::safety_check(
            contract,
            tier,
            self.memory_overlay.as_deref(),
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return Ok(None);
        };
        crate::memory_strategy::begin_request(
            contract,
            tier,
            self.memory_overlay.as_deref(),
            self.config.num_layers as usize,
            context,
        )
    }
}
