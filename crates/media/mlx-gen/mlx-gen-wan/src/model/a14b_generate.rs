//! The A14B MoE `Generator` surface and generation body.

use super::*;

impl Generator for Wan14b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.adapter_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn validate(&self, request: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(request).map_err(Into::into)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        // SC-20686 Metal lane: a direct `generate_impl` call unless a campaign is armed.
        mlx_gen::sc20686::observe_generation(
            &self.root,
            &request.cancel,
            self.descriptor.id,
            || mlx_gen::sc20686::RequestFacts::from_request(request),
            on_progress,
            |on_progress| self.generate_impl(request, on_progress),
        )
        .map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory.as_ref().map(|prepared| &prepared.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        self.i2v_memory.as_ref().map_or_else(
            || {
                if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                    mlx_gen::gen_core::MemorySafetyDecision::Accept
                } else {
                    mlx_gen::gen_core::MemorySafetyDecision::Reject {
                        reason: format!(
                            "{} has no prepared I2V memory receipt",
                            self.descriptor.id
                        ),
                    }
                }
            },
            |prepared| crate::i2v_memory_strategy::safety_check(prepared, context),
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        match &self.i2v_memory {
            Some(prepared) => crate::i2v_memory_strategy::begin_request(prepared, context),
            None => Ok(None),
        }
    }
}

impl Wan14b {
    /// Validate body — kept on the crate's own [`mlx_gen::Error`] so `?` on the capability check
    /// lifts transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    pub(super) fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        let id = self.descriptor.id;
        // Shared capability floor: size range (the advertised `min_size` = patch×vae_stride = 16 is
        // the sub-tile lower bound; `max_size` caps the long edge), count, guidance/negative/true_cfg,
        // sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and conditioning (none for T2V, `Reference`
        // for I2V).
        self.descriptor.capabilities.validate_request(id, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "{id}: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
            if frames as usize > crate::MAX_WAN_FRAMES {
                return Err(Error::Msg(format!(
                    "{id}: num_frames {frames} exceeds the maximum {}",
                    crate::MAX_WAN_FRAMES
                )));
            }
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it.
        // sc-12607: the 16-px grid stride (candle rejects via `is_multiple_of(SIZE_MULTIPLE_14B)`).
        // sc-12308: the area cap (T2V was uncapped entirely; I2V refit 1280×720 → 1264×704). Both A14B
        // variants share the 14B family's 921 600 budget.
        let (dw, dh) = grid(&self.config, A14bProviderVae::VAE_TILING);
        reject_off_grid(id, req, dw, dh)?;
        reject_over_area(id, req, dw, dh, self.config.max_area)?;
        // I2V channel-concat requires a single reference image (the first conditioning frame), and
        // does not support `trim_first_frames` (the reference builds `y` from `num_frames`, so an
        // extended noise length would mismatch the conditioning's temporal dim).
        if self.config.is_i2v_concat() {
            let Some((_, strength)) = i2v_reference(req) else {
                return Err(Error::Msg(format!(
                    "{id}: image-to-video requires a Reference conditioning image"
                )));
            };
            // sc-19571 — **refuse, do not ignore.** The A14B I2V conditions by CHANNEL-CONCAT: the
            // reference is VAE-encoded into `y` and concatenated onto every forward's latent, with
            // no denoise mask anywhere in the path. There is therefore nothing for a conditioning
            // strength to weight, and the 5B's `1 − strength` mask-blend cannot be extended here
            // (sc-19504). Accepting the knob and quietly rendering a full pin is the exact failure
            // this story exists to close, so a request that asks for anything other than a full pin
            // gets a typed error naming the mechanism.
            if strength != 1.0 {
                return Err(Error::Msg(format!(
                    "{id}: conditioning strength is not supported for A14B image-to-video (got \
                     {strength}) — the reference is channel-concatenated, not mask-blended, so \
                     there is no denoise mask to weight; use strength 1.0, or wan2_2_ti2v_5b, \
                     whose mask-blend honors it"
                )));
            }
            if req.trim_first_frames.unwrap_or(0) > 0 {
                return Err(Error::Msg(format!(
                    "{id}: trim_first_frames is not supported for I2V (the conditioning `y` is built \
                     from num_frames)"
                )));
            }
        }
        Ok(())
    }

    /// The full dual-expert MoE pipeline (port of `generate_wan.py`'s dual-model path) — serves both
    /// **T2V-A14B** and **I2V-A14B** (the struct's config selects). Resolves request knobs against the
    /// config defaults, then **stages** the phases to bound memory: (1) load UMT5, encode the prompt +
    /// negative prompt, drop the encoder; (1b, I2V only) load the z16 VAE encoder, build the
    /// channel-concat conditioning `y` from the reference image, drop it; (2) load both 14B experts,
    /// embed the contexts per expert, run the boundary-switched [`denoise_moe`] loop (with `y` for
    /// I2V), drop the experts; (3) load the z16 VAE, decode to RGB8 frames. CFG runs with the
    /// per-expert (low, high) guidance.
    pub(super) fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)
                .map_err(Error::from)?;
        }
        // Every A14B route is an expert-swap route: the trajectory changes transformers at the
        // boundary, so a retained trunk residual or a rotation phase captured under the high-noise
        // expert is meaningless under the low-noise one (and retaining it would keep the outgoing
        // expert's weights alive). This provider therefore declares no approximate mechanism, and
        // refuses a non-exact plan **by name** here rather than by omission — resolving the plan is
        // what makes that refusal exist at all, and its absence was why the guard's doc claimed a
        // coverage it did not have (sc-18322).
        refuse_unwired_approximation(
            self.descriptor.id,
            "MoE expert swap",
            &self
                .descriptor
                .capabilities
                .approximation_plan(self.descriptor.id, req)?,
        )?;
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12736): free the UMT5 TE / VAE off-GPU during denoise and
        // hold only the ACTIVE MoE expert resident (the expert swap). `Resident` (default) is the
        // byte-identical pre-swap path (both experts co-resident, no per-stage `clear_cache`).
        let sequential = self.offload_policy == OffloadPolicy::Sequential
            || crate::i2v_memory_strategy::staged(req);

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // trim_first_frames: generate `trim` extra leading temporal chunks (each = vae_stride_t = 4
        // latent frames → 4 output frames after the non-causal T→4T decode) and discard them after
        // decode, so the first kept frame sees a full temporal receptive field (port of
        // generate_wan.py). gen_frames stays 1+4k since frames is and we add a multiple of 4.
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let vae_stride = provider_vae_stride(A14bProviderVae::VAE_TILING);
        let trim_out = trim * vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim * vae_stride.0;
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid, then
        // enforce the model's max-area cap (I2V-14B / TI2V-5B: 704×1280; no-op for T2V's max_area 0).
        let (width, height) = resolve_capped_dims(req, cfg, A14bProviderVae::VAE_TILING);
        let (steps, shift, kind, seed) =
            resolve_sampler_knobs(req, cfg.sample_steps, cfg.sample_shift);
        // A scalar request `guidance` overrides both experts; otherwise use the config (low, high).
        let (low_gs, high_gs) = cfg.sample_guide_scale.resolve_dual(req.guidance);
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        // Init-noise latent geometry: [z_dim, t_lat, h_lat, w_lat] for the (possibly trim-extended)
        // generation length.
        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, vae_stride)?;

        // sc-4986 — fail fast (catchable) if the DiT-denoise stage won't fit, before any heavy load.
        // sc-12736/sc-12795: every `Sequential` route keeps only ONE expert resident; `Resident`
        // budgets BOTH so an oversized both-resident job is rejected before loading. The
        // denoise always runs cond+uncond (batch 2).
        let low_bytes =
            dit_resident_bytes(&[self.root.join("low_noise_model.safetensors")], self.quant);
        let high_bytes = dit_resident_bytes(
            &[self.root.join("high_noise_model.safetensors")],
            self.quant,
        );
        let adapter_mode = if self.config.quantization.is_some() {
            AdapterResidencyMode::Additive
        } else {
            AdapterResidencyMode::Folded
        };
        let (low_adapter_bytes, high_adapter_bytes) =
            wan14b_adapter_bytes_per_expert(&self.adapters, adapter_mode).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: cannot size every additive adapter before the denoise fit gate",
                    self.descriptor.id
                ))
            })?;
        preflight_denoise_memory_guard(
            self.descriptor.id,
            wan14b_denoise_resident_bytes(
                self.offload_policy,
                req.sampler.as_deref(),
                low_bytes.saturating_add(low_adapter_bytes),
                high_bytes.saturating_add(high_adapter_bytes),
            ),
            seq_len(lat, cfg.patch_size),
            cfg.dim,
            true,
        )?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        // A14B always encodes both prompts (the dual-expert default), so `skip_neg = false`; unwrap
        // the always-present negative context back to the `Array` the MoE denoise expects.
        let (context, context_null) = encode_text_staged_for_tier(
            &self.root,
            cfg,
            &req.prompt,
            &neg_prompt,
            false,
            self.quant,
        )?;
        let context_null = context_null.expect("a14b always encodes the negative context");
        // sc-12736: `encode_text_staged` already drops the UMT5 encoder before returning (only the
        // small `[1, L, dim]` raw contexts survive — each expert projects them through its own
        // `text_embedder` below). Under `Sequential`, flush its ~11 GB f32 weights out of MLX's buffer
        // cache to the OS now, so the TE is truly off-GPU for the whole denoise. `Resident` leaves the
        // cache warm (byte-identical pre-swap behavior). `context`/`context_null` are already eval'd.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // Seeded init noise (f32, no batch dim) — matches the reference's `mx.random.normal(shape)`
        // shape; exact seeded-RNG values differ across the mlx-python/mlx-rs split (expected). I2V
        // (like the reference) starts from pure noise — the image enters via the `y` channel-concat.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (I2V only): build the channel-concat conditioning `y` (→ VAE encoder freed) ---
        // First frame = the reference image, the rest zero, VAE-encoded under a temporal mask →
        // `[20, T_lat, h_lat, w_lat]` (f32), concatenated onto each forward's noise latent in
        // `denoise_moe`. `frames` (not `gen_frames`) — validate() rejected `trim` for I2V.
        let y = if cfg.is_i2v_concat() {
            // sc-19571: the strength is deliberately NOT read here — the A14B I2V is channel-concat,
            // which has no denoise mask to weight, so `validate_impl` refuses a non-default strength
            // outright rather than accepting it and dropping it silently.
            let (image, _) = i2v_reference(req).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: image-to-video requires a Reference conditioning image",
                    self.descriptor.id
                ))
            })?;
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = A14bProviderVae::from_weights(&w)?;
            let y = build_i2v_y(&vae, image, frames, height, width, vae_stride)?;
            mlx_rs::transforms::eval([&y])?;
            Some(y)
        } else {
            None
        };
        // sc-12736: the VAE encoder (loaded only to build `y`) drops at the brace above; `y` is eval'd
        // and independent of it. Under `Sequential`, flush the VAE out of the cache before the experts
        // load so the denoise peak is one expert, not one expert + a resident VAE. No-op for T2V.
        if sequential && y.is_some() {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 2: dual-expert MoE denoise (→ freed). Under `Sequential`, native solvers swap at
        // the step boundary and curated solvers at the exact model-evaluation boundary; both keep one
        // expert resident. `Resident` keeps both experts for byte-identical legacy behavior. ---
        let latents = if sequential && is_wan_curated(req.sampler.as_deref()) {
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let low_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
            let high_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
            on_progress(Progress::Loading(LoadPhase::Renderer));
            let latents = denoise_moe_curated_swapped(
                boundary_timestep,
                req.sampler.as_deref().expect("curated sampler"),
                cfg.num_train_timesteps,
                steps,
                shift,
                &init_noise,
                y.as_ref(),
                seed,
                &req.cancel,
                on_progress,
                |high| {
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let which = if high {
                        MoeExpert::High
                    } else {
                        MoeExpert::Low
                    };
                    let (dit, report) = self.build_expert_staged(which)?;
                    let cond = dit.embed_text(&context)?;
                    let uncond = Some(dit.embed_text(&context_null)?);
                    if high {
                        high_report.set(report);
                    } else {
                        low_report.set(report);
                        if !self.adapters.is_empty() {
                            self.finalize_dual_report(
                                low_report.take().unwrap_or_default(),
                                high_report.take().unwrap_or_default(),
                            )?;
                        }
                    }
                    Ok((dit, cond, uncond, if high { high_gs } else { low_gs }))
                },
            )?;
            latents
        } else if sequential {
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            self.denoise_moe_swapped(
                &context,
                &context_null,
                low_gs,
                high_gs,
                kind,
                cfg.num_train_timesteps,
                steps,
                shift,
                boundary_timestep,
                &init_noise,
                y.as_ref(),
                &req.cancel,
                on_progress,
            )?
        } else {
            let mut low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let mut high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            // Adapter routing (sc-2683 / sc-2393 / sc-10045), two mutually-exclusive paths:
            //   • DENSE bf16 snapshot → FOLD LoRA/LoKr into each expert's dense weights BEFORE
            //     building (the fork order: fold, then `spec.quantize` may quantize the merged weight).
            //   • PRE-QUANTIZED (packed Q4/Q8) snapshot → build both experts packed first, then install
            //     LoRA as forward-time residuals per expert (bases stay packed; LoKr/LoHa rejected up
            //     front, deferred to sc-10050/sc-10051). Removes the old `model.rs:614` hard error.
            // Q4/Q8 (sc-2682) is transformer-only (attn q/k/v/o + ffn.fc1/fc2; T5 above + VAE below stay
            // f32 — the reference's quant scope), two routes:
            //   • pre-quantized snapshot (config.json `quantization` block) → `from_weights` already
            //     built the experts quantized from the on-disk packed weights (`self.quant` is None,
            //     resolved in load), so they load at the reduced ~Q4/Q8 size — the low-peak path;
            //   • dense bf16 snapshot + `spec.quantize` → quantize each expert in-memory after the fold.
            // Either way both experts are quantized independently.
            let prequantized = self.config.quantization.is_some();
            if !prequantized {
                self.merge_adapters(&mut low_w, &mut high_w)?;
            }
            let mut low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let mut high_dit = WanTransformer::from_weights(&high_w, cfg)?;
            if let Some(q) = self.quant {
                low_dit.quantize(q.bits(), None)?;
                high_dit.quantize(q.bits(), None)?;
            }
            if prequantized {
                self.install_adapters_additive(&mut low_dit, &mut high_dit)?;
            }

            // Each expert has its own text_embedding weights, so contexts are embedded per expert.
            let low = Expert {
                transformer: &low_dit,
                ctx_cond: low_dit.embed_text(&context)?,
                ctx_uncond: Some(low_dit.embed_text(&context_null)?),
                guidance: low_gs,
            };
            let high = Expert {
                transformer: &high_dit,
                ctx_cond: high_dit.embed_text(&context)?,
                ctx_uncond: Some(high_dit.embed_text(&context_null)?),
                guidance: high_gs,
            };
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            // Curated unified solver (epic 7114, sc-7121): the gen-core-only solvers route through
            // `denoise_moe_curated` (the boundary expert swap happens inside its predict closure); the
            // native unipc/euler/dpmpp2m stay on `scheduler.rs` (N1). Works for both T2V (`y = None`)
            // and I2V (`y = Some` channel-concat conditioning).
            if is_wan_curated(req.sampler.as_deref()) {
                denoise_moe_curated(
                    &low,
                    &high,
                    boundary_timestep,
                    req.sampler.as_deref().expect("is_wan_curated ⇒ Some"),
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    &init_noise,
                    y.as_ref(),
                    seed,
                    &req.cancel,
                    on_progress,
                )?
            } else {
                let mut on_step = |i: usize| {
                    on_progress(Progress::Step {
                        current: i as u32,
                        total,
                    })
                };
                denoise_moe(
                    &low,
                    &high,
                    boundary_timestep,
                    kind,
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    &init_noise,
                    y.as_ref(),
                    &req.cancel,
                    &mut on_step,
                )?
            }
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        // sc-6894 F-009 — memory-**budgeted** (catchable) z16 decode tiling from the decoded output
        // dims (t_lat·4 frames after the non-causal decode), replacing the unbudgeted
        // `TilingConfig::auto` that could pick an over-budget tile and OOM the largest-resident model.
        // `Ok(None)` for small outputs → single-pass; an over-budget decode returns a catchable error
        // here instead of a SIGKILL in the decode. decode_to_frames re-checks `needs_tiling`.
        let out_frames = lat[1] * A14bProviderVae::VAE_TILING.temporal_scale;
        let out_height = lat[2] * A14bProviderVae::VAE_TILING.spatial_scale;
        let out_width = lat[3] * A14bProviderVae::VAE_TILING.spatial_scale;
        let tiling = a14b_decode_tiling(req, out_width, out_height, out_frames)?;
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = A14bProviderVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref(), Some(&req.cancel))?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}
