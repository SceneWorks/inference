//! The TI2V-5B generation body.

use super::*;

impl Wan {
    /// Validate body — kept on the crate's own [`mlx_gen::Error`] so `?` on the capability check
    /// lifts transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    pub(super) fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor: size range (the advertised `min_size` = MIN_SIZE = 480 is the
        // 5B coherence floor below which the z48 vae22 renders garbage, sc-10306/sc-12636 — NOT the
        // 32-px grid stride, which `reject_off_grid` enforces below; `max_size` caps the long edge),
        // count, guidance/negative/true_cfg, sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and
        // conditioning (`Reference`/`Keyframe`).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "wan2_2_ti2v_5b: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
            if frames as usize > crate::MAX_WAN_FRAMES {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: num_frames {frames} exceeds the maximum {}",
                    crate::MAX_WAN_FRAMES
                )));
            }
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it — one
        // request means one geometry on both backends. sc-12607: the 32-px grid stride (candle rejects
        // via `is_multiple_of(SIZE_MULTIPLE)`). sc-12308: the 5B's OWN 901 120 area budget — its 32-px
        // grid makes 1280×704 the geometry it genuinely renders.
        let (dw, dh) = grid(&self.config, Ti2vProviderVae::VAE_TILING);
        reject_off_grid(MODEL_ID, req, dw, dh)?;
        reject_over_area(MODEL_ID, req, dw, dh, self.config.max_area)?;
        // sc-19571: the conditioning strengths now reach the mask builder, so their range is load
        // bearing — reject out-of-range/non-finite here rather than letting `1 − strength` become a
        // negative mask deep in the denoise.
        reject_out_of_range_strengths(MODEL_ID, req)?;
        // The TI2V mask-blend path (the `ti2v = Some(_)` branch of `generate_impl`) is entered by
        // Keyframe conditioning OR a Reference image — the contract checks below must cover both.
        let image_conditioned = !req.keyframes().is_empty() || i2v_reference(req).is_some();
        // F-074(b): a `Reference` image AND `Keyframe`s together previously fell into the keyframe
        // branch, silently dropping the Reference (undocumented precedence). The combination is
        // redundant — a Reference *is* a first-frame pin — so reject it; the first frame is
        // expressed as a Keyframe with frame_idx 0 alongside the others.
        if i2v_reference(req).is_some() && !req.keyframes().is_empty() {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: Reference and Keyframe conditioning cannot be combined (the \
                 keyframe path would silently drop the Reference) — express the first frame as a \
                 Keyframe with frame_idx 0"
                    .into(),
            ));
        }
        // F-074(a): the curated-sampler × image-conditioned-TI2V mask-blend rejection previously
        // fired in Stage 2 (denoise), AFTER the ~11 GB UMT5 load + VAE encode. Both inputs are known
        // at validate time, so reject here — the mask-blend path has no single-eval curated-sampler
        // hook. (The 14B sibling has no mask-blend path: its I2V is channel-concat, which the
        // curated solvers DO serve via `denoise_moe_curated`.)
        if is_wan_curated(req.sampler.as_deref()) && image_conditioned {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: curated samplers (euler_ancestral/heun/dpmpp_sde/ddim) are not \
                 supported with image-conditioned TI2V mask-blend — use unipc/euler/dpmpp2m"
                    .into(),
            ));
        }
        // F-015: `trim_first_frames` generates extra leading latent frames and drops them after
        // decode — but the mask-blend pins the Reference at latent frame 0 (into the discarded
        // prefix) and Keyframe indices resolve against the trim-EXTENDED grid (shifted vs the
        // delivered video). The reference (mlx_video generate_wan.py) documents trim as a 14B
        // first-frame-artifact fix and, run this way, trims its own pinned frame off — a degenerate
        // outcome, not a supported mode. Reject, mirroring the 14B's trim×I2V rejection.
        if req.trim_first_frames.unwrap_or(0) > 0 && image_conditioned {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: trim_first_frames is not supported with Reference/Keyframe \
                 conditioning (the pinned frame would be trimmed off, and keyframe positions \
                 would shift vs the delivered video)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The dense 5B pipeline (port of `generate_wan.py`'s single-model path, sc-2680) — **T2V** when
    /// no image is given, **TI2V** mask-blend when a `Reference` image is. Resolves request knobs,
    /// then **stages** the phases to bound memory: (1) UMT5 encode the prompt (+ neg, unless CFG is
    /// off); (1b, TI2V) load the z48 vae22, encode the conditioning image → `z_img`, build the
    /// first-frame mask + per-token mask, blend the noise init; (2) load the 5B DiT (merge adapters,
    /// quantize), embed the contexts, run the dense [`crate::pipeline::denoise`] (T2V) or [`denoise_ti2v`] mask-blend
    /// loop; (3) load the vae22 decoder → RGB8 frames. CFG runs with the single guidance scale.
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
        // The resolved approximate-capability plan (sc-18322). `validate` already ran the same
        // resolution at the shared floor, so this cannot fail here; resolving again is how the route
        // dispatch below gets a plan without re-reading the request and re-deriving a policy.
        //
        // Always `Exact` today — the contract refuses every approximate selection until a
        // quality-characterization artifact family exists — but the three unwired routes refuse a
        // non-exact plan **by name** rather than by omission, so the day selection becomes possible they
        // fail closed instead of silently running the exact denoise.
        let approximation = self
            .descriptor
            .capabilities
            .approximation_plan(self.descriptor.id, req)?;
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12796): the dense render is already staged (TE → DiT → z48
        // VAE, each loaded → used → dropped in turn), so there is no expert swap. Under `Sequential`,
        // `clear_cache`-flush each dead component's buffers off-GPU as soon as it is dropped, so the
        // freed UMT5 TE / VAE-encoder don't linger in MLX's buffer cache (RSS / wired footprint) through
        // the denoise + decode that follow. `Resident` (default) leaves the cache warm — the
        // byte-identical pre-offload path (residency/lifetime change only, numerics untouched).
        let sequential = self.offload_policy == OffloadPolicy::Sequential
            || crate::i2v_memory_strategy::staged(req);

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let vae_stride = provider_vae_stride(Ti2vProviderVae::VAE_TILING);
        let trim_out = trim * vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim_out;
        let (width, height) = resolve_capped_dims(req, cfg, Ti2vProviderVae::VAE_TILING);
        let (steps, shift, kind, seed) =
            resolve_sampler_knobs(req, cfg.sample_steps, cfg.sample_shift);
        // The 5B is dense → a single guidance scale (config Single(5.0), overridable per request).
        let guidance = cfg.sample_guide_scale.resolve_single(req.guidance);
        let cfg_disabled = guidance <= 1.0;
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, vae_stride)?;

        // sc-4986 — fail fast (catchable) if the DiT-denoise stage won't fit, before any heavy load.
        // sc-12796: this is policy-INdependent for the dense 5B (unlike the A14B's `moe_denoise_resident_bytes`,
        // whose resident-expert count is 1 under the swap vs 2 otherwise). The 5B holds exactly ONE DiT
        // resident during denoise under both `Resident` and `Sequential` — the TE/VAE are staged out
        // either way; `Sequential` only additionally `clear_cache`-flushes their dead buffers — so the
        // single-DiT byte count is already the correct budget for both policies.
        let adapter_bytes = adapter_stack_resident_bytes(
            &self.adapters,
            if self.config.quantization.is_some() {
                AdapterResidencyMode::Additive
            } else {
                AdapterResidencyMode::Folded
            },
        )
        .ok_or_else(|| {
            Error::Msg(format!(
                "{}: cannot size every additive adapter before the denoise fit gate",
                self.descriptor.id
            ))
        })?;
        preflight_denoise_memory_guard(
            self.descriptor.id,
            dit_resident_bytes(&[self.root.join("model.safetensors")], self.quant)
                .saturating_add(adapter_bytes),
            seq_len(lat, cfg.patch_size),
            cfg.dim,
            !cfg_disabled,
        )?;

        // sc-4998 — pick the memory-budgeted z48 vae22 decode tiling now, so an over-budget decode
        // (the 60 GB / 4.3 min blowup the px-threshold `auto` produced at moderate res) fails
        // catchably *before* the heavy denoise rather than OOM-ing in the post-loop decode stage.
        // sc-5039 — the decode runs **bf16** (visually lossless, cosine 0.999954 real-weight; lower
        // peak ⇒ the budget fits bigger tiles), so the plan uses the bf16 cost coefficient.
        let out_frames = 1 + (lat[1] - 1) * Ti2vProviderVae::VAE_TILING.temporal_scale;
        let out_height = lat[2] * Ti2vProviderVae::VAE_TILING.spatial_scale;
        let out_width = lat[3] * Ti2vProviderVae::VAE_TILING.spatial_scale;
        let decode_tiling =
            dense_decode_tiling(req, out_width as u32, out_height as u32, out_frames as u32)?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let (context, context_null) = encode_text_staged_for_tier(
            &self.root,
            cfg,
            &req.prompt,
            &neg_prompt,
            cfg_disabled,
            self.quant,
        )?;
        // sc-12796: `encode_text_staged` already drops the UMT5 encoder before returning (only the small
        // `[L, dim]` contexts survive — the DiT projects them through its own `text_embedding` below).
        // Under `Sequential`, flush its ~11 GB bf16 weights (f32 activations — text_encoder.rs) out of
        // MLX's buffer cache to the OS now, so the TE is truly off-GPU for the whole denoise + decode.
        // `Resident` leaves the cache warm (byte-identical pre-offload behavior). `context`/`context_null`
        // are already eval'd.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // Seeded init noise (f32) — shape matches the reference; exact RNG values differ across the
        // mlx-python/mlx-rs split (expected).
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (TI2V only): encode the conditioning image + build the mask-blend init ---
        // A `Reference` image → z48-VAE-encode to `z_img [z,1,h,w]`, build the first-frame mask
        // (`[z,T,h,w]`, 0 at frame 0) + per-token mask (`[1,L]`), and blend `(1−mask)·z_img +
        // mask·noise`. Without an image this is pure-noise T2V.
        // Channels-first `[z,1,h,w]` latent for one preprocessed TI2V image (z48-VAE-encode → reshape).
        let encode_kf = |vae: &Ti2vProviderVae, image: &Image| -> Result<Array> {
            let img_thwc = preprocess_ti2v_image(image, width, height)?; // [1,1,H,W,3]
            let z = vae.encode(&img_thwc)?; // [1,1,h,w,z]
            Ok(z.reshape(&z.shape()[1..])?.transpose_axes(&[3, 0, 1, 2])?) // [z,1,h,w]
        };
        let (t_lat, h_lat, w_lat) = (lat[1] as usize, lat[2] as usize, lat[3] as usize);
        let keyframes = req.keyframes();
        // **The seam** (sc-19571). Every conditioning strength the request carries is resolved into
        // `(latent_frame, strength)` pins HERE, once, and the mask builder below is driven by these
        // and by nothing else — so a strength cannot be "accepted at the boundary and defaulted on
        // the way in" the way it was when the builder took bare indices. `resolve_ti2v_pins` is pure
        // and unit-tested against non-default strengths (`ti2v_pins_carry_*`).
        let pins = resolve_ti2v_pins(req, t_lat)?;
        let (latents_init, ti2v) = if pins.is_empty() {
            (init_noise.clone(), None)
        } else {
            // Wan-native first_last_frame / multi-keyframe (sc-3357) or a single `Reference` image:
            // encode the conditioning frame(s), scatter them into the clean latent, then mask-blend.
            // The VAE encoder is scoped to this block so it drops before the mask build (sc-12796).
            let z = {
                let w = Weights::from_file(self.root.join("vae.safetensors"))?;
                let vae = Ti2vProviderVae::from_weights(&w)?;
                if keyframes.is_empty() {
                    // `[z,1,h,w]` — broadcasts over T_lat in the blend (the pin is frame 0).
                    let (image, _) = i2v_reference(req).expect("pins are non-empty");
                    encode_kf(&vae, image)?
                } else {
                    // Positional with `pins`: `resolve_ti2v_pins` walks `req.keyframes()` in the same
                    // request order, so `pins[i]` is this keyframe's resolved latent frame.
                    let mut frames: Vec<(Array, usize)> = Vec::with_capacity(keyframes.len());
                    for (kf, &(idx, _)) in keyframes.iter().zip(pins.iter()) {
                        frames.push((encode_kf(&vae, kf.image)?, idx));
                    }
                    build_ti2v_keyframe_z(&frames, cfg.vae_z_dim, t_lat, h_lat, w_lat)?
                }
            };
            let (mask, mask_tokens) =
                build_ti2v_mask(&pins, cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
            let latents = ti2v_blend_init(&z, &mask, &init_noise)?;
            mlx_rs::transforms::eval([&latents, &z])?;
            (latents, Some((z, mask, mask_tokens)))
        };
        // sc-12796: the TI2V/keyframe path above loads the z48 VAE **encoder** to encode the reference
        // image(s), then drops it at the brace (`latents_init`/`z` are eval'd and independent of it).
        // Under `Sequential`, flush the VAE encoder out of the cache before the DiT loads so the denoise
        // stage's footprint is the DiT alone, not the DiT + a cache-resident VAE encoder. Pure-noise T2V
        // (`ti2v` is `None`) loaded no VAE here — nothing to flush.
        if sequential && ti2v.is_some() {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 2: load the DiT, merge adapters + quantize, embed contexts, denoise (→ freed) ---
        let latents = {
            let mut w = Weights::from_file(self.root.join("model.safetensors"))?;
            // Adapter routing (sc-2683 / sc-2393 / sc-10045), two mutually-exclusive paths:
            //   • DENSE bf16 snapshot → FOLD LoRA/LoKr into the dense weights BEFORE building (the fork
            //     order: fold, then `spec.quantize` may quantize the merged weight). No-op w/o adapters.
            //   • PRE-QUANTIZED (packed Q4/Q8) snapshot → build packed first, then install LoRA as
            //     forward-time residuals on the built DiT (base stays packed; LoKr/LoHa rejected up
            //     front, deferred to sc-10050/sc-10051).
            let prequantized = self.config.quantization.is_some();
            if !prequantized {
                self.merge_adapters(&mut w)?;
            }
            let mut dit = WanTransformer::from_weights(&w, cfg)?;
            if let Some(q) = self.quant {
                dit.quantize(q.bits(), None)?;
            }
            if prequantized {
                self.install_adapters_additive(&mut dit)?;
            }
            let ctx_cond = dit.embed_text(&context)?;
            let ctx_uncond = match &context_null {
                Some(cn) => Some(dit.embed_text(cn)?),
                None => None,
            };
            let total = steps as u32;
            // Provider-neutral benchmark boundary: model construction, adapter/quant work, and
            // context embedding above are pre-denoise preparation. Emit only when the solver is
            // ready to consume its first latent, matching Qwen and SDXL.
            mlx_gen::diagnostics::record_phase_boundary(
                mlx_gen::diagnostics::BenchmarkPhaseBoundary::DenoiseStart,
            );
            // Curated unified solver (epic 7114, sc-7121): the gen-core-only solvers route through the
            // shared `denoise_curated`; the native unipc/euler/dpmpp2m stay on `scheduler.rs` (N1). The
            // image-conditioned TI2V mask-blend (per-token timesteps + a post-step re-blend) has no
            // single-eval curated-sampler hook, so it stays native-only.
            match (&ti2v, is_wan_curated(req.sampler.as_deref())) {
                (Some(_), true) => {
                    // Unreachable via requests — `validate_impl` rejects curated × mask-blend up
                    // front (F-074a); kept as a defensive backstop for direct callers.
                    return Err(Error::Msg(
                        "wan: curated samplers (euler_ancestral/heun/dpmpp_sde/ddim) are not \
                         supported with image-conditioned TI2V mask-blend — use unipc/euler/dpmpp2m"
                            .into(),
                    ));
                }
                (Some((z_img, mask, mask_tokens)), false) => {
                    // Unwired for the denoise feature cache: the mask-blend re-mixes the conditioning
                    // latent into the trajectory after every step (sc-18322).
                    refuse_unwired_approximation(
                        self.descriptor.id,
                        "TI2V mask-blend",
                        &approximation,
                    )?;
                    let mut on_step = |i: usize| {
                        on_progress(Progress::Step {
                            current: i as u32,
                            total,
                        })
                    };
                    denoise_ti2v(
                        &dit,
                        kind,
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        z_img,
                        mask,
                        mask_tokens,
                        &req.cancel,
                        &mut on_step,
                    )?
                }
                (None, true) => {
                    // Unwired: a curated solver evaluates the model 1..N times per solver step, so
                    // "the previous step's residual" has no single meaning (sc-18322).
                    refuse_unwired_approximation(
                        self.descriptor.id,
                        "curated unified solver",
                        &approximation,
                    )?;
                    denoise_curated(
                        &dit,
                        req.sampler.as_deref().expect("is_wan_curated ⇒ Some"),
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        seed,
                        &req.cancel,
                        on_progress,
                    )?
                }
                (None, false) => {
                    let mut on_step = |i: usize| {
                        on_progress(Progress::Step {
                            current: i as u32,
                            total,
                        })
                    };
                    // The one wired route: native dense T2V, one forward per step over one
                    // transformer, with the step index in hand (sc-18322). Neither `TrunkCache` nor
                    // `TokenPruner` has a production constructor, so both are `None` in every non-test
                    // build — the plan being `Exact` is the contract-level reason, this is the
                    // mechanism-level one. The step count is only known here, which is why the
                    // inert-over-the-trajectory refusal lives in these two bridges.
                    let grid = dit.patch_grid(&latents_init);
                    let mut trunk = trunk_cache(self.descriptor.id, &approximation, steps)?;
                    let mut prune = token_pruner(
                        self.descriptor.id,
                        &approximation,
                        steps,
                        grid.0 * grid.1 * grid.2,
                    )?;
                    denoise_approx(
                        &dit,
                        kind,
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        &req.cancel,
                        &mut on_step,
                        trunk.as_mut(),
                        prune.as_mut(),
                    )?
                }
            }
        };
        // sc-12796: the DiT drops at the brace above; `latents` is fully materialized (the per-step
        // `eval` in `denoise` / `denoise_ti2v` / `denoise_curated` detached it from the DiT weights), so
        // nothing live references the transformer. Under `Sequential`, flush the dropped DiT out of the
        // cache before the z48 VAE **decoder** loads, so the decode stage doesn't stack on a
        // cache-resident DiT. On the dense 5B the DiT is small (~10 GB bf16 / ~2.9 GB q4) but the z48
        // `vae22` decode is the heavy stage, so keeping its dead buffers off-GPU is the point.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 3: z48 vae22 decode → RGB8 frames ---
        mlx_gen::diagnostics::record_phase_boundary(
            mlx_gen::diagnostics::BenchmarkPhaseBoundary::DecodeStart,
        );
        on_progress(Progress::Decoding);
        // Causal temporal decode: t_lat → 1 + (t_lat−1)·4 output frames (= gen_frames). The tiling
        // was chosen (and budget-checked) up front by `auto_tiling_budgeted` (sc-4998). sc-5039 casts
        // the decoder weights to bf16 (the conv-heavy body runs bf16, RMS_norm + denorm stay f32) —
        // `Wan22Vae` infers its compute dtype from the loaded weights. Decode-only: the conditioning
        // image was already encoded above through a separate f32 VAE load.
        let frames_u8 = {
            let mut w = Weights::from_file(self.root.join("vae.safetensors"))?;
            w.cast_all(mlx_rs::Dtype::Bfloat16)?;
            let vae = Ti2vProviderVae::from_weights(&w)?;
            decode_to_frames_22(&vae, &latents, decode_tiling.as_ref(), Some(&req.cancel))?
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
