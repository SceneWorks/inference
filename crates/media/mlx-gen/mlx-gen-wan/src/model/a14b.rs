//! The A14B MoE providers (T2V and I2V): load-time quant, decode tiling and construction.

use super::*;

/// The loaded Wan2.2 T2V-A14B. Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the two 14B experts, the z16 VAE) are **staged** inside
/// [`Wan14b::generate`] — loaded, used, then dropped in turn — to bound peak memory (mirrors
/// `generate_wan.py`, which never holds the T5 encoder and both 27 GB experts resident at once).
pub struct Wan14b {
    pub(super) descriptor: ModelDescriptor,
    pub(super) config: WanModelConfig,
    pub(super) root: PathBuf,
    /// LoRA adapters merged onto the experts at generate time (sc-2683). Empty for a plain load;
    /// `moe_expert`-tagged specs route to the high/low expert (shared = both).
    pub(super) adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer experts (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest —
    /// see [`resolve_load_time_quant`]). When `Some`, [`Wan14b::generate`] quantizes **each** expert
    /// independently after load (transformer-only: attn `q/k/v/o` + `ffn.fc1/fc2`; T5 + VAE stay f32).
    pub(super) quant: Option<Quant>,
    /// Component-residency strategy for the denoise (epic 12732, sc-12736). [`OffloadPolicy::Resident`]
    /// (default) holds **both** ~8-9 GB MoE experts resident for the whole loop (the byte-identical
    /// pre-swap path). [`OffloadPolicy::Sequential`] runs the **expert swap** — only the ACTIVE expert
    /// is resident (high `0..k`, then evict-then-load the low `k..steps`, never co-resident) and the
    /// UMT5 TE / VAE encoder are `clear_cache`-freed off-GPU during denoise — bounding the unified-memory
    /// peak to ~one expert. Advertised via `supports_sequential_offload` so the worker's fit-gate can
    /// select it under a memory ceiling. Captured from [`LoadSpec::offload_policy`] at load.
    pub(super) offload_policy: OffloadPolicy,
    /// Exact provider-owned public-I2V artifact and adapter receipt. Absent on direct T2V loads.
    pub(super) i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
    /// Per-adapter apply outcomes from the most recent adapter fold/install (sc-27451), published
    /// through [`Generator::adapter_apply_reports`]. Interior-mutable because the adapters apply
    /// inside `generate(&self)`.
    pub(super) adapter_reports: std::sync::Mutex<Vec<AdapterApplyReport>>,
}

impl Wan14b {
    /// The resolved model config.
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Enforce the "matched no module across either expert" error + surface the combined partial skips
    /// after an adapter apply (fold OR additive) — shared by both paths so the message can't drift. A
    /// non-empty adapter set that applied nothing across BOTH experts is a format/prefix
    /// misconfiguration; per-key skips (a target absent from this checkpoint) are surfaced as a
    /// warning, not fatal, mirroring the reference.
    pub(super) fn finalize_dual_report(
        &self,
        low: WanLoraReport,
        high: WanLoraReport,
    ) -> Result<()> {
        if low.applied + high.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} LoRA file(s) matched no module across either expert — check the format \
                 (expected PEFT `lora_A/B` or kohya `lora_down/up`, `diffusion_model.`-prefixed Wan \
                 module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        *self
            .adapter_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = adapter_apply_reports(
            &self.adapters,
            &[low.per_spec.as_slice(), high.per_spec.as_slice()],
        );
        let mut skipped = low.skipped;
        skipped.extend(high.skipped);
        skipped.sort();
        skipped.dedup();
        warn_skipped_adapters(self.descriptor.id, &skipped);
        Ok(())
    }

    /// **Dense-bf16 path.** Fold the load-time LoRA/LoKr adapters onto the two expert weight maps in
    /// place (sc-2683), before the [`WanTransformer`]s are built (the reference fold order: LoRA folds
    /// into the dense weight, then `spec.quantize` may quantize it). No-op without adapters (the
    /// no-adapter path is byte-identical). Shared specs merge onto both experts, `moe_expert`-tagged
    /// specs onto their own (the reference `(loras)+(loras_high/low)` split). Called only for a dense
    /// snapshot; a pre-quantized snapshot uses [`Wan14b::install_adapters_additive`] (sc-10045).
    pub(super) fn merge_adapters(&self, low_w: &mut Weights, high_w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        let low = merge_wan_adapters(low_w, &self.adapters, MoeExpert::Low)?;
        let high = merge_wan_adapters(high_w, &self.adapters, MoeExpert::High)?;
        self.finalize_dual_report(low, high)
    }

    /// **Pre-quantized (packed Q4/Q8) path (sc-10045).** Install the load-time LoRA adapters onto the
    /// two already-built (packed) expert [`WanTransformer`]s as forward-time residuals — the bases
    /// stay packed, never dequantized (removing the old `model.rs:614` rejection). No-op without
    /// adapters. Shared specs install onto both experts, `moe_expert`-tagged onto their own (same
    /// high/low routing as the fold path). Plain LoRA and **LoKr** both apply (LoKr via the structured
    /// deferred-Kronecker vec-trick, sc-10050); only LoHa on a packed tier is rejected up front with an
    /// actionable error (deferred to sc-10051). Called only for a pre-quantized snapshot; a dense
    /// snapshot uses [`Wan14b::merge_adapters`] instead.
    pub(super) fn install_adapters_additive(
        &self,
        low_dit: &mut WanTransformer,
        high_dit: &mut WanTransformer,
    ) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
        let low = apply_wan_adapters_additive(low_dit, &self.adapters, MoeExpert::Low)?;
        let high = apply_wan_adapters_additive(high_dit, &self.adapters, MoeExpert::High)?;
        self.finalize_dual_report(low, high)
    }

    /// Build **one** MoE expert's transformer, staged (sc-12736): load its
    /// `{high,low}_noise_model.safetensors`, apply *this* expert's adapters (dense FOLD before the
    /// build / packed ADDITIVE install after it), and quantize if requested — the per-expert half of
    /// the resident Stage-2 build, extracted so the sequential expert swap can materialize exactly one
    /// expert at a time. Returns the transformer plus its adapter report (`Some` iff adapters are
    /// configured); the caller feeds both experts' reports to [`Wan14b::finalize_dual_report`] for the
    /// cross-expert "matched no module across either expert" guard, exactly as the resident path does.
    /// The build is byte-identical to the resident per-expert build — the experts are independent
    /// weight files, so building one never touches the other (the numerics-preserving requirement).
    pub(super) fn build_expert_staged(
        &self,
        which: MoeExpert,
    ) -> Result<(WanTransformer, Option<WanLoraReport>)> {
        let cfg = &self.config;
        let file = match which {
            MoeExpert::Low => "low_noise_model.safetensors",
            MoeExpert::High => "high_noise_model.safetensors",
        };
        let mut w = Weights::from_file(self.root.join(file))?;
        let prequantized = cfg.quantization.is_some();
        let has_adapters = !self.adapters.is_empty();
        // DENSE bf16 snapshot: FOLD this expert's LoRA/LoKr into the weights BEFORE building (fold,
        // then `spec.quantize` may quantize the merged weight) — mirrors `merge_adapters`, per expert.
        let fold_report = if !prequantized && has_adapters {
            Some(merge_wan_adapters(&mut w, &self.adapters, which)?)
        } else {
            None
        };
        let mut dit = WanTransformer::from_weights(&w, cfg)?;
        if let Some(q) = self.quant {
            dit.quantize(q.bits(), None)?;
        }
        // PRE-QUANTIZED (packed Q4/Q8) snapshot: install this expert's adapters as forward-time
        // residuals AFTER building (the bases stay packed) — mirrors `install_adapters_additive`.
        let report = if prequantized && has_adapters {
            reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
            Some(apply_wan_adapters_additive(
                &mut dit,
                &self.adapters,
                which,
            )?)
        } else {
            fold_report
        };
        Ok((dit, report))
    }

    /// The **sequential-offload** dual-expert denoise (sc-12736, epic 12732) — the staged twin of the
    /// resident [`denoise_moe`] that keeps only the ACTIVE ~8-9 GB expert resident, dropping the
    /// unified-memory denoise peak to ~one expert on a Mac ceiling. High runs steps `0..k`
    /// (`t ≥ boundary_timestep`); then, via [`staged_expert_swap`]'s **evict-then-load** discipline, the
    /// high expert is dropped and its buffers `clear_cache`-flushed **before** the low expert loads,
    /// which runs `k..steps`. Flow-match timesteps decrease monotonically ⇒ exactly one boundary
    /// crossing ⇒ at most one swap; the two experts are **never co-resident** (the Pillar-1 win).
    ///
    /// **Parity:** one continuous scheduler advances across the swap and each expert drives the
    /// identical per-step [`denoise_range`] the resident [`denoise_moe`] drives, so the prefix/suffix
    /// split reproduces the resident per-step `t ≥ boundary` choice bit-for-bit — only residency /
    /// lifetime changes. Each expert projects the raw UMT5 `context`/`context_null` through its **own**
    /// `text_embedder` after it loads (the entangled per-expert projection), so the small raw contexts
    /// stay resident across the swap. Curated solvers use a separate per-eval swap because Heun and
    /// DPM++ SDE may cross the boundary during a sub-evaluation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn denoise_moe_swapped(
        &self,
        context: &Array,
        context_null: &Array,
        low_gs: f32,
        high_gs: f32,
        kind: SolverKind,
        num_train_timesteps: usize,
        steps: usize,
        shift: f32,
        boundary_timestep: f32,
        init_noise: &Array,
        y: Option<&Array>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // sc-2957 compiled elementwise glue, active for the whole (both-expert) denoise — see
        // `denoise_moe`. Scoped + restored on drop by the RAII guard.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();
        let mut sched = make_scheduler(kind, num_train_timesteps);
        sched.set_timesteps(steps, shift);
        let timesteps: Vec<f32> = sched.timesteps().to_vec();
        // High runs `0..k` (`t ≥ boundary`), low runs `k..steps` — the single high→low crossing.
        let k = crossing_index(&timesteps, boundary_timestep);
        let mut latents = init_noise.clone();
        let total = steps as u32;

        // Each expert's adapter report, carried out of its (dropped-before-the-next-loads) load closure
        // so the cross-expert "matched no module across either expert" guard runs once both are known.
        let high_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
        let low_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);

        // Build a borrowed `Expert` (per-expert `text_embedder` projection of the raw contexts) and run
        // its step range through the shared scheduler/latents — the identical per-step math as resident.
        let denoise_expert = |dit: &WanTransformer,
                              guidance: f32,
                              range: std::ops::Range<usize>,
                              st: &mut SwapState| {
            let e = Expert {
                transformer: dit,
                ctx_cond: dit.embed_text(context)?,
                ctx_uncond: Some(dit.embed_text(context_null)?),
                guidance,
            };
            let grid = dit.patch_grid(init_noise);
            let op = &mut *st.on_progress;
            let mut on_step = |i: usize| {
                op(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_range(
                &mut *st.sched,
                &e,
                grid,
                y,
                &mut *st.latents,
                &timesteps,
                range,
                cancel,
                &mut on_step,
            )
        };

        let mut state = SwapState {
            sched: &mut *sched,
            latents: &mut latents,
            on_progress: &mut *on_progress,
        };
        staged_expert_swap(
            k,
            steps,
            &mut state,
            // load high
            |st| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (dit, report) = self.build_expert_staged(MoeExpert::High)?;
                high_report.set(report);
                Ok(dit)
            },
            // use high over `0..k`
            |dit, st| denoise_expert(dit, high_gs, 0..k, st),
            // load low
            |st| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (dit, report) = self.build_expert_staged(MoeExpert::Low)?;
                low_report.set(report);
                // Both experts' reports are now known → run the cross-expert zero-match guard before
                // the low denoise (fail fast). An expert that owned no steps (didn't load) contributes
                // a default (applied 0) report, so the guard still requires SOME loaded expert to have
                // matched. No-op without adapters. (For the A14B the schedule always spans ~999→0, so
                // both experts always own steps — `k` is strictly inside `0..steps`.)
                if !self.adapters.is_empty() {
                    self.finalize_dual_report(
                        low_report.take().unwrap_or_default(),
                        high_report.take().unwrap_or_default(),
                    )?;
                }
                Ok(dit)
            },
            // use low over `k..steps`
            |dit, st| denoise_expert(dit, low_gs, k..steps, st),
            // evict: flush the just-dropped expert's buffers to the OS BEFORE the next loads.
            || {
                mlx_rs::memory::clear_cache();
                Ok(())
            },
        )?;
        Ok(latents)
    }
}

/// The mutable denoise state threaded through [`staged_expert_swap`] (sc-12736): the continuous
/// scheduler, the in-place latents, and the progress sink. Held as `&mut` references so exclusive
/// access moves between the load/use closures via the `&mut SwapState` param (disjoint-field borrows)
/// rather than being captured by each closure — the borrow-checker-clean way to let the two swap
/// stages share the scheduler/latents/progress without a `RefCell`.
pub(super) struct SwapState<'a> {
    pub(super) sched: &'a mut dyn WanScheduler,
    pub(super) latents: &'a mut Array,
    pub(super) on_progress: &'a mut dyn FnMut(Progress),
}

/// Resolve the **load-time** quantization to apply in [`Wan14b::generate`], reconciling the requested
/// `spec.quantize` against a pre-quantized snapshot's `config.json` manifest (`cfg.quantization`).
///
/// A *pre-quantized* snapshot (manifest present) ships packed weights on disk → [`WanTransformer::
/// from_weights`] builds the experts quantized directly (the `loading.py` consume path), so **no**
/// load-time re-quantization is applied (returns `None`). A *dense bf16* snapshot honors
/// `spec.quantize` (quantized in-memory after load). A bits conflict is a hard error: the on-disk
/// manifest is authoritative, so we don't silently ignore (or re-quantize at) a different width.
/// (This is a deliberately *loud* "stored wins" — a pre-quantized snapshot at a different width is a
/// hard error here, not a silent downgrade.)
pub(super) fn resolve_load_time_quant(
    id: &str,
    cfg: &WanModelConfig,
    requested: Option<Quant>,
) -> Result<Option<Quant>> {
    match (cfg.quantization, requested) {
        (Some(stored), Some(req)) if stored.bits != req.bits() => Err(Error::Msg(format!(
            "{id}: snapshot is pre-quantized {}-bit (config.json quantization block), but \
             spec.quantize requested {}-bit — the on-disk manifest is authoritative; drop the \
             precision override or convert a snapshot at the requested width",
            stored.bits,
            req.bits()
        ))),
        // Pre-quantized snapshot: `from_weights` builds it quantized; no load-time requant.
        (Some(_), _) => Ok(None),
        // Dense bf16 snapshot: quantize at load if requested.
        (None, req) => Ok(req),
    }
}

/// The z16 decode tiling for one A14B render, from the decoded output dims.
///
/// A BoundedDecode the route's own contract admitted has to be REACHED: the sealed edge/overlap
/// pair only ever comes out of [`crate::i2v_memory_strategy::decode_tiling`], never out of the auto
/// selector. The I2V arm reaches it through its `image_to_video` mode; the T2V route has no such
/// mode, so the `tile_vae_decode` knob is the second door — otherwise the T2V provider would
/// publish a BoundedDecode contract, admit the rung, and silently decode with the auto tile
/// (sc-22738). Everything else keeps the z16 selector, which is the A14B autoencoder's own cost
/// model.
pub(crate) fn a14b_decode_tiling(
    req: &GenerationRequest,
    out_width: i32,
    out_height: i32,
    out_frames: i32,
) -> Result<Option<mlx_gen::TilingConfig>> {
    let bounded_decode = req.memory.is_some_and(|memory| memory.tile_vae_decode);
    if bounded_decode
        || (req.video_mode.as_deref() == Some("image_to_video") && req.memory.is_some())
    {
        crate::i2v_memory_strategy::decode_tiling(
            req,
            out_width as u32,
            out_height as u32,
            out_frames as u32,
        )
    } else {
        auto_tiling_budgeted_z16(out_height, out_width, out_frames)
    }
}

/// Load the Wan2.2 T2V-A14B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
/// `vae.safetensors` + `tokenizer.json` + `config.json`). LoRA adapters merge per-expert at generate
/// time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot. LoKr is the
/// sibling sc-2393.
pub fn load_t2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_t2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "wan2_2_t2v_14b: config.json is not a dual-expert model (dual_model=false, \
             model_type={}); expected the converted Wan2.2 A14B MoE checkpoint",
            config.model_type
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID_T2V_14B, &config, spec.quantize)?;
    // The T2V-A14B route publishes the same pre-load memory surface as its I2V sibling (`lib.rs`
    // registers `i2v_memory_strategy::t2v_14b::MEMORY_{REGISTRATION,FIXTURE,BEHAVIOR}`), so the
    // LOADED provider has to seal the same receipt. Without it the registry declares a contract
    // the generator never publishes: `memory_strategy_contract()` answers `None`,
    // `memory_strategy_safety_check` refuses every optimized rung with "has no prepared I2V memory
    // receipt", and `begin_memory_strategy_request` opens no scope — so a selection the video gate
    // admits dies at the consumer's scope check (sc-22738).
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_T2V_14B)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_T2V_14B).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        i2v_memory,
        adapter_reports: Default::default(),
    }))
}

/// Load the Wan2.2 I2V-A14B from a converted MLX snapshot directory (same layout as the T2V-A14B:
/// `low_noise_model` + `high_noise_model` + `t5_encoder` + `vae` (with encoder) + `tokenizer.json` +
/// `config.json`). Requires `model_type == "i2v"` (in_dim 36) and a dual-expert checkpoint. LoRA
/// adapters merge per-expert at generate time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize`
/// or a pre-quantized snapshot. LoKr is the sibling sc-2393.
pub fn load_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_i2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.is_i2v_concat() {
        return Err(Error::Msg(format!(
            "wan2_2_i2v_14b: config.json is not a channel-concat I2V model (model_type={}, \
             in_dim={}); expected the converted Wan2.2 I2V-A14B checkpoint (model_type=i2v, \
             in_dim=36)",
            config.model_type, config.in_dim
        )));
    }
    if !config.dual_model {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: config.json is not a dual-expert model (dual_model=false); expected \
             the converted Wan2.2 I2V-A14B MoE checkpoint"
                .into(),
        ));
    }
    let quant = resolve_load_time_quant(MODEL_ID_I2V_14B, &config, spec.quantize)?;
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_I2V_14B)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_I2V_14B).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan14b {
        descriptor: descriptor_i2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        i2v_memory,
        adapter_reports: Default::default(),
    }))
}
