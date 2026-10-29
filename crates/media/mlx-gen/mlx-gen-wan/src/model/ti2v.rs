//! The TI2V-5B provider: text-encoder tier, construction and its `Generator` surface.

use super::*;

/// The projection width the UMT5 text encoder packs to on a quantized tier: **Q8** (sc-12831). Q8 is
/// near-lossless for this drift-sensitive encoder — measured prompt-embedding cosine **0.9998** vs the
/// bf16 baseline on the real 5B TE, vs **0.976** at Q4 (a visible ~12° shift; the "biggest numeric-drift
/// risk" the candle sc-12757 finding flagged). Q8 already cuts the TE-encode active peak **11.83 → 7.72
/// GiB** — well under the epic's <16 GB-with-margin target — so we floor the TE at Q8 even when the DiT
/// tier is Q4: the user's Q4 *DiT* creative choice is untouched, and the TE stays regression-free (the
/// extra ~2 GiB a Q4 TE would save is not needed and not worth the drift).
pub(crate) const TE_QUANT_BITS: i32 = 8;

/// The effective UMT5 text-encoder quantization for a Wan tier (sc-12831). The DiT is packed on an
/// MLX-affine tier iff a pre-quantized snapshot manifest is present (`config.quantization`) **or** a
/// load-time `Q4`/`Q8` `spec.quantize` was requested (`Nvfp4` is candle-only and never reaches the MLX
/// Wan path — excluded so its `bits()` = 4 is not routed through `mlx quantize`). On such a tier the TE
/// packs to `TE_QUANT_BITS` (Q8), retiring the residual ~12 GiB f32-TE-encode active peak (sc-12796)
/// that no further component offload could lower — the epic's binding 5B constraint. On the bf16 tier
/// this is `None` (the TE stays dense / bit-exact). Shared by the 5B [`Wan`], the A14B [`Wan14b`], and
/// the VACE paths (`model_vace.rs`) — and, since sc-15203, by `mlx-gen-krea-realtime`, whose Krea
/// Realtime 14B tiers reuse this same stock-Wan UMT5-XXL encoder and therefore this same Q8 floor.
/// `pub` for that cross-crate reuse: the floor's rationale (and `TE_QUANT_BITS`) must live in exactly
/// one place rather than being re-derived per Wan-family provider.
pub fn effective_te_quant(config: &WanModelConfig, load_quant: Option<Quant>) -> Option<WanQuant> {
    let dit_affine_quantized =
        config.quantization.is_some() || matches!(load_quant, Some(Quant::Q4) | Some(Quant::Q8));
    dit_affine_quantized.then_some(WanQuant {
        bits: TE_QUANT_BITS,
        group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
    })
}

/// The loaded Wan2.2 TI2V-5B (dense). Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the single 5B DiT, the z48 vae22) are **staged** inside [`Wan::generate`] —
/// loaded, used, then dropped in turn — to bound peak memory (mirrors `generate_wan.py`, which never
/// holds the T5 encoder + the 10 GB transformer resident at once).
pub struct Wan {
    pub(super) descriptor: ModelDescriptor,
    pub(super) config: WanModelConfig,
    pub(super) root: PathBuf,
    /// LoRA/LoKr adapters merged onto the single dense model at generate time (the reference
    /// `_loras_single` path). Empty for a plain load. `moe_expert`-tagged specs are rejected (dense).
    pub(super) adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest).
    pub(super) quant: Option<Quant>,
    /// Component-residency strategy for the dense render (epic 12732, sc-12796). [`OffloadPolicy::Resident`]
    /// (default) is the byte-identical pre-offload path — the staged components are dropped by scope but
    /// their freed buffers stay warm in MLX's cache. [`OffloadPolicy::Sequential`] additionally
    /// `clear_cache`-flushes each dead component (UMT5 TE, TI2V VAE-encoder, DiT) out to the OS as soon
    /// as it is dropped, so it is truly off-GPU (not just off the active set) for the stages that
    /// follow — bounding the unified-memory RSS / wired footprint. No expert swap (single dense DiT).
    /// Captured from [`LoadSpec::offload_policy`] at load.
    pub(super) offload_policy: OffloadPolicy,
    /// Load-exact memory contract and numeric tier for the one calibrated plain Resident/Eager
    /// route. Other supported Wan loads remain available but deliberately publish no contract.
    pub(super) memory_strategy: Option<mlx_gen::gen_core::MemoryProviderContract>,
    pub(super) memory_tier: Option<mlx_gen::gen_core::MemoryNumericTier>,
    /// Exact provider-owned public-I2V artifact and adapter receipt, when the caller prepared one.
    pub(super) i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
    /// Per-adapter apply outcomes from the most recent adapter fold/install (sc-27451), published
    /// through [`Generator::adapter_apply_reports`]. Interior-mutable because the adapters apply
    /// inside `generate(&self)`.
    pub(super) adapter_reports: std::sync::Mutex<Vec<AdapterApplyReport>>,
}

impl Wan {
    /// The resolved model config (exposed for tests).
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Reject a `moe_expert`-tagged spec on the dense 5B (a misconfiguration — high/low tagging is
    /// only for the dual-expert A14B). The dense 5B takes only **shared** (untagged) specs — the
    /// reference's `_loras_single` (`--lora`, not `--lora-high/low`).
    pub(super) fn check_dense_specs(&self) -> Result<()> {
        if self.adapters.iter().any(|s| s.moe_expert.is_some()) {
            return Err(Error::Msg(format!(
                "{}: `moe_expert` (high/low) tagging is only for the dual-expert A14B — the dense \
                 5B takes shared (untagged) adapters",
                self.descriptor.id
            )));
        }
        Ok(())
    }

    /// Enforce the "matched no module" error + surface partial skips after an adapter apply (fold OR
    /// additive) — shared by both paths so the message can't drift. A non-empty adapter set that
    /// applied nothing is a format/prefix misconfiguration; per-key skips are surfaced, not fatal.
    pub(super) fn finalize_report(&self, report: &WanLoraReport) -> Result<()> {
        if report.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} adapter file(s) matched no module — check the format (PEFT `lora_A/B` or \
                 kohya `lora_down/up`, `diffusion_model.`-prefixed Wan module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        *self
            .adapter_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            adapter_apply_reports(&self.adapters, &[report.per_spec.as_slice()]);
        warn_skipped_adapters(self.descriptor.id, &report.skipped);
        Ok(())
    }

    /// **Dense-bf16 path.** Fold the load-time LoRA/LoKr adapters into the single dense model weight
    /// map in place, before the [`WanTransformer`] is built (the reference fold order: LoRA folds into
    /// the dense weight, then `spec.quantize` may quantize it). No-op without adapters. Reuses the
    /// sc-2683/sc-2393 [`merge_wan_adapters`] seam (`MoeExpert::High` ⇒ only the `moe_expert == None`
    /// pass fires, since all specs are untagged). Called only for a dense snapshot; a pre-quantized
    /// snapshot uses [`Wan::install_adapters_additive`] instead (sc-10045).
    pub(super) fn merge_adapters(&self, w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        self.check_dense_specs()?;
        let report = merge_wan_adapters(w, &self.adapters, MoeExpert::High)?;
        self.finalize_report(&report)
    }

    /// **Pre-quantized (packed Q4/Q8) path (sc-10045 / sc-10050).** Install the load-time adapters onto
    /// the already-built (packed) [`WanTransformer`] as forward-time residuals — the base stays packed,
    /// never dequantized. No-op without adapters. Plain LoRA and **LoKr** both apply here (LoKr via the
    /// structured deferred-Kronecker vec-trick, sc-10050 — no full delta materialized); only LoHa on a
    /// packed tier is rejected up front with an actionable error (deferred to sc-10051). Called only for
    /// a pre-quantized snapshot; a dense snapshot uses [`Wan::merge_adapters`] instead.
    pub(super) fn install_adapters_additive(&self, dit: &mut WanTransformer) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        self.check_dense_specs()?;
        reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
        let report = apply_wan_adapters_additive(dit, &self.adapters, MoeExpert::High)?;
        self.finalize_report(&report)
    }
}

/// Load the Wan2.2 TI2V-5B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `model.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` + `tokenizer.json` +
/// `config.json`). The DiT runs bf16 GEMMs over an f32 residual (the S3 parity regime). Q4/Q8
/// (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot; LoRA/LoKr (sc-2683 / sc-2393)
/// merge onto the single dense model at generate time.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_ti2v_5b: expected a model directory (converted MLX snapshot), not a single file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: precision override is not wired (the DiT runs bf16 GEMMs over an f32 \
             residual stream — the parity regime)"
                .into(),
        ));
    }
    let config = WanModelConfig::from_model_dir(&root)?;
    if config.dual_model || !config.is_ti2v() {
        return Err(Error::Msg(format!(
            "wan2_2_ti2v_5b: config.json is not the dense TI2V-5B (model_type={}, dual_model={}); \
             expected the converted Wan2.2 TI2V-5B checkpoint (model_type=ti2v, dual_model=false)",
            config.model_type, config.dual_model
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID, &config, spec.quantize)?;
    let memory = crate::memory_strategy::contract_for_loaded(spec).map_err(Error::from)?;
    let (memory_strategy, memory_tier) = memory
        .map(|(contract, tier)| (Some(contract), Some(tier)))
        .unwrap_or((None, None));
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        memory_strategy,
        memory_tier,
        i2v_memory,
        adapter_reports: Default::default(),
    }))
}

// Hand-written so the load-exact SC-19236 contract is a property of the loaded provider. The
// descriptor/validation/generation arms are otherwise the same delegation emitted by
// `impl_generator!` for the other Wan routes.
impl Generator for Wan {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.adapter_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        // SC-20686 Metal lane: a direct `generate_impl` call unless a campaign is armed.
        mlx_gen::sc20686::observe_generation(
            &self.root,
            &req.cancel,
            self.descriptor.id,
            || mlx_gen::sc20686::RequestFacts::from_request(req),
            on_progress,
            |on_progress| self.generate_impl(req, on_progress),
        )
        .map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory
            .as_ref()
            .map(|prepared| &prepared.contract)
            .or(self.memory_strategy.as_ref())
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        if let Some(prepared) = &self.i2v_memory {
            return crate::i2v_memory_strategy::safety_check(prepared, context);
        }
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                mlx_gen::gen_core::MemorySafetyDecision::Accept
            } else {
                mlx_gen::gen_core::MemorySafetyDecision::Reject {
                    reason: format!(
                        "{MODEL_ID} loaded route has no calibrated memory-strategy contract"
                    ),
                }
            };
        };
        crate::memory_strategy::safety_check(contract, tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        if let Some(prepared) = &self.i2v_memory {
            return crate::i2v_memory_strategy::begin_request(prepared, context);
        }
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return Ok(None);
        };
        crate::memory_strategy::begin_request(contract, tier, context)
    }
}
