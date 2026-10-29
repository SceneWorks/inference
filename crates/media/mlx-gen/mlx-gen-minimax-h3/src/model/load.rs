//! Provider load: component directories, text-encoder / tier reconciliation, construction and
//! the per-task DiT and text-encoder loads.

use super::*;

/// Resolve the DiT directory: the staged [`DIT_COMPONENT`] if the caller provided one, else
/// `root/transformer` (the flat upstream layout).
pub(super) fn resolve_dit_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(DIT_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        // A single file is not a component directory; fall back and let the config probe below
        // produce the actionable "missing transformer/config.json" error against the root.
        _ => root.join(DIT_COMPONENT),
    }
}

/// Resolve the text-encoder directory: the staged [`TEXT_ENCODER_COMPONENT`] if the caller provided
/// one, else `root/text_encoder`.
pub(super) fn resolve_text_encoder_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        _ => root.join(TEXT_ENCODER_COMPONENT),
    }
}

/// Validate the staged text encoder's quantization marker, if it carries one.
///
/// Unlike [`reconcile_tier`] this does **not** compare against `spec.quantize` — see
/// [`TEXT_ENCODER_COMPONENT`] for why the two components' tiers are independent. What it does check
/// is the one thing a packed component can get wrong in a way no loader can detect: the **group
/// size**.
///
/// [`mlx_gen::quant::packed_bits`] *infers* the bit width from the weight/scales column ratio
/// **assuming** [`crate::text_encoder::GROUP_SIZE`]. A component packed at a different group size
/// therefore does not fail cleanly — Mage's q8 vision tower (packed at 32, read at 64) derived
/// "bits 16" from a perfectly good artifact and was diagnosed as a bad upload (sc-15154). Reading
/// the declared group size and rejecting a mismatch by name turns that into one actionable line.
pub(super) fn reconcile_text_encoder(te_dir: &Path) -> Result<()> {
    let Some(declared) = mlx_gen::quant::packed_quant_group_size_at(te_dir)? else {
        // Dense, or packed without a declared group size — the loaders auto-detect on `.scales`
        // and `packed_bits` will reject anything that does not divide cleanly at our own width.
        return Ok(());
    };
    if declared != crate::text_encoder::GROUP_SIZE {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: the staged text encoder in {} declares quantization group_size {declared}, \
             but this engine reads packed text-encoder weights at {} — a mismatched group size does \
             not fail cleanly (the inferred bit width comes out legal-looking and wrong), so it is \
             rejected here; re-pack the tier at {}",
            te_dir.display(),
            crate::text_encoder::GROUP_SIZE,
            crate::text_encoder::GROUP_SIZE
        )));
    }
    Ok(())
}

/// Reconcile a caller's `spec.quantize` against the tier actually staged on disk.
///
/// **MiniMax-H3 never quantizes at load.** Every tier ships pre-quantized (`crate::convert`), and
/// the reason is not tidiness: quantizing the DiT at load would materialize its 66_280_430_080
/// dense bytes *and* the growing packed output at once, which is the install-time peak this model
/// cannot afford on any Mac it targets. So `spec.quantize` is an **assertion** about which tier was
/// staged, not an instruction — the mochi / LTX `split_model.json` shape, read here from the
/// `quantization` marker [`mlx_gen::quant::write_quantized_config`] writes into each tier's
/// `config.json`.
///
/// A disagreement is a hard error rather than a silent downgrade, because an unmarked packed
/// component and a genuinely dense one are indistinguishable to a loader that guesses.
pub(super) fn reconcile_tier(
    dit_dir: &Path,
    requested: Option<mlx_gen::gen_core::Quant>,
) -> Result<()> {
    let packed = mlx_gen::quant::packed_quant_bits_at(dit_dir)?;
    match (packed, requested) {
        (Some(bits), Some(q)) if bits != q.bits() => Err(Error::Msg(format!(
            "{MODEL_ID}: spec.quantize={q:?} (bits {}) disagrees with the staged tier's \
             config.json quantization marker (bits {bits}) in {} — the on-disk tier is \
             authoritative; stage the tier you asked for",
            q.bits(),
            dit_dir.display()
        ))),
        // Packed and agreed (or asserted nothing): the loader builds it packed from `.scales`.
        (Some(_), _) => Ok(()),
        (None, Some(q)) => Err(Error::Unsupported(format!(
            "{MODEL_ID}: spec.quantize={q:?} but {} carries no quantization marker — MiniMax-H3 \
             does not quantize at load (the DiT's 66_280_430_080 dense bytes plus the packed \
             output will not co-reside); stage the pre-quantized tier's `transformer` directory as \
             the '{DIT_COMPONENT}' component",
            dit_dir.display()
        ))),
        (None, None) => Ok(()),
    }
}

/// Load the generator from a `MiniMaxAI/MiniMax-H3` snapshot root.
///
/// Validates that every partition the `t2va` render reads is present — `transformer/`,
/// `text_encoder/`, `tokenizer/`, `vae/`, `audio_vae/` — and reads none of them. An absent partition
/// is a **load-time** error with the missing path named; this loader never derives a cache location
/// and never self-fetches (epic 13657).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: expected a MiniMax-H3 snapshot directory (the dir holding \
                 `transformer/`), not a single file"
            )))
        }
    };
    reject_unknown_components(
        spec,
        &[
            DIT_COMPONENT,
            TEXT_ENCODER_COMPONENT,
            crate::veda::contract::COMPONENT,
        ],
        MODEL_ID,
    )?;
    // Adapter paths are probed at load so a mistyped one fails here rather than 20 minutes into a
    // render, after the DiT is mapped. The factors themselves are read and installed per render,
    // onto the task's own DiT (sc-18724) — `t2va`/`fl2va` and `ref2va` are different checkpoints.
    for adapter in &spec.adapters {
        if !adapter.path.is_file() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: adapter {} does not exist",
                adapter.path.display()
            )));
        }
        // Two shared-spec knobs no MiniMax-H3 path can honor. Surfaced rather than silently ignored,
        // per their own contract: `pass_scales` is LTX's per-distilled-stage strength and this
        // denoise is single-pass, `moe_expert` addresses a dual-expert Wan MoE and this DiT is
        // single-stream.
        if adapter.pass_scales.is_some() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} sets per-pass scales, which only LTX's two-stage denoise \
                 has; this model runs one transformer pass per step — use a uniform `scale`",
                adapter.path.display()
            )));
        }
        if adapter.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} targets a MoE expert; this DiT is single-stream and has \
                 none",
                adapter.path.display()
            )));
        }
    }
    // The DiT is the one tiered component, so it is probed at its resolved location rather than
    // under the root — a split install has no `root/transformer` at all.
    let dit_dir = resolve_dit_dir(&root, spec);
    let dit_config = dit_dir.join("config.json");
    if !dit_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — stage the tier's `transformer` directory as the \
             '{DIT_COMPONENT}' component, or point at a snapshot root that holds `transformer/`",
            dit_config.display()
        )));
    }
    reconcile_tier(&dit_dir, spec.quantize)?;
    // `ref2va` is a first-class task of this engine, not an optional extra, so its checkpoint is
    // probed at load like every other partition. A snapshot missing it fails here, naming the path
    // — rather than 20 minutes into a render when the reference arm finally reaches for it.
    //
    // Probed as `dit_dir`'s SIBLING, not under the root: the reference DiT is tiered exactly like
    // the base one (`crate::convert` pre-quantizes `transformer/` or `transformer_ref/` alike), so
    // on a split install it sits at `{tier}/transformer_ref` and there is no `root/transformer_ref`
    // to find. See [`MiniMaxH3::task_dit_dir`], which resolves the same way at denoise time.
    let reference_dit = dit_dir.with_file_name(REFERENCE_DIT_PARTITION);
    let reference_config = reference_dit.join("config.json");
    if !reference_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — `ref2va` reads its own DiT partition, which is tiered \
             alongside the base one; stage the tier's `{REFERENCE_DIT_PARTITION}` directory next \
             to its `{BASE_DIT_PARTITION}`, or point at a snapshot root that holds \
             `{REFERENCE_DIT_PARTITION}/`",
            reference_config.display()
        )));
    }
    reconcile_tier(&reference_dit, spec.quantize)?;
    // The text encoder is the second tiered component (sc-19120), so it is probed at its resolved
    // location like the DiT and not under the root: a split install that stages a packed TE has no
    // `root/text_encoder` at all.
    let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
    let te_config = text_encoder_dir.join("config.json");
    if !te_config.is_file() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: missing {} — stage the tier's `text_encoder` directory as the \
             '{TEXT_ENCODER_COMPONENT}' component, or point at a snapshot root that holds \
             `text_encoder/`",
            te_config.display()
        )));
    }
    reconcile_text_encoder(&text_encoder_dir)?;
    for (partition, probe) in [
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("tokenizer", "tokenizer.json"),
        // The audio VAE's *constructor* arguments live only in the three FL2VA source documents;
        // the repackaged root config carries none of them (see `MiniMaxH3AudioVaeConfig`).
        ("FL2VA/audio_vae", "config.json"),
        ("FL2VA/audio_vae", "config.yaml"),
        ("FL2VA/audio_vae", "metadata.json"),
    ] {
        let path = root.join(partition).join(probe);
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: missing {} in the snapshot root {}",
                path.strip_prefix(&root).unwrap_or(&path).display(),
                root.display()
            )));
        }
    }
    let dtype = match spec.precision {
        Precision::Fp32 => Dtype::Float32,
        _ => Dtype::Bfloat16,
    };
    // Resolve the same source-aware contract used by registry preflight and retain it on the
    // generator. This reads component metadata, source headers and configs, never tensor payloads.
    let memory_strategy = crate::memory_strategy::contract_for(spec).map_err(Error::from)?;
    let memory_tier = crate::memory_strategy::numeric_tier(spec);
    Ok(Box::new(MiniMaxH3 {
        descriptor: descriptor(),
        #[cfg(test)]
        fixture: None,
        veda_component: match spec.components.get(crate::veda::contract::COMPONENT) {
            Some(WeightsSource::File(path)) => Some(path.clone()),
            _ => None,
        },
        veda_report: Default::default(),
        root,
        dit_dir,
        text_encoder_dir,
        dtype,
        adapters: spec.adapters.clone(),
        load_shape: spec.load_shape,
        memory_strategy,
        memory_tier,
    }))
}

/// Release a component's device memory for real: drop the Rust handle, then drain MLX's allocator
/// cache so the buffers go back to the system rather than migrating active → cache.
///
/// The drain is the **shared retried** one ([`mlx_gen::residency::drain_allocator_cache`]), not a
/// bare `clear_cache()`. sc-17145 measured a single sweep leaving a straggler counted as active in
/// roughly one run in five under load, because the Metal command buffer that used the weights had
/// not retired by the time `eval` returned; every phase boundary in [`MiniMaxH3::generate_impl`]
/// and [`MiniMaxH3::generate_ref2va`] hands 10–67 GB back through this function, so the straggler
/// is a whole component's worth.
pub(super) fn release<T>(component: T) {
    drop(component);
    mlx_gen::residency::drain_allocator_cache();
}

impl MiniMaxH3 {
    /// The snapshot root this generator reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The resolved DiT directory — `root/transformer` on a flat install, or the staged
    /// [`DIT_COMPONENT`] on a tiered one.
    pub fn dit_dir(&self) -> &Path {
        &self.dit_dir
    }

    /// The DiT directory a given task denoises from.
    ///
    /// `ref2va` reads its own [`REFERENCE_DIT_PARTITION`] checkpoint (sc-17149) and every tier
    /// carries both partitions side by side (sc-17150 — `crate::convert` pre-quantizes
    /// `transformer/` and `transformer_ref/` identically), so the reference partition is resolved
    /// as [`Self::dit_dir`]'s **sibling** rather than against the snapshot root. That is the only
    /// form that works on both layouts: a split install has no `root/transformer_ref` at all, and a
    /// flat one has `dit_dir == root/transformer`, whose sibling is exactly `root/transformer_ref`.
    /// `load` probes the same path, so a snapshot missing it fails at load rather than here.
    pub(super) fn task_dit_dir(&self, task: MiniMaxH3Task) -> PathBuf {
        match task.partition() {
            BASE_DIT_PARTITION => self.dit_dir.clone(),
            partition => self.dit_dir.with_file_name(partition),
        }
    }

    /// Read only the selected partition's JSON header, before any conditioner/VAE weights.
    /// The resolved tier/sibling directory is the same one the eventual DiT loader uses.
    pub(crate) fn task_dit_config(
        &self,
        task: MiniMaxH3Task,
    ) -> Result<crate::dit::MiniMaxH3DitConfig> {
        let path = self.task_dit_dir(task).join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Msg(format!("{MODEL_ID}: no task config for selected partition `{}` at {}; reimport with its config or select a partition that provides one", task.partition(), path.display()))
            } else {
                Error::Msg(format!("{MODEL_ID}: cannot read task config for selected partition `{}` at {}: {e}; verify the source and permissions", task.partition(), path.display()))
            }
        })?;
        crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
    }

    /// The block stack's precision.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Map `task`'s DiT partition and fold in every configured adapter (sc-18724).
    ///
    /// The **single** seam both denoise paths go through, so a LoRA cannot be honored on `t2va` and
    /// forgotten on `ref2va`. The install happens here rather than in [`load`] because the two tasks
    /// run different checkpoints and the DiT is mapped and released per render — and it happens
    /// *before* `JointDit::new`, whose `PrecomputeAndEvict` consumes the model. That ordering is
    /// safe: nothing in the published turbo set targets `adaln_proj`, and
    /// [`crate::adapters`] cannot reach it anyway, so the eviction lever is untouched.
    ///
    /// The install is **strict**: an unmatched target, or a spec list that matched nothing at all,
    /// fails the render here rather than producing a plausible one the LoRA barely touched. The
    /// returned report can therefore carry nothing a caller has to act on, and is dropped.
    pub(super) fn load_task_dit(&self, task: MiniMaxH3Task) -> Result<MiniMaxH3Dit> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return crate::veda_fixture::dit();
        }
        let mut dit = MiniMaxH3Dit::load_dir(self.task_dit_dir(task), self.dtype)?;
        if !self.adapters.is_empty() {
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.adapters)?;
        }
        Ok(dit)
    }

    /// The rung-4 admission for one request (sc-18662): `Some(window)` routes the render through
    /// the deferred loaders ([`MiniMaxH3Dit::load_dir_deferred`],
    /// [`MiniMaxH3TextEncoder::from_dir_deferred`]), `None` is the resident staged path.
    ///
    /// The guards mirror the contract exactly, because **declaration is not reachability in either
    /// direction**: a request the contract's `validate_selection` refuses must be refused here even
    /// when it arrives without going through a request scope, and a request the contract admits
    /// must not be silently downgraded to the resident path.
    ///
    /// That mirror is a **standing obligation on two files**, and it was broken between sc-18662 and
    /// sc-18650: the adapter guard below existed here with no counterpart in the declaration, so an
    /// adapter-carrying deferred load advertised rung 4 `Implemented`, was admitted by the safety
    /// check, opened a request scope, had `stream_transformer_blocks` set from that engagement by
    /// `MemoryProviderContract::generation_memory` — and was then refused right here. Every guard
    /// below now names the declaration clause it mirrors; adding a guard without one reopens that
    /// loop.
    ///
    /// * The generator must have been loaded at [`LoadShape::DeferredMaterialization`] — the rung's
    ///   first shared prerequisite, and half of what
    ///   [`crate::memory_strategy::streamable`] resolves `Implemented` on. On a resident load the
    ///   rung is `Missing` and streaming is a typed refusal, not a fallback.
    /// * Adapters are refused. A window rebuilds each block from the staged directory per step, so
    ///   the forward-time residual factors sc-18724 installs would be dropped from every windowed
    ///   block — a plausible render the adapter barely touched, which is the exact failure
    ///   `apply_minimax_h3_adapters`' strict matching exists to prevent. Mirrored by the other half
    ///   of [`crate::memory_strategy::streamable`], which must live in the declaration because
    ///   `MemoryRunContext` has no adapter axis for a route gate to read.
    /// * The component scope must be [`TransformerComponent::Both`] — the only declared arm.
    ///   `None` defaults to `Dit` by the shared convention (SC-15794), and `Dit` is deliberately
    ///   outside this provider's published domain: declaring the DiT alone would leave the
    ///   conditioning phase, the taller stage at every tier, with no lever (AC3).
    /// * The window size must be inside the published singleton domain
    ///   ([`crate::memory_strategy::TRANSFORMER_WINDOW_SIZE`]) — 1 is the measured residency floor
    ///   on every text-encoder tier, so any other value would advertise a lever that only trades
    ///   memory away.
    ///   (Until sc-17153 this said the parameter was *inert* above 1. It is not: the packed
    ///   encoders spread 2.17-3.12x across `[1, 5, 10, 50]`. The domain is unchanged, its reason is
    ///   not — see that constant.)
    pub(super) fn requested_transformer_window(
        &self,
        req: &GenerationRequest,
    ) -> Result<Option<u32>> {
        use mlx_gen::gen_core::LoadShape;
        let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
            return Ok(None);
        };
        if self.load_shape != LoadShape::DeferredMaterialization {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming (rung 4) requires a \
                 LoadShape::DeferredMaterialization load; this generator was loaded resident, \
                 where the rung is declared Missing"
            )));
        }
        if !self.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming rebuilds blocks from the staged tier per \
                 window, so a configured adapter would be silently dropped from every windowed \
                 block; run adapters on the resident path"
            )));
        }
        let component = memory
            .transformer_window_component
            .unwrap_or(mlx_gen::gen_core::TransformerComponent::Dit);
        if component != mlx_gen::gen_core::TransformerComponent::Both {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer streaming is declared for TransformerComponent::Both \
                 only — got {component:?}. The conditioning stage is the taller of the two at \
                 every tier, so a Dit-only window would leave the request peak untouched"
            )));
        }
        let size = memory
            .transformer_window_size
            .unwrap_or(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE);
        if size != crate::memory_strategy::TRANSFORMER_WINDOW_SIZE {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer window {size} is outside the published domain \
                 [{}] — 1 is the measured residency floor on every text-encoder tier, so larger \
                 windows only raise the peak this rung exists to bound and are not advertised",
                crate::memory_strategy::TRANSFORMER_WINDOW_SIZE
            )));
        }
        Ok(Some(size))
    }

    /// [`Self::load_task_dit`]'s rung-4 twin: defer the 600 block tensors and hold only the I/O
    /// projections and the refiner. Adapters were already refused by
    /// [`Self::requested_transformer_window`]; the assertion keeps the two seams from drifting.
    pub(super) fn load_task_dit_deferred(&self, task: MiniMaxH3Task) -> Result<MiniMaxH3Dit> {
        assert!(
            self.adapters.is_empty(),
            "{MODEL_ID}: a deferred DiT load with adapters configured — \
             requested_transformer_window must refuse this before the load is reached"
        );
        MiniMaxH3Dit::load_dir_deferred(self.task_dit_dir(task), self.dtype)
    }

    /// Map the text-encoder shards a presentation needs, out of the **resolved** text-encoder
    /// directory — so a staged packed tier ([`TEXT_ENCODER_COMPONENT`], sc-19120) is mapped instead
    /// of the dense root component.
    ///
    /// The window itself lives with the encoder ([`crate::text_encoder::map_shards`]), which is
    /// also what the tier measurement drives, so a per-stage memory figure is taken from this exact
    /// mapping rather than from a copy of it.
    pub(super) fn map_te_shards(&self, with_vision: bool) -> Result<Weights> {
        crate::text_encoder::map_shards(&self.text_encoder_dir, with_vision)
    }
}
