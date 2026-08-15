//! `ZImageTurboControl` — the Z-Image-turbo **Fun-Controlnet-Union** variant (sc-2349 / sc-2257):
//! strict pose (VACE-style) conditioning via `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`,
//! registered as its own `Generator` (`z_image_turbo_control`).
//!
//! Identical to [`crate::model::ZImageTurbo`] except the transformer is a
//! [`ZImageControlTransformer`] (base DiT + control branch) and `generate` threads a VAE-encoded
//! control context through it. [`load`] needs the base snapshot (`spec.weights`) **and** the
//! control checkpoint (`spec.control`); it applies both dense, then quantizes the whole transformer
//! together (the fork's `d32454c` ordering — quantizing before the overlay would leave the control
//! Linears unable to accept their real weights). The control patch embedder stays dense (its
//! in-features is not divisible by the quant group size).
//!
//! Parity-proven against the frozen Python fork (sc-2257): the control branch is bit-identical to
//! the base transformer at `control_context_scale = 0`, and the full control render matches the
//! fork's control golden — see `tests/z_control_transformer.rs` and `tests/control_real_weights.rs`.

use mlx_gen::gen_core;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, require_base_dir,
    require_control, resolve_flow_schedule, Capabilities, ConditioningKind, ControlBranch, Error,
    FlowMatchEuler, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, Residency, Result, SizeFloor, StagedHeavy,
    WeightsSource,
};
use mlx_rs::Dtype;
use std::path::Path;

use crate::control_transformer::ZImageControlTransformer;
use crate::loader;
use crate::model::{validate_request, ZImageDecodeView, DEFAULT_STEPS, SCHEDULE_SHIFT};
use crate::pipeline::{
    self, denoise_control_with_progress_and_preview, encode_control_context, encode_init_latents,
    init_time_step,
};
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;

/// Registry id for the Z-Image-turbo Fun-Controlnet-Union variant.
pub const MODEL_ID: &str = "z_image_turbo_control";

/// The control variant's identity + capabilities. Same distilled turbo base (no CFG / negative
/// prompt) as `z_image_turbo`, plus `Control` conditioning (the required pose/union skeleton) and
/// `Reference` (an optional img2img init — the fork's `generate_image` accepts both).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: Some(crate::ENCODER_CONTRACT),
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::FLUX1_LATENT_SPACE),
        control_kinds: Some(crate::model_base_control::accepted_kinds()),
        required_components: &[],
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Advertised so a weights-free caller can reject a bad `kind` before
            // paying for a load. Same expression the `ControlBranch` override
            // returns, so the two cannot drift.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Control (required) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3), as the base turbo variant.
            samplers: curated_sampler_names(),
            // Curated scheduler menu (epic 7114), as the base turbo variant — static-shift default.
            schedulers: curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
            unconditionally_engages_staged_residency: false,
            supports_preview: true,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
}

/// A loaded control generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// component-residency strategy (base text encoder + control transformer + VAE), driven through the
/// shared [`Residency`] seam. SC-15806 makes residency request-scoped; the legacy
/// [`LoadSpec::offload_policy`] no longer selects the lifecycle for Z-Image.
pub struct ZImageTurboControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// The provider's half of the shared memory-strategy handshake (SC-15449 / SC-15615), built from the
    /// `LoadSpec` at load so its asset facts describe the snapshot this generator actually loaded.
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    loaded_tier: mlx_gen::gen_core::MemoryNumericTier,
    /// Request-scoped residency shared with the base control variant.
    residency: Residency<TextEncoder, ZImageControlHeavyOwned>,
}

/// The heavy render-phase components for both Z-Image ControlNet variants (the composed base+control
/// transformer and the VAE) — everything but the text encoder. There is no PiD overlay on the control
/// path (sc-7846 is base-`z_image_turbo`-only), so the seam's `use_pid` loader flag is ignored. Owned
/// by the warm pair or by a staged request. `pub(crate)` so the **base** control
/// sibling ([`crate::model_base_control`]) shares the identical bundle + seam (sc-11124).
pub(crate) struct ZImageControlHeavyOwned {
    pub(crate) transformer: ZImageControlTransformer,
    pub(crate) vae: Vae,
}

/// The light (decode-only) control bundle that survives the DiT drop during staged decode
/// (sc-13571): just the VAE (the control path has no PiD overlay). [`StagedHeavy::shed_dit`] drops the
/// control DiT so the tiled VAE decode peak excludes it.
pub(crate) struct ZImageControlLight {
    pub(crate) vae: Vae,
}

impl StagedHeavy for ZImageControlHeavyOwned {
    type Light = ZImageControlLight;
    // Reuse the base decode view; the control path never has a PiD decoder, so `pid` is always `None`.
    type DecodeView<'a> = ZImageDecodeView<'a>;
    fn shed_dit(self) -> ZImageControlLight {
        // `self.transformer` (the control DiT) drops here; the VAE moves into the light bundle.
        ZImageControlLight { vae: self.vae }
    }
    fn decode_view(&self) -> ZImageDecodeView<'_> {
        ZImageDecodeView {
            vae: &self.vae,
            pid: None,
        }
    }
    fn light_view(light: &ZImageControlLight) -> ZImageDecodeView<'_> {
        ZImageDecodeView {
            vae: &light.vae,
            pid: None,
        }
    }
}

/// Precision guard (only dense bf16 is wired) + base-snapshot-dir resolution + the **required**
/// control-checkpoint resolution, shared by [`load_control_residency`]'s `Resident` composition and its
/// `Sequential` per-phase loaders (sc-11124). Preserves the original message order: a single-file base
/// is rejected first (via [`require_base_dir`]), then a missing control checkpoint (via
/// [`require_control`]). `precision_msg` is the per-id override-rejection text (turbo vs base control).
pub(crate) fn resolve_control_base_and_control<'a>(
    spec: &'a LoadSpec,
    model_id: &str,
    precision_msg: &str,
) -> Result<(&'a Path, &'a WeightsSource)> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(precision_msg.into()));
    }
    let root = require_base_dir(spec, model_id, "a base snapshot directory")?;
    let control = require_control(spec, model_id, "Fun-Controlnet-Union")?;
    Ok((root, control))
}

/// Load the text encoder — the phase-A component dropped first under `Sequential`. Quantized with the
/// whole-model bits when `quant` is set (the Z-Image control quant scope covers the text encoder), so
/// the `Resident` and `Sequential` paths build byte-identical encoders.
pub(crate) fn load_control_text_encoder_only(
    source: &gen_core::ValidatedEncoderSource,
    load_time_quant_bits: Option<i32>,
    streamable: bool,
) -> Result<TextEncoder> {
    let mut text_encoder = if streamable {
        loader::load_validated_text_encoder_streamable(source)?
    } else {
        loader::load_validated_text_encoder(source)?
    };
    if let Some(bits) = load_time_quant_bits {
        text_encoder.quantize(bits)?;
    }
    Ok(text_encoder)
}

/// Load the heavy control render components — the composed base+control transformer (+ Q4/Q8 + the
/// base's LoRA/LoKr residuals) and the VAE (+ Q4/Q8) — everything but the text encoder. The
/// overlay-then-quantize order (dense base + dense control, THEN quantize) matches the pre-sc-11124
/// hand-written `load`; the components are independent of the text encoder (separate weight files,
/// deterministic RNG-free quant), so the `Resident` composition is byte-identical. Shared by both
/// control variants (turbo + base) — they differ only in the generate-time schedule + CFG.
pub(crate) fn load_control_heavy(
    spec: &LoadSpec,
    root: &Path,
    control: &WeightsSource,
    streamable: bool,
    model_id: &str,
) -> Result<ZImageControlHeavyOwned> {
    // F-009 (sc-12461): the tier guard runs here too, BEFORE the component loads, so it fires on
    // both residency policies — `Resident` eager-loads through here at load time and `Sequential`
    // re-loads through here on every generate. A requested-vs-packed mismatch on the BASE snapshot
    // (e.g. Q4 over a pre-quantized Q8 turnkey) hard-errors instead of falling through to the
    // no-op base `quantize()` below and silently serving the packed tier. Do NOT gate the quantize
    // calls on the guard's bool: over a matching packed base the composed transformer still holds
    // the DENSE control branch, which needs the load-time quantize (`quantize()` no-ops on the
    // packed base Linears but packs the dense control ones).
    if let Some(q) = spec.quantize {
        mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), model_id)?;
    }
    // Base + control applied dense first, THEN quantize together (the fork's ordering): quantizing
    // before the overlay would replace the control Linears with QuantizedLinear that can't accept
    // the raw bf16 control weights.
    let mut transformer = loader::load_control_transformer_with_stream(root, control, streamable)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): install onto the composed base DiT (the control branch is not an adapter
    // target). Same load-time, post-quantize, residual-over-base path. No-op when `spec.adapters` is
    // empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    // SC-15754: capture the base blocks' final adapter state for the rung-4 stream (see
    // `model::load_heavy`). The control branch carries none by construction.
    transformer.capture_block_adapters();
    Ok(ZImageControlHeavyOwned { transformer, vae })
}

/// Build the tokenizer + request-scoped [`Residency`] seam for either Z-Image ControlNet variant.
/// Both legacy offload-policy values retain the same loaders; `GenerationMemory::stage_residency`
/// chooses warm or staged execution per request. Parameterized by `model_id` + the per-id
/// precision-override message so the base control sibling shares it.
pub(crate) fn load_control_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
) -> Result<(
    TextTokenizer,
    Residency<TextEncoder, ZImageControlHeavyOwned>,
)> {
    // Validate precision + base dir + the required control checkpoint up front (fail fast, same for
    // BOTH policies); then the always-warm tokenizer, then the shared [`build_control_residency`]
    // dispatch.
    let (root, _control) = resolve_control_base_and_control(spec, model_id, precision_msg)?;
    let requantize_bits = if let Some(q) = spec.quantize {
        // F-009 (sc-12461): run the tier guard for BOTH residency policies, before any component
        // load — a Q4 request over a pre-quantized Q8 base turnkey hard-errors here instead of
        // silently serving Q8 (`quantize()` is a no-op on packed weights). Before this fix only the
        // Sequential warn gate below evaluated it, so the DEFAULT `Resident` load skipped the guard
        // entirely; `load_control_heavy` re-checks for the Sequential per-generate reload path.
        mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), model_id)?
            .then_some(q.bits())
    } else {
        None
    };
    let text_encoder_source = crate::ENCODER_CONTRACT.source_for_load(spec, root)?;
    let effective_quant_bits = crate::memory_strategy::loaded_tier(spec, model_id)?
        .quant
        .map(Quant::bits);
    let text_encoder_quant_bits =
        text_encoder_source.load_time_quant_bits(effective_quant_bits, model_id)?;
    let tokenizer = loader::load_validated_tokenizer(&text_encoder_source)?;
    let mut residency = build_control_residency_with_source(
        spec,
        model_id,
        precision_msg,
        text_encoder_source,
        text_encoder_quant_bits,
    )?;
    if let Some(bits) = requantize_bits {
        let warned = std::sync::atomic::AtomicBool::new(false);
        residency = residency.with_staged_advisory(move || {
            if !warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
                mlx_gen::residency::warn_sequential_requantize(model_id, bits);
            }
        });
    }
    Ok((tokenizer, residency))
}

/// Request-scoped builder shared by both Z-Image control variants. Construction touches no component
/// weights. The pose branch carries no PiD overlay, so the seam's `use_pid` argument is unused.
#[cfg(test)]
pub(crate) fn build_control_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
) -> Result<Residency<TextEncoder, ZImageControlHeavyOwned>> {
    let (root, _control) = resolve_control_base_and_control(spec, model_id, precision_msg)?;
    let text_encoder_source = crate::ENCODER_CONTRACT.source_for_load(spec, root)?;
    let effective_quant_bits = crate::memory_strategy::loaded_tier(spec, model_id)?
        .quant
        .map(Quant::bits);
    let text_encoder_quant_bits =
        text_encoder_source.load_time_quant_bits(effective_quant_bits, model_id)?;
    build_control_residency_with_source(
        spec,
        model_id,
        precision_msg,
        text_encoder_source,
        text_encoder_quant_bits,
    )
}

fn build_control_residency_with_source(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
    text_encoder_source: gen_core::ValidatedEncoderSource,
    text_encoder_quant_bits: Option<i32>,
) -> Result<Residency<TextEncoder, ZImageControlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Ok(Residency::request_scoped(
        move |streamable| {
            let (_root, _control) =
                resolve_control_base_and_control(&spec_text, model_id, precision_msg)?;
            load_control_text_encoder_only(
                &text_encoder_source,
                text_encoder_quant_bits,
                streamable,
            )
        },
        move |_use_pid, streamable| {
            let (root, control) =
                resolve_control_base_and_control(&spec_heavy, model_id, precision_msg)?;
            load_control_heavy(&spec_heavy, root, control, streamable, model_id)
        },
    ))
}

/// The per-id precision-override rejection message for the turbo control variant, shared by
/// [`load_control_residency`]'s eager guard and its `Sequential` per-phase loaders.
const PRECISION_MSG: &str =
    "z_image_turbo_control: only dense bf16 is wired (the text encoder runs \
     f32 internally); drop the precision override";

/// Construct a [`ZImageTurboControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] base `Tongyi-MAI/Z-Image-Turbo` snapshot, and
/// `spec.control` (required) the Fun-Controlnet-Union checkpoint (a single `.safetensors` `File`,
/// or a `Dir` of them). Weights load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes the whole
/// transformer (base + control, group_size 64) plus the text encoder + VAE — the fork's whole-model
/// quant, with the control patch embedder left dense (its in-features is not a multiple of 64).
///
/// Component residency (sc-11124, F-172): `Resident` (default) holds every heavy component warm;
/// `Sequential` re-loads per generate in phase order to bound peak memory — routed through the shared
/// `load_control_residency` builder.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, residency) = load_control_residency(spec, MODEL_ID, PRECISION_MSG)?;
    let loaded_tier = crate::memory_strategy::loaded_tier(spec, MODEL_ID)?;
    Ok(Box::new(ZImageTurboControl {
        memory_strategy: crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?,
        loaded_tier,
        descriptor: descriptor(),
        tokenizer,
        residency,
    }))
}

/// The Fun-Controlnet-Union is a *union* ControlNet (pose/canny/depth share one VAE-encoded control
/// path), so all the control boilerplate (resolve/validate-present + the load helpers above) comes
/// from the shared trait (sc-8241). F-089: this is the SAME union checkpoint as the base variant, so
/// it shares the base `accepted_kinds()` (`Only([Pose, Canny, Depth])`) — previously it fell back to
/// the trait default `AcceptedControlKinds::Any`, accepting `Other("scribble")` the base rejects.
impl ControlBranch for ZImageTurboControl {
    fn model_id(&self) -> &'static str {
        MODEL_ID
    }
}

impl ZImageTurboControl {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Required pose/union control + optional img2img init.
        let (control_image, control_scale) = self.resolve_control(req)?;
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // The staged residency lifecycle (encode → drop the text encoder under `Sequential` → load the
        // control DiT/VAE → denoise/decode → free the heavy bundle) is driven by the shared
        // [`Residency::run`] seam (sc-11124), owning the eval/drop/clear discipline, the stage-boundary
        // cancel checks, and the error-safe cache flush — identically to the base `z_image_turbo`. The
        // control variant is guidance-distilled (no CFG / negative prompt), so the encode phase is a
        // single cond `cap`.
        // sc-13571 / GitHub #1658: DiT-dropping staged decode (see `crate::model` for the turbo path).
        let pipeline::RequestRungs {
            stage_residency,
            streamable,
            tiling,
            attention_budget,
            block_window,
            encoder_window,
        } = pipeline::resolve_request_rungs(req, &self.memory_strategy, MODEL_ID)?;
        let images = self.residency.run_staged_request_scoped(
            stage_residency,
            streamable,
            &req.cancel,
            // No PiD overlay on the control path (sc-7846 is base-turbo-only); the heavy loader ignores
            // this flag, so `false` avoids loading a student that would never be used.
            false,
            on_progress,
            // ── Phase A: prompt → cap_feats. The fork's control path is **mixed precision**, NOT pure
            // bf16: it feeds the latents (`x`) and `cap_feats` as bf16 but `control_context` as **f32**
            // (sc-2720, verified against the fork). The f32 control branch then promotes the bf16
            // image/caption stream to f32 when its hints are added, and `latents += dt·velocity` makes
            // the latents f32 after step 0 — so most of the loop runs f32. We match that exactly: bf16
            // cap (txt2img) + f32 control_context below. (img2img keeps f32 cap, mirroring the base
            // img2img; the DiT promotes per-op either way.)
            |text_encoder: &TextEncoder| {
                // Calibration-only fault injection at a physical phase boundary (SC-15449);
                // `None` for every production request, so this is a `None` comparison.
                pipeline::calibration_fault(
                    req,
                    mlx_gen::gen_core::MemoryPhase::Conditioning,
                    MODEL_ID,
                )?;
                let cap = pipeline::encode_prompt(
                    &self.tokenizer,
                    text_encoder,
                    &req.prompt,
                    MODEL_ID,
                    encoder_window,
                )?;
                if is_img2img {
                    Ok(cap)
                } else {
                    // PARITY-BF16 (sc-2609): round the text embeddings to bf16 to match the fork's cap.
                    Ok(cap.as_dtype(Dtype::Bfloat16)?)
                }
            },
            // Materialize the post-cast `cap` while the encoder is still alive (Sequential only) — MLX
            // is lazy, so an un-evaluated `cap` keeps the encoder referenced through the graph and the
            // drop would free nothing.
            |cap| match cap {
                Some(cap) => Ok(mlx_rs::transforms::eval([cap])?),
                None => Ok(()),
            },
            // ── Phase B (denoise): heavy bundle + cap → evaluated latents.
            |heavy: &ZImageControlHeavyOwned, cap, on_progress| {
                pipeline::calibration_fault(
                    req,
                    mlx_gen::gen_core::MemoryPhase::Denoise,
                    MODEL_ID,
                )?;
                // Static shift=3.0 schedule (shared with the base turbo, sc-2536) — build once. An
                // unset `req.scheduler` keeps it byte-exact (epic 7114 N1); a curated name re-shapes σ
                // over the shift.
                let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
                let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
                    req.scheduler.as_deref(),
                    SCHEDULE_SHIFT.ln(),
                    steps,
                    &native.sigmas,
                ))?;

                // The 33ch control context is constant across steps + the batch — build once. It stays
                // **f32** (the fork feeds it f32, which promotes the whole control branch to f32).
                let control_context =
                    encode_control_context(&heavy.vae, control_image, req.width, req.height)?;

                // VAE-encode the init image once too (img2img), constant across the batch (F-034).
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    Some(encode_init_latents(
                        &heavy.vae, image, req.width, req.height,
                    )?)
                } else {
                    None
                };

                let sampler_name = req.sampler.as_deref();
                let latents = pipeline::denoise_batch(
                    &scheduler,
                    clean.as_ref(),
                    start_step,
                    base_seed,
                    req,
                    on_progress,
                    |latents, seed, op| {
                        denoise_control_with_progress_and_preview(
                            &heavy.transformer,
                            &scheduler,
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            &control_context,
                            control_scale,
                            start_step,
                            attention_budget,
                            block_window,
                            &req.cancel,
                            &req.preview,
                            op,
                        )
                    },
                )?;
                Ok(latents)
            },
            // Materialize the latents so the control DiT is no longer held via the lazy graph, then shed.
            |latents| Ok(mlx_rs::transforms::eval(latents.iter())?),
            // ── Phase C (decode): light (VAE) view + latents → images (no PiD on control). Tiled under
            // `Sequential`.
            |view, latents, on_progress| {
                pipeline::calibration_fault(req, mlx_gen::gen_core::MemoryPhase::Decode, MODEL_ID)?;
                let images = pipeline::decode_batch(
                    view.vae,
                    tiling.as_ref(),
                    latents,
                    &req.cancel,
                    on_progress,
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )?;
        Ok(images)
    }
}

impl Generator for ZImageTurboControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability checks (size/count/guidance/negative/accepted conditioning), then the
        // shared control-present check (sc-8241's `ControlBranch::require_control_present`).
        validate_request(self.descriptor.id, &self.descriptor.capabilities, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.memory_strategy, self.loaded_tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            MODEL_ID,
            &self.memory_strategy,
            self.loaded_tier,
            context,
        )
    }
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`. The `impl
// Generator` above stays hand-written because `validate` adds a control-specific check beyond the
// shared `validate_request`, so it is not the plain delegation `impl_generator!` expresses.
mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = crate::model::component_footprint
}

/// The shared memory-strategy contract registration (SC-15449) — resolvable before any weights load, so
/// the worker can select a strategy from the static declaration plus its own measured evidence.
pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec),
        safety_check: crate::memory_strategy::registered_safety_check,
    };
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::OffloadPolicy;
    // `WeightsSource` + `OffloadPolicy` come in via `super::*` (both used by `load`/its helpers).

    #[test]
    fn descriptor_is_z_image_turbo_control() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo_control");
        assert_eq!(d.family, "z-image");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (not on the
        // missing snapshot) — proving the control overlay is wired as a hard requirement.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("base snapshot directory"), "got: {err}");
    }

    #[test]
    fn both_legacy_offload_values_share_fail_fast_validation() {
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()))
                .with_control(WeightsSource::File("/tmp/control.safetensors".into()))
                .with_offload_policy(policy);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(
                err.contains("base snapshot directory"),
                "policy {policy:?} must reach the shared base-dir validation, got: {err}"
            );
        }
    }

    // SC-15806 construction proof: both legacy values retain loaders and touch no component weights.
    fn incomplete_control_spec(policy: OffloadPolicy) -> (tempfile::TempDir, LoadSpec) {
        let snapshot = tempfile::tempdir().expect("snapshot fixture dir");
        gen_core_testkit::write_encoder_contract_fixture(
            &snapshot.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .expect("validation-complete encoder and tokenizer fixture");
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.path().to_path_buf()))
            .with_control(WeightsSource::File(
                snapshot.path().join("control.safetensors"),
            ))
            .with_offload_policy(policy);
        (snapshot, spec)
    }

    // ── F-009 (sc-12461): the control lane's tier-mismatch guard must fire on the DEFAULT
    // `Resident` policy too, not just behind the Sequential F-181 warn gate. Weight-free: the
    // fixture is only the packed `transformer/config.json` marker on the BASE snapshot, and the
    // guard errors before any base/control weights load (the nonexistent control file would
    // otherwise produce a missing-weights error, not the tier message).
    #[test]
    fn control_tier_mismatch_errors_on_resident_and_sequential_load() {
        let tmp = tempfile::tempdir().unwrap();
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let root = loader::packed_snapshot_fixture(&tmp, "control-load", 8);
            gen_core_testkit::write_encoder_contract_fixture(
                &root.join("text_encoder"),
                crate::ENCODER_CONTRACT,
            )
            .unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_control(WeightsSource::File(
                    "/nonexistent/z-image-control-overlay.safetensors".into(),
                ))
                .with_quant(mlx_gen::Quant::Q4)
                .with_offload_policy(policy);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(
                err.contains("pre-quantized Q8"),
                "policy {policy:?}: Q4 over a packed Q8 base turnkey must hard-error, got: {err}"
            );
            assert!(
                err.contains(MODEL_ID),
                "policy {policy:?}: the error must name the model id, got: {err}"
            );
            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn load_control_heavy_runs_tier_guard_before_weights() {
        let tmp = tempfile::tempdir().unwrap();
        // F-009 (sc-12461): the heavy control loader itself re-checks the tier guard — the seam the
        // Sequential path re-loads through on every generate. The fixture has no weights, so
        // reaching the base/control load would fail with a missing-weights error instead —
        // asserting on the tier message proves the guard runs first.
        let root = loader::packed_snapshot_fixture(&tmp, "control-heavy", 8);
        let control =
            WeightsSource::File("/nonexistent/z-image-control-overlay.safetensors".into());
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_control(control.clone())
            .with_quant(mlx_gen::Quant::Q4);
        let err = load_control_heavy(&spec, &root, &control, false, MODEL_ID)
            .err()
            .expect("expected a tier-mismatch error")
            .to_string();
        assert!(err.contains("pre-quantized Q8"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn build_control_residency_defers_for_both_legacy_offload_values() {
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let (snapshot, spec) = incomplete_control_spec(policy);
            assert!(!snapshot.path().join("transformer").exists());
            assert!(!snapshot.path().join("vae").exists());
            assert!(!snapshot.path().join("control.safetensors").exists());
            let res =
                build_control_residency(&spec, MODEL_ID, PRECISION_MSG).unwrap_or_else(|error| {
                    panic!("{policy:?} must defer absent heavy components: {error}")
                });
            assert!(
                res.with_resident_parts(|_, _| ()).unwrap().is_none(),
                "{policy:?} must begin with no warm request-scoped pair"
            );
        }
    }
}
