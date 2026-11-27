//! # candle-gen-flux
//!
//! The **FLUX.1** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-flux`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! exposes both FLUX.1 variants through its explicit family catalog.
//!
//! **txt2img (sc-3694):** [`FluxGenerator::generate`] adapts the `candle-transformers` `flux`
//! reference model (`pipeline`) through the contract: dual **CLIP-L + T5-XXL** text encoders → the
//! FLUX **DiT** (flow-match Euler) → FLUX **AutoEncoder** VAE, emitting `Progress` and honoring
//! `req.cancel`, with **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per
//! seed. The two variants:
//!
//! - **`flux1_schnell`** — Apache-2.0, timestep-distilled: a fixed **4-step** schedule, **no
//!   guidance** (the DiT has no guidance embedding), no negative prompt.
//! - **`flux1_dev`** — guidance-distilled: **25 steps** by default with a resolution-dependent
//!   time-shifted schedule and an embedded **guidance** scale (default 3.5, mlx parity). FLUX.1`dev`
//!   is a **gated** model (a non-commercial license + an accepted HF license agreement); the engine
//!   consumes already-staged weights and does not itself perform credential/license gating — that
//!   stays upstream in the worker's weight-staging layer, **consistent with the mlx provider** (which
//!   likewise carries no gating flag on the descriptor).
//!
//! The descriptors advertise **only** the wired txt2img surface — NOT the full mlx-gen-flux
//! Reference/IP-adapter, LoRA, or Q4/Q8 surface — so the worker routes the rest to the Python
//! fallback rather than the candle backend silently dropping a control (the false-capability trap,
//! exactly as the SDXL and Z-Image slices did). `backend` is `"candle"` and `mac_only` is `false`.

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_core::Tensor, ()>::Err(()));
};

mod pipeline;

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub mod memory_strategy;

// Per-step latent previews (epic 16948, sc-16956) — the 16-channel FLUX.1 fit reused verbatim from
// epic 16624's `mlx-gen-flux`, plus the packed-token → native-latent recovery every route projects
// through. `candle-gen-chroma` and `candle-gen-pulid` share this latent space and reach it here rather
// than restating the constants; see the module docs for the tensor-byte provenance.
pub mod preview;

// Vendored, i32-overflow-safe FLUX.1 VAEs (sc-11154 / F-081): faithful copies of the BFL/native
// `flux::autoencoder` and the diffusers `z_image::vae` with the mid-block spatial self-attention routed
// through the shared budgeted helper (the stock upstream overflows i32 on CUDA at a 2048² decode).
mod vae;

// Shared FLUX.1 component-loading stack (sc-9003 / F-023): the CLIP-L + T5-XXL text encoders, the DiT
// checkpoint mmap, the AutoEncoder VAE, and the CPU-seeded initial noise — the block the three FLUX.1
// providers (txt2img / IP-Adapter / control) used to copy-paste. One home over `candle_gen::loader`
// (F-019), parameterized only by the provider's error `label`; the genuine drift (which DiT wrapper is
// built, the mean-encoder mirror) stays with each caller.
mod flux1_load;

/// Compatibility imports for the shared Candle text encoders.
pub mod packed_te {
    pub use candle_gen::text_encoder::{
        ClipConfig, ClipTextEncoder as PackedClipText, T5Config, T5Encoder as PackedT5Encoder,
    };
}
mod quant;
#[cfg(test)]
mod text_encoder_tests;

// XLabs FLUX IP-Adapter (sc-5872, epic 5480) — reference-image (identity) conditioning. The FLUX DiT
// exposes the per-double-block decoupled cross-attention seam. `ip_adapter` is the XLabs projector
// and K/V weights; `ip_image_encoder` is the pooled
// CLIP-ViT-L tower; `ip_provider` composes them into the bespoke reference stream the worker drives
// directly (not gen-core-registered — the `flux1_*` descriptors stay txt2img-only).
pub mod ip_adapter;
pub mod ip_image_encoder;
pub mod ip_provider;
mod transformer;
pub use ip_provider::{
    compile_ip_plan, ip_load_spec, IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest,
    DEFAULT_IP_SCALE,
};

// FLUX.1-dev Fun-Controlnet-Union (sc-8412) — strict structural conditioning (pose/canny/depth,
// input-agnostic) via `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`. `control` is the diffusers
// residual-emitter control branch (a 6-block partial copy of the dev MMDiT) + the
// `FluxControlTransformer` that injects its 6 residuals into the BFL base double stream at interval
// ceil(19/6)=4; `control_provider` composes it with the reused FLUX text encoders / VAE into the
// bespoke control stream the worker drives by name (NOT gen-core-registered — the `flux1_*` descriptors
// stay txt2img-only; the `flux1_dev_control` worker lane is Phase-B, sc-8304/sc-8246).
pub mod control;
pub mod control_provider;
pub use control::{FluxControlNet, FluxControlNetConfig, FluxControlTransformer};
pub use control_provider::{
    compile_control_plan, control_load_spec, Flux1ControlPaths, Flux1ControlRequest,
    Flux1DevControl, DEFAULT_CONTROL_SCALE,
};

// The vendored FLUX DiT + its post-block image-stream injector seam, re-exported for the PuLID-FLUX
// provider (`candle-gen-pulid`, sc-5492), which composes the FLUX backbone with the EVA-CLIP tower +
// IDFormer + the 20 PerceiverAttentionCA modules driven through [`DitImageInjector`]. `Config` is the
// candle-transformers FLUX config the fork reuses (so it cannot drift on hyperparameters).
pub use transformer::{Config as FluxConfig, DitImageInjector, FluxDit};
// Compatibility name for callers constructing the native BFL layout.
pub use transformer::FluxDit as IpFlux;
// FLUX backbone helpers shared with the PuLID provider so the two never drift on the parity-critical
// tokenization / VAE decode / config (the candle twin of `mlx-gen-flux`'s shared `Flux1` surface). The
// IP-Adapter provider reaches these as `pub(crate)`; PuLID is a separate crate, hence `pub`.
pub use pipeline::{
    ae_config, clip_config, decode_latents, encode_text, execute_candle_flux_denoise_trajectory,
    flux_config, transition_candle_flux_boundary_state, FluxTokenizers,
};

/// Combine unperturbed (`out`) and attention-perturbed (`pert`) FLUX velocity predictions in f32:
/// `out + step_pag * (out - pert)`, cast back to `out.dtype()`.
pub fn apply_pag_blend(
    out: &candle_core::Tensor,
    pert: &candle_core::Tensor,
    step_pag: f32,
) -> candle_gen::Result<candle_core::Tensor> {
    let out_dtype = out.dtype();
    let out_f32 = out.to_dtype(candle_core::DType::F32)?;
    let pert_f32 = pert.to_dtype(candle_core::DType::F32)?;
    let diff = (&out_f32 - &pert_f32)?.affine(step_pag as f64, 0.0)?;
    Ok((&out_f32 + &diff)?.to_dtype(out_dtype)?)
}
// The FLUX dev flow-match time-shift constants + `flow_mu` linear map — one home (sc-11249 / F-140).
// The PuLID reference stream (`candle-gen-pulid`) shares these exact parity-critical schedule pieces
// rather than maintaining a third copy; the IP-Adapter provider reaches them in-crate.
pub use pipeline::{flow_mu, BASE_SHIFT, MAX_SHIFT};

// The tier-detecting reference backbone (sc-10103, epic 9083) — the candle twin of
// `mlx_gen_flux::load_flux1`. Reuses the txt2img `Pipeline`'s tier detect-and-load so the reference
// lanes (`candle-gen-pulid`, the FLUX IP-adapter) consume the SAME `SceneWorks/flux1-dev-mlx`
// q4/q8/bf16 turnkey tiers the base generator does, driving the post-block `DitImageInjector` seam on
// the shared FLUX transformer across both checkpoint layouts.
mod ref_backbone;
pub use ref_backbone::{FluxRefBackbone, FluxRefHeavy};

/// FLUX XLabs IP-Adapter real-weight GPU validation (sc-5872) — env-driven, `#[ignore]`d integration
/// test (the analog of the SDXL/Kolors IP-Adapter Phase-5 harnesses).
#[cfg(test)]
mod ip_validate;

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{
    self, GenerationOutput, GenerationReport, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, RenderPlan, WeightsSource,
};

use pipeline::{Pipeline, SeqHeavy, SeqTextEncoders};
use std::sync::{Arc, Mutex};

/// Registry id for FLUX.1 `schnell` — matches the SceneWorks worker's engine id and the macOS
/// `mlx-gen-flux` descriptor.
pub const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
/// Registry id for FLUX.1 `dev`.
pub const FLUX1_DEV_ID: &str = "flux1_dev";
/// Content identity for the CUDA resident/staged real-weight calibration harness.
pub const RESIDENCY_CALIBRATION_FINGERPRINT: &str = "flux1-cuda-residency-v1";

/// FLUX works in the VAE's /8 latent and the DiT packs that 2×2, so both image dims must be multiples
/// of **16** for a clean pack. Enforced in [`validate`](Generator::validate). Exposed as the
/// pinned-engine stride SceneWorks ties each advertised FLUX image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`; the control / IP-Adapter sibling providers import this same
/// crate-root const so no copy can drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// The two FLUX.1 variants. Carries the parity-critical per-variant metadata (id, step/guidance
/// defaults, T5 length, checkpoint filename) as primitives so `lib.rs` stays candle-light — the
/// pipeline maps the variant onto candle's `flux`/`autoencoder` configs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Schnell,
    Dev,
}

impl Variant {
    /// The registry / engine id.
    pub const fn model_id(self) -> &'static str {
        match self {
            Variant::Schnell => FLUX1_SCHNELL_ID,
            Variant::Dev => FLUX1_DEV_ID,
        }
    }

    /// The variant a registry / engine id names — the exact inverse of [`Self::model_id`].
    ///
    /// `None` for any other id rather than a defaulted variant: a caller holding an unknown id has
    /// not selected a FLUX variant at all, and silently answering `Dev` is how a schnell route ends
    /// up described by dev's geometry.
    pub fn from_model_id(model_id: &str) -> Option<Self> {
        [Variant::Schnell, Variant::Dev]
            .into_iter()
            .find(|variant| variant.model_id() == model_id)
    }

    /// Distilled default step count (mlx parity): schnell 4, dev 25.
    pub const fn default_steps(self) -> u32 {
        match self {
            Variant::Schnell => 4,
            Variant::Dev => 25,
        }
    }

    /// Whether the DiT embeds a guidance scale. schnell is timestep-distilled (no guidance); dev is
    /// guidance-distilled. Drives both the descriptor's `supports_guidance` and the denoise.
    pub const fn supports_guidance(self) -> bool {
        matches!(self, Variant::Dev)
    }

    /// Default guidance scale when a dev request omits one (mlx `DEFAULT_GUIDANCE`). Inert for schnell.
    pub const fn default_guidance(self) -> f32 {
        3.5
    }

    pub(crate) fn resolved_guidance(self, requested: Option<f32>) -> f32 {
        let guidance = if self.supports_guidance() {
            requested.unwrap_or(self.default_guidance())
        } else {
            0.0
        };
        candle_gen::execution::f32_policy("flux/effective_guidance", guidance)
    }

    /// T5 sequence length the prompt is padded to (diffusers FluxPipeline default): schnell 256,
    /// dev 512. FLUX attends every T5 position, so this is parity-critical.
    pub const fn t5_max_len(self) -> usize {
        match self {
            Variant::Schnell => 256,
            Variant::Dev => 512,
        }
    }

    /// The root BFL DiT checkpoint filename in the snapshot.
    pub const fn transformer_file(self) -> &'static str {
        match self {
            Variant::Schnell => "flux1-schnell.safetensors",
            Variant::Dev => "flux1-dev.safetensors",
        }
    }

    /// Whether this is the dev variant (guidance + time-shifted schedule).
    pub const fn is_dev(self) -> bool {
        matches!(self, Variant::Dev)
    }

    pub fn pag_capability(self) -> gen_core::PagCapability {
        let (default_stop, recommended_scale) = match self {
            Variant::Schnell => (0.75, 1.2),
            Variant::Dev => (1.0, 1.5),
        };
        gen_core::PagCapability {
            num_blocks: 57,
            block_domain: std::borrow::Cow::Borrowed("flux_blocks"),
            default_layers: vec![28],
            default_start: 0.0,
            default_stop,
            recommended_scale,
            max_scale: gen_core::DEFAULT_PAG_MAX_SCALE,
        }
    }
}

/// A loaded candle FLUX generator (one per variant). The shared residency owner holds either the
/// warm phase pair or the deferred per-request loaders.
pub struct FluxGenerator {
    variant: Variant,
    descriptor: ModelDescriptor,
    pipe: Pipeline,
    residency: candle_gen::Residency<SeqTextEncoders, SeqHeavy>,
    lifecycle: Mutex<()>,
    stream_cancel: Arc<Mutex<gen_core::CancelFlag>>,
    bounded_host_decode: Arc<Mutex<bool>>,
    loaded_quant: Option<gen_core::Quant>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    load_binding: gen_core::RenderLoadBinding,
}

impl FluxGenerator {
    /// Load-time selections bound to every reported attempt.
    fn observe_loaded(&self) {
        self.pipe.observe_adapter_provenance();
        candle_gen::execution::policy(
            "loaded/loaded_quant",
            format!("{:?}", self.loaded_quant).into(),
        );
    }
}

impl Generator for FluxGenerator {
    fn generate_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || self.observe_loaded())
    }

    /// The admitted runner's entry: the same planned body, captured as one attempt.
    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            self.observe_loaded()
        })
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        memory_strategy::admission_safety_check(contract, context, self.loaded_quant)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(contract, context, self.loaded_quant)?;
        Ok(Some(Box::new(memory_strategy::FluxMemoryScope::new(
            self.variant.model_id(),
            self.pipe.device().clone(),
            contract,
            context,
        ))))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_base_request(self.variant, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| gen_core::Error::Msg("FLUX.1 base render produced no images".into()))
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        let plan = compile_base_plan(self.variant, req)?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&self.load_binding)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        let stage_residency = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stage_residency);
        let stream_transformer_blocks = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stream_transformer_blocks);
        *candle_gen::lock_recover(&self.stream_cancel) = req.cancel.clone();
        *candle_gen::lock_recover(&self.bounded_host_decode) =
            req.memory.is_some_and(|memory| memory.tile_vae_decode);
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let result = self.residency.run_request_scoped(
            stage_residency,
            stream_transformer_blocks,
            &req.cancel,
            req.use_pid(),
            on_progress,
            // PiD's Gemma caption encode is text conditioning, and its engine arrives with the
            // heavy bundle; an img2img `source_image` also runs its VAE encode inside
            // `render_residency` once the heavy bundle is resident. Both hold Conditioning open
            // until `render_residency` has finished their conditioning work.
            |text| {
                if req.use_pid() || req.source_image.is_some() {
                    run.begin_conditioning()?;
                    self.pipe.encode_residency(text, &req.prompt)
                } else {
                    // Synchronize before Conditioning closes so the queued T5/CLIP kernels (and
                    // any failure they raise) stay inside it.
                    run.encode(|| {
                        let encoded = self.pipe.encode_residency(text, &req.prompt)?;
                        self.pipe.device().synchronize()?;
                        Ok::<_, candle_gen::CandleError>(encoded)
                    })
                }
            },
            |_| Ok(self.pipe.device().synchronize()?),
            |heavy, encoded, on_progress| {
                let result =
                    self.pipe
                        .render_residency(req, plan, &run, heavy, encoded, on_progress);
                candle_gen::synchronize_result(self.pipe.device(), result)
            },
        );
        *candle_gen::lock_recover(&self.bounded_host_decode) = false;
        *candle_gen::lock_recover(&self.stream_cancel) = gen_core::CancelFlag::default();
        let images = result?;
        let pag_report = self
            .descriptor
            .capabilities
            .resolve_pag(self.variant.model_id(), req)?
            .map(|p| p.execution_report(plan.denoise_sigmas().len().saturating_sub(1) as u32));
        let low_res_report = plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
        let mut report = GenerationReport::from_output(GenerationOutput::Images(images));
        report.pag = pag_report;
        report.low_res_start = low_res_report;
        Ok(report)
    }
}

/// The same request floor is used before load by the registry plan compiler and by a loaded
/// generator.  No tensor or checkpoint is touched on this path.
fn validate_base_request(variant: Variant, req: &GenerationRequest) -> gen_core::Result<()> {
    // The shared capability floor: the descriptor advertises no conditioning and (for schnell) no
    // guidance / negative prompt, so any of those is rejected here (distilled-model honesty).
    descriptor_for(variant)
        .capabilities
        .validate_request(variant.model_id(), req)?;
    let id = variant.model_id();
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    // An explicit `steps: Some(0)` would VAE-decode pure noise — reject loudly (txt2img-only).
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    let stage_residency = req.memory.is_some_and(|memory| memory.stage_residency);
    if req.memory.as_ref().is_some_and(|memory| {
        memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
    }) && !stage_residency
    {
        return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode, attention, and transformer residency require request-scoped staged residency",
                variant.model_id()
            )));
    }
    if req.use_pid()
        && req.memory.is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        })
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: optimized native-VAE memory strategies do not support PiD decode",
            variant.model_id()
        )));
    }
    if req.features.low_res_start.is_some() || req.source_image.is_some() {
        let sigmas = pipeline::base_schedule(variant, req)?;
        let executed_steps = sigmas.len().saturating_sub(1);
        if let Some(spec) = &req.features.low_res_start {
            spec.resolve_for_descriptor(
                &descriptor_for(variant),
                req,
                executed_steps,
                SIZE_MULTIPLE,
            )?;
        }
    }
    Ok(())
}

/// Resolve the base FLUX.1 schedule and request defaults while the provider is still unloaded.
/// A plan is bound to one variant and one request, so the heavy path cannot silently choose a
/// different seed, solver, guidance or step count after memory admission.
fn compile_base_plan(variant: Variant, req: &GenerationRequest) -> gen_core::Result<RenderPlan> {
    validate_base_request(variant, req)?;
    let sigmas = pipeline::base_schedule(variant, req)?;
    let executed_steps = sigmas.len().saturating_sub(1);
    let guidance = if variant.supports_guidance() {
        req.guidance.unwrap_or(variant.default_guidance())
    } else {
        0.0
    };
    RenderPlan::base_image(
        &descriptor_for(variant),
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 64,
            sampler: req.sampler.as_deref().unwrap_or("euler"),
            guidance,
            sigmas,
            executed_steps,
            capture_sigma: 0.0,
        },
    )
}

/// Compile the same native FLUX trajectory for a bespoke conditioned provider. Its descriptor
/// and full load contract remain owned by that provider; no base txt2img capability is implied.
pub fn add_conditioned_native_aliases(descriptor: &mut ModelDescriptor) {
    // Bespoke Control/IP/PuLID routes historically pass these names to the shared flow driver,
    // where they select Euler over the native FLUX schedule. Keep the registered base menu intact.
    descriptor.capabilities.samplers = candle_gen::menu_with_aliases(
        std::mem::take(&mut descriptor.capabilities.samplers),
        &["flow_match", "flow_match_euler"],
    );
    descriptor.capabilities.schedulers = candle_gen::menu_with_aliases(
        std::mem::take(&mut descriptor.capabilities.schedulers),
        &["native", "flow_match", "flow_match_euler"],
    );
}

pub fn compile_conditioned_plan(
    variant: Variant,
    descriptor: &ModelDescriptor,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<RenderPlan> {
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    let sigmas = pipeline::base_schedule(variant, req)?;
    let executed_steps = sigmas.len().saturating_sub(1);
    let guidance = if variant.supports_guidance() {
        req.guidance.unwrap_or(variant.default_guidance())
    } else {
        0.0
    };
    RenderPlan::packed_image_trajectory(
        descriptor,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 64,
            sampler: req.sampler.as_deref().unwrap_or("euler"),
            guidance,
            sigmas,
            executed_steps,
            capture_sigma: 0.0,
        },
    )
    .map(|plan| plan.bind_load(spec))
}

fn compile_schnell_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<RenderPlan>> {
    validate_base_load_spec(Variant::Schnell, spec)?;
    validate_preload_pid(spec, req, Variant::Schnell)?;
    compile_base_plan(Variant::Schnell, req).map(Some)
}

fn compile_dev_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<RenderPlan>> {
    validate_base_load_spec(Variant::Dev, spec)?;
    validate_preload_pid(spec, req, Variant::Dev)?;
    compile_base_plan(Variant::Dev, req).map(Some)
}

fn validate_preload_pid(
    spec: &LoadSpec,
    req: &GenerationRequest,
    variant: Variant,
) -> gen_core::Result<()> {
    if !req.use_pid() {
        return Ok(());
    }
    if spec.pid.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)",
            variant.model_id()
        )));
    }
    if req.pid_capture_sigma().is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: pid_capture_sigma (from_ldm early-stop) is not wired for this latent space yet",
            variant.model_id()
        )));
    }
    Ok(())
}

/// Primary typed route declaration for a Candle FLUX.1 base variant (`REG-1`, `AP5`).
pub(crate) fn route_base(variant: Variant) -> gen_core::RouteRecord {
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bindings = match variant {
        Variant::Schnell => gen_core::RouteBindings::generator(descriptor_schnell, load_schnell)
            .with_render_plan(compile_schnell_plan),
        Variant::Dev => gen_core::RouteBindings::generator(descriptor_dev, load_dev)
            .with_render_plan(compile_dev_plan),
    };
    #[cfg(feature = "cuda")]
    {
        bindings = match variant {
            Variant::Schnell => bindings
                .with_memory_registration(SCHNELL_MEMORY_REGISTRATION)
                .with_memory_fixture(
                    gen_core::candle_memory_contract_surface_specs,
                    registered_schnell_memory_contract,
                )
                .with_memory_behavior_registration(SCHNELL_MEMORY_BEHAVIOR),
            Variant::Dev => bindings
                .with_memory_registration(DEV_MEMORY_REGISTRATION)
                .with_memory_fixture(
                    gen_core::candle_memory_contract_surface_specs,
                    registered_dev_memory_contract,
                )
                .with_memory_behavior_registration(DEV_MEMORY_BEHAVIOR),
        };
    }

    gen_core::RouteRecord {
        route_id: variant.model_id(),
        family: "flux",
        backend: "candle",
        modality: Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("clip_l", Some(77)),
                    gen_core::TextEncoderDeclaration::new(
                        "t5_xxl",
                        Some(variant.t5_max_len() as u32),
                    ),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: false,
                modules: vec![],
                low_res_start_forbids_reference_conditioning: true,
            },
            sampling: gen_core::SamplingSurface {
                samplers: candle_gen::curated_sampler_names(),
                schedulers: candle_gen::curated_scheduler_names(),
                guidance_methods: vec![],
                supports_guidance: variant.supports_guidance(),
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("flux1_mmdit")
                    .with_low_res_start()
                    .with_pag(variant.pag_capability()),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&gen_core::FLUX1_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: Some(64),
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface {
                components: vec!["pid_decoder"],
                max_temporal_upsample_rounds: 0,
            },
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::ForwardResidual,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["clip_l", "t5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["flux1_mmdit"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["flux1_ae"],
                        false,
                    ),
                ],
                supports_sequential_offload: true,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
                ..Default::default()
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::LatentProjection,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(variant.default_steps()),
            guidance: variant
                .supports_guidance()
                .then_some(variant.default_guidance()),
            sampler: Some("euler"),
            schedule: None,
            shift: variant.is_dev().then_some(1.15),
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings,
    }
}

/// The descriptor for a FLUX.1 `variant`, computed from [`route_base`].
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    route_base(variant).compute_descriptor()
}

/// FLUX.1 `schnell` descriptor (registry).
pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(Variant::Schnell)
}

/// FLUX.1 `dev` descriptor (registry).
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(Variant::Dev)
}

/// Validate the base FLUX.1 load shape before compiling a plan or constructing the lazy generator.
/// `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a black-forest-labs `FLUX.1-{schnell,dev}` snapshot (root
/// `flux1-*.safetensors` + `ae.safetensors`, plus the `text_encoder/`, `text_encoder_2/`,
/// `tokenizer_2/` subdirs). LoRA adapters, on-the-fly quantization, and control/IP-adapter overlays
/// are rejected — none are wired in this slice, so refusing is more honest than silently dropping
/// them (the worker falls back to Python).
fn validate_base_load_spec(
    variant: Variant,
    spec: &LoadSpec,
) -> gen_core::Result<std::path::PathBuf> {
    let id = variant.model_id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (flux1-*.safetensors, ae.safetensors, \
                 text_encoder/, text_encoder_2/, tokenizer_2/), not a single .safetensors file"
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays yet (txt2img only)"
        )));
    }
    Ok(root)
}

/// Construct the lazy base generator after the shared pre-load shape check.
fn load_variant(variant: Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.model_id();
    let root = validate_base_load_spec(variant, spec)?;
    // FLUX is a bf16 model; load at bf16 regardless of the CPU-default dtype. The device is the
    // backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    let pipe = Pipeline::load(
        variant,
        &root,
        &device,
        DType::BF16,
        spec.pid.clone(),
        spec.adapters.clone(),
    );
    let loaded_quant = memory_strategy::snapshot_quant_tier(spec, id)?;
    #[cfg(any(feature = "cuda", test))]
    let memory_strategy = Some(memory_strategy::provider_contract(id, spec)?);
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_strategy = None;
    let resident_pipe = pipe.clone();
    let text_pipe = pipe.clone();
    let heavy_pipe = pipe.clone();
    let stream_cancel = Arc::new(Mutex::new(gen_core::CancelFlag::default()));
    let heavy_cancel = stream_cancel.clone();
    let bounded_host_decode = Arc::new(Mutex::new(false));
    let heavy_bounded_host_decode = bounded_host_decode.clone();
    let residency = candle_gen::Residency::request_scoped_with_resident(
        move |_| {
            Ok((
                resident_pipe.load_text_residency()?,
                resident_pipe.load_heavy_residency(true)?,
            ))
        },
        move |_| text_pipe.load_text_residency(),
        move |use_pid, stream_transformer_blocks| {
            heavy_pipe.load_heavy_residency_with_memory(
                use_pid,
                stream_transformer_blocks,
                *candle_gen::lock_recover(&heavy_bounded_host_decode),
                &candle_gen::lock_recover(&heavy_cancel),
            )
        },
    );
    Ok(Box::new(FluxGenerator {
        variant,
        descriptor: descriptor_for(variant),
        pipe,
        residency,
        lifecycle: Mutex::new(()),
        stream_cancel,
        bounded_host_decode,
        loaded_quant,
        memory_strategy,
        load_binding: gen_core::RenderLoadBinding::from_spec(spec),
    }))
}

/// Registry entry point for FLUX.1 `schnell`.
pub fn load_schnell(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Variant::Schnell, spec)
}

/// Registry entry point for FLUX.1 `dev`.
pub fn load_dev(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Variant::Dev, spec)
}

fn route_pulid_feature() -> gen_core::RouteRecord {
    let mut route = ip_provider::route_ip(Variant::Dev);
    route.route_id = "flux1_dev_pulid";
    route.family = "pulid";
    route.structure.conditioning.modules = vec!["pulid_idformer"];
    route.structure.weight_plan.requires_ip_adapter_overlay = false;
    route.structure.weight_plan.requires_identity_overlay = true;
    route.bindings = gen_core::RouteBindings::feature_route(
        Some(FLUX1_DEV_ID),
        pulid_feature_descriptor,
    )
    .with_render_plan(compile_pulid_feature_route);
    route
}

fn pulid_feature_descriptor() -> ModelDescriptor {
    route_pulid_feature().compute_descriptor()
}

fn compile_pulid_feature_route(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<RenderPlan>> {
    let descriptor = pulid_feature_descriptor();
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !matches!(
        req.conditioning.as_slice(),
        [gen_core::Conditioning::Reference { .. }]
    ) || spec.identity.is_none()
    {
        return Err(gen_core::Error::Unsupported(
            "flux1_dev_pulid requires a Reference face image and identity weights".into(),
        ));
    }
    compile_conditioned_plan(Variant::Dev, &descriptor, spec, req).map(Some)
}

/// Add all Candle FLUX.1 providers to an explicit media registry builder via one-call
/// [`ProviderRegistryBuilder::register_route`](candle_gen::gen_core::ProviderRegistryBuilder::register_route)
/// (`SW-6979`, `REG-1`).
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(route_base(Variant::Schnell))
        .register_route(route_base(Variant::Dev))
        .register_route(control_provider::route_control())
        .register_route(ip_provider::route_ip(Variant::Dev))
        .register_route(ip_provider::route_ip(Variant::Schnell))
        .register_route(route_pulid_feature())
}

/// Register only weights-free memory-contract surfaces; safe on every build platform.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(SCHNELL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: FLUX1_SCHNELL_ID,
            contract: registered_schnell_memory_contract,
        })
        .register_memory_strategy(DEV_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: FLUX1_DEV_ID,
            contract: registered_dev_memory_contract,
        })
}

fn registered_schnell_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(FLUX1_SCHNELL_ID, spec)
}

fn registered_dev_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(FLUX1_DEV_ID, spec)
}

const SCHNELL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: FLUX1_SCHNELL_ID,
    contract: registered_schnell_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

const DEV_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: FLUX1_DEV_ID,
    contract: registered_dev_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

#[cfg(feature = "cuda")]
const SCHNELL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: FLUX1_SCHNELL_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(FLUX1_SCHNELL_ID, spec, contract, context)
        },
    };

#[cfg(feature = "cuda")]
const DEV_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: FLUX1_DEV_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(FLUX1_DEV_ID, spec, contract, context)
        },
    };

/// Build the complete explicit Candle FLUX.1 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["flux1_schnell", "flux1_dev"]);
    }

    #[test]
    fn route_records_preserve_exact_pre_migration_descriptor_and_capability_parity() {
        use candle_gen::gen_core::{
            AcceptedControlKinds, ConditioningKind, ControlKind, FeatureInapplicabilityCode,
            Modality, SizeFloor, StepSupport, FLUX1_LATENT_SPACE,
        };

        let registry = super::provider_registry().unwrap();
        assert_eq!(registry.route_records().len(), 6);
        assert!(
            registry.route_conformance_errors().is_empty(),
            "route conformance errors: {:?}",
            registry.route_conformance_errors()
        );
        assert!(
            registry.descriptor_conformance_errors().is_empty(),
            "descriptor conformance errors: {:?}",
            registry.descriptor_conformance_errors()
        );

        let schnell = registry.generator_descriptor("flux1_schnell").unwrap();
        assert_eq!(schnell.family, "flux");
        assert_eq!(schnell.backend, "candle");
        assert_eq!(schnell.modality, Modality::Image);
        assert_eq!(schnell.denoiser_output_latent_space, Some(&FLUX1_LATENT_SPACE));
        assert!(schnell.encoder_contract.is_none());
        assert!(schnell.control_kinds.is_none());
        assert!(schnell.required_components.is_empty());
        assert!(!schnell.capabilities.supports_negative_prompt);
        assert!(!schnell.capabilities.supports_guidance);
        assert!(!schnell.capabilities.supports_true_cfg);
        assert!(schnell.capabilities.conditioning.is_empty());
        assert!(schnell.capabilities.supported_quants.is_empty());
        assert!(schnell.capabilities.supports_lora);
        assert!(schnell.capabilities.supports_lokr);
        assert_eq!(
            schnell.capabilities.samplers,
            candle_gen::curated_sampler_names()
        );
        assert_eq!(
            schnell.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
        assert_eq!(schnell.capabilities.min_size, 256);
        assert_eq!(schnell.capabilities.max_size, 2048);
        assert_eq!(schnell.capabilities.max_count, 8);
        assert_eq!(schnell.capabilities.size_floor, SizeFloor::RangeChecked);
        assert_eq!(schnell.capabilities.supported_steps, StepSupport::Unconstrained);
        assert!(!schnell.capabilities.mac_only);
        assert!(!schnell.capabilities.requires_sigma_shift);
        assert!(schnell.capabilities.supports_sequential_offload);
        assert!(schnell.capabilities.supports_preview);
        assert!(schnell.capabilities.supports_low_res_start);
        assert!(!schnell.capabilities.supports_phases);
        assert_eq!(
            schnell.capabilities.pag,
            Some(super::Variant::Schnell.pag_capability())
        );

        let dev = registry.generator_descriptor("flux1_dev").unwrap();
        assert!(dev.capabilities.supports_guidance);
        assert!(dev.capabilities.supports_low_res_start);
        assert_eq!(dev.capabilities.pag, Some(super::Variant::Dev.pag_capability()));
        assert_eq!(registry.route_record("flux1_dev").unwrap().defaults.steps, Some(25));
        assert_eq!(
            registry.route_record("flux1_dev").unwrap().defaults.guidance,
            Some(3.5)
        );

        let ctrl_desc = registry
            .route_record("flux1_dev_control")
            .unwrap()
            .compute_descriptor();
        assert_eq!(ctrl_desc.capabilities.conditioning, vec![ConditioningKind::Control]);
        assert_eq!(
            ctrl_desc.control_kinds,
            Some(AcceptedControlKinds::Only(vec![
                ControlKind::Pose,
                ControlKind::Canny,
                ControlKind::Depth,
            ]))
        );
        assert_eq!(ctrl_desc.capabilities.min_size, super::SIZE_MULTIPLE);
        assert_eq!(ctrl_desc.capabilities.max_size, u32::MAX);
        assert_eq!(ctrl_desc.capabilities.max_count, 1);
        assert!(!ctrl_desc.capabilities.supports_low_res_start);

        let all = registry.all_route_feature_applicability();
        assert!(all["flux1_dev"]["low_res_start"].applicable);
        assert!(all["flux1_schnell"]["low_res_start"].applicable);
        assert!(!all["flux1_dev_control"]["low_res_start"].applicable);
        assert!(all["flux1_dev_control"]["low_res_start"]
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));
        for ref_route in [
            "flux1_dev_ip_adapter",
            "flux1_schnell_ip_adapter",
            "flux1_dev_pulid",
        ] {
            assert!(!all[ref_route]["low_res_start"].applicable);
            assert!(all[ref_route]["low_res_start"]
                .reasons
                .iter()
                .any(|r| r.code
                    == FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// Both base variants compile their complete trajectory without reading a checkpoint.  This
    /// freezes the same seed, guidance and native schedule the old per-seed render resolved after
    /// loading text and heavy components.
    #[test]
    fn base_flux_plan_is_tensor_free_and_variant_exact() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let registry = provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        for (route_id, expected_steps, expected_max_k) in
            [("flux1_schnell", 4u32, 3u64), ("flux1_dev", 25u32, 24u64)]
        {
            let app = &all_features[route_id]["low_res_start"];
            assert!(
                app.applicable,
                "candle {route_id} must support low_res_start"
            );
            let json = app.to_json();
            assert_eq!(json["class"], "recipe");
            assert_eq!(json["version"], 1);
            assert_eq!(
                json["parameter_schema"]["k"]["default_executed_steps"],
                expected_steps
            );
            assert_eq!(json["parameter_schema"]["k"]["max"], expected_max_k);
        }
        assert!(!all_features["flux1_dev_control"]["low_res_start"].applicable);
        assert!(all_features["flux1_dev_control"]["low_res_start"]
            .reasons
            .iter()
            .any(|r| r.code
                == gen_core::FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));
        for ref_route in [
            "flux1_dev_ip_adapter",
            "flux1_schnell_ip_adapter",
            "flux1_dev_pulid",
        ] {
            let app = &all_features[ref_route]["low_res_start"];
            assert!(
                !app.applicable,
                "candle {ref_route} must refuse low_res_start"
            );
            assert!(app.reasons.iter().any(|r| r.code
                == gen_core::FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning));
        }

        // Unsupported `phases` on every migrated Candle FLUX.1 route returns a typed Error::Unsupported.
        let phased_req = GenerationRequest {
            prompt: "a blue lamp".into(),
            width: 512,
            height: 512,
            features: candle_gen::gen_core::FeatureParams { phases: Some(vec![gen_core::GenerationPhase {
                steps: 4,
                guidance: Some(3.5),
                adapters: vec![],
            }]), ..Default::default() },
            ..Default::default()
        };
        for id in [FLUX1_SCHNELL_ID, FLUX1_DEV_ID] {
            assert!(matches!(
                registry.compile_render_plan(id, &spec, &phased_req),
                Err(gen_core::Error::Unsupported(_))
            ));
        }
        assert!(matches!(
            compile_conditioned_plan(
                Variant::Dev,
                &control_provider::control_descriptor(),
                &spec,
                &phased_req
            ),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            compile_conditioned_plan(
                Variant::Dev,
                &ip_provider::ip_descriptor(Variant::Dev),
                &spec,
                &phased_req
            ),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            compile_conditioned_plan(
                Variant::Schnell,
                &ip_provider::ip_descriptor(Variant::Schnell),
                &spec,
                &phased_req
            ),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            compile_conditioned_plan(
                Variant::Dev,
                &pulid_feature_descriptor(),
                &spec,
                &phased_req
            ),
            Err(gen_core::Error::Unsupported(_))
        ));

        for (variant, expected_steps, expected_guidance) in
            [(Variant::Schnell, 4, 0.0), (Variant::Dev, 25, 3.5)]
        {
            let req = GenerationRequest {
                prompt: "a blue lamp".into(),
                seed: Some(91),
                width: 1024,
                height: 768,
                count: 2,
                ..Default::default()
            };
            let plan = registry
                .compile_render_plan(variant.model_id(), &spec, &req)
                .unwrap()
                .expect("base FLUX.1 must have a pre-load plan");
            assert_eq!(plan.base_seed(), 91);
            assert_eq!(plan.guidance(), expected_guidance);
            assert_eq!(plan.sigmas().len(), expected_steps + 1);
            assert_eq!(plan.capture_sigma(), 0.0);
            plan.check_request(&descriptor_for(variant), &req).unwrap();

            let mut different = req.clone();
            different.width = 768;
            assert!(plan
                .check_request(&descriptor_for(variant), &different)
                .is_err());
        }
    }

    #[test]
    fn base_plan_rejects_unloadable_spec_and_unwired_pid_before_weights() {
        use candle_gen::gen_core::Quant;
        let registry = provider_registry().unwrap();
        let req = GenerationRequest {
            prompt: "a blue lamp".into(),
            seed: Some(91),
            ..Default::default()
        };
        let file = LoadSpec::new(WeightsSource::File("/nonexistent/flux.safetensors".into()));
        let quant = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q8);
        let control = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_control(WeightsSource::Dir("/nonexistent/control".into()));
        for id in [FLUX1_SCHNELL_ID, FLUX1_DEV_ID] {
            for spec in [&file, &quant, &control] {
                assert!(registry.compile_render_plan(id, spec, &req).is_err());
            }
            let pid_req = GenerationRequest {
                features: candle_gen::gen_core::FeatureParams { pid: Some(candle_gen::gen_core::PidParams::enabled()), ..Default::default() },
                ..req.clone()
            };
            let no_pid = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            assert!(registry.compile_render_plan(id, &no_pid, &pid_req).is_err());
            let with_pid = no_pid.with_pid(
                WeightsSource::File("/nonexistent/pid.safetensors".into()),
                WeightsSource::Dir("/nonexistent/gemma".into()),
            );
            let early_capture = GenerationRequest {
                features: candle_gen::gen_core::FeatureParams { pid: (Some(0.2)).map(candle_gen::gen_core::PidParams::with_capture_sigma), ..Default::default() },
                ..pid_req
            };
            assert!(registry
                .compile_render_plan(id, &with_pid, &early_capture)
                .is_err());
        }
    }

    /// `from_model_id` is the exact inverse of `model_id`, and answers nothing for an id it does
    /// not own. The memory contract selects its trunk geometry through it (sc-22661), so a version
    /// that defaulted to `Dev` would publish dev's geometry for schnell.
    #[test]
    fn a_model_id_resolves_to_its_own_variant_and_no_other() {
        for variant in [Variant::Schnell, Variant::Dev] {
            assert_eq!(Variant::from_model_id(variant.model_id()), Some(variant));
        }
        assert_eq!(Variant::from_model_id("flux2_dev"), None);
        assert_eq!(Variant::from_model_id(""), None);
    }

    /// Both variants resolve as candle generators through the family registry. `load` is lazy, so a nonexistent
    /// weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn both_variants_register_and_resolve_as_candle() {
        for (id, family) in [(FLUX1_SCHNELL_ID, "flux"), (FLUX1_DEV_ID, "flux")] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("candle {id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, family);
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Image);
        }
    }

    /// schnell advertises no guidance (timestep-distilled); dev advertises guidance. Neither
    /// advertises negative prompt, true-CFG, conditioning, LoRA, or quantization, and neither is
    /// Mac-only.
    #[test]
    fn descriptors_advertise_only_wired_txt2img_surface() {
        let schnell = descriptor_schnell();
        let dev = descriptor_dev();
        assert!(
            !schnell.capabilities.supports_guidance,
            "schnell is distilled"
        );
        assert!(
            dev.capabilities.supports_guidance,
            "dev is guidance-distilled"
        );
        for d in [&schnell, &dev] {
            assert!(!d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(!d.capabilities.mac_only);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(d.capabilities.supports_lora);
            assert!(d.capabilities.supports_lokr);
            assert!(d.capabilities.supported_quants.is_empty());
            assert_eq!(d.capabilities.min_size, 256);
            assert_eq!(d.capabilities.max_size, 2048);
            assert_eq!(d.capabilities.max_count, 8);
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123) — the denoise routes
            // through the shared driver, so both variants now advertise the full curated vocabulary.
            assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
            assert_eq!(
                d.capabilities.schedulers,
                candle_gen::curated_scheduler_names()
            );
        }
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly. dev accepts a
    /// guidance value (advertised), schnell rejects it (not advertised). Lazy generator → no weights.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let schnell = crate::provider_registry()
            .unwrap()
            .load(FLUX1_SCHNELL_ID, &spec)
            .unwrap();
        let dev = crate::provider_registry()
            .unwrap()
            .load(FLUX1_DEV_ID, &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(schnell.validate(&ok).is_ok());
        assert!(dev.validate(&ok).is_ok());

        // dev advertises guidance, so a guidance request is accepted; schnell rejects it.
        let with_guidance = GenerationRequest {
            prompt: "x".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(dev.validate(&with_guidance).is_ok());
        assert!(
            schnell.validate(&with_guidance).is_err(),
            "schnell advertises no guidance"
        );

        // Shapes rejected on both variants.
        for g in [&schnell, &dev] {
            for bad in [
                GenerationRequest::default(), // empty prompt
                GenerationRequest {
                    prompt: "x".into(),
                    negative_prompt: Some("blurry".into()),
                    ..Default::default()
                },
                GenerationRequest {
                    prompt: "x".into(),
                    width: 1000, // not a multiple of 16
                    ..Default::default()
                },
                GenerationRequest {
                    prompt: "x".into(),
                    steps: Some(0),
                    ..Default::default()
                },
                GenerationRequest {
                    prompt: "x".into(),
                    conditioning: vec![Conditioning::Reference {
                        image: Image::default(),
                        strength: None,
                    }],
                    ..Default::default()
                },
            ] {
                assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
            }
        }
        // Neither variant advertises img2img Reference.
        assert!(!descriptor_schnell()
            .capabilities
            .accepts(ConditioningKind::Reference));
        assert!(!descriptor_dev()
            .capabilities
            .accepts(ConditioningKind::Reference));

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised FLUX bucket
        // to. Pin the value and mutation-check that a multiple of 8 that is not SIZE_MULTIPLE (16) is
        // rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = dev
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(dev
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    /// Adapters are accepted; unsupported quant/control overlays remain typed rejections.
    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        for load in [load_schnell, load_dev] {
            let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
                AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
            ]);
            assert!(load(&lora).is_ok());

            let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
            assert!(matches!(
                load(&quant).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));

            let control = LoadSpec::new(WeightsSource::Dir("/snap".into()))
                .with_control(WeightsSource::Dir("/ctrl".into()));
            assert!(matches!(
                load(&control).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        for load in [load_schnell, load_dev] {
            let spec = LoadSpec::new(WeightsSource::File("/tmp/flux.safetensors".into()));
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(err.contains("snapshot directory"), "got: {err}");
        }
    }

    #[test]
    fn pag_capability_validation_and_joint_attention_match_flux1_contract() {
        use candle_core::{DType, Device, Tensor};
        use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
        use candle_gen::gen_core::PagConfig;

        let dev = descriptor_dev();
        let dev_pag = dev.capabilities.pag.as_ref().expect("dev pag");
        assert_eq!(dev_pag.num_blocks, 57);
        assert_eq!(dev_pag.block_domain.as_ref(), "flux_blocks");
        assert_eq!(dev_pag.default_layers, vec![28]);
        assert_eq!((dev_pag.default_start, dev_pag.default_stop), (0.0, 1.0));
        assert_eq!((dev_pag.recommended_scale, dev_pag.max_scale), (1.5, 15.0));

        let schnell = descriptor_schnell();
        let schnell_pag = schnell.capabilities.pag.as_ref().expect("schnell pag");
        assert_eq!(schnell_pag.num_blocks, 57);
        assert_eq!(schnell_pag.block_domain.as_ref(), "flux_blocks");
        assert_eq!(schnell_pag.default_layers, vec![28]);
        assert_eq!(
            (schnell_pag.default_start, schnell_pag.default_stop),
            (0.0, 0.75)
        );
        assert_eq!(
            (schnell_pag.recommended_scale, schnell_pag.max_scale),
            (1.2, 15.0)
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = load_dev(&spec).unwrap();
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                features: candle_gen::gen_core::FeatureParams { pag: Some(PagConfig {
                    scale: 1.5,
                    layers: Some(vec![0, 28, 56]),
                    ..Default::default()
                }), ..Default::default() },
                ..Default::default()
            })
            .is_ok());
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                features: candle_gen::gen_core::FeatureParams { pag: Some(PagConfig {
                    scale: 0.0,
                    layers: Some(vec![57]),
                    ..Default::default()
                }), ..Default::default() },
                ..Default::default()
            })
            .is_err());

        let cpu = Device::Cpu;
        let (b, h, txt_seq, img_seq, d) = (1usize, 2usize, 3usize, 4usize, 4usize);
        let total = txt_seq + img_seq;
        let q = Tensor::randn(0f32, 1f32, (b, h, total, d), &cpu).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, total, d), &cpu).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, total, d), &cpu).unwrap();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let (o_txt, o_img) =
            crate::transformer::pag_joint_attention_post_rope(&q, &k, &v, txt_seq, plan).unwrap();
        let full = crate::transformer::scaled_dot_product_attention_planned(&q, &k, &v, plan)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .flatten_from(2)
            .unwrap();
        let full_txt = full.narrow(1, 0, txt_seq).unwrap();
        let diff_txt = (&o_txt - &full_txt)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff_txt < 1e-6,
            "Q_txt must match full SDPA, diff={diff_txt}"
        );
        let want_img = v
            .narrow(2, txt_seq, img_seq)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .flatten_from(2)
            .unwrap();
        assert_eq!(
            o_img
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            want_img.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// Shared real-weight sequential-residency A/B body for FLUX.1 dev and schnell (sc-12138). Runs ONE
    /// probed generation whose mode is carried by `GenerationMemory::stage_residency`, writes raw RGB
    /// pixels to `FLUX_OUT`, and prints the device peak. Run it TWICE in SEPARATE processes (resident vs
    /// sequential): pixels must be byte-identical and the sequential peak materially lower because the
    /// ~9 GB T5-XXL drops before the DiT loads. Separate processes are required because candle's cudarc
    /// caching allocator never returns pages to the driver.
    #[cfg(feature = "cuda")]
    fn run_probed_offload_ab(
        label: &str,
        dir_env: &str,
        load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
        steps: u32,
    ) {
        let dir = std::env::var(dir_env).unwrap_or_else(|_| {
            panic!("set {dir_env} to a real-file (hardlink-staged) {label} snapshot")
        });
        let out = std::env::var("FLUX_OUT").expect("set FLUX_OUT to the pixel-dump path");
        let rung = std::env::var("FLUX_MEMORY_RUNG").unwrap_or_else(|_| "resident".into());
        let (strategy, memory) = match rung.as_str() {
            "resident" => (gen_core::MemoryStrategy::Resident, Default::default()),
            "staged" => (
                gen_core::MemoryStrategy::StagedResidency,
                gen_core::GenerationMemory {
                    stage_residency: true,
                    ..Default::default()
                },
            ),
            "bounded-decode" => (
                gen_core::MemoryStrategy::BoundedDecode,
                gen_core::GenerationMemory {
                    stage_residency: true,
                    tile_vae_decode: true,
                    decode_tile_edge: Some(memory_strategy::DECODE_TILE_EDGE),
                    decode_overlap: Some(memory_strategy::DECODE_OVERLAP),
                    ..Default::default()
                },
            ),
            "bounded-attention" => (
                gen_core::MemoryStrategy::BoundedAttention,
                gen_core::GenerationMemory {
                    stage_residency: true,
                    tile_vae_decode: true,
                    chunk_attention: true,
                    decode_tile_edge: Some(memory_strategy::DECODE_TILE_EDGE),
                    decode_overlap: Some(memory_strategy::DECODE_OVERLAP),
                    attention_chunk_size: Some(memory_strategy::ATTENTION_CHUNK_SIZE),
                    ..Default::default()
                },
            ),
            "bounded-transformer" => (
                gen_core::MemoryStrategy::BoundedTransformerResidency,
                gen_core::GenerationMemory {
                    stage_residency: true,
                    tile_vae_decode: true,
                    chunk_attention: true,
                    stream_transformer_blocks: true,
                    decode_tile_edge: Some(memory_strategy::DECODE_TILE_EDGE),
                    decode_overlap: Some(memory_strategy::DECODE_OVERLAP),
                    attention_chunk_size: Some(memory_strategy::ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(
                        memory_strategy::DEFAULT_TRANSFORMER_WINDOW as u32,
                    ),
                    transformer_window_component: Some(gen_core::TransformerComponent::Dit),
                    ..Default::default()
                },
            ),
            other => panic!("unknown FLUX_MEMORY_RUNG `{other}`"),
        };
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        if strategy == gen_core::MemoryStrategy::BoundedTransformerResidency {
            spec = spec.with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        }
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            steps: Some(steps),
            seed: Some(42),
            count: 1,
            memory: Some(memory),
            ..Default::default()
        };
        assert!(
            candle_gen::testkit::reset_cuda_mempool_high_water(0),
            "reset CUDA live-allocation high-water"
        );
        // Reject a contaminated GPU before spending minutes loading and generating. The final
        // report is still validated below so evidence cannot become trustworthy merely because
        // the device happened to be idle at this boundary.
        let mut probe = candle_gen::testkit::VramProbe::start_rendered().assert_idle(1.0);
        let load_phase = probe.phase();
        let g = load(&spec).unwrap_or_else(|e| panic!("load {label}: {e}"));
        probe.end_load(load_phase);
        let generate_phase = probe.phase();
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        probe.end_gen(generate_phase);
        let report = probe.report().assert_trustworthy(1.0);
        let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
            .expect("read CUDA live-allocation high-water");
        assert!(
            live_peak_bytes > 0,
            "CUDA live-allocation peak must be positive"
        );
        let img = match output {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        std::fs::write(&out, &img.pixels).expect("write pixels");
        let contract = memory_strategy::provider_contract(label, &spec).expect("memory contract");
        let selection = gen_core::MemorySelection {
            strategy,
            parameters: gen_core::MemoryStrategyParameters {
                stage_residency: None,
                decode_tile_edge: memory.decode_tile_edge,
                decode_overlap: memory.decode_overlap,
                attention_chunk_size: memory.attention_chunk_size,
                transformer_window_size: memory.transformer_window_size,
                transformer_window_component: memory.transformer_window_component,
            },
            tier: memory_strategy::resolved_numeric_tier(&spec, label).expect("numeric tier"),
        };
        eprintln!(
            "{}",
            candle_gen::testkit::memory_evidence_v1_line(
                candle_gen::testkit::MemoryEvidenceProbe {
                    resolved_route: label,
                    declared_calibration: candle_gen::testkit::expected_memory_calibration(
                        spec.load_shape,
                    ),
                    observed_calibration: contract.calibration.clone().expect("calibration"),
                    // Packed q4/q8 snapshots intentionally leave `LoadSpec::quantize` empty:
                    // their actual tier comes from transformer/config.json. Preserve the exact
                    // tier already resolved for admission instead of restamping the request hint.
                    tier: selection.tier,
                    load_shape: spec.load_shape,
                    mode: gen_core::MemoryMode::TextToImage,
                    overlay: None,
                    geometry: gen_core::MemoryGeometry {
                        width: req.width,
                        height: req.height,
                        batch: req.count,
                        frames: 1,
                        reference_count: 0,
                    },
                    strategy,
                    engaged_composition: contract.engaged_composition(strategy),
                    parameters: selection.parameters,
                    observed_peak_bytes: live_peak_bytes,
                    harness_version: "candle-flux1-memory-ladder-v1",
                    output_bytes: &img.pixels,
                }
            )
        );
        eprintln!(
            "MEMORY_EVIDENCE_DIAGNOSTIC gpu={} {report} bytes={} {}x{} out={out}",
            candle_gen::testkit::probe_gpu(),
            img.pixels.len(),
            img.width,
            img.height
        );
    }

    /// FLUX.1-dev real-weight A/B (epic 10765 Phase 1, sc-10769/sc-12138). Needs a real-file snapshot
    /// in `FLUX_DEV_DIR` and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux_dev_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux1_dev", "FLUX_DEV_DIR", load_dev, 8);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED tests::flux_dev_probed_generate_for_offload_ab");
    }

    /// FLUX.1-schnell sibling required by the live worker gate (sc-12138). Needs a real-file snapshot
    /// in `FLUX_SCHNELL_DIR` and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux_schnell_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux1_schnell", "FLUX_SCHNELL_DIR", load_schnell, 4);

        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED tests::flux_schnell_probed_generate_for_offload_ab"
        );
    }
}

#[cfg(test)]
mod execution_policy_tests {
    use super::*;
    #[test]
    fn execution_guidance_distinguishes_defaults_overrides_and_loaded_variant() {
        let report = |variant: Variant, requested| {
            candle_gen::testkit::execution_policy(|| {
                variant.resolved_guidance(requested);
            })
        };
        assert_eq!(report(Variant::Dev, None), report(Variant::Dev, Some(3.5)));
        assert_ne!(report(Variant::Dev, None), report(Variant::Dev, Some(4.5)));
        assert_eq!(
            report(Variant::Schnell, None),
            report(Variant::Schnell, Some(99.0))
        );
        assert_ne!(
            descriptor_for(Variant::Schnell).id,
            descriptor_for(Variant::Dev).id
        );
        assert_ne!(report(Variant::Dev, None), report(Variant::Schnell, None));
    }

    #[test]
    fn sw6939_candle_flux_inactive_hook_is_byte_identical_and_two_phase_low_res_executes_conformance(
    ) {
        use candle_core::{DType, Device, Tensor};
        use candle_transformers::models::flux::sampling::State;
        use gen_core::render::RenderPhaseIdentity;
        use gen_core::sampling::{FlowModelSampling, TimestepConvention};
        use gen_core::{
            BoundaryState, Image, LowResStartSpec, MemoryGeometry, MemoryPhase, MemoryRequestScope,
            MemoryRunOutcome, RenderRun,
        };
        use std::cell::RefCell;

        let dev = Device::Cpu;
        let base_req = GenerationRequest {
            prompt: "a lighthouse in a storm".into(),
            width: 512,
            height: 512,
            count: 1,
            steps: Some(4),
            seed: Some(42),
            sampler: Some("dpmpp_2m".into()),
            ..Default::default()
        };
        let active_req = GenerationRequest {
            features: candle_gen::gen_core::FeatureParams { low_res_start: Some(LowResStartSpec::new(2, Some(0.5))), ..Default::default() },
            ..base_req.clone()
        };

        // 1. Schedule & flow_mu invariance across both Variant::Dev and Variant::Schnell.
        for variant in [Variant::Dev, Variant::Schnell] {
            let plan_off = compile_base_plan(variant, &base_req).unwrap();
            let plan_on = compile_base_plan(variant, &active_req).unwrap();
            assert_eq!(
                plan_off.denoise_sigmas(),
                plan_on.denoise_sigmas(),
                "{variant:?}: full-pass sigma schedule and mu must stay invariant under low_res_start"
            );
            assert_eq!(plan_off.denoise_spans(), vec![0..4]);
            assert_eq!(plan_on.denoise_spans(), vec![0..2, 2..4]);
            assert_eq!(plan_on.phase_output_size(0), (256, 256));
            assert_eq!(plan_on.phase_output_size(1), (512, 512));
        }

        // 2. Ordinary single-source img2img strength truncation resolves M = executed_steps before checking K.
        let src_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        let img2img_ok = GenerationRequest {
            steps: Some(10),
            strength: Some(0.6),
            source_image: Some(src_img),
            features: candle_gen::gen_core::FeatureParams { low_res_start: Some(LowResStartSpec::new(3, Some(0.5))), ..Default::default() },
            ..base_req.clone()
        };
        let img2img_plan = compile_base_plan(Variant::Dev, &img2img_ok).unwrap();
        assert_eq!(img2img_plan.executed_steps(), 4);
        assert_eq!(img2img_plan.denoise_spans(), vec![0..3, 3..4]);

        let img2img_bad_k = GenerationRequest {
            features: candle_gen::gen_core::FeatureParams { low_res_start: Some(LowResStartSpec::new(4, Some(0.5))), ..Default::default() },
            ..img2img_ok
        };
        assert!(validate_base_request(Variant::Dev, &img2img_bad_k).is_err());

        // 3. BoundaryState::XSigma is refused by transition_candle_flux_boundary_state (cannot double-noise).
        let plan_on = compile_base_plan(Variant::Dev, &active_req).unwrap();
        let low_res_plan = *plan_on.low_res_start().unwrap();
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let dummy_low = Tensor::zeros((1, 256, 64), DType::F32, &dev).unwrap();
        assert!(transition_candle_flux_boundary_state(
            BoundaryState::XSigma(dummy_low),
            &low_res_plan,
            plan_on.denoise_sigmas()[2],
            &ms,
            42,
            0,
            0,
            0,
            &dev,
            DType::F32,
        )
        .is_err());

        // 4. Inactive hook byte-identity: execute_candle_flux_denoise_trajectory with low_res_start: None
        // produces byte-identical latents, evaluation trace, and progress to run_flow_sampler_spans.
        let plan_off = compile_base_plan(Variant::Dev, &base_req).unwrap();
        let noise_full = flux1_load::seeded_noise(42, 16, 64, 64, &dev, DType::F32).unwrap();
        let t5_dummy = Tensor::zeros((1, 16, 4096), DType::F32, &dev).unwrap();
        let clip_dummy = Tensor::zeros((1, 768), DType::F32, &dev).unwrap();
        let state_full = State::new(&t5_dummy, &clip_dummy, &noise_full).unwrap();

        let trace_off = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
        let mut progress_off = Vec::new();
        let run_off = RenderRun::new(&plan_off, &base_req.cancel, None);
        run_off.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_off = run_off
            .images_with_denoise_phases(
                &mut |p| progress_off.push(p),
                |_| Ok::<_, candle_gen::CandleError>(state_full.img.clone()),
                |seed, lat, prog| {
                    execute_candle_flux_denoise_trajectory(
                        &run_off,
                        &base_req,
                        seed,
                        lat,
                        prog,
                        |x_in, step_idx, timestep, _guidance, pw, ph| {
                            trace_off.borrow_mut().push((
                                step_idx,
                                pw,
                                ph,
                                x_in.dim(1)?,
                                timestep.to_bits(),
                            ));
                            Ok((x_in * 0.15)?)
                        },
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);

        let trace_ref = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
        let mut progress_ref = Vec::new();
        let run_ref = RenderRun::new(&plan_off, &base_req.cancel, None);
        run_ref.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_ref = run_ref
            .images_with_denoise_phases(
                &mut |p| progress_ref.push(p),
                |_| Ok::<_, candle_gen::CandleError>(state_full.img.clone()),
                |seed, lat, prog| {
                    let mut eval_idx = 0usize;
                    candle_gen::run_flow_sampler_with_phase_transition_spans(
                        Some(plan_off.sampler()),
                        TimestepConvention::Sigma,
                        plan_off.denoise_sigmas(),
                        &plan_off.denoise_spans(),
                        Some(&run_ref),
                        false,
                        lat,
                        seed,
                        &base_req.cancel,
                        prog,
                        None,
                        |x_in, timestep| {
                            let step_idx = eval_idx;
                            eval_idx += 1;
                            trace_ref.borrow_mut().push((
                                step_idx,
                                512,
                                512,
                                x_in.dim(1)?,
                                timestep.to_bits(),
                            ));
                            Ok((x_in * 0.15)?)
                        },
                        None,
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);

        let vec_off: Vec<f32> = out_off.flatten_all().unwrap().to_vec1().unwrap();
        let vec_ref: Vec<f32> = out_ref.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vec_off, vec_ref);
        assert_eq!(*trace_off.borrow(), *trace_ref.borrow());
        assert_eq!(progress_off, progress_ref);

        // 5. Active two-phase execution with FluxMemoryScope:
        // Phase 0 (steps 0..2) runs at 256x256 (256 tokens), boundary K=2 performs read-only endpoint
        // evaluation at 256x256, transitions in fp32 to 512x512 (1024 tokens), resets Dpmpp2m history,
        // and Phase 1 (steps 2..4) runs at 512x512 (1024 tokens).
        struct RecordingFluxScope {
            inner: memory_strategy::FluxMemoryScope,
            phase_identities: Vec<(MemoryPhase, RenderPhaseIdentity)>,
        }
        impl MemoryRequestScope for RecordingFluxScope {
            fn configure_request(&mut self, req: &mut GenerationRequest) -> gen_core::Result<()> {
                self.inner.configure_request(req)
            }
            fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
                self.inner.enter_phase(phase)
            }
            fn enter_render_phase(
                &mut self,
                phase: MemoryPhase,
                identity: &RenderPhaseIdentity,
            ) -> gen_core::Result<()> {
                self.inner.enter_render_phase(phase, identity)?;
                self.phase_identities.push((phase, *identity));
                Ok(())
            }
            fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
                self.inner.leave_phase(phase)
            }
            fn configure_decode(
                &mut self,
                tile_edge: u32,
                overlap: u32,
                geometry: MemoryGeometry,
            ) -> gen_core::Result<()> {
                self.inner.configure_decode(tile_edge, overlap, geometry)
            }
            fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
                self.inner.configure_attention(chunk_size)
            }
            fn materialize_transformer_window(
                &mut self,
                first_block: u32,
                block_count: u32,
            ) -> gen_core::Result<()> {
                self.inner
                    .materialize_transformer_window(first_block, block_count)
            }
            fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
                self.inner.finish(outcome)
            }
        }

        let spec = LoadSpec::new(gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ))
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        let contract = memory_strategy::provider_contract(FLUX1_DEV_ID, &spec).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            gen_core::MemoryStrategy::Resident,
            gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            gen_core::MemoryBehaviorRoute {
                mode: gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry.width = 512;
        context.geometry.height = 512;
        let mut scope = RecordingFluxScope {
            inner: memory_strategy::FluxMemoryScope::new(
                FLUX1_DEV_ID,
                dev.clone(),
                &contract,
                &context,
            ),
            phase_identities: Vec::new(),
        };

        let noise_low = flux1_load::seeded_noise(42, 16, 32, 32, &dev, DType::F32).unwrap();
        let state_low = State::new(&t5_dummy, &clip_dummy, &noise_low).unwrap();
        assert_eq!(state_low.img.dims(), &[1, 256, 64]);
        assert_eq!(state_low.img_ids.dims(), &[1, 256, 3]);
        assert_eq!(state_full.img_ids.dims(), &[1, 1024, 3]);

        let captured_previews =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::<(u32, u32, u32, u32)>::new()));
        let preview_capture = captured_previews.clone();
        let mut active_req = active_req;
        active_req.preview = gen_core::PreviewSink::new(move |frame: gen_core::PreviewFrame| {
            candle_gen::lock_recover(&preview_capture).push((
                frame.current,
                frame.total,
                frame.image.width,
                frame.image.height,
            ));
        });

        let trace_on = RefCell::new(Vec::<(usize, u32, u32, usize)>::new());
        let mut progress_on = Vec::new();
        let run_on = RenderRun::new(&plan_on, &active_req.cancel, Some(&mut scope));
        run_on.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_on = run_on
            .images_with_denoise_phases(
                &mut |p| progress_on.push(p),
                |_| Ok::<_, candle_gen::CandleError>(state_low.img.clone()),
                |seed, lat, prog| {
                    execute_candle_flux_denoise_trajectory(
                        &run_on,
                        &active_req,
                        seed,
                        lat,
                        prog,
                        |x_in, step_idx, _timestep, _guidance, pw, ph| {
                            // Verify per-phase State/img_ids derivation matches the phase geometry and token count.
                            let lat_h = (ph as usize).div_ceil(16) * 2;
                            let lat_w = (pw as usize).div_ceil(16) * 2;
                            let dummy = Tensor::zeros((1, 16, lat_h, lat_w), DType::F32, &dev)?;
                            let phase_state = State::new(&t5_dummy, &clip_dummy, &dummy)?;
                            let expected_tokens = ((ph / 16) * (pw / 16)) as usize;
                            assert_eq!(x_in.dim(1)?, expected_tokens);
                            assert_eq!(phase_state.img_ids.dims(), &[1, expected_tokens, 3]);
                            trace_on.borrow_mut().push((step_idx, pw, ph, x_in.dim(1)?));
                            Ok((x_in * 0.15)?)
                        },
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        assert_eq!(out_on.dims(), &[1, 1024, 64]);
        // Preview frames track Phase-0 (256/8 = 32x32) and Phase-1 (512/8 = 64x64) geometries,
        // and the read-only boundary endpoint evaluation does not emit an extra frame.
        assert_eq!(
            *candle_gen::lock_recover(&captured_previews),
            vec![
                (1, 4, 32, 32),
                (2, 4, 32, 32),
                (3, 4, 64, 64),
                (4, 4, 64, 64),
            ]
        );
        assert_eq!(
            progress_on,
            vec![
                Progress::Step {
                    current: 1,
                    total: 4
                },
                Progress::Step {
                    current: 2,
                    total: 4
                },
                Progress::Step {
                    current: 3,
                    total: 4
                },
                Progress::Step {
                    current: 4,
                    total: 4
                },
            ]
        );
        assert_eq!(
            *trace_on.borrow(),
            vec![
                (0, 256, 256, 256),
                (1, 256, 256, 256),
                (2, 256, 256, 256),
                (2, 512, 512, 1024),
                (3, 512, 512, 1024),
            ]
        );
        let denoise_identities: Vec<RenderPhaseIdentity> = scope
            .phase_identities
            .iter()
            .filter_map(|(phase, id)| (*phase == MemoryPhase::Denoise).then_some(*id))
            .collect();
        assert_eq!(denoise_identities.len(), 2);
        assert_eq!(
            (
                denoise_identities[0].pass,
                denoise_identities[0].phase,
                denoise_identities[0].geometry.width,
                denoise_identities[0].geometry.height
            ),
            (0, 0, 256, 256)
        );
        assert_eq!(
            (
                denoise_identities[1].pass,
                denoise_identities[1].phase,
                denoise_identities[1].geometry.width,
                denoise_identities[1].geometry.height
            ),
            (0, 1, 512, 512)
        );
    }

    #[test]
    fn sw6940_candle_flux_endpoint_and_previous_evaluation_end_to_end_and_provenance_report() {
        use candle_core::{DType, Device, Tensor};
        use candle_transformers::models::flux::sampling::State;
        use gen_core::{EstimateSource, EvaluationKind, LowResStartSpec, RenderRun};
        use std::cell::RefCell;

        let dev = Device::Cpu;
        let t5_dummy = Tensor::zeros((1, 16, 4096), DType::F32, &dev).unwrap();
        let clip_dummy = Tensor::zeros((1, 768), DType::F32, &dev).unwrap();
        let noise_low = flux1_load::seeded_noise(42, 16, 32, 32, &dev, DType::F32).unwrap();
        let state_low = State::new(&t5_dummy, &clip_dummy, &noise_low).unwrap();

        for (sampler_name, expected_p0_evals, expected_p1_evals) in [
            ("euler", 2usize, 2usize),
            ("dpmpp_2m", 2usize, 2usize),
            ("heun", 4usize, 3usize),
            ("dpmpp_sde", 4usize, 3usize),
        ] {
            let make_req = |source: EstimateSource| GenerationRequest {
                prompt: "a lighthouse in a storm".into(),
                width: 512,
                height: 512,
                count: 1,
                steps: Some(4),
                seed: Some(42),
                sampler: Some(sampler_name.into()),
                features: candle_gen::gen_core::FeatureParams { low_res_start: Some(
                    LowResStartSpec::new(2, Some(0.5)).with_estimate_source(source),
                ), ..Default::default() },
                ..Default::default()
            };

            let req_endpoint = make_req(EstimateSource::EndpointEvaluation);
            let req_previous = make_req(EstimateSource::PreviousEvaluation);
            let plan_endpoint = compile_base_plan(Variant::Dev, &req_endpoint).unwrap();
            let plan_previous = compile_base_plan(Variant::Dev, &req_previous).unwrap();

            let report_endpoint = plan_endpoint
                .low_res_start_execution_report("bf16", 2, "base_dit")
                .unwrap()
                .unwrap();
            let report_previous = plan_previous
                .low_res_start_execution_report("bf16", 2, "base_dit")
                .unwrap()
                .unwrap();

            assert_eq!(report_endpoint.estimator, "endpoint_evaluation_v1");
            assert_eq!(
                report_endpoint.estimate_provenance.evaluation_kind,
                EvaluationKind::BoundaryEstimate
            );
            assert_eq!(report_endpoint.estimate_provenance.step_index, 2);
            assert_eq!(
                report_endpoint.estimate_provenance.sigma.to_bits(),
                plan_endpoint.denoise_sigmas()[2].to_bits()
            );
            assert_eq!(report_endpoint.boundary_evaluations, 1);
            assert_eq!(
                report_endpoint.total_model_evaluations,
                (expected_p0_evals + 1 + expected_p1_evals) as u32
            );

            assert_eq!(report_previous.estimator, "previous_evaluation_v1");
            assert_eq!(
                report_previous.estimate_provenance.evaluation_kind,
                EvaluationKind::GuidedPrimaryX0
            );
            assert_eq!(report_previous.estimate_provenance.step_index, 1);
            assert_eq!(
                report_previous.estimate_provenance.sigma.to_bits(),
                plan_previous.denoise_sigmas()[1].to_bits()
            );
            assert_eq!(report_previous.boundary_evaluations, 0);
            assert_eq!(
                report_previous.total_model_evaluations,
                (expected_p0_evals + expected_p1_evals) as u32
            );
            assert_eq!(
                report_previous
                    .estimate_provenance
                    .retained_primary_estimate_bytes,
                256 * 64 * 2
            );

            let run_policy = |req: &GenerationRequest, plan: &gen_core::RenderPlan| {
                let trace = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
                let mut progress = Vec::new();
                let run = RenderRun::new(plan, &req.cancel, None);
                run.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
                let out = run
                    .images_with_denoise_phases(
                        &mut |p| progress.push(p),
                        |_| Ok::<_, candle_gen::CandleError>(state_low.img.clone()),
                        |seed, lat, prog| {
                            execute_candle_flux_denoise_trajectory(
                                &run,
                                req,
                                seed,
                                lat,
                                prog,
                                |x_in, step_idx, timestep, _guidance, pw, ph| {
                                    trace.borrow_mut().push((
                                        step_idx,
                                        pw,
                                        ph,
                                        x_in.dim(1)?,
                                        timestep.to_bits(),
                                    ));
                                    Ok((x_in * 0.15)?)
                                },
                            )
                        },
                        |_, lat, _| Ok(lat),
                    )
                    .unwrap()
                    .remove(0);
                let vec_out: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
                (vec_out, trace.into_inner(), progress)
            };

            let (out_ep, trace_ep, prog_ep) = run_policy(&req_endpoint, &plan_endpoint);
            let (out_pr, trace_pr, prog_pr) = run_policy(&req_previous, &plan_previous);

            assert_eq!(out_ep.len(), 1024 * 64);
            assert_eq!(out_pr.len(), 1024 * 64);
            assert!(out_ep.iter().all(|v| v.is_finite()));
            assert!(out_pr.iter().all(|v| v.is_finite()));
            assert_ne!(
                out_ep, out_pr,
                "{sampler_name}: EndpointEvaluation and PreviousEvaluation must produce distinct trajectories"
            );
            assert_eq!(prog_ep, prog_pr);
            assert_eq!(
                &trace_ep[..expected_p0_evals],
                &trace_pr[..expected_p0_evals],
                "{sampler_name}: Phase-0 evaluations must be bit-identical between EndpointEvaluation and PreviousEvaluation"
            );
            assert_eq!(trace_ep.len(), expected_p0_evals + 1 + expected_p1_evals);
            assert_eq!(trace_pr.len(), expected_p0_evals + expected_p1_evals);
        }
    }
}
