//! `QwenImageControl` — the Qwen-Image **ControlNet (strict pose)** variant (epic 3401 / sc-8267),
//! registered as its own `Generator` (`qwen_image_control`) via the alibaba-pai
//! `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (a VACE-style Fun-Controlnet-Union, Apache-2.0,
//! ungated — it **replaces** the retired InstantX `Qwen-Image-ControlNet-Union` on the Qwen path).
//!
//! Identical to [`crate::model::QwenImage`] (T2I) except it also loads a [`QwenFunControlBranch`]
//! VACE control branch and `generate` threads a VAE-encoded pose skeleton through it: each denoise
//! step the control branch computes 5 per-block hints from the post-embedder streams + the (constant)
//! 132-ch packed control context, which the frozen base 60-layer MMDiT adds into its image stream at
//! `control_layers = [0, 12, 24, 36, 48]` scaled by the request's control scale. [`load`] needs the
//! base snapshot (`spec.weights`) **and** the control checkpoint (`spec.control`); it applies both
//! dense, then quantizes base + control together (Q4/Q8, transformer-only — the fork's
//! overlay-then-quantize ordering). Identity comes from a character LoRA on the **base**
//! (`spec.adapters`); the control branch is never an adapter target.
//!
//! Accepts the three structural control signals — **pose/canny/depth** — which the 2512-Fun Union
//! admits via one input-agnostic VACE control path (no mode index; sc-8250). **Base pose-from-prompt**
//! (composing with the edit model is a later reach). Pose parity vs the fork's
//! `pipeline_qwenimage_control` is gated by `tests/control_real_weights.rs` (`#[ignore]`, M-series).
//!
//! Component residency (epic 10834 Phase 1, sc-11006 — the fan-out sibling of the T2I sc-11000):
//! under [`OffloadPolicy::Sequential`] the Qwen2.5-VL text encoder (~15 GB) is dropped after the
//! text-encode phase, so peak unified memory is bounded to `max(text-encoder, DiT+control+VAE)`
//! instead of the sum. The control branch is an **extra** heavy component (loaded + quantized with the
//! base transformer), so it stays on the heavy side of the split; the pose VAE-encode uses the VAE
//! (not the text encoder) and so runs after the drop, byte-identically.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    gen_core, require_base_dir, require_control, AcceptedControlKinds, ConditioningKind,
    ControlBranch, ControlKind, Error, GenerationOutput, GenerationRequest, Generator,
    LatentDecoder, LoadSpec, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Residency,
    Result, WeightsSource, VAE_COMPONENT,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_wan::OwnedWanSingleFrameDecoder;
use std::path::Path;

use crate::control_transformer::QwenFunControlBranch;
use crate::loader;
use crate::model::{pag_capability, validate_request, SIZE_MULTIPLE};
use crate::pipeline::{
    create_noise, encode_fun_control_context, encode_prompt, negative_or_fallback,
    resolve_run_params, PID_BACKBONE,
};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Registry id for the Qwen-Image ControlNet (strict pose) variant.
pub const MODEL_ID: &str = "qwen_image_control";

/// Primary typed route declaration for Qwen-Image ControlNet (`REG-1`, `AP5`).
pub(crate) fn route_qwen_image_control() -> gen_core::RouteRecord {
    let mut route = crate::model::route_qwen_image();
    route.route_id = MODEL_ID;
    route.structure.conditioning.kinds = vec![ConditioningKind::Control];
    route.structure.conditioning.control_kinds = Some(accepted_kinds());
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.conditioning.modules = vec!["qwen_fun_controlnet_union"];
    route.structure.denoiser = gen_core::ModelBinding::family_model(
        gen_core::FamilyModelSpec::new("qwen_image_mmdit_control").with_pag(pag_capability()),
    );
    route.structure.adapters.install_policy = gen_core::AdapterInstallPolicy::BaseDenoiserOnly;
    route.structure.weight_plan.requires_control_overlay = true;
    route.structure.output.preview = gen_core::PreviewSupport::Unsupported;
    route.bindings = gen_core::RouteBindings::generator(descriptor, |spec| {
        load(spec).map_err(Into::into)
    })
    .with_footprint(crate::model::control_component_footprint)
    .with_render_plan(compile_render_plan)
    .with_memory_registration(MEMORY_REGISTRATION)
    .with_memory_fixture(gen_core::mlx_memory_contract_surface_specs, |spec| {
        crate::memory_strategy::weights_free_memory_strategy_contract(MODEL_ID, spec)
    })
    .with_memory_behavior_registration(MEMORY_BEHAVIOR_REGISTRATION);
    route
}

/// The control variant's identity + capabilities — the base Qwen-Image T2I surface (true CFG /
/// negative prompt / guidance / Lightning) plus the **required** `Control` (pose skeleton)
/// conditioning. LoRA/LoKr (character identity) is on the base transformer.
pub fn descriptor() -> ModelDescriptor {
    route_qwen_image_control().compute_descriptor()
}

/// A loaded control generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// heavy-component residency strategy (base components + the control branch) — sc-11006.
pub struct QwenImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11006; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the Qwen2.5-VL
    /// text encoder + DiT + control branch + VAE warm; `Sequential` holds only the per-phase loader
    /// closures and re-loads per generation in phase order (encode → **drop the text encoder** →
    /// denoise/decode), bounding peak unified memory to `max(text-encoder, DiT+control+VAE)` instead of
    /// the sum. The [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel
    /// checks, and the error-safe cache flush.
    residency: Residency<QwenTextEncoder, QwenControlHeavyOwned>,
    memory_strategy: gen_core::MemoryProviderContract,
    precision: Precision,
    quant: Option<Quant>,
    adapters: Vec<mlx_gen::AdapterSpec>,
    loaded_spec: LoadSpec,
}

/// The heavy render-phase components — the base MMDiT transformer, the VACE control branch, the VAE,
/// and the optional PiD decoder — everything but the text encoder. Owned by the `Resident` components
/// or by a `Sequential` generate. The control branch is an **extra** component vs T2I: it is loaded +
/// quantized alongside the base transformer and lives on the heavy side of the residency split.
struct QwenControlHeavyOwned {
    transformer: QwenTransformer,
    controlnet: QwenFunControlBranch,
    vae: QwenVae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845); see [`crate::model::QwenImage`].
    pid: Option<PidEngine>,
    alternate_decoder: Option<OwnedWanSingleFrameDecoder>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically whether
/// they are held resident or were just loaded by the `Sequential` path (candle's `DitRef`).
struct QwenControlHeavy<'a> {
    transformer: &'a QwenTransformer,
    controlnet: &'a QwenFunControlBranch,
    vae: &'a QwenVae,
    pid: Option<&'a PidEngine>,
    alternate_decoder: Option<&'a OwnedWanSingleFrameDecoder>,
}

impl QwenControlHeavyOwned {
    fn as_ref(&self) -> QwenControlHeavy<'_> {
        QwenControlHeavy {
            transformer: &self.transformer,
            controlnet: &self.controlnet,
            vae: &self.vae,
            pid: self.pid.as_ref(),
            alternate_decoder: self.alternate_decoder.as_ref(),
        }
    }
}

/// Construct a [`QwenImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a base `Qwen/Qwen-Image-2512` snapshot directory and `spec.control`
/// (required) the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (a single
/// `.safetensors` `File`, or a `Dir`). Base + control load dense (bf16); `spec.quantize` (Q4/Q8) then
/// quantizes both transformers (group_size 64). The text encoder + VAE stay dense (the fork's
/// transformer-only quant scope — see [`crate::model::load`]).
///
/// Component residency (epic 10834 Phase 1, sc-11006; hoisted to the shared [`Residency::from_policy`]
/// seam in sc-11126, F-180): `Resident` (default) builds every heavy component now via
/// `build_residency` and holds it warm; `Sequential` keeps only the spec and re-loads per generate in
/// phase order (encode → drop the text encoder → denoise/decode) to bound peak memory to
/// `max(text-encoder, DiT+control+VAE)`. Both use the same per-phase loaders, so the components are
/// byte-identical.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    mlx_gen::gen_core::reject_unknown_components(spec, &[VAE_COMPONENT], MODEL_ID)?;
    mlx_gen_wan::validate_selected_single_frame_decoder(spec, &descriptor())?;
    // Resolve the base dir + required control checkpoint up front — fail-fast for BOTH policies — then
    // the always-warm tokenizer, then the shared [`build_residency`] dispatch.
    let (root, _control) = resolve_base_and_control(spec)?;
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), MODEL_ID)?
        {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    let text_encoder_source = crate::active_encoder_contract().source_for_load(spec, root)?;
    text_encoder_source.load_time_quant_bits(None, MODEL_ID)?;
    let tokenizer = loader::load_validated_tokenizer(&text_encoder_source)?;
    Ok(Box::new(QwenImageControl {
        descriptor: descriptor(),
        tokenizer,
        residency: build_residency_with_source(spec, text_encoder_source)?,
        memory_strategy: crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?,
        precision: spec.precision,
        quant: spec.quantize,
        adapters: spec.adapters.clone(),
        loaded_spec: spec.clone(),
    }))
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-11006; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in one
/// place rather than a bespoke per-crate copy. `Resident` eager-loads the text encoder + heavy bundle
/// now (the heavy loader with `use_pid = true`, loading any PiD overlay once and reusing it);
/// `Sequential` captures the two per-phase loaders and loads nothing now, deferring each to
/// [`Residency::run`]. Both use the same [`load_text_encoder_only`] / [`load_heavy`], so the `Resident`
/// composition is byte-identical to the pre-seam one. The deferral is weight-free-testable: under
/// `Sequential` this touches no component weights, so a dispatch that ignored `offload_policy` would
/// eager-load and fail the "Sequential defers" unit test.
fn build_residency_with_source(
    spec: &LoadSpec,
    text_encoder_source: mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<Residency<QwenTextEncoder, QwenControlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    crate::model::residency_from_spec(
        spec,
        move || {
            let (_root, _control) = resolve_base_and_control(&spec_text)?;
            load_text_encoder_only(&text_encoder_source)
        },
        move |use_pid| {
            let (root, control) = resolve_base_and_control(&spec_heavy)?;
            load_heavy(&spec_heavy, root, control, use_pid)
        },
    )
}

/// Precision guard (only dense bf16 is wired) + base-snapshot-dir resolution + the **required**
/// control-checkpoint resolution, shared by [`load`] and [`build_residency`]'s per-phase loaders
/// (sc-11006). Preserves the original message order: a single-file base is rejected first (via
/// [`require_base_dir`]), then a missing control checkpoint (via [`require_control`]).
fn resolve_base_and_control(spec: &LoadSpec) -> Result<(&Path, &WeightsSource)> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_control: only dense bf16 is wired in the Rust port (drop the precision \
             override)"
                .into(),
        ));
    }
    // Shared load boilerplate (sc-8241): the base must be a snapshot dir, the control checkpoint is
    // required. The model id + labels keep the messages byte-identical to the hand-written originals.
    let root = require_base_dir(spec, MODEL_ID, "a base snapshot directory")?;
    let control = require_control(spec, MODEL_ID, "Qwen-Image-2512-Fun-Controlnet-Union")?;
    Ok((root, control))
}

/// Load the Qwen2.5-VL text encoder — the phase-A component dropped first under `Sequential`. Never
/// quantized (the fork's transformer-only quant scope), so the `Resident` and `Sequential` paths build
/// byte-identical encoders.
fn load_text_encoder_only(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<QwenTextEncoder> {
    source.read_unchanged(loader::load_text_encoder_from_source)
}

/// Load the heavy render-phase components — the base MMDiT transformer, the VACE control branch (both
/// Q4/Q8 + the base's LoRA/LoKr residuals), the VAE, and the optional PiD overlay — everything but the
/// text encoder. Factored so the `Sequential` path loads these AFTER the encoder is dropped (bounding
/// peak to `max(text-encoder, DiT+control+VAE)`). The overlay-then-quantize order (dense base + dense
/// control, THEN quantize both) matches the pre-sc-11006 `load`; the components are independent of the
/// text encoder (separate weight files, deterministic RNG-free quant), so the `Resident` composition
/// is byte-identical.
fn load_heavy(
    spec: &LoadSpec,
    root: &Path,
    control: &WeightsSource,
    load_pid: bool,
) -> Result<QwenControlHeavyOwned> {
    // Base + control applied dense first, THEN quantize together (the overlay-then-quantize ordering,
    // matching the Z-Image control port): quantizing before loading the control branch would not let
    // the dense control Linears compose. The text encoder + VAE stay dense (fork's quant scope).
    // The control contract deliberately publishes no transformer-window lifecycle: its separate
    // five-block branch is unbounded. Do not arm the base transformer's block stream here; without
    // a selectable window it can never be driven. Base and Edit retain their streamed load path.
    let mut transformer = loader::load_transformer(root)?;
    let mut controlnet = loader::load_controlnet(control)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        // F-076: reject a requested-vs-packed quant-tier mismatch on the base snapshot instead of
        // silently serving its tier; skip the no-op base quantize when the turnkey is already packed
        // at the requested bits (see `loader::needs_load_time_quant`). The control checkpoint has no
        // packed-marker convention (it ships dense), so it always takes the load-time quantize.
        if mlx_gen::quant::needs_load_time_quant(root, "transformer", bits, MODEL_ID)? {
            transformer.quantize(bits)?;
        }
        // F-076 parity for the control tier (sc-9517): a **published packed** control tier (built by
        // `convert::quantize_qwen_control_branch`) loads packed — its projections packed-detect via
        // `linear_from` — and `QwenFunControlBranch::quantize` no-ops on it (`AdaptableLinear::quantize`
        // is a no-op once quantized), so it renders as-is. But a requested-vs-packed BIT mismatch would
        // then silently serve the packed tier's bits — reject it, mirroring the base
        // `loader::needs_load_time_quant`. A dense checkpoint reports no packed bits → the request stands.
        match controlnet.packed_bits() {
            Some(packed) if packed != bits => {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: control checkpoint is a pre-quantized Q{packed} tier but Q{bits} was \
                     requested; quantize() is a no-op on packed weights so the request would silently \
                     serve Q{packed}. Point at a Q{bits} control tier (or a dense checkpoint)."
                )));
            }
            Some(_) => {}
            None => controlnet.quantize(bits)?,
        }
    }
    // Character-identity LoRA/LoKr targets the base transformer only (the control branch is never an
    // adapter target). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    transformer.capture_block_adapters();
    // Optional PiD overlay, loaded only when the spec carries it AND this generate uses it (`load_pid`,
    // F-177) — Resident passes `true`, Sequential passes `req.use_pid`.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    let vae = loader::load_vae(root)?;
    let alternate_decoder = mlx_gen_wan::load_selected_single_frame_decoder(spec, &descriptor())?;
    Ok(QwenControlHeavyOwned {
        transformer,
        controlnet,
        vae,
        pid,
        alternate_decoder,
    })
}

/// The 2512-Fun-Controlnet-Union VACE checkpoint is input-agnostic: pose, canny, and depth differ
/// only by the preprocessor-produced control image (no mode index — sc-8250). Spelled out as
/// `Only([Pose, Canny, Depth])` so a free-form `ControlKind::Other` is rejected rather than silently
/// coerced into the union path. A free function so the policy is unit-testable without a loaded model.
fn accepted_kinds() -> AcceptedControlKinds {
    AcceptedControlKinds::Only(vec![
        ControlKind::Pose,
        ControlKind::Canny,
        ControlKind::Depth,
    ])
}

/// The 2512-Fun Union admits the three structural control signals — pose/canny/depth share one
/// VACE control path, so all are accepted (sc-8250); only a free-form `ControlKind::Other` is
/// rejected. The control boilerplate (resolve/validate-present + the load helpers above) comes from
/// the shared trait (sc-8241).
impl ControlBranch for QwenImageControl {
    fn model_id(&self) -> &'static str {
        MODEL_ID
    }

    /// Fun-Union accepts pose/canny/depth; only the catch-all `Other` reaches this rejection, so the
    /// default Qwen "pose control only" wording is replaced with the union family's actual surface.
    fn unsupported_kind_message(&self, kind: &ControlKind) -> String {
        format!(
            "{MODEL_ID}: 2512-Fun-Controlnet-Union accepts pose/canny/depth control, got {kind:?}"
        )
    }

    fn missing_control_message(&self) -> String {
        format!("{MODEL_ID} requires a Control (pose/canny/depth) conditioning")
    }
}

impl Generator for QwenImageControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        mlx_gen::image_execution::without_capture(|| self.generate_observed(req, on_progress))
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            MODEL_ID,
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        let plan = compile_trajectory_for(req)?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        let plan = compile_trajectory_for(req)?;
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let mut report = mlx_gen::image_execution::without_capture(|| {
            self.generate_with_report_observed_with_run(req, Some(&run), on_progress)
        })?;
        report.low_res_start = plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
        Ok(report)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                mlx_gen::image_execution::policy_debug(
                    "QwenImageControl.precision",
                    &self.precision,
                );
                mlx_gen::image_execution::policy_debug("QwenImageControl.quant", &self.quant);
                mlx_gen::image_execution::policy_debug(
                    "QwenImageControl.loaded_sequential_residency",
                    &self.residency.is_sequential(),
                );
                mlx_gen::image_execution::applied_adapter_controls(
                    "QwenImageControl.applied_adapter_controls",
                    &self.adapters,
                );
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                let mut report =
                    self.generate_with_report_observed_with_run(req, Some(&run), &mut progress)?;
                report.low_res_start =
                    plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
                Ok(report)
            },
        )
    }
}

impl QwenImageControl {
    fn generate_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let plan = compile_trajectory_for(req)?;
        self.generate_planned(req, &plan, None, on_progress)?
            .output
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{MODEL_ID}: the render produced no output"))
            })
    }

    fn generate_with_report_observed_with_run(
        &self,
        req: &GenerationRequest,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.generate_impl_with_run(req, run, on_progress)
            .map_err(Into::into)
    }
}

impl QwenImageControl {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor, then the shared control-present check (sc-8241's
        // `ControlBranch::require_control_present`, which uses Qwen's "(pose skeleton)" message).
        validate_request(self.descriptor.id, &self.descriptor.capabilities, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (encode pos+neg → drop the Qwen2.5-VL encoder under `Sequential`
    /// → load the base DiT/control/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the
    /// shared [`Residency::run`] seam (sc-11125).
    fn generate_impl_with_run(
        &self,
        req: &GenerationRequest,
        run: Option<&gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<gen_core::GenerationReport> {
        self.validate(req)?;
        let resolved_pag = self.descriptor.capabilities.resolve_pag(MODEL_ID, req)?;
        mlx_gen::image_execution::policy_debug("qwen-image.pag", &resolved_pag);
        let crate::pipeline::RequestRungs {
            block_window,
            attention_budget,
            decode_tiling,
        } = crate::pipeline::resolve_request_rungs(req, &self.memory_strategy)?;

        // Shared step/sampler/guidance/seed resolution (F-117).
        let mut params = resolve_run_params(req, req.width, req.height)?;
        if let Some(run) = run {
            params.base_seed = run.plan().base_seed();
        }
        mlx_gen::image_execution::scalar("qwen-image.guidance", params.guidance);
        mlx_gen::image_execution::policy_debug("qwen-image.lightning", &params.is_lightning);
        let (control_image, control_scale) = self.resolve_control(req)?;
        let mut actual_steps = 0u32;

        // Phase A: prompt → embeds (epic 10834 Phase 1, sc-11006; sc-11125). Under `Sequential` the
        // shared seam loads the Qwen2.5-VL encoder, encodes pos+neg, materializes, then DROPS it +
        // `clear_cache()` so its ~15 GB frees before the DiT/control load below. Under `Resident` it
        // borrows the warm encoder. `neg` is `None` under Lightning (CFG-distilled → one forward/step).
        let output = self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |te: &QwenTextEncoder| {
                let encode_body = || -> Result<_> {
                    let pos = encode_prompt(&self.tokenizer, te, &req.prompt, MODEL_ID)?;
                    let neg = if params.guidance == 1.0 {
                        None
                    } else {
                        Some(encode_prompt(
                            &self.tokenizer,
                            te,
                            negative_or_fallback(req),
                            MODEL_ID,
                        )?)
                    };
                    let encoded = (pos, neg);
                    crate::pipeline::finish_conditioning(req, MODEL_ID, || {
                        match &encoded.1 {
                            Some(neg) => mlx_rs::transforms::eval([&encoded.0, neg])?,
                            None => mlx_rs::transforms::eval([&encoded.0])?,
                        }
                        Ok(())
                    })?;
                    Ok(encoded)
                };
                match run {
                    Some(run) => run.encode(encode_body),
                    None => encode_body(),
                }
            },
            // The shared boundary already forced pos (+neg) while the encoder was alive, for both
            // residency modes; Sequential has nothing left to materialize before dropping it.
            |_| Ok(()),
            // ── Establish the heavy render components (base DiT + control branch + VAE + PiD) and run
            // the denoise/decode body once against the `heavy` borrow — identical for both residencies.
            |heavy_owned, enc, on_progress| {
                let heavy = heavy_owned.as_ref();
                let (pos, neg) = enc;
                // VAE-encode + pack the pose skeleton to the 132-ch control context `[1, seq, 132]` (constant
                // across steps + the batch). The 2512-Fun control path VAE-encodes the control image and
                // concatenates a zero mask + zero inpaint latent before packing 2×2 (pose-only layout). This is
                // a deterministic VAE encode, independent of `pos`/`neg`, so under `Sequential` running it here
                // — after the text-encoder drop, with the VAE just loaded — is byte-identical to the Resident
                // order (same hoist argument as the T2I img2img `encode_init_latents`).
                let control_cond =
                    encode_fun_control_context(heavy.vae, control_image, req.width, req.height)?;

                // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): the partially-denoised x_k at the
                // achieved σ (truncated schedule) when use_pid + pid_capture_sigma; else the clean σ=0 path.
                // Control denoises from full noise (the pose is a constant conditioning), so `start_step = 0`.
                let (capture_sigma, keep) = flow_capture_for_request(req, &params.sigmas, 0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    params.base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let denoise_sigmas = &params.sigmas[..keep];
                actual_steps = denoise_sigmas.len().saturating_sub(1) as u32;
                let decoder = pid_decoder
                    .as_ref()
                    .map(|decoder| decoder as &dyn LatentDecoder)
                    .or_else(|| {
                        heavy
                            .alternate_decoder
                            .map(|decoder| decoder as &dyn LatentDecoder)
                    });
                if let Some(run) = run {
                    run.begin_rendering()?;
                }
                let images = crate::pipeline::decode_and_collect_with_run(
                    heavy.vae,
                    decoder,
                    decode_tiling.as_ref(),
                    req,
                    MODEL_ID,
                    req.count,
                    params.base_seed,
                    req.width,
                    req.height,
                    run,
                    on_progress,
                    |seed, progress| {
                        let noise = create_noise(seed, req.width, req.height)?;
                        let latents =
                            crate::pipeline::denoise_control_with_progress_windowed_pag_and_run(
                                heavy.transformer,
                                heavy.controlnet,
                                params.sampler_name.as_deref(),
                                denoise_sigmas,
                                seed,
                                noise,
                                &control_cond,
                                &pos,
                                neg.as_ref(),
                                params.guidance,
                                control_scale,
                                req.width,
                                req.height,
                                attention_budget,
                                block_window,
                                resolved_pag.as_ref(),
                                run,
                                &req.cancel,
                                progress,
                            )?;
                        crate::pipeline::calibration_fault(
                            req,
                            mlx_gen::gen_core::MemoryPhase::Denoise,
                            MODEL_ID,
                        )?;
                        Ok(latents)
                    },
                )?;
                if let Some(run) = run {
                    run.finish_rendering();
                }
                Ok(GenerationOutput::Images(images))
            },
        )?;
        Ok(gen_core::GenerationReport {
            pag: resolved_pag
                .as_ref()
                .map(|p| p.execution_report(actual_steps)),
            ..gen_core::GenerationReport::from_output(output)
        })
    }
}

pub fn compile_trajectory_for(req: &GenerationRequest) -> gen_core::Result<gen_core::RenderPlan> {
    let desc = descriptor();
    validate_request(MODEL_ID, &desc.capabilities, req)?;
    let has_control = req
        .conditioning
        .iter()
        .any(|c| matches!(c, mlx_gen::Conditioning::Control { .. }));
    if !has_control {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID} requires a Control (pose/canny/depth) conditioning"
        )));
    }
    for c in &req.conditioning {
        if let mlx_gen::Conditioning::Control { kind, .. } = c {
            if !matches!(
                kind,
                ControlKind::Pose | ControlKind::Canny | ControlKind::Depth
            ) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_ID}: 2512-Fun-Controlnet-Union accepts pose/canny/depth control, got {kind:?}"
                )));
            }
        }
    }
    let params = resolve_run_params(req, req.width, req.height)?;
    let (capture_sigma, keep) = flow_capture_for_request(req, &params.sigmas, 0);
    let executed_steps = keep.saturating_sub(1);
    let solver = gen_core::sampling::canonical_solver_or_euler(params.sampler_name.as_deref());
    let pass = gen_core::render::PackedImagePass {
        pixel_stride: SIZE_MULTIPLE,
        token_width: 64,
        sampler: solver.name(),
        guidance: params.guidance,
        sigmas: params.sigmas,
        executed_steps,
        capture_sigma,
    };
    gen_core::RenderPlan::packed_image_trajectory(&desc, req, pass)
}

pub fn compile_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    crate::model::validate_plan_load_spec(spec, MODEL_ID)?;
    compile_trajectory_for(req).map(|plan| Some(plan.bind_load(spec)))
}

pub const RENDER_PLAN_REGISTRATION: gen_core::RenderPlanRegistration =
    gen_core::RenderPlanRegistration {
        provider_id: MODEL_ID,
        compile: compile_render_plan,
    };

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
    use mlx_gen::Modality;

    #[test]
    fn descriptor_is_qwen_image_control() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_control");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.supports_lora);
    }

    #[test]
    fn accepts_pose_canny_depth_via_control_branch() {
        // The 2512-Fun Union is input-agnostic: pose, canny, and depth are all accepted (they differ
        // only by the preprocessor-produced control image, no mode index — sc-8250). A free-form
        // `Other` kind is rejected. This is exactly the `accepted_control_kinds()` policy the
        // `ControlBranch` impl returns.
        let accepted = accepted_kinds();
        assert!(accepted.accepts(&ControlKind::Pose));
        assert!(accepted.accepts(&ControlKind::Canny));
        assert!(accepted.accepts(&ControlKind::Depth));
        assert!(!accepted.accepts(&ControlKind::Other("scribble".into())));
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights, proving the overlay
        // is a hard requirement (it fails here before touching the missing base snapshot).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            err.contains("Qwen-Image-2512-Fun-Controlnet-Union"),
            "got: {err}"
        );
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    fn sealed_then_mutated_source(
        policy: OffloadPolicy,
    ) -> (
        tempfile::TempDir,
        LoadSpec,
        mlx_gen::gen_core::ValidatedEncoderSource,
    ) {
        let fixture = tempfile::tempdir().unwrap();
        let component = fixture.path().join("text_encoder");
        let contract = crate::active_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(&component, contract).unwrap();
        let control = fixture.path().join("control.safetensors");
        std::fs::write(&control, b"bounded control probe").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(fixture.path().to_path_buf()))
            .with_control(WeightsSource::File(control))
            .with_offload_policy(policy);
        let source = contract.source_for_load(&spec, fixture.path()).unwrap();
        std::fs::write(component.join("config.json"), b"{}\n").unwrap();
        (fixture, spec, source)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let _guard = crate::scoped_bounded_encoder_contract();
        let (_fixture, spec, source) = sealed_then_mutated_source(OffloadPolicy::Sequential);
        let residency = build_residency_with_source(&spec, source).unwrap();
        assert!(residency.is_sequential());
    }

    #[test]
    fn build_residency_resident_eagerly_invokes_the_text_loader() {
        let _guard = crate::scoped_bounded_encoder_contract();
        let (_fixture, spec, source) = sealed_then_mutated_source(OffloadPolicy::Resident);
        let error = build_residency_with_source(&spec, source)
            .err()
            .expect("Resident must eagerly invoke the production control text loader")
            .to_string();
        assert!(error.contains("changed after load"), "{error}");
    }
}
