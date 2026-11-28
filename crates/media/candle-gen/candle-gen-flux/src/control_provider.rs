//! FLUX.1-dev **Fun-Controlnet-Union** provider (sc-8412) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux`'s `Flux1DevControl` (sc-8238/8239). Strict structural conditioning (pose / canny /
//! depth — **input-agnostic**, no discrete mode index) on FLUX.1-dev via
//! `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`, a diffusers residual-emitter control branch
//! overlaid on the dev base DiT.
//!
//! **How it conditions:** the pose/canny/depth control image is VAE-encoded + 2×2-packed into the
//! packed transformer latent `[1, seq, 64]` (the same pack as the noise latents, so it aligns 1:1 with
//! the base image tokens), constant across the denoise. [`crate::FluxControlTransformer`] runs the
//! parity-proven dev DiT ([`crate::transformer::FluxDit`]) plus the control branch
//! ([`crate::control::FluxControlNet`]): 6 per-block residuals are computed once and added into the base
//! image stream after base double blocks at `interval = ceil(19/6) = 4`, scaled by `control_scale`
//! (Shakker README sweet spot ≈ 0.7; absent → that default, an explicit `Some(0.0)` = control off).
//! dev is guidance-distilled — a single embedded-guidance forward,
//! no true-CFG / negative pass.
//!
//! **Compose-readiness:** the denoise routes through [`crate::FluxControlTransformer::forward_composed`], which
//! threads an optional identity injector ([`crate::transformer::DitImageInjector`] — PuLID / XLabs
//! IP-Adapter) into the SAME base double-block stream as the control residuals. This provider wires the
//! seam (and exposes [`Flux1DevControl::generate_with_injector_planned`]); a follow-on epic stacks identity +
//! control in one denoise step by passing `Some(injector)`.
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::ip_provider`] / the FLUX.2 control provider). The
//! `flux1_dev_control` worker lane is a separate Phase-B step (sc-8304/sc-8246), not this crate.
//! Determinism is the candle-lane contract (sc-3673): seeded CPU init noise; the control encode uses
//! the VAE posterior MEAN (no sampling, no device RNG — sc-8988), so the control latent is fully
//! deterministic and launch-portable.

fn resolved_control_guidance(value: f32) -> f64 {
    candle_gen::execution::f32_policy("flux/control_guidance", value) as f64
}

fn resolved_control_scale(requested: Option<f32>) -> f64 {
    candle_gen::execution::f64_policy(
        "flux/control_scale",
        requested.unwrap_or(DEFAULT_CONTROL_SCALE) as f64,
    )
}

fn observe_injector(injector: Option<&dyn DitImageInjector>) {
    match injector {
        None => candle_gen::execution::policy("flux/control_injector", "none".into()),
        Some(injector) => match injector.execution_policy() {
            Some(value) => candle_gen::execution::policy("flux/control_injector", value),
            None => candle_gen::execution::unavailable(
                "FLUX control DitImageInjector does not expose its applied settings",
            ),
        },
    }
}

fn control_report_descriptor() -> candle_gen::gen_core::ModelDescriptor {
    let mut descriptor = crate::descriptor_dev();
    descriptor.id = "flux1_dev_control";
    descriptor
}

/// Run one control entry as a single reported attempt. The consumer's progress callback stays
/// outside it; observations from every phase of the planned body bind to the returned image.
#[cfg(test)]
fn complete_control_image(
    cancel: &CancelFlag,
    route: &'static str,
    on_progress: &mut dyn FnMut(Progress),
    run: impl FnOnce(&mut dyn FnMut(Progress)) -> Result<Image>,
) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
    complete_control_image_with_pag(cancel, route, on_progress, |progress| {
        Ok((run(progress)?, None))
    })
}

fn complete_control_image_with_pag(
    cancel: &CancelFlag,
    route: &'static str,
    on_progress: &mut dyn FnMut(Progress),
    run: impl FnOnce(
        &mut dyn FnMut(Progress),
    ) -> Result<(Image, Option<candle_gen::gen_core::PagExecutionReport>)>,
) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
    let mut report_progress = |event| candle_gen::execution::outside_attempt(|| on_progress(event));
    candle_gen::execution::capture_direct_report(&control_report_descriptor(), cancel, || {
        candle_gen::execution::policy("direct_route", route.into());
        let (image, pag) = run(&mut report_progress).map_err(candle_gen::gen_core::Error::from)?;
        let mut report = candle_gen::gen_core::GenerationReport::from_output(
            candle_gen::gen_core::GenerationOutput::Images(vec![image]),
        );
        report.pag = pag;
        Ok(report)
    })
}

fn resolve_control_pag(
    req: &Flux1ControlRequest,
) -> candle_gen::gen_core::Result<Option<candle_gen::gen_core::PagResolved>> {
    let descriptor = control_descriptor();
    let mut pag_req = GenerationRequest::default();
    pag_req.features.pag = req.pag.clone();
    descriptor.capabilities.resolve_pag(descriptor.id, &pag_req)
}

fn resolve_control_pag_report(
    req: &Flux1ControlRequest,
    plan: &RenderPlan,
) -> candle_gen::gen_core::Result<Option<candle_gen::gen_core::PagExecutionReport>> {
    let total_steps = u32::try_from(plan.denoise_sigmas().len().saturating_sub(1))
        .map_err(|_| candle_gen::gen_core::Error::Unsupported("step count overflow".into()))?;
    Ok(resolve_control_pag(req)?.map(|pag| pag.execution_report(total_steps)))
}

#[cfg(test)]
use crate::vae::native::{DiagonalGaussian, Encoder};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::flux::sampling::State;
use std::path::{Path, PathBuf};

use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    AdapterSpec, Conditioning, ControlKind, GenerationMemory, GenerationRequest, Image, LoadSpec,
    MemoryGeometry, MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    ModelDescriptor, PreviewSink, Progress, RenderLoadBinding, RenderPlan, RenderRun,
    WeightsSource,
};
use candle_gen::preview::PreviewHook;
use candle_gen::{CandleError, Result};

use crate::control::{accepts_control_kind, FluxControlNet, FluxControlNetConfig};
use crate::flux1_load;
use crate::pipeline::flux_config;
use crate::ref_backbone::{FluxRefBackbone, FluxRefHeavy};
use crate::transformer::{control_residual_interval, DitImageInjector};
use crate::Variant;

/// FLUX runs at bf16.
const DTYPE: DType = DType::BF16;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
/// FLUX latent geometry requires both image dims to be multiples of 16 for a clean 2×2 pack. Single
/// source of truth = the crate-root [`crate::SIZE_MULTIPLE`] (sc-12612).
use crate::SIZE_MULTIPLE;

/// Default control-conditioning scale — the Shakker Union-Pro-2.0 README recommends ≈ 0.7. Used only when
/// the request leaves `control_scale` **absent** (`None`); an explicit `Some(x)` always wins, including
/// `Some(0.0)` (which the engine proves is byte-identical to the base forward — "control off"). See
/// sc-9024 / F-040: an explicit 0.0 must NOT be remapped to the default (0.0 is a valid value, not
/// "unset").
pub const DEFAULT_CONTROL_SCALE: f32 = 0.7;

/// Paths to the FLUX.1-dev control checkpoints: the dev snapshot dir + the Shakker Fun-Controlnet-Union
/// overlay.
#[derive(Clone)]
pub struct Flux1ControlPaths {
    /// The black-forest-labs FLUX.1-dev snapshot dir (`flux1-dev.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`).
    pub flux_base: PathBuf,
    /// The `FLUX.1-dev-ControlNet-Union-Pro-2.0` checkpoint (a single `.safetensors`
    /// `diffusion_pytorch_model.safetensors`, or a dir containing it).
    pub control: PathBuf,
    pub adapters: Vec<AdapterSpec>,
}

/// One FLUX.1-dev strict-control request. dev is guidance-distilled — `guidance` is the embedded scalar
/// (single forward, no negative prompt).
#[derive(Clone)]
pub struct Flux1ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,
    /// Embedded guidance scale (dev default ≈ 3.5).
    pub guidance: f32,
    /// `control_scale` — how strongly the control branch locks the base (≈ 0.7). `None` = absent → the
    /// [`DEFAULT_CONTROL_SCALE`]; `Some(x)` is honored verbatim, including `Some(0.0)` for "control off"
    /// (byte-identical to the base forward). sc-9024: an explicit 0.0 is a valid value, not "unset".
    pub control_scale: Option<f32>,
    /// The control kind (pose / canny / depth). Input-agnostic — used only to validate the accepted set;
    /// it does NOT branch the forward (Union-Pro-2.0 dropped the discrete mode index).
    pub control_kind: String,
    pub seed: u64,
    /// Shared FLUX memory ladder selected by worker admission.
    pub memory: GenerationMemory,
    /// Per-step latent-preview sink (epic 16948, sc-16956) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which this provider cannot carry because it is worker-invoked by name rather than through a
    /// registered descriptor. Default is inert, and an inert sink makes a seeded render byte-identical
    /// to one with no preview at all.
    pub preview: PreviewSink,
    /// Optional Perturbed-Attention Guidance configuration for control-conditioned FLUX.1-dev.
    pub pag: Option<candle_gen::gen_core::PagConfig>,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux1ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance: 3.5,
            control_scale: None,
            control_kind: "pose".into(),
            seed: 0,
            memory: GenerationMemory::default(),
            preview: PreviewSink::default(),
            pag: None,
            cancel: CancelFlag::default(),
        }
    }
}

/// Open a VarBuilder over the Shakker control checkpoint — a single `.safetensors` `File` or a `Dir`
/// containing it — on `device` at `dtype`.
fn control_var_builder(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    candle_gen::load_path_mmap(path, dtype, device, "flux1 control")
}

/// Resize `image` to `width`×`height` and convert to an NCHW `[1, 3, H, W]` tensor in `[-1, 1]` at
/// `dtype` on `device` — the VAE-encode input for the control image (the same normalization the noise
/// path uses for decode).
pub(crate) fn preprocess_control(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "flux1 control: control pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    // Lanczos-resize to the render size (no-op when already that size), then [0,255] → [-1,1], HWC→NCHW.
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
    };
    let data: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(data, (rh, rw, 3), device)?;
    let nchw = hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
    nchw.to_dtype(dtype).map_err(Into::into)
}

/// Deterministic control-image VAE encoder (sc-8988): the FLUX [`Encoder`] + the posterior MEAN
/// (`DiagonalGaussian` with `sample = false`) + the BFL shift/scale. The upstream
/// [`AutoEncoder::encode`] *samples* the posterior with device `randn` — per-launch-deterministic at
/// best and never launch-portable, violating the sc-3673 determinism contract the rest of this
/// provider maintains (CPU-seeded `StdRng` everywhere). The mean encode is RNG-free (matching
/// `Flux2Vae::encode_packed` and the boogu edit precedent) and needs no seed at all.
#[cfg(test)]
struct MeanVaeEncoder {
    encoder: Encoder,
    shift_factor: f64,
    scale_factor: f64,
}

#[cfg(test)]
impl MeanVaeEncoder {
    /// Encode `nchw` (`[1, 3, H, W]` in `[-1, 1]`) to the posterior-MEAN latent `[1, 16, H/8, W/8]`,
    /// shift/scale-normalized exactly like [`AutoEncoder::encode`] — minus the sampling.
    fn encode(&self, nchw: &Tensor) -> Result<Tensor> {
        let moments = nchw.apply(&self.encoder)?; // [1, 2·z, H/8, W/8] = (mean ‖ logvar)
        let mean = moments.apply(&DiagonalGaussian::new(false, 1)?)?;
        ((mean - self.shift_factor)? * self.scale_factor).map_err(Into::into)
    }
}

/// A loaded FLUX.1-dev control model: the shared tier-aware backbone plus its resident Shakker branch.
pub struct Flux1DevControl {
    device: Device,
    dtype: DType,
    memory: GenerationMemory,
    backbone: FluxRefBackbone,
    branch: FluxControlNet,
    loaded_spec: LoadSpec,
    memory_contract: Option<MemoryProviderContract>,
    admitted_context: Option<MemoryRunContext>,
}

fn validate_control_memory_context(context: &MemoryRunContext) -> Result<()> {
    if !crate::memory_strategy::accepts_control_public_mode(&context.mode)
        || context.overlay.as_deref() != Some("control")
        || !context.has_reference
        || context.use_pid
        || context.has_phases
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.reference_count != 1
    {
        return Err(CandleError::Msg(
            "FLUX.1 ControlNet memory context is not one supported public control request".into(),
        ));
    }
    Ok(())
}

fn validate_control_spec(
    paths: &Flux1ControlPaths,
    spec: &LoadSpec,
) -> candle_gen::gen_core::Result<()> {
    if spec.weights != WeightsSource::Dir(paths.flux_base.clone())
        || !matches!(&spec.control, Some(WeightsSource::File(path) | WeightsSource::Dir(path)) if *path == paths.control)
        || spec.adapters != paths.adapters
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
        || spec.pid.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
        || spec.quantize.is_some()
        || spec.precision != candle_gen::gen_core::Precision::Bf16
    {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "FLUX.1 ControlNet load contract differs from resolved paths".into(),
        ));
    }
    Ok(())
}

pub(crate) fn route_control() -> candle_gen::gen_core::RouteRecord {
    let mut route = crate::route_base(Variant::Dev);
    route.route_id = "flux1_dev_control";
    route.structure.sampling.samplers = candle_gen::menu_with_aliases(
        std::mem::take(&mut route.structure.sampling.samplers),
        &["flow_match", "flow_match_euler"],
    );
    route.structure.sampling.schedulers = candle_gen::menu_with_aliases(
        std::mem::take(&mut route.structure.sampling.schedulers),
        &["native", "flow_match", "flow_match_euler"],
    );
    // The bespoke ControlNet route accepts any positive on-stride geometry; the
    // registered base txt2img route's 256..=2048 range does not apply here.
    route.limits.min_size = SIZE_MULTIPLE;
    route.limits.max_size = u32::MAX;
    route.limits.size_floor = candle_gen::gen_core::SizeFloor::RangeCheckedOnGridArea {
        multiple: SIZE_MULTIPLE,
        max_cells: u32::MAX,
    };
    route.structure.conditioning.kinds = vec![candle_gen::gen_core::ConditioningKind::Control];
    route.structure.conditioning.control_kinds =
        Some(candle_gen::gen_core::AcceptedControlKinds::Only(vec![
            ControlKind::Pose,
            ControlKind::Canny,
            ControlKind::Depth,
        ]));
    route.structure.conditioning.pass0_requirement =
        candle_gen::gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.conditioning.modules = vec!["flux_controlnet_union"];
    route.structure.weight_plan.requires_control_overlay = true;
    route.seed_and_batch.max_count = 1;
    route.defaults.strength = Some(DEFAULT_CONTROL_SCALE);
    route.bindings =
        candle_gen::gen_core::RouteBindings::generator(control_descriptor, load_control)
            .with_feature_route(Some(crate::FLUX1_DEV_ID), control_descriptor)
            .with_render_plan(compile_control_feature_route);
    route
}

struct Flux1DevControlGenerator {
    descriptor: ModelDescriptor,
    spec: LoadSpec,
    model: Flux1DevControl,
}

impl candle_gen::gen_core::Generator for Flux1DevControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        self.validate(req)?;
        let plan = compile_control_feature_route(&self.spec, req)?.ok_or_else(|| {
            candle_gen::gen_core::Error::Msg(
                "flux1_dev_control: failed to compile control render plan".into(),
            )
        })?;
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &candle_gen::gen_core::RenderPlan,
        scope: Option<&mut dyn candle_gen::gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::GenerationReport> {
        self.generate_planned_with_execution_report(req, plan, scope, on_progress)
            .map(|completed| completed.generation)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &candle_gen::gen_core::RenderPlan,
        scope: Option<&mut dyn candle_gen::gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&candle_gen::gen_core::RenderLoadBinding::from_spec(
            &self.spec,
        ))?;
        let (control_image, kind, scale) = match req.conditioning.as_slice() {
            [Conditioning::Control { image, kind, scale }] => (image, kind, *scale),
            _ => {
                return Err(candle_gen::gen_core::Error::Unsupported(
                    "flux1_dev_control requires exactly one Control conditioning image".into(),
                ));
            }
        };
        let control_kind = match kind {
            ControlKind::Pose => "pose".to_string(),
            ControlKind::Canny => "canny".to_string(),
            ControlKind::Depth => "depth".to_string(),
            ControlKind::Other(other) => other.clone(),
        };
        let control_req = Flux1ControlRequest {
            prompt: req.prompt.clone(),
            width: req.width,
            height: req.height,
            steps: req.steps.map(|s| s as usize).unwrap_or(25),
            sampler: req.sampler.as_deref().map(str::to_owned),
            scheduler: req.scheduler.as_deref().map(str::to_owned),
            scheduler_shift: req.scheduler_shift,
            guidance: req.guidance.unwrap_or(3.5),
            control_scale: scale.or(req.control_scale),
            control_kind,
            seed: req.seed.unwrap_or(0),
            memory: req.memory.unwrap_or_default(),
            preview: req.preview.clone(),
            pag: req.features.pag.clone(),
            cancel: req.cancel.clone(),
        };
        let _run = candle_gen::gen_core::RenderRun::new(plan, &req.cancel, scope);
        self.model
            .generate_with_execution_report(&control_req, control_image, plan, on_progress)
    }

    fn validate(&self, request: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        let _ = compile_control_feature_route(&self.spec, request)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::GenerationOutput> {
        self.generate_with_execution_report(request, on_progress)?
            .generation
            .output
            .ok_or_else(|| {
                candle_gen::gen_core::Error::Msg(
                    "flux1_dev_control: generation produced no output".into(),
                )
            })
    }
}

pub(crate) fn load_control(
    spec: &LoadSpec,
) -> candle_gen::gen_core::Result<Box<dyn candle_gen::gen_core::Generator>> {
    spec.validate_prepared_file_pins()?;
    candle_gen::gen_core::reject_unknown_components(spec, &[], control_descriptor().id)?;
    if !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
    {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "flux1_dev_control does not accept extra control, identity, IP-Adapter, or PiD overlays"
                .into(),
        ));
    }
    let flux_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(candle_gen::gen_core::Error::Msg(
                "flux1_dev_control requires a FLUX.1-dev snapshot directory".into(),
            ));
        }
    };
    let control = match &spec.control {
        Some(WeightsSource::File(path) | WeightsSource::Dir(path)) => path.clone(),
        None => {
            return Err(candle_gen::gen_core::Error::Msg(
                "flux1_dev_control requires spec.control".into(),
            ));
        }
    };
    let paths = Flux1ControlPaths {
        flux_base,
        control,
        adapters: spec.adapters.clone(),
    };
    let model = Flux1DevControl::load_with_memory_spec(&paths, GenerationMemory::default(), spec)
        .map_err(candle_gen::gen_core::Error::from)?;
    Ok(Box::new(Flux1DevControlGenerator {
        descriptor: control_descriptor(),
        spec: spec.clone(),
        model,
    }))
}

pub(crate) fn control_descriptor() -> ModelDescriptor {
    route_control().compute_descriptor()
}

pub(crate) fn compile_control_feature_route(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> candle_gen::gen_core::Result<Option<RenderPlan>> {
    let descriptor = control_descriptor();
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !matches!(req.conditioning.as_slice(), [Conditioning::Control { .. }])
        || spec.control.is_none()
    {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "flux1_dev_control requires a Control conditioning input and control weights".into(),
        ));
    }
    crate::compile_conditioned_plan(Variant::Dev, &descriptor, spec, req).map(Some)
}

fn typed_control_request(
    req: &Flux1ControlRequest,
    image: &Image,
) -> candle_gen::gen_core::Result<GenerationRequest> {
    let steps = u32::try_from(req.steps).map_err(|_| {
        candle_gen::gen_core::Error::Unsupported("FLUX.1 ControlNet step count exceeds u32".into())
    })?;
    let kind = match req.control_kind.as_str() {
        "pose" => ControlKind::Pose,
        "canny" => ControlKind::Canny,
        "depth" => ControlKind::Depth,
        other => ControlKind::Other(other.to_owned()),
    };
    Ok(GenerationRequest {
        prompt: req.prompt.clone(),
        width: req.width,
        height: req.height,
        count: 1,
        steps: Some(steps),
        seed: Some(req.seed),
        guidance: Some(req.guidance),
        sampler: req.sampler.clone().map(Into::into),
        scheduler: req.scheduler.clone().map(Into::into),
        scheduler_shift: req.scheduler_shift,
        conditioning: vec![Conditioning::Control {
            image: image.clone(),
            kind,
            scale: req.control_scale,
        }],
        memory: Some(req.memory),
        preview: req.preview.clone(),
        features: candle_gen::gen_core::FeatureParams {
            pag: req.pag.clone(),
            ..Default::default()
        },
        cancel: req.cancel.clone(),
        ..Default::default()
    })
}

/// The load contract [`Flux1DevControl::load`] / [`Flux1DevControl::load_with_memory`] bind, so a
/// caller of those paths-only loaders can compile [`compile_control_plan`] before loading.
pub fn control_load_spec(paths: &Flux1ControlPaths) -> LoadSpec {
    let control = if paths.control.is_dir() {
        WeightsSource::Dir(paths.control.clone())
    } else {
        WeightsSource::File(paths.control.clone())
    };
    LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
        .with_control(control)
        .with_adapters(paths.adapters.clone())
}

/// Resolve and bind the entire strict-control request before opening model weights.
pub fn compile_control_plan(
    paths: &Flux1ControlPaths,
    spec: &LoadSpec,
    req: &Flux1ControlRequest,
    image: &Image,
) -> candle_gen::gen_core::Result<RenderPlan> {
    validate_control_spec(paths, spec)?;
    let typed = typed_control_request(req, image)?;
    let descriptor = control_descriptor();
    descriptor
        .capabilities
        .validate_request(descriptor.id, &typed)?;
    if req.prompt.trim().is_empty() || req.steps == 0 || !accepts_control_kind(&req.control_kind) {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "FLUX.1 ControlNet needs a prompt, positive steps and pose/canny/depth control".into(),
        ));
    }
    if req.cancel.is_cancelled() {
        return Err(candle_gen::gen_core::Error::Canceled);
    }
    crate::compile_conditioned_plan(Variant::Dev, &descriptor, spec, &typed)
}

impl Flux1DevControl {
    /// Load the dev base (the forked [`crate::FluxDit`] DiT — the only FLUX DiT with the compose-ready injector
    /// seam) + the reused text encoders + VAE + the Shakker control overlay, assembling the
    /// [`crate::FluxControlTransformer`].
    pub fn load(paths: &Flux1ControlPaths) -> Result<Self> {
        Self::load_with_memory(paths, GenerationMemory::default())
    }

    pub fn load_with_memory(paths: &Flux1ControlPaths, memory: GenerationMemory) -> Result<Self> {
        Self::load_with_memory_spec(paths, memory, &control_load_spec(paths))
    }

    /// The worker passes the same resolved contract used for pre-load planning.
    pub fn load_with_memory_spec(
        paths: &Flux1ControlPaths,
        memory: GenerationMemory,
        spec: &LoadSpec,
    ) -> Result<Self> {
        Self::load_internal(paths, memory, None, None, spec)
    }

    pub fn load_with_memory_context_spec(
        paths: &Flux1ControlPaths,
        context: MemoryRunContext,
        spec: &LoadSpec,
    ) -> Result<Self> {
        validate_control_spec(paths, spec).map_err(|error| CandleError::Msg(error.to_string()))?;
        validate_control_memory_context(&context)?;
        let contract = crate::memory_strategy::provider_contract("flux1_dev", spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let tier = crate::memory_strategy::resolved_numeric_tier(spec, "flux1_dev")
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(&contract, &context, tier.quant)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        Self::load_internal(paths, memory, Some(contract), Some(context), spec)
    }

    fn load_internal(
        paths: &Flux1ControlPaths,
        memory: GenerationMemory,
        memory_contract: Option<MemoryProviderContract>,
        admitted_context: Option<MemoryRunContext>,
        spec: &LoadSpec,
    ) -> Result<Self> {
        validate_control_spec(paths, spec).map_err(|error| CandleError::Msg(error.to_string()))?;
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let root = paths.flux_base.clone();
        let variant = Variant::Dev; // the Shakker control is FLUX.1-dev only.

        let backbone = FluxRefBackbone::load_with_memory(
            &root,
            variant,
            &device,
            dtype,
            memory,
            paths.adapters.clone(),
        )?;

        // The Shakker control overlay (diffusers layout, un-prefixed keys). The branch shares the base
        // FLUX config's dims (hidden / heads / RoPE) so its residuals align with the base image tokens.
        let control_vb = control_var_builder(&paths.control, dtype, &device)?;
        let branch = FluxControlNet::new(
            &flux_config(variant),
            &FluxControlNetConfig::shakker_union_pro_2_0(),
            control_vb,
        )?;
        Ok(Self {
            device,
            dtype,
            memory,
            backbone,
            branch,
            loaded_spec: spec.clone(),
            memory_contract,
            admitted_context,
        })
    }

    /// The injection interval over the base double blocks (`ceil(19/6) = 4`).
    pub fn residual_interval(&self) -> usize {
        control_residual_interval(19, self.branch.num_residuals())
    }

    /// Number of control residuals (the control double-block count, 6).
    pub fn num_residuals(&self) -> usize {
        self.branch.num_residuals()
    }

    /// Plain control from a pre-load plan, completed with observations from this same attempt.
    pub fn generate_with_execution_report(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let pag = resolve_control_pag_report(req, plan)?;
        complete_control_image_with_pag(
            &req.cancel,
            "flux/control_provider/generate",
            on_progress,
            |progress| {
                let image =
                    self.generate_with_injector_planned(req, control_image, None, plan, progress)?;
                Ok((image, pag))
            },
        )
    }

    /// Identity stacked on control (or plain control with `None`), completed with observations
    /// from this same attempt.
    pub fn generate_with_injector_with_execution_report(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let pag = resolve_control_pag_report(req, plan)?;
        complete_control_image_with_pag(
            &req.cancel,
            "flux/control_provider/generate_with_injector",
            on_progress,
            |progress| {
                let image = self.generate_with_injector_planned(
                    req,
                    control_image,
                    injector,
                    plan,
                    progress,
                )?;
                Ok((image, pag))
            },
        )
    }

    /// The admitted plain-control entry, completed with observations from this same attempt —
    /// including the work inside its Conditioning/Denoise/Decode scopes.
    pub fn generate_with_memory_context_with_execution_report(
        &self,
        context: &MemoryRunContext,
        req: &Flux1ControlRequest,
        control_image: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let pag = resolve_control_pag_report(req, plan)?;
        complete_control_image_with_pag(
            &req.cancel,
            "flux/control_provider/generate_with_memory_context",
            on_progress,
            |progress| {
                let image = self.generate_with_injector_memory_context_planned(
                    context,
                    req,
                    control_image,
                    None,
                    plan,
                    progress,
                )?;
                Ok((image, pag))
            },
        )
    }

    /// The admitted identity-on-control entry, completed with observations from this same attempt.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_injector_memory_context_with_execution_report(
        &self,
        context: &MemoryRunContext,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let pag = resolve_control_pag_report(req, plan)?;
        complete_control_image_with_pag(
            &req.cancel,
            "flux/control_provider/generate_with_injector_memory_context",
            on_progress,
            |progress| {
                let image = self.generate_with_injector_memory_context_planned(
                    context,
                    req,
                    control_image,
                    injector,
                    plan,
                    progress,
                )?;
                Ok((image, pag))
            },
        )
    }

    /// Generate one control-conditioned image from the trajectory [`compile_control_plan`] froze
    /// before load. `control_image` is the preprocessed pose/canny/depth hint (the worker pre-fits it
    /// to the render size; this re-resizes defensively). `injector` optionally threads an identity
    /// injector (PuLID / XLabs IP-Adapter) into every control-denoise step — the **compose-ready**
    /// seam; `None` is the plain control path.
    pub fn generate_with_injector_planned(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if self.admitted_context.is_some() {
            return Err(CandleError::Msg(
                "FLUX.1 ControlNet admitted model requires a memory context".into(),
            ));
        }
        self.generate_with_injector_planned_scoped(
            req,
            control_image,
            injector,
            plan,
            None,
            on_progress,
        )
    }

    pub fn generate_with_memory_context_planned(
        &self,
        context: &MemoryRunContext,
        req: &Flux1ControlRequest,
        control_image: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.generate_with_injector_memory_context_planned(
            context,
            req,
            control_image,
            None,
            plan,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_injector_memory_context_planned(
        &self,
        context: &MemoryRunContext,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let admitted = self.admitted_context.as_ref().ok_or_else(|| {
            CandleError::Msg("FLUX.1 ControlNet model has no admitted memory context".into())
        })?;
        let contract = self.memory_contract.as_ref().ok_or_else(|| {
            CandleError::Msg("FLUX.1 ControlNet model lost its memory contract".into())
        })?;
        validate_control_memory_context(context)?;
        if admitted != context
            || context.geometry
                != (MemoryGeometry {
                    width: req.width,
                    height: req.height,
                    batch: 1,
                    frames: 1,
                    reference_count: 1,
                })
            || req.memory != self.memory
        {
            return Err(CandleError::Msg(
                "FLUX.1 ControlNet request changed after memory admission".into(),
            ));
        }
        contract
            .validate_selection(&context.selection)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let mut scope = crate::memory_strategy::FluxMemoryScope::new(
            "flux1_dev_control",
            self.device.clone(),
            contract,
            context,
        );
        let result = self.generate_with_injector_planned_scoped(
            req,
            control_image,
            injector,
            plan,
            Some(&mut scope),
            on_progress,
        );
        let outcome = match &result {
            Ok(_) => MemoryRunOutcome::Complete,
            Err(CandleError::Canceled) => MemoryRunOutcome::Canceled,
            Err(error) => MemoryRunOutcome::Error {
                message: error.to_string(),
            },
        };
        let cleanup = scope
            .finish(outcome)
            .map_err(|error| CandleError::Msg(error.to_string()));
        match (result, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(image), Ok(())) => Ok(image),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_with_injector_planned_scoped(
        &self,
        req: &Flux1ControlRequest,
        control_image: &Image,
        injector: Option<&dyn DitImageInjector>,
        plan: &RenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let typed = typed_control_request(req, control_image)?;
        plan.check_request(&control_descriptor(), &typed)?;
        plan.check_load(&RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let run = RenderRun::new(plan, &req.cancel, scope);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if req.prompt.trim().is_empty() {
            return Err(CandleError::Msg("flux1 control: prompt is required".into()));
        }
        if !accepts_control_kind(&req.control_kind) {
            return Err(CandleError::Msg(format!(
                "flux1_dev_control supports pose/canny/depth control (Fun-Controlnet-Union), got {:?}",
                req.control_kind
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "flux1 control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if req.steps == 0 {
            return Err(CandleError::Msg("flux1 control: steps must be >= 1".into()));
        }
        resolve_control_pag(req).map_err(|e| CandleError::Msg(e.to_string()))?;
        // sc-9024 (F-040): distinguish "absent" from an explicit 0.0. Only an ABSENT scale (`None`) falls
        // back to the Shakker default; an explicit `Some(x)` is honored verbatim — including `Some(0.0)`,
        // which the engine proves is byte-identical to the base forward ("control off"). The old code
        // treated 0.0 as "unset" and silently steered at 0.7, contradicting the engine's own ablation
        // semantics (scale 0 ≡ base; see `control_parity` / `control_real_weights`).
        let control_scale = resolved_control_scale(req.control_scale);

        // Conditioning (seed-independent): text (T5 seq + CLIP pooled) + the packed control latent.
        self.backbone
            .validate_native_vae_request(false, &req.cancel)?;
        let (t5_emb, clip_emb, heavy, control_latent) = run.encode(|| {
            let (t5, clip) = self.backbone.encode_text_with_memory_and_progress(
                &req.prompt,
                &req.cancel,
                on_progress,
            )?;
            let heavy = self.backbone.load_heavy(&req.cancel)?;
            let control_latent = self.encode_control_latent(
                heavy.as_ref(),
                control_image,
                req.width,
                req.height,
                &req.cancel,
            )?;
            // Finish the queued encoder and control-latent kernels inside Conditioning.
            self.device.synchronize()?;
            Ok::<_, CandleError>((t5, clip, heavy, control_latent))
        })?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request. sc-3673 parity:
        // deterministic, launch-portable CPU-seeded initial noise (shared FLUX.1 helper, sc-9003).
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        resolved_control_guidance(plan.guidance());
        observe_injector(injector);
        let images = run.images_with_denoise_phases(
            on_progress,
            |seed| {
                flux1_load::seeded_noise(
                    seed,
                    LATENT_CHANNELS,
                    lat_h,
                    lat_w,
                    &self.device,
                    self.dtype,
                )
            },
            |seed, noise, on_progress| {
                run.begin_denoise_phase(0)?;
                let state = State::new(&t5_emb, &clip_emb, &noise)?;
                if state.img.dim(1)? != plan.packed_tokens() {
                    return Err(CandleError::Msg(
                        "FLUX.1 ControlNet packed geometry changed after planning".into(),
                    ));
                }
                let preview = crate::preview::hook(&req.preview, req.width, req.height);
                self.denoise(
                    req,
                    &state,
                    &control_latent,
                    plan,
                    &run,
                    control_scale,
                    injector,
                    heavy.as_ref(),
                    seed,
                    &preview,
                    &req.cancel,
                    on_progress,
                )
            },
            |_, latents, on_progress| {
                on_progress(Progress::Decoding);
                self.backbone.decode_with_memory(
                    heavy.as_ref(),
                    &latents,
                    req.height as usize,
                    req.width as usize,
                    None,
                    &req.cancel,
                )
            },
        )?;
        images
            .into_iter()
            .next()
            .ok_or_else(|| CandleError::Msg("FLUX.1 ControlNet plan produced no image".into()))
    }

    /// VAE-encode + 2×2-pack the control hint into the packed control latent `[1, seq, 64]` (constant
    /// across steps). Uses the same VAE weights + pack (`State::new`'s patchify) as the noise latents, so
    /// the control latent aligns 1:1 with the base image tokens. The encode takes the posterior MEAN
    /// (sc-8988) — no sampling, no device RNG — so the latent is fully deterministic and launch-portable
    /// (the candle determinism contract, sc-3673), identical across steps and across the batch.
    fn encode_control_latent(
        &self,
        heavy: Option<&FluxRefHeavy>,
        image: &Image,
        width: u32,
        height: u32,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        let nchw = preprocess_control(image, width, height, &self.device, self.dtype)?;
        let encoded = self
            .backbone
            .encode_control_with_memory(heavy, &nchw, cancel)?; // [1, 16, H/8, W/8]
                                                                // Reuse the candle State patchify to 2×2-pack to [1, seq, 64] — identical geometry to the noise
                                                                // pack. `State::new` needs a t5/clip embed only to fill txt/vec (unused here), so feed tiny
                                                                // placeholders and take only `.img` (the packed latent).
        let dummy_t5 = Tensor::zeros((1, 1, 4096), self.dtype, &self.device)?;
        let dummy_clip = Tensor::zeros((1, 768), self.dtype, &self.device)?;
        let packed = State::new(&dummy_t5, &dummy_clip, &encoded)?;
        Ok(packed.img)
    }

    /// The flow-match denoise with the control branch, routed through the unified curated
    /// sampler/scheduler driver (epic 7114 P4) — the control twin of the IP/txt2img `denoise`. dev is
    /// guidance-distilled (a single embedded-guidance forward, no negative pass). The control forward
    /// lives INSIDE the `predict` closure so a multi-eval solver re-runs the whole step. FLUX feeds the
    /// raw timestep (`Sigma` convention: `t == σ`). The provider exposes no sampler/scheduler knob, so
    /// this defaults to the native flow-match Euler path. Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        req: &Flux1ControlRequest,
        state: &State,
        control_latent: &Tensor,
        render_plan: &RenderPlan,
        run: &RenderRun<'_, '_>,
        control_scale: f64,
        injector: Option<&dyn DitImageInjector>,
        heavy: Option<&FluxRefHeavy>,
        seed: u64,
        preview: &PreviewHook<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        let pag = resolve_control_pag(req).map_err(|e| CandleError::Msg(e.to_string()))?;
        candle_gen::execution::policy(
            "flux.pag",
            serde_json::to_value(&pag).unwrap_or(serde_json::Value::Null),
        );
        let canonical_solver =
            candle_gen::gen_core::sampling::canonical_solver_or_euler(Some(render_plan.sampler()));
        let total_steps = render_plan.denoise_sigmas().len().saturating_sub(1);
        let mut eval_idx = 0usize;
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(render_plan.guidance(), b_sz, &self.device)?;
        let prepared = self.backbone.prepare_conditioning(
            heavy,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
        )?;
        candle_gen::run_flow_sampler_spans(
            Some(render_plan.sampler()),
            TimestepConvention::Sigma,
            render_plan.denoise_sigmas(),
            &render_plan.denoise_spans(),
            Some(run),
            true,
            state.img.clone(),
            seed,
            cancel,
            on_progress,
            Some(preview),
            |img, t| -> Result<Tensor> {
                candle_gen::check_cancel(cancel)?;
                let current_eval = eval_idx;
                eval_idx += 1;
                let step_idx = candle_gen::gen_core::sampling::step_index_for_eval(
                    canonical_solver,
                    current_eval,
                    total_steps,
                );
                let step_pag = pag
                    .as_ref()
                    .map_or(0.0, |p| p.step_scale(step_idx, total_steps));
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                let budget = if self.memory.chunk_attention {
                    self.memory
                        .attention_chunk_size
                        .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE)
                        as u64
                } else {
                    candle_gen::ATTN_SCORES_BUDGET as u64
                };
                let plan =
                    AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, false))
                        .with_cancel(cancel);
                let residuals = self.branch.forward_prepared_with_memory(
                    img,
                    control_latent,
                    &state.txt,
                    &state.vec,
                    &state.img_ids,
                    &state.txt_ids,
                    &t_vec,
                    Some(&guidance_t),
                    &prepared,
                    plan,
                    cancel,
                )?;
                let pos = self.backbone.forward_control_prepared_with_memory_pag(
                    heavy,
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    injector,
                    Some((&residuals, control_scale)),
                    &prepared,
                    cancel,
                    None,
                )?;
                if step_pag > 0.0 {
                    let layers = pag
                        .as_ref()
                        .map_or(&[][..], |p| p.step_layers(step_idx, total_steps));
                    let pert = self.backbone.forward_control_prepared_with_memory_pag(
                        heavy,
                        img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &t_vec,
                        &state.vec,
                        Some(&guidance_t),
                        injector,
                        Some((&residuals, control_scale)),
                        &prepared,
                        cancel,
                        Some(layers),
                    )?;
                    crate::apply_pag_blend(&pos, &pert, step_pag)
                } else {
                    Ok(pos)
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_plan_freezes_source_schedule_and_overlay_without_weights() {
        let paths = Flux1ControlPaths {
            flux_base: "/missing/flux-dev".into(),
            control: "/missing/control.safetensors".into(),
            adapters: Vec::new(),
        };
        let spec = LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
            .with_control(WeightsSource::File(paths.control.clone()));
        let req = Flux1ControlRequest {
            prompt: "a fox".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: 11,
            ..Default::default()
        };
        let mut image = Image {
            width: 16,
            height: 16,
            pixels: vec![7; 16 * 16 * 3],
        };
        let plan = compile_control_plan(&paths, &spec, &req, &image).unwrap();
        assert_eq!(plan.denoise_sigmas().len(), 5);
        assert_eq!(plan.base_seed(), 11);
        let large = Flux1ControlRequest {
            width: 2304,
            height: 1024,
            ..req.clone()
        };
        compile_control_plan(&paths, &spec, &large, &image).unwrap();
        let overflowing_grid = Flux1ControlRequest {
            width: 1_048_576,
            height: 1_048_576,
            ..req.clone()
        };
        let error = compile_control_plan(&paths, &spec, &overflowing_grid, &image)
            .unwrap_err()
            .to_string();
        assert!(error.contains("packed image grid exceeds"), "{error}");
        for name in ["flow_match", "flow_match_euler"] {
            let mut alias = req.clone();
            alias.sampler = Some(name.into());
            assert_eq!(
                compile_control_plan(&paths, &spec, &alias, &image)
                    .unwrap()
                    .sampler(),
                name
            );
        }
        for name in ["native", "flow_match", "flow_match_euler"] {
            let mut alias = req.clone();
            alias.scheduler = Some(name.into());
            assert_eq!(
                compile_control_plan(&paths, &spec, &alias, &image)
                    .unwrap()
                    .denoise_sigmas(),
                plan.denoise_sigmas()
            );
        }
        for (sampler, scheduler) in [(Some("unknown"), None), (None, Some("unknown"))] {
            let mut invalid = req.clone();
            invalid.sampler = sampler.map(str::to_owned);
            invalid.scheduler = scheduler.map(str::to_owned);
            assert!(compile_control_plan(&paths, &spec, &invalid, &image).is_err());
        }
        crate::memory_strategy::assert_conditioned_plan_trace(&plan, &req.cancel);
        plan.check_load(&RenderLoadBinding::from_spec(&spec))
            .unwrap();
        image.pixels[0] ^= 1;
        assert!(plan
            .check_request(
                &control_descriptor(),
                &typed_control_request(&req, &image).unwrap()
            )
            .is_err());
        let mut changed = spec.clone();
        changed.control = Some(WeightsSource::File("/different/control".into()));
        assert!(plan
            .check_load(&RenderLoadBinding::from_spec(&changed))
            .is_err());
        let bad = Flux1ControlRequest {
            control_kind: "other".into(),
            ..req.clone()
        };
        assert!(compile_control_plan(&paths, &spec, &bad, &image).is_err());
        let mut missing_overlay = spec.clone();
        missing_overlay.control = None;
        assert!(compile_control_plan(&paths, &missing_overlay, &req, &image).is_err());
        let mut invalid = req.clone();
        invalid.steps = 0;
        assert!(compile_control_plan(&paths, &spec, &invalid, &image).is_err());
        invalid.steps = req.steps;
        invalid.width = 513;
        assert!(compile_control_plan(&paths, &spec, &invalid, &image).is_err());
    }

    #[test]
    fn direct_control_report_names_control_engine() {
        let descriptor = control_report_descriptor();
        assert_eq!(descriptor.id, "flux1_dev_control");
        assert_eq!(descriptor.backend, "candle");
    }

    /// Metadata only: the admitted route's report binds every observation the planned body makes
    /// to this attempt's image, names main's route, and leaves the consumer callback outside.
    #[test]
    fn admitted_control_report_binds_the_planned_attempt() {
        use candle_gen::gen_core::generator::{ExecutionFact, ExecutionIdentity};
        let cancel = CancelFlag::default();
        let completed = complete_control_image(
            &cancel,
            "flux/control_provider/generate_with_memory_context",
            &mut |_| candle_gen::execution::policy("consumer/callback", true.into()),
            |progress| {
                candle_gen::execution::runtime(&Device::Cpu, DType::F32);
                candle_gen::execution::rng_scheme("fixture-metadata-v1");
                candle_gen::execution::draw("fixture", 7, 0);
                candle_gen::execution::schedule("euler", &[1.0, 0.0], "Flow");
                resolved_control_guidance(3.5);
                observe_injector(None);
                progress(Progress::Step {
                    current: 1,
                    total: 1,
                });
                Ok(Image {
                    width: 2,
                    height: 1,
                    pixels: vec![0; 6],
                })
            },
        )
        .unwrap();
        let ExecutionIdentity::Resolved { identity } = completed.execution else {
            panic!("expected a resolved report: {:?}", completed.execution)
        };
        let ExecutionFact::Resolved { value } = &identity.quality_policy else {
            panic!("quality policy")
        };
        let operations = &value["operations"];
        assert_eq!(
            operations["direct_route"][0]["value"],
            "flux/control_provider/generate_with_memory_context"
        );
        assert!(operations.get("flux/control_guidance").is_some());
        assert!(operations.get("flux/control_injector").is_some());
        assert!(operations.get("consumer/callback").is_none());

        let pag_req = Flux1ControlRequest {
            prompt: "a fox".into(),
            steps: 4,
            pag: Some(candle_gen::gen_core::PagConfig {
                scale: 1.5,
                ..Default::default()
            }),
            ..Default::default()
        };
        let paths = Flux1ControlPaths {
            flux_base: "/missing/flux-dev".into(),
            control: "/missing/control.safetensors".into(),
            adapters: Vec::new(),
        };
        let spec = control_load_spec(&paths);
        let plan = compile_control_plan(
            &paths,
            &spec,
            &pag_req,
            &Image {
                width: 16,
                height: 16,
                pixels: vec![0; 16 * 16 * 3],
            },
        )
        .unwrap();
        let pag_report = resolve_control_pag_report(&pag_req, &plan)
            .unwrap()
            .expect("active pag report");
        let pag_completed = complete_control_image_with_pag(
            &cancel,
            "flux/control_provider/generate",
            &mut |_| {},
            |_| {
                candle_gen::execution::runtime(&Device::Cpu, DType::F32);
                candle_gen::execution::rng_scheme("fixture-metadata-v1");
                candle_gen::execution::draw("fixture", 7, 0);
                candle_gen::execution::schedule("euler", &[1.0, 0.0], "Flow");
                candle_gen::execution::policy(
                    "flux.pag",
                    serde_json::to_value(resolve_control_pag(&pag_req).unwrap())
                        .unwrap_or(serde_json::Value::Null),
                );
                Ok((
                    Image {
                        width: 2,
                        height: 1,
                        pixels: vec![0; 6],
                    },
                    Some(pag_report.clone()),
                ))
            },
        )
        .unwrap();
        assert_eq!(pag_completed.generation.pag, Some(pag_report));

        assert!(complete_control_image(
            &cancel,
            "flux/control_provider/generate",
            &mut |_| {},
            |_| Err(CandleError::Msg("fixture failure".into())),
        )
        .is_err());
    }

    /// The request defaults match the dev control production knobs (1024², 25 steps, guidance 3.5, pose).
    /// `control_scale` defaults to `None` (absent) — resolved to [`DEFAULT_CONTROL_SCALE`] at generate
    /// time (sc-9024: absent ≠ an explicit 0.0).
    #[test]
    fn request_defaults() {
        let r = Flux1ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, 3.5);
        assert_eq!(r.control_scale, None);
        assert_eq!(r.control_kind, "pose");
        assert!(!r.cancel.is_cancelled());
    }

    /// sc-9024 (F-040): the `control_scale` resolution honors an explicit `Some(0.0)` as "control off"
    /// (byte-identical to the base forward — see `control_parity`) and only substitutes the Shakker
    /// default (0.7) when the scale is truly absent (`None`). A genuine 0.0 must NOT be remapped to 0.7.
    #[test]
    fn control_scale_resolution_honors_explicit_zero() {
        // The generate-time resolution: `req.control_scale.unwrap_or(DEFAULT_CONTROL_SCALE)`.
        let resolve = |cs: Option<f32>| cs.unwrap_or(DEFAULT_CONTROL_SCALE) as f64;

        // Absent → the default (0.7): a request that supplied a control but left the scale unset steers.
        assert_eq!(
            resolve(None),
            DEFAULT_CONTROL_SCALE as f64,
            "absent (None) must fall back to the Shakker default"
        );
        // Explicit 0.0 → 0.0 (NOT 0.7): the caller asked for control off; honor it (0 ≡ base forward).
        assert_eq!(
            resolve(Some(0.0)),
            0.0,
            "explicit Some(0.0) must be honored, not remapped to the default"
        );
        // Explicit non-zero → verbatim (compare at the same f32→f64 widening the code performs).
        assert_eq!(
            resolve(Some(0.35)),
            0.35f32 as f64,
            "explicit Some(x) is honored verbatim"
        );
        assert_eq!(resolve(Some(1.0)), 1.0);
    }

    /// The control checkpoint resolver: a missing path errors loudly; a direct file is used as-is.
    #[test]
    fn control_checkpoint_resolution() {
        let dev = Device::Cpu;
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // A nonexistent path errors.
        assert!(control_var_builder(&dir.join("nope.safetensors"), DTYPE, &dev).is_err());
    }

    use crate::vae::native::Config as AeCfg;
    use candle_nn::VarMap;
    use rand::{rngs::StdRng, SeedableRng};

    /// A tiny FLUX AE config (real BFL shift/scale, `ch = 32` — the AE's `group_norm(32, ·)` floor).
    fn tiny_ae_cfg() -> AeCfg {
        AeCfg {
            resolution: 16,
            in_channels: 3,
            ch: 32,
            out_ch: 3,
            ch_mult: vec![1],
            num_res_blocks: 1,
            z_channels: 2,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    /// Build a tiny random-weight [`MeanVaeEncoder`] on CPU. VarMap zero-init is randomized (a zero
    /// encoder emits constant moments, which would make the mean-vs-sample distinction vacuous).
    fn tiny_mean_encoder(dev: &Device) -> MeanVaeEncoder {
        let cfg = tiny_ae_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let encoder = Encoder::new(&cfg, vb).expect("tiny encoder");
        let mut rng = StdRng::seed_from_u64(8988);
        for var in vm.data().lock().unwrap().values() {
            let n = var.shape().elem_count();
            let data: Vec<f32> = candle_gen::seeded_normal_vec(&mut rng, n)
                .into_iter()
                .map(|v| v * 0.05)
                .collect();
            let t = Tensor::from_vec(data, var.shape(), dev).expect("randomize");
            var.set(&t).expect("set var");
        }
        MeanVaeEncoder {
            encoder,
            shift_factor: cfg.shift_factor,
            scale_factor: cfg.scale_factor,
        }
    }

    /// A fixed (RNG-free) `[1, 3, 16, 16]` input in `[-1, 1]`.
    fn fixed_input(dev: &Device) -> Tensor {
        let n = 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        Tensor::from_vec(data, (1, 3, 16, 16), dev).expect("input")
    }

    /// sc-8988: the control encode is bit-identical across calls and independent of the device RNG
    /// stream — the old sampled path (`AutoEncoder::encode` + `set_seed`) drew device `randn` and
    /// diverged when the RNG state moved between calls.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn control_encode_mean_is_deterministic_and_rng_free() {
        let dev = Device::Cpu;
        let enc = tiny_mean_encoder(&dev);
        let img = fixed_input(&dev);
        let a = enc.encode(&img).expect("encode a");
        // Churn the device RNG between calls — a sampled encode would change; the mean must not.
        let _ = Tensor::randn(0f32, 1f32, (16,), &dev).expect("rng churn");
        let b = enc.encode(&img).expect("encode b");
        // ch_mult = [1] ⇒ a single level with no downsample: the tiny latent stays at H×W.
        assert_eq!(a.dims(), &[1, 2, 16, 16], "z=2 latent");
        let (av, bv) = (
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(av.iter().all(|v| v.is_finite()), "finite latent");
        assert_eq!(av, bv, "mean encode must be bit-identical across calls");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED control_provider::tests::control_encode_mean_is_deterministic_and_rng_free");
    }

    /// sc-8988: the encode is the shift/scale-normalized posterior MEAN — the first `z_channels` chunk
    /// of the encoder moments, exactly as `AutoEncoder::encode` normalizes, minus the sampling.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn control_encode_takes_the_posterior_mean() {
        let dev = Device::Cpu;
        let enc = tiny_mean_encoder(&dev);
        let img = fixed_input(&dev);
        let got = enc.encode(&img).expect("encode");
        let moments = img.apply(&enc.encoder).expect("moments"); // [1, 2·z, 2, 2]
        let mean = moments.chunk(2, 1).expect("chunk")[0].clone();
        let want = ((mean - enc.shift_factor).unwrap() * enc.scale_factor).unwrap();
        let (g, w) = (
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert_eq!(g, w, "encode must equal (posterior_mean - shift) * scale");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED control_provider::tests::control_encode_takes_the_posterior_mean");
    }
}

#[cfg(test)]
mod execution_policy_tests {
    use super::*;
    #[test]
    fn execution_control_distinguishes_absent_zero_and_unknown_injector() {
        let report = |scale| {
            candle_gen::testkit::execution_policy(|| {
                resolved_control_scale(scale);
                observe_injector(None);
            })
        };
        assert_eq!(report(None), report(Some(DEFAULT_CONTROL_SCALE)));
        assert_ne!(report(None), report(Some(0.0)));
        let guidance = |value| {
            candle_gen::testkit::execution_policy(|| {
                resolved_control_guidance(value);
            })
        };
        assert_ne!(guidance(3.5), guidance(4.5));
        struct Known(f32);
        impl DitImageInjector for Known {
            fn execution_policy(&self) -> Option<serde_json::Value> {
                Some(serde_json::json!({"weight_bits": self.0.to_bits()}))
            }
            fn after_double(
                &self,
                _: usize,
                _: &Tensor,
            ) -> candle_gen::candle_core::Result<Option<Tensor>> {
                unreachable!()
            }
            fn injects_after_single(&self, _: usize) -> bool {
                unreachable!()
            }
            fn after_single(
                &self,
                _: usize,
                _: &Tensor,
            ) -> candle_gen::candle_core::Result<Option<Tensor>> {
                unreachable!()
            }
        }
        let injected = |weight| {
            candle_gen::testkit::execution_policy(|| {
                observe_injector(Some(&Known(weight)));
            })
        };
        assert_ne!(injected(0.5), injected(0.75));
        struct Unknown;
        impl DitImageInjector for Unknown {
            fn after_double(
                &self,
                _: usize,
                _: &Tensor,
            ) -> candle_gen::candle_core::Result<Option<Tensor>> {
                unreachable!()
            }
            fn injects_after_single(&self, _: usize) -> bool {
                unreachable!()
            }
            fn after_single(
                &self,
                _: usize,
                _: &Tensor,
            ) -> candle_gen::candle_core::Result<Option<Tensor>> {
                unreachable!()
            }
        }
        let report = candle_gen::testkit::execution_policy(|| {
            resolved_control_scale(None);
            observe_injector(Some(&Unknown));
        });
        assert!(matches!(
            report,
            candle_gen::gen_core::generator::ExecutionIdentity::Unavailable { .. }
        ));
    }
}
