//! `Boogu` — the [`mlx_gen::Generator`] implementation for Boogu-Image-0.1, plus its
//! [`descriptor`]/[`load`] entry points and registration constants for the three variants:
//! `"boogu_image"` (Base, true-CFG T2I),
//! `"boogu_image_turbo"` (DMD few-step, CFG-free), and `"boogu_image_edit"` (instruction
//! image-edit).
//!
//! All three variants share one architecture/loader (the [`crate::pipeline`] `BooguEncoders` +
//! `BooguHeavy` bundles, staged onto the shared [`mlx_gen::Residency`] seam); they differ only in which
//! snapshot they load (Base / Turbo / Edit checkpoint) and which sampler
//! [`Boogu::generate`] runs. `spec.quantize` (Q4/Q8) quantizes the dense base in place after the
//! load — a **no-op** when the snapshot is already a packed Q8/Q4 turnkey (the turnkey's default),
//! so pointing at a pre-quantized snapshot skips the dense transient. A precision override and LoRA
//! adapters are rejected rather than silently ignored.

use std::path::Path;

use mlx_gen::gen_core::{
    AdapterSurface, ConditioningSurface, FamilyDriverSpec, FamilyModelSpec, LatentCodecDeclaration,
    ModelBinding, OutputSurface, Pass0ConditioningRequirement, PreviewSupport, ProductLimits,
    ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface, RouteBindings, RouteDefaults,
    RouteRecord, RouteStructure, SamplingSurface, SeedAndBatchSemantics, SeedScheme, SizeFloor,
    StepSupport, TextEncoderDeclaration, TextEncoderSurface, UpsamplerSurface,
    WeightPlanDeclaration, WeightSourceKind, FLUX1_LATENT_SPACE,
};
use mlx_gen::img2img::init_time_step;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Conditioning, ConditioningKind,
    Error, GenerationOutput, GenerationRequest, Generator, Image, LatentDecoder, LoadSpec,
    Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Residency, Result,
    WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};

use crate::pipeline::{
    resolve_reference, BooguBaseCond, BooguEncoders, BooguHeavy, EditOptions, GenerateOptions,
    TurboOptions,
};
use crate::tokenizer::BooguTokenizer;

/// Registry id for the Base text-to-image variant (true-CFG). Matches the SceneWorks worker's
/// `payload.model`.
pub const BOOGU_IMAGE_ID: &str = "boogu_image";
/// Registry id for the Turbo variant (DMD few-step, CFG-free).
pub const BOOGU_IMAGE_TURBO_ID: &str = "boogu_image_turbo";
/// Registry id for the instruction image-edit variant.
pub const BOOGU_IMAGE_EDIT_ID: &str = "boogu_image_edit";

/// PiD backbone (latent-space) tag for Boogu (epic 7840, sc-7846). All three Boogu variants use the
/// FLUX.1 16-ch VAE (the shared `mlx_gen_z_image::vae::Vae`), so they reuse the `flux` PiD student.
/// Used only at load time to build the [`PidEngine`].
pub const PID_BACKBONE: &str = "flux";

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Resolution bounds (W/H), both multiples of 16. The catalog/worker gate the actual UI options
/// tighter; this is the engine validation ceiling.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Patch(2)·ae_scale(8) = 16 — `patchify` requires dims divisible by this. Exposed as the
/// pinned-engine stride SceneWorks ties each advertised Boogu image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`. `validate_request` enforces exactly this value, so the const
/// cannot drift from the check.
pub const RES_MULTIPLE: u32 = 16;

/// Max reference images the Edit checkpoint supports — the DiT's `image_index_embedding` carries 5
/// per-image index slots (`[5, hidden]`, OmniGen2 lineage), so `N ∈ [1, 5]` references can be packed.
const MAX_EDIT_REFS: usize = 5;

/// Base/Edit default steps + guidance (the reference `__call__`: 50-step true-CFG, guidance 4.0).
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Turbo default steps (DMD student few-step) + the lowest sigma in the DMD schedule.
const DEFAULT_TURBO_STEPS: u32 = 4;
const DEFAULT_TURBO_SIGMA: f32 = 0.001;

/// Primary typed route declaration for Boogu Base (`REG-1`, `AP5`).
pub fn route_base() -> RouteRecord {
    RouteRecord {
        route_id: BOOGU_IMAGE_ID,
        family: "boogu",
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("qwen3_vl_8b", None)],
                ..Default::default()
            },
            conditioning: ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                control_kinds: None,
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: vec![],
                ..Default::default()
            },
            sampling: SamplingSurface {
                samplers: curated_sampler_names(),
                schedulers: curated_scheduler_names(),
                guidance_methods: vec![],
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: ModelBinding::family_model(
                FamilyModelSpec::new("boogu_dit").with_low_res_start(),
            ),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&FLUX1_LATENT_SPACE),
                pixel_stride: Some(RES_MULTIPLE),
                token_width: Some(64),
                supports_diffusion_decoder: false,
            },
            upsamplers: UpsamplerSurface {
                components: vec!["pid_decoder"],
                max_temporal_upsample_rounds: 0,
            },
            adapters: AdapterSurface::unsupported(),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(ResidencyPhaseKind::PromptEncode, ["qwen3_vl_8b"], true),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Denoise, ["boogu_dit"], false),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Decode, ["flux1_ae"], false),
                ],
                supports_sequential_offload: true,
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
                ..Default::default()
            },
            output: OutputSurface {
                preview: PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: RES_MIN,
            max_size: RES_MAX,
            size_floor: SizeFloor::RangeChecked,
            supported_steps: StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: MAX_COUNT,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(DEFAULT_STEPS),
            guidance: Some(DEFAULT_GUIDANCE),
            sampler: Some("euler"),
            schedule: None,
            shift: Some(1.15),
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(descriptor, |spec| load(spec).map_err(Into::into))
            .with_footprint(component_footprint)
            .with_render_plan(compile_render_plan),
    }
}

/// Boogu Base's identity + capabilities, computed from [`route_base`].
pub fn descriptor() -> ModelDescriptor {
    route_base().compute_descriptor()
}

/// The curated samplers the Turbo DMD student stays coherent under (real-weight survey, sc-7491). The
/// student was distilled against a **stochastic** (re-noised) trajectory — predict the clean estimate,
/// then renoise to the next level with fresh noise — so the curated *stochastic* solvers match its
/// training regime and render at native quality. `lcm` is the closest match (it IS the consistency
/// predict→renoise loop, like ComfyUI's `lcm`/`sgm_uniform` combo), once `lcm` re-noises through the
/// FLOW `noise_scaling` convex blend rather than the VE additive form (the gen-core sc-7491 fix). The
/// deterministic ODE solvers (`euler`/`ddim`/`heun`/`dpmpp_2m`/`uni_pc`) feed the few-step student
/// out-of-regime latents (background artifacts), so they stay off the menu; the native DMD loop
/// (`req.sampler == None`) stays the byte-exact default.
const TURBO_SAMPLERS: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

/// Primary typed route declaration for Boogu Turbo (`REG-1`, `AP5`).
pub fn route_turbo() -> RouteRecord {
    let mut route = route_base();
    route.route_id = BOOGU_IMAGE_TURBO_ID;
    route.structure.sampling.samplers = TURBO_SAMPLERS.to_vec();
    route.structure.sampling.schedulers = curated_scheduler_names();
    route.structure.sampling.supports_guidance = true;
    route.structure.denoiser = ModelBinding::family_driver(FamilyDriverSpec::new(
        crate::pipeline::boogu_turbo_driver_support(),
    ));
    route.defaults = RouteDefaults {
        steps: Some(DEFAULT_TURBO_STEPS),
        guidance: Some(1.0),
        sampler: None,
        schedule: None,
        shift: None,
        strength: None,
        dimensions: Some((1024, 1024)),
        ..Default::default()
    };
    route.bindings =
        RouteBindings::generator(descriptor_turbo, |spec| load_turbo(spec).map_err(Into::into))
            .with_footprint(component_footprint);
    route
}

/// Boogu **Turbo** identity + capabilities, computed from [`route_turbo`].
pub fn descriptor_turbo() -> ModelDescriptor {
    route_turbo().compute_descriptor()
}

/// Primary typed route declaration for Boogu Edit (`REG-1`, `AP5`).
pub fn route_edit() -> RouteRecord {
    let mut route = route_base();
    route.route_id = BOOGU_IMAGE_EDIT_ID;
    route.structure.conditioning.kinds = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    route.structure.conditioning.pass0_requirement = Pass0ConditioningRequirement::RequiresEdit;
    route.structure.conditioning.modules = vec!["qwen3_vl_vision_tower"];
    route.structure.denoiser = ModelBinding::family_model(FamilyModelSpec::new("boogu_dit"));
    route.bindings =
        RouteBindings::generator(descriptor_edit, |spec| load_edit(spec).map_err(Into::into))
            .with_footprint(component_footprint)
            .with_render_plan(compile_edit_render_plan);
    route
}

/// Boogu **Edit** identity + capabilities, computed from [`route_edit`].
pub fn descriptor_edit() -> ModelDescriptor {
    route_edit().compute_descriptor()
}

/// A loaded Boogu generator: the always-warm tokenizer + PiD overlay, the cached descriptor (which
/// selects the sampler path), and the component-residency strategy.
pub struct Boogu {
    descriptor: ModelDescriptor,
    loaded_spec: LoadSpec,
    /// Effective Q4/Q8 tier for the text tower and DiT; `None` means source metadata could not
    /// prove one of the two tiers, so a completed render must refuse exact execution identity.
    quantization_bits: Option<(Option<i32>, Option<i32>)>,
    /// The tiny BPE-ish tokenizer stays always-warm on the generator (cheap); the heavy mllm↔DiT
    /// dispatch is the shared [`Residency`] seam.
    tokenizer: BooguTokenizer,
    /// Component-residency strategy (epic 10834, sc-10840; the shared seam sc-11125), selected from
    /// [`LoadSpec::offload_policy`] at [`load_with`]. `Resident` (default) holds the Qwen3-VL `mllm/`
    /// encoder (+ lazy vision tower) and the DiT + VAE warm for the whole job and across jobs;
    /// `Sequential` holds only the per-phase loader closures and re-loads each per generation in phase
    /// order (encode → **drop the mllm** → denoise/decode), bounding peak unified memory to
    /// `max(mllm, DiT+VAE)` instead of the sum (the ~17.5 GB Qwen3-VL encoder is comparable to the
    /// ~20.6 GB DiT). The seam owns the eval/drop/clear discipline, the stage-boundary cancel checks,
    /// and the error-safe cache flush.
    residency: Residency<BooguEncoders, BooguHeavy>,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7846). `Some` only when the
    /// `LoadSpec` carried `pid`; selected per-generation by `req.use_pid` and threaded into the
    /// render phase's decode tail. Shared across all three Boogu variants (FLUX.1 VAE latent space).
    /// The overlay is small (a student decoder + caption encoder) and, unlike the mllm/DiT, stays
    /// resident on the generator (loaded once at [`load_with`], like the tokenizer) — the peak-bounding
    /// win is the mllm↔DiT drop, so the overlay is not re-loaded per generate.
    pid: Option<PidEngine>,
}

/// Load a Boogu generator from a [`LoadSpec`] under the given `descriptor`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a Boogu snapshot (`mllm/ transformer/ vae/`). The loader
/// auto-detects a packed Q8/Q4 turnkey (the shipped default) vs a dense bf16 snapshot; `spec.quantize`
/// then quantizes the dense base in place (a no-op on an already-packed snapshot). A precision
/// override and LoRA/LoKr adapters are rejected rather than silently ignored.
///
/// Component residency (epic 10834, sc-10840; the shared [`Residency::from_policy`] seam sc-11126):
/// `Resident` (default) builds the mllm encoder + heavy DiT/VAE now via [`build_residency`] and holds
/// them warm; `Sequential` keeps only the per-phase loader closures and re-loads per generate in phase
/// order (encode → drop the mllm → denoise/decode) to bound peak memory to `max(mllm, DiT+VAE)`. Both
/// use the same per-phase loaders, so the components are byte-identical.
fn load_with(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Box<dyn Generator>> {
    let id = descriptor.id;
    let quant_bits = spec.quantize.map(Quant::affine_bits).transpose()?;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    // Resolve the snapshot dir up front — a fail-fast for BOTH residencies (Sequential defers the heavy
    // component build to each generate, but a single-file source is still wrong here).
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory (mllm/ transformer/ vae/), not a single \
                 .safetensors file"
            )))
        }
    };
    let quantization_bits = (
        mlx_gen::quant::packed_bits_from_headers(&root.join("mllm")),
        mlx_gen::quant::packed_bits_from_headers(&root.join("transformer")),
    );
    let quantization_bits = match quantization_bits {
        (Ok(text), Ok(dit)) => effective_component_quant_bits(text, quant_bits)
            .zip(effective_component_quant_bits(dit, quant_bits)),
        _ => None,
    };
    // F-181: a `Sequential` + `quantize` load re-quantizes over the dense snapshot on EVERY generate
    // (repeated compute + the dense per-phase transient). Boogu's per-phase `quantize` is a no-op on an
    // already-packed turnkey (the shipped default). The header-only tier read above records the
    // effective policy, but dense sequential loads still pay the repeated quantization cost.
    if let Some(bits) = quant_bits {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(id, bits);
        }
    }
    let tokenizer = BooguTokenizer::from_snapshot(&root)?;
    let residency = build_residency(spec, &root)?;
    // Optional PiD decoder overlay (epic 7840, sc-7846): Boogu's FLUX.1 16-ch VAE latent space has a
    // PiD student (the `flux` backbone), so the final decode can route through `mlx_gen_pid` when
    // `req.use_pid` is set. Loaded only when the spec carries `pid`; native VAE decode otherwise.
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(Box::new(Boogu {
        descriptor,
        loaded_spec: spec.clone(),
        quantization_bits,
        tokenizer,
        residency,
        pid,
    }))
}

/// The load quantizer changes dense layers but leaves packed layers alone. A conflicting packed
/// tier and request may therefore produce mixed component numerics; do not claim one effective tier.
fn effective_component_quant_bits(
    stored: Option<i32>,
    requested: Option<i32>,
) -> Option<Option<i32>> {
    match (stored, requested) {
        (Some(stored), Some(requested)) if stored != requested => None,
        (Some(stored), _) => Some(Some(stored)),
        (None, requested) => Some(requested),
    }
}

/// The policy→[`Residency`] dispatch every Boogu variant shares (sc-11126, F-180), routed through the
/// single [`Residency::from_policy`] seam so no variant re-derives the `match offload_policy`.
/// `Resident` eager-loads the mllm encoder + heavy DiT/VAE now (byte-identical to the pre-seam
/// `BooguPipeline::from_snapshot` composition — the same per-phase loaders over independent weight
/// files, deterministic RNG-free quant); `Sequential` captures the two loader closures and loads
/// nothing now, deferring each to [`Residency::run`]. The heavy loader's `use_pid` is ignored (Boogu's
/// PiD overlay is held on the generator, not the heavy bundle). The deferral is weight-free-testable:
/// under `Sequential` this touches no component weights, so a dispatch that ignored `offload_policy`
/// would eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    root: &Path,
) -> Result<Residency<BooguEncoders, BooguHeavy>> {
    let quant_bits = spec.quantize.map(Quant::affine_bits).transpose()?;
    let root_text = root.to_path_buf();
    let root_heavy = root.to_path_buf();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let mut enc = BooguEncoders::load(&root_text)?;
            // No-op when the snapshot is already packed (the turnkey default); quantizes the dense
            // Qwen3-VL text tower otherwise (`AdaptableLinear::quantize` skips already-quantized bases).
            if let Some(bits) = quant_bits {
                enc.quantize(bits)?;
            }
            Ok(enc)
        },
        move |_use_pid| {
            let mut heavy = BooguHeavy::load(&root_heavy)?;
            if let Some(bits) = quant_bits {
                heavy.quantize(bits)?;
            }
            Ok(heavy)
        },
    )
}

/// Construct a Boogu **Base** generator (true-CFG text-to-image) from a [`LoadSpec`].
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor())
}

/// Construct a Boogu **Turbo** generator (DMD few-step, CFG-free) from a [`LoadSpec`].
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor_turbo())
}

/// Construct a Boogu **Edit** generator (instruction image-edit) from a [`LoadSpec`].
pub fn load_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor_edit())
}

impl Generator for Boogu {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            return mlx_gen::image_execution::without_capture(|| {
                self.generate_observed(req, None, on_progress)
            });
        }
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| mlx_gen::gen_core::Error::Msg("Boogu render produced no images".into()))
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::GenerationReport> {
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            let output = mlx_gen::image_execution::without_capture(|| {
                self.generate_observed(req, None, on_progress)
            })?;
            return Ok(mlx_gen::gen_core::GenerationReport::from_output(output));
        }
        let plan = compile_trajectory_for(&self.descriptor, req)?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &mlx_gen::gen_core::RenderPlan,
        scope: Option<&mut dyn mlx_gen::gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::GenerationReport> {
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: planned execution is not supported on the Boogu Turbo DMD student path (ENG-5)",
                self.descriptor.id
            )));
        }
        validate_request(&self.descriptor, req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&mlx_gen::gen_core::RenderLoadBinding::from_spec(
            &self.loaded_spec,
        ))?;
        let run = mlx_gen::gen_core::RenderRun::new(plan, &req.cancel, scope);
        let output = mlx_gen::image_execution::without_capture(|| {
            self.generate_observed(req, Some(&run), on_progress)
        })?;
        let mut report = mlx_gen::gen_core::GenerationReport::from_output(output);
        report.low_res_start = low_res_start_report(plan)?;
        Ok(report)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            return mlx_gen::image_execution::complete_image_report_attempt(
                self.descriptor(),
                &req.cancel,
                || {
                    self.record_execution_metadata();
                    let mut progress =
                        |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                    self.generate_observed(req, None, &mut progress)
                        .map(mlx_gen::gen_core::GenerationReport::from_output)
                },
            );
        }
        let plan = compile_trajectory_for(&self.descriptor, req)?;
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &mlx_gen::gen_core::RenderPlan,
        scope: Option<&mut dyn mlx_gen::gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: planned execution is not supported on the Boogu Turbo DMD student path (ENG-5)",
                self.descriptor.id
            )));
        }
        validate_request(&self.descriptor, req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&mlx_gen::gen_core::RenderLoadBinding::from_spec(
            &self.loaded_spec,
        ))?;
        let run = mlx_gen::gen_core::RenderRun::new(plan, &req.cancel, scope);
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                self.record_execution_metadata();
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                let output = self.generate_observed(req, Some(&run), &mut progress)?;
                let mut report = mlx_gen::gen_core::GenerationReport::from_output(output);
                report.low_res_start = low_res_start_report(plan)?;
                Ok(report)
            },
        )
    }
}

fn low_res_start_report(
    plan: &mlx_gen::gen_core::RenderPlan,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::LowResStartExecutionReport>> {
    // The retained boundary estimate lives in the f32 sampler trajectory.
    plan.low_res_start_execution_report("f32", 4, "base_dit")
}

impl Boogu {
    fn record_execution_metadata(&self) {
        if let Some((text_bits, dit_bits)) = self.quantization_bits {
            mlx_gen::image_execution::policy(
                "Boogu.component_quantization",
                serde_json::json!({
                    "text_encoder_bits": text_bits,
                    "transformer_bits": dit_bits
                }),
            );
        } else {
            mlx_gen::image_execution::unavailable(
                "boogu_component_quantization_unavailable",
                "the loaded Boogu text encoder or transformer has no single proven quantization tier",
                "component_quantization",
            );
        }
        mlx_gen::image_execution::policy_debug(
            "Boogu.loaded_sequential_residency",
            &self.residency.is_sequential(),
        );
    }

    fn generate_observed(
        &self,
        req: &GenerationRequest,
        run: Option<&mlx_gen::gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, run, on_progress)
            .map_err(Into::into)
    }
}

impl Boogu {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`].
    ///
    /// Dispatches to the per-variant residency lifecycle. Each variant drives the shared
    /// [`Residency::run`] seam (encode the mllm conditioning → **drop the mllm** under `Sequential` →
    /// load the DiT/VAE → denoise/decode → free the heavy bundle); the seam owns the eval/drop/clear
    /// discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        run: Option<&mlx_gen::gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = run.map_or_else(
            || req.seed.unwrap_or_else(default_seed),
            |r| r.plan().base_seed(),
        );
        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            self.generate_turbo(req, base_seed, on_progress)
        } else if self.descriptor.id == BOOGU_IMAGE_EDIT_ID {
            self.generate_edit(req, base_seed, run, on_progress)
        } else {
            self.generate_base(req, base_seed, run, on_progress)
        }
    }

    /// Base (true-CFG text-to-image, with a single-`Reference` img2img latent-init). Phase A encodes
    /// the positive instruction + (guidance > 1) the empty/drop CFG-negative; phase B resolves the PiD
    /// decoder + `from_ldm` truncation, VAE-encodes any img2img reference ONCE (seed-independent), and
    /// runs the true-CFG denoise/decode over the `count` loop.
    fn generate_base(
        &self,
        req: &GenerationRequest,
        base_seed: u64,
        run: Option<&mlx_gen::gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let id = self.descriptor.id;
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        mlx_gen::image_execution::scalar("boogu.guidance", guidance);
        let preflight =
            crate::pipeline::preflight_text(&self.tokenizer, &req.prompt, guidance != 1.0)?
                .with_negative_prompt(&self.tokenizer, req.negative_prompt.as_deref())?;
        // img2img (epic 8588 A4.3, sc-10191): a single `Reference` seeds the true-CFG denoise from the
        // VAE-encoded reference at a strength-derived start step; no reference → pure t2i (start 0).
        // Seed-independent — resolved above the residency lifecycle.
        let reference = resolve_reference(req, id)?;
        let start_step = reference
            .map(|(_, strength)| init_time_step(steps, Some(strength)))
            .unwrap_or(0);
        let sigmas = crate::pipeline::base_request_schedule(req, steps)?;

        self.residency.run(
            &req.cancel,
            // Boogu's PiD overlay is held on the generator, not the heavy bundle; the loader ignores it.
            req.use_pid(),
            on_progress,
            // ── Phase A: true-CFG mllm conditioning (Sequential loads the mllm, encodes, materializes,
            // drops it + clear_cache before the DiT/VAE load; Resident borrows the warm encoder).
            |enc: &BooguEncoders| match run {
                Some(run) => run.encode(|| enc.encode_base_preflight(&preflight)),
                None => enc.encode_base_preflight(&preflight),
            },
            |c: Option<&BooguBaseCond>| match c {
                Some(c) => c.materialize(),
                None => Ok(()),
            },
            // ── Phase B: denoise/decode from the heavy DiT/VAE bundle over the count loop.
            |heavy: &BooguHeavy, c, on_progress| {
                // PiD decode overlay (sc-7846) + `from_ldm` early-stop (sc-8048): the img2img start
                // offsets the schedule; `flow_capture_for_request` folds any PiD σ ceiling against the
                // *start-offset* schedule so the two compose. No reference → `start_step = 0`,
                // byte-identical to a plain txt2img.
                let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, start_step);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    self.pid.as_ref(),
                    req,
                    base_seed,
                    id,
                    capture_sigma,
                )?;
                let denoise_sigmas = &sigmas[..keep];
                let (init_w, init_h) =
                    run.map_or((req.width, req.height), |r| r.plan().phase_output_size(0));
                // VAE-encode the img2img reference ONCE (seed-independent) — hoisted out of the count
                // loop (the qwen-image F-118 fix). `None` for pure txt2img.
                let clean = match reference {
                    Some((image, _)) if start_step > 0 => {
                        Some(heavy.encode_init_clean(image, init_w, init_h)?)
                    }
                    _ => None,
                };
                if let Some(run) = run {
                    run.begin_rendering()?;
                }
                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    let opts = GenerateOptions {
                        height: req.height,
                        width: req.width,
                        steps,
                        text_guidance_scale: guidance,
                        seed: base_seed.wrapping_add(n as u64),
                        sampler: req.sampler.as_deref().map(str::to_owned),
                        scheduler: req.scheduler.as_deref().map(str::to_owned),
                    };
                    let decoder = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
                    let img = match &clean {
                        Some(clean) => heavy.render_base_img2img_with_run(
                            &c,
                            clean,
                            start_step,
                            &opts,
                            denoise_sigmas,
                            decoder,
                            &req.cancel,
                            on_progress,
                            run,
                        )?,
                        None => heavy.render_base_t2i_with_run(
                            &c,
                            &opts,
                            denoise_sigmas,
                            decoder,
                            &req.cancel,
                            on_progress,
                            run,
                        )?,
                    };
                    images.push(img);
                }
                if let Some(run) = run {
                    run.finish_rendering();
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }

    /// Instruction-Edit (true-CFG, 1..=`MAX_EDIT_REFS` source references packed into the DiT sequence).
    /// Phase A encodes the (optionally image-conditioned) instruction; phase B VAE-encodes the reference
    /// spatial latents ONCE and runs the true-CFG denoise/decode over the `count` loop.
    fn generate_edit(
        &self,
        req: &GenerationRequest,
        base_seed: u64,
        run: Option<&mlx_gen::gen_core::RenderRun<'_, '_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let id = self.descriptor.id;
        // Source images arrive as `Reference` / `MultiReference` (1..=MAX_EDIT_REFS); the prompt is the
        // edit instruction. Clone once into an owned slice (cheap next to the multi-step DiT denoise).
        let references: Vec<Image> = resolve_edit_references(req)?.into_iter().cloned().collect();
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        mlx_gen::image_execution::scalar("boogu.guidance", guidance);
        // Host-only smart-resize/token admission runs before Residency can load or call the MLX
        // text/vision tower or VAE phase. Both CFG legs reuse the exact checked ids/counts.
        let preflight = crate::pipeline::preflight_edit(
            &self.tokenizer,
            &references,
            &req.prompt,
            guidance != 1.0,
            true,
            false,
        )?
        .with_negative_prompt(&self.tokenizer, req.negative_prompt.as_deref())?;
        let sigmas = crate::pipeline::base_request_schedule(req, steps)?;
        // The faithful edit runs each reference through the vision tower (image-conditioned) and the
        // CFG-negative is the text-only empty/drop instruction — the reference defaults.
        let edit_opts = EditOptions {
            height: req.height,
            width: req.width,
            steps,
            text_guidance_scale: guidance,
            seed: base_seed,
            condition_on_image: true,
            use_input_images_4_neg_instruct: false,
            sampler: req.sampler.as_deref().map(str::to_owned),
            scheduler: req.scheduler.as_deref().map(str::to_owned),
        };

        self.residency.run(
            &req.cancel,
            req.use_pid(),
            on_progress,
            |enc: &BooguEncoders| match run {
                Some(run) => run.encode(|| {
                    enc.encode_edit_preflight(&references, &edit_opts, Some(&preflight))
                }),
                None => enc.encode_edit_preflight(&references, &edit_opts, Some(&preflight)),
            },
            |c: Option<&BooguBaseCond>| match c {
                Some(c) => c.materialize(),
                None => Ok(()),
            },
            |heavy: &BooguHeavy, c, on_progress| {
                // Edit always starts the output from pure noise (the references shape the DiT sequence,
                // not the init latent), so there is no img2img start-step; `flow_capture_for_request`
                // folds any PiD σ ceiling against the full schedule (`start_step = 0`).
                let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    self.pid.as_ref(),
                    req,
                    base_seed,
                    id,
                    capture_sigma,
                )?;
                let denoise_sigmas = &sigmas[..keep];
                // VAE-encode the reference spatial latents ONCE (seed-independent) — hoisted out of the
                // count loop (the qwen-image F-118 fix).
                let ref_latents = heavy.encode_ref_latents(&references)?;
                if let Some(run) = run {
                    run.begin_rendering()?;
                }
                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    let opts = EditOptions {
                        seed: base_seed.wrapping_add(n as u64),
                        ..edit_opts.clone()
                    };
                    let img = heavy.render_edit_with_run(
                        &c,
                        &ref_latents,
                        &opts,
                        denoise_sigmas,
                        pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
                        &req.cancel,
                        on_progress,
                        run,
                    )?;
                    images.push(img);
                }
                if let Some(run) = run {
                    run.finish_rendering();
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }

    /// Turbo (DMD few-step, CFG-free, with a single-`Reference` img2img latent-init). Phase A encodes
    /// the positive instruction only (no unconditional branch); phase B runs the few-step DMD
    /// denoise/decode. The `from_ldm` early-stop is rejected (the DMD loop yields clean x0 estimates,
    /// not a σ>0 latent, and is decode-bound); the PiD decoder is minted at the clean terminal σ=0.
    fn generate_turbo(
        &self,
        req: &GenerationRequest,
        base_seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let id = self.descriptor.id;
        // The Turbo few-step DMD loop predicts a CLEAN x0 estimate each step and re-noises; there is no
        // "noisy x_k at σ>0" to hand PiD, and Turbo is decode-bound (sc-7993), so the `from_ldm`
        // early-stop has no benefit. Reject `pid_capture_sigma` loudly; the Base variant is the
        // supported path for from_ldm (sc-8048). Resolved before the residency (validation-shaped).
        if req.use_pid() && req.pid_capture_sigma().is_some() {
            return Err(Error::Msg(format!(
                "{id}: pid_capture_sigma (from_ldm early-stop) is not supported on the Boogu Turbo \
                 few-step DMD path — it is decode-bound with no early-stop benefit and the DMD loop \
                 produces clean x0 estimates, not a σ>0 latent; use the Base variant for from_ldm \
                 (sc-8048)"
            )));
        }
        let steps = req.steps.unwrap_or(DEFAULT_TURBO_STEPS) as usize;
        let preflight = crate::pipeline::preflight_text(
            &self.tokenizer,
            &req.prompt,
            req.guidance.unwrap_or(0.0) != 0.0,
        )?
        .with_negative_prompt(&self.tokenizer, req.negative_prompt.as_deref())?;
        // img2img (epic 8588 A4.3, sc-10191): a single `Reference` seeds the few-step DMD denoise from
        // the VAE-encoded reference at a strength-derived start step; no reference → pure t2i.
        let reference = resolve_reference(req, id)?;
        let start_step = reference
            .map(|(_, strength)| init_time_step(steps, Some(strength)))
            .unwrap_or(0);

        self.residency.run(
            &req.cancel,
            req.use_pid(),
            on_progress,
            |enc: &BooguEncoders| enc.encode_turbo_preflight(&preflight),
            |encoded: Option<&BooguBaseCond>| match encoded {
                Some(c) => c.materialize(),
                None => Ok(()),
            },
            |heavy: &BooguHeavy, c, on_progress| {
                // No early-stop on Turbo: the PiD decoder (when `use_pid`) is minted at the clean
                // terminal σ=0 (the DMD loop's final x0 estimate), matching the full-loop latent.
                let pid_decoder =
                    resolve_pid_decoder_at_sigma(self.pid.as_ref(), req, base_seed, id, 0.0)?;
                // VAE-encode the img2img reference ONCE (seed-independent) — hoisted out of the loop.
                let clean = match reference {
                    Some((image, _)) if start_step > 0 => {
                        Some(heavy.encode_init_clean(image, req.width, req.height)?)
                    }
                    _ => None,
                };
                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    let opts = TurboOptions {
                        height: req.height,
                        width: req.width,
                        steps,
                        seed: base_seed.wrapping_add(n as u64),
                        conditioning_sigma: DEFAULT_TURBO_SIGMA,
                        guidance: req.guidance.unwrap_or(0.0),
                        scheduler_shift: req.scheduler_shift,
                        sampler: req.sampler.as_deref().map(str::to_owned),
                        scheduler: req.scheduler.as_deref().map(str::to_owned),
                    };
                    let decoder = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
                    let img = match &clean {
                        Some(clean) => heavy.render_turbo_img2img(
                            &c,
                            clean,
                            start_step,
                            &opts,
                            decoder,
                            &req.cancel,
                            on_progress,
                        )?,
                        None => {
                            heavy.render_turbo_t2i(&c, &opts, decoder, &req.cancel, on_progress)?
                        }
                    };
                    images.push(img);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// The instruction-edit source images, in order: any mix of [`Conditioning::Reference`] (single) and
/// [`Conditioning::MultiReference`] (a list), flattened. The Edit path needs at least one and at most
/// [`MAX_EDIT_REFS`] (the DiT's `image_index_embedding` slot count); none, or more than the cap, is an
/// error.
fn resolve_edit_references(req: &GenerationRequest) -> Result<Vec<&Image>> {
    // Preserve the typed strength failure independently of cardinality: scan the whole request for
    // the edit-inert field before performing the bounded count pass.
    for c in &req.conditioning {
        if let Conditioning::Reference {
            strength: Some(_), ..
        } = c
        {
            return Err(Error::Unsupported(
                "boogu_image_edit: per-reference strength is not supported for instruction edit; \
                 omit conditioning.reference.strength"
                    .into(),
            ));
        }
    }

    // Count first, checking every cumulative MultiReference length before allocating/extending the
    // flattened vector. A pathological list can therefore never cause a large pointer allocation.
    let mut count = 0usize;
    for c in &req.conditioning {
        let add = match c {
            Conditioning::Reference { .. } => 1,
            Conditioning::MultiReference { images } => images.len(),
            _ => 0,
        };
        count = count
            .checked_add(add)
            .ok_or_else(|| Error::Msg("boogu_image_edit: reference image count overflow".into()))?;
        validate_max_edit_reference_count(count)?;
    }
    validate_edit_reference_count(count)?;

    let mut refs: Vec<&Image> = Vec::with_capacity(count);
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {}
        }
    }
    debug_assert_eq!(refs.len(), count);
    for (index, image) in refs.iter().enumerate() {
        validate_edit_reference(index, image)?;
    }
    Ok(refs)
}

/// Reject an over-cap reference count before any flattened or per-reference admission vectors can
/// be allocated. Kept separate from the final non-empty check because registered requests may carry
/// unrelated conditioning entries before their first reference.
fn validate_max_edit_reference_count(count: usize) -> Result<()> {
    if count > MAX_EDIT_REFS {
        return Err(Error::Msg(format!(
            "boogu_image_edit: at most {MAX_EDIT_REFS} reference images are supported (got {count})"
        )));
    }
    Ok(())
}

fn validate_edit_reference_count(count: usize) -> Result<()> {
    validate_max_edit_reference_count(count)?;
    if count == 0 {
        return Err(Error::Msg(
            "boogu_image_edit: an instruction edit requires at least one source reference image"
                .into(),
        ));
    }
    Ok(())
}

/// Apply the registered generator's complete raw-reference carrier contract to a direct resident
/// pipeline slice. This is allocation-free and must run before token/grid admission or component
/// entry so the public [`crate::BooguPipeline`] route cannot bypass the generator boundary.
pub(crate) fn validate_raw_edit_references(references: &[Image]) -> Result<()> {
    validate_edit_reference_count(references.len())?;
    for (index, image) in references.iter().enumerate() {
        validate_edit_reference(index, image)?;
    }
    Ok(())
}

/// Validate one Edit source at the public generator boundary, before the request can enter either
/// the vision encoder or the raw-resolution VAE/DiT spatial path. References retain their own
/// geometry but must fit the advertised Boogu envelope, the patch/VAE stride, and the RGB8 carrier
/// contract.
fn validate_edit_reference(index: usize, image: &Image) -> Result<()> {
    if image.width == 0 || image.height == 0 {
        return Err(Error::Msg(format!(
            "boogu_image_edit: reference {index} must have non-zero dimensions (got {}x{})",
            image.width, image.height
        )));
    }
    if image.width > RES_MAX || image.height > RES_MAX {
        return Err(Error::Msg(format!(
            "boogu_image_edit: reference {index} {}x{} exceeds the {RES_MAX}x{RES_MAX} reference envelope",
            image.width, image.height
        )));
    }
    if !image.width.is_multiple_of(RES_MULTIPLE) || !image.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "boogu_image_edit: reference {index} dimensions must be multiples of {RES_MULTIPLE} (got {}x{})",
            image.width, image.height
        )));
    }
    let expected = mlx_gen::gen_core::imageops::checked_image_buffer_len(
        image.width as usize,
        image.height as usize,
        3,
    )
    .ok_or_else(|| {
        Error::Msg(format!(
            "boogu_image_edit: reference {index} RGB8 dimensions overflow host size"
        ))
    })?;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "boogu_image_edit: reference {index} RGB8 buffer has {} bytes; expected {expected} for {}x{}x3",
            image.pixels.len(),
            image.width,
            image.height
        )));
    }
    Ok(())
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Layers Boogu's model-specific constraints (non-empty prompt, size multiple-of-16, steps
/// ≥ 1, the Edit variant requires a reference) on top of the shared [`Capabilities::validate_request`]
/// floor (count/size range, negative/guidance/true_cfg flags, conditioning kinds).
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    // F-146: `trim()` so a whitespace-only prompt ("   ") is rejected too — it tokenizes to an empty
    // (or padding-only) conditioning, the same degenerate state as `""`.
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    let (default, off) = if id == BOOGU_IMAGE_TURBO_ID {
        (0.0, 0.0)
    } else {
        (DEFAULT_GUIDANCE, 1.0)
    };
    if req
        .negative_prompt
        .as_ref()
        .is_some_and(|prompt| !prompt.trim().is_empty())
        && req.guidance.unwrap_or(default) == off
    {
        return Err(Error::Msg(format!("{id}: a negative prompt requires CFG")));
    }

    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    // The Edit variant needs 1..=MAX_EDIT_REFS bounded RGB8 source references. Resolve the same
    // production boundary generate uses so validate cannot accept a request that later fails only
    // after vision/VAE allocation, and so edit-only per-reference strength is never discarded.
    if id == BOOGU_IMAGE_EDIT_ID {
        resolve_edit_references(req)?;
    } else {
        resolve_reference(req, id)?;
    }
    if let Some(low_res) = &req.features.low_res_start {
        if id == BOOGU_IMAGE_TURBO_ID {
            return Err(Error::Unsupported(format!(
                "{id}: low_res_start is not supported on the Boogu Turbo DMD student path (ENG-5)"
            )));
        }
        if id == BOOGU_IMAGE_EDIT_ID {
            return Err(Error::Unsupported(format!(
                "{id}: low_res_start conflicts with pass-0 edit conditioning"
            )));
        }
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let sigmas = crate::pipeline::base_request_schedule(req, steps)?;
        let start_step = resolve_reference(req, id)?
            .map(|(_, strength)| init_time_step(steps, Some(strength)))
            .unwrap_or(0);
        let (_capture_sigma, keep) = flow_capture_for_request(req, &sigmas, start_step);
        let denoise_sigmas = &sigmas[start_step.min(keep.saturating_sub(1))..keep];
        let executed_steps = denoise_sigmas.len().saturating_sub(1);
        low_res.resolve_for_descriptor(desc, req, executed_steps, RES_MULTIPLE)?;
    }
    Ok(())
}

/// Compile a weights-free [`mlx_gen::gen_core::RenderPlan`] for Boogu Base (`boogu_image`) or Edit (`boogu_image_edit`).
pub fn compile_trajectory_for(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::RenderPlan> {
    if desc.id == BOOGU_IMAGE_TURBO_ID {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{}: RenderPlan compilation is not supported on the Boogu Turbo DMD student path (ENG-5)",
            desc.id
        )));
    }
    validate_request(desc, req)?;
    let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
    let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
    let solver = mlx_gen::sampler::canonical_solver(req.sampler.as_deref(), mlx_gen::Solver::Euler);
    let sigmas = crate::pipeline::base_request_schedule(req, steps)?;
    let start_step = if desc.id == BOOGU_IMAGE_EDIT_ID {
        0
    } else {
        resolve_reference(req, desc.id)?
            .map(|(_, strength)| init_time_step(steps, Some(strength)))
            .unwrap_or(0)
    };
    let (_capture_sigma, keep) = flow_capture_for_request(req, &sigmas, start_step);
    let start = start_step.min(keep.saturating_sub(1));
    let sliced = sigmas[start..].to_vec();
    let executed_steps = keep.saturating_sub(start).saturating_sub(1);
    let capture_sigma = *sliced.get(executed_steps).unwrap_or(&0.0);
    let pass = mlx_gen::gen_core::render::PackedImagePass {
        pixel_stride: RES_MULTIPLE,
        token_width: 64,
        sampler: solver.name(),
        guidance,
        sigmas: sliced,
        executed_steps,
        capture_sigma,
    };
    if desc.id == BOOGU_IMAGE_ID && req.conditioning.is_empty() {
        mlx_gen::gen_core::RenderPlan::base_image(desc, req, pass)
    } else {
        mlx_gen::gen_core::RenderPlan::packed_image_trajectory(desc, req, pass)
    }
}

fn validate_plan_load_spec(spec: &LoadSpec, id: &str) -> mlx_gen::gen_core::Result<()> {
    let _ = spec.quantize.map(Quant::affine_bits).transpose()?;
    if spec.precision != Precision::Bf16 {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    if matches!(spec.weights, WeightsSource::File(_)) {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "{id} expects a snapshot directory (mllm/ transformer/ vae/), not a single .safetensors file"
        )));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{id}: ControlNet / IP-Adapter overlays and component overrides are not supported"
        )));
    }
    Ok(())
}

/// Pre-load [`mlx_gen::gen_core::RenderPlanRegistration`] compiler for `boogu_image`.
pub fn compile_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::RenderPlan>> {
    validate_plan_load_spec(spec, BOOGU_IMAGE_ID)?;
    compile_trajectory_for(&descriptor(), req).map(|plan| Some(plan.bind_load(spec)))
}

/// Pre-load [`mlx_gen::gen_core::RenderPlanRegistration`] compiler for `boogu_image_edit`.
pub fn compile_edit_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::RenderPlan>> {
    validate_plan_load_spec(spec, BOOGU_IMAGE_EDIT_ID)?;
    compile_trajectory_for(&descriptor_edit(), req).map(|plan| Some(plan.bind_load(spec)))
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the
/// Qwen3-VL text/vision encoder under **`mllm/`** (NOT `text_encoder/`, so a name-guessing consumer
/// would read it as ZERO — the seam exists so the provider reports the real bytes), the DiT
/// (`transformer/`), and the VAE (`vae/`), summed from the subdirs [`crate::loader`] loads. Shared by
/// boogu image/turbo/edit.
///
/// The engine advertises `supports_sequential_offload` (sc-10840) and this split is the staged
/// peak the shared `Residency` seam bounds (`max(mllm, DiT+VAE)`). The worker's fit-gate keys on
/// that descriptor capability bit (there is no allowlist) and consumes this split generically through
/// the registered footprint seam (this crate reports the bytes; the worker decides how to use them).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(spec, &["mllm"], &["transformer"], &["vae"])
}

pub const BASE_RENDER_PLAN_REGISTRATION: mlx_gen::gen_core::RenderPlanRegistration =
    mlx_gen::gen_core::RenderPlanRegistration {
        provider_id: BOOGU_IMAGE_ID,
        compile: compile_render_plan,
    };

pub const EDIT_RENDER_PLAN_REGISTRATION: mlx_gen::gen_core::RenderPlanRegistration =
    mlx_gen::gen_core::RenderPlanRegistration {
        provider_id: BOOGU_IMAGE_EDIT_ID,
        compile: compile_edit_render_plan,
    };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_quantization_reports_effective_affine_tier() {
        assert_eq!(effective_component_quant_bits(None, None), Some(None));
        assert_eq!(effective_component_quant_bits(None, Some(4)), Some(Some(4)));
        assert_eq!(effective_component_quant_bits(Some(8), None), Some(Some(8)));
        assert_eq!(
            effective_component_quant_bits(Some(8), Some(8)),
            Some(Some(8))
        );
        assert_eq!(effective_component_quant_bits(Some(8), Some(4)), None);
        let mut unsupported = LoadSpec::new(WeightsSource::Dir("/unused".into()));
        unsupported.quantize = Some(Quant::Nvfp4);
        assert!(build_residency(&unsupported, Path::new("/unused")).is_err());
    }

    fn write_tiny_tokenizer(root: &Path, overlong_negative: bool) {
        let mllm = root.join("mllm");
        std::fs::create_dir_all(&mllm).unwrap();
        let literals = [
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
        ];
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".into(), serde_json::json!(0));
        let added = literals
            .iter()
            .enumerate()
            .map(|(index, literal)| {
                let id = index + 1;
                vocab.insert((*literal).into(), serde_json::json!(id));
                serde_json::json!({
                    "id": id,
                    "content": literal,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                })
            })
            .collect::<Vec<_>>();
        vocab.insert("burst".into(), serde_json::json!(literals.len() + 1));
        let normalizer = overlong_negative.then(|| {
            serde_json::json!({
                "type": "Replace",
                "pattern": { "String": "Describe" },
                "content": "burst ".repeat(crate::tokenizer::MAX_TEXT_TOKENS + 1)
            })
        });
        let fixture = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": normalizer,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
        });
        std::fs::write(
            mllm.join("tokenizer.json"),
            serde_json::to_vec(&fixture).unwrap(),
        )
        .unwrap();
    }

    fn write_tiny_edit_tokenizer(root: &Path) {
        write_tiny_tokenizer(root, false);
    }

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    /// Registered-route mutation table: a Sequential snapshot containing only the host tokenizer
    /// proves that every ordinary positive/negative failure happens before the text/heavy loaders.
    /// The negative fixture expands `SYSTEM_PROMPT_DROP`; grounded Edit remains below 8192 while its
    /// ordinary CFG-negative crosses 1280, independently discriminating that leg.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn ordinary_text_preflight_precedes_every_registered_component_route() {
        let positive_root = tempfile::tempdir().unwrap();
        write_tiny_tokenizer(positive_root.path(), false);
        let positive_spec = LoadSpec::new(WeightsSource::Dir(positive_root.path().into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        for load_route in [
            load as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
            load_turbo as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
        ] {
            let generator = load_route(&positive_spec).unwrap();
            let request = GenerationRequest {
                prompt: "<|image_pad|>".repeat(crate::tokenizer::MAX_TEXT_TOKENS + 1),
                ..req(512, 512)
            };
            let error = generator
                .generate(&request, &mut |_| {})
                .unwrap_err()
                .to_string();
            assert!(error.contains("max_text_tokens=1280"), "{error}");
        }

        let negative_root = tempfile::tempdir().unwrap();
        write_tiny_tokenizer(negative_root.path(), true);
        let negative_spec = LoadSpec::new(WeightsSource::Dir(negative_root.path().into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        for (load_route, conditioning) in [
            (
                load as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
                Vec::new(),
            ),
            (
                load_edit as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
                vec![Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                }],
            ),
        ] {
            let generator = load_route(&negative_spec).unwrap();
            let request = GenerationRequest {
                prompt: "ordinary positive".into(),
                guidance: Some(4.0),
                conditioning,
                ..req(512, 512)
            };
            let error = generator
                .generate(&request, &mut |_| {})
                .unwrap_err()
                .to_string();
            assert!(error.contains("max_text_tokens=1280"), "{error}");
        }
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED model::tests::ordinary_text_preflight_precedes_every_registered_component_route");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn ordinary_text_preflight_admits_exactly_1280_on_every_positive_route() {
        let root = tempfile::tempdir().unwrap();
        write_tiny_tokenizer(root.path(), false);
        let tok = BooguTokenizer::from_snapshot(root.path()).unwrap();
        let template_len = tok.preflight_t2i("").unwrap().len();
        let prompt = "<|image_pad|>".repeat(crate::tokenizer::MAX_TEXT_TOKENS - template_len);
        assert_eq!(
            tok.preflight_t2i(&prompt).unwrap().len(),
            crate::tokenizer::MAX_TEXT_TOKENS
        );

        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        for load_route in [
            load as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
            load_turbo as fn(&LoadSpec) -> Result<Box<dyn Generator>>,
        ] {
            let generator = load_route(&spec).unwrap();
            let error = generator
                .generate(
                    &GenerationRequest {
                        prompt: prompt.clone(),
                        ..req(512, 512)
                    },
                    &mut |_| {},
                )
                .unwrap_err()
                .to_string();
            assert!(
                !error.contains("max_text_tokens"),
                "exactly 1280 must pass admission: {error}"
            );
        }
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED model::tests::ordinary_text_preflight_admits_exactly_1280_on_every_positive_route");
    }

    #[test]
    fn edit_preflight_discriminates_text_and_optional_grounded_cfg_legs() {
        let root = tempfile::tempdir().unwrap();
        write_tiny_tokenizer(root.path(), false);
        let tok = BooguTokenizer::from_snapshot(root.path()).unwrap();
        // Exercise the direct resident seam at the inclusive five-reference boundary while each
        // conditioning/CFG mode proves that the exact admitted leg set reaches component entry.
        let references = (0..MAX_EDIT_REFS).map(|_| img(16, 16)).collect::<Vec<_>>();

        for (condition_on_image, grounded_negative, expected) in [
            (true, false, (false, true, true, false)),
            (true, true, (false, true, false, true)),
            (false, false, (true, false, true, false)),
        ] {
            let opts = crate::pipeline::EditOptions {
                condition_on_image,
                use_input_images_4_neg_instruct: grounded_negative,
                ..Default::default()
            };
            let admitted = crate::pipeline::with_resident_edit_preflight(
                &tok,
                &references,
                "edit",
                &opts,
                |preflight| Ok(preflight.admitted_legs()),
            )
            .unwrap();
            assert_eq!(admitted, expected);
        }
    }

    /// The exported resident `BooguPipeline` seam must reject every raw-reference boundary before
    /// it enters the closure that owns text/vision/VAE work. Each row is independently
    /// mutation-sensitive: weakening count, geometry, alignment, or RGB8 admission makes its entry
    /// counter non-zero instead of returning the named typed diagnostic.
    #[test]
    fn resident_edit_reference_preflight_has_zero_component_entry_on_every_invalid_carrier() {
        use std::cell::Cell;

        let root = tempfile::tempdir().unwrap();
        write_tiny_edit_tokenizer(root.path());
        let tok = BooguTokenizer::from_snapshot(root.path()).unwrap();
        let opts = crate::pipeline::EditOptions::default();
        let invalid = [
            ("zero references", Vec::new(), "at least one"),
            (
                "six references",
                (0..=MAX_EDIT_REFS).map(|_| img(16, 16)).collect(),
                "at most 5",
            ),
            (
                "zero geometry",
                vec![Image {
                    width: 0,
                    height: 16,
                    pixels: Vec::new(),
                }],
                "non-zero",
            ),
            (
                "oversized geometry",
                vec![img(RES_MAX + RES_MULTIPLE, 16)],
                "reference envelope",
            ),
            (
                "misaligned geometry",
                vec![img(RES_MULTIPLE + 1, RES_MULTIPLE)],
                "multiples of 16",
            ),
            (
                "malformed RGB8",
                vec![Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0; 16 * 16 * 3 - 1],
                }],
                "RGB8 buffer",
            ),
        ];

        for (case, references, expected) in invalid {
            let component_entries = Cell::new(0usize);
            let error = crate::pipeline::with_resident_edit_preflight(
                &tok,
                &references,
                "edit",
                &opts,
                |_| {
                    component_entries.set(component_entries.get() + 1);
                    Ok(())
                },
            )
            .unwrap_err()
            .to_string();
            assert!(
                error.contains(expected),
                "{case}: expected {expected:?}, got {error}"
            );
            assert_eq!(
                component_entries.get(),
                0,
                "{case}: invalid carrier entered resident components"
            );
        }
    }

    /// The real generator boundary must reject combined edit budgets before entering either
    /// sequential component loader. The snapshot contains only a tokenizer: reaching the MLX tower
    /// or VAE loader would therefore replace the declared budget error with a missing-weight error.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn edit_overbudget_preflight_never_enters_tower_or_vae_loaders() {
        let root = tempfile::tempdir().unwrap();
        write_tiny_edit_tokenizer(root.path());
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let generator = load_edit(&spec).unwrap();

        let two_max = GenerationRequest {
            prompt: "edit".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img(2048, 2048), img(2048, 2048)],
            }],
            ..Default::default()
        };
        let error = generator
            .generate(&two_max, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("max_edit_tokens=8192"), "{error}");

        let prompt_over = GenerationRequest {
            prompt: "<|image_pad|>".repeat(crate::tokenizer::MAX_EDIT_TOKENS),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..Default::default()
        };
        let error = generator
            .generate(&prompt_over, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("max_edit_tokens=8192"), "{error}");
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED model::tests::edit_overbudget_preflight_never_enters_tower_or_vae_loaders");
    }

    #[test]
    fn descriptor_is_boogu_image() {
        let d = descriptor();
        assert_eq!(d.id, "boogu_image");
        assert_eq!(d.family, "boogu");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // Base is text-to-image with a single-`Reference` img2img surface (sc-10191); no
        // MultiReference (that is the Edit checkpoint's).
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn descriptor_turbo_is_cfg_free_else_matches_base() {
        let (b, t) = (descriptor(), descriptor_turbo());
        assert_eq!(t.id, "boogu_image_turbo");
        assert_eq!(t.family, b.family);
        assert_eq!(t.modality, b.modality);
        assert!(b.capabilities.supports_guidance);
        assert!(t.capabilities.supports_guidance);
        // Turbo inherits the Base img2img `Reference` surface via clone.
        assert_eq!(t.capabilities.conditioning, b.capabilities.conditioning);
        assert_eq!(
            t.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(
            t.capabilities.supported_quants,
            b.capabilities.supported_quants
        );
    }

    #[test]
    fn descriptor_turbo_advertises_the_stochastic_sampler_subset_and_scheduler_axis() {
        let (b, t) = (descriptor(), descriptor_turbo());
        // Turbo exposes the DMD-compatible stochastic samplers (incl. flow-aware `lcm`), a strict subset
        // of the Base sampler menu, plus the full curated scheduler axis (the ComfyUI lcm/sgm_uniform).
        assert_eq!(
            t.capabilities.samplers,
            vec!["lcm", "euler_ancestral", "dpmpp_sde"]
        );
        assert_eq!(t.capabilities.schedulers, b.capabilities.schedulers);
        assert!(t.capabilities.schedulers.contains(&"sgm_uniform"));
        for s in &t.capabilities.samplers {
            assert!(
                b.capabilities.samplers.contains(s),
                "turbo sampler {s:?} must be a subset of the Base curated menu"
            );
        }
        // The deterministic ODE solvers degrade on the few-step student and are NOT advertised.
        for excluded in ["euler", "ddim", "heun", "dpmpp_2m", "uni_pc"] {
            assert!(!t.capabilities.samplers.contains(&excluded));
        }
    }

    #[test]
    fn turbo_validate_gates_to_the_advertised_sampler_subset() {
        let d = descriptor_turbo();
        // Advertised stochastic samplers are accepted, optionally with a curated scheduler
        // (the ComfyUI lcm/sgm_uniform combo).
        for s in ["lcm", "euler_ancestral", "dpmpp_sde"] {
            let r = GenerationRequest {
                sampler: Some(s.into()),
                scheduler: Some("sgm_uniform".into()),
                ..req(512, 512)
            };
            assert!(
                validate_request(&d, &r).is_ok(),
                "turbo should accept {s}+sgm_uniform"
            );
        }
        // An unadvertised sampler (degraded on the few-step student) is rejected before any work.
        let bad = GenerationRequest {
            sampler: Some("dpmpp_2m".into()),
            ..req(512, 512)
        };
        assert!(validate_request(&d, &bad).is_err());
    }

    #[test]
    fn descriptor_edit_adds_reference() {
        let d = descriptor_edit();
        assert_eq!(d.id, "boogu_image_edit");
        assert!(d.capabilities.supports_guidance);
        // Both single and multi reference are advertised (the DiT carries 5 image-index slots).
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::MultiReference));
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Mask));
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                guidance: Some(4.0),
                ..req(512, 512)
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt_and_bad_size() {
        assert!(validate_request(&descriptor(), &GenerationRequest::default()).is_err());
        for (w, h) in [(1000, 1000), (257, 256)] {
            let e = validate_request(&descriptor(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
        assert!(validate_request(&descriptor(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&descriptor(), &req(2064, 256)).is_err()); // above max

        // sc-12612: `RES_MULTIPLE` is the pinned stride SceneWorks ties every advertised Boogu bucket
        // to. Pin the value and mutation-check that an in-range size which is a multiple of 8 (the VAE
        // scale) but not RES_MULTIPLE (16) is still rejected with the stride error, and on-stride passes.
        assert_eq!(RES_MULTIPLE, 16);
        let off_stride = validate_request(&descriptor(), &req(1000, 1024)) // 125×8 — mult of 8, not 16
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiple of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
    }

    #[test]
    fn validate_accepts_experimental_turbo_cfg_and_negative_prompt() {
        assert!(validate_request(
            &descriptor_turbo(),
            &GenerationRequest {
                guidance: Some(4.0),
                ..req(512, 512)
            }
        )
        .is_ok());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_ok());
    }

    #[test]
    fn base_and_turbo_accept_a_single_img2img_reference() {
        // sc-10191: Base/Turbo now advertise a single-`Reference` img2img surface, so the capability
        // floor accepts one reference (with or without a strength) but rejects a second.
        let one = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: Some(0.6),
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor(), &one).is_ok());
        assert!(validate_request(&descriptor_turbo(), &one).is_ok());
        // resolve_reference returns the image + strength (falling back to req.strength when unset).
        let (_, strength) = resolve_reference(&one, "boogu_image").unwrap().unwrap();
        assert_eq!(strength, 0.6);

        // Two references on the t2i path → error (single img2img init only; multi is the Edit path).
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
            ],
            ..req(512, 512)
        };
        assert!(resolve_reference(&two, "boogu_image").is_err());

        // No reference → None (pure t2i), and a per-reference strength falls back to req.strength.
        assert!(resolve_reference(&req(512, 512), "boogu_image")
            .unwrap()
            .is_none());
        let fallback = GenerationRequest {
            strength: Some(0.4),
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..req(512, 512)
        };
        assert_eq!(
            resolve_reference(&fallback, "boogu_image")
                .unwrap()
                .unwrap()
                .1,
            0.4
        );
    }

    #[test]
    fn img2img_strength_precedence_default_and_start_step_are_explicit() {
        let resolve = |per_reference, request| {
            let request = GenerationRequest {
                strength: request,
                conditioning: vec![Conditioning::Reference {
                    image: img(512, 512),
                    strength: per_reference,
                }],
                ..req(512, 512)
            };
            resolve_reference(&request, BOOGU_IMAGE_ID)
                .unwrap()
                .unwrap()
                .1
        };

        assert_eq!(resolve(Some(0.7), Some(0.3)), 0.7);
        assert_eq!(resolve(None, Some(0.3)), 0.3);
        assert_eq!(
            resolve(None, None),
            mlx_gen::img2img::DEFAULT_IMG2IMG_STRENGTH
        );
        assert_eq!(init_time_step(20, Some(resolve(None, None))), 10);
        assert_eq!(resolve(Some(0.0), Some(0.3)), 0.0);
        assert_eq!(init_time_step(20, Some(resolve(Some(0.0), None))), 0);
    }

    #[test]
    fn edit_accepts_one_to_five_references() {
        // No reference → error.
        assert!(validate_request(&descriptor_edit(), &req(512, 512)).is_err());
        assert!(resolve_edit_references(&req(512, 512)).is_err());

        // A request carrying `n` single `Reference` conditionings.
        let with_refs = |n: usize| GenerationRequest {
            conditioning: (0..n)
                .map(|_| Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                })
                .collect(),
            ..req(512, 512)
        };

        // 1..=MAX_EDIT_REFS single references → ok (the DiT has 5 image-index slots), flattened in order.
        for n in 1..=MAX_EDIT_REFS {
            assert!(
                validate_request(&descriptor_edit(), &with_refs(n)).is_ok(),
                "{n} refs should validate"
            );
            assert_eq!(resolve_edit_references(&with_refs(n)).unwrap().len(), n);
        }
        // One past the cap → error.
        assert!(validate_request(&descriptor_edit(), &with_refs(MAX_EDIT_REFS + 1)).is_err());
        assert!(resolve_edit_references(&with_refs(MAX_EDIT_REFS + 1)).is_err());

        // A `MultiReference` list is accepted and flattened the same way.
        let multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img(512, 512), img(512, 512), img(512, 512)],
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &multi).is_ok());
        assert_eq!(resolve_edit_references(&multi).unwrap().len(), 3);

        // A `MultiReference` list past the cap → error.
        let multi_over = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..=MAX_EDIT_REFS).map(|_| img(512, 512)).collect(),
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &multi_over).is_err());

        // Mutation guard: the cap is cumulative across multiple carriers, not per conditioning.
        let split_over = GenerationRequest {
            conditioning: vec![
                Conditioning::MultiReference {
                    images: (0..3).map(|_| img(16, 16)).collect(),
                },
                Conditioning::MultiReference {
                    images: (0..3).map(|_| img(16, 16)).collect(),
                },
            ],
            ..req(512, 512)
        };
        let error = resolve_edit_references(&split_over)
            .unwrap_err()
            .to_string();
        assert!(error.contains("at most 5"), "{error}");
    }

    #[test]
    fn edit_reference_geometry_and_rgb8_are_rejected_at_validation_boundary() {
        let edit_request = |image| GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image,
                strength: None,
            }],
            ..req(512, 512)
        };

        // Small-but-aligned sources remain supported; the output minimum is not imposed on source
        // geometry. This guards against accidentally narrowing Edit while adding the upper bound.
        assert!(validate_request(&descriptor_edit(), &edit_request(img(16, 16))).is_ok());
        assert!(validate_request(&descriptor_edit(), &edit_request(img(RES_MAX, 16))).is_ok());
        assert!(validate_request(&descriptor_edit(), &edit_request(img(16, RES_MAX))).is_ok());

        let cases = [
            (
                Image {
                    width: 0,
                    height: 16,
                    pixels: vec![],
                },
                "non-zero",
            ),
            (img(RES_MAX + RES_MULTIPLE, 16), "reference envelope"),
            (img(RES_MAX - 1, 512), "multiples of 16"),
            (
                Image {
                    width: 512,
                    height: 512,
                    pixels: vec![0; 512 * 512 * 3 - 1],
                },
                "RGB8 buffer",
            ),
        ];
        for (image, expected) in cases {
            let error = validate_request(&descriptor_edit(), &edit_request(image))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} from early edit validation, got: {error}"
            );
        }

        // Every ordered member is validated, including images carried by MultiReference. Checking
        // only the first item would let this malformed second source reach the vision/VAE path.
        let malformed_second = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![
                    img(512, 512),
                    Image {
                        width: 512,
                        height: 512,
                        pixels: vec![0; 12],
                    },
                ],
            }],
            ..req(512, 512)
        };
        let error = validate_request(&descriptor_edit(), &malformed_second)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reference 1 RGB8 buffer"), "{error}");
    }

    #[test]
    fn edit_rejects_per_reference_strength_as_typed_unsupported_only_on_edit() {
        let request = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: Some(0.6),
            }],
            ..req(512, 512)
        };
        let error = validate_request(&descriptor_edit(), &request).unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
        assert!(
            error.to_string().contains("per-reference strength"),
            "{error}"
        );

        // Preserve the typed field failure even when an earlier carrier is independently over the
        // cardinality cap.
        let strength_after_over_cap = GenerationRequest {
            conditioning: vec![
                Conditioning::MultiReference {
                    images: (0..6).map(|_| img(16, 16)).collect(),
                },
                Conditioning::Reference {
                    image: img(16, 16),
                    strength: Some(0.6),
                },
            ],
            ..req(512, 512)
        };
        assert!(matches!(
            resolve_edit_references(&strength_after_over_cap),
            Err(Error::Unsupported(_))
        ));

        // Base/Turbo consume this field as img2img start strength; the edit-only guard must not
        // narrow either supported route.
        assert!(validate_request(&descriptor(), &request).is_ok());
        assert!(validate_request(&descriptor_turbo(), &request).is_ok());
        assert_eq!(
            resolve_reference(&request, BOOGU_IMAGE_ID)
                .unwrap()
                .unwrap()
                .1,
            0.6
        );
    }

    #[test]
    fn load_rejects_single_file_and_adapters() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_accepts_quant_spec() {
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let e = load(&spec).err().expect("error").to_string();
            assert!(
                !e.contains("not supported"),
                "quant should be accepted: {e}"
            );
        }
    }

    #[test]
    fn all_three_reachable_via_registry_by_id() {
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            assert!(
                crate::provider_registry()
                    .unwrap()
                    .generators()
                    .copied()
                    .any(|r| (r.descriptor)().id == id),
                "id {id} not registered"
            );
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-boogu".into()));
            let e = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .err()
                .expect("missing weights → err")
                .to_string();
            assert!(
                !e.contains("no generator registered"),
                "id {id} not resolved: {e}"
            );
        }
    }

    // ── Component residency (epic 10834, sc-10840) -----------------------------------------------

    #[test]
    fn all_three_advertise_sequential_offload() {
        // Every Boogu id honors the shared `Residency` seam (the descriptor bit the worker fit-gate
        // reads to consume the TE/DiT/VAE footprint split).
        for d in [descriptor(), descriptor_turbo(), descriptor_edit()] {
            assert!(
                d.capabilities.supports_sequential_offload,
                "{} must advertise supports_sequential_offload",
                d.id
            );
        }
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Boogu's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Qwen3-VL mllm from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weight A/B is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/boogu-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let spec = missing_snapshot_spec(OffloadPolicy::Sequential);
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => unreachable!(),
        };
        let res = build_residency(&spec, &root)
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let spec = missing_snapshot_spec(OffloadPolicy::Resident);
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => unreachable!(),
        };
        let err = build_residency(&spec, &root)
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        // An eager-load failure (missing weights), not a policy/dispatch guard.
        assert!(
            res_is_load_failure(&err.to_string()),
            "expected an eager-load failure on the missing snapshot: {err}"
        );
    }

    /// The eager-load failure surfaces as a weights/file error, not one of the up-front guards.
    fn res_is_load_failure(msg: &str) -> bool {
        !msg.contains("single .safetensors file")
            && !msg.contains("precision override")
            && !msg.contains("not supported")
    }

    #[test]
    fn previous_evaluation_reports_f32_retained_estimate_bytes() {
        use mlx_gen::gen_core::sampling::{
            FlowModelSampling, Solver, SolverRunState, TimestepConvention,
        };
        use mlx_gen::gen_core::{EstimateSource, EvaluationKind, LowResStartSpec};
        use mlx_gen::sampler::MlxLatentOps;

        let req = GenerationRequest {
            prompt: "a fox in the forest".into(),
            width: 512,
            height: 512,
            seed: Some(42),
            steps: Some(4),
            sampler: Some("euler".into()),
            features: mlx_gen::gen_core::FeatureParams { low_res_start: Some(
                LowResStartSpec::new(2, Some(0.5))
                    .with_estimate_source(EstimateSource::PreviousEvaluation),
            ), ..Default::default() },
            ..Default::default()
        };
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/boogu".into()));
        let registry = crate::provider_registry().unwrap();
        let plan = registry
            .compile_render_plan(BOOGU_IMAGE_ID, &spec, &req)
            .unwrap()
            .unwrap();
        let (width, height) = plan.phase_output_size(0);
        let noise = crate::pipeline::init_noise(height, width, 42, 0).unwrap();
        let sigmas = plan.denoise_sigmas();
        let mut state = SolverRunState::new(Solver::Euler, sigmas);
        state
            .configure_low_res_start(plan.low_res_start().unwrap())
            .unwrap();
        let _ = Solver::Euler
            .boxed::<MlxLatentOps>()
            .sample_span(
                &MlxLatentOps,
                &FlowModelSampling::new(TimestepConvention::Sigma),
                &mut |x, _| Ok(x.clone()),
                noise,
                sigmas,
                42,
                plan.denoise_spans()[0].clone(),
                &mut state,
            )
            .unwrap();
        let retained = state.take_captured_primary_estimate(1).unwrap().latent;
        assert_eq!(
            retained.shape(),
            vec![1, 16, (height / 8) as i32, (width / 8) as i32]
        );
        assert_eq!(retained.dtype(), mlx_rs::Dtype::Float32);

        let estimate = low_res_start_report(&plan)
            .unwrap()
            .unwrap()
            .estimate_provenance;
        assert_eq!(estimate.evaluation_kind, EvaluationKind::GuidedPrimaryX0);
        assert_eq!(estimate.dtype, "f32");
        assert_eq!(
            estimate.retained_primary_estimate_bytes,
            retained.nbytes() as u64
        );
    }

    #[test]
    fn tier0_render_plan_and_low_res_start_conformance() {
        use mlx_gen::gen_core::{FeatureInapplicabilityCode, LowResStartSpec};

        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/boogu".into()));

        // Base T2I with low_res_start compiles a two-phase RenderPlan
        let mut t2i = req(1024, 1024);
        t2i.steps = Some(10);
        t2i.features.low_res_start = Some(LowResStartSpec::new(4, Some(0.5)));
        let plan = registry
            .compile_render_plan(BOOGU_IMAGE_ID, &spec, &t2i)
            .unwrap()
            .expect("boogu_image compiles a RenderPlan");
        assert_eq!(plan.denoise_spans(), vec![0..4, 4..10]);
        assert_eq!(plan.phase_output_size(0), (512, 512));
        assert_eq!(plan.phase_output_size(1), (1024, 1024));
        let lrs = plan.low_res_start().expect("low_res_start plan resolved");
        assert_eq!(lrs.k, 4);
        assert_eq!(lrs.executed_steps, 10);
        assert_eq!(lrs.low_width, 512);
        assert_eq!(lrs.low_height, 512);

        // Base single-reference img2img with low_res_start truncates schedule and compiles two phases
        let mut i2i = t2i.clone();
        i2i.conditioning = vec![Conditioning::Reference {
            image: img(512, 512),
            strength: Some(0.6),
        }];
        i2i.features.low_res_start = Some(LowResStartSpec::new(2, Some(0.5)));
        let i2i_plan = registry
            .compile_render_plan(BOOGU_IMAGE_ID, &spec, &i2i)
            .unwrap()
            .expect("boogu_image img2img compiles a RenderPlan");
        assert_eq!(i2i_plan.denoise_spans(), vec![0..2, 2..4]);
        assert_eq!(i2i_plan.phase_output_size(0), (512, 512));
        assert_eq!(i2i_plan.phase_output_size(1), (1024, 1024));

        // Edit compiles a single-phase RenderPlan without low_res_start, and refuses low_res_start with ConflictsWithPass0EditConditioning
        let edit_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..req(1024, 1024)
        };
        let edit_plan = registry
            .compile_render_plan(BOOGU_IMAGE_EDIT_ID, &spec, &edit_req)
            .unwrap()
            .expect("boogu_image_edit compiles a RenderPlan");
        assert_eq!(edit_plan.phase_output_size(0), (1024, 1024));

        let mut edit_lrs = edit_req.clone();
        edit_lrs.features.low_res_start = Some(LowResStartSpec::new(4, Some(0.5)));
        assert!(registry
            .compile_render_plan(BOOGU_IMAGE_EDIT_ID, &spec, &edit_lrs)
            .is_err());

        let base_app = registry
            .route_feature_applicability(BOOGU_IMAGE_ID)
            .expect("boogu_image applicability");
        assert!(base_app["low_res_start"].applicable);

        let edit_app = registry
            .route_feature_applicability(BOOGU_IMAGE_EDIT_ID)
            .expect("boogu_image_edit applicability");
        assert!(!edit_app["low_res_start"].applicable);
        assert!(edit_app["low_res_start"]
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0EditConditioning));

        // Turbo (ENG-5) refuses low_res_start with typed refusal
        let turbo_app = registry
            .route_feature_applicability(BOOGU_IMAGE_TURBO_ID)
            .expect("boogu_image_turbo applicability");
        assert!(!turbo_app["low_res_start"].applicable);
        assert!(turbo_app["low_res_start"]
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::MissingRenderPlan));
        assert!(validate_request(&descriptor_turbo(), &t2i).is_err());
    }
}
