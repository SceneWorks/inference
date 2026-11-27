//! # candle-gen-chroma
//!
//! The **Chroma** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-chroma`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! explicitly registers all three Chroma variants in its family catalog.
//!
//! **txt2img (sc-5484):** Chroma is a FLUX.1-schnell-derived DiT — the MMDiT skeleton with a
//! distilled-guidance **Approximator** replacing FLUX's modulation stack, **T5-XXL-only**
//! conditioning, and **true CFG** (a real negative prompt). `pipeline` adapts it through the
//! contract: T5-XXL encode → the Chroma DiT (flow-match Euler, static-shift / beta sigmas) → the
//! FLUX 16-ch AutoencoderKL, emitting `Progress` and honoring `req.cancel`, with **deterministic
//! CPU-seeded noise** (sc-3673) so output is launch-portable per seed. The three variants:
//!
//! - **`chroma1_hd`** — high-detail full-CFG (28 steps, true_cfg 4.0, sigma shift 3.0).
//! - **`chroma1_base`** — base full-CFG (28 steps, true_cfg 4.0, beta-spaced sigmas).
//! - **`chroma1_flash`** — few-step distilled (8 steps, true_cfg 1.0 → single forward).
//!
//! The descriptors advertise **only** the wired txt2img surface (true-CFG + negative prompt, but NOT
//! LoRA/LoKr, quantization, or ControlNet/IP-adapter) — so the worker routes the rest to the Python
//! fallback rather than the candle backend silently dropping a control (the false-capability trap,
//! exactly as the SDXL / FLUX / Z-Image slices did). `backend` is `"candle"` and `mac_only` is `false`.

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_gen::candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_gen::candle_core::Tensor, ()>::Err(()));
};

mod beta;
mod config;
mod memory_strategy;
mod pipeline;
// Per-step latent previews (epic 16948, sc-16956). A re-export of `candle_gen_flux::preview`: Chroma
// packs, unpacks and decodes the same 16-channel `AutoencoderKL` bytes as FLUX.1, so it shares that
// crate's committed fit rather than restating it. See the module docs for the tensor-byte provenance.
pub mod preview;
mod quant;
mod rope;
mod text;
mod transformer;
mod vae;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    PidWeights, Progress, WeightsSource,
};

pub use config::{ChromaVariant, CHROMA1_BASE_ID, CHROMA1_FLASH_ID, CHROMA1_HD_ID, SIZE_MULTIPLE};

use pipeline::{Components, Pipeline};

/// Dense Chroma components that retain their checkpoint-native width.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeComponent {
    T5,
    Vae,
}

/// Chroma publishes its dense T5-XXL and FLUX.1 VAE tensors as BF16. Keeping the selections explicit
/// makes either loader's precision policy independently testable.
pub(crate) fn native_component_dtype(component: NativeComponent) -> DType {
    match component {
        NativeComponent::T5 => DType::BF16,
        NativeComponent::Vae => DType::BF16,
    }
}

/// A loaded candle Chroma generator (one per variant). Loading is **lazy**: `load` does no file I/O,
/// and the heavy components (T5 + DiT + VAE) are built on the first [`generate`](Generator::generate)
/// call and then cached.
pub struct ChromaGenerator {
    variant: ChromaVariant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Components>>,
    lifecycle: Mutex<()>,
    memory_contract: gen_core::MemoryProviderContract,
    load_receipt: Option<memory_strategy::ChromaLoadReceipt>,
    memory_admission: memory_strategy::AdmissionRegistry,
    load_binding: gen_core::RenderLoadBinding,
    loaded_spec: LoadSpec,
}

impl ChromaGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

fn evict_warm_for_staged<T>(slot: &Mutex<Option<T>>) -> Option<T> {
    candle_gen::lock_recover(slot).take()
}

fn release_warm_after_synchronize<T>(device: &Device, component: T) -> gen_core::Result<()> {
    match device.synchronize() {
        Ok(()) => {
            drop(component);
            Ok(())
        }
        Err(error) => {
            // Releasing storage while queued work may still reference it is unsafe. A failed device
            // synchronization is already terminal for the generator, so retain the allocation.
            std::mem::forget(component);
            Err(gen_core::Error::backend(error))
        }
    }
}

impl Generator for ChromaGenerator {
    fn generate_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let plan = compile_trajectory_for(self.variant, req)?.bind_load(&self.loaded_spec);
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            candle_gen::execution::unobserved_adapters(
                self.adapters.len(),
                "candle-gen-chroma/src/lib.rs",
            );
        })
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        if let Some(receipt) = &self.load_receipt {
            if let Err(error) = receipt.ensure_unchanged() {
                self.memory_admission.clear_approval();
                return gen_core::MemorySafetyDecision::rejected(error);
            }
        }
        if context.use_pid
            && self
                .load_receipt
                .as_ref()
                .is_none_or(|receipt| receipt.pid.is_none())
        {
            self.memory_admission.clear_approval();
            return gen_core::MemorySafetyDecision::Reject {
                reason: format!(
                    "{}: PiD was requested without the loaded PiD artifact receipt",
                    self.descriptor.id
                ),
            };
        }
        match memory_strategy::safety_check(
            self.descriptor.id,
            &self.memory_contract,
            context,
            self.load_receipt.as_ref().and_then(|receipt| receipt.tier),
        ) {
            gen_core::MemorySafetyDecision::Accept => {
                gen_core::MemorySafetyDecision::from_validation(
                    self.memory_admission.approve(context),
                )
            }
            reject => {
                self.memory_admission.clear_approval();
                reject
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        memory_strategy::validate_context(
            self.descriptor.id,
            &self.memory_contract,
            context,
            self.load_receipt.as_ref().and_then(|receipt| receipt.tier),
        )?;
        Ok(Some(Box::new(
            memory_strategy::ChromaMemoryScope::new_bound(
                self.descriptor.id,
                self.device.clone(),
                &self.memory_contract,
                context,
                self.memory_admission.clone(),
            )?,
        )))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request_for_variant(self.variant, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{}: generation completed without output",
                    self.descriptor.id
                ))
            })
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        let plan = compile_trajectory_for(self.variant, req)?.bind_load(&self.loaded_spec);
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&self.load_binding)?;
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume_for_generate(req)?;
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        let memory = req.memory.unwrap_or_default();
        candle_gen::execution::policy("chroma/lib/memory", format!("{memory:?}").into());
        if memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode, attention, and transformer residency are Missing",
                self.descriptor.id
            )));
        }
        let pipe = Pipeline::load(
            self.variant,
            &self.root,
            &self.device,
            self.pid_spec.clone(),
            self.adapters.clone(),
        );
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let images = if memory.stage_residency {
            if let Some(warm) = evict_warm_for_staged(&self.components) {
                release_warm_after_synchronize(&self.device, warm)?;
            } else {
                self.device
                    .synchronize()
                    .map_err(gen_core::Error::backend)?;
            }
            pipe.render_staged_planned(req, &run, on_progress)?
        } else {
            let components = self.components(&pipe)?;
            pipe.render_planned(req, &run, &components, on_progress)?
        };
        let mut report = gen_core::GenerationReport::from_output(GenerationOutput::Images(images));
        report.low_res_start = plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
        Ok(report)
    }
}

fn validate_request_for_variant(
    variant: ChromaVariant,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let descriptor = variant.descriptor();
    descriptor
        .capabilities
        .validate_request(variant.id(), req)?;
    let id = variant.id();
    if req.source_image.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: source_image is not supported on this route"
        )));
    }
    if req.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
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
    let memory = req.memory.unwrap_or_default();
    if memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: bounded decode, attention, and transformer residency are Missing"
        )));
    }
    if let Some(low_res) = &req.features.low_res_start {
        let sigmas = pipeline::resolved_sigmas(variant, req)?;
        low_res.resolve_for_descriptor(
            &descriptor,
            req,
            sigmas.len().saturating_sub(1),
            SIZE_MULTIPLE,
        )?;
    }
    Ok(())
}

/// Compile the weights-free Chroma trajectory for `variant` and `req`.
pub fn compile_trajectory_for(
    variant: ChromaVariant,
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    validate_request_for_variant(variant, req)?;
    let sigmas = pipeline::resolved_sigmas(variant, req)?;
    let executed_steps = sigmas.len().saturating_sub(1);
    let guidance = req.true_cfg.unwrap_or_else(|| variant.default_true_cfg());
    gen_core::RenderPlan::base_image(
        &variant.descriptor(),
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

fn validate_load_spec(variant: ChromaVariant, spec: &LoadSpec) -> gen_core::Result<PathBuf> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a Chroma diffusers snapshot directory (tokenizer/, text_encoder/, \
                 transformer/, vae/), not a single .safetensors file"
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

fn compile_variant_plan(
    variant: ChromaVariant,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_load_spec(variant, spec)?;
    if req.use_pid() && spec.pid.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)",
            variant.id()
        )));
    }
    compile_trajectory_for(variant, req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

/// Descriptors (registry introspection — no weights).
pub fn descriptor_hd() -> ModelDescriptor {
    ChromaVariant::Hd.descriptor()
}
pub fn descriptor_base() -> ModelDescriptor {
    ChromaVariant::Base.descriptor()
}
pub fn descriptor_flash() -> ModelDescriptor {
    ChromaVariant::Flash.descriptor()
}

/// Construct a lazy candle Chroma generator for `variant`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a Chroma diffusers snapshot (`tokenizer/`, `text_encoder/`,
/// `transformer/`, `vae/`). Ordered LoRA/LoKr adapters and the optional PiD decoder are admitted and
/// receipt-pinned; control/IP-adapter overlays and on-the-fly quantization remain unsupported.
fn load_variant(variant: ChromaVariant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = validate_load_spec(variant, spec)?;
    let device = candle_gen::default_device()?;
    let (load_receipt, memory_contract) =
        match memory_strategy::ChromaLoadReceipt::capture(id, spec) {
            Ok(receipt) => {
                let contract = memory_strategy::contract_from_receipt(id, spec, &receipt);
                (Some(receipt), contract)
            }
            #[cfg(test)]
            Err(_) if !root.exists() => {
                let mut fixture_spec = spec.clone();
                fixture_spec.resolved_route = Some(id.to_owned());
                (
                    None,
                    memory_strategy::uncalibrated_contract(id, &fixture_spec)?,
                )
            }
            Err(error) => return Err(error),
        };
    Ok(Box::new(ChromaGenerator {
        variant,
        descriptor: variant.descriptor(),
        root,
        device,
        pid_spec: spec.pid.clone(),
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        memory_contract,
        load_receipt,
        memory_admission: memory_strategy::AdmissionRegistry::new(id),
        load_binding: gen_core::RenderLoadBinding::from_spec(spec),
        loaded_spec: spec.clone(),
    }))
}

pub fn load_hd(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Hd, spec)
}
pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Base, spec)
}
pub fn load_flash(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(ChromaVariant::Flash, spec)
}

/// Primary typed route declaration for a Candle Chroma variant (`SW-6979`, `REG-1`).
pub(crate) fn route_for(variant: ChromaVariant) -> gen_core::RouteRecord {
    let bindings = match variant {
        ChromaVariant::Hd => gen_core::RouteBindings::generator(descriptor_hd, load_hd)
            .with_render_plan(|spec, req| compile_variant_plan(ChromaVariant::Hd, spec, req))
            .with_memory_registration(HD_MEMORY_REGISTRATION)
            .with_memory_fixture(
                gen_core::candle_memory_contract_surface_specs,
                hd_weights_free_contract,
            )
            .with_memory_surface_resolver(hd_surface_contract)
            .with_memory_behavior_registration(HD_MEMORY_BEHAVIOR),
        ChromaVariant::Base => gen_core::RouteBindings::generator(descriptor_base, load_base)
            .with_render_plan(|spec, req| compile_variant_plan(ChromaVariant::Base, spec, req))
            .with_memory_registration(BASE_MEMORY_REGISTRATION)
            .with_memory_fixture(
                gen_core::candle_memory_contract_surface_specs,
                base_weights_free_contract,
            )
            .with_memory_surface_resolver(base_surface_contract)
            .with_memory_behavior_registration(BASE_MEMORY_BEHAVIOR),
        ChromaVariant::Flash => gen_core::RouteBindings::generator(descriptor_flash, load_flash)
            .with_render_plan(|spec, req| compile_variant_plan(ChromaVariant::Flash, spec, req))
            .with_memory_registration(FLASH_MEMORY_REGISTRATION)
            .with_memory_fixture(
                gen_core::candle_memory_contract_surface_specs,
                flash_weights_free_contract,
            )
            .with_memory_surface_resolver(flash_surface_contract)
            .with_memory_behavior_registration(FLASH_MEMORY_BEHAVIOR),
    };

    gen_core::RouteRecord {
        route_id: variant.id(),
        family: "chroma",
        backend: "candle",
        modality: gen_core::Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new("t5_xxl", Some(512))],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: vec![],
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123) plus the legacy
                // aliases (`flow_match` / `linear`), which fall back to euler / the native
                // per-variant schedule (N3) so a request the worker builds for either backend still
                // validates.
                samplers: candle_gen::menu_with_aliases(
                    candle_gen::curated_sampler_names(),
                    &[config::DEFAULT_SAMPLER],
                ),
                schedulers: candle_gen::menu_with_aliases(
                    candle_gen::curated_scheduler_names(),
                    &["linear"],
                ),
                guidance_methods: vec![],
                supports_guidance: false,
                supports_true_cfg: true,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("chroma_mmdit").with_low_res_start(),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&candle_gen::gen_core::FLUX1_LATENT_SPACE),
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
                        ["t5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["chroma_mmdit"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["flux1_ae"],
                        false,
                    ),
                ],
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
            guidance: None,
            sampler: Some("euler"),
            schedule: None,
            shift: (!variant.use_beta_sigmas()).then_some(variant.sigma_shift()),
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings,
    }
}

/// Add all Candle Chroma providers to an explicit media registry builder via one-call
/// [`ProviderRegistryBuilder::register_route`](candle_gen::gen_core::ProviderRegistryBuilder::register_route)
/// (`SW-6979`, `REG-1`).
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(route_for(ChromaVariant::Hd))
        .register_route(route_for(ChromaVariant::Base))
        .register_route(route_for(ChromaVariant::Flash))
}

fn hd_memory_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(CHROMA1_HD_ID, spec)
}
fn base_memory_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(CHROMA1_BASE_ID, spec)
}
fn flash_memory_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(CHROMA1_FLASH_ID, spec)
}

fn hd_weights_free_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_contract(CHROMA1_HD_ID, spec)
}
fn base_weights_free_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_contract(CHROMA1_BASE_ID, spec)
}
fn flash_weights_free_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_contract(CHROMA1_FLASH_ID, spec)
}

fn hd_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(CHROMA1_HD_ID, surface)
}
fn base_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(CHROMA1_BASE_ID, surface)
}
fn flash_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(CHROMA1_FLASH_ID, surface)
}

const HD_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: CHROMA1_HD_ID,
    contract: hd_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};
const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: CHROMA1_BASE_ID,
    contract: base_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};
const FLASH_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: CHROMA1_FLASH_ID,
    contract: flash_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

macro_rules! memory_behavior {
    ($name:ident, $id:expr) => {
        const $name: gen_core::MemoryBehaviorRegistration = gen_core::MemoryBehaviorRegistration {
            provider_id: $id,
            valid_fixtures: memory_strategy::registered_valid_fixture,
            begin_request: |spec, contract, context| {
                memory_strategy::registered_begin_request($id, spec, contract, context)
            },
        };
    };
}

memory_behavior!(HD_MEMORY_BEHAVIOR, CHROMA1_HD_ID);
memory_behavior!(BASE_MEMORY_BEHAVIOR, CHROMA1_BASE_ID);
memory_behavior!(FLASH_MEMORY_BEHAVIOR, CHROMA1_FLASH_ID);

/// Register Chroma's exhaustive weights-free q4/q8/bf16 contract surfaces on every platform.
pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(HD_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: CHROMA1_HD_ID,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: hd_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: CHROMA1_HD_ID,
                contract: hd_surface_contract,
            },
        )
        .register_memory_strategy(BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: CHROMA1_BASE_ID,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: base_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: CHROMA1_BASE_ID,
                contract: base_surface_contract,
            },
        )
        .register_memory_strategy(FLASH_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: CHROMA1_FLASH_ID,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: flash_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: CHROMA1_FLASH_ID,
                contract: flash_surface_contract,
            },
        )
}

/// Build the complete explicit Candle Chroma provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod sw6979_route_conformance_tests {
    #[test]
    fn sw6979_route_records_pass_tier0_conformance() {
        let registry = super::provider_registry().unwrap();
        assert_eq!(registry.route_records().len(), 3);
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
    }
}

#[cfg(test)]
mod explicit_registry_tests {
    use candle_gen::candle_core::DType;

    #[test]
    fn dense_t5_and_vae_stay_at_their_native_bf16_width() {
        assert_eq!(
            super::native_component_dtype(super::NativeComponent::T5),
            DType::BF16
        );
        assert_eq!(
            super::native_component_dtype(super::NativeComponent::Vae),
            DType::BF16
        );
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["chroma1_hd", "chroma1_base", "chroma1_flash"]);
    }

    /// The registry-level memory lifecycle seams must be reachable on a build with no CUDA
    /// feature: building the provider catalog is contract-only (no device, no weights), so
    /// `register_providers` publishes the memory-strategy, weights-free contract-fixture and
    /// memory-behavior rows on every platform. Gating these behind `cuda` left registry
    /// lifecycle conformance running on no CPU CI configuration at all.
    #[test]
    fn register_providers_publishes_memory_lifecycle_seams_without_cuda() {
        let registry = super::provider_registry().unwrap();

        let strategies: Vec<&str> = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        let fixtures: Vec<&str> = registry
            .memory_contract_fixture_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        let behaviors: Vec<&str> = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect();

        assert_eq!(strategies, ["chroma1_hd", "chroma1_base", "chroma1_flash"]);
        assert_eq!(fixtures, ["chroma1_hd", "chroma1_base", "chroma1_flash"]);
        assert_eq!(behaviors, ["chroma1_hd", "chroma1_base", "chroma1_flash"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, Image, Modality};

    #[test]
    fn staged_warm_eviction_never_repopulates_the_generator_cache() {
        let cache = Mutex::new(Some(7_u8));
        assert_eq!(evict_warm_for_staged(&cache), Some(7));
        assert!(candle_gen::lock_recover(&cache).is_none());
        assert_eq!(evict_warm_for_staged(&cache), None);
    }

    #[test]
    fn all_three_variants_register_and_resolve_as_candle() {
        for id in [CHROMA1_HD_ID, CHROMA1_BASE_ID, CHROMA1_FLASH_ID] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("candle {id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "chroma");
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Image);
        }
    }

    #[test]
    fn validate_accepts_true_cfg_and_negative_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let hd = crate::provider_registry()
            .unwrap()
            .load(CHROMA1_HD_ID, &spec)
            .unwrap();

        // True CFG + negative prompt are advertised → accepted.
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            negative_prompt: Some("blurry".into()),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(hd.validate(&ok).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(3.5), // distilled guidance not advertised
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
            assert!(hd.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Chroma
        // bucket to. Pin the value and mutation-check that a size which is a multiple of 8 but not
        // SIZE_MULTIPLE (16) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = hd
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
        assert!(hd
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    #[test]
    fn load_accepts_adapters_and_rejects_unwired_quant_and_single_file() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        let temp = tempfile::tempdir().unwrap();
        for (route, load) in memory_strategy::ROUTES
            .iter()
            .zip([load_hd, load_base, load_flash])
        {
            // `/snap` exists on Ubuntu hosts, which accidentally turns this weights-free loader
            // fixture into a physical-receipt probe. Use a guaranteed-missing path that still
            // carries the exact repository/revision/tier and resolved provider identity.
            let root = temp
                .path()
                .join(format!("models--SceneWorks--{}", route.repository))
                .join("snapshots")
                .join(route.revision)
                .join("bf16");
            let lora = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_resolved_route(route.provider)
                .with_adapters(vec![AdapterSpec::new(
                    "/lora.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                )]);
            assert!(load(&lora).is_ok());

            let quant = LoadSpec::new(WeightsSource::Dir(root))
                .with_resolved_route(route.provider)
                .with_quant(Quant::Q8);
            assert!(matches!(
                load(&quant).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));

            let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()))
                .with_resolved_route(route.provider);
            let err = load(&single).err().expect("err").to_string();
            assert!(err.contains("snapshot directory"), "got: {err}");
        }
    }

    /// A pre-cancelled request aborts before any forward with the typed `Canceled` (the cancellation
    /// contract — sc-4481). The per-image cancel check runs before `denoise`, so loading lazily means
    /// this needs no weights... except `generate` loads components first. Kept as a descriptor-level
    /// check instead: `generate` on a nonexistent snapshot errors (not a panic).
    #[test]
    fn descriptors_are_image_modality() {
        assert_eq!(descriptor_hd().modality, Modality::Image);
        assert_eq!(descriptor_base().modality, Modality::Image);
        assert_eq!(descriptor_flash().modality, Modality::Image);
    }

    #[test]
    fn sw6945_candle_chroma_render_plan_and_low_res_start_conformance() {
        use candle_gen::gen_core::{GenerationRequest, LoadSpec, LowResStartSpec, WeightsSource};

        let registry = provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
            assert!(all_features[id]["low_res_start"].applicable);
            let req = GenerationRequest {
                prompt: "a blue lamp".into(),
                width: 512,
                height: 512,
                steps: Some(6),
                seed: Some(77),
                features: candle_gen::gen_core::FeatureParams { low_res_start: Some(LowResStartSpec::new(3, Some(0.5))), ..Default::default() },
                ..Default::default()
            };
            let plan = registry
                .compile_render_plan(id, &spec, &req)
                .unwrap()
                .expect("chroma render plan");
            assert_eq!(plan.denoise_spans(), vec![0..3, 3..6]);
            assert_eq!(plan.phase_output_size(0), (256, 256));
            assert_eq!(plan.phase_output_size(1), (512, 512));
        }
    }
}
