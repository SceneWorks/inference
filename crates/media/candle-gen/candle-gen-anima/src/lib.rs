//! # candle-gen-anima
//!
//! The **Anima** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-anima` (epic 10512, sc-10525). Three variants share **one architecture** and differ only
//! in the DiT weights file:
//! - **`anima_base`** — the base model (30 steps, CFG 4.5),
//! - **`anima_aesthetic`** — the aesthetic fine-tune (30 steps, CFG 4.5),
//! - **`anima_turbo`** — the merged CFG-free few-step student (10 steps, CFG 1.0).
//!
//! ## Architecture (transcribed from the MLX port; no candle-transformers reference)
//! - **DiT** — the **Cosmos-Predict2** `CosmosTransformer3DModel` (28 layers, hidden 2048 = 16×128,
//!   patch `(1,2,2)`, adaLN-LoRA 256, 3-axis NTK RoPE `rope_scale (1,4,4)`, `concat_padding_mask` ⇒
//!   **17-channel** patch-embed input) — [`transformer::CosmosDiT`].
//! - **Text conditioner** — the **`AnimaTextConditioner`** (bundled under `{prefix}.llm_adapter.*`):
//!   `nn.Embedding(32128, 1024)` over T5 ids → 6 × [self-attn → cross-attn into Qwen3 states → GELU
//!   MLP] → out_proj + RMSNorm, right-padded to **512** — [`conditioner::AnimaTextConditioner`].
//! - **Text encoder** — **Qwen3-0.6B base** (`last_hidden_state`, GQA 16/8 handled with `repeat_kv`) —
//!   [`text_encoder::AnimaQwen3`].
//! - **VAE** — the **Qwen-Image** `AutoencoderKLQwenImage`, reusing [`vae::QwenVae`] (from
//!   `candle-gen-qwen-image`) via the original→diffusers key rename [`vae::convert_vae_key`].
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler` static `shift=3.0`, `sigmas = linspace(1, 1/N, N)`
//!   ([`pipeline::anima_sigmas`]); default solver the recommended `er_sde` (sc-10519), carried by the
//!   `441ecec` gen-core pin ([`pipeline::DEFAULT_SAMPLER`]).
//!
//! **`backend = "candle"`, `mac_only = false`** — this crate is what lets the manifest drop the
//! `macOnly: true` gate the MLX-only port carried (sc-10523 wires it worker-side).
//!
//! **Surface:** txt2img at the single-file dense checkpoint, with **LoRA/LoKr injection** (448 DiT + 60
//! conditioner targets folded at load, stacked + mixed, strict routing — [`adapters`]). Q4/Q8 candle
//! quant tiers are the counterpart of MLX sc-10517 (see the `quant` gap note in [`loader`]).

pub mod adapt;
pub mod adapters;
pub mod conditioner;
pub mod config;
pub mod loader;
pub mod memory_strategy;
pub mod nn;
pub mod pipeline;
// Per-step latent previews (epic 16948, sc-16953). Anima carries no fit of its own: it reuses the
// `candle-gen-qwen-image` QwenVae constants unchanged, because its VAE tensors are bit-identical to
// the file that fit was measured against. This module owns only the 5-D Cosmos → `[1, C, h, w]`
// layout adaptation.
pub mod preview;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod vae;

pub use conditioner::AnimaTextConditioner;
pub use config::{ConditionerConfig, DitConfig, Qwen3Config, Variant, RES_MULTIPLE};
pub use loader::{detect_dit_prefix, AnimaComponents};
pub use pipeline::{anima_sigmas, AnimaPipeline, GenOptions, DEFAULT_SAMPLER};
pub use text_encoder::AnimaQwen3;
pub use transformer::CosmosDiT;
pub use vae::{load_vae, QwenVae};

use std::sync::Arc;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant,
};

/// The candle quant tiers Anima advertises — Q4 + Q8 (the counterpart of MLX sc-10517). The DiT loads
/// packed (dequant-dense per step, CPU-capable); the conditioner / Qwen3 TE / VAE stay dense bf16.
const ANIMA_QUANTS: &[Quant] = &[Quant::Q4, Quant::Q8];

const MAX_COUNT: u32 = 8;
const RES_MIN: u32 = 512;
/// Above ~1920 px/side the Cosmos RoPE would index out of its trained range; `rope.rs` **rejects**
/// such a request rather than clamping. The shipped ceiling is 1536² (post-patch 96 < the 120-position
/// max_size), so the guard is unreachable via the normal path. See [`crate::rope`].
const RES_MAX: u32 = 1536;

/// Build the descriptor for a variant. Turbo is the merged CFG-free student (no guidance / negative
/// prompt); Base/Aesthetic run true classifier-free guidance.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let cfg_capable = variant.uses_cfg();
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::QWEN_KREA_Z16_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: variant.id(),
        family: "anima",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: cfg_capable,
            supports_guidance: cfg_capable,
            // LoRA/LoKr injection is wired (the candle counterpart of MLX sc-10521): every trained
            // adapter's 448 DiT + 60 conditioner targets fold at load, stacked + mixed, strict routing
            // (`adapters::apply_anima_adapters`). Weight-level fold, validated bit-exact on CPU.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow over the unified curated-sampler framework (epic 7114). The native default
            // (req.sampler == None) is the recommended er_sde solver; the full curated menu is advertised.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            // Q4 + Q8 (the candle counterpart of MLX sc-10517): the DiT packed-detects and runs the
            // dequant-dense forward (CPU-capable — NOT the CUDA-only int8 fast GEMM); conditioner /
            // Qwen3 TE / VAE stay dense bf16. A pre-packed tier is a real, loadable snapshot.
            supported_quants: ANIMA_QUANTS,
            requires_sigma_shift: true,
            // Per-step latent previews: wired by sc-16953 for all three variants at once — they share
            // one render lane and differ only in the DiT weights file — and advertised behind the
            // source-verified bidirectional guard in `candle-gen-catalog` (sc-16951), which derives
            // from this crate's shipped sources whether it actually emits.
            supports_preview: true,
            ..Default::default()
        },
    }
}

pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(Variant::Base)
}
pub fn descriptor_aesthetic() -> ModelDescriptor {
    descriptor_for(Variant::Aesthetic)
}
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::Turbo)
}

/// A loaded Anima generator: the cached descriptor + variant + lazily-built pipeline (mirrors the
/// candle-gen-qwen-image lazy component cache).
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    device: Device,
    /// LoRA/LoKr adapters to bake onto the DiT + conditioner at pipeline build (empty for the plain
    /// model). Captured at load; folded lazily when the pipeline is first assembled.
    adapters: Vec<gen_core::AdapterSpec>,
    /// The loaded numeric tier is an admission identity, not an inferred descriptor property.
    /// Retain the exact spec so generator-side admission cannot accept crossed-tier evidence.
    load_spec: LoadSpec,
    /// The request-scoped owner keeps the warm resident pipeline correct after a staged request:
    /// staging evicts it, and a subsequent resident request rebuilds the exact same variant.
    residency: candle_gen::Residency<AnimaTextPhase, AnimaHeavyPhase>,
    memory_contract: Option<gen_core::MemoryProviderContract>,
}

enum AnimaTextPhase {
    Resident(Arc<AnimaPipeline>),
    Staged(Box<loader::AnimaConditioningComponents>),
}

enum AnimaHeavyPhase {
    Resident(Arc<AnimaPipeline>),
    Staged(Box<loader::AnimaRenderComponents>),
}

struct AnimaMemoryScope {
    memory: Option<gen_core::GenerationMemory>,
    geometry: gen_core::MemoryGeometry,
    finished: bool,
}

impl AnimaMemoryScope {
    fn new(
        contract: &gen_core::MemoryProviderContract,
        context: &gen_core::MemoryRunContext,
    ) -> Self {
        Self {
            memory: contract.generation_memory(&context.selection),
            geometry: context.geometry,
            finished: false,
        }
    }

    fn active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(
                "anima memory request scope is already finished".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

impl gen_core::MemoryRequestScope for AnimaMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.active()?;
        let geometry = gen_core::MemoryGeometry {
            width: request.width,
            height: request.height,
            batch: request.count,
            frames: request.frames.unwrap_or(1),
            reference_count: request.image_reference_count(),
        };
        if geometry != self.geometry || !request.conditioning.is_empty() {
            return Err(gen_core::Error::Unsupported(
                "anima: request geometry or conditioning changed after memory admission".to_owned(),
            ));
        }
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }
    fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }
    fn configure_decode(
        &mut self,
        _: u32,
        _: u32,
        _: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.active()?;
        Err(gen_core::Error::Unsupported(
            "anima: bounded decode is not implemented".to_owned(),
        ))
    }
    fn configure_attention(&mut self, _: u32) -> gen_core::Result<()> {
        self.active()?;
        Err(gen_core::Error::Unsupported(
            "anima: bounded attention is not implemented".to_owned(),
        ))
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> gen_core::Result<()> {
        self.active()?;
        Err(gen_core::Error::Unsupported(
            "anima: transformer block streaming is not implemented".to_owned(),
        ))
    }
    fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.active()?;
        self.finished = true;
        Ok(())
    }
}

pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Base)
}
pub fn load_aesthetic(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Aesthetic)
}
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Turbo)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: candle Anima is txt2img only (no control / IP-adapter)"
        )));
    }
    // A Q4/Q8 tier is a **pre-packed** snapshot (the worker points `spec.weights` at it; the DiT
    // packed-detects it at load). LoRA/LoKr on a packed tier is now wired (sc-10640): the loader builds
    // the packed model and installs each adapter as a **forward-time residual** (`y = xW_packed +
    // scale·(xA)B`, epic 10043 prior art) rather than folding into the codes — so no rejection here. The
    // dense-checkpoint fold and the packed residual are both handled in `loader::AnimaComponents::load`
    // (a LoKr/LoHa on a packed tier still errors, in the loader — sc-10713).
    //
    // A requested Q4/Q8 tier MUST be an actually-packed checkpoint (u32 codes + `.scales`). Anima ships
    // no packed tier yet, and `lin()` packed-detects PER-TENSOR — so a `quantize = Q8` request against a
    // DENSE DiT would silently build bf16 and return success (a tier downgrade the caller never sees).
    // Assert the DiT is packed; otherwise reject naming the requested tier and what was found. Same
    // runtime-lie class as sc-10515 (advertising a tier the load can't honor).
    if let Some(q) = spec.quantize {
        if !loader::dit_is_packed(&spec.weights, variant).map_err(gen_core::Error::from)? {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: {q:?} tier requested but the DiT checkpoint is DENSE (no packed `.scales` \
                 tensors) — Anima ships no packed Q4/Q8 tier yet; load the dense tier (no quantize)"
            )));
        }
    }
    // LoRA/LoKr adapters (`spec.adapters`) are accepted — folded onto the DiT + conditioner when the
    // pipeline is assembled (`adapters::apply_anima_adapters`).
    let root = spec.weights.clone();
    let device = candle_gen::default_device().map_err(gen_core::Error::from)?;
    let resident_root = root.clone();
    let resident_device = device.clone();
    let resident_adapters = spec.adapters.clone();
    let text_root = root.clone();
    let text_device = device.clone();
    let heavy_root = root.clone();
    let heavy_device = device.clone();
    let staged_adapters = spec.adapters.clone();
    let residency = candle_gen::Residency::request_scoped_with_resident_cancelable(
        move |_| {
            let pipeline = AnimaPipeline::from_source(
                &resident_root,
                variant,
                &resident_device,
                &resident_adapters,
            )
            .map(Arc::new)?;
            Ok((
                AnimaTextPhase::Resident(pipeline.clone()),
                AnimaHeavyPhase::Resident(pipeline),
            ))
        },
        move |_, cancel| {
            candle_gen::check_cancel(cancel)?;
            if !staged_adapters.is_empty() {
                return Err(candle_gen::CandleError::Msg(
                    "anima: staged residency does not support adapter overlays".to_owned(),
                ));
            }
            loader::AnimaConditioningComponents::load(&text_root, variant, &text_device)
                .map(Box::new)
                .map(AnimaTextPhase::Staged)
        },
        move |_, _, cancel| {
            candle_gen::check_cancel(cancel)?;
            loader::AnimaRenderComponents::load(&heavy_root, variant, &heavy_device)
                .map(Box::new)
                .map(AnimaHeavyPhase::Staged)
        },
    );
    #[cfg(any(feature = "cuda", test))]
    let memory_contract = Some(memory_strategy::contract(id, spec)?);
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_contract = None;
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        device,
        adapters: spec.adapters.clone(),
        load_spec: spec.clone(),
        residency,
        memory_contract,
    }))
}

impl Generator for Anima {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        self.memory_contract
            .as_ref()
            .map_or(gen_core::MemorySafetyDecision::Accept, |contract| {
                memory_strategy::provider_safety_check(&self.load_spec, contract, context)
            })
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_contract.as_ref() else {
            return Ok(None);
        };
        if let gen_core::MemorySafetyDecision::Reject { reason } =
            self.memory_strategy_safety_check(context)
        {
            return Err(gen_core::Error::Unsupported(reason));
        }
        Ok(Some(Box::new(AnimaMemoryScope::new(contract, context))))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = if self.variant.uses_cfg() {
            req.guidance.unwrap_or(self.variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        // Shared batch frame (sc-7792): the `0..count` loop + per-image `image_seed(base_seed, n)`
        // derivation + `Vec` collect that every provider repeats. The model body stays hand-written in
        // the closure (captures `on_progress` + the borrowed pipeline).
        let stage_residency = req.memory.is_some_and(|memory| memory.stage_residency);
        if stage_residency && !self.adapters.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: staged residency refuses LoRA/LoKr overlays because their conditioner and DiT loads are one atomic artifact",
                self.descriptor.id
            )));
        }
        let images = candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let opts = GenOptions {
                width: req.width,
                height: req.height,
                steps,
                guidance,
                seed,
                sampler: req.sampler.clone(),
                // The caller's live preview sink (epic 16948, sc-16953). Cloning is an `Arc` bump;
                // an absent sink is inert and the denoise never projects anything.
                preview: req.preview.clone(),
            };
            self.residency.run_request_scoped(
                stage_residency,
                false,
                &req.cancel,
                false,
                on_progress,
                |text| match text {
                    AnimaTextPhase::Resident(pipeline) => {
                        let cond = pipeline.encode_prompt(&req.prompt)?;
                        let uncond = self
                            .variant
                            .uses_cfg()
                            .then(|| pipeline.encode_prompt(&negative))
                            .transpose()?;
                        Ok((cond, uncond))
                    }
                    AnimaTextPhase::Staged(components) => {
                        let cond =
                            AnimaPipeline::encode_staged(components, &self.device, &req.prompt)?;
                        let uncond = self
                            .variant
                            .uses_cfg()
                            .then(|| {
                                AnimaPipeline::encode_staged(components, &self.device, &negative)
                            })
                            .transpose()?;
                        Ok((cond, uncond))
                    }
                },
                |_| Ok(self.device.synchronize()?),
                |heavy, (cond, uncond), progress| match heavy {
                    AnimaHeavyPhase::Resident(pipeline) => AnimaPipeline::render_encoded(
                        pipeline.components(),
                        &self.device,
                        cond,
                        uncond,
                        &opts,
                        &req.cancel,
                        progress,
                    ),
                    AnimaHeavyPhase::Staged(components) => AnimaPipeline::render_encoded(
                        components.as_ref(),
                        &self.device,
                        cond,
                        uncond,
                        &opts,
                        &req.cancel,
                        progress,
                    ),
                },
            )
        })?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (testable without loaded weights): non-empty prompt, size a
/// multiple of 16, on top of the shared [`Capabilities::validate_request`] floor.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    // Reject an explicit `steps: Some(0)` loudly: `anima_sigmas` clamps `steps.max(1)`, so a 0 silently
    // becomes a single-step render rather than the fast typed error its sibling bespoke lanes give
    // (`reject_zero_steps`, sc-9016, F-032; swept here by sc-11182, F-102). A `None` legitimately falls
    // through to `variant.default_steps()`.
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders a single step of undenoised noise)"
        )));
    }
    desc.capabilities.validate_request(id, req)?;
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration of all three variants.
candle_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base
}
candle_gen::register_generators! {
    pub(crate) const AESTHETIC_REGISTRATION = descriptor_aesthetic => load_aesthetic
}
candle_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo
}

/// Add all Candle Anima providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(BASE_REGISTRATION)
        .register_generator(AESTHETIC_REGISTRATION)
        .register_generator(TURBO_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = register_memory_contract_surfaces(registry);
    registry
}

/// Register the weights-free contracts on non-CUDA catalog builds and the executable contracts on
/// CUDA builds.  The three ids deliberately receive independent registrations: they share code but
/// have distinct checkpoint, tier, mode, and calibration identities.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_memory_strategy(memory_strategy::BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(candle_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: candle_gen::gen_core::candle_memory_contract_surface_specs,
            provider_id: "anima_base",
            contract: |spec| memory_strategy::contract("anima_base", spec),
        })
        .register_memory_strategy(memory_strategy::AESTHETIC_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(candle_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: candle_gen::gen_core::candle_memory_contract_surface_specs,
            provider_id: "anima_aesthetic",
            contract: |spec| memory_strategy::contract("anima_aesthetic", spec),
        })
        .register_memory_strategy(memory_strategy::TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(
            candle_gen::gen_core::MemoryContractFixtureRegistration {
                surface_specs: candle_gen::gen_core::candle_memory_contract_surface_specs,
                provider_id: "anima_turbo",
                contract: |spec| memory_strategy::contract("anima_turbo", spec),
            },
        );
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR)
        .register_memory_behavior(AESTHETIC_MEMORY_BEHAVIOR)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR);
    registry
}

#[cfg(feature = "cuda")]
const BASE_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "anima_base",
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: memory_strategy::registered_begin_request,
    };

#[cfg(feature = "cuda")]
const AESTHETIC_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "anima_aesthetic",
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: memory_strategy::registered_begin_request,
    };

#[cfg(feature = "cuda")]
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "anima_turbo",
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: memory_strategy::registered_begin_request,
    };

/// Build the complete explicit Candle Anima provider catalog.
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

        assert_eq!(explicit, ["anima_base", "anima_aesthetic", "anima_turbo"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "an anime girl with silver hair, detailed".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn three_variants_registered_as_candle() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            let g = crate::provider_registry()
                .unwrap()
                .load(
                    id,
                    &LoadSpec::new(gen_core::WeightsSource::Dir("/nonexistent".into())),
                )
                .unwrap_or_else(|_| panic!("id {id} not registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "anima");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn descriptors_surface() {
        let b = descriptor_base();
        assert_eq!(b.id, "anima_base");
        assert_eq!(b.backend, "candle");
        assert_eq!(b.modality, Modality::Image);
        assert!(b.capabilities.supports_guidance);
        assert!(b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.requires_sigma_shift);
        // The candle port removes the Mac-only gate.
        assert!(!b.capabilities.mac_only);
        // LoRA/LoKr injection is wired; Q4/Q8 tiers are advertised (packed-detect, dequant-dense).
        assert!(b.capabilities.supports_lora && b.capabilities.supports_lokr);
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(b.capabilities.min_size, 512);
        assert_eq!(b.capabilities.max_size, 1536);
        // Per-step latent previews are wired for ALL THREE variants at once (sc-16953): they share one
        // `pipeline::AnimaPipeline::generate` render lane and differ only in the DiT weights file, so
        // there is no configuration in which one emits and another does not. Asserted per id rather
        // than once, because per id is the claim `candle-gen-catalog`'s bidirectional guard checks
        // against the sources.
        for descriptor in [
            descriptor_base(),
            descriptor_aesthetic(),
            descriptor_turbo(),
        ] {
            assert!(
                descriptor.capabilities.supports_preview,
                "{} must advertise per-step latent previews",
                descriptor.id
            );
        }
        // Turbo is the CFG-free merged student.
        let t = descriptor_turbo();
        assert!(!t.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_negative_prompt);
        // The default flow solver (er_sde on the 441ecec gen-core pin) is a real curated sampler.
        assert_eq!(pipeline::DEFAULT_SAMPLER, "er_sde");
        assert!(
            b.capabilities.samplers.contains(&pipeline::DEFAULT_SAMPLER),
            "er_sde advertised in the curated menu (441ecec gen-core pin carries the ErSde solver)"
        );
    }

    #[test]
    fn validate_rejects_bad_requests() {
        assert!(validate_request(&descriptor_base(), &GenerationRequest::default()).is_err()); // empty prompt
        assert!(validate_request(&descriptor_base(), &req(1000, 1024)).is_err()); // not mult of 16
        assert!(validate_request(&descriptor_base(), &req(256, 256)).is_err()); // below min
        assert!(validate_request(&descriptor_base(), &req(2048, 2048)).is_err()); // above max
                                                                                  // Explicit `steps: Some(0)` is rejected (sc-11182, F-102) — it would otherwise clamp to a
                                                                                  // silent 1-step render in `anima_sigmas`; `None` (the default) is fine.
        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..req(1024, 1024)
        };
        let err = validate_request(&descriptor_base(), &zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor_base(), &req(1536, 1536)).is_ok());

        // sc-12612: `RES_MULTIPLE` is the pinned stride SceneWorks ties every advertised Anima bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE scale) but
        // not RES_MULTIPLE (16) — 1000 = 125×8, in range [512, 1536] — is rejected with the stride
        // error, and an on-stride in-range size passes.
        assert_eq!(RES_MULTIPLE, 16);
        let off_stride = validate_request(&descriptor_base(), &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiple of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
    }

    /// Write a minimal **dense** DiT split_files layout (one anchor tensor, NO `.scales` codes) so the
    /// quant-guard can header-detect it as dense. Returns the split_files root.
    fn write_dense_split_files(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        use candle_gen::candle_core::{DType, Device, Tensor};
        let root = tmp.path().join("anima_quant_guard");
        let dm = root.join("diffusion_models");
        std::fs::create_dir_all(&dm).unwrap();
        let mut m = std::collections::HashMap::new();
        m.insert(
            "net.x_embedder.proj.1.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&m, dm.join(Variant::Base.dit_filename()))
            .unwrap();
        root
    }

    /// Write a minimal **packed** DiT split_files layout (an anchor tensor WITH a `.scales`/`.biases`
    /// sibling) so the quant-guard header-detects it as packed. Returns the split_files root.
    fn write_packed_split_files(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        use candle_gen::candle_core::{DType, Device, Tensor};
        let root = tmp.path().join("anima_packed_guard");
        let dm = root.join("diffusion_models");
        std::fs::create_dir_all(&dm).unwrap();
        let mut m = std::collections::HashMap::new();
        // The anchor `.weight` (u32 codes) + `.scales`/`.biases` — enough for the header-only packed
        // detect (`dit_path_is_packed` looks only for a `.scales` sibling).
        m.insert(
            "net.x_embedder.proj.1.weight".to_string(),
            Tensor::zeros((2, 2), DType::U32, &Device::Cpu).unwrap(),
        );
        m.insert(
            "net.x_embedder.proj.1.scales".to_string(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        m.insert(
            "net.x_embedder.proj.1.biases".to_string(),
            Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&m, dm.join(Variant::Base.dit_filename()))
            .unwrap();
        root
    }

    #[test]
    fn load_accepts_lora_and_packed_combo_but_rejects_quant_on_dense() {
        let tmp = tempfile::tempdir().unwrap();
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let root = write_dense_split_files(&tmp);
        let base = gen_core::WeightsSource::Dir(root.clone());
        let lora_spec = || {
            vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]
        };

        // A plain LoRA load SUCCEEDS (lazily built; the fold happens at first generate). Advertising the
        // capability and then rejecting at load would be a lie.
        assert!(load_base(&LoadSpec::new(base.clone()).with_adapters(lora_spec())).is_ok());

        // A Q4/Q8 request against a DENSE checkpoint must be REJECTED at load, not silently downgraded to
        // bf16 and returned Ok (the sc-10525 blocker: a tier the load can't honor). The message names the
        // requested tier and that the DiT is dense.
        for q in [Quant::Q4, Quant::Q8] {
            let err = load_base(&LoadSpec::new(base.clone()).with_quant(q))
                .err()
                .expect("Q-tier on a dense DiT must error");
            let gen_core::Error::Unsupported(msg) = &err else {
                panic!("expected Unsupported, got {err:?}");
            };
            assert!(
                msg.contains(&format!("{q:?}")) && msg.contains("DENSE"),
                "message must name the tier + dense: {msg}"
            );
        }

        // Q8 + LoRA against a DENSE checkpoint still errors — but for the **tier-mismatch** reason (Q8 on
        // a dense DiT), NOT a packed+adapter combo rejection. That combo rejection was REMOVED in sc-10640
        // (the combo is now wired via forward-time residuals); the guard that fires here is the same dense
        // tier-mismatch as the no-adapter case above, so the message names DENSE.
        let dense_combo = LoadSpec::new(base.clone())
            .with_quant(Quant::Q8)
            .with_adapters(lora_spec());
        let gen_core::Error::Unsupported(msg) = load_base(&dense_combo).err().expect("err") else {
            panic!("expected Unsupported");
        };
        assert!(
            msg.contains("DENSE"),
            "dense-tier mismatch (Q8 on dense), not a packed-combo rejection: {msg}"
        );

        // sc-10640: Q4/Q8 + LoRA on a **packed** checkpoint is now ACCEPTED at load (built lazily; the
        // residual install runs at first generate). This is exactly the combo that used to be rejected.
        let packed_root = write_packed_split_files(&tmp);
        let packed = gen_core::WeightsSource::Dir(packed_root.clone());
        assert!(
            load_base(
                &LoadSpec::new(packed)
                    .with_quant(Quant::Q8)
                    .with_adapters(lora_spec())
            )
            .is_ok(),
            "packed tier + LoRA must be accepted at load (sc-10640) — no combo rejection"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&packed_root);
    }

    /// The pipeline handle must be `Send + Sync` so the denoise can run outside the cache lock
    /// (sc-10608). `Anima` is `Sync` (required by the `Generator` bound), its cache is
    /// `Mutex<Option<Arc<AnimaPipeline>>>`, and `Mutex<Option<Arc<T>>>` is only `Sync` when
    /// `T: Send + Sync`. A change that made `AnimaPipeline` non-`Sync` would break `Anima: Sync` — this
    /// static assertion fails to compile first, with a clear pointer to the reason.
    #[test]
    fn pipeline_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<AnimaPipeline>>();
        assert_send_sync::<Anima>();
    }

    /// Exercise Anima's actual production lifecycle owner, rather than the removed `cached` stand-in.
    /// The sequence proves warm → staged → warm rebuilding, text-before-heavy ordering, and that
    /// cancellation/error exits leave the owner usable for the next admitted request.
    #[test]
    fn residency_warm_staged_warm_and_failed_requests_leave_no_poisoned_cache() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let events = Arc::new(Mutex::new(Vec::new()));
        let warm_loads = Arc::new(AtomicUsize::new(0));
        let text_loads = Arc::new(AtomicUsize::new(0));
        let heavy_loads = Arc::new(AtomicUsize::new(0));
        let residency = candle_gen::Residency::request_scoped_with_resident_cancelable(
            {
                let events = Arc::clone(&events);
                let warm_loads = Arc::clone(&warm_loads);
                move |_| {
                    warm_loads.fetch_add(1, Ordering::SeqCst);
                    events.lock().unwrap().push("warm-load");
                    Ok::<_, candle_gen::CandleError>((10u8, 20u16))
                }
            },
            {
                let events = Arc::clone(&events);
                let text_loads = Arc::clone(&text_loads);
                move |_, cancel| {
                    candle_gen::check_cancel(cancel)?;
                    text_loads.fetch_add(1, Ordering::SeqCst);
                    events.lock().unwrap().push("text-load");
                    Ok::<_, candle_gen::CandleError>(30u8)
                }
            },
            {
                let events = Arc::clone(&events);
                let heavy_loads = Arc::clone(&heavy_loads);
                move |_, _, cancel| {
                    candle_gen::check_cancel(cancel)?;
                    heavy_loads.fetch_add(1, Ordering::SeqCst);
                    events.lock().unwrap().push("heavy-load");
                    Ok::<_, candle_gen::CandleError>(40u16)
                }
            },
        );
        let run = |staged: bool, fail_encode: bool, cancel: &gen_core::CancelFlag| {
            residency.run_request_scoped(
                staged,
                false,
                cancel,
                false,
                &mut |_| {},
                |text| {
                    events.lock().unwrap().push("encode");
                    if fail_encode {
                        Err(candle_gen::CandleError::Msg(
                            "synthetic encode failure".to_owned(),
                        ))
                    } else {
                        Ok(*text)
                    }
                },
                |_| {
                    events.lock().unwrap().push("synchronize");
                    Ok(())
                },
                |heavy, text, _| {
                    events.lock().unwrap().push("render");
                    Ok::<_, candle_gen::CandleError>(u32::from(*heavy) + u32::from(text))
                },
            )
        };

        let live = gen_core::CancelFlag::new();
        assert_eq!(run(false, false, &live).unwrap(), 30);
        assert_eq!(run(true, false, &live).unwrap(), 70);
        assert_eq!(run(false, false, &live).unwrap(), 30);
        assert_eq!(
            warm_loads.load(Ordering::SeqCst),
            2,
            "staged evicts and the next warm request rebuilds"
        );
        assert_eq!(text_loads.load(Ordering::SeqCst), 1);
        assert_eq!(heavy_loads.load(Ordering::SeqCst), 1);
        assert_eq!(
            *events.lock().unwrap(),
            [
                "warm-load",
                "encode",
                "render",
                "text-load",
                "encode",
                "synchronize",
                "heavy-load",
                "render",
                "warm-load",
                "encode",
                "render"
            ],
            "staged request must encode and synchronize before heavy materialization"
        );

        let cancelled = gen_core::CancelFlag::new();
        cancelled.cancel();
        assert!(run(true, false, &cancelled).is_err());
        assert!(
            run(true, true, &live).is_err(),
            "error path must release the lifecycle owner"
        );
        assert_eq!(
            run(false, false, &live).unwrap(),
            30,
            "warm request recovers after cancellation/error"
        );
    }
}
