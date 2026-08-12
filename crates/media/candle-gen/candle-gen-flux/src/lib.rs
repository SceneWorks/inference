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

// Packed-tier (MLX diffusers-layout q4/q8) load path (sc-9407, sc-9089 umbrella). `quant` is the thin
// per-crate dense-or-packed enum delegating to the shared `candle_gen::quant` packed-load module
// (sc-9086); `packed_dit`/`packed_te` are minimal vendored diffusers-layout FLUX.1 DiT + CLIP-L/T5-XXL
// encoders that build every projection through it, so q4/q8 load straight from the packed parts (no
// dense bf16 staging). The dense BFL snapshot path (stock `candle-transformers`) is unchanged.
mod packed_dit;
// `packed_te` is `pub` so the Mochi provider (`candle-gen-mochi`) can reuse the vendored T5-XXL encoder
// stack (`PackedT5Encoder` / `T5Config::xxl`) — the same google/t5-v1.1-xxl geometry Mochi conditions
// on — through its masked-forward path (`forward_masked`, added for Mochi's padded/masked encode).
pub mod packed_te;
mod quant;

// XLabs FLUX IP-Adapter (sc-5872, epic 5480) — reference-image (identity) conditioning. `ip_dit` is the
// forked FLUX DiT carrying the per-double-block decoupled-cross-attn seam (the stock candle-transformers
// `Flux` has none); `ip_adapter` is the XLabs projector + K/V weights; `ip_image_encoder` is the pooled
// CLIP-ViT-L tower; `ip_provider` composes them into the bespoke reference stream the worker drives
// directly (not gen-core-registered — the `flux1_*` descriptors stay txt2img-only).
pub mod ip_adapter;
mod ip_dit;
pub mod ip_image_encoder;
pub mod ip_provider;
pub use ip_provider::{IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest, DEFAULT_IP_SCALE};

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
    Flux1ControlPaths, Flux1ControlRequest, Flux1DevControl, DEFAULT_CONTROL_SCALE,
};

// The vendored FLUX DiT + its post-block image-stream injector seam, re-exported for the PuLID-FLUX
// provider (`candle-gen-pulid`, sc-5492), which composes the FLUX backbone with the EVA-CLIP tower +
// IDFormer + the 20 PerceiverAttentionCA modules driven through [`DitImageInjector`]. `Config` is the
// candle-transformers FLUX config the fork reuses (so it cannot drift on hyperparameters).
pub use ip_dit::{Config as FluxConfig, DitImageInjector, IpFlux};
// FLUX backbone helpers shared with the PuLID provider so the two never drift on the parity-critical
// tokenization / VAE decode / config (the candle twin of `mlx-gen-flux`'s shared `Flux1` surface). The
// IP-Adapter provider reaches these as `pub(crate)`; PuLID is a separate crate, hence `pub`.
pub use pipeline::{
    ae_config, clip_config, decode_latents, encode_text, flux_config, FluxTokenizers,
};
// The FLUX dev flow-match time-shift constants + `flow_mu` linear map — one home (sc-11249 / F-140).
// The PuLID reference stream (`candle-gen-pulid`) shares these exact parity-critical schedule pieces
// rather than maintaining a third copy; the IP-Adapter provider reaches them in-crate.
pub use pipeline::{flow_mu, BASE_SHIFT, MAX_SHIFT};

// The tier-detecting reference backbone (sc-10103, epic 9083) — the candle twin of
// `mlx_gen_flux::load_flux1`. Reuses the txt2img `Pipeline`'s tier detect-and-load so the reference
// lanes (`candle-gen-pulid`, the FLUX IP-adapter) consume the SAME `SceneWorks/flux1-dev-mlx`
// q4/q8/bf16 turnkey tiers the base generator does, driving the post-block `DitImageInjector` seam on
// either the BFL `IpFlux` or the diffusers `PackedFluxDit`.
mod ref_backbone;
pub use ref_backbone::{FluxRefBackbone, FluxRefHeavy};

/// FLUX XLabs IP-Adapter real-weight GPU validation (sc-5872) — env-driven, `#[ignore]`d integration
/// test (the analog of the SDXL/Kolors IP-Adapter Phase-5 harnesses).
#[cfg(test)]
mod ip_validate;

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, SizeFloor, WeightsSource,
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
}

impl Generator for FluxGenerator {
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
        memory_strategy::admission_safety_check(
            self.variant.model_id(),
            contract,
            context,
            self.loaded_quant,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(
            self.variant.model_id(),
            contract,
            context,
            self.loaded_quant,
        )?;
        Ok(Some(Box::new(memory_strategy::FluxMemoryScope::new(
            self.variant.model_id(),
            self.pipe.device().clone(),
            contract,
            context,
        ))))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor: the descriptor advertises no conditioning and (for schnell) no
        // guidance / negative prompt, so any of those is rejected here (distilled-model honesty).
        self.descriptor
            .capabilities
            .validate_request(self.variant.model_id(), req)?;
        let id = self.variant.model_id();
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
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        let stage_residency = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stage_residency);
        let stream_transformer_blocks = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stream_transformer_blocks);
        if req.memory.as_ref().is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        }) && !stage_residency
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode, attention, and transformer residency require request-scoped staged residency",
                self.variant.model_id()
            )));
        }
        if req.use_pid
            && req.memory.is_some_and(|memory| {
                memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
            })
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: optimized native-VAE memory strategies do not support PiD decode",
                self.variant.model_id()
            )));
        }
        *candle_gen::lock_recover(&self.stream_cancel) = req.cancel.clone();
        *candle_gen::lock_recover(&self.bounded_host_decode) =
            req.memory.is_some_and(|memory| memory.tile_vae_decode);
        let result = self.residency.run_request_scoped(
            stage_residency,
            stream_transformer_blocks,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text| self.pipe.encode_residency(text, &req.prompt),
            |_| Ok(self.pipe.device().synchronize()?),
            |heavy, encoded, on_progress| {
                let result = self.pipe.render_residency(req, heavy, encoded, on_progress);
                candle_gen::synchronize_result(self.pipe.device(), result)
            },
        );
        *candle_gen::lock_recover(&self.bounded_host_decode) = false;
        *candle_gen::lock_recover(&self.stream_cancel) = gen_core::CancelFlag::default();
        let images = result?;
        Ok(GenerationOutput::Images(images))
    }
}

/// The descriptor for a FLUX.1 `variant` — the surface sc-3694 actually wires: txt2img only (no
/// conditioning / LoRA / quantization advertised — those are the Python fallback's job until candle
/// wires them), dev exposes guidance (schnell does not), no negative prompt / true-CFG. `backend` is
/// `"candle"` and `mac_only` is `false` (the two backend-correct deviations from `mlx-gen-flux`).
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: variant.model_id(),
        family: "flux",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Negative prompt / true-CFG ride the mlx Reference path (dev + reference = real CFG),
            // which this txt2img slice does not wire — so neither is advertised on either variant.
            supports_negative_prompt: false,
            supports_guidance: variant.supports_guidance(),
            supports_true_cfg: false,
            // txt2img only in sc-3694 — Reference/IP-adapter lands later; an empty list means the
            // shared `validate_request` rejects any conditioning and the worker keeps those shapes on
            // the Python path.
            conditioning: vec![],
            // LoRA/LoKr (mlx supports both) and Q4/Q8 quantization are deferred to a later slice; not
            // advertised, and rejected at load rather than silently dropped.
            supports_lora: true,
            supports_lokr: true,
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123): the denoise routes
            // through the shared driver, so the per-generation `sampler`/`scheduler` knob can select any
            // curated integrator/schedule. The DEFAULT (None/None) reproduces the native flow-match
            // Euler path (N1). FLUX had no legacy sampler/scheduler aliases, so no `menu_with_aliases`.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            supported_quants: &[],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: true,
            // Per-step latent previews (epic 16948, sc-16956): the registered txt2img route hands
            // `crate::preview::hook` to the shared flow driver, projecting the unpacked 16-channel
            // latent through the reused epic-16624 fit. Both variants share one render lane, so both
            // advertise. `candle-gen-catalog`'s `preview_advertising` guard derives this from the
            // sources and fails if the flag and the wiring ever disagree.
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

/// FLUX.1 `schnell` descriptor (registry).
pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(Variant::Schnell)
}

/// FLUX.1 `dev` descriptor (registry).
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(Variant::Dev)
}

/// Construct a lazy candle FLUX generator for `variant` from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a black-forest-labs `FLUX.1-{schnell,dev}` snapshot (root
/// `flux1-*.safetensors` + `ae.safetensors`, plus the `text_encoder/`, `text_encoder_2/`,
/// `tokenizer_2/` subdirs). LoRA adapters, on-the-fly quantization, and control/IP-adapter overlays
/// are rejected — none are wired in this slice, so refusing is more honest than silently dropping
/// them (the worker falls back to Python).
fn load_variant(variant: Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
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

// Link-time self-registration into gen-core's model registry — one descriptor per variant. Linking
// the explicit family and platform catalogs resolve both candle generators
// with no central match to edit.
candle_gen::register_generators! {
    pub(crate) const SCHNELL_REGISTRATION = descriptor_schnell => load_schnell
}
candle_gen::register_generators! {
    pub(crate) const DEV_REGISTRATION = descriptor_dev => load_dev
}

/// Add all Candle FLUX.1 providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(SCHNELL_REGISTRATION)
        .register_generator(DEV_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(SCHNELL_MEMORY_REGISTRATION)
        .register_memory_behavior(SCHNELL_MEMORY_BEHAVIOR)
        .register_memory_strategy(DEV_MEMORY_REGISTRATION)
        .register_memory_behavior(DEV_MEMORY_BEHAVIOR);
    registry
}

#[cfg(feature = "cuda")]
fn registered_schnell_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(FLUX1_SCHNELL_ID, spec)
}

#[cfg(feature = "cuda")]
fn registered_dev_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(FLUX1_DEV_ID, spec)
}

#[cfg(feature = "cuda")]
const SCHNELL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: FLUX1_SCHNELL_ID,
    contract: registered_schnell_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

#[cfg(feature = "cuda")]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

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
    }

    /// FLUX.1-schnell sibling required by the live worker gate (sc-12138). Needs a real-file snapshot
    /// in `FLUX_SCHNELL_DIR` and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux_schnell_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux1_schnell", "FLUX_SCHNELL_DIR", load_schnell, 4);
    }
}
