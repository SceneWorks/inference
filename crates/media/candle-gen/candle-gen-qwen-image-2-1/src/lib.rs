//! # candle-gen-qwen-image-2-1
//!
//! The **Qwen-Image 2.1** provider crate for [`candle-gen`](candle_gen) (sc-24109) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-qwen-image-2-1` (sc-24108) and a native port of the frozen
//! upstream `QwenImage21Pipeline`: Qwen3-VL-8B text conditioning, the 32-layer single-stream
//! block-causal DiT, the resolution-shifted flow-match Euler schedule, and the 64-channel / 16×
//! **RGBA** autoencoder — registered as the distinct engine id [`MODEL_ID`] (`qwen_image_2_1`).
//! It is **not** a checkpoint alias of the 2512 `candle-gen-qwen-image` crate: the DiT
//! (single-stream, shared modulation, block-causal joint sequence, unpatched 64-ch latents), the
//! text tower (Qwen3-VL, last-layer pre-norm hidden states) and the VAE (4-channel in/out, 16×
//! spatial, learned-shortcut-free residual up/down) are all new architectures.
//!
//! ## Frozen upstream
//!
//! Every numeric leaf mirrors the pinned sources recorded in [`UPSTREAM_HF_REVISION`],
//! [`UPSTREAM_DIFFUSERS_REVISION`] and [`UPSTREAM_GITHUB_REVISION`] (see `UPSTREAM.md`); the parity
//! fixtures this crate's tests read are the **same** committed `.safetensors` the MLX twin reads
//! (`crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/`, produced by
//! `crates/media/mlx-gen/tools/dump_qwen21_*.py` from those exact revisions), so both backends are
//! held to one numeric reference. The weights are distributed under the **Qwen Research License**
//! (research/evaluation only) — [`UPSTREAM_LICENSE_NOTICE`] is the attribution the licence requires
//! every derived bundle to carry (`NOTICE`).
//!
//! ## What ships in this story
//!
//! Text-to-image at the seven upstream presets (any 32-px-multiple size in range), steps, seed,
//! true-CFG guidance with a negative prompt, seeded determinism, progress + cancellation through
//! the shared `run_flow_sampler` contract, and Resident/Sequential residency. The VAE decodes RGBA
//! ([`QwenImage21Vae::decode_rgba`]), and the request's `output_channels` selects what
//! reaches the caller: `Rgb` (the default) composites that decode over white
//! ([`pipeline::rgba_to_rgb_over_white`]) and emits an `Image`, while `Rgba` emits the
//! four-channel decode unflattened as an `RgbaImage` with straight alpha (sc-24111). There is
//! ONE decode either way — upstream has no transparency flag, always decodes four channels,
//! and leaves *whether* a render is transparent to the prompt; see `UPSTREAM.md`.
//! A request asking for `GenerationMemory::tile_vae_decode` gets the **bounded** decode
//! ([`QwenImage21Vae::decode_rgba_tiled`]): the decoder's global head runs once and only the
//! up-sampling tail — where the 144-channel full-resolution spike lives — is tiled and
//! trapezoidally blended, over the one [`VaeTiling::QWEN_IMAGE_2_1`] geometry gen-core declares for
//! both engines.
//!
//! Reference conditioning (sc-24110) rides the **same** upstream call: `Conditioning::Reference`
//! and `Conditioning::MultiReference` are flattened, in request order, into one ordered list of
//! one to ten condition images, which reach the Qwen3-VL vision tower (via [`mod@reference`] and
//! [`text_encoder::QwenImage21TextEncoder::encode_conditioning`]) as well as the VAE (via
//! [`pipeline::encode_references`]). Upstream has no mask tensor and performs no inpainting, so
//! `Conditioning::Mask` is deliberately unadvertised and refused with the workaround named — see
//! `UPSTREAM.md`.
//!
//! ## Deliberate differences from the MLX twin
//!
//! * `backend = "candle"`, `mac_only = false`.
//! * **No on-the-fly Q4/Q8.** MLX quantizes the DiT's Linears at load (`spec.quantize`); candle has
//!   no affine-quantize-at-load path, so [`load`] refuses `LoadSpec::quantize` with a typed
//!   `Unsupported` and the descriptor advertises no `supported_quants`. A snapshot that is
//!   **already** an MLX-packed tier still loads: every DiT Linear goes through
//!   `candle_gen::quant::AdaptLinear::linear_detect_gs`, the same packed-detect seam
//!   `candle-gen-qwen-image` uses.
//! * Noise is drawn from the shared launch-portable CPU `StdRng` (`candle_gen::seed`), not MLX's
//!   RNG, so a seed reproduces within a backend but not across the two.
//! * The Qwen3 decoder block is ported here rather than reused from `candle-gen-z-image` (whose
//!   copy is a private module pinned to Z-Image's layer[-2] / `model.` conventions); the MLX twin
//!   could reuse `mlx_gen_z_image::text_encoder::EncoderLayer` because that one is public and
//!   convention-free.
//!
//! [`QwenImage21Vae::decode_rgba`]: crate::vae::QwenImage21Vae::decode_rgba
//! [`QwenImage21Vae::decode_rgba_tiled`]: crate::vae::QwenImage21Vae::decode_rgba_tiled
//! [`VaeTiling::QWEN_IMAGE_2_1`]: candle_gen::gen_core::tiling::VaeTiling::QWEN_IMAGE_2_1

use std::path::Path;
use std::sync::Mutex;

use candle_core::Device;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, OutputChannels, Precision, Progress, Quant, SizeFloor,
};
use candle_gen::residency::Residency;
use candle_gen::{CandleError as Error, Result};

pub mod config;
pub mod loader;
pub mod memory_strategy;
pub mod pipeline;
pub mod quant;
pub mod reference;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

/// Hugging Face repository the production snapshot layout is frozen from.
pub const UPSTREAM_HF_REPO: &str = "Qwen/Qwen-Image-2.1";
/// Pinned `Qwen/Qwen-Image-2.1` revision (weights, configs, tokenizer, scheduler config).
pub const UPSTREAM_HF_REVISION: &str = "790c92633540aa0cb11d9abf19eb46d861714758";
/// GitHub repository carrying the upstream presets, defaults and prompt-rewrite tooling.
pub const UPSTREAM_GITHUB_REPO: &str = "QwenLM/Qwen-Image-2.1";
/// Pinned `QwenLM/Qwen-Image-2.1` commit (README presets / 40-step default / RGBA prompt form).
pub const UPSTREAM_GITHUB_REVISION: &str = "fb7ae1d1f9611cd91524d03c53c5246b36ac8577";
/// Pinned `huggingface/diffusers` commit whose `QwenImage21Pipeline`,
/// `QwenImage21Transformer2DModel`, `AutoencoderKLQwenImage21` and
/// `FlowMatchEulerDiscreteScheduler` are the numeric reference for every port here.
pub const UPSTREAM_DIFFUSERS_REVISION: &str = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65";
/// Licence the pinned weights are distributed under.
pub const UPSTREAM_LICENSE: &str = "Qwen Research License Agreement (research/evaluation only)";
/// The attribution notice the Qwen Research License §3(c) requires in every distributed copy.
pub const UPSTREAM_LICENSE_NOTICE: &str = "Qwen is licensed under the Qwen RESEARCH LICENSE \
    AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights \
    Reserved.";

pub use config::{
    SchedulerConfig, SizePreset, TextEncoderConfig, TransformerConfig, VaeConfig, VisionConfig,
    DEFAULT_STEPS, DEFAULT_TRUE_CFG, IMAGE_TOKENS_PER_SLOT, MAX_REFERENCE_IMAGES,
    OUTPUT_RESOLUTION, PRESETS, SIZE_MULTIPLE, SYSTEM_PROMPT, VAE_SCALE_FACTOR, VISION_MERGE_SIZE,
    VISION_PATCH_SIZE,
};
pub use loader::{
    load_scheduler_config, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    load_vision_config,
};
pub use memory_strategy::{admission_geometry, AdmissionGeometry};
pub use pipeline::{
    create_noise, decode_rgb, decode_rgba, decode_tiling, denoise, encode_prompt,
    encode_references, joint_layout, pack_latents, rgba_to_rgb_over_white, text_rows,
    unpack_latents, DenoiseInputs, ReferenceConditioning, DECODE_OVERLAP, DECODE_TILE_EDGE,
};
pub use quant::{Tier, GROUP_SIZE};
pub use reference::{
    calculate_dimensions, collect_references, prepare_reference, prepare_references,
    reference_derived_size, reference_target_size, validate_reference_count, PreparedReference,
};
pub use text_encoder::{
    image_pad_token_id, prompt_template, prompt_template_ti2i, system_prefix,
    system_prompt_drop_count, QwenImage21TextEncoder, TextConditioning, IMAGE_PAD_TOKEN,
};
pub use transformer::{JointLayout, QwenImage21Transformer, Segment};
pub use vae::QwenImage21Vae;

/// Registry id for Qwen-Image 2.1 (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image_2_1";
/// Descriptor family — distinct from the 2512 crate's `qwen-image`.
pub const FAMILY: &str = "qwen-image-2-1";
/// Smallest side upstream's pipeline can render: one vision slot (a 2×2 latent) at 16 px/token.
pub const MIN_SIZE: u32 = 32;

/// Qwen-Image 2.1's identity + capabilities — constructible without loading weights.
pub fn descriptor() -> ModelDescriptor {
    let max_size = PRESETS
        .iter()
        .map(|p| p.width.max(p.height))
        .max()
        .unwrap_or(2048);
    ModelDescriptor {
        // The Qwen3-VL tower is loaded from the snapshot's own `text_encoder/`; substituting one
        // through `LoadSpec::text_encoder` is not advertised for this route.
        encoder_contract: None,
        denoiser_output_latent_space: Some(&gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG: a negative prompt plus `true_cfg`/`guidance` above 1 runs the negative
            // branch (upstream default 1.0 = off).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Reference conditioning (sc-24110). `Reference` and `MultiReference` reach the same
            // upstream call — one ordered list of one to ten condition images — so both kinds are
            // advertised and flattened in request order, matching the seam the 2512 edit provider
            // already exposes to SceneWorks routing. `Mask` is deliberately **not** advertised:
            // upstream has no mask input (see UPSTREAM.md), and a mask travels as an ordinary
            // extra reference the prompt names.
            // `ReferenceRgba` (sc-24111) joins them: upstream converts every condition image
            // to RGBA anyway, so a reference that already carries alpha is the general case and
            // an RGB one the opaque special case. Advertising it is what lets a transparent layer
            // be edited (its alpha reaches the VAE encode) instead of being flattened at the
            // contract boundary.
            conditioning: vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            // Native transparency (sc-24111). The VAE is four-channel in and out on **every**
            // route behind this descriptor — T2I and reference alike — so there is no sub-path
            // that would have to fabricate an alpha to honour the flag. Whether a given render is
            // actually transparent is decided by the prompt, not by this bit; see UPSTREAM.md.
            supports_alpha_output: true,
            supports_lora: false,
            supports_lokr: false,
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: MIN_SIZE,
            max_size,
            max_count: 8,
            // **Installable** tiers, not on-the-fly ones (sc-24112). Candle still has no
            // affine-quantize-at-load path — `validate_load_spec` refuses `quantize` against a
            // dense snapshot with the same typed `Unsupported` as before — but the pre-quantized
            // Q4/Q8 tiers `mlx_gen_qwen_image_2_1::convert` writes DO install here: the DiT and the
            // Qwen3 tower both bind through `AdaptLinear::linear_detect_gs`. Advertising the tiers
            // is what lets the worker's A-B tier toggle reach this backend; see `crate::quant`.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // No `component_precision_floors`: a tier is a whole-pipeline contract, so every
            // packable component runs the tier the caller selected (`crate::quant`).
            requires_sigma_shift: true,
            supports_sequential_offload: true,
            // Both sides must be multiples of 32 px (16× VAE × 2×2 vision slot).
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// The heavy render-phase components — everything but the text encoder.
pub(crate) struct Heavy {
    transformer: QwenImage21Transformer,
    vae: QwenImage21Vae,
}

/// A loaded Qwen-Image 2.1 generator.
pub struct QwenImage21 {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops, derived from the tokenizer.
    drop_count: usize,
    scheduler: SchedulerConfig,
    device: Device,
    residency: Residency<QwenImage21TextEncoder, Heavy>,
    lifecycle: Mutex<()>,
    /// The spec this generator was loaded from — the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against (sc-24114).
    spec: LoadSpec,
    /// The loaded memory contract ([`memory_strategy::memory_strategy_contract`] over the snapshot
    /// on disk), published through [`Generator::memory_strategy_contract`] so the SceneWorks
    /// worker's optimized selections reach this generator instead of the trait's
    /// "has not adopted the shared memory-strategy contract" default.
    memory_strategy: gen_core::MemoryProviderContract,
}

/// The one refusal both Qwen-Image 2.1 backends raise for a `LoadSpec::precision` override. The
/// wording is shared with `mlx_gen_qwen_image_2_1::model::PRECISION_OVERRIDE_REFUSAL` by literal
/// (the two provider crates cannot share a symbol without routing it through gen-core); the tests
/// on both sides pin this exact text so the two cannot drift.
pub const PRECISION_OVERRIDE_REFUSAL: &str = "qwen_image_2_1: components load at the backend's \
    own compute dtype (the checkpoint's bf16 on MLX, CUDA and Metal; f32 on the candle CPU lane); \
    drop the precision override";

/// The one refusal both backends raise for a `LoadSpec::quantize` that is not an affine tier —
/// shared wording, see [`PRECISION_OVERRIDE_REFUSAL`].
pub fn non_affine_quant_refusal(quant: Quant) -> String {
    format!("qwen_image_2_1: {quant:?} is not a Qwen-Image 2.1 affine tier (Q4/Q8)")
}

/// Reject every overlay this route does not wire, with a typed, actionable error.
pub(crate) fn validate_load_spec(spec: &LoadSpec) -> gen_core::Result<()> {
    gen_core::reject_unknown_components(spec, &[], MODEL_ID)?;
    if spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(
            PRECISION_OVERRIDE_REFUSAL.into(),
        ));
    }
    if let Some(quant) = spec.quantize {
        if !matches!(quant, Quant::Q4 | Quant::Q8) {
            return Err(gen_core::Error::Unsupported(non_affine_quant_refusal(
                quant,
            )));
        }
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_2_1: LoRA/LoKr adapters are not wired for Qwen-Image 2.1 yet".into(),
        ));
    }
    if spec.text_encoder.is_some() {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_2_1: the Qwen3-VL text encoder is loaded from the snapshot's own \
             text_encoder/; LoadSpec::text_encoder substitution is not advertised"
                .into(),
        ));
    }
    // `spec.quantize` is a TIER SELECTOR here, resolved against the snapshot on disk: it matches a
    // pre-quantized tier (loaded packed, no quantization pass), or it is refused. A dense snapshot
    // with a Q4/Q8 request is still the same typed `Unsupported` — candle cannot produce that tier
    // itself. See `crate::quant::resolve_requested_tier`.
    if let gen_core::WeightsSource::Dir(root) = &spec.weights {
        crate::quant::resolve_requested_tier(root, spec.quantize)?;
    } else if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_2_1: candle has no on-the-fly Q4/Q8 quantization; provision an \
             already-packed snapshot instead"
                .into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_2_1: control / IP-adapter overlays are not wired (the reference/edit \
             route takes ordered condition images, not a control overlay)"
                .into(),
        ));
    }
    if spec.identity.is_some() {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_2_1: identity weights are not wired".into(),
        ));
    }
    Ok(())
}

/// Construct a [`QwenImage21`] from a [`LoadSpec`] whose `weights` is a `Qwen/Qwen-Image-2.1`
/// snapshot directory (see [`loader`]). `Resident` (the default) holds every component warm;
/// `Sequential` loads the text encoder, encodes, drops it, then loads the DiT + VAE — bounding
/// peak memory to `max(text encoder, DiT + VAE)`.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    validate_load_spec(spec)?;
    let root = loader::snapshot_root(&spec.weights)?;
    let device = candle_gen::default_device()?;
    let tokenizer = loader::load_tokenizer(root)?;
    let drop_count = system_prompt_drop_count(&tokenizer)?;
    let scheduler = loader::load_scheduler_config(root)?;
    // The loaded contract is priced from the snapshot on disk BEFORE any weight is read, so a
    // tier/request disagreement is the same refusal whether it is asked for through `load` or
    // through the memory registration.
    let memory_strategy = memory_strategy::memory_strategy_contract(MODEL_ID, spec)?;
    let residency = build_residency(spec, &device)?;
    Ok(Box::new(QwenImage21 {
        descriptor: descriptor(),
        tokenizer,
        drop_count,
        scheduler,
        device,
        residency,
        lifecycle: Mutex::new(()),
        spec: spec.clone(),
        memory_strategy,
    }))
}

fn build_residency(
    spec: &LoadSpec,
    device: &Device,
) -> Result<Residency<QwenImage21TextEncoder, Heavy>> {
    let text_spec = spec.clone();
    let text_device = device.clone();
    let heavy_spec = spec.clone();
    let heavy_device = device.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || loader::load_text_encoder(loader::snapshot_root(&text_spec.weights)?, &text_device),
        move |_use_pid| load_heavy(&heavy_spec, &heavy_device),
    )
}

fn load_heavy(spec: &LoadSpec, device: &Device) -> Result<Heavy> {
    let root: &Path = loader::snapshot_root(&spec.weights)?;
    let transformer = loader::load_transformer(root, device)?;
    let vae = loader::load_vae(root, device)?;
    Ok(Heavy { transformer, vae })
}

impl Generator for QwenImage21 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// The registered contract, priced from the loaded snapshot — the same one
    /// [`memory_strategy::MEMORY_REGISTRATION`] builds for this spec.
    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        memory_strategy::safety_check(&self.spec, &self.memory_strategy, context)
    }

    /// The registered executable seam (`memory_strategy::registered_begin_request`) on the
    /// loaded generator: the scope's `configure_request` writes the admitted
    /// `GenerationMemory` onto the request, which [`Self::generate`] then honours — staged
    /// residency through `Residency::run_request_scoped`, bounded decode through
    /// [`pipeline::decode_tiling`].
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        memory_strategy::registered_begin_request(&self.spec, &self.memory_strategy, context)
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

/// The resolved run parameters of one request.
pub(crate) struct RunParams {
    pub(crate) true_cfg: f32,
    pub(crate) use_negative: bool,
    pub(crate) base_seed: u64,
    pub(crate) sigmas: Vec<f32>,
}

pub(crate) fn resolve_run_params(
    scheduler: &SchedulerConfig,
    req: &GenerationRequest,
) -> Result<RunParams> {
    let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
    let tokens = scheduler::image_tokens(req.width, req.height);
    let native = scheduler::sigmas(scheduler, steps, tokens)?;
    let mu = scheduler::mu_for_tokens(scheduler, tokens) as f32;
    let sigmas = candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);
    // `true_cfg_scale`: honour a caller-supplied `true_cfg`, then `guidance`, then the upstream
    // default (off). Guidance engages only with a negative prompt, as upstream's `do_true_cfg`.
    let true_cfg = req.true_cfg.or(req.guidance).unwrap_or(DEFAULT_TRUE_CFG);
    let has_negative = req
        .negative_prompt
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty());
    Ok(RunParams {
        true_cfg,
        use_negative: true_cfg > 1.0 && has_negative,
        base_seed: req.seed.unwrap_or_else(gen_core::default_seed),
        sigmas,
    })
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy scope) OR a `Sequential` load policy. Mirrors the MLX twin.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

impl QwenImage21 {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let params = resolve_run_params(&self.scheduler, req)?;
        let tiling = crate::pipeline::decode_tiling(req);
        let drop = self.drop_count;
        let device = self.device.clone();
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        // Staged residency is a per-request execution decision (sc-24114): the shared worker
        // selects it through the memory-strategy scope, which writes `GenerationMemory` onto the
        // request. A `Sequential` load policy stages every request whatever the request says, so
        // the two are OR'd — a request can stage a warm-loaded generator, never un-stage a
        // sequential one.
        let stage_residency = stage_residency_for(&self.spec, req);
        self.residency.run_request_scoped(
            stage_residency,
            false,
            &req.cancel,
            false,
            on_progress,
            |te: &QwenImage21TextEncoder| {
                // The ordered reference list, host-preprocessed against the snapshot's own
                // Qwen3-VL processor geometry. Empty ⇒ the text-to-image route, unchanged.
                let sources = crate::reference::collect_references(req)?;
                let references = if sources.is_empty() {
                    Vec::new()
                } else {
                    let vision = te.vision_config().ok_or_else(|| {
                        Error::Unsupported(
                            "qwen_image_2_1: reference conditioning needs the snapshot's Qwen3-VL \
                             vision tower (`text_encoder/config.json` `vision_config` + \
                             `model.visual.*`), which this snapshot does not carry"
                                .to_string(),
                        )
                    })?;
                    crate::reference::prepare_references(&sources, vision, te.device())?
                };
                let pos =
                    te.encode_conditioning(&self.tokenizer, &req.prompt, drop, &references)?;
                let neg = if params.use_negative {
                    Some(te.encode_conditioning(
                        &self.tokenizer,
                        req.negative_prompt.as_deref().unwrap_or(""),
                        drop,
                        &references,
                    )?)
                } else {
                    None
                };
                Ok((pos, neg, references))
            },
            |_| Ok(()),
            |heavy, (pos, neg, references), on_progress| {
                let channels = heavy.transformer.config().in_channels;
                // The joint layout + the condition latents: both branches share one reference
                // encode, but a different prompt is a different text length, hence two layouts.
                let reference_latents =
                    crate::pipeline::encode_references(&heavy.vae, &references)?;
                let pos_layout = crate::pipeline::joint_layout(
                    &pos.image_pad_mask,
                    &references,
                    req.width,
                    req.height,
                )?;
                let neg_layout = neg
                    .as_ref()
                    .map(|neg| {
                        crate::pipeline::joint_layout(
                            &neg.image_pad_mask,
                            &references,
                            req.width,
                            req.height,
                        )
                    })
                    .transpose()?;
                let pos_text = crate::pipeline::text_rows(&pos.hidden, &pos.image_pad_mask)?;
                let neg_text = neg
                    .as_ref()
                    .map(|neg| crate::pipeline::text_rows(&neg.hidden, &neg.image_pad_mask))
                    .transpose()?;
                let conditioned = !references.is_empty();
                // ONE decode per image either way — upstream always decodes four channels and
                // has no transparency flag — so this branch chooses only whether the alpha is
                // carried (`Rgba`) or applied over white (`Rgb`, the default). The latents,
                // the denoise and the VAE call are identical.
                let rgba_out = req.output_channels == OutputChannels::Rgba;
                let mut images = Vec::with_capacity(if rgba_out { 0 } else { req.count as usize });
                let mut rgba_images =
                    Vec::with_capacity(if rgba_out { req.count as usize } else { 0 });
                for i in 0..req.count {
                    let seed = candle_gen::image_seed(params.base_seed, i);
                    let latents = create_noise(seed, req.width, req.height, channels, &device)?;
                    let latents = denoise(
                        DenoiseInputs {
                            transformer: &heavy.transformer,
                            sigmas: &params.sigmas,
                            latents,
                            prompt_embeds: &pos_text,
                            negative_embeds: neg_text.as_ref(),
                            true_cfg_scale: params.true_cfg,
                            width: req.width,
                            height: req.height,
                            sampler: req.sampler.as_deref(),
                            seed,
                            cancel: &req.cancel,
                            references: conditioned.then(|| ReferenceConditioning {
                                latents: &reference_latents,
                                layout: &pos_layout,
                                negative_layout: neg_layout.as_ref(),
                            }),
                        },
                        on_progress,
                    )?;
                    on_progress(Progress::Decoding);
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    if rgba_out {
                        rgba_images.push(decode_rgba(
                            &heavy.vae,
                            &latents,
                            req.width,
                            req.height,
                            tiling.as_ref(),
                            Some(&req.cancel),
                        )?);
                    } else {
                        images.push(decode_rgb(
                            &heavy.vae,
                            &latents,
                            req.width,
                            req.height,
                            tiling.as_ref(),
                            Some(&req.cancel),
                        )?);
                    }
                }
                if rgba_out {
                    Ok(GenerationOutput::ImagesRgba(rgba_images))
                } else {
                    Ok(GenerationOutput::Images(images))
                }
            },
        )
    }
}

/// Capability-driven request validation: the shared floor (count, size range + 32-px grid,
/// negative/guidance support, sampler/scheduler membership, finiteness) plus the family's own
/// `steps >= 2` (the terminal-sigma stretch is undefined at one step).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // The reference list first, so a caller sees the actionable message (the `Mask` workaround,
    // the one-to-ten window) rather than the shared floor's generic "unsupported conditioning",
    // and so a bad count is refused before any weight is touched. Mirrors the MLX twin.
    crate::reference::collect_references(req)?;
    caps.validate_request(MODEL_ID, req)?;
    if req.prompt.trim().is_empty() && req.negative_prompt.is_none() {
        // Upstream renders an empty prompt as a single space; accept it, but a whitespace-only
        // prompt with nothing else is almost certainly a caller bug, so say so.
        return Err(Error::Msg(
            "qwen_image_2_1: the prompt is empty (upstream would condition on a single space)"
                .into(),
        ));
    }
    if let Some(steps) = req.steps {
        if steps < 2 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: steps must be >= 2 (got {steps}); the terminal-sigma stretch is undefined at one step"
            )));
        }
    }
    Ok(())
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

/// Add the Candle Qwen-Image 2.1 generator to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    register_memory_contract_surfaces(registry.register_generator(REGISTRATION))
}

/// The shared-ladder registrations (sc-24112). Split out the way the sibling Candle providers do so
/// a CUDA-less catalog build can append the contract surface to a registry that already carries the
/// generator, without registering it twice.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(candle_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: candle_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: MODEL_ID,
            contract: |spec| memory_strategy::weights_free_memory_strategy_contract(MODEL_ID, spec),
        })
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit Candle Qwen-Image 2.1 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{OffloadPolicy, WeightsSource};

    fn req(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red fox".into(),
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn frozen_revisions_are_full_shas() {
        for sha in [
            UPSTREAM_HF_REVISION,
            UPSTREAM_GITHUB_REVISION,
            UPSTREAM_DIFFUSERS_REVISION,
        ] {
            assert_eq!(sha.len(), 40);
            assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
        }
        assert!(UPSTREAM_LICENSE_NOTICE.contains("Qwen RESEARCH LICENSE AGREEMENT"));
    }

    /// The two load-spec refusals both backends share **by literal** (sc-24114): the exact text
    /// is pinned here and, character for character, in `mlx_gen_qwen_image_2_1::model`'s tests,
    /// so the wording cannot drift on either side without one of the two pins going red. Both are
    /// the typed `Unsupported`, not `Msg`.
    #[test]
    fn precision_and_non_affine_quant_refusals_share_the_mlx_wording() {
        assert_eq!(
            PRECISION_OVERRIDE_REFUSAL,
            "qwen_image_2_1: components load at the backend's own compute dtype (the checkpoint's \
             bf16 on MLX, CUDA and Metal; f32 on the candle CPU lane); drop the precision override"
        );
        assert_eq!(
            non_affine_quant_refusal(Quant::Nvfp4),
            "qwen_image_2_1: Nvfp4 is not a Qwen-Image 2.1 affine tier (Q4/Q8)"
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let mut fp32 = spec.clone();
        fp32.precision = Precision::Fp32;
        match validate_load_spec(&fp32) {
            Err(gen_core::Error::Unsupported(message)) => {
                assert_eq!(message, PRECISION_OVERRIDE_REFUSAL);
            }
            other => panic!("a precision override must be a typed Unsupported, got {other:?}"),
        }
        match validate_load_spec(&spec.with_quant(Quant::Nvfp4)) {
            Err(gen_core::Error::Unsupported(message)) => {
                assert_eq!(message, non_affine_quant_refusal(Quant::Nvfp4));
            }
            other => panic!("a non-affine tier must be a typed Unsupported, got {other:?}"),
        }
    }

    /// A request stages when the memory-strategy scope said so OR the load policy is
    /// `Sequential`; a plain request on a `Resident` load does not.
    ///
    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use candle_gen::gen_core::GenerationMemory;

        let resident = LoadSpec::new(WeightsSource::Dir("/qwen-image-2-1".into()));
        let sequential = resident
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential);
        let plain = req(64, 64);
        let mut staged = req(64, 64);
        staged.memory = Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        });
        let mut unstaged = req(64, 64);
        unstaged.memory = Some(GenerationMemory::default());

        assert!(!stage_residency_for(&resident, &plain));
        assert!(!stage_residency_for(&resident, &unstaged));
        assert!(stage_residency_for(&resident, &staged));
        assert!(stage_residency_for(&sequential, &plain));
        assert!(stage_residency_for(&sequential, &staged));
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = provider_registry().unwrap();
        let ids: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, ["qwen_image_2_1"]);
        assert!(registry.descriptor_conformance_errors().is_empty());
    }

    #[test]
    fn descriptor_is_distinct_from_the_2512_route() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_2_1");
        assert_eq!(d.family, "qwen-image-2-1");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        assert_eq!(d.capabilities.max_size, 2752);
        assert_eq!(d.capabilities.min_size, 32);
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 32 }
        );
        assert!(d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            "the reference/edit route is the same upstream call as text-to-image, and a \
             transparent reference is an ordinary reference that kept its alpha (sc-24111)"
        );
        assert!(
            d.capabilities.supports_alpha_output,
            "the VAE is four-channel on every route behind this descriptor (sc-24111)"
        );
        assert!(
            !d.capabilities.accepts(gen_core::ConditioningKind::Mask),
            "upstream has no mask input; a mask travels as an ordinary extra reference"
        );
        assert_eq!(d.denoiser_output_latent_space.map(|s| s.channels), Some(64));
        assert!(gen_core::registry::model_descriptor_errors(&d).is_empty());
    }

    #[test]
    fn every_preset_validates_and_off_grid_sizes_do_not() {
        let caps = descriptor().capabilities;
        for p in PRESETS {
            assert!(
                validate_request(&caps, &req(p.width, p.height)).is_ok(),
                "{}",
                p.ratio
            );
        }
        let err = validate_request(&caps, &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(err.contains("32"), "{err}");
        assert!(validate_request(&caps, &req(4096, 4096)).is_err());
        assert!(validate_request(&caps, &req(32, 32)).is_ok());
    }

    #[test]
    fn steps_below_two_and_empty_prompts_are_refused() {
        let caps = descriptor().capabilities;
        let mut r = req(2048, 2048);
        r.steps = Some(1);
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
        r.steps = Some(2);
        assert!(validate_request(&caps, &r).is_ok());
        r.prompt = "   ".into();
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("prompt is empty"), "{err}");
    }

    #[test]
    fn reference_shapes_are_refused_at_validate_before_any_weight_loads() {
        use candle_gen::gen_core::{Conditioning, Image};

        let caps = descriptor().capabilities;
        let image = |n: u32| Image {
            width: n,
            height: n,
            pixels: vec![0; (n * n * 3) as usize],
        };
        let mut r = req(2048, 2048);
        assert!(
            validate_request(&caps, &r).is_ok(),
            "no conditioning is T2I"
        );

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..11).map(|_| image(8)).collect(),
        }];
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("at most 10"), "{err}");

        // The `Mask` refusal must be this route's actionable message, not the shared floor's
        // generic "unsupported conditioning" — which is why the reference check runs first.
        r.conditioning = vec![Conditioning::Mask { image: image(8) }];
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("no mask input"), "{err}");
        assert!(err.contains("extra reference"), "{err}");

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..10).map(|_| image(8)).collect(),
        }];
        assert!(
            validate_request(&caps, &r).is_ok(),
            "ten references is the documented boundary"
        );
    }

    #[test]
    fn run_params_follow_upstream_defaults() {
        let s = SchedulerConfig::production();
        let p = resolve_run_params(&s, &req(2048, 2048)).unwrap();
        assert_eq!(p.sigmas.len(), 41, "40 default steps + the trailing 0");
        assert_eq!(p.true_cfg, 1.0);
        assert!(!p.use_negative);
        let mut r = req(2048, 2048);
        r.negative_prompt = Some("blurry".into());
        r.guidance = Some(3.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(p.use_negative);
        assert_eq!(p.true_cfg, 3.0);
        r.true_cfg = Some(1.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(
            !p.use_negative,
            "true_cfg 1.0 disables guidance even with a negative"
        );
    }

    #[test]
    fn load_rejects_single_file_and_unwired_overlays() {
        let err = load(&LoadSpec::new(WeightsSource::File(
            "/tmp/q.safetensors".into(),
        )))
        .err()
        .expect("a single file is refused")
        .to_string();
        assert!(err.contains("snapshot directory"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_text_encoder(WeightsSource::Dir("/elsewhere".into()));
        let err = load(&spec)
            .err()
            .expect("substitution is refused")
            .to_string();
        assert!(err.contains("text_encoder"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q8);
        let err = load(&spec).err().expect("quantize is refused").to_string();
        assert!(err.contains("on-the-fly"), "{err}");
        assert!(
            load(
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
                    .with_offload_policy(OffloadPolicy::Sequential)
            )
            .is_err(),
            "a missing snapshot fails at the tokenizer, for both policies"
        );
    }
}
