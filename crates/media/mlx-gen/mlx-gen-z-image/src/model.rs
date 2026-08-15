//! `ZImageTurbo` — the Z-Image-turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and explicit registration constant.
//!
//! [`load`] assembles the full model from a `Tongyi-MAI/Z-Image-Turbo` snapshot directory (see
//! [`crate::loader`]) — tokenizer, Qwen text encoder, DiT transformer, VAE decoder — and
//! [`ZImageTurbo::generate`] runs the complete prompt→image pipeline: tokenize → encode →
//! seeded noise → flow-match Euler denoise over the DiT → VAE decode → RGB8. The chain is
//! parity-proven against the frozen Python fork on real bf16 weights (sc-2352).

use mlx_gen::gen_core;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, resolve_flow_schedule,
    ActivationMemoryAnchor, Capabilities, ConditioningKind, Error, FlowMatchEuler,
    GenerationOutput, GenerationRequest, Generator, LatentDecoder, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, Residency, Result, StagedHeavy, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidDecoder, PidEngine};
use mlx_rs::Dtype;
use std::path::Path;

use crate::loader;
use crate::pipeline::{
    self, denoise_with_progress_and_preview, encode_init_latents, init_time_step,
};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image-turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. (`pub(crate)` so the ControlNet variant shares the same default.)
pub(crate) const DEFAULT_STEPS: u32 = 4;

/// Flow-match time-shift for Z-Image-Turbo: the model's own published schedule from
/// `scheduler/scheduler_config.json` (`FlowMatchEulerDiscreteScheduler`, `shift=3.0`,
/// `use_dynamic_shifting=false`) — static, resolution-independent.
///
/// **Deliberate choice (sc-2536; Michael, 2026-06-01) — do NOT "restore" `linear`.** The mflux
/// MLX path this port replaces (`MlxZImageAdapter` → `generate_image`'s default `linear`
/// scheduler) actually uses a *dynamic*, resolution-dependent shift (≈3.16 @1024², 1.88 @512²,
/// 25 @2048²). We use the model's static 3.0 instead: A/B renders (`tools/compare_z_image_
/// schedulers.py`) are visually identical at 1024² and only differ at lower resolutions, where
/// 3.0 reads slightly crisper — the preferred look. So 3.0 is an intentional, model-config-backed
/// deviation from the MLX path, not drift.
///
/// (The *original* port's bug — replaced by sc-2536 — was using `FlowMatchEuler::for_image`'s
/// empirical per-step `mu`, the *full* Z-Image model's scheduler, ≈shift 10. That was wrong;
/// `linear` and 3.0 are both reasonable, empirical-`mu` was not.)
///
/// `pub(crate)` so the ControlNet variant ([`crate::model_control`]) uses the identical schedule —
/// it is the same base turbo model, and the parity golden holds the schedule fixed on both sides.
pub(crate) const SCHEDULE_SHIFT: f32 = 3.0;

/// Registry id for Z-Image-turbo (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "z_image_turbo";

/// PiD backbone (latent-space) tag for Z-Image (epic 7840, sc-7846). Z-Image ships Flux1-dev's 16-ch
/// VAE, so it reuses the `flux` PiD student — the `mlx_gen_pid::registry` `zimage-turbo` alias resolves
/// to exactly that checkpoint/space. Used only at load time to build the [`PidEngine`].
pub const PID_BACKBONE: &str = "zimage-turbo";

/// Z-Image-turbo's identity + capabilities — constructible without loading weights (registry
/// introspection). Values are conservative-but-real; sampler/scheduler lists fill in with the
/// scheduler port.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            // img2img reference; ControlNet is a separate variant (sc-2349).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3). Turbo is guidance-distilled to
            // ~4 steps; an unset `req.sampler` is the curated Euler over the static-shift schedule.
            samplers: curated_sampler_names(),
            // Scheduler axis (epic 7114): the static-shift schedule is the byte-exact default (an unset
            // `req.scheduler`); a curated name re-shapes the σ schedule over the same `shift=3.0`.
            schedulers: curated_scheduler_names(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
            supports_preview: true,
            ..Default::default()
        },
    }
}

/// A loaded Z-Image-turbo generator: the four model components assembled from a snapshot
/// directory, plus the cached descriptor.
pub struct ZImageTurbo {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// The provider's half of the shared memory-strategy handshake (SC-15449 / SC-15615), built from the
    /// `LoadSpec` at load so its asset facts describe the snapshot this generator actually loaded.
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    loaded_tier: mlx_gen::gen_core::MemoryNumericTier,
    /// Request-scoped component residency (SC-15806). Construction retains the two phase loaders;
    /// each request either populates/reuses one warm pair or runs encode → drop text → load heavy →
    /// denoise/decode according to [`GenerationMemory::stage_residency`](gen_core::GenerationMemory).
    /// The [`Residency`] seam owns eviction-before-reload, eval/drop/clear discipline, cancellation,
    /// and error-safe cache flushing.
    residency: Residency<TextEncoder, ZImageHeavyOwned>,
}

/// The heavy render-phase components (the DiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the text encoder. Owned by the warm pair or by a staged request.
/// `pub(crate)` so the **base** (non-distilled) `z_image` sibling ([`crate::model_base`]) shares the
/// identical heavy bundle on the same shared [`load_residency`] seam (sc-11124, F-172).
pub(crate) struct ZImageHeavyOwned {
    pub(crate) transformer: ZImageTransformer,
    pub(crate) vae: Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7846). `Some` only when the
    /// `LoadSpec` carried `pid`; selected per-generation by `req.use_pid`. Z-Image shares Flux1's VAE
    /// latent space, so it reuses the `flux` PiD student via the `zimage-turbo` registry alias.
    pub(crate) pid: Option<PidEngine>,
}

/// The light (decode-only) z-image bundle that survives the DiT drop during staged decode
/// (sc-13571, GitHub #1658): the VAE plus the optional PiD overlay. [`StagedHeavy::shed_dit`] drops the
/// multi-GB DiT and moves these here, so the tiled VAE decode peak excludes the transformer.
pub(crate) struct ZImageLight {
    pub(crate) vae: Vae,
    pub(crate) pid: Option<PidEngine>,
}

/// A borrowed decode view (VAE + optional PiD) — from the owned [`ZImageLight`] under `Sequential`
/// (post-shed), or from the still-warm [`ZImageHeavyOwned`] under `Resident` — so the decode body is
/// written once for both policies.
pub(crate) struct ZImageDecodeView<'a> {
    pub(crate) vae: &'a Vae,
    #[allow(dead_code)]
    // carried for symmetry; the staged path mints its PiD decoder in the denoise phase
    pub(crate) pid: Option<&'a PidEngine>,
}

impl StagedHeavy for ZImageHeavyOwned {
    type Light = ZImageLight;
    type DecodeView<'a> = ZImageDecodeView<'a>;
    fn shed_dit(self) -> ZImageLight {
        // `self.transformer` (the DiT) drops here; the VAE + PiD move into the light bundle.
        ZImageLight {
            vae: self.vae,
            pid: self.pid,
        }
    }
    fn decode_view(&self) -> ZImageDecodeView<'_> {
        ZImageDecodeView {
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
    fn light_view(light: &ZImageLight) -> ZImageDecodeView<'_> {
        ZImageDecodeView {
            vae: &light.vae,
            pid: light.pid.as_ref(),
        }
    }
}

/// Construct a [`ZImageTurbo`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image-Turbo`
/// snapshot (the diffusers multi-component tree — `tokenizer/`, `text_encoder/`, `transformer/`,
/// `vae/`). Weights load dense at their on-disk dtype (bf16); the text encoder promotes to f32
/// internally. `spec.quantize` (Q4/Q8) quantizes the **whole model** — transformer, text encoder,
/// and VAE (group_size 64) — after the dense load, matching the mflux fork's `nn.quantize` over
/// every quantizable Linear (plus the text encoder's token Embedding) so a Q4/Q8 consumer gets the
/// full memory saving and fork-matching output (sc-2532). An fp32 precision override is not wired
/// (the validated dense path is bf16) and is rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, residency) = load_residency(spec, MODEL_ID, PRECISION_MSG, FILE_MSG)?;
    let loaded_tier = crate::memory_strategy::loaded_tier(spec, MODEL_ID)?;
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer,
        residency,
        memory_strategy: crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?,
        loaded_tier,
    }))
}

#[derive(Clone)]
enum ComfyUiSource {
    Components {
        transformer: std::path::PathBuf,
        text_encoder: std::path::PathBuf,
        vae: std::path::PathBuf,
    },
    Checkpoint(std::path::PathBuf),
}

impl ComfyUiSource {
    fn load_text(&self) -> Result<mlx_gen::weights::Weights> {
        match self {
            Self::Components { text_encoder, .. } => {
                mlx_gen::weights::Weights::from_file_with_fp8(text_encoder)
            }
            Self::Checkpoint(path) => crate::comfyui::split_combined_text(path),
        }
    }

    fn load_heavy(&self) -> Result<(mlx_gen::weights::Weights, mlx_gen::weights::Weights)> {
        match self {
            Self::Components {
                transformer, vae, ..
            } => Ok((
                mlx_gen::weights::Weights::from_file_with_fp8(transformer)?,
                mlx_gen::weights::Weights::from_file_with_fp8(vae)?,
            )),
            Self::Checkpoint(path) => crate::comfyui::split_combined_heavy(path),
        }
    }
}

fn build_comfyui_generator(
    source: ComfyUiSource,
    tokenizer_root: &Path,
) -> Result<Box<dyn Generator>> {
    let text_source = source.clone();
    let heavy_source = source;
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(tokenizer_root)?,
        // A community single-file / three-file load has no snapshot component tree, so the contract's
        // asset facts are the truthful zero (see `memory_strategy::asset_facts`).
        memory_strategy: crate::memory_strategy::memory_strategy_contract(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::File(std::path::PathBuf::from(
                "comfyui-in-place",
            ))),
        )?,
        loaded_tier: mlx_gen::gen_core::MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        residency: Residency::request_scoped(
            move |streamable| {
                if streamable {
                    return Err(Error::Unsupported(
                        "Z-Image ComfyUI in-memory weights cannot stream transformer blocks".into(),
                    ));
                }
                loader::load_text_encoder_from_weights(crate::comfyui::normalize_fp8(
                    text_source.load_text()?,
                    "z-image text encoder",
                )?)
            },
            move |_use_pid, streamable| {
                if streamable {
                    return Err(Error::Unsupported(
                        "Z-Image ComfyUI in-memory weights cannot stream transformer blocks".into(),
                    ));
                }
                let (transformer_weights, vae_weights) = heavy_source.load_heavy()?;
                let transformer =
                    loader::load_transformer_from_weights(crate::comfyui::remap_dit(
                        crate::comfyui::normalize_fp8(transformer_weights, "z-image transformer")?,
                    )?)?;
                let vae = loader::load_vae_from_weights(crate::comfyui::remap_vae(
                    crate::comfyui::normalize_fp8(vae_weights, "z-image VAE")?,
                )?)?;
                Ok(ZImageHeavyOwned {
                    transformer,
                    vae,
                    pid: None,
                })
            },
        ),
    }))
}

/// Load three in-place ComfyUI Z-Image component files on MLX.
pub fn load_from_comfyui_components(
    transformer_file: impl AsRef<Path>,
    text_encoder_file: impl AsRef<Path>,
    vae_file: impl AsRef<Path>,
    tokenizer_root: impl AsRef<Path>,
) -> Result<Box<dyn Generator>> {
    build_comfyui_generator(
        ComfyUiSource::Components {
            transformer: transformer_file.as_ref().to_owned(),
            text_encoder: text_encoder_file.as_ref().to_owned(),
            vae: vae_file.as_ref().to_owned(),
        },
        tokenizer_root.as_ref(),
    )
}

/// Load one fused community Z-Image checkpoint on MLX.
pub fn load_from_comfyui_checkpoint(
    checkpoint_file: impl AsRef<Path>,
    tokenizer_root: impl AsRef<Path>,
) -> Result<Box<dyn Generator>> {
    build_comfyui_generator(
        ComfyUiSource::Checkpoint(checkpoint_file.as_ref().to_owned()),
        tokenizer_root.as_ref(),
    )
}

/// Build the tokenizer + request-scoped [`Residency`] seam for a non-control Z-Image generator.
/// `LoadSpec::offload_policy` is deliberately ignored: both values retain loaders and defer component
/// loading until a request carries [`GenerationMemory::stage_residency`].
///
/// `pub(crate)` and parameterized by `model_id` + the two per-id error strings (precision override /
/// single-file rejection) so the **base** `z_image` sibling ([`crate::model_base`]) shares the
/// identical policy routing rather than re-deriving it (sc-11124, F-172 — before which the base
/// always loaded `Resident`, silently OOMing a fit-gated Sequential request). The tier guard and the
/// F-181 warn name the actual variant via `model_id`.
pub(crate) fn load_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
    file_msg: &'static str,
) -> Result<(TextTokenizer, Residency<TextEncoder, ZImageHeavyOwned>)> {
    // Precision + snapshot-dir guard up front for BOTH policies (fail fast), then the always-warm
    // tokenizer; the heavy component dispatch is the shared [`build_residency`] seam below.
    let root = resolve_precision_and_root(spec, precision_msg, file_msg)?;
    let requantize_bits = if let Some(q) = spec.quantize {
        // F-009 (sc-12461): run the tier guard for BOTH residency policies, before any component
        // load — a Q4 request over a pre-quantized Q8 turnkey hard-errors here instead of silently
        // serving Q8 (`quantize()` is a no-op on packed weights). Before this fix only the
        // Sequential warn gate below evaluated it, so the DEFAULT `Resident` load skipped the guard
        // entirely; `load_heavy` re-checks for the Sequential per-generate reload path.
        mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), model_id)?
            .then_some(q.bits())
    } else {
        None
    };
    let tokenizer = loader::load_tokenizer(root)?;
    let mut residency = build_residency(spec, model_id, precision_msg, file_msg)?;
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

/// Request-scoped residency builder shared by every Z-Image variant. Both loader closures receive the
/// request's `streamable` component form; phase-level staging is a separate argument to
/// [`Residency::run_staged_request_scoped`].
pub(crate) fn build_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
    file_msg: &'static str,
) -> Result<Residency<TextEncoder, ZImageHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Ok(Residency::request_scoped(
        move |streamable| {
            let root = resolve_precision_and_root(&spec_text, precision_msg, file_msg)?;
            load_text_encoder_only(root, spec_text.quantize, streamable)
        },
        move |use_pid, streamable| {
            let root = resolve_precision_and_root(&spec_heavy, precision_msg, file_msg)?;
            load_heavy(&spec_heavy, root, use_pid, streamable, model_id)
        },
    ))
}

/// The `z_image_turbo` precision-override / single-file rejection messages, shared by the
/// [`build_residency`] dispatch and the `Sequential` [`resolve_precision_and_root`] guard.
const PRECISION_MSG: &str = "z_image_turbo: only dense bf16 is wired in the Rust port; the text \
     encoder already runs f32 internally (drop the precision override)";
const FILE_MSG: &str = "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ \
     transformer/ vae/), not a single .safetensors file";

/// Precision guard + snapshot-dir resolution (rejecting a single-file source), shared by
/// [`build_residency`]'s per-phase loaders (sc-10839, sc-11126).
fn resolve_precision_and_root<'a>(
    spec: &'a LoadSpec,
    precision_msg: &str,
    file_msg: &str,
) -> Result<&'a Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(precision_msg.into()));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(file_msg.into())),
    }
}

/// Load the Qwen text encoder (+ optional whole-model Q4/Q8), the phase-A component dropped first
/// under `Sequential`. Q4/Q8 is `group_size 64` over every quantizable Linear + the token Embedding
/// (sc-2532); factored out so the `Resident` and `Sequential` paths build byte-identical encoders.
/// `streamable` builds the rung-4 text-encoder stream (SC-15794). SC-15998 derives it from the
/// independent [`mlx_gen::LoadShape::DeferredMaterialization`] request, not from phase-level residency.
fn load_text_encoder_only(
    root: &Path,
    quant: Option<Quant>,
    streamable: bool,
) -> Result<TextEncoder> {
    let mut text_encoder = if streamable {
        loader::load_text_encoder_streamable(root)?
    } else {
        loader::load_text_encoder(root)?
    };
    if let Some(q) = quant {
        text_encoder.quantize(q.bits())?;
    }
    Ok(text_encoder)
}

/// Load the heavy render-phase components — DiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), VAE
/// (+ Q4/Q8), and the optional PiD overlay — everything but the text encoder. Factored so the
/// `Sequential` path loads these AFTER the encoder is dropped (bounding peak to
/// `max(text-encoder, DiT+VAE)`). Quantize-then-adapters order matches the pre-sc-10839
/// resident composition; the components are independent of the text encoder (separate weight files,
/// deterministic RNG-free quant), so the `Resident` composition below is byte-identical.
fn load_heavy(
    spec: &LoadSpec,
    root: &Path,
    load_pid: bool,
    streamable: bool,
    model_id: &str,
) -> Result<ZImageHeavyOwned> {
    // F-009 (sc-12461): the tier guard runs here too, BEFORE the dense loads, so it fires on both
    // residency policies — `Resident` eager-loads through here at load time and `Sequential`
    // re-loads through here on every generate. A requested-vs-packed mismatch (e.g. Q4 over a
    // pre-quantized Q8 turnkey) hard-errors instead of falling through to the no-op `quantize()`
    // below and silently serving the packed tier. On a matching packed turnkey the quantizes below
    // are documented no-ops (`AdaptableLinear::quantize` on a packed base); on a dense snapshot
    // they do the load-time quant — either way the request stands, so no gating on the bool.
    if let Some(q) = spec.quantize {
        mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), model_id)?;
    }
    let mut transformer = loader::load_transformer_with_stream(root, streamable)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): applied after quantization, as forward-time residuals over the
    // (possibly quantized) base — fork-faithful (the fork applies adapters in its initializer over
    // the quantized model). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    // SC-15754: capture what the resident blocks ended up carrying — AFTER quantization and adapters,
    // so a streamed block replays the final state rather than re-deriving it. A no-op when rung 4 is
    // unarmed or no adapter was installed.
    transformer.capture_block_adapters();
    // Optional PiD decoder overlay (epic 7840, sc-7846): Z-Image is the Flux1 latent space, so it
    // reuses the `flux` PiD student (the `zimage-turbo` registry alias). Loaded only when the spec
    // carries `pid` AND this generate uses it (`load_pid`, F-177) — the Resident path passes `true`
    // (loaded once, reused), the Sequential path passes `req.use_pid` so a non-PiD generate skips
    // the student + its Gemma caption encoder entirely. The native VAE decode path is untouched.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(ZImageHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

// Hand-written rather than `impl_generator!` because this provider also implements the shared
// memory-strategy hooks (SC-15449 / SC-15615); `descriptor` / `validate` / `generate` are the identical
// plain delegation the macro would have emitted.
impl Generator for ZImageTurbo {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(self.descriptor.id, &self.descriptor.capabilities, req).map_err(Into::into)
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

impl ZImageTurbo {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (encode → drop the text encoder under `Sequential` → load the
    /// DiT/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // sc-13571 / GitHub #1658: the DiT-dropping staged decode. Under `Sequential` (the small-Mac
        // fit-gate signal) `run_staged` frees the DiT after denoise and before the (tiled) VAE decode,
        // so the decode peak excludes the ~3.2 GiB DiT; under `Resident` nothing is shed. `tiling` bounds
        // the decode transient on the same small-Mac signal ([`pipeline::decode_tiling`]).
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
            req.use_pid,
            on_progress,
            // ── Phase A: prompt → cap_feats. txt2img runs the DiT in bf16 (the parity-proven path);
            // img2img matches the fork's f32 init latents, so keep cap f32. PARITY-BF16 (sc-2609):
            // round the text embeddings to bf16 to match the fork's golden; f32 is sharper — flip to
            // f32 once parity is not the goal.
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
                    Ok(cap.as_dtype(Dtype::Bfloat16)?)
                }
            },
            // Materialize `cap` while the encoder is still alive (Sequential only) — MLX is lazy, so an
            // un-evaluated `cap` keeps the encoder referenced through the graph and the drop frees nothing.
            |cap| match cap {
                Some(cap) => Ok(mlx_rs::transforms::eval([cap])?),
                None => Ok(()),
            },
            // ── Phase B (denoise): heavy bundle + cap → (evaluated latents, minted PiD decoder). The
            // PiD decoder is minted here (owned, no borrow of `heavy.pid`) so it survives the shed.
            |heavy: &ZImageHeavyOwned, cap, on_progress| {
                pipeline::calibration_fault(
                    req,
                    mlx_gen::gen_core::MemoryPhase::Denoise,
                    MODEL_ID,
                )?;
                // Static shift=3.0 schedule (the model's scheduler_config.json), resolution- and
                // seed-independent. An unset `req.scheduler` keeps this native schedule byte-exact
                // (epic 7114 N1); a curated name re-shapes the σ schedule over the same `shift=3.0`.
                let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
                let resolved_sigmas = resolve_flow_schedule(
                    req.scheduler.as_deref(),
                    SCHEDULE_SHIFT.ln(),
                    steps,
                    &native.sigmas,
                );
                // PiD decode overlay (sc-7846) + `from_ldm` early-stop (sc-8048): mint a decoder when
                // `use_pid` is set AND an overlay was loaded, else `None` → the native VAE. The
                // `(capture_sigma, keep)` fold truncates the denoise to the achieved-σ step; the clean
                // path yields `(0.0, len())` → full schedule, byte-identical.
                let (capture_sigma, keep) =
                    flow_capture_for_request(req, &resolved_sigmas, start_step);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid.as_ref(),
                    req,
                    base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let scheduler = FlowMatchEuler::from_sigmas(resolved_sigmas[..keep].to_vec())?;

                // VAE-encode the init image once (img2img): constant across the count loop (F-034).
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
                        denoise_with_progress_and_preview(
                            &heavy.transformer,
                            &scheduler,
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            start_step,
                            attention_budget,
                            block_window,
                            &req.cancel,
                            &req.preview,
                            op,
                        )
                    },
                )?;
                Ok((latents, pid_decoder))
            },
            // Materialize the latents so the DiT is no longer held via the lazy graph, then it is shed.
            |(latents, _pid): &(Vec<_>, Option<PidDecoder>)| {
                Ok(mlx_rs::transforms::eval(latents.iter())?)
            },
            // ── Phase C (decode): light (VAE) view + latents → images. Tiled under `Sequential`.
            |view, (latents, pid_decoder), on_progress| {
                pipeline::calibration_fault(req, mlx_gen::gen_core::MemoryPhase::Decode, MODEL_ID)?;
                let images = pipeline::decode_batch(
                    view.vae,
                    pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
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

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
/// Required divisor for requested image dims: VAE downsample (8) × transformer patch (2). Exposed as
/// the pinned-engine stride SceneWorks ties each advertised Z-Image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`. `validate_request` enforces exactly this value, so the const
/// cannot drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // Shared capability floor (F-030): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. The hand-rolled copy validated NONE of sampler/scheduler/true_cfg — a typo'd
    // sampler silently fell back to Euler in `run_flow_sampler`. Delegating to core (like Kolors,
    // F-132) rejects it. `id` is threaded from each of the four registered variants so the error
    // strings name the actual model instead of a hardcoded `z_image_turbo:` (F-089a). The `?` keeps
    // the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(id, req)?;

    // Z-Image-specific checks layered on top of the shared floor:
    // F-096: reject a whitespace-only prompt too (`"   "` renders an effectively unconditioned image),
    // not just the empty string — the boogu F-146 `trim()` fix that didn't travel back to the crate the
    // empty-prompt class originated from.
    if req.prompt.trim().is_empty() {
        return Err(mlx_gen::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    // The pipeline needs dims divisible by VAE downsample (8) × patch (2) = 16. A non-multiple either
    // blows up deep in `patchify`'s reshape with a cryptic mlx error, or truncates in `create_noise`
    // and silently returns a smaller image than requested (F-033) — reject it clearly at the boundary.
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(mlx_gen::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {SIZE_MULTIPLE} (VAE 8 × patch 2)",
            req.width, req.height
        )));
    }
    Ok(())
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// sc-16195 Apple-Silicon warm sweep: q4 peaked at 14.043 GiB at 1024². Rounded upward
/// to a 14.05 GiB family anchor; activations remain bf16 across weight tiers.
pub const ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: MODEL_ID,
        anchor: ActivationMemoryAnchor {
            bytes_1024: 15_086_072_628,
        },
    };

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

    #[test]
    fn descriptor_is_z_image_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        // An empty prompt must surface as a typed error (it would otherwise panic in encode via
        // `as_slice` on the size-0 token array — F-001).
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(MODEL_ID, &caps, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "got: {err}");

        // F-096: a whitespace-only prompt is also rejected (it would otherwise render an effectively
        // unconditioned image) — the boogu `trim()` fix travelling back to z-image.
        let ws = GenerationRequest {
            prompt: "   \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &ws)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "whitespace-only prompt got: {err}");
    }

    #[test]
    fn validate_rejects_guidance_and_bad_size() {
        let caps = descriptor().capabilities;
        // guidance on a distilled model (non-empty prompt so the empty-prompt guard doesn't mask it).
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // out-of-range size.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // a plain valid request passes.
        req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
    }

    #[test]
    fn validate_rejects_zero_steps() {
        // F-032: an explicit `steps = 0` builds a degenerate 1-element schedule; img2img then indexes
        // `sigmas[init_time_step]` (>= 1) out of bounds (process abort) and txt2img silently decodes
        // pure noise. Reject it at the boundary; `None` (default) and any positive count still pass.
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");

        // `None` falls back to DEFAULT_STEPS and a positive count both pass.
        for steps in [None, Some(1), Some(20)] {
            let ok = GenerationRequest {
                prompt: "a fox".into(),
                steps,
                ..Default::default()
            };
            assert!(
                validate_request(MODEL_ID, &caps, &ok).is_ok(),
                "steps={steps:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_non_multiple_of_16_size() {
        // F-033: in-range dims that aren't a multiple of 16 must be rejected at the boundary, not
        // crash in patchify (1000×1000) or silently truncate (257×257).
        // sc-12612: pin the exported stride so it cannot drift from the check SceneWorks ties to.
        assert_eq!(SIZE_MULTIPLE, 16);
        let caps = descriptor().capabilities;
        for (w, h) in [(1000, 1000), (257, 257), (512, 520)] {
            let req = GenerationRequest {
                prompt: "a fox".into(),
                width: w,
                height: h,
                ..Default::default()
            };
            let err = validate_request(MODEL_ID, &caps, &req)
                .unwrap_err()
                .to_string();
            assert!(err.contains("multiple of 16"), "{w}x{h} got: {err}");
        }
        // A multiple-of-16 in-range request still passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 512,
            height: 768,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &ok).is_ok());
    }

    #[test]
    fn validate_rejects_typoed_sampler_and_scheduler() {
        // F-030: the hand-rolled copy never validated sampler/scheduler, so a typo'd sampler silently
        // fell back to Euler in `run_flow_sampler`. Delegating to the floor now rejects an
        // un-advertised name as a typed Unsupported gap. The error string threads the actual model id.
        let caps = descriptor().capabilities;
        let bad_sampler = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("eular".into()), // typo of "euler"
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &bad_sampler).unwrap_err();
        assert!(
            matches!(err, mlx_gen::Error::Unsupported(_)),
            "typo'd sampler must be a typed Unsupported gap, got {err:?}"
        );
        assert!(
            err.to_string().contains(MODEL_ID),
            "error must name the model id, got: {err}"
        );
        let bad_scheduler = GenerationRequest {
            prompt: "a fox".into(),
            scheduler: Some("not_a_scheduler".into()),
            ..Default::default()
        };
        assert!(matches!(
            validate_request(MODEL_ID, &caps, &bad_scheduler).unwrap_err(),
            mlx_gen::Error::Unsupported(_)
        ));
        // An advertised curated sampler passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("euler".into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &ok).is_ok());
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![mlx_gen::Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        // Z-Image is a multi-component snapshot, not a single safetensors file.
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        // `Box<dyn Generator>` isn't Debug, so use `.err()` rather than `unwrap_err()`.
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_quantization_spec() {
        // Q4/Q8 is wired (whole model: transformer + text encoder + VAE); a quant spec must get
        // past the load entry point and fail later on the missing snapshot, not on quantization
        // being unsupported.
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }

    // SC-15806: Z-Image construction is request-scoped. Both legacy load-policy values retain the
    // same loaders and defer component loading; GenerationMemory chooses the lifecycle per request.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/z-image-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_defers_for_both_legacy_offload_values() {
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let res = build_residency(
                &missing_snapshot_spec(policy),
                MODEL_ID,
                PRECISION_MSG,
                FILE_MSG,
            )
            .unwrap_or_else(|error| {
                panic!("{policy:?} must defer and ignore the missing snapshot: {error}")
            });
            assert!(
                res.with_resident_parts(|_, _| ()).unwrap().is_none(),
                "{policy:?} must begin with no warm request-scoped pair"
            );
        }
    }

    // ── F-009 (sc-12461): the tier-mismatch guard must fire on the DEFAULT `Resident` policy, not
    // just `Sequential`. Before the fix, `needs_load_time_quant` only ran behind the Sequential
    // F-181 warn gate, so a Resident Q4 request over a pre-quantized Q8 turnkey silently served Q8
    // (`quantize()` is a no-op on packed weights). Weight-free: the fixture is only the packed
    // `transformer/config.json` marker, and the guard errors before any component weights load.
    #[test]
    fn tier_mismatch_errors_on_resident_and_sequential_load() {
        let tmp = tempfile::tempdir().unwrap();
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let root = loader::packed_snapshot_fixture(&tmp, "model-load", 8);
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_quant(mlx_gen::Quant::Q4)
                .with_offload_policy(policy);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(
                err.contains("pre-quantized Q8"),
                "policy {policy:?}: Q4 over a packed Q8 turnkey must hard-error, got: {err}"
            );
            assert!(
                err.contains(MODEL_ID),
                "policy {policy:?}: the error must name the model id, got: {err}"
            );
            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn load_heavy_runs_tier_guard_before_weights() {
        let tmp = tempfile::tempdir().unwrap();
        // F-009 (sc-12461): the heavy loader itself re-checks the tier guard — this is the seam the
        // Sequential path re-loads through on every generate, and the defense-in-depth for any
        // composition that reaches `load_heavy` without the `load_residency` entry guard. The
        // fixture has no weights at all, so reaching the transformer load would fail with a
        // missing-weights error instead — asserting on the tier message proves the guard runs first.
        let root = loader::packed_snapshot_fixture(&tmp, "model-heavy", 8);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(mlx_gen::Quant::Q4);
        let err = load_heavy(&spec, &root, false, false, MODEL_ID)
            .err()
            .expect("expected a tier-mismatch error")
            .to_string();
        assert!(err.contains("pre-quantized Q8"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn matching_packed_tier_passes_the_guard() {
        let tmp = tempfile::tempdir().unwrap();
        // A Q8 request over a Q8-packed turnkey must get PAST the guard (and fail later on the
        // missing component weights, not on the tier) — the guard rejects mismatches only.
        let root = loader::packed_snapshot_fixture(&tmp, "model-match", 8);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            !err.contains("pre-quantized"),
            "a matching packed tier must not trip the mismatch guard, got: {err}"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
