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
    curated_sampler_names, curated_scheduler_names, ActivationMemoryAnchor, Capabilities,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, LatentDecoder,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Residency, Result,
    StagedHeavy, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidDecoder, PidEngine};
use std::path::Path;

use crate::loader;
use crate::pipeline::{self, encode_init_latents, init_time_step};
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

/// Perturbed-Attention Guidance (PAG) capability descriptor for Z-Image Turbo.
pub fn turbo_pag_capability() -> gen_core::PagCapability {
    gen_core::PagCapability {
        num_blocks: 30,
        block_domain: std::borrow::Cow::Borrowed("layers"),
        default_layers: vec![15],
        default_start: 0.0,
        default_stop: 0.75,
        recommended_scale: 1.2,
        max_scale: gen_core::DEFAULT_PAG_MAX_SCALE,
    }
}

/// Primary typed route declaration for Z-Image-turbo (`REG-1`, `AP5`).
pub(crate) fn route_turbo() -> gen_core::RouteRecord {
    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new("qwen3", Some(512))],
                encoder_contract: Some(crate::active_encoder_contract()),
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                // img2img reference; ControlNet is a separate variant (sc-2349).
                kinds: vec![ConditioningKind::Reference],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: Vec::new(),
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                samplers: curated_sampler_names(),
                schedulers: curated_scheduler_names(),
                guidance_methods: Vec::new(),
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("z_image_dit")
                    .with_low_res_start()
                    .with_pag(turbo_pag_capability()),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&mlx_gen::gen_core::FLUX1_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: Some(16),
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
                        ["qwen3"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["z_image_dit"],
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
                supported_quants: &[Quant::Q4, Quant::Q8],
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
            seed_scheme: gen_core::SeedScheme::BackendGlobalSeeded,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(DEFAULT_STEPS),
            guidance: Some(1.0),
            sampler: Some("euler"),
            schedule: Some("normal"),
            shift: Some(SCHEDULE_SHIFT),
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings: gen_core::RouteBindings::generator(descriptor, |spec| {
            load(spec).map_err(Into::into)
        })
        .with_footprint(component_footprint)
        .with_render_plan(compile_turbo)
        .with_activation_memory(ACTIVATION_MEMORY_REGISTRATION.anchor)
        .with_memory_registration(MEMORY_REGISTRATION)
        .with_memory_fixture(gen_core::mlx_memory_contract_surface_specs, |spec| {
            crate::memory_strategy::weights_free_memory_strategy_contract(MODEL_ID, spec)
        })
        .with_memory_behavior_registration(MEMORY_BEHAVIOR_REGISTRATION),
    }
}

/// Z-Image-turbo's identity + capabilities — constructible without loading weights (registry
/// introspection).
pub fn descriptor() -> ModelDescriptor {
    route_turbo().compute_descriptor()
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
    loaded_spec: Option<LoadSpec>,
    adapters: Vec<mlx_gen::AdapterSpec>,
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
    let (tokenizer, residency) = load_residency(spec, MODEL_ID, PRECISION_ERROR, FILE_ERROR)?;
    let loaded_tier = crate::memory_strategy::loaded_tier(spec, MODEL_ID)?;
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer,
        residency,
        memory_strategy: crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?,
        loaded_tier,
        loaded_spec: Some(spec.clone()),
        adapters: spec.adapters.clone(),
    }))
}

#[derive(Clone)]
enum ComfyUiSource {
    Components {
        transformer: Box<gen_core::PinnedWeightsFile>,
        text_encoder: Box<gen_core::ValidatedEncoderSource>,
        vae: Box<gen_core::PinnedWeightsFile>,
    },
    Checkpoint {
        checkpoint: Box<gen_core::PinnedWeightsFile>,
        tokenizer: Option<Box<gen_core::ValidatedTokenizerSource>>,
    },
}

impl ComfyUiSource {
    fn load_tokenizer(&self) -> Result<TextTokenizer> {
        match self {
            Self::Components { text_encoder, .. } => loader::load_validated_tokenizer(text_encoder),
            Self::Checkpoint { tokenizer, .. } => {
                loader::load_tokenizer_from_receipt(tokenizer.as_deref().ok_or_else(|| {
                    Error::Unsupported(
                        "Z-Image fused checkpoint has no retained tokenizer receipt".into(),
                    )
                })?)
            }
        }
    }

    fn load_text(&self) -> Result<mlx_gen::weights::Weights> {
        let weights = match self {
            Self::Components { text_encoder, .. } => text_encoder.read_unchanged(|source| {
                let WeightsSource::File(path) = source else {
                    return Err(Error::Unsupported(
                        "Z-Image ComfyUI text encoder must remain one File source".into(),
                    ));
                };
                let weights = mlx_gen::weights::Weights::from_file_with_fp8(path)?;
                weights.materialize()?;
                Ok(weights)
            }),
            Self::Checkpoint { checkpoint, .. } => checkpoint.read_unchanged(|path| {
                let weights = crate::comfyui::split_combined_text(path)?;
                weights.materialize()?;
                Ok(weights)
            }),
        }?;
        Ok(weights)
    }

    fn load_heavy(&self) -> Result<(mlx_gen::weights::Weights, mlx_gen::weights::Weights)> {
        let weights = match self {
            Self::Components {
                transformer, vae, ..
            } => {
                let transformer = transformer.read_unchanged(|path| {
                    let weights = mlx_gen::weights::Weights::from_file_with_fp8(path)?;
                    weights.materialize()?;
                    Ok::<_, Error>(weights)
                })?;
                let vae = vae.read_unchanged(|path| {
                    let weights = mlx_gen::weights::Weights::from_file_with_fp8(path)?;
                    weights.materialize()?;
                    Ok::<_, Error>(weights)
                })?;
                Ok::<_, Error>((transformer, vae))
            }
            Self::Checkpoint { checkpoint, .. } => checkpoint.read_unchanged(|path| {
                let (transformer, vae) = crate::comfyui::split_combined_heavy(path)?;
                transformer.materialize()?;
                vae.materialize()?;
                Ok((transformer, vae))
            }),
        }?;
        Ok(weights)
    }
}

fn build_comfyui_generator(source: ComfyUiSource) -> Result<Box<dyn Generator>> {
    let tokenizer = source.load_tokenizer()?;
    let text_source = source.clone();
    let heavy_source = source;
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer,
        adapters: Vec::new(),
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
        loaded_spec: None,
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
    let text_encoder_file = text_encoder_file.as_ref();
    let text_encoder = crate::active_encoder_contract().validate_comfyui_source_against_base(
        &WeightsSource::File(text_encoder_file.to_owned()),
        tokenizer_root.as_ref(),
    )?;
    build_comfyui_generator(ComfyUiSource::Components {
        transformer: Box::new(gen_core::PinnedWeightsFile::pin(transformer_file.as_ref())?),
        text_encoder: Box::new(text_encoder),
        vae: Box::new(gen_core::PinnedWeightsFile::pin(vae_file.as_ref())?),
    })
}

/// Load one fused community Z-Image checkpoint on MLX.
pub fn load_from_comfyui_checkpoint(
    checkpoint_file: impl AsRef<Path>,
    tokenizer_root: impl AsRef<Path>,
) -> Result<Box<dyn Generator>> {
    let checkpoint = gen_core::PinnedWeightsFile::pin(checkpoint_file.as_ref())?;
    let tokenizer = checkpoint.read_unchanged(|path| {
        crate::active_encoder_contract().validate_embedded_comfyui_file_against_base(
            path,
            &[
                "conditioner.embedders.0.transformer.",
                "text_encoders.qwen3_4b.transformer.",
                "text_encoder.",
            ],
            tokenizer_root.as_ref(),
        )
    })?;
    build_comfyui_generator(ComfyUiSource::Checkpoint {
        checkpoint: Box::new(checkpoint),
        tokenizer: Some(Box::new(tokenizer)),
    })
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
    let text_encoder_source = crate::active_encoder_contract().source_for_load(spec, root)?;
    let effective_quant_bits = crate::memory_strategy::loaded_tier(spec, model_id)?
        .quant
        .map(Quant::bits);
    let text_encoder_quant_bits =
        text_encoder_source.load_time_quant_bits(effective_quant_bits, model_id)?;
    let tokenizer = loader::load_validated_tokenizer(&text_encoder_source)?;
    let mut residency = build_residency_with_source(
        spec,
        model_id,
        precision_msg,
        file_msg,
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

/// Request-scoped residency builder shared by every Z-Image variant. Both loader closures receive the
/// request's `streamable` component form; phase-level staging is a separate argument to
/// [`Residency::run_staged_request_scoped`].
fn build_residency_with_source(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
    file_msg: &'static str,
    text_encoder_source: gen_core::ValidatedEncoderSource,
    text_encoder_quant_bits: Option<i32>,
) -> Result<Residency<TextEncoder, ZImageHeavyOwned>> {
    let spec_heavy = spec.clone();
    Ok(Residency::request_scoped(
        move |streamable| {
            load_text_encoder_only(&text_encoder_source, text_encoder_quant_bits, streamable)
        },
        move |use_pid, streamable| {
            let root = resolve_precision_and_root(&spec_heavy, precision_msg, file_msg)?;
            load_heavy(&spec_heavy, root, use_pid, streamable, model_id)
        },
    ))
}

/// The `z_image_turbo` precision-override / single-file rejection messages, shared by the
/// [`build_residency`] dispatch and the `Sequential` [`resolve_precision_and_root`] guard.
pub(crate) const PRECISION_ERROR: &str =
    "z_image_turbo: only dense bf16 is wired in the Rust port; the text \
     encoder already runs f32 internally (drop the precision override)";
pub(crate) const FILE_ERROR: &str =
    "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ \
     transformer/ vae/), not a single .safetensors file";

/// Precision guard + snapshot-dir resolution (rejecting a single-file source), shared by
/// [`build_residency`]'s per-phase loaders (sc-10839, sc-11126).
pub(crate) fn resolve_precision_and_root<'a>(
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
    let mut transformer = loader::load_transformer_with_stream(root, streamable, streamable)?;
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
        let plan = compile_turbo_trajectory(req)?;
        self.generate_planned(req, &plan, None, on_progress)?
            .output
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{MODEL_ID}: generator returned no output"))
            })
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        let plan = compile_turbo_trajectory(req)?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        let plan = compile_turbo_trajectory(req)?;
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.render_planned(req, plan, scope, on_progress)
                .map_err(Into::into)
        })
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::image_execution::complete_image_report_attempt(
            &self.descriptor,
            &req.cancel,
            || {
                mlx_gen::image_execution::applied_adapter_controls(
                    "ZImageTurbo.applied_adapter_controls",
                    &self.adapters,
                );
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.render_planned(req, plan, scope, &mut progress)
                    .map_err(Into::into)
            },
        )
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
    /// The rich-`Result` body behind [`Generator::generate`] and [`Generator::generate_planned`].
    ///
    /// The staged residency lifecycle (encode → drop the text encoder under `Sequential` → load the
    /// DiT/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run_staged_request_scoped`] seam (sc-11125) and [`gen_core::RenderRun`] (`SW-6947`).
    fn render_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        if let Some(spec) = &self.loaded_spec {
            plan.check_load(&gen_core::RenderLoadBinding::from_spec(spec))?;
        }
        let resolved_pag = self.descriptor.capabilities.resolve_pag(MODEL_ID, req)?;
        mlx_gen::ios_quantum::diagnostic_metadata(
            MODEL_ID,
            "request",
            &[
                ("schema", 1),
                ("serial_count", req.count as u64),
                ("forward_rows", 1),
                ("requested_width", req.width as u64),
                ("requested_height", req.height as u64),
                ("requested_steps", req.steps.unwrap_or(DEFAULT_STEPS) as u64),
            ],
        );

        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let guidance = run.plan().guidance();
        let cfg_on = guidance != 1.0;
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = run.plan().base_seed();

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => {
                if req.features.low_res_start.is_some() {
                    gen_core::resolve_img2img_init_time_step(steps, strength)?
                } else {
                    init_time_step(steps, strength)
                }
            }
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
        mlx_gen::image_execution::policy_debug("stage_residency", &stage_residency);
        mlx_gen::image_execution::policy_debug("streamable", &streamable);
        mlx_gen::image_execution::policy_debug("decode_tiling", &tiling);
        mlx_gen::image_execution::policy_debug("attention_budget", &attention_budget);
        mlx_gen::image_execution::policy_debug("transformer_window", &block_window);
        mlx_gen::image_execution::policy_debug(
            "encoder_window",
            &encoder_window.map(|scope| scope.window),
        );
        if let Some(pag) = resolved_pag.as_ref() {
            mlx_gen::image_execution::policy_debug("pag", pag);
        }
        let images = self.residency.run_staged_request_scoped(
            stage_residency,
            streamable,
            &req.cancel,
            req.use_pid(),
            on_progress,
            // ── Phase A: prompt → cap_feats. txt2img runs the DiT in bf16 (the parity-proven path);
            // img2img matches the fork's f32 init latents, so keep cap f32. PARITY-BF16 (sc-2609):
            // round the text embeddings to bf16 to match the fork's golden; f32 is sharper — flip to
            // f32 once parity is not the goal.
            |text_encoder: &TextEncoder| {
                run.encode(|| {
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
                    let cap = if is_img2img {
                        cap
                    } else {
                        pipeline::caption_cast_quantum(&cap)?
                    };
                    let negative = if cfg_on {
                        Some(pipeline::encode_uncond(
                            &self.tokenizer,
                            text_encoder,
                            req.negative_prompt.as_deref().unwrap_or(""),
                            encoder_window,
                        )?)
                    } else {
                        None
                    };
                    Ok((cap, negative))
                })
            },
            // Materialize `cap` while the encoder is still alive (Sequential only) — MLX is lazy, so an
            // un-evaluated `cap` keeps the encoder referenced through the graph and the drop frees nothing.
            |encoded| {
                if let Some((cap, negative)) = encoded {
                    match negative {
                        Some(negative) => mlx_rs::transforms::eval([cap, negative])?,
                        None => mlx_rs::transforms::eval([cap])?,
                    }
                }
                Ok(())
            },
            // ── Phase B (denoise): heavy bundle + cap → (evaluated latents, minted PiD decoder). The
            // PiD decoder is minted here (owned, no borrow of `heavy.pid`) so it survives the shed.
            |heavy: &ZImageHeavyOwned, (cap, negative), on_progress| {
                run.begin_rendering()?;
                pipeline::calibration_fault(
                    req,
                    mlx_gen::gen_core::MemoryPhase::Denoise,
                    MODEL_ID,
                )?;
                let capture_sigma = run.plan().capture_sigma();
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid.as_ref(),
                    req,
                    base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;

                // VAE-encode the init image once (img2img) at Phase-0 output dimensions.
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    let (init_w, init_h) = run.plan().phase_output_size(0);
                    Some(encode_init_latents(&heavy.vae, image, init_w, init_h)?)
                } else {
                    None
                };

                let sampler_name = req.sampler.as_deref();
                let sigmas = run.plan().denoise_sigmas();
                let spans = run.plan().denoise_spans();
                let resolved_pag_ref = resolved_pag.as_ref();
                let actual_steps = run.plan().executed_steps() as u32;
                let latents = pipeline::denoise_batch_planned(
                    &run,
                    req.count,
                    clean.as_ref(),
                    on_progress,
                    |latents, seed, op| {
                        pipeline::denoise_cfg_planned(
                            &heavy.transformer,
                            sigmas,
                            &spans,
                            Some(&run),
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            negative.as_ref(),
                            guidance,
                            attention_budget,
                            block_window,
                            &req.cancel,
                            &req.preview,
                            resolved_pag_ref,
                            op,
                        )
                    },
                )?;
                Ok((latents, pid_decoder, actual_steps))
            },
            // Materialize the latents so the DiT is no longer held via the lazy graph, then it is shed.
            |(latents, _pid, _steps): &(Vec<_>, Option<PidDecoder>, u32)| {
                Ok(mlx_rs::transforms::eval(latents.iter())?)
            },
            // ── Phase C (decode): light (VAE) view + latents → images. Tiled under `Sequential`.
            |view, (latents, pid_decoder, actual_steps), on_progress| {
                let report = run.decode_phase(|| -> Result<_> {
                    pipeline::calibration_fault(
                        req,
                        mlx_gen::gen_core::MemoryPhase::Decode,
                        MODEL_ID,
                    )?;
                    let decoder: &dyn LatentDecoder = pid_decoder
                        .as_ref()
                        .map(|d| d as &dyn LatentDecoder)
                        .unwrap_or(view.vae);
                    let images = pipeline::decode_batch(
                        decoder,
                        tiling.as_ref(),
                        latents,
                        &req.cancel,
                        on_progress,
                    )?;
                    let mut report =
                        gen_core::GenerationReport::from_output(GenerationOutput::Images(images));
                    report.pag = resolved_pag
                        .as_ref()
                        .map(|resolved| resolved.execution_report(actual_steps));
                    report.low_res_start =
                        plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
                    Ok(report)
                })?;
                run.finish_rendering();
                Ok(report)
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
    if let Some(low_res) = &req.features.low_res_start {
        if id == crate::model_control::MODEL_ID || id == crate::model_base_control::MODEL_ID {
            let (desc, shift, default_steps) = if id == crate::model_base_control::MODEL_ID {
                (
                    crate::model_base_control::descriptor(),
                    crate::model_base::SCHEDULE_SHIFT,
                    crate::model_base::DEFAULT_STEPS,
                )
            } else {
                (
                    crate::model_control::descriptor(),
                    SCHEDULE_SHIFT,
                    DEFAULT_STEPS,
                )
            };
            let sampling = crate::sampling::resolve_full_with_shift(req, shift, default_steps)?;
            let executed_steps = sampling.sigmas().len().saturating_sub(1);
            low_res.resolve_for_descriptor(&desc, req, executed_steps, SIZE_MULTIPLE)?;
        }
    }

    // Shared capability floor (F-030): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. The hand-rolled copy validated NONE of sampler/scheduler/true_cfg — a typo'd
    // sampler silently fell back to Euler in `run_flow_sampler`. Delegating to core (like Kolors,
    // F-132) rejects it. `id` is threaded from each of the four registered variants so the error
    // strings name the actual model instead of a hardcoded `z_image_turbo:` (F-089a). The `?` keeps
    // the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(id, req)?;
    if req
        .negative_prompt
        .as_ref()
        .is_some_and(|negative| !negative.is_empty())
        && id.starts_with("z_image_turbo")
        && req.guidance.unwrap_or(0.0) == 0.0
    {
        return Err(Error::Msg(format!(
            "{id}: a negative prompt requires CFG different from 0.0"
        )));
    }

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

    if let Some(low_res) = &req.features.low_res_start {
        let (desc, shift, default_steps) = if id == crate::model_base::MODEL_ID {
            (
                crate::model_base::descriptor(),
                crate::model_base::SCHEDULE_SHIFT,
                crate::model_base::DEFAULT_STEPS,
            )
        } else {
            (descriptor(), SCHEDULE_SHIFT, DEFAULT_STEPS)
        };
        let steps = req.steps.unwrap_or(default_steps) as usize;
        let reference = pipeline::resolve_reference(req, id)?;
        let start_step = match reference {
            Some((img, strength)) => {
                if img.width == 0
                    || img.height == 0
                    || img.pixels.len()
                        != (img.width as usize)
                            .saturating_mul(img.height as usize)
                            .saturating_mul(3)
                {
                    return Err(Error::Msg(format!(
                        "{id}: invalid img2img reference dimensions {}x{}",
                        img.width, img.height
                    )));
                }
                gen_core::resolve_img2img_init_time_step(steps, strength)?
            }
            None => 0,
        };
        let sampling = crate::sampling::resolve_full_with_shift(req, shift, default_steps)?;
        let (_, keep) = flow_capture_for_request(req, sampling.sigmas(), start_step);
        let start = start_step.min(keep.saturating_sub(1));
        let executed_steps = keep.saturating_sub(1).saturating_sub(start);
        low_res.resolve_for_descriptor(&desc, req, executed_steps, SIZE_MULTIPLE)?;
    }
    Ok(())
}

/// Compile the weights-free Z-Image trajectory for `desc` (`z_image_turbo` or `z_image`) (`SW-6947`).
pub(crate) fn compile_trajectory_for(
    desc: ModelDescriptor,
    shift: f32,
    default_steps: u32,
    guidance: f32,
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    validate_request(desc.id, &desc.capabilities, req)?;
    let steps = req.steps.unwrap_or(default_steps) as usize;
    let reference = pipeline::resolve_reference(req, desc.id)?;
    let start_step = match reference {
        Some((img, strength)) => {
            if img.width == 0
                || img.height == 0
                || img.pixels.len()
                    != (img.width as usize)
                        .saturating_mul(img.height as usize)
                        .saturating_mul(3)
            {
                return Err(gen_core::Error::Msg(format!(
                    "{}: invalid img2img reference dimensions {}x{}",
                    desc.id, img.width, img.height
                )));
            }
            if req.features.low_res_start.is_some() {
                gen_core::resolve_img2img_init_time_step(steps, strength)?
            } else {
                init_time_step(steps, strength)
            }
        }
        None => 0,
    };
    let sampling = crate::sampling::resolve_full_with_shift(req, shift, default_steps)?;
    let (_capture_sigma, keep) = flow_capture_for_request(req, sampling.sigmas(), start_step);
    let start = start_step.min(keep.saturating_sub(1));
    let sigmas = sampling.sigmas()[start..].to_vec();
    let executed_steps = keep.saturating_sub(start).saturating_sub(1);
    let capture_sigma = *sigmas.get(executed_steps).unwrap_or(&0.0);
    gen_core::RenderPlan::packed_image_trajectory(
        &desc,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 16,
            sampler: sampling.solver().name(),
            guidance,
            sigmas,
            executed_steps,
            capture_sigma,
        },
    )
}

pub fn compile_turbo_trajectory(req: &GenerationRequest) -> gen_core::Result<gen_core::RenderPlan> {
    let guidance = 1.0 + req.guidance.unwrap_or(0.0);
    compile_trajectory_for(descriptor(), SCHEDULE_SHIFT, DEFAULT_STEPS, guidance, req)
}

pub(crate) fn compile_turbo(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_plan_load_spec(spec, MODEL_ID)?;
    compile_turbo_trajectory(req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

pub fn compile_base_trajectory(req: &GenerationRequest) -> gen_core::Result<gen_core::RenderPlan> {
    let guidance = req.guidance.unwrap_or(crate::model_base::DEFAULT_GUIDANCE);
    compile_trajectory_for(
        crate::model_base::descriptor(),
        crate::model_base::SCHEDULE_SHIFT,
        crate::model_base::DEFAULT_STEPS,
        guidance,
        req,
    )
}

pub(crate) fn compile_base(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_plan_load_spec(spec, crate::model_base::MODEL_ID)?;
    compile_base_trajectory(req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

pub(crate) fn compile_control_for(
    desc: ModelDescriptor,
    shift: f32,
    default_steps: u32,
    guidance: f32,
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    validate_request(desc.id, &desc.capabilities, req)?;
    let accepted = desc
        .control_kinds
        .clone()
        .unwrap_or(gen_core::AcceptedControlKinds::Any);
    let mut found_control = false;
    for c in &req.conditioning {
        if let gen_core::Conditioning::Control { kind, .. } = c {
            if !accepted.accepts(kind) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: unsupported control kind {kind:?}",
                    desc.id
                )));
            }
            if found_control {
                return Err(gen_core::Error::Msg(format!(
                    "{}: a single control image is supported",
                    desc.id
                )));
            }
            found_control = true;
        }
    }
    if !found_control {
        return Err(gen_core::Error::Msg(format!(
            "{} requires a Control conditioning (the pose/union skeleton)",
            desc.id
        )));
    }
    let steps = req.steps.unwrap_or(default_steps) as usize;
    let reference = pipeline::resolve_reference(req, desc.id)?;
    let start_step = match reference {
        Some((_, strength)) => init_time_step(steps, strength),
        None => 0,
    };
    let sampling = crate::sampling::resolve_full_with_shift(req, shift, default_steps)?;
    let (_capture_sigma, keep) = flow_capture_for_request(req, sampling.sigmas(), start_step);
    let start = start_step.min(keep.saturating_sub(1));
    let sigmas = sampling.sigmas()[start..].to_vec();
    let executed_steps = keep.saturating_sub(start).saturating_sub(1);
    let capture_sigma = *sigmas.get(executed_steps).unwrap_or(&0.0);
    gen_core::RenderPlan::packed_image_trajectory(
        &desc,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 16,
            sampler: sampling.solver().name(),
            guidance,
            sigmas,
            executed_steps,
            capture_sigma,
        },
    )
}

pub(crate) fn compile_turbo_control(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    let desc = crate::model_control::descriptor();
    validate_request(desc.id, &desc.capabilities, req)?;
    validate_plan_load_spec(spec, crate::model_control::MODEL_ID)?;
    validate_control_load_spec(spec, crate::model_control::MODEL_ID)?;
    let guidance = 1.0 + req.guidance.unwrap_or(0.0);
    compile_control_for(desc, SCHEDULE_SHIFT, DEFAULT_STEPS, guidance, req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

pub(crate) fn compile_base_control(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    let desc = crate::model_base_control::descriptor();
    validate_request(desc.id, &desc.capabilities, req)?;
    validate_plan_load_spec(spec, crate::model_base_control::MODEL_ID)?;
    validate_control_load_spec(spec, crate::model_base_control::MODEL_ID)?;
    let guidance = req.guidance.unwrap_or(crate::model_base::DEFAULT_GUIDANCE);
    compile_control_for(
        desc,
        crate::model_base::SCHEDULE_SHIFT,
        crate::model_base::DEFAULT_STEPS,
        guidance,
        req,
    )
    .map(|plan| plan.bind_load(spec))
    .map(Some)
}

fn validate_control_load_spec(spec: &LoadSpec, id: &str) -> gen_core::Result<()> {
    match &spec.control {
        Some(WeightsSource::File(_)) => Ok(()),
        Some(WeightsSource::Dir(_)) => Err(gen_core::Error::Msg(format!(
            "{id}: LoadSpec::control must be a WeightsSource::File pointing at the ControlNet .safetensors"
        ))),
        None => Err(gen_core::Error::Msg(format!(
            "{id}: LoadSpec::control is required (pass the Fun-Controlnet-Union .safetensors via LoadSpec::with_control)"
        ))),
    }
}

fn validate_plan_load_spec(spec: &LoadSpec, id: &str) -> gen_core::Result<()> {
    if spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Msg(format!(
            "{id}: only dense bf16 is wired in the Rust port; the text encoder already runs f32 \
             internally (drop the precision override)"
        )));
    }
    if req_is_file_without_comfyui(spec) {
        return Err(gen_core::Error::Msg(format!(
            "{id} expects a snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
             not a single .safetensors file"
        )));
    }
    Ok(())
}

fn req_is_file_without_comfyui(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::File(_)) && spec.text_encoder.is_none()
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.as_path(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "z-image component footprint requires a snapshot directory".to_owned(),
            ))
        }
    };
    let selected = crate::active_encoder_contract().source_for_load(spec, root)?;
    let text_encoder = selected.read_unchanged(|source| {
        Ok::<u64, gen_core::Error>(match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => {
                mlx_gen::safetensors_path_bytes(path)
            }
        })
    })?;
    let mut footprint = mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )?;
    footprint.text_encoder = text_encoder;
    Ok(footprint)
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
pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration = mlx_gen::gen_core::profile_memory_registration!(
    crate::memory_strategy::MEMORY_PROFILE,
    MODEL_ID,
    |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec),
);
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration = mlx_gen::gen_core::profile_memory_behavior_registration!(
    crate::memory_strategy::MEMORY_PROFILE,
    MODEL_ID,
    crate::memory_strategy::registered_scope,
);

#[cfg(test)]
mod tests {
    #[test]
    fn turbo_control_negative_prompt_requires_effective_cfg() {
        let descriptor = crate::model_control::descriptor();
        for guidance in [None, Some(0.0), Some(1.5)] {
            let request = GenerationRequest {
                prompt: "p".into(),
                negative_prompt: Some("blurry".into()),
                guidance,
                ..Default::default()
            };
            assert_eq!(
                validate_request(descriptor.id, &descriptor.capabilities, &request).is_ok(),
                guidance == Some(1.5)
            );
        }
    }

    use super::*;
    use mlx_gen::OffloadPolicy;

    #[test]
    fn comfyui_checkpoint_rechecks_retained_pin_before_deferred_text_load() {
        let fixture = tempfile::tempdir().unwrap();
        let checkpoint = fixture.path().join("combined.safetensors");
        std::fs::write(&checkpoint, b"initial checkpoint").unwrap();
        let source = ComfyUiSource::Checkpoint {
            checkpoint: Box::new(gen_core::PinnedWeightsFile::pin(&checkpoint).unwrap()),
            tokenizer: None,
        };

        std::fs::write(&checkpoint, b"replacement checkpoint with different bytes").unwrap();
        let error = source
            .load_text()
            .err()
            .expect("request-scoped ComfyUI load must retain its original checkpoint pin");
        match error {
            Error::Unsupported(reason)
                if reason.starts_with("artifact seal mismatch after load: ") => {}
            Error::Unsupported(reason) => panic!("unexpected artifact-seal reason: {reason}"),
            other => panic!("expected a typed artifact-seal rejection, got: {other:?}"),
        }
    }

    #[test]
    fn comfyui_components_recheck_validated_text_source_before_deferred_load() {
        let fixture = tempfile::tempdir().unwrap();
        let encoder = fixture.path().join("encoder");
        let contract = crate::bounded_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(&encoder, contract).unwrap();
        let encoder_file = encoder.join("model.safetensors");
        let text_encoder = contract
            .validate_comfyui_source(&WeightsSource::File(encoder_file.clone()))
            .unwrap();
        let transformer = fixture.path().join("transformer.safetensors");
        let vae = fixture.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let source = ComfyUiSource::Components {
            transformer: Box::new(gen_core::PinnedWeightsFile::pin(&transformer).unwrap()),
            text_encoder: Box::new(text_encoder),
            vae: Box::new(gen_core::PinnedWeightsFile::pin(&vae).unwrap()),
        };

        std::fs::write(&encoder_file, b"replacement encoder payload").unwrap();
        let error = source
            .load_text()
            .err()
            .expect("request-scoped ComfyUI load must retain the validated encoder receipt");
        let error = error.to_string();
        assert!(error.contains("changed after load"), "{error}");
    }

    #[test]
    fn descriptor_is_z_image_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert!(d.capabilities.supports_guidance);
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
            guidance: Some(1.5),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
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
    fn incomplete_snapshot_spec(policy: OffloadPolicy) -> (tempfile::TempDir, LoadSpec) {
        let snapshot = tempfile::tempdir().expect("snapshot fixture dir");
        gen_core_testkit::write_encoder_contract_fixture(
            &snapshot.path().join("text_encoder"),
            crate::bounded_encoder_contract(),
        )
        .expect("bounded validation-complete encoder and tokenizer fixture");
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.path().to_path_buf()))
            .with_offload_policy(policy);
        (snapshot, spec)
    }

    #[test]
    fn build_residency_defers_for_both_legacy_offload_values() {
        let _guard = crate::scoped_bounded_encoder_contract();
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let (snapshot, spec) = incomplete_snapshot_spec(policy);
            assert!(!snapshot.path().join("transformer").exists());
            assert!(!snapshot.path().join("vae").exists());
            let (_tokenizer, res) = load_residency(&spec, MODEL_ID, PRECISION_ERROR, FILE_ERROR)
                .unwrap_or_else(|error| {
                    panic!("{policy:?} must defer absent heavy components: {error}")
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
        // A Q8 request over a Q8-packed turnkey must get past the exact shared guard — the guard
        // rejects mismatches only. The mismatch tests above exercise the production load dispatch.
        let root = loader::packed_snapshot_fixture(&tmp, "model-match", 8);
        assert!(!mlx_gen::quant::needs_load_time_quant(
            &root,
            "transformer",
            mlx_gen::Quant::Q8.bits(),
            MODEL_ID,
        )
        .unwrap());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn selected_encoder_quant_policy_rejects_packed_mismatch_and_accepts_dense_inheritance() {
        let _guard = crate::scoped_bounded_encoder_contract();
        let contract = crate::bounded_encoder_contract();
        let base = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base.path().join("transformer")).unwrap();
        std::fs::write(
            base.path().join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        gen_core_testkit::write_encoder_contract_tokenizer_fixture(base.path(), contract).unwrap();

        let packed_q4 = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            packed_q4.path(),
            contract,
            Some(4),
        )
        .unwrap();
        let mismatch = LoadSpec::new(WeightsSource::Dir(base.path().to_path_buf()))
            .with_text_encoder(WeightsSource::Dir(packed_q4.path().to_path_buf()));
        let error = load_residency(&mismatch, MODEL_ID, PRECISION_ERROR, FILE_ERROR)
            .err()
            .expect("packed encoder mismatch")
            .to_string();
        assert!(error.contains("pre-quantized Q4"), "{error}");
        assert!(error.contains("model policy is Q8"), "{error}");

        let dense = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_fixture(dense.path(), contract).unwrap();
        let inherited = LoadSpec::new(WeightsSource::Dir(base.path().to_path_buf()))
            .with_text_encoder(WeightsSource::Dir(dense.path().to_path_buf()));
        load_residency(&inherited, MODEL_ID, PRECISION_ERROR, FILE_ERROR).unwrap();
    }

    #[test]
    fn turbo_pag_capability_and_validation_match_contract() {
        let cap = turbo_pag_capability();
        assert_eq!(cap.num_blocks, 30);
        assert_eq!(cap.block_domain, "layers");
        assert_eq!(cap.default_layers, vec![15]);
        assert_eq!(cap.default_start, 0.0);
        assert_eq!(cap.default_stop, 0.75);
        assert_eq!(cap.recommended_scale, 1.2);
        assert_eq!(cap.max_scale, gen_core::DEFAULT_PAG_MAX_SCALE);

        let d = descriptor();
        assert_eq!(d.capabilities.pag.as_ref(), Some(&cap));

        let active = GenerationRequest {
            prompt: "a fox".into(),
            features: mlx_gen::gen_core::FeatureParams {
                pag: Some(gen_core::PagConfig {
                    scale: 1.2,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &d.capabilities, &active).is_ok());
        let resolved = d
            .capabilities
            .resolve_pag(MODEL_ID, &active)
            .unwrap()
            .expect("active PAG resolves");
        assert_eq!(resolved.layers, vec![15]);
        assert_eq!(resolved.execution_report(4).active_steps, 3);

        let noop = GenerationRequest {
            prompt: "a fox".into(),
            features: mlx_gen::gen_core::FeatureParams {
                pag: Some(gen_core::PagConfig {
                    scale: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &d.capabilities, &noop).is_ok());
        assert_eq!(d.capabilities.resolve_pag(MODEL_ID, &noop).unwrap(), None);

        let bad_layer = GenerationRequest {
            prompt: "a fox".into(),
            features: mlx_gen::gen_core::FeatureParams {
                pag: Some(gen_core::PagConfig {
                    scale: 1.0,
                    layers: Some(vec![30]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &d.capabilities, &bad_layer).is_err());
    }
}
