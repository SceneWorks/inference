//! `LensGenerator` — the [`mlx_gen::Generator`] impl wiring the Lens pipeline ([`crate::pipeline`])
//! into `mlx_gen`'s registry under **two** ids (sc-3173):
//!
//! - **`lens_turbo`** — the distilled turbo variant: **4 steps, guidance 1.0** (≈ no CFG).
//! - **`lens`** — the base variant: **20 steps, CFG 5.0**.
//!
//! Both ids share the identical crate/architecture/weights tree and differ **only** in their default
//! `num_steps` / `guidance_scale` (the reference ships them as separate model cards with the same
//! arch). A request's explicit `steps` / `guidance` still override the per-id default.
//!
//! **Surface.** This is a pure **T2I** generator: no img2img / ControlNet / IP conditioning (none
//! exists in the Lens port). **LoRA + LoKr** merge into the DiT's joint-attention projections at load
//! (sc-3174 — inference consumption; native-MLX *training* is [`crate::training`], sc-5148). The dense path is bf16; the `Fp32`
//! precision override is honored. **Q4/Q8** quantize the gpt-oss encoder's MoE experts (sc-3172 —
//! the ~38 GB / 20 B-param bulk → ~12 GB) **and** the DiT's linears (sc-3175) at load.
//!
//! **Registration mechanism:** the two named constants below are composed by the family registry,
//! which is in turn composed by the MLX platform catalog.

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::residency::StagedHeavy;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LatentDecoder, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, Residency, Result, SizeFloor, WeightsSource,
};
use mlx_gen_flux2::model::PID_BACKBONE;
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};

use crate::pipeline::{LensHeavy, LensText, DEFAULT_DATE, VAE_SCALE_FACTOR};

/// Registry id — the distilled turbo variant.
pub const MODEL_ID_TURBO: &str = "lens_turbo";
/// Registry id — the base variant.
pub const MODEL_ID_BASE: &str = "lens";

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`) baked into the loaded generator.
#[derive(Clone, Copy)]
struct Defaults {
    id: &'static str,
    steps: u32,
    guidance: f32,
}

// The step/guidance numbers are the single source of truth in [`crate::schedule`] (`TURBO`/`BASE`);
// the registry just re-tags them with the model id.
const TURBO_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_TURBO,
    steps: crate::schedule::TURBO.num_steps as u32,
    guidance: crate::schedule::TURBO.guidance_scale,
};
const BASE_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_BASE,
    steps: crate::schedule::BASE.num_steps as u32,
    guidance: crate::schedule::BASE.guidance_scale,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextEncoderStorage {
    PackedAffine,
    Mxfp4,
    DenseBf16,
    Unknown,
}

/// Classify the Lens text encoder's on-disk representation from its provider-owned config.
///
/// Packed affine takes precedence because the re-hosted q4/q8 turnkeys intentionally retain the
/// upstream `quantization_config.quant_method = "mxfp4"` provenance while adding the load-bearing
/// `quantization.bits` marker for their converted weights. Missing or unrecognized metadata stays
/// `Unknown` so footprint accounting remains conservative instead of silently under-predicting an
/// MXFP4 source.
fn text_encoder_storage(root: &Path) -> Result<TextEncoderStorage> {
    if mlx_gen::quant::packed_quant_bits(root, "text_encoder")?.is_some() {
        return Ok(TextEncoderStorage::PackedAffine);
    }

    let config_path = root.join("text_encoder").join("config.json");
    let bytes = match std::fs::read(&config_path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TextEncoderStorage::Unknown)
        }
        Err(err) => {
            return Err(Error::Msg(format!(
                "lens text encoder: read {}: {err}",
                config_path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        Error::Msg(format!(
            "lens text encoder: parse {}: {err}",
            config_path.display()
        ))
    })?;
    if config
        .pointer("/quantization_config/quant_method")
        .and_then(serde_json::Value::as_str)
        == Some("mxfp4")
    {
        return Ok(TextEncoderStorage::Mxfp4);
    }
    if config.get("dtype").and_then(serde_json::Value::as_str) == Some("bfloat16") {
        return Ok(TextEncoderStorage::DenseBf16);
    }
    Ok(TextEncoderStorage::Unknown)
}

/// Lens' identity + capabilities for `id` — constructible without loading weights (registry
/// introspection). Advertises the wired + parity-proven surface: T2I with negative-prompt /
/// guidance CFG, no conditioning, LoRA + LoKr (DiT joint-attention, sc-3174), and Q4/Q8 load-time
/// quant (gpt-oss MoE experts sc-3172 + DiT linears sc-3175).
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::FLUX2_PACKED_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // The norm-rescaled CFG path is always present; turbo simply defaults guidance to 1.0.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![], // pure T2I — no img2img / control / IP in the Lens port
            // sc-3174: LoRA + LoKr merge into the DiT's joint-attention projections at load.
            supports_lora: true,
            supports_lokr: true,
            // epic 7114 sc-7305: advertise the curated sampler/scheduler menu (mirrors the candle Lens
            // adoption) so the per-generation knobs route through the unified `Sampler<MlxLatentOps>` +
            // `FlowModelSampling`. The legacy native aliases stay valid for old recipes; both N3-fall
            // back to the default (`flow_match_euler` → euler, `flow_match` → the native empirical-μ
            // schedule), so they never hard-fail a generation.
            samplers: {
                let mut s = curated_sampler_names();
                s.push("flow_match_euler");
                s
            },
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("flow_match");
                s
            },
            // Buckets span 736..2080 (all ÷16); allow any ÷16 size in a sane range.
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2080,
            max_count: 8,
            mac_only: true,
            // Q4/Q8 quantize the gpt-oss encoder's MoE experts (sc-3172 — the ~38 GB / 20 B-param
            // bulk → ~12 GB) and the DiT's linears (sc-3175) at load.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            // The Lens schedule computes its own empirical-μ shift internally (not a loader hint).
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
            unconditionally_engages_staged_residency: false,
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

/// Public descriptor accessors (used by the registry submits + tests).
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(MODEL_ID_TURBO)
}
pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(MODEL_ID_BASE)
}

/// A loaded, dispatchable Lens generator: the variant's descriptor & sampling defaults + the
/// component-residency strategy (epic 10834 Phase 1, sc-11030). Both `lens` and `lens_turbo` share
/// this and differ only in the baked sampling defaults.
pub struct LensGenerator {
    descriptor: ModelDescriptor,
    defaults: Defaults,
    precision: Precision,
    quant: Option<Quant>,
    streamable_text_encoder: bool,
    streamable_dit: bool,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    default_stage_residency: bool,
    /// Request-scoped component owner (sc-11030; hoisted to the shared seam in sc-11125). The load
    /// policy is retained separately in `default_stage_residency`: both defaults keep this owner lazy
    /// so a later memory-ladder request can choose its physical materialization before any full warm
    /// pair exists. The [`Residency`] seam owns eval/drop/clear, stage-boundary cancellation, and the
    /// error-safe cache flush.
    residency: Residency<LensText, LensHeavyOwned>,
    text_stream_residency: Residency<LensText, LensHeavyOwned>,
    dit_stream_residency: Residency<LensText, LensHeavyOwned>,
    both_stream_residency: Residency<LensText, LensHeavyOwned>,
}

/// The heavy render-phase components (the DiT + VAE via [`LensHeavy`], plus the optional PiD decoder) —
/// everything but the text encoder. Owned for the duration selected by the request's residency plan.
pub(crate) struct LensHeavyOwned {
    heavy: LensHeavy,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7847): loaded when the spec carries
    /// `LoadSpec::pid`. `Some` → a `req.use_pid` generation decodes through the `flux2` student (4× SR).
    pid: Option<PidEngine>,
}

pub(crate) struct LensLightOwned {
    vae: mlx_gen_flux2::Flux2Vae,
    pid: Option<PidEngine>,
}

pub(crate) struct LensDecodeRef<'a> {
    vae: &'a mlx_gen_flux2::Flux2Vae,
    pid: Option<&'a PidEngine>,
}

impl StagedHeavy for LensHeavyOwned {
    type Light = LensLightOwned;
    type DecodeView<'a> = LensDecodeRef<'a>;

    fn shed_dit(self) -> Self::Light {
        LensLightOwned {
            vae: self.heavy.into_vae(),
            pid: self.pid,
        }
    }

    fn decode_view(&self) -> Self::DecodeView<'_> {
        LensDecodeRef {
            vae: self.heavy.vae(),
            pid: self.pid.as_ref(),
        }
    }

    fn light_view(light: &Self::Light) -> Self::DecodeView<'_> {
        LensDecodeRef {
            vae: &light.vae,
            pid: light.pid.as_ref(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamScope {
    TextEncoder,
    Dit,
    Both,
}

impl StreamScope {
    fn text(self) -> bool {
        matches!(self, Self::TextEncoder | Self::Both)
    }

    fn dit(self) -> bool {
        matches!(self, Self::Dit | Self::Both)
    }
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically whether
/// they are held resident or were just loaded by the `Sequential` path.
struct LensHeavyRef<'a> {
    heavy: &'a LensHeavy,
    pid: Option<&'a PidEngine>,
}

impl LensHeavyOwned {
    fn as_ref(&self) -> LensHeavyRef<'_> {
        LensHeavyRef {
            heavy: &self.heavy,
            pid: self.pid.as_ref(),
        }
    }
}

/// Measured production domain for the 24-layer Lens encoder. Keeping the resolver explicit makes a
/// hand-built request fail closed rather than silently accepting an uncalibrated window.
const TEXT_ENCODER_WINDOW_DOMAIN: &[u32] = &[crate::memory_strategy::TEXT_ENCODER_WINDOW];

fn resolve_encoder_window(req: &GenerationRequest, streamable: bool) -> Result<Option<usize>> {
    Ok(resolve_transformer_windows(req, streamable, false)?.0)
}

fn resolve_transformer_windows(
    req: &GenerationRequest,
    text_streamable: bool,
    dit_streamable: bool,
) -> Result<(Option<usize>, Option<usize>)> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok((None, None));
    };
    let component = memory.transformer_window_component.unwrap_or_default();
    let needs_text = matches!(
        component,
        mlx_gen::gen_core::TransformerComponent::TextEncoder
            | mlx_gen::gen_core::TransformerComponent::Both
    );
    let needs_dit = matches!(
        component,
        mlx_gen::gen_core::TransformerComponent::Dit
            | mlx_gen::gen_core::TransformerComponent::Both
    );
    if needs_text && !text_streamable {
        return Err(Error::Unsupported(
            "lens: text-encoder streaming requires a deferred directory load whose numeric tier can be replayed without a load-time conversion"
                .to_owned(),
        ));
    }
    if needs_dit && !dit_streamable {
        return Err(Error::Unsupported(
            "lens: DiT streaming requires a deferred directory load with dense bf16 or exact prepacked weights and no adapters"
                .to_owned(),
        ));
    }
    let window = memory
        .transformer_window_size
        .unwrap_or(crate::memory_strategy::TEXT_ENCODER_WINDOW);
    if !TEXT_ENCODER_WINDOW_DOMAIN.contains(&window) {
        return Err(Error::Unsupported(format!(
            "lens: transformer_window_size={window} is outside the measured production domain \
             {TEXT_ENCODER_WINDOW_DOMAIN:?}"
        )));
    }
    Ok((
        needs_text.then_some(window as usize),
        needs_dit.then_some(window as usize),
    ))
}

/// Build a [`LensGenerator`] from a [`LoadSpec`] with the given per-variant defaults.
///
/// `spec.weights` is a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot dir (the diffusers
/// multi-component tree). Dense runs **bf16**; `Precision::Fp32` loads the tight-gate f32 path.
/// `spec.quantize` (Q4/Q8) quantizes the encoder's MoE experts at load (sc-3172); `spec.adapters`
/// (LoRA/LoKr) merge into the DiT (sc-3174). `control` / `ip_adapter` are not part of the Lens port.
///
/// Component residency (epic 10834 Phase 1, sc-11030): both load defaults retain request-scoped
/// loader closures here. The requested policy becomes the no-explicit-memory generation default:
/// `Resident` holds both phases for that request, while `Sequential` drops the encoder before loading
/// denoise/decode. Keeping construction lazy also lets an explicit memory plan choose TE, DiT, or Both
/// bounded materialization without first allocating a full resident pair.
fn load_with(spec: &LoadSpec, defaults: Defaults) -> Result<Box<dyn Generator>> {
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(defaults.id, spec)?;
    Ok(Box::new(LensGenerator {
        descriptor: descriptor_for(defaults.id),
        defaults,
        precision: spec.precision,
        quant: spec.quantize,
        streamable_text_encoder: crate::memory_strategy::can_stream_text(spec)?,
        streamable_dit: crate::memory_strategy::can_stream_dit(spec)?,
        memory_strategy,
        default_stage_residency: matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential),
        residency: build_residency(spec, defaults.id)?,
        text_stream_residency: build_request_residency(spec, defaults.id, StreamScope::TextEncoder),
        dit_stream_residency: build_request_residency(spec, defaults.id, StreamScope::Dit),
        both_stream_residency: build_request_residency(spec, defaults.id, StreamScope::Both),
    }))
}

fn build_request_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    scope: StreamScope,
) -> Residency<LensText, LensHeavyOwned> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::request_scoped(
        move |_| {
            let (root, dtype) = resolve_root(&spec_text)?;
            load_text_phase_scoped(&spec_text, &root, dtype, model_id, scope.text())
        },
        move |use_pid, _| {
            let (root, dtype) = resolve_root(&spec_heavy)?;
            load_heavy_phase_scoped(&spec_heavy, &root, dtype, use_pid, model_id, scope.dit())
        },
    )
}

/// The ordinary request-scoped [`Residency`] owner both Lens variants share. Component construction
/// remains lazy for both load defaults so an explicit memory-ladder request cannot inherit a full
/// eager pair. [`LoadSpec::offload_policy`] is retained by [`LensGenerator`] and supplied as the
/// default stage choice for requests without an explicit memory plan. The up-front [`resolve_root`]
/// and packed-tier checks still fail fast for both policies before any request begins.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    model_id: &'static str,
) -> Result<Residency<LensText, LensHeavyOwned>> {
    // Up-front fail-fast for both policies (mirrors the pre-seam load order).
    let (root, _) = resolve_root(spec)?;
    // F-010 (sc-12462): fail-fast requested-vs-packed tier guard for BOTH policies — `Sequential`
    // defers the phase loaders to the first generate, so without this an e.g. Q4 request over a Q8
    // turnkey would only surface mid-job (Resident re-checks inside the phase loaders below). Both
    // quantized components carry the converter's marker; check both so a half-converted snapshot
    // still errors.
    if let Some(q) = spec.quantize {
        let text_needs_quant =
            mlx_gen::quant::needs_load_time_quant(&root, "text_encoder", q.bits(), model_id)?;
        let transformer_needs_quant =
            mlx_gen::quant::needs_load_time_quant(&root, "transformer", q.bits(), model_id)?;
        if matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential)
            && (text_needs_quant || transformer_needs_quant)
        {
            mlx_gen::residency::warn_sequential_requantize(model_id, q.bits());
        }
    }
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Ok(Residency::request_scoped(
        move |_| {
            let (root, dtype) = resolve_root(&spec_text)?;
            load_text_phase(&spec_text, &root, dtype, model_id)
        },
        move |use_pid, _| {
            let (root, dtype) = resolve_root(&spec_heavy)?;
            load_heavy_phase(&spec_heavy, &root, dtype, use_pid, model_id)
        },
    ))
}

/// Snapshot-dir + precision→dtype resolution (rejecting a single-file source / unsupported overlays),
/// shared by the `Resident` build and the `Sequential` per-phase loaders (sc-11030).
fn resolve_root(spec: &LoadSpec) -> Result<(std::path::PathBuf, Dtype)> {
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Msg(
            "lens: ControlNet / IP-Adapter conditioning is not part of the Lens port".into(),
        ));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "lens: expects a Lens snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok((root, dtype))
}

/// Load the text-encode phase — the gpt-oss encoder dropped first under `Sequential`. `spec.quantize`
/// quantizes the encoder's MoE experts at load (sc-3172).
fn load_text_phase(spec: &LoadSpec, root: &Path, dtype: Dtype, model_id: &str) -> Result<LensText> {
    load_text_phase_scoped(
        spec,
        root,
        dtype,
        model_id,
        crate::memory_strategy::is_streamable_spec(spec),
    )
}

fn load_text_phase_scoped(
    spec: &LoadSpec,
    root: &Path,
    dtype: Dtype,
    model_id: &str,
    streamable: bool,
) -> Result<LensText> {
    // F-010 (sc-12462): reject a requested-vs-packed tier mismatch BEFORE any weights load — a
    // packed turnkey's experts build `ExpertBank::Quant` from the on-disk shapes, so e.g. a Q4
    // request over a Q8 turnkey would otherwise silently serve Q8. The returned bool is unused:
    // `from_weights_quant` auto-detects packed vs dense itself ("lens" — the snapshot tree is
    // shared by both registry ids).
    if let Some(q) = spec.quantize {
        mlx_gen::quant::needs_load_time_quant(root, "text_encoder", q.bits(), model_id)?;
    }
    if streamable {
        LensText::load_streamable(root, dtype, spec.quantize)
    } else {
        LensText::load(root, dtype, spec.quantize)
    }
}

/// Load the heavy render phase — DiT (+ LoRA/LoKr merge, then Q4/Q8) + VAE + the optional PiD overlay —
/// everything but the text encoder. Factored so `Sequential` loads these AFTER the encoder is dropped.
/// The DiT quantizes **after** any adapter merge (sc-3175 — adapters are forward-time residuals over
/// the quantized base); the components are byte-identical to the `Resident` composition.
fn load_heavy_phase(
    spec: &LoadSpec,
    root: &Path,
    dtype: Dtype,
    load_pid: bool,
    model_id: &str,
) -> Result<LensHeavyOwned> {
    load_heavy_phase_scoped(spec, root, dtype, load_pid, model_id, false)
}

fn load_heavy_phase_scoped(
    spec: &LoadSpec,
    root: &Path,
    dtype: Dtype,
    load_pid: bool,
    model_id: &str,
    streamable: bool,
) -> Result<LensHeavyOwned> {
    // F-010 (sc-12462): reject a requested-vs-packed tier mismatch BEFORE any weights load — the
    // DiT projections load packed via `quant::lin` (a Quantized base on which
    // `AdaptableLinear::quantize` no-ops), so e.g. a Q4 request over a Q8 turnkey would otherwise
    // silently serve Q8. `false` (already packed at the requested bits) also skips the no-op
    // `quantize_dit` below.
    let needs_quant = match spec.quantize {
        Some(q) => mlx_gen::quant::needs_load_time_quant(root, "transformer", q.bits(), model_id)?,
        None => false,
    };
    let heavy = if streamable {
        if needs_quant {
            return Err(Error::Unsupported(
                "lens: deferred DiT windows cannot replay load-time quantization; use an exact prepacked snapshot"
                    .to_owned(),
            ));
        }
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(
                "lens: deferred DiT windows do not replay LoRA/LoKr mutations; use a non-streamed memory rung"
                    .to_owned(),
            ));
        }
        LensHeavy::load_streamable(root, dtype, spec.quantize)?
    } else {
        let mut heavy = LensHeavy::load(root, dtype)?;
        if !spec.adapters.is_empty() {
            heavy.apply_adapters(&spec.adapters)?;
        }
        if let Some(q) = spec.quantize {
            if needs_quant {
                heavy.quantize_dit(q)?;
            }
        }
        heavy
    };
    // PiD decoder overlay (epic 7840, sc-7847): load the shared `flux2` student + Gemma once when the
    // spec carries it AND this generate uses it (`load_pid`, F-177) — Resident passes `true` (loaded
    // once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips the student + Gemma.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(LensHeavyOwned { heavy, pid })
}

impl Generator for LensGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::begin_request(
            self.defaults.id,
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl LensGenerator {
    /// The rich-`Result` body behind [`Generator::validate`].
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(self.defaults.id, &self.descriptor.capabilities, req)?;
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]: map the request onto the residency,
    /// looping `count` with per-image seeds and streaming step/decode progress. The staged residency
    /// lifecycle (encode → drop the gpt-oss encoder under `Sequential` → load the DiT/VAE/PiD →
    /// denoise/decode → free the heavy bundle) is driven by the shared request-scoped [`Residency`]
    /// seam (sc-11125), which owns the eval/drop/clear discipline, stage-boundary cancellation, and
    /// the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;

        if req.memory.is_some() {
            return self.generate_memory_impl(req, on_progress);
        }

        let steps = req.steps.unwrap_or(self.defaults.steps) as usize;
        let guidance = req.guidance.unwrap_or(self.defaults.guidance);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let total = steps as u32;
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;
        let encoder_window = resolve_encoder_window(req, self.streamable_text_encoder)?;

        // Phase A: prompt → embeds (sc-11030; sc-11125). Under `Sequential` the shared seam loads the
        // gpt-oss encoder, encodes, materializes, then DROPS it + `clear_cache()` so its ~13 GB frees
        // before the DiT/VAE load below — the peak-bounding win. Encoding once (deterministic, no RNG
        // draw) is byte-identical to the pre-sc-11030 per-image re-encode (the init noise reseeds per
        // image inside `render`). Under the Resident default it remains live through this request.
        self.residency.run_request_scoped(
            self.default_stage_residency,
            false,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &LensText| {
                text.encode_prompt_windowed(
                    &req.prompt,
                    negative,
                    DEFAULT_DATE,
                    guidance,
                    Some(&req.cancel),
                    encoder_window,
                )
            },
            // Materialize the features + mask while the encoder is still alive (Sequential only) — MLX
            // is lazy, so un-evaluated outputs keep the encoder referenced and the drop frees nothing.
            |encoded: Option<&(Vec<Array>, Array)>| {
                let Some((features, mask)) = encoded else {
                    return Ok(());
                };
                let mut to_eval: Vec<&Array> = features.iter().collect();
                to_eval.push(mask);
                mlx_rs::transforms::eval(to_eval)?;
                Ok(())
            },
            // ── Establish the heavy render components (DiT + VAE + PiD) and run the render body once
            // against the `heavy` borrow — identical for both residencies.
            |heavy_owned, enc, on_progress| {
                let heavy = heavy_owned.as_ref();
                let (encoder_features, encoder_mask) = enc;

                // PiD decode overlay (epic 7840, sc-7847) + `from_ldm` early-stop (sc-8048): one decoder serves
                // the whole count loop (same prompt). Errors if `req.use_pid` but the model wasn't loaded with
                // `LoadSpec::pid`; `None` (the default) → the byte-exact native Flux.2 VAE path. Lens is
                // `vp_frame=false` (schedule σ *is* the degrade σ) and pure T2I (`start_step = 0`); resolve the
                // plan against the SAME descending schedule `render` runs. `None` capture → full schedule.
                let sigmas =
                    heavy
                        .heavy
                        .resolve_sigmas(latent_h, latent_w, steps, req.scheduler.as_deref());
                let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
                let keep = (keep < sigmas.len()).then_some(keep);
                // F-030 (sc-11133): the PiD `from_ldm` early-stop truncates the descending schedule to
                // `keep` σ nodes, so `render` (→ `run_curated_sampler`) runs and reports exactly
                // `sigmas[..keep].len() - 1 == keep - 1` steps — NOT the requested `steps`. Deriving the
                // emitted `total` from `keep` keeps the bar monotone AND lets it reach its total, so the
                // `cur >= total` Decoding trigger below fires on the shortened schedule (without this the
                // job froze at `(keep-1)/steps` and the 4×-SR decode was invisible).
                let effective_total = effective_step_total(keep, total);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    base_seed,
                    self.defaults.id,
                    capture_sigma,
                )?;
                let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);

                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = base_seed.wrapping_add(i as u64);
                    // The one render body (sc-11030): the same `LensHeavy::render` for both residencies, so a
                    // Sequential job (encoder already dropped) is byte-identical to Resident. The reasoner
                    // (sc-3176) is a standalone struct-API opt-in; the registry path leaves it off.
                    let image = heavy.heavy.render_with_preview(
                        &encoder_features,
                        &encoder_mask,
                        latent_h,
                        latent_w,
                        steps,
                        guidance,
                        // epic 7114 sc-7305: per-generation curated sampler/scheduler (N3 fallback inside the
                        // unified framework; the worker also pre-normalizes unadvertised names).
                        req.sampler.as_deref(),
                        req.scheduler.as_deref(),
                        seed,
                        keep,
                        pid_ref,
                        &req.cancel,
                        &mut |cur| {
                            on_progress(Progress::Step {
                                current: cur as u32,
                                total: effective_total,
                            });
                            // F-106: `render` decodes immediately after the final step (it exposes only a step
                            // callback, not a Progress sink), so emit `Decoding` when the last step lands —
                            // BEFORE the VAE/PiD decode. F-030: gate on `effective_total` so the truncated
                            // early-stop schedule still trips it exactly once.
                            if cur as u32 >= effective_total {
                                on_progress(Progress::Decoding);
                            }
                        },
                        &req.preview,
                    )?;
                    images.push(image);
                    // F-030 residual (sc-11133): a `keep == 1` early-stop runs 0 real steps, so the
                    // per-step callback above never fires — the bar stalls at 0/1 and `Decoding` never
                    // trips. Synthesize the terminal `Step` + one `Decoding` in that case (no-op for a
                    // schedule that ran ≥ 1 real step and drove its own terminal above).
                    emit_terminal_if_no_steps(keep, total, on_progress);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }

    /// Shared-ladder execution. The legacy no-memory path above is intentionally unchanged; an
    /// admitted request selects all lifecycle and scratch levers explicitly here.
    fn generate_memory_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let memory = req.memory.unwrap_or_default();
        let (encoder_window, dit_window) =
            resolve_transformer_windows(req, self.streamable_text_encoder, self.streamable_dit)?;
        let component = memory.transformer_window_component.unwrap_or_default();
        let residency = if memory.stream_transformer_blocks {
            match component {
                mlx_gen::gen_core::TransformerComponent::TextEncoder => &self.text_stream_residency,
                mlx_gen::gen_core::TransformerComponent::Dit => &self.dit_stream_residency,
                mlx_gen::gen_core::TransformerComponent::Both => &self.both_stream_residency,
            }
        } else {
            &self.residency
        };

        // A cached generator has one physical warm shape at a time. Evict non-selected owners before
        // loading the chosen component scope so switching requests cannot retain duplicate trunks.
        for other in [
            &self.residency,
            &self.text_stream_residency,
            &self.dit_stream_residency,
            &self.both_stream_residency,
        ] {
            if !std::ptr::eq(other, residency) {
                other.evict_warm()?;
            }
        }

        let attention = if memory.chunk_attention {
            mlx_gen::attention::AttentionPlan::budgeted(
                mlx_gen::attention::AttentionBudget::from_score_elements(
                    memory
                        .attention_chunk_size
                        .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE)
                        as u64,
                    true,
                ),
            )
            .with_cancel(&req.cancel)
        } else {
            mlx_gen::attention::AttentionPlan::UNBOUNDED
        };
        let tiling = memory.tile_vae_decode.then(|| {
            mlx_gen::tiling::TilingConfig::spatial_only(
                memory
                    .decode_tile_edge
                    .unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE) as i32,
                memory
                    .decode_overlap
                    .unwrap_or(crate::memory_strategy::DECODE_OVERLAP) as i32,
            )
        });

        let steps = req.steps.unwrap_or(self.defaults.steps) as usize;
        let guidance = req.guidance.unwrap_or(self.defaults.guidance);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;

        struct Denoised {
            latents: Array,
        }

        residency.run_staged_request_scoped(
            memory.stage_residency,
            memory.stream_transformer_blocks,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text| {
                calibration_fault(req, mlx_gen::gen_core::MemoryPhase::Conditioning)?;
                text.encode_prompt_windowed(
                    &req.prompt,
                    negative,
                    DEFAULT_DATE,
                    guidance,
                    Some(&req.cancel),
                    encoder_window,
                )
            },
            |encoded| {
                let Some((features, mask)) = encoded else {
                    return Ok(());
                };
                let mut arrays: Vec<&Array> = features.iter().collect();
                arrays.push(mask);
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            |heavy, (features, mask), progress| {
                calibration_fault(req, mlx_gen::gen_core::MemoryPhase::Denoise)?;
                let mut out = Vec::with_capacity(req.count as usize);
                for index in 0..req.count {
                    let seed = base_seed.wrapping_add(index as u64);
                    mlx_rs::random::seed(seed)?;
                    let init = mlx_rs::random::normal::<f32>(
                        &[1, (latent_h * latent_w) as i32, 128],
                        None,
                        None,
                        None,
                    )?;
                    let latents = heavy.heavy.denoise_with_sampler_keep_with_preview_memory(
                        &features,
                        &mask,
                        &init,
                        latent_h,
                        latent_w,
                        steps,
                        guidance,
                        req.sampler.as_deref(),
                        req.scheduler.as_deref(),
                        seed,
                        None,
                        &req.cancel,
                        &mut |current, total| {
                            progress(Progress::Step {
                                current: current as u32,
                                total: total as u32,
                            })
                        },
                        &req.preview,
                        attention,
                        dit_window,
                    )?;
                    out.push(Denoised { latents });
                }
                Ok(out)
            },
            |denoised| {
                let arrays: Vec<&Array> = denoised.iter().map(|item| &item.latents).collect();
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            |decode, denoised, progress| {
                calibration_fault(req, mlx_gen::gen_core::MemoryPhase::Decode)?;
                if decode.pid.is_some() {
                    return Err(Error::Unsupported(
                        "lens: the shared native-VAE memory ladder does not cover the PiD overlay"
                            .to_owned(),
                    ));
                }
                let mut images = Vec::with_capacity(denoised.len());
                for item in denoised {
                    progress(Progress::Decoding);
                    let decoded = crate::vae::decode_with_tiling(
                        decode.vae,
                        &item.latents,
                        latent_h,
                        latent_w,
                        None,
                        tiling.as_ref(),
                        Some(&req.cancel),
                    )?;
                    images.push(crate::pipeline::decoded_to_image(&decoded)?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

fn calibration_fault(req: &GenerationRequest, phase: mlx_gen::gen_core::MemoryPhase) -> Result<()> {
    if req.memory.is_some_and(|memory| {
        memory.calibration_fault_harness_authorized && memory.calibration_error_phase == Some(phase)
    }) {
        return Err(Error::Msg(format!("lens calibration fault at {phase:?}")));
    }
    Ok(())
}

/// The number of denoise steps the sampler actually runs — and therefore the `Progress::Step.total`
/// and the `Decoding` trigger (F-030, sc-11133). With no PiD early-stop (`keep == None`) it is the
/// requested `steps`; under a `from_ldm` early-stop that truncates the schedule to `keep` σ nodes,
/// `run_curated_sampler` reports exactly `sigmas[..keep].len() - 1 == keep - 1` steps, so the bar
/// must be sized to that (never below 1) or it freezes below its stale `steps` total and never trips
/// `Decoding`.
fn effective_step_total(keep: Option<usize>, steps: u32) -> u32 {
    match keep {
        Some(k) => (k.saturating_sub(1) as u32).max(1),
        None => steps,
    }
}

/// The number of real denoise transitions a schedule actually runs: `keep - 1` σ steps under a PiD
/// early-stop (0 when `keep <= 1`), else the full `steps`. Distinct from [`effective_step_total`],
/// which floors the *bar size* at 1 — a `keep == 1` schedule sizes the bar to 1 yet runs ZERO
/// transitions, so `run_curated_sampler` never invokes the per-step callback (sc-11133).
fn real_step_count(keep: Option<usize>, steps: u32) -> u32 {
    match keep {
        Some(k) => k.saturating_sub(1) as u32,
        None => steps,
    }
}

/// F-030 residual (sc-11133): a `keep == 1` PiD early-stop truncates the schedule to a single σ
/// node, so `run_curated_sampler` runs zero transitions and `render`'s per-step callback never
/// fires — the bar would freeze at `0/total` and the `cur >= total` `Decoding` trigger never trip.
/// When no real step runs, synthesize the terminal `Step{total,total}` + one `Decoding` so the bar
/// reaches its total and `Decoding` fires exactly once. A schedule with ≥ 1 real step drives the
/// bar (and its own `Decoding`) through the per-step callback and needs no synthetic terminal.
/// Returns whether a terminal was emitted (weight-free unit-testable).
fn emit_terminal_if_no_steps(
    keep: Option<usize>,
    steps: u32,
    on_progress: &mut dyn FnMut(Progress),
) -> bool {
    if real_step_count(keep, steps) != 0 {
        return false;
    }
    let total = effective_step_total(keep, steps);
    on_progress(Progress::Step {
        current: total,
        total,
    });
    on_progress(Progress::Decoding);
    true
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg, sampler,
    // scheduler, conditioning kinds.
    caps.validate_request(id, req)?;

    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    // The Flux.2 VAE + DiT patchify downsample by 16; non-multiple-of-16 dims mismatch latent shapes.
    if !req.width.is_multiple_of(VAE_SCALE_FACTOR) || !req.height.is_multiple_of(VAE_SCALE_FACTOR) {
        return Err(Error::Msg(format!(
            "{id}: width/height must be multiples of {VAE_SCALE_FACTOR} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// Thin id-binding loaders: each pins the variant defaults onto `load_with`, so they can't be a
// plain `load` path. They return the crate's rich `Result`; `register_generators!` adds the
// `gen_core::Result` bridge.
fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, TURBO_DEFAULTS)
}
fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, BASE_DEFAULTS)
}

/// Per-component resident-weight estimate (sc-10894/sc-11924) for the MLX fit-gate's staged split —
/// gpt-oss MoE text encoder (`text_encoder/`), the DiT (`transformer/`), and the Flux.2 VAE (`vae/`),
/// summed from the exact snapshot subdirs [`crate::pipeline`] loads. The text encoder is the ~38 GB /
/// 20B-param bulk the `Sequential` schedule drops before the DiT loads, so an accurate split here is
/// what lets the fit-gate select staged residency for `lens` / `lens_turbo`.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mut footprint = mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )?;
    let root = match &spec.weights {
        mlx_gen::WeightsSource::Dir(root) => root,
        mlx_gen::WeightsSource::File(_) => return Ok(footprint),
    };
    let storage = text_encoder_storage(root)?;
    if spec.quantize.is_none()
        && matches!(
            storage,
            TextEncoderStorage::Mxfp4 | TextEncoderStorage::Unknown
        )
    {
        // sc-11924: the dense Lens snapshot stores the gpt-oss MoE experts as MXFP4 but the loader
        // materializes them at bf16. The 1024² real-weight calibration measured 30.07 GiB resident
        // for the encoder (vs 12.83 GiB on disk). Keep this provider- and FORMAT-specific: measured
        // q4/q8 and packed turnkeys retain their disk-derived footprint, an explicit bf16-on-disk
        // encoder is not inflated, and unknown metadata stays conservative so it cannot hide MXFP4.
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        footprint.text_encoder = footprint.text_encoder.max((30.07 * GIB).ceil() as u64);
    }
    Ok(footprint)
}

mlx_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo;
    footprint = component_footprint
}

pub const TURBO_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID_TURBO,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID_TURBO, spec),
        safety_check: crate::memory_strategy::registered_safety_check,
    };
pub const TURBO_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_TURBO,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(
                MODEL_ID_TURBO,
                spec,
                contract,
                context,
            )
        },
    };

pub const BASE_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID_BASE,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID_BASE, spec),
        safety_check: crate::memory_strategy::registered_safety_check,
    };
pub const BASE_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_BASE,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID_BASE, spec, contract, context)
        },
    };
mlx_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    /// Measurement-only construction for the Q4 non-win record. Production goes through
    /// `is_streamable_spec` and correctly refuses to advertise Q4; this test deliberately injects
    /// the streamable text-phase loader so the rejected tier's request-level result stays
    /// reproducible without opening a production bypass.
    fn q4_measurement_generator(spec: &LoadSpec) -> Result<LensGenerator> {
        let spec_text = spec.clone();
        let spec_heavy = spec.clone();
        let residency = Residency::from_policy(
            mlx_gen::OffloadPolicy::Sequential,
            move || {
                let (root, dtype) = resolve_root(&spec_text)?;
                if let Some(q) = spec_text.quantize {
                    mlx_gen::quant::needs_load_time_quant(
                        &root,
                        "text_encoder",
                        q.bits(),
                        MODEL_ID_TURBO,
                    )?;
                }
                LensText::load_streamable(&root, dtype, spec_text.quantize)
            },
            move |use_pid| {
                let (root, dtype) = resolve_root(&spec_heavy)?;
                load_heavy_phase(&spec_heavy, &root, dtype, use_pid, MODEL_ID_TURBO)
            },
        )?;
        Ok(LensGenerator {
            descriptor: descriptor_turbo(),
            defaults: TURBO_DEFAULTS,
            precision: spec.precision,
            quant: spec.quantize,
            streamable_text_encoder: true,
            streamable_dit: false,
            memory_strategy: crate::memory_strategy::memory_strategy_contract(
                MODEL_ID_TURBO,
                spec,
            )?,
            default_stage_residency: true,
            residency,
            text_stream_residency: build_request_residency(
                spec,
                MODEL_ID_TURBO,
                StreamScope::TextEncoder,
            ),
            dit_stream_residency: build_request_residency(spec, MODEL_ID_TURBO, StreamScope::Dit),
            both_stream_residency: build_request_residency(spec, MODEL_ID_TURBO, StreamScope::Both),
        })
    }

    #[test]
    #[ignore = "SC-15800 Q4 request non-win; needs an explicit LENS_DIR q4 turnkey and Apple/Metal"]
    fn q4_request_non_improvement_remains_reproducible_but_unadvertised() {
        use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
        use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

        let root = std::path::PathBuf::from(
            std::env::var("LENS_DIR").expect("set LENS_DIR to the explicit Lens q4 tier"),
        );
        assert_eq!(root.file_name().and_then(|name| name.to_str()), Some("q4"));
        let spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(Quant::Q4);
        assert!(matches!(
            crate::memory_strategy::memory_strategy_contract(MODEL_ID_TURBO, &spec)
                .unwrap()
                .capability(mlx_gen::gen_core::MemoryStrategy::BoundedTransformerResidency)
                .map(|capability| &capability.support),
            Some(mlx_gen::gen_core::MemoryStrategySupport::Missing)
        ));

        let run = |window: Option<u32>| {
            let generator = q4_measurement_generator(&spec).unwrap();
            let request = GenerationRequest {
                prompt: "a red fox crossing a snowy clearing at dawn, documentary photograph"
                    .into(),
                width: 256,
                height: 256,
                count: 1,
                steps: Some(1),
                guidance: Some(1.0),
                seed: Some(15800),
                memory: window.map(|window| GenerationMemory {
                    stream_transformer_blocks: true,
                    transformer_window_size: Some(window),
                    transformer_window_component: Some(TransformerComponent::TextEncoder),
                    ..Default::default()
                }),
                ..Default::default()
            };
            clear_cache();
            reset_peak_memory();
            let output = generator.generate_impl(&request, &mut |_| {}).unwrap();
            let peak = get_peak_memory() as u64;
            let image = match output {
                GenerationOutput::Images(mut images) => images.pop().unwrap(),
                other => panic!("expected image output, got {other:?}"),
            };
            drop(generator);
            clear_cache();
            (peak, image)
        };

        let (unscoped_peak, unscoped_image) = run(None);
        let (window_peak, window_image) = run(Some(crate::memory_strategy::TEXT_ENCODER_WINDOW));
        let gib = 1024.0 * 1024.0 * 1024.0;
        println!(
            "SC-15800 Lens Q4 request peak: unscoped={:.3} GiB text-w=1={:.3} GiB",
            unscoped_peak as f64 / gib,
            window_peak as f64 / gib
        );
        assert_eq!(unscoped_image.pixels, window_image.pixels);
        assert!(
            window_peak <= unscoped_peak + unscoped_peak / 20,
            "the measurement path unexpectedly raised Q4 request peak by more than 5%"
        );
    }

    fn window_request(
        component: mlx_gen::gen_core::TransformerComponent,
        window: Option<u32>,
    ) -> GenerationRequest {
        GenerationRequest {
            memory: Some(mlx_gen::gen_core::GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: window,
                transformer_window_component: Some(component),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn text_encoder_scope_requires_a_replayable_source() {
        let request = window_request(
            mlx_gen::gen_core::TransformerComponent::TextEncoder,
            Some(1),
        );
        assert_eq!(resolve_encoder_window(&request, true).unwrap(), Some(1));
        let error = resolve_encoder_window(&request, false).unwrap_err();
        assert!(
            error.to_string().contains("deferred directory load"),
            "refusal must name the replayable-source reason: {error}"
        );
    }

    #[test]
    fn component_scope_maps_to_the_exact_physical_trunks_and_unknown_windows_fail_closed() {
        use mlx_gen::gen_core::TransformerComponent;
        assert_eq!(
            resolve_transformer_windows(
                &window_request(TransformerComponent::TextEncoder, Some(1)),
                true,
                true,
            )
            .unwrap(),
            (Some(1), None)
        );
        assert_eq!(
            resolve_transformer_windows(
                &window_request(TransformerComponent::Dit, Some(1)),
                true,
                true,
            )
            .unwrap(),
            (None, Some(1))
        );
        assert_eq!(
            resolve_transformer_windows(
                &window_request(TransformerComponent::Both, Some(1)),
                true,
                true,
            )
            .unwrap(),
            (Some(1), Some(1))
        );
        let error = resolve_transformer_windows(
            &window_request(TransformerComponent::Both, Some(3)),
            true,
            true,
        )
        .expect_err("an unswept window must fail closed");
        assert!(error
            .to_string()
            .contains("outside the measured production domain"));
    }

    #[test]
    fn an_unselected_request_does_not_stream_and_the_default_is_explicit() {
        assert_eq!(
            resolve_encoder_window(&GenerationRequest::default(), true).unwrap(),
            None
        );
        let request = window_request(mlx_gen::gen_core::TransformerComponent::TextEncoder, None);
        assert_eq!(
            resolve_encoder_window(&request, true).unwrap(),
            Some(crate::memory_strategy::TEXT_ENCODER_WINDOW as usize)
        );
    }

    fn footprint_spec(
        tmp: &tempfile::TempDir,
        quantize: Option<mlx_gen::Quant>,
    ) -> (std::path::PathBuf, LoadSpec) {
        let tier = if quantize.is_some() { "q8" } else { "dense" };
        let root = tmp.path().join(format!(
            "mlx_gen_lens_sc16014_{}_{}",
            tier,
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).expect("tempdir");
        for (component, bytes) in [("text_encoder", 13), ("transformer", 11), ("vae", 3)] {
            let dir = root.join(component);
            std::fs::create_dir(&dir).expect("component dir");
            std::fs::write(dir.join("model.safetensors"), vec![0; bytes]).expect("fixture");
        }
        let mut spec = LoadSpec::new(mlx_gen::WeightsSource::Dir(root.clone()));
        spec.quantize = quantize;
        (root, spec)
    }

    fn write_text_encoder_config(root: &std::path::Path, body: &str) {
        std::fs::write(root.join("text_encoder").join("config.json"), body)
            .expect("text encoder config");
    }

    #[test]
    fn dense_footprint_accounts_for_mxfp4_materialization() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = footprint_spec(&tmp, None);
        write_text_encoder_config(
            &root,
            r#"{"dtype":"bfloat16","quantization_config":{"quant_method":"mxfp4"}}"#,
        );
        let fp = component_footprint(&spec).expect("footprint");
        let gib: f64 = 1024.0 * 1024.0 * 1024.0;
        assert_eq!(fp.text_encoder, (30.07 * gib).ceil() as u64);
        assert_eq!(fp.dit, 11);
        assert_eq!(fp.vae, 3);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn bf16_on_disk_footprint_is_not_inflated_as_mxfp4() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = footprint_spec(&tmp, None);
        write_text_encoder_config(&root, r#"{"dtype":"bfloat16"}"#);
        assert_eq!(
            component_footprint(&spec).expect("footprint"),
            mlx_gen::PerComponentBytes {
                text_encoder: 13,
                dit: 11,
                vae: 3,
            },
            "an explicit bf16-on-disk encoder has no MXFP4 materialization delta"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn unknown_storage_remains_conservative() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = footprint_spec(&tmp, None);
        let gib: f64 = 1024.0 * 1024.0 * 1024.0;
        assert_eq!(
            component_footprint(&spec).expect("footprint").text_encoder,
            (30.07 * gib).ceil() as u64,
            "missing format metadata must not hide a possible MXFP4 materialization"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn quantized_footprint_remains_disk_derived() {
        let tmp = tempfile::tempdir().unwrap();
        for quant in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let (root, spec) = footprint_spec(&tmp, Some(quant));
            assert_eq!(
                component_footprint(&spec).expect("footprint"),
                mlx_gen::PerComponentBytes {
                    text_encoder: 13,
                    dit: 11,
                    vae: 3,
                }
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn packed_turnkey_without_quant_request_remains_disk_derived() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = footprint_spec(&tmp, None);
        std::fs::write(
            root.join("text_encoder").join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .expect("packed marker");
        assert_eq!(
            component_footprint(&spec).expect("footprint").text_encoder,
            13
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn dense_calibration_never_reduces_a_larger_disk_estimate() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = footprint_spec(&tmp, None);
        let larger = (30.07_f64 * 1024.0 * 1024.0 * 1024.0).ceil() as u64 + 1;
        std::fs::OpenOptions::new()
            .write(true)
            .open(root.join("text_encoder").join("model.safetensors"))
            .expect("fixture")
            .set_len(larger)
            .expect("sparse fixture");
        assert_eq!(
            component_footprint(&spec).expect("footprint").text_encoder,
            larger
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// F-030 (sc-11133): the emitted `total` tracks the (possibly truncated) schedule so the bar
    /// reaches its total and the `cur >= total` Decoding trigger fires. Full schedule → `steps`;
    /// PiD early-stop (`keep` σ nodes) → `keep - 1`; degenerate `keep` floors at 1 (never 0).
    #[test]
    fn effective_step_total_tracks_pid_early_stop() {
        assert_eq!(effective_step_total(None, 20), 20, "full schedule = steps");
        // keep=13 σ nodes → 12 steps run and reported (not the requested 20).
        assert_eq!(effective_step_total(Some(13), 20), 12);
        // Degenerate: keep=1 (or 0) must still leave a 1-step bar the Decoding trigger can reach.
        assert_eq!(effective_step_total(Some(1), 20), 1);
        assert_eq!(effective_step_total(Some(0), 20), 1);
    }

    /// F-030 residual (sc-11133): a `keep == 1` schedule runs ZERO real steps (`real_step_count`),
    /// so `render`'s per-step callback never fires. `emit_terminal_if_no_steps` must synthesize a
    /// terminal `Step` reaching total plus exactly one `Decoding`, so the bar completes and Decoding
    /// trips once. A multi-step or full schedule drives its own bar and must emit nothing.
    #[test]
    fn zero_step_schedule_fills_bar_and_fires_decoding_once() {
        // Real transitions actually run: keep-1 (0 for keep<=1), else the full steps.
        assert_eq!(real_step_count(None, 20), 20);
        assert_eq!(real_step_count(Some(13), 20), 12);
        assert_eq!(real_step_count(Some(1), 20), 0, "keep==1 runs 0 real steps");
        assert_eq!(real_step_count(Some(0), 20), 0);

        // keep == 1 (0-step): synthesize the terminal so the bar reaches total and Decoding fires once.
        let mut events: Vec<Progress> = Vec::new();
        let emitted = {
            let mut sink = |p: Progress| events.push(p);
            emit_terminal_if_no_steps(Some(1), 20, &mut sink)
        };
        assert!(emitted, "a 0-step schedule must synthesize a terminal");
        let steps: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|p| match p {
                Progress::Step { current, total } => Some((*current, *total)),
                _ => None,
            })
            .collect();
        let decodings = events
            .iter()
            .filter(|p| matches!(p, Progress::Decoding))
            .count();
        assert_eq!(
            steps,
            vec![(1, 1)],
            "the bar must reach its total so it does not freeze at 0/1"
        );
        assert_eq!(decodings, 1, "Decoding must fire exactly once");

        // A multi-step (keep=13) schedule drives its own bar — no synthetic terminal.
        let mut multi: Vec<Progress> = Vec::new();
        let emitted_multi = {
            let mut sink = |p: Progress| multi.push(p);
            emit_terminal_if_no_steps(Some(13), 20, &mut sink)
        };
        assert!(!emitted_multi, "a multi-step schedule needs no terminal");
        assert!(multi.is_empty());

        // A full schedule (keep == None) likewise drives its own bar.
        let mut full: Vec<Progress> = Vec::new();
        {
            let mut sink = |p: Progress| full.push(p);
            assert!(!emit_terminal_if_no_steps(None, 20, &mut sink));
        }
        assert!(full.is_empty());
    }

    #[test]
    fn descriptors_are_lens() {
        for (d, id, steps, g) in [
            (descriptor_turbo(), MODEL_ID_TURBO, 4u32, 1.0f32),
            (descriptor_base(), MODEL_ID_BASE, 20, 5.0),
        ] {
            assert_eq!(d.id, id);
            assert_eq!(d.family, "lens");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_guidance);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(d.capabilities.conditioning.is_empty());
            // sc-3174: LoRA + LoKr merge into the DiT joint-attention projections at load.
            assert!(d.capabilities.supports_lora);
            assert!(d.capabilities.supports_lokr);
            // sc-3172: encoder MoE experts quantize to Q4/Q8 at load.
            assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
            // sc-7305: the curated sampler/scheduler menu is advertised (the unified framework) with the
            // legacy native aliases retained — both backends (mlx + candle) now expose the same menu.
            assert!(d.capabilities.samplers.contains(&"euler"));
            assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
            assert!(d.capabilities.samplers.contains(&"uni_pc"));
            assert!(d.capabilities.samplers.contains(&"flow_match_euler"));
            assert!(d.capabilities.schedulers.contains(&"karras"));
            assert!(d.capabilities.schedulers.contains(&"exponential"));
            assert!(d.capabilities.schedulers.contains(&"flow_match"));
            // The defaults are exercised end-to-end in the e2e test; assert the constants here.
            let def = if id == MODEL_ID_TURBO {
                TURBO_DEFAULTS
            } else {
                BASE_DEFAULTS
            };
            assert_eq!((def.steps, def.guidance), (steps, g));
        }
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        // The family catalog resolves both ids. Component access is intentionally request-scoped,
        // so a missing snapshot is not touched until generation begins.
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
            let generator = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .unwrap_or_else(|err| panic!("{id} should resolve in the registry; got: {err}"));
            assert_eq!(generator.descriptor().id, id);
        }
    }

    #[test]
    fn load_rejects_unsupported_overlays_not_quant() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
        // A ControlNet overlay is rejected (not part of the Lens port) — the message names it, before
        // any weights load.
        let mut with_control = base.clone();
        with_control.control = Some(WeightsSource::Dir("/nonexistent/cn".into()));
        let err = match load_with(&with_control, TURBO_DEFAULTS) {
            Ok(_) => panic!("control must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not part of the Lens port"), "got: {err}");

        // Quantize is NOT rejected (sc-3172). Construction remains lazy, so the bogus weights path
        // is deferred until generation just like the unquantized path.
        let mut quant = base;
        quant.quantize = Some(Quant::Q8);
        let generator = load_with(&quant, TURBO_DEFAULTS)
            .unwrap_or_else(|err| panic!("quantize must be accepted (sc-3172); got: {err}"));
        assert_eq!(generator.descriptor().id, MODEL_ID_TURBO);
    }

    #[test]
    fn validate_rejects_bad_inputs() {
        let caps = descriptor_turbo().capabilities;
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &ok).is_ok());

        let empty = GenerationRequest {
            prompt: "".into(),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &empty).is_err());

        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &zero_steps).is_err());

        let bad_dims = GenerationRequest {
            width: 1000, // not ÷16
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &bad_dims).is_err());

        // sc-12612: `VAE_SCALE_FACTOR` is the pinned stride SceneWorks ties every advertised Lens
        // image bucket to. Pin the value and mutation-check that a size which is a multiple of 8 (a
        // lower divisor) but not VAE_SCALE_FACTOR (16) is still rejected with the stride error, and
        // an on-stride in-range size passes.
        assert_eq!(VAE_SCALE_FACTOR, 16);
        let off_stride = validate_request(
            MODEL_ID_TURBO,
            &caps,
            &GenerationRequest {
                width: 1000, // 125×8 — a multiple of 8 but not VAE_SCALE_FACTOR
                ..ok.clone()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(
            MODEL_ID_TURBO,
            &caps,
            &GenerationRequest {
                width: 1024, // 64×16 — on-stride
                ..ok.clone()
            }
        )
        .is_ok());
    }

    // Request-scoped residency keeps both load policies lazy so a later rung-4 request can choose a
    // different physical materialization shape without first loading a full warm pair. The generator
    // retains `offload_policy` as `default_stage_residency` for unscoped compatibility requests.
    fn missing_snapshot_spec(policy: mlx_gen::OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/lens-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    // ── F-010 (sc-12462): requested-vs-packed quant-tier guard. Lens ships pre-quantized packed
    // turnkeys (sc-8763) whose converter writes the `"quantization": {"bits"}` marker into BOTH
    // quantized component dirs (`transformer/`, `text_encoder/`); the packed load paths infer bits
    // from the on-disk shapes and the load-time `quantize` no-ops, so a Q4 request over a Q8
    // turnkey would silently serve Q8 in both components, on both policies. Weight-free fixtures:
    // only the component `config.json` markers are written.

    /// Temp snapshot root with a Q8 marker in each of `components` (others absent = dense).
    fn tier_fixture(tmp: &tempfile::TempDir, components: &[&str], bits: i32) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "lens-registry-tier-{}-{:?}",
            components.join("-"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        for c in components {
            let dir = root.join(c);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("config.json"),
                format!(r#"{{"quantization": {{"bits": {bits}, "group_size": 64}}}}"#),
            )
            .unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn q4_spec(root: &std::path::Path, policy: mlx_gen::OffloadPolicy) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.into()));
        spec.quantize = Some(Quant::Q4);
        spec.with_offload_policy(policy)
    }

    /// Q4-over-Q8 must hard-error for the **DiT**: `load_heavy_phase` checks the `transformer/`
    /// marker BEFORE any weights load (the projections would otherwise load packed Q8 and
    /// `quantize_dit` no-op).
    #[test]
    fn heavy_phase_rejects_q4_over_q8_turnkey() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tier_fixture(&tmp, &["transformer"], 8);
        let spec = q4_spec(&root, mlx_gen::OffloadPolicy::Resident);
        let err = load_heavy_phase(&spec, &root, Dtype::Bfloat16, false, MODEL_ID_BASE)
            .err()
            .expect("Q4 over a packed Q8 DiT must error");
        let msg = err.to_string();
        assert!(
            msg.contains("pre-quantized Q8") && msg.contains("transformer"),
            "expected the DiT tier-mismatch error, got: {msg}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// Q4-over-Q8 must hard-error for the **gpt-oss encoder**: `load_text_phase` checks the
    /// `text_encoder/` marker BEFORE any weights load (`from_weights_quant` would otherwise build
    /// `ExpertBank::Quant` at the on-disk Q8, never consulting the request).
    #[test]
    fn text_phase_rejects_q4_over_q8_turnkey() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tier_fixture(&tmp, &["text_encoder"], 8);
        let spec = q4_spec(&root, mlx_gen::OffloadPolicy::Resident);
        let err = load_text_phase(&spec, &root, Dtype::Bfloat16, MODEL_ID_BASE)
            .err()
            .expect("Q4 over a packed Q8 encoder must error");
        let msg = err.to_string();
        assert!(
            msg.contains("pre-quantized Q8") && msg.contains("text_encoder"),
            "expected the encoder tier-mismatch error, got: {msg}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// The guard covers BOTH registry ids end-to-end: a `lens` / `lens_turbo` load of a Q8 turnkey
    /// with Q4 requested fails with the tier-mismatch error (not a missing-weights error).
    #[test]
    fn both_ids_reject_q4_over_q8_turnkey() {
        let tmp = tempfile::tempdir().unwrap();
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let root = tier_fixture(&tmp, &["transformer", "text_encoder"], 8);
            let spec = q4_spec(&root, mlx_gen::OffloadPolicy::Resident);
            let err = match crate::provider_registry().unwrap().load(id, &spec) {
                Ok(_) => panic!("{id}: Q4 over a packed Q8 turnkey must fail to load"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("pre-quantized Q8") && err.contains(id),
                "{id}: expected the tier-mismatch error, got: {err}"
            );
            std::fs::remove_dir_all(&root).ok();
        }
    }

    /// `Sequential` defers the phase loaders to the first generate, so the mismatch must be caught
    /// by the up-front `build_residency` check — at LOAD time, not mid-job.
    #[test]
    fn sequential_fails_fast_on_tier_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tier_fixture(&tmp, &["transformer", "text_encoder"], 8);
        let err = build_residency(
            &q4_spec(&root, mlx_gen::OffloadPolicy::Sequential),
            MODEL_ID_BASE,
        )
        .err()
        .expect("Sequential must fail-fast on a tier mismatch at load, not at first generate");
        assert!(err.to_string().contains("pre-quantized Q8"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Pinned per sibling semantics: a matching request (Q8 over a Q8 turnkey) and a no-quantize
    /// request over a packed turnkey both pass the guard (the turnkey loads packed at its shipped
    /// tier). Weight-free via `Sequential`, which runs only the up-front checks.
    #[test]
    fn matching_or_absent_request_passes_the_guard() {
        let tmp = tempfile::tempdir().unwrap();
        // Q8 over Q8: no tier error (build succeeds — Sequential touches no weights).
        let root = tier_fixture(&tmp, &["transformer", "text_encoder"], 8);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential);
        spec.quantize = Some(Quant::Q8);
        build_residency(&spec, MODEL_ID_BASE)
            .expect("Q8 over a packed Q8 turnkey must pass the tier guard");

        // No quantize requested over a packed turnkey: guard not consulted, load proceeds.
        spec.quantize = None;
        build_residency(&spec, MODEL_ID_BASE)
            .expect("a packed turnkey with no quantize requested must load at its shipped tier");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn build_residency_defers_component_loads_for_both_load_defaults() {
        for policy in [
            mlx_gen::OffloadPolicy::Sequential,
            mlx_gen::OffloadPolicy::Resident,
        ] {
            let res = build_residency(&missing_snapshot_spec(policy), MODEL_ID_BASE)
                .expect("request-scoped owner must defer component loads");
            assert!(
                !res.is_sequential(),
                "phase staging is selected per request, not baked into the shared owner"
            );
        }
    }
}
