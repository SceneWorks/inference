//! The XLabs FLUX **IP-Adapter** provider (sc-5872, epic 5480) — reference-image (identity)
//! conditioning on FLUX.1 `dev`/`schnell`, the candle (Windows/CUDA) sibling of `mlx-gen-flux`'s XLabs
//! IP path. It composes the reused FLUX text encoders / VAE / flow-match schedule with the forked DiT
//! ([`crate::ip_dit::IpFlux`], the only FLUX DiT with an IP seam) + the XLabs adapter
//! ([`crate::ip_adapter`]) + the pooled CLIP-ViT-L image encoder ([`crate::ip_image_encoder`]).
//!
//! **Single distilled forward** (no true-CFG): FLUX is guidance/timestep-distilled, so — like the
//! candle txt2img path — each denoise step is a single DiT forward (dev embeds the guidance scalar;
//! schnell ignores it), with the XLabs IP residual injected per double block. The reference's identity
//! tokens are computed **once** (constant across the denoise) and bound into a [`FluxIpInjector`] at
//! `ip_adapter_scale`; at `scale = 0` the forked DiT is byte-identical to the stock FLUX path — the
//! no-IP arm of the validation ablation (`crate::ip_validate`).
//!
//! The provider is a plain struct driven **directly** by the worker (a bespoke reference stream, like
//! `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered `Generator` — the
//! registered `flux1_*` descriptors stay txt2img-only.

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::flux::sampling::State;
use std::path::{Path, PathBuf};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    AdapterSpec, Conditioning, ConditioningKind, GenerationMemory, GenerationRequest, Image,
    LoadSpec, MemoryGeometry, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemoryRunOutcome, ModelDescriptor, PreviewSink, Progress, RenderLoadBinding, RenderPlan,
    RenderRun, WeightsSource,
};
use candle_gen::preview::PreviewHook;
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};

use crate::flux1_load;
use crate::ip_adapter::{FluxIpAdapter, FluxIpInjector};
use crate::ip_image_encoder::FluxIpImageEncoder;
use crate::ref_backbone::{FluxRefBackbone, FluxRefHeavy};
use crate::Variant;

/// FLUX runs at bf16.
const DTYPE: DType = DType::BF16;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
// FLUX dev's flow-match time-shift endpoints (`BASE_SHIFT`/`MAX_SHIFT`) are shared from
// `crate::pipeline` (sc-11249 / F-140) rather than re-declared here.

/// Default `ip_adapter_scale` for the XLabs FLUX IP-Adapter (mlx-gen-flux `DEFAULT_IP_SCALE`).
pub const DEFAULT_IP_SCALE: f32 = 0.7;

/// FLUX packs the /8 VAE latent 2×2, so both render dims must be multiples of 16 (matching the flux1
/// txt2img `validate` and the control/edit siblings). Single source of truth = the crate-root
/// [`crate::SIZE_MULTIPLE`] (sc-12612).
use crate::SIZE_MULTIPLE;

/// Paths to the FLUX IP-Adapter checkpoints.
#[derive(Clone)]
pub struct IpAdapterFluxPaths {
    /// The black-forest-labs FLUX.1 snapshot dir (`flux1-{dev,schnell}.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`). The variant is detected from which DiT file
    /// is present.
    pub flux_base: PathBuf,
    /// The XLabs adapter (`XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`: `ip_adapter_proj_model.*`
    /// + `double_blocks.{0..18}.processor.ip_adapter_double_stream_{k,v}_proj.*`).
    pub ip_adapter: PathBuf,
    /// The CLIP ViT-L/14 image encoder (`openai/clip-vit-large-patch14`) — a dir (`model.safetensors`)
    /// or the file directly.
    pub image_encoder: PathBuf,
    pub adapters: Vec<AdapterSpec>,
}

/// One FLUX IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterFluxRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,
    /// Guidance scale — embedded by the dev DiT, inert on schnell.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled-cross-attn weight on the reference image tokens).
    pub ip_adapter_scale: f32,
    pub seed: u64,
    /// Shared FLUX memory ladder selected by worker admission.
    pub memory: GenerationMemory,
    /// Per-step latent-preview sink (epic 16948, sc-16956) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which this provider cannot carry because it is worker-invoked by name rather than through a
    /// registered descriptor. Default is inert, and an inert sink makes a seeded render byte-identical
    /// to one with no preview at all.
    pub preview: PreviewSink,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterFluxRequest {
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
            ip_adapter_scale: DEFAULT_IP_SCALE,
            seed: 0,
            memory: GenerationMemory::default(),
            preview: PreviewSink::default(),
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the CLIP image-encoder weight file from a dir-or-file path (a file is used directly; a dir
/// resolves `model.safetensors` then `model.fp16.safetensors`).
fn resolve_image_encoder(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in ["model.safetensors", "model.fp16.safetensors"] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "flux ip-adapter: CLIP image encoder not found under {} (expected a model.safetensors or a \
         direct .safetensors file)",
        path.display()
    )))
}

/// Detect the FLUX variant from the snapshot by which DiT checkpoint is present (dev preferred if both).
fn detect_variant(flux_base: &Path) -> Result<Variant> {
    if flux_base.join(Variant::Dev.transformer_file()).is_file() {
        Ok(Variant::Dev)
    } else if flux_base
        .join(Variant::Schnell.transformer_file())
        .is_file()
    {
        Ok(Variant::Schnell)
    } else if flux_base.join("transformer/config.json").is_file() {
        let path = flux_base.join("transformer/config.json");
        let bytes = candle_gen::gen_core::__model_input_reader!(
std::fs::read(&path),
"crates/media/candle-gen/candle-gen-flux/src/ip_provider.rs::detect_variant::std::fs::read@6286",
ArtifactInput,
&path,
|__model_input_path| std::fs::read(__model_input_path)
)
        .map_err(|error| {
            CandleError::Msg(format!(
                "flux ip-adapter: read packed transformer config {}: {error}",
                path.display()
            ))
        })?;
        let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            CandleError::Msg(format!(
                "flux ip-adapter: parse packed transformer config {}: {error}",
                path.display()
            ))
        })?;
        Ok(
            if config
                .get("guidance_embeds")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                Variant::Dev
            } else {
                Variant::Schnell
            },
        )
    } else {
        Err(CandleError::Msg(format!(
            "flux ip-adapter: no BFL checkpoint or packed transformer/config.json in {}",
            flux_base.display()
        )))
    }
}

/// The loaded FLUX IP-Adapter model: the reused FLUX text encoders + VAE, the forked IP DiT, the XLabs
/// adapter, and the CLIP ViT-L image encoder.
pub struct IpAdapterFlux {
    variant: Variant,
    device: Device,
    dtype: DType,
    backbone: FluxRefBackbone,
    ip_encoder: FluxIpImageEncoder,
    adapter: FluxIpAdapter,
    loaded_paths: IpAdapterFluxPaths,
    loaded_spec: LoadSpec,
    memory: GenerationMemory,
    memory_contract: Option<MemoryProviderContract>,
    admitted_context: Option<MemoryRunContext>,
}

fn validate_ip_memory_context(context: &MemoryRunContext) -> Result<()> {
    if !crate::memory_strategy::accepts_ip_public_mode(&context.mode)
        || context.overlay.as_deref() != Some("identity")
        || !context.has_reference
        || context.use_pid
        || context.has_phases
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.reference_count != 1
    {
        return Err(CandleError::Msg(
            "FLUX.1 IP memory context is not one supported public identity reference".into(),
        ));
    }
    Ok(())
}

fn validate_ip_spec(
    paths: &IpAdapterFluxPaths,
    spec: &LoadSpec,
) -> candle_gen::gen_core::Result<()> {
    if spec.weights != WeightsSource::Dir(paths.flux_base.clone())
        || spec.ip_adapter != Some(WeightsSource::File(paths.ip_adapter.clone()))
        || !matches!(spec.components.get("flux_ip_image_encoder"), Some(WeightsSource::Dir(path) | WeightsSource::File(path)) if *path == paths.image_encoder)
        || spec.components.len() != 1
        || spec.adapters != paths.adapters
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.pid.is_some()
        || spec.text_encoder.is_some()
        || spec.quantize.is_some()
        || spec.precision != candle_gen::gen_core::Precision::Bf16
    {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "FLUX.1 IP load contract differs from resolved paths".into(),
        ));
    }
    Ok(())
}

fn ip_descriptor(variant: Variant) -> ModelDescriptor {
    let mut descriptor = match variant {
        Variant::Dev => crate::descriptor_dev(),
        Variant::Schnell => crate::descriptor_schnell(),
    };
    descriptor.id = if variant.is_dev() {
        "flux1_dev_ip_adapter"
    } else {
        "flux1_schnell_ip_adapter"
    };
    descriptor.capabilities.conditioning = vec![ConditioningKind::Reference];
    descriptor.capabilities.max_count = 1;
    descriptor
}

fn typed_ip_request(
    req: &IpAdapterFluxRequest,
    reference: &Image,
    variant: Variant,
) -> candle_gen::gen_core::Result<GenerationRequest> {
    let steps = u32::try_from(req.steps).map_err(|_| {
        candle_gen::gen_core::Error::Unsupported("FLUX.1 IP step count exceeds u32".into())
    })?;
    Ok(GenerationRequest {
        prompt: req.prompt.clone(),
        width: req.width,
        height: req.height,
        count: 1,
        steps: Some(steps),
        seed: Some(req.seed),
        guidance: variant.supports_guidance().then_some(req.guidance),
        sampler: req.sampler.clone(),
        scheduler: req.scheduler.clone(),
        scheduler_shift: req.scheduler_shift,
        conditioning: vec![Conditioning::Reference {
            image: reference.clone(),
            strength: Some(req.ip_adapter_scale),
        }],
        memory: Some(req.memory),
        preview: req.preview.clone(),
        cancel: req.cancel.clone(),
        ..Default::default()
    })
}

/// Freeze the XLabs reference, encoder, adapter, variant and native trajectory before load.
pub fn compile_ip_plan(
    paths: &IpAdapterFluxPaths,
    spec: &LoadSpec,
    req: &IpAdapterFluxRequest,
    reference: &Image,
) -> candle_gen::gen_core::Result<RenderPlan> {
    validate_ip_spec(paths, spec)?;
    let variant = detect_variant(&paths.flux_base)
        .map_err(|error| candle_gen::gen_core::Error::Unsupported(error.to_string()))?;
    let typed = typed_ip_request(req, reference, variant)?;
    let descriptor = ip_descriptor(variant);
    descriptor
        .capabilities
        .validate_request(descriptor.id, &typed)?;
    validate_request(req)
        .map_err(|error| candle_gen::gen_core::Error::Unsupported(error.to_string()))?;
    if req.cancel.is_cancelled() {
        return Err(candle_gen::gen_core::Error::Canceled);
    }
    crate::compile_conditioned_plan(variant, &descriptor, spec, &typed)
}

impl IpAdapterFlux {
    /// Load the FLUX backbone (text encoders + forked DiT + VAE) + the XLabs adapter + the CLIP ViT-L
    /// image encoder from a FLUX snapshot, the XLabs `ip_adapter.safetensors`, and a CLIP image encoder.
    pub fn load(paths: &IpAdapterFluxPaths) -> Result<Self> {
        Self::load_with_memory(paths, GenerationMemory::default())
    }

    pub fn load_with_memory(paths: &IpAdapterFluxPaths, memory: GenerationMemory) -> Result<Self> {
        let encoder = if paths.image_encoder.is_dir() {
            WeightsSource::Dir(paths.image_encoder.clone())
        } else {
            WeightsSource::File(paths.image_encoder.clone())
        };
        let spec = LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
            .with_ip_adapter(WeightsSource::File(paths.ip_adapter.clone()))
            .with_component("flux_ip_image_encoder", encoder)
            .with_adapters(paths.adapters.clone());
        Self::load_with_memory_spec(paths, memory, &spec)
    }

    pub fn load_with_memory_spec(
        paths: &IpAdapterFluxPaths,
        memory: GenerationMemory,
        spec: &LoadSpec,
    ) -> Result<Self> {
        Self::load_internal(paths, memory, None, None, spec)
    }

    pub fn load_with_memory_context_spec(
        paths: &IpAdapterFluxPaths,
        context: MemoryRunContext,
        spec: &LoadSpec,
    ) -> Result<Self> {
        validate_ip_spec(paths, spec)?;
        validate_ip_memory_context(&context)?;
        let variant = detect_variant(&paths.flux_base)?;
        let provider_id = variant.model_id();
        let contract = crate::memory_strategy::provider_contract(provider_id, spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let tier = crate::memory_strategy::resolved_numeric_tier(spec, provider_id)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(&contract, &context, tier.quant)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        Self::load_internal(paths, memory, Some(contract), Some(context), spec)
    }

    fn load_internal(
        paths: &IpAdapterFluxPaths,
        memory: GenerationMemory,
        memory_contract: Option<MemoryProviderContract>,
        admitted_context: Option<MemoryRunContext>,
        spec: &LoadSpec,
    ) -> Result<Self> {
        validate_ip_spec(paths, spec)?;
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let root = paths.flux_base.clone();
        let variant = detect_variant(&root)?;

        // CLIP-L + T5-XXL text encoders (shared FLUX.1 backbone load, sc-9003).
        let backbone = FluxRefBackbone::load_with_memory(
            &root,
            variant,
            &device,
            dtype,
            memory,
            paths.adapters.clone(),
        )?;

        // The forked FLUX DiT (the IP seam) from the root BFL checkpoint — the genuine per-provider drift
        // (IpFlux, not the stock Flux), so the wrapper choice stays here over the shared mmap.
        // XLabs adapter weights (`ip_adapter.safetensors`).
        let ipa = Weights::from_file(&paths.ip_adapter, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load adapter {:?}: {e}",
                paths.ip_adapter
            ))
        })?;
        let adapter = FluxIpAdapter::from_weights(&ipa)?;
        if adapter.num_blocks() != 19 {
            return Err(CandleError::Msg(format!(
                "flux ip-adapter: adapter has {} double-block pairs but FLUX.1 has 19 double blocks",
                adapter.num_blocks(),
            )));
        }

        // CLIP ViT-L/14 image encoder (`vision_model.*` + `visual_projection.*`).
        let enc_path = resolve_image_encoder(&paths.image_encoder)?;
        let enc_w = Weights::from_file(&enc_path, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        let ip_encoder = FluxIpImageEncoder::from_weights(&enc_w)?;

        Ok(Self {
            variant,
            device,
            dtype,
            backbone,
            ip_encoder,
            adapter,
            loaded_paths: paths.clone(),
            loaded_spec: spec.clone(),
            memory,
            memory_contract,
            admitted_context,
        })
    }

    /// Reference-image T2I: condition the FLUX generation on `reference`'s CLIP-ViT-L identity tokens at
    /// `req.ip_adapter_scale` (a single distilled forward per step — no true-CFG).
    pub fn generate(
        &self,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = compile_ip_plan(&self.loaded_paths, &self.loaded_spec, req, reference)?;
        self.generate_planned(req, reference, &plan, on_progress)
    }

    pub fn generate_planned(
        &self,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if self.admitted_context.is_some() {
            return Err(CandleError::Msg(
                "FLUX.1 IP admitted model requires a memory context".into(),
            ));
        }
        self.generate_planned_scoped(req, reference, plan, None, on_progress)
    }

    pub fn generate_with_memory_context_planned(
        &self,
        context: &MemoryRunContext,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let admitted = self.admitted_context.as_ref().ok_or_else(|| {
            CandleError::Msg("FLUX.1 IP model has no admitted memory context".into())
        })?;
        let contract = self
            .memory_contract
            .as_ref()
            .ok_or_else(|| CandleError::Msg("FLUX.1 IP model lost its memory contract".into()))?;
        validate_ip_memory_context(context)?;
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
                "FLUX.1 IP request changed after memory admission".into(),
            ));
        }
        contract
            .validate_selection(&context.selection)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let mut scope = crate::memory_strategy::FluxMemoryScope::new(
            self.variant.model_id(),
            self.device.clone(),
            contract,
            context,
        );
        let result =
            self.generate_planned_scoped(req, reference, plan, Some(&mut scope), on_progress);
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

    fn generate_planned_scoped(
        &self,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        plan: &RenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let typed = typed_ip_request(req, reference, self.variant)?;
        plan.check_request(&ip_descriptor(self.variant), &typed)?;
        plan.check_load(&RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let run = RenderRun::new(plan, &req.cancel, scope);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        validate_request(req)?;

        // Conditioning: text (T5 seq + CLIP pooled) and the reference image tokens (computed once).
        self.backbone
            .validate_native_vae_request(false, &req.cancel)?;
        let (t5_emb, clip_emb, tokens) = run.encode(|| {
            let (t5, clip) = self.backbone.encode_text_with_memory_and_progress(
                &req.prompt,
                &req.cancel,
                on_progress,
            )?;
            candle_gen::check_cancel(&req.cancel)?;
            let embeds = self
                .ip_encoder
                .image_embeds(reference)?
                .to_dtype(self.dtype)?;
            candle_gen::check_cancel(&req.cancel)?;
            let tokens = self.adapter.tokens(&embeds)?;
            Ok::<_, CandleError>((t5, clip, tokens))
        })?;
        let injector = FluxIpInjector::new(&self.adapter, tokens, req.ip_adapter_scale as f64);
        let heavy = self.backbone.load_heavy(&req.cancel)?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request. sc-3673 parity:
        // deterministic, launch-portable CPU-seeded initial noise (shared FLUX.1 helper, sc-9003).
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
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
                        "FLUX.1 IP packed geometry changed after planning".into(),
                    ));
                }
                let preview = crate::preview::hook(&req.preview, req.width, req.height);
                self.denoise(
                    req,
                    &state,
                    plan,
                    &run,
                    &injector,
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
            .ok_or_else(|| CandleError::Msg("FLUX.1 IP plan produced no image".into()))
    }

    /// The flow-match denoise with the XLabs IP injector, routed through the unified curated
    /// sampler/scheduler driver (epic 7114 P4, sc-7123) — the IP twin of the txt2img
    /// [`crate::pipeline`] `denoise`. The `scheduler` axis re-strides FLUX's native `get_schedule(..)`
    /// over the time-shift `mu`; the `sampler` axis picks the integrator. The forked
    /// [`IpFlux::forward`] (`Some(injector)`) replaces the stock FLUX forward, and the XLabs IP residual
    /// injection stays INSIDE the `predict` closure so a multi-eval solver re-runs the whole step. The
    /// DEFAULT (`euler` over the native schedule) is the N1 no-op for the legacy inline flow-match Euler
    /// loop `img += pred·(σ_{i+1} − σ_i)`. FLUX feeds the raw timestep (`Sigma` convention: `t == σ`);
    /// guidance is a per-batch tensor only embedded by the dev DiT. The provider request carries no
    /// sampler/scheduler knob (the worker drives this stream directly), so it defaults to the native
    /// flow-match Euler path. Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        _req: &IpAdapterFluxRequest,
        state: &State,
        render_plan: &RenderPlan,
        run: &RenderRun<'_, '_>,
        injector: &FluxIpInjector,
        heavy: Option<&FluxRefHeavy>,
        seed: u64,
        preview: &PreviewHook<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
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
                // The forked DiT forward returns a `candle_core::Result`; `?` bridges it into the
                // driver's `CandleError`. The XLabs IP residual injection lives inside this closure.
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                self.backbone.forward_ip_prepared_with_memory(
                    heavy,
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    injector,
                    &prepared,
                    cancel,
                )
            },
        )
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard
/// (sc-9171, the sc-8646 bug class) mirrors the flux1 **control** provider
/// ([`crate::control_provider`]) and the registered flux1 txt2img `validate`. flux1-IP conditions on
/// **T5 + CLIP** (`encode_text`): an empty prompt reaches the T5 encoder as an all-pad sequence and
/// CLIP as bare BOS/EOS, i.e. degenerate identity-free conditioning rather than the intended
/// text-plus-reference blend. Rejecting it up front turns a silent quality collapse (or, on the CLIP
/// side, the deeper `"empty CLIP tokenization"` error in `encode_text`) into a clean, actionable
/// validation error. The provider is a single distilled forward (no true-CFG), so there is no
/// negative/uncond prompt — the exposure is solely the positive prompt.
fn validate_request(req: &IpAdapterFluxRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg(
            "flux ip-adapter: prompt is required".into(),
        ));
    }
    // Bring this lane up to parity with its three siblings (flux1 txt2img/control/edit): a
    // multiple-of-16 size floor and a `steps == 0` reject. Without them `get_schedule(0, …)` builds an
    // empty flow-match trajectory (zero sampler steps ⇒ pure seeded noise decoded and returned as
    // success), burning GPU time for garbage rather than the fast typed error the siblings give
    // (sc-9016/F-032 established the guard; sc-11182/F-102 sweeps it here).
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "flux ip-adapter: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg(
            "flux ip-adapter: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ip_plan_binds_reference_and_encoder_before_loading_weights() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join(Variant::Dev.transformer_file()),
            b"header only",
        )
        .unwrap();
        let paths = IpAdapterFluxPaths {
            flux_base: root.path().to_path_buf(),
            ip_adapter: "/missing/ip.safetensors".into(),
            image_encoder: "/missing/clip".into(),
            adapters: Vec::new(),
        };
        let spec = LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
            .with_ip_adapter(WeightsSource::File(paths.ip_adapter.clone()))
            .with_component(
                "flux_ip_image_encoder",
                WeightsSource::Dir(paths.image_encoder.clone()),
            );
        let req = IpAdapterFluxRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: 13,
            ..Default::default()
        };
        let mut reference = Image {
            width: 16,
            height: 16,
            pixels: vec![3; 16 * 16 * 3],
        };
        let plan = compile_ip_plan(&paths, &spec, &req, &reference).unwrap();
        assert_eq!(plan.base_seed(), 13);
        crate::memory_strategy::assert_conditioned_plan_trace(&plan, &req.cancel);
        reference.pixels[0] ^= 1;
        assert!(plan
            .check_request(
                &ip_descriptor(Variant::Dev),
                &typed_ip_request(&req, &reference, Variant::Dev).unwrap()
            )
            .is_err());
        let mut changed = spec.clone();
        changed.components.insert(
            "flux_ip_image_encoder".into(),
            WeightsSource::Dir("/other/clip".into()),
        );
        assert!(plan
            .check_load(&RenderLoadBinding::from_spec(&changed))
            .is_err());
        let mut missing_overlay = spec.clone();
        missing_overlay.ip_adapter = None;
        assert!(compile_ip_plan(&paths, &missing_overlay, &req, &reference).is_err());
        let mut invalid = req.clone();
        invalid.steps = 0;
        assert!(compile_ip_plan(&paths, &spec, &invalid, &reference).is_err());
        invalid.steps = req.steps;
        invalid.width = 513;
        assert!(compile_ip_plan(&paths, &spec, &invalid, &reference).is_err());
    }

    /// The empty-prompt guard (sc-9171, the sc-8646 bug class): an empty or whitespace-only prompt is
    /// a clean validation error — never a degenerate all-pad T5 / bare CLIP encode — while a real
    /// prompt passes. flux1-IP conditions on T5+CLIP, so this is a genuinely degenerate site (unlike
    /// the CLIP-fixed-pad SDXL edit/IP or ChatGLM-framed Kolors-IP siblings audited in sc-9171).
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = IpAdapterFluxRequest::default();
        assert!(empty.prompt.is_empty());
        let err = validate_request(&empty).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let whitespace = IpAdapterFluxRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(&whitespace).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let ok = IpAdapterFluxRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// Parity with the three flux1 siblings (sc-11182, F-102): `steps == 0` and a non-multiple-of-16
    /// size are fast typed errors (never a decoded pure-noise "success"); the defaults pass.
    #[test]
    fn validate_request_floors_steps_and_size() {
        let base = IpAdapterFluxRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };

        let zero_steps = IpAdapterFluxRequest {
            steps: 0,
            ..base.clone()
        };
        let err = validate_request(&zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");

        let bad_size = IpAdapterFluxRequest {
            width: 1000, // not a multiple of 16
            ..base.clone()
        };
        let err = validate_request(&bad_size).unwrap_err();
        assert!(err.to_string().contains("multiples of 16"), "{err}");

        assert!(validate_request(&base).is_ok());
    }

    /// The request defaults match the FLUX dev IP-Adapter knobs (1024², 25 steps, guidance 3.5, ip 0.7).
    #[test]
    fn request_defaults() {
        let r = IpAdapterFluxRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, 3.5);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a missing dir errors loudly;
    /// a direct file is used as-is.
    #[test]
    fn image_encoder_resolution() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(resolve_image_encoder(&dir).is_err());
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
    }

    /// `detect_variant` keys off the DiT checkpoint filename and errors when neither is present.
    #[test]
    fn variant_detection() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(detect_variant(&dir).is_err());
        std::fs::write(dir.join(Variant::Schnell.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Schnell);
        std::fs::write(dir.join(Variant::Dev.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Dev); // dev preferred if both

        // Clear the checkpoint files so the next leg exercises the transformer/config.json
        // route rather than re-reading the filenames above. A reset, not cleanup.
        std::fs::remove_file(dir.join(Variant::Schnell.transformer_file())).unwrap();
        std::fs::remove_file(dir.join(Variant::Dev.transformer_file())).unwrap();
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(
            dir.join("transformer/config.json"),
            br#"{"guidance_embeds":false}"#,
        )
        .unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Schnell);
        std::fs::write(
            dir.join("transformer/config.json"),
            br#"{"guidance_embeds":true}"#,
        )
        .unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Dev);
    }
}
