//! `Flux3Klein` — the distilled FLUX 3 Image Klein text-to-image [`Generator`]: descriptor, load,
//! request validation and the staged generate (text encoder → DiT + autoencoder), registered as
//! [`MODEL_ID`].

use std::path::{Path, PathBuf};

use gen_core::flux3::{
    prompt_seed, validate_generation_request, Flux3Config, GenerationParams, GenerationResources,
    FAMILY, SAMPLERS,
};
use gen_core::GenerationReport;
use mlx_gen::weights::Weights;
use mlx_gen::{
    gen_core, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, Residency, Result, SizeFloor,
};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::Flux3Dit;
use crate::pipeline::{decode_tokens, denoise_one_with_references};
use crate::text_encoder::{Flux3TextEncoder, TextConditioning};
use gen_core::flux3::references::{self, PreparedReferences};

/// Registry id (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = gen_core::flux3::DISTILLED_MODEL_ID;
/// Output sides must be multiples of the token size.
pub const SIZE_MULTIPLE: u32 = gen_core::flux3::DOWNSAMPLING_RATIO;
/// Smallest side (`pipeline.MIN_SIDE`).
pub const MIN_SIZE: u32 = gen_core::flux3::MIN_SIDE;
/// Largest side (`pipeline.MAX_SIDE`).
pub const MAX_SIZE: u32 = gen_core::flux3::MAX_SIDE;
/// Images per request (rendered one after another, each from `seed + i`).
pub const MAX_COUNT: u32 = 8;

fn new_image_fence() -> gen_core::core_llm::text_llm::OwnedExecutionFence<mlx_rs::Stream> {
    gen_core::core_llm::text_llm::OwnedExecutionFence::new(
        mlx_rs::Stream::cpu(),
        mlx_rs::Stream::task_local_or_default(),
    )
}
fn synchronize_image_stream(stream: &mlx_rs::Stream) -> Result<()> {
    mlx_llm::primitives::synchronize_stream(stream).map_err(|error| {
        Error::Msg(format!(
            "FLUX3 image stream fence failed; no release claimed: {error}"
        ))
    })
}

fn fenced_image_load<T>(load: impl FnOnce() -> Result<T>) -> Result<T> {
    // Created INSIDE the Send+Sync loader invocation; no non-Send stream handle
    // is captured by its reusable path/config closure.
    let streams = new_image_fence();
    let primary = load();
    if primary.is_err() {
        if let Err(cleanup) = streams.synchronize(synchronize_image_stream) {
            eprintln!("FLUX3 partial image-load cleanup fence failed; primary error preserved, no release claimed: {cleanup}");
        }
    }
    primary
}

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // The packed 16× latent space of FLUX 3's own autoencoder (BatchNorm-stat tokens): no
        // shared decoder consumes it, so nothing is advertised and every decoder swap fails closed.
        denoiser_output_latent_space: None,
        control_kinds: None,
        // Both components default to the package's bundled copies; neither is required.
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // A flow map: no CFG, so no negative prompt and no guidance scale (upstream refuses
            // a guidance on the distilled checkpoint).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // The one integrator, the flow-map Euler step; the schedule has no name and its
            // alpha is fixed, so no scheduler is advertised.
            samplers: SAMPLERS.to_vec(),
            schedulers: Vec::new(),
            // Compatible LoRA stacks are admitted by exact base/revision, rank/alpha and target mapping (S6).
            supports_lora: true,
            supports_lokr: false,
            // No per-step preview: the packed latent has no fitted RGB projection and the
            // autoencoder decode is the only exact view of a step.
            supports_preview: false,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: MAX_COUNT,
            mac_only: true,
            // Quantized variants are sc-26241.
            supported_quants: &[],
            supports_sequential_offload: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            // sc-26238: `enhance_prompt` runs the separate FLUX 3 prompt upsampler (the optional
            // `prompt_upsampler` component) before encoding; without it the request is a typed
            // refusal, never a silent pass-through.
            supports_prompt_enhancement: true,
            conditioning: vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            ..Default::default()
        },
    }
}

/// Base checkpoint public surface: CFG and independently encoded negative conditioning.
pub fn descriptor_base() -> ModelDescriptor {
    let mut descriptor = descriptor();
    descriptor.id = gen_core::flux3::BASE_MODEL_ID;
    descriptor.capabilities.supports_guidance = true;
    // CFG uses the standard guidance axis; the separate true_cfg control is not consumed.
    descriptor.capabilities.supports_true_cfg = false;
    descriptor.capabilities.supports_negative_prompt = true;
    descriptor
}

/// The heavy half: the DiT and the autoencoder.
pub struct Heavy {
    pub dit: Flux3Dit,
    pub ae: Flux3AutoEncoder,
}

impl gen_core::StagedHeavy for Heavy {
    type Light = Flux3AutoEncoder;
    type DecodeView<'a> = &'a Flux3AutoEncoder;
    fn shed_dit(self) -> Self::Light {
        let Self { dit, ae } = self;
        drop(dit);
        ae
    }
    fn decode_view(&self) -> Self::DecodeView<'_> {
        &self.ae
    }
    fn light_view(light: &Self::Light) -> Self::DecodeView<'_> {
        light
    }
}

/// A loaded distilled generator.
pub struct Flux3Klein {
    descriptor: ModelDescriptor,
    adapter_reports: Vec<gen_core::AdapterApplyReport>,
    config: Flux3Config,
    residency: Residency<Flux3TextEncoder, Heavy>,
    /// The spec this generator was loaded from: the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against.
    spec: LoadSpec,
    /// The loaded memory contract, published through [`Generator::memory_strategy_contract`].
    memory_strategy: gen_core::MemoryProviderContract,
    /// No VLM weights or prefix cache exist until an enabled prepare phase.
    upsampler_dir: Option<PathBuf>,
    upsampler: gen_core::flux3::phase_resource::PhaseResourceSlot<crate::pup::Flux3PromptUpsampler>,
    image_execution_fence: gen_core::core_llm::text_llm::OwnedExecutionFence<mlx_rs::Stream>,
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy request scope) OR a `Sequential` load policy.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

/// The DiT compute dtype a load spec selects: the release's bf16 autocast by default, the
/// oracle's FP32 path on an explicit `Precision::Fp32`. The autoencoder is f32 either way.
pub fn compute_dtype(spec: &LoadSpec) -> Dtype {
    match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    }
}

/// Load the DiT from the package's `model.safetensors` and evaluate every parameter before
/// returning, so no forward ever reads a weight lazily.
pub fn load_dit(weights: &Path, config: &Flux3Config, compute: Dtype) -> Result<Flux3Dit> {
    let mut weights = Weights::from_file(weights)
        .map_err(|e| Error::Msg(format!("flux3: loading {}: {e}", weights.display())))?;
    // The release ships bf16. Cast every tensor to the compute dtype ONE TENSOR AT A TIME and
    // drop its source handle, so the load peak is the compute-dtype model plus a single source
    // tensor rather than both full copies (the f32 path doubles the DiT; bf16 is a no-op cast).
    let mut keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    keys.sort();
    for key in keys {
        let Some(array) = weights.remove(&key) else {
            continue;
        };
        // Read the file bytes on MLX's CPU stream first: a GPU cast over an unread `Load` makes
        // the Metal command buffer wait on the disk (sc-24245).
        mlx_rs::transforms::eval_pending_loads([&array])?;
        let array = if array.ndim() >= 2 {
            array.as_dtype(compute)?
        } else {
            array
        };
        eval([&array])?;
        weights.insert(key, array);
    }
    let dit = Flux3Dit::from_weights(&weights, &config.dit, compute)?;
    let arrays = dit.arrays();
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    drop(weights);
    Ok(dit)
}

/// Load the autoencoder (f32) and evaluate every parameter.
pub fn load_autoencoder(path: &Path) -> Result<Flux3AutoEncoder> {
    let weights = Weights::from_file(path)
        .map_err(|e| Error::Msg(format!("flux3: loading {}: {e}", path.display())))?;
    let ae = Flux3AutoEncoder::from_weights(&weights)?;
    let arrays = ae.arrays();
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    Ok(ae)
}

fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_deferred_load_shape(MODEL_ID, spec)?;
    let refusals: [(&str, bool); 7] = [
        ("quantize", spec.quantize.is_some()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: LoadSpec::{field} is not supported by the distilled FLUX 3 route \
                 (quantized variants are sc-26241, LoRA is sc-26239; the text encoder is the \
                 '{}' component)",
                gen_core::flux3::TEXT_ENCODER_COMPONENT
            )));
        }
    }
    Ok(())
}

/// Construct the generator from the package: `spec.weights` = the package root (`config.json` +
/// `model.safetensors`), with the bundled `text_encoder/` and `ae.safetensors` unless the
/// `text_encoder` / `autoencoder` components override them. `Resident` (default) loads every
/// component now; `Sequential` loads the encoder, encodes, drops it, then loads the DiT and the
/// autoencoder, per request. A `Resident` generator runs one request the same staged way when
/// its admitted `GenerationMemory::stage_residency` is set.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, false)
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, true)
}

fn load_inner(spec: &LoadSpec, base: bool) -> Result<Box<dyn Generator>> {
    refuse_unsupported_spec(spec)?;
    let package_dir = match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => dir.clone(),
        gen_core::WeightsSource::File(file) => {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the weights resource must be the package directory, not {}",
                file.display()
            )))
        }
    };
    let config = Flux3Config::from_dir(&package_dir)?;
    config.validate_supported()?;
    if config.dit.flow_map == base {
        return Err(Error::Unsupported(format!("flux3: selected {} route but package {} has flow_map={}; select the matching CFG checkpoint or distilled flow-map route", if base { "base CFG" } else { "distilled flow-map" }, config.model, config.dit.flow_map)));
    }
    crate::adapters::validate(&spec.adapters, &config.dit)?;
    let adapter_reports = spec
        .adapters
        .iter()
        .map(|adapter| {
            Ok(gen_core::AdapterApplyReport {
                adapter_path: adapter.path.clone(),
                applied: gen_core::flux3::adapters::inspect(adapter, &config.dit)?.len(),
                skipped: Vec::new(),
            })
        })
        .collect::<gen_core::Result<Vec<_>>>()?;

    let resources = GenerationResources::from_spec(spec, &config, MODEL_ID)?;
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let text_in = config.dit.text_in_channels;
    let dit_path = resources.dit_weights();
    let ae_path = resources.autoencoder_file.clone();
    let heavy_cfg = config.clone();
    let adapters = spec.adapters.clone();
    // Header/config inventory precedes any model-weight materialization.
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || fenced_image_load(|| Flux3TextEncoder::load(&te_dir, &te_cfg, text_in)),
        move |_use_pid| {
            fenced_image_load(|| {
                let mut dit = load_dit(&dit_path, &heavy_cfg, compute)?;
                crate::adapters::install(&mut dit, &adapters)?;
                let ae = load_autoencoder(&ae_path)?;
                if ae.latent_channels() as usize != heavy_cfg.dit.image_in_channels {
                    return Err(Error::Msg(format!(
                        "{MODEL_ID}: the autoencoder at {} packs {} channels per token, the DiT \
                     reads {}",
                        ae_path.display(),
                        ae.latent_channels(),
                        heavy_cfg.dit.image_in_channels
                    )));
                }
                Ok(Heavy { dit, ae })
            })
        },
    )?;
    // Only an optional local source path is retained here. An enabled prepare
    // phase owns the VLM load and a bounded phase drops it before image encoding.
    let upsampler_dir = gen_core::flux3::pup::UpsamplePackage::component_dir(spec)?;
    Ok(Box::new(Flux3Klein {
        adapter_reports,
        descriptor: if base {
            descriptor_base()
        } else {
            descriptor()
        },
        config,
        residency,
        spec: spec.clone(),
        memory_strategy,
        upsampler_dir,
        upsampler: Default::default(),
        image_execution_fence: new_image_fence(),
    }))
}

// Keep cancellation at the image decode/completion boundaries too: the shared residency
// driver returns the renderer's result directly, and a Metal decode cannot be interrupted mid-op.
fn decode_batch<T, U>(
    tokens: &[T],
    cancel: &gen_core::CancelFlag,
    mut decode: impl FnMut(&T) -> Result<U>,
) -> Result<Vec<U>> {
    let mut images = Vec::with_capacity(tokens.len());
    for token in tokens {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        images.push(decode(token)?);
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    Ok(images)
}

fn checked_completion<T>(cancel: &gen_core::CancelFlag, value: T) -> Result<T> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    Ok(value)
}

impl Flux3Klein {
    fn staged_images(
        &self,
        req: &GenerationRequest,
        prepared: Option<&PreparedReferences>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<gen_core::Image>> {
        self.residency.run_staged_request_scoped_with_prepare(
            true,false,&req.cancel,false,on_progress,
            Some(||self.synchronize_image()),
            |progress|self.effective_params(req,progress),
            |te:&Flux3TextEncoder,params:GenerationParams| ->Result<(TextConditioning,Option<TextConditioning>,GenerationParams)> {
                let text=te.encode(&params.prompt)?;
                let negative=if params.guidance.is_some(){Some(te.encode(&params.negative_prompt)?)}else{None};
                Ok((text,negative,params))
            },
            |encoded:Option<&(TextConditioning,Option<TextConditioning>,GenerationParams)>| {
                if let Some((text,negative,_))=encoded {
                    text.tokens.eval()?;if let Some(negative)=negative {negative.tokens.eval()?;}
                }
                self.synchronize_image()
            },
            |heavy:&Heavy,(text,negative,params):(TextConditioning,Option<TextConditioning>,GenerationParams),progress| {
                let reference_tokens=prepared.map(|p|crate::references::encode(&heavy.ae,p,&req.cancel)).transpose()?;
                let total=params.steps*req.count;
                let mut tokens=Vec::with_capacity(req.count as usize);
                for index in 0..req.count {
                    let done=index*params.steps;
                    let seed=params.seed.wrapping_add(u64::from(index));
                    tokens.push(denoise_one_with_references(&heavy.dit,&text,negative.as_ref(),reference_tokens.as_ref(),&params,seed,&req.cancel,|step|progress(Progress::Step{current:done+step as u32,total}))?);
                }
                Ok((tokens,params))
            },
            |mid:Option<&(Vec<mlx_rs::Array>,GenerationParams)>| {
                if let Some((tokens,_))=mid {eval(tokens.iter().collect::<Vec<_>>())?;}
                self.synchronize_image()
            },
            |ae,(tokens,params),progress| {
                progress(Progress::Decoding);
                if req.cancel.is_cancelled(){return Err(Error::Canceled);}
                let (h,w)=params.latent_shape()?;
                let primary=decode_batch(&tokens,&req.cancel,|token| {
                    let image=decode_tokens(ae,token,h,w)?;
                    match prepared {Some(p)=>Ok(references::crop_output(image,&p.output)?),None=>Ok(image)}
                });
                let fence=self.synchronize_image();
                match primary {
                    Err(primary)=>{if let Err(cleanup)=fence{eprintln!("FLUX3 decode cleanup fence failed; primary error preserved: {cleanup}");}Err(primary)},
                    Ok(images)=>{fence?;mlx_rs::memory::clear_cache();Ok(images)},
                }
            },
        )
    }

    fn effective_params(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationParams> {
        let staged = stage_residency_for(&self.spec, req);
        let outcome=self.upsampler.with_request(
            req.enhance_prompt,staged,&req.cancel,
            || {
                let dir=self.upsampler_dir.as_ref().ok_or_else(||gen_core::Error::Unsupported(
                    "FLUX3 prompt enhancement requires the separate pinned prompt_upsampler component; install it or explicitly disable enhancement".into()
                ))?;
                let loaded=if staged {crate::pup::Flux3PromptUpsampler::load_bounded(dir)}else{crate::pup::Flux3PromptUpsampler::load(dir)};
                loaded.map_err(gen_core::Error::from)
            },
            |upsampler|crate::pup::effective_prompt(upsampler,req,&references::upsampler_input(req)?,on_progress).map_err(gen_core::Error::from),
            crate::pup::Flux3PromptUpsampler::synchronize_before_release,
            ||{mlx_gen::residency::drain_allocator_cache();},
        )?;
        let mut effective = req.clone();
        effective.prompt = outcome.caption;
        Ok(GenerationParams::resolve(
            &effective,
            prompt_seed(&effective.prompt),
            &self.config,
        )?)
    }

    fn synchronize_image(&self) -> Result<()> {
        self.image_execution_fence
            .synchronize(synchronize_image_stream)
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        let result = self.generate_inner(req, on_progress);
        if result.is_err() {
            match self.upsampler.cleanup_pending() {
                Ok(false)=>if let Err(cleanup)=self.upsampler.release_cached(
                    crate::pup::Flux3PromptUpsampler::synchronize_before_release,
                    ||{mlx_gen::residency::drain_allocator_cache();},
                ) {eprintln!("FLUX3 PUP cleanup failed; primary generation error preserved, owner retained: {cleanup}");},
                Ok(true)=>eprintln!("FLUX3 PUP owner remains retained after failed cleanup; no duplicate fence or reclamation claimed"),
                Err(status)=>eprintln!("FLUX3 PUP cleanup state unavailable; primary generation error preserved: {status}"),
            }
        }
        result
    }

    fn generate_inner(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate_impl(req)?;
        // Validate all originals/geometry before PUP, text encoding, or heavy GPU work.
        let originals = references::request_images(req)?;
        let prepared: Option<PreparedReferences> = if originals.is_empty() {
            None
        } else {
            Some(references::prepare_borrowed(
                &originals
                    .iter()
                    .map(|image| image.as_ref())
                    .collect::<Vec<_>>(),
                references::request_output(req),
            )?)
        };
        self.image_execution_fence.before_execution(
            mlx_rs::Stream::task_local_or_default(),
            synchronize_image_stream,
        )?;
        // Preparation belongs to the exclusive request lifecycle. In particular a staged
        // request must evict a reused resident pair before the optional VLM runs.
        let images = if stage_residency_for(&self.spec, req) {
            self.staged_images(req, prepared.as_ref(), on_progress)?
        } else {
            self.residency.run_request_scoped_with_prepare(
            stage_residency_for(&self.spec, req),
            false,
            &req.cancel,
            false,
            on_progress,
            |on_progress| self.effective_params(req,on_progress),
            |te: &Flux3TextEncoder,
             params: GenerationParams|
             -> Result<(TextConditioning, Option<TextConditioning>, GenerationParams)> {
                let text = te.encode(&params.prompt)?;
                let negative = if params.guidance.is_some() {
                    Some(te.encode(&params.negative_prompt)?)
                } else {
                    None
                };
                Ok((text, negative, params))
            },
            |_| Ok(()),
            |heavy: &Heavy,
             (text, negative, params): (
                TextConditioning,
                Option<TextConditioning>,
                GenerationParams,
            ),
             on_progress| {
                let reference_tokens = prepared
                    .as_ref()
                    .map(|p| crate::references::encode(&heavy.ae, p, &req.cancel))
                    .transpose()?;
                let total = params.steps * req.count;
                // Every image's Euler loop first, then ONE `Decoding` and the decodes: the
                // progress contract reports the decode phase once per request.
                let mut tokens = Vec::with_capacity(req.count as usize);
                for c in 0..req.count {
                    let done = c * params.steps;
                    let seed = params.seed.wrapping_add(u64::from(c));
                    tokens.push(denoise_one_with_references(
                        &heavy.dit,
                        &text,
                        negative.as_ref(),
                        reference_tokens.as_ref(),
                        &params,
                        seed,
                        &req.cancel,
                        |step| {
                            on_progress(Progress::Step {
                                current: done + step as u32,
                                total,
                            })
                        },
                    )?);
                }
                on_progress(Progress::Decoding);
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let (h, w) = params.latent_shape()?;
                let images = decode_batch(&tokens, &req.cancel, |t| {
                    let image = decode_tokens(&heavy.ae, t, h, w)?;
                    match prepared.as_ref() {
                        Some(p) => Ok(references::crop_output(image, &p.output)?),
                        None => Ok(image),
                    }
                })?;
                mlx_rs::memory::clear_cache();
                Ok(images)
            },
        )?
        };
        checked_completion(
            &req.cancel,
            GenerationReport {
                output: Some(GenerationOutput::Images(images)),
                artifacts: None,
                warnings: Vec::new(),
                prefix_kv_cache: None,
            },
        )
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        validate_generation_request(self.descriptor.id, req, &self.config)?;
        refuse_unhonored_memory(req)?;
        Ok(())
    }
}

/// Refuse the memory-ladder controls this route has no mechanism for, so none is accepted and
/// ignored: the autoencoder decode is not tiled (sc-26241), there is no bounded-attention path
/// through the DiT and no block-streaming loader. Staged residency is the one control honoured
/// ([`stage_residency_for`]), and it is the only one the route's request scope ever writes.
fn refuse_unhonored_memory(req: &GenerationRequest) -> Result<()> {
    let Some(memory) = req.memory else {
        return Ok(());
    };
    let refusals = [
        ("tile_vae_decode", memory.tile_vae_decode),
        ("chunk_attention", memory.chunk_attention),
        (
            "stream_transformer_blocks",
            memory.stream_transformer_blocks,
        ),
    ];
    for (control, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: GenerationMemory::{control} is not a memory strategy of the MLX \
                 distilled FLUX 3 route (it stages residency only)"
            )));
        }
    }
    Ok(())
}

impl Generator for Flux3Klein {
    fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
        self.adapter_reports.clone()
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// The registered contract, priced from the loaded resources: the same one
    /// [`crate::memory_strategy::MEMORY_REGISTRATION`] builds for this spec.
    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.spec, &self.memory_strategy, context)
    }

    /// The registered executable seam on the loaded generator: the scope's `configure_request`
    /// writes the admitted `GenerationMemory` onto the request, which [`Self::generate`] honours
    /// through `Residency::run_request_scoped`.
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::registered_begin_request(&self.spec, &self.memory_strategy, context)
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let report = self.generate_impl(req, on_progress)?;
        report
            .output
            .ok_or_else(|| gen_core::Error::Msg("flux3: the render produced no output".into()))
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

mlx_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hosted-only here (MLX does not build on Windows). Mutations: remove the per-item guard
    /// (second decode runs), remove final guard (last decode publishes), or remove completion guard.
    #[test]
    fn late_decode_cancellation_never_returns_partial_or_completed_output() {
        let cancel = gen_core::CancelFlag::new();
        let mut calls = 0;
        let output = decode_batch(&[1u8, 2], &cancel, |token| {
            calls += 1;
            cancel.cancel();
            Ok(*token)
        });
        assert!(matches!(output, Err(Error::Canceled)));
        assert_eq!(calls, 1);
        let cancel = gen_core::CancelFlag::new();
        let output = decode_batch(&[1u8], &cancel, |token| {
            cancel.cancel();
            Ok(*token)
        });
        assert!(matches!(output, Err(Error::Canceled)));
        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            checked_completion(&cancel, vec![1u8]),
            Err(Error::Canceled)
        ));
        let cancel = gen_core::CancelFlag::new();
        assert_eq!(
            decode_batch(&[1u8, 2], &cancel, |token| Ok(*token * 2)).unwrap(),
            [2, 4]
        );
    }

    #[test]
    fn decode_primary_error_is_preserved_when_cancellation_arrives_with_it() {
        let cancel = gen_core::CancelFlag::new();
        let output = decode_batch(&[1u8], &cancel, |_| {
            cancel.cancel();
            Err::<u8, _>(Error::Msg("primary decode error".into()))
        });
        assert!(matches!(output, Err(Error::Msg(message)) if message == "primary decode error"));
    }
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_is_a_flow_map_and_honest() {
        let d = descriptor();
        assert_eq!(d.id, "flux3_klein");
        assert_eq!(d.family, "flux3");
        assert_eq!(d.backend, "mlx");
        assert!(d.denoiser_output_latent_space.is_none());
        assert_eq!(d.capabilities.samplers, ["euler"]);
        assert!(d.capabilities.schedulers.is_empty());
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_lora && !d.capabilities.supports_lokr);
        assert!(!d.capabilities.supports_preview);
        assert!(d.required_components.is_empty());
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 16 }
        );
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (64, 4096)
        );
    }

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Q4);
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
        let spec = LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()))
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("load_shape"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }

    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use mlx_gen::gen_core::{GenerationMemory, OffloadPolicy};
        let resident = LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()));
        let sequential = resident
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential);
        let plain = GenerationRequest::default();
        let staged = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!stage_residency_for(&resident, &plain));
        assert!(stage_residency_for(&resident, &staged));
        assert!(stage_residency_for(&sequential, &plain));
    }

    /// *Mutation that reds this:* removing any row of `refuse_unhonored_memory`'s table.
    #[test]
    fn unhonoured_memory_controls_are_typed_refusals() {
        use mlx_gen::gen_core::GenerationMemory;
        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        refuse_unhonored_memory(&GenerationRequest::default()).unwrap();
        refuse_unhonored_memory(&with(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        }))
        .unwrap();
        for (control, memory) in [
            (
                "tile_vae_decode",
                GenerationMemory {
                    tile_vae_decode: true,
                    ..Default::default()
                },
            ),
            (
                "chunk_attention",
                GenerationMemory {
                    chunk_attention: true,
                    attention_chunk_size: Some(1 << 26),
                    ..Default::default()
                },
            ),
            (
                "stream_transformer_blocks",
                GenerationMemory {
                    stream_transformer_blocks: true,
                    ..Default::default()
                },
            ),
        ] {
            match refuse_unhonored_memory(&with(memory)) {
                Err(Error::Unsupported(message)) => {
                    assert!(message.contains(control), "{message}")
                }
                other => panic!("{control}: expected a typed refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_missing_package_is_a_load_error_naming_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let err = load(&spec).err().expect("an empty package must fail");
        assert!(err.to_string().contains("config.json"), "{err}");
    }
}
