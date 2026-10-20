//! `Flux3Klein` — the distilled FLUX 3 Image Klein text-to-image [`Generator`] on Candle:
//! descriptor, load, request validation and the staged generate (text encoder → DiT +
//! autoencoder), registered as [`MODEL_ID`]. The request/load surface is the MLX twin's, field for
//! field (both read `gen_core::flux3`).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::flux3::artifacts::{ArtifactBackend, NativeDitStorage};
use candle_gen::gen_core::flux3::conversion::ComponentRole;
use candle_gen::gen_core::flux3::quantization::DitFormat;
use candle_gen::gen_core::flux3::{
    validate_generation_request, Flux3Config, GenerationParams, GenerationResources,
    DISTILLED_MODEL_ID, FAMILY, SAMPLERS, TEXT_ENCODER_COMPONENT,
};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationReport, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, SizeFloor,
};
use candle_gen::residency::Residency;
use candle_gen::{CandleError as Error, Result};

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::Flux3Dit;
use crate::pipeline::{decode_tokens, denoise_one_with_references};
use crate::text_encoder::{Flux3TextEncoder, TextConditioning};
use gen_core::flux3::references::{self, PreparedReferences};

/// Registry id (the SceneWorks worker's `payload.model`) — the same id as the MLX twin, bound to
/// the `candle` backend by the descriptor.
pub const MODEL_ID: &str = DISTILLED_MODEL_ID;
/// Output sides must be multiples of the token size.
pub const SIZE_MULTIPLE: u32 = gen_core::flux3::DOWNSAMPLING_RATIO;
/// Smallest side (`pipeline.MIN_SIDE`).
pub const MIN_SIZE: u32 = gen_core::flux3::MIN_SIDE;
/// Largest side (`pipeline.MAX_SIDE`).
pub const MAX_SIZE: u32 = gen_core::flux3::MAX_SIDE;
/// Images per request (rendered one after another, each from `seed + i`).
pub const MAX_COUNT: u32 = 8;

fn fenced_image_load<T>(device: &Device, load: impl FnOnce() -> Result<T>) -> Result<T> {
    let primary = load();
    if primary.is_err() {
        if let Err(cleanup) = device.synchronize() {
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
        backend: "candle",
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
            mac_only: false,
            // Native Candle tiers (sc-26241): q4 = GGML Q4_1 and q8 = GGML Q8_0 packages converted
            // from the ORIGINAL dense checkpoint (`crate::convert`). Distinct from the MLX twin's
            // affine q4/q8: no cross-backend artifact reuse. Publisher FP8R/NVFP4 have no Candle
            // FLUX3 kernel on this build and are therefore not advertised.
            #[cfg(not(feature = "cuda"))]
            supported_quants: &[gen_core::Quant::Q4, gen_core::Quant::Q8],
            // The CUDA build also loads the publisher FP8R / NVFP4 artifacts as shipped onto
            // their frozen-recipe kernels (device floors are checked at load).
            #[cfg(feature = "cuda")]
            supported_quants: &[
                gen_core::Quant::Q4,
                gen_core::Quant::Q8,
                gen_core::Quant::Fp8Rowwise,
                gen_core::Quant::Nvfp4,
            ],
            supports_sequential_offload: true,
            // The optional native VLM resolves the effective caption before encoding.
            supports_prompt_enhancement: true,
            conditioning: vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::ReferenceRgba,
                gen_core::ConditioningKind::MultiReference,
            ],
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
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
    device: Device,
    residency: Residency<Flux3TextEncoder, Heavy>,
    prompt_upsampler_dir: Option<PathBuf>,
    prompt_upsampler:
        gen_core::flux3::phase_resource::PhaseResourceSlot<crate::pup::Flux3PromptUpsampler>,
    /// The spec this generator was loaded from: the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against.
    spec: LoadSpec,
    /// The loaded memory contract, published through [`Generator::memory_strategy_contract`].
    memory_strategy: gen_core::MemoryProviderContract,
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy request scope) OR a `Sequential` load policy.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

/// The DiT compute dtype a load spec selects: the release's bf16 autocast by default, the
/// oracle's FP32 path on an explicit `Precision::Fp32`. The autoencoder is f32 either way.
pub fn compute_dtype(spec: &LoadSpec) -> DType {
    match spec.precision {
        Precision::Bf16 => DType::BF16,
        Precision::Fp32 => DType::F32,
    }
}

/// Load the DiT from the package's `model.safetensors` onto `device` at `compute`.
pub fn load_dit(
    weights: &Path,
    config: &Flux3Config,
    compute: DType,
    device: &Device,
) -> Result<Flux3Dit> {
    Flux3Dit::load(weights, &config.dit, compute, device)
}

/// Load the autoencoder (f32) onto `device`.
pub fn load_autoencoder(path: &Path, device: &Device) -> Result<Flux3AutoEncoder> {
    Flux3AutoEncoder::load(path, device)
}

#[cfg(feature = "cuda")]
fn cuda_backend(device: &Device) -> Result<ArtifactBackend> {
    let Device::Cuda(cuda) = device else {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: expected a CUDA device"
        )));
    };
    let (major, minor) = candle_quant_kernels::nvrtc::device_compute_cap(cuda).map_err(|e| {
        Error::Unsupported(format!(
            "{MODEL_ID}: cannot read the CUDA compute capability: {e}"
        ))
    })?;
    Ok(ArtifactBackend::CandleCuda {
        major: major as u32,
        minor: minor as u32,
    })
}

/// A non-CUDA build never opens a CUDA device.
#[cfg(not(feature = "cuda"))]
fn cuda_backend(_device: &Device) -> Result<ArtifactBackend> {
    Err(Error::Unsupported(format!(
        "{MODEL_ID}: this build has no CUDA backend"
    )))
}

/// A native package loads only with its conversion receipt: the owned outputs full-hash
/// verified, the receipt's algorithm identical to the declared storage, and every EXTERNAL
/// component full-hash verified where this load actually reads it (the resolved text-encoder
/// directory and autoencoder file). A mismatch is refused before any weight is read.
fn verify_native_package(
    package_dir: &Path,
    storage: NativeDitStorage,
    config: &Flux3Config,
    resources: &GenerationResources,
) -> Result<()> {
    let receipt = gen_core::flux3::conversion::open_conversion(package_dir)?;
    let (algorithm, bits, group_size) = storage.receipt_algorithm()?;
    if receipt.algorithm != algorithm || receipt.bits != bits || receipt.group_size != group_size {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: {} receipt records {} (bits {:?}, group {:?}), not the declared {}",
            package_dir.display(),
            receipt.algorithm,
            receipt.bits,
            receipt.group_size,
            storage.key()?
        )));
    }
    let te_prefix = format!("{}/", config.text_encoder.path);
    receipt.verify_external_with(|role, path| match role {
        ComponentRole::TextEncoder => path
            .strip_prefix(&te_prefix)
            .map(|rest| resources.text_encoder_dir.join(rest))
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!("{MODEL_ID}: text-encoder path {path}"))
            }),
        ComponentRole::Autoencoder if path == config.vae.path => {
            Ok(resources.autoencoder_file.clone())
        }
        other => Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: external {} file {path} has no resolved load location",
            other.name()
        ))),
    })?;
    Ok(())
}

pub(crate) fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_deferred_load_shape(MODEL_ID, spec)?;
    crate::memory_strategy::selected_storage(spec.quantize)?;
    let refusals: [(&str, bool); 6] = [
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
                 (the text encoder is the '{TEXT_ENCODER_COMPONENT}' component)"
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
/// its admitted `GenerationMemory::stage_residency` is set. Everything runs on the build's device
/// ([`candle_gen::default_device`]: `cuda:0` on a CUDA build).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_inner(spec, false)?))
}

pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_inner(spec, true)?))
}

fn load_inner(spec: &LoadSpec, base: bool) -> Result<Flux3Klein> {
    refuse_unsupported_spec(spec)?;
    let package_dir = match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => dir.clone(),
        gen_core::WeightsSource::File(file) => {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: the weights resource must be the package directory, not {}",
                file.display()
            )))
        }
    };
    let config = Flux3Config::from_dir(&package_dir)?;
    let storage = crate::memory_strategy::selected_storage(spec.quantize)?;
    let package_storage = NativeDitStorage::from_package(&package_dir)?;
    if package_storage != storage {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: selected {} but the quantization marker of {} is {}; install or \
             convert the exact requested native tier (no tier or model fallback)",
            storage.key()?,
            package_dir.display(),
            package_storage.key()?
        )));
    }
    storage.validate_selection(spec.quantize, spec.precision)?;
    let dense = storage == NativeDitStorage::Publisher(DitFormat::Dense);
    if dense {
        config.validate_supported()?;
    } else {
        config.validate_native_storage_graph()?;
    }
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
    // Converted packages carry a receipt; a publisher FP8R/NVFP4 package is the pinned upstream
    // repository itself, whose identity is the install manifest's, not a conversion's.
    if !dense && !matches!(storage, NativeDitStorage::Publisher(_)) {
        verify_native_package(&package_dir, storage, &config, &resources)?;
    }
    // Priced from the resources' tensor headers BEFORE any weight is read, so a spec the
    // registration refuses is the same refusal through `load` on every offload policy
    // (`Sequential` defers every component to the first render).
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    // Resolve only the optional source path. No VLM/model/prefix cache is loaded
    // until an enabled request enters the exclusive prepare phase.
    let prompt_upsampler_dir = gen_core::flux3::pup::UpsamplePackage::component_dir(spec)?;
    let device = candle_gen::default_device().map_err(|e| {
        Error::Unsupported(format!(
        "{MODEL_ID}: cannot initialize the selected native Candle device: {e}; use a supported \
         NVIDIA CUDA device and driver, or the native MLX backend on Apple Silicon"
    ))
    })?;
    if !dense {
        // The device's REAL compute capability: the publisher FP8R/NVFP4 floors depend on it
        // (GGML ignores it). Metal is a typed refusal, never a dequantized substitute.
        let backend = if device.is_cpu() {
            ArtifactBackend::CandleCpu
        } else if device.is_metal() {
            ArtifactBackend::CandleMetal
        } else {
            cuda_backend(&device)?
        };
        storage.check_backend(backend)?;
    }
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let te_device = device.clone();
    let text_in = config.dit.text_in_channels;
    let dit_path = resources.dit_weights();
    let ae_path = resources.autoencoder_file.clone();
    let heavy_cfg = config.clone();
    let adapters = spec.adapters.clone();
    let heavy_device = device.clone();
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || {
            fenced_image_load(&te_device, || {
                Flux3TextEncoder::load(&te_dir, &te_cfg, text_in, &te_device)
            })
        },
        move |_use_pid| {
            fenced_image_load(&heavy_device, || {
                let mut dit = Flux3Dit::load_with_storage(
                    &dit_path,
                    &heavy_cfg.dit,
                    compute,
                    &heavy_device,
                    storage,
                )?;
                crate::adapters::install(&mut dit, &adapters)?;
                let ae = load_autoencoder(&ae_path, &heavy_device)?;
                if ae.latent_channels() != heavy_cfg.dit.image_in_channels {
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
    Ok(Flux3Klein {
        adapter_reports,
        descriptor: if base {
            descriptor_base()
        } else {
            descriptor()
        },
        config,
        device,
        residency,
        prompt_upsampler_dir,
        prompt_upsampler: Default::default(),
        spec: spec.clone(),
        memory_strategy,
    })
}

/// The residency driver also calls this boundary with `None` after an encoding error. Do not
/// condition synchronization on a successful encoding: queued work must finish before text drops.
fn encoder_materialization_boundary<T>(
    synchronize: impl FnOnce() -> Result<()>,
) -> impl FnOnce(Option<&T>) -> Result<()> {
    move |_| synchronize()
}

/// Every decode is a cancellation boundary, including the last one before results can escape.
fn decode_batch<T, Out>(
    tokens: &[T],
    cancel: &gen_core::CancelFlag,
    mut decode: impl FnMut(&T) -> Result<Out>,
) -> Result<Vec<Out>> {
    let mut images = Vec::with_capacity(tokens.len());
    for token in tokens {
        candle_gen::residency::check_cancel(cancel)?;
        images.push(decode(token)?);
    }
    candle_gen::residency::check_cancel(cancel)?;
    Ok(images)
}

fn checked_completion<T>(cancel: &gen_core::CancelFlag, value: T) -> Result<T> {
    candle_gen::residency::check_cancel(cancel)?;
    Ok(value)
}

/// Fixed-flags Candle image adapter to the shared prepared-request driver. This forwards phase
/// ordering/locking/eviction to gen-core; the seam lets CPU witnesses exercise the production call.
#[allow(clippy::too_many_arguments)]
fn prepared_image_request<Text, HeavyPhase, Prepared, Encoded, Out>(
    residency: &Residency<Text, HeavyPhase>,
    stage: bool,
    cancel: &gen_core::CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    prepare: impl FnOnce(&mut dyn FnMut(Progress)) -> Result<Prepared>,
    encode: impl FnOnce(&Text, Prepared) -> Result<Encoded>,
    materialize: impl FnOnce(Option<&Encoded>) -> Result<()>,
    render: impl FnOnce(&HeavyPhase, Encoded, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    residency.run_request_scoped_with_prepare(
        stage,
        false,
        cancel,
        false,
        on_progress,
        prepare,
        encode,
        materialize,
        render,
    )
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
            Some(||Ok(self.device.synchronize()?)),
            |progress|self.effective_params(req,progress),
            |te:&Flux3TextEncoder,params:GenerationParams| ->Result<(TextConditioning,Option<TextConditioning>,GenerationParams)> {
                let text=te.encode(&params.prompt)?;
                let negative=if params.guidance.is_some(){Some(te.encode(&params.negative_prompt)?)}else{None};
                Ok((text,negative,params))
            },
            encoder_materialization_boundary(||Ok(self.device.synchronize()?)),
            |heavy:&Heavy,(text,negative,params):(TextConditioning,Option<TextConditioning>,GenerationParams),progress| {
                let reference_tokens=prepared.map(|p|crate::references::encode(&heavy.ae,p,&self.device,&req.cancel)).transpose()?;
                gen_core::flux3::phase_resource::conditioning_complete(&req.cancel, || {
                    self.device.synchronize().map_err(Error::from).map_err(gen_core::Error::from)
                }, progress)?;
                let total=params.steps*req.count;
                let mut tokens=Vec::with_capacity(req.count as usize);
                for index in 0..req.count {
                    let done=index*params.steps;
                    let seed=candle_gen::image_seed(params.seed,index);
                    tokens.push(denoise_one_with_references(&heavy.dit,&text,negative.as_ref(),reference_tokens.as_ref(),&params,seed,&req.cancel,|step|progress(Progress::Step{current:done+step as u32,total}))?);
                }
                Ok((tokens,params))
            },
            |_mid|Ok(self.device.synchronize()?),
            |ae,(tokens,params),progress| {
                progress(Progress::Decoding);
                if req.cancel.is_cancelled(){return Err(Error::Canceled);}
                let (h,w)=params.latent_shape()?;
                let primary=decode_batch(&tokens,&req.cancel,|token| {
                    let image=decode_tokens(ae,token,h,w)?;
                    match prepared {Some(p)=>Ok(references::crop_output(image,&p.output)?),None=>Ok(image)}
                });
                let fence=self.device.synchronize().map_err(Error::from);
                match primary {
                    Err(primary)=>{if let Err(cleanup)=fence{eprintln!("FLUX3 decode cleanup fence failed; primary error preserved: {cleanup}");}Err(primary)},
                    Ok(images)=>{fence?;Ok(images)},
                }
            },
        )
    }

    fn effective_params(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationParams> {
        let outcome=self.prompt_upsampler.with_prompt_upsampling(
            req.enhance_prompt,stage_residency_for(&self.spec,req),&req.cancel,
            || {
                let dir=self.prompt_upsampler_dir.as_ref().ok_or_else(||gen_core::Error::Unsupported(
                    "FLUX3 prompt enhancement requires the separate pinned prompt_upsampler component; install it or explicitly disable enhancement".into()
                ))?;
                crate::pup::Flux3PromptUpsampler::load(dir).map_err(gen_core::Error::from)
            },
            |upsampler, progress|crate::pup::effective_prompt(upsampler,req,progress).map_err(gen_core::Error::from),
            crate::pup::Flux3PromptUpsampler::synchronize_before_release,
            || {}, // Candle storage owns deallocation; unlike MLX there is no allocator-drain hook.
            on_progress,
        )?;
        Ok(GenerationParams::resolve_caption(
            req,
            outcome.caption,
            &self.config,
        )?)
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        let result = self.generate_inner(req, on_progress);
        if result.is_err() {
            match self.prompt_upsampler.cleanup_pending() {
                Ok(false)=>if let Err(cleanup)=self.prompt_upsampler.release_cached(
                    crate::pup::Flux3PromptUpsampler::synchronize_before_release,||{},
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
        let images = if stage_residency_for(&self.spec, req) {
            self.staged_images(req, prepared.as_ref(), on_progress)?
        } else {
            prepared_image_request(
            &self.residency,
            stage_residency_for(&self.spec, req),
            &req.cancel,
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
            encoder_materialization_boundary(|| Ok(self.device.synchronize()?)),
            |heavy: &Heavy,
             (text, negative, params): (
                TextConditioning,
                Option<TextConditioning>,
                GenerationParams,
            ),
             on_progress| {
                let reference_tokens = prepared
                    .as_ref()
                    .map(|p| crate::references::encode(&heavy.ae, p, &self.device, &req.cancel))
                    .transpose()?;
                gen_core::flux3::phase_resource::conditioning_complete(&req.cancel, || {
                    self.device.synchronize().map_err(Error::from).map_err(gen_core::Error::from)
                }, on_progress)?;
                let total = params.steps * req.count;
                // Every image's Euler loop first, then ONE `Decoding` and the decodes: the
                // progress contract reports the decode phase once per request.
                let mut tokens = Vec::with_capacity(req.count as usize);
                for c in 0..req.count {
                    let done = c * params.steps;
                    let seed = candle_gen::image_seed(params.seed, c);
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
                decode_batch(&tokens, &req.cancel, |t| {
                    let image = decode_tokens(&heavy.ae, t, h, w)?;
                    match prepared.as_ref() {
                        Some(p) => Ok(references::crop_output(image, &p.output)?),
                        None => Ok(image),
                    }
                })
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
        references::validate_request_with_resolved_output(
            &self.descriptor.capabilities,
            self.descriptor.id,
            req,
        )?;
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
                "{MODEL_ID}: GenerationMemory::{control} is not a memory strategy of the Candle \
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

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

candle_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

    fn render(model: &Flux3Klein) -> Vec<u8> {
        let req = GenerationRequest {
            prompt: "a red fox in the snow".into(),
            width: 64,
            height: 64,
            steps: Some(2),
            seed: Some(7),
            ..Default::default()
        };
        match model.generate_impl(&req, &mut |_| {}).unwrap().output {
            Some(GenerationOutput::Images(images)) => images[0].pixels.clone(),
            other => panic!("expected an image, got {other:?}"),
        }
    }

    fn mean_abs(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    /// A converted GGML package loads ONLY as its exact tier, with its external encoder/AE
    /// full-hash verified where they are read, and renders through the packed GEMM close to the
    /// dense package (same steps, size and seed — no quality setting changed).
    ///
    /// *Mutations that red this:* accepting a dense request on a packed package (or another
    /// tier), skipping `verify_native_package`, or loading packed targets as dense.
    #[test]
    fn converted_ggml_package_loads_only_as_its_exact_tier_and_renders_close_to_dense() {
        use gen_core::flux3::AUTOENCODER_COMPONENT;
        let dir = tempfile::tempdir().unwrap();
        let source = crate::convert::tests::source_copy(dir.path());
        let dense = load_inner(
            &LoadSpec::new(WeightsSource::Dir(source.root.clone())),
            false,
        )
        .unwrap();
        let dense_pixels = render(&dense);
        let packed_spec = |root: &Path, quant: Option<gen_core::Quant>, ae: &Path| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
            spec.quantize = quant;
            spec.components.insert(
                TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(source.root.join("text_encoder")),
            );
            spec.components.insert(
                AUTOENCODER_COMPONENT.into(),
                WeightsSource::File(ae.to_path_buf()),
            );
            spec
        };
        let source_ae = source.root.join("ae.safetensors");
        let mut diffs = Vec::new();
        for (storage, quant) in [
            (NativeDitStorage::GgmlQ8_0, gen_core::Quant::Q8),
            (NativeDitStorage::GgmlQ4_1, gen_core::Quant::Q4),
        ] {
            let out = dir.path().join(storage.key().unwrap());
            crate::convert::convert_ggml(&source, storage, &out, &gen_core::CancelFlag::new())
                .unwrap();
            let model = load_inner(&packed_spec(&out, Some(quant), &source_ae), false).unwrap();
            let pixels = render(&model);
            assert_ne!(
                pixels, dense_pixels,
                "{storage:?}: the packed GEMM must actually run"
            );
            diffs.push(mean_abs(&pixels, &dense_pixels));
            assert!(
                load_inner(&packed_spec(&out, None, &source_ae), false).is_err(),
                "a dense request never reads packed codes"
            );
            let other = if quant == gen_core::Quant::Q8 {
                gen_core::Quant::Q4
            } else {
                gen_core::Quant::Q8
            };
            assert!(
                load_inner(&packed_spec(&out, Some(other), &source_ae), false).is_err(),
                "no cross-tier load"
            );
            let tampered = dir
                .path()
                .join(format!("{}-ae.safetensors", storage.key().unwrap()));
            let mut bytes = std::fs::read(&source_ae).unwrap();
            let last = bytes.len() - 1;
            bytes[last] ^= 1;
            std::fs::write(&tampered, bytes).unwrap();
            assert!(
                load_inner(&packed_spec(&out, Some(quant), &tampered), false).is_err(),
                "an external AE that is not the converted source is refused"
            );
        }
        eprintln!(
            "mean |packed - dense| per u8 channel: q8 {:.3}, q4 {:.3}",
            diffs[0], diffs[1]
        );
        assert!(diffs[0] < 12.0 && diffs[1] < 32.0, "{diffs:?}");
        assert!(
            load_inner(
                &LoadSpec::new(WeightsSource::Dir(source.root.clone()))
                    .with_quant(gen_core::Quant::Q8),
                false
            )
            .is_err(),
            "a q8 request never quantizes a dense package at load"
        );
    }

    #[test]
    fn actual_generator_lazy_pup_bypass_and_exclusive_ownership_are_honest() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures");
        for policy in [
            gen_core::OffloadPolicy::Resident,
            gen_core::OffloadPolicy::Sequential,
        ] {
            let mut spec = LoadSpec::new(WeightsSource::Dir(fixtures.join("tiny-snapshot")))
                .with_offload_policy(policy);
            spec.precision = Precision::Fp32;
            spec.components.insert(
                gen_core::flux3::pup::UPSAMPLER_COMPONENT.into(),
                WeightsSource::Dir(fixtures.join("tiny-pup-snapshot")),
            );
            let model = load_inner(&spec, false).unwrap();
            assert!(
                !model.prompt_upsampler.is_loaded().unwrap(),
                "loading the image generator must not load an unused VLM"
            );
            let mut req = GenerationRequest {
                prompt: "a red bicycle leaning against a brick wall".into(),
                width: 80,
                height: 64,
                steps: Some(1),
                seed: Some(17),
                enhance_max_tokens: None,
                ..Default::default()
            };
            model.generate_impl(&req, &mut |_| {}).unwrap();
            assert!(
                !model.prompt_upsampler.is_loaded().unwrap(),
                "bypass must not secretly load PUP"
            );
            req.enhance_prompt = true;
            req.enhance_max_tokens = Some(12);
            model.generate_impl(&req, &mut |_| {}).unwrap();
            assert_eq!(
                model.prompt_upsampler.is_loaded().unwrap(),
                policy == gen_core::OffloadPolicy::Resident
            );
            if policy == gen_core::OffloadPolicy::Sequential {
                assert!(model
                    .residency
                    .with_resident_parts(|_, _| ())
                    .unwrap()
                    .is_none());
            }
            req.enhance_prompt = false;
            req.enhance_max_tokens = None;
            model.generate_impl(&req, &mut |_| {}).unwrap();
            assert!(
                !model.prompt_upsampler.is_loaded().unwrap(),
                "bypass must evict a prior resident PUP owner"
            );
        }
    }

    #[test]
    fn candle_prepare_driver_evicts_before_pup_once_and_observes_prepare_cancellation() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        };
        struct Component {
            kind: &'static str,
            id: usize,
            events: Arc<Mutex<Vec<String>>>,
        }
        impl Drop for Component {
            fn drop(&mut self) {
                candle_gen::lock_recover(&self.events)
                    .push(format!("drop-{}-{}", self.kind, self.id));
            }
        }
        for stage in [true, false] {
            for cancel_prepare in [false, true] {
                let events = Arc::new(Mutex::new(Vec::new()));
                let text_count = Arc::new(AtomicUsize::new(0));
                let heavy_count = Arc::new(AtomicUsize::new(0));
                let te = events.clone();
                let tc = text_count.clone();
                let he = events.clone();
                let hc = heavy_count.clone();
                let advisory = events.clone();
                let residency = Residency::from_policy(
                    gen_core::OffloadPolicy::Resident,
                    move || {
                        let id = tc.fetch_add(1, Ordering::SeqCst);
                        candle_gen::lock_recover(&te).push(format!("load-text-{id}"));
                        Ok(Component {
                            kind: "text",
                            id,
                            events: te.clone(),
                        })
                    },
                    move |_| {
                        let id = hc.fetch_add(1, Ordering::SeqCst);
                        candle_gen::lock_recover(&he).push(format!("load-heavy-{id}"));
                        Ok(Component {
                            kind: "heavy",
                            id,
                            events: he.clone(),
                        })
                    },
                )
                .unwrap()
                .with_staged_advisory(move || {
                    candle_gen::lock_recover(&advisory).push("stage".into())
                });
                candle_gen::lock_recover(&events).clear(); // The old resident pair is already warm.
                let cancel = gen_core::CancelFlag::new();
                let result = prepared_image_request(
                    &residency,
                    stage,
                    &cancel,
                    &mut |_| {},
                    |_| {
                        candle_gen::lock_recover(&events).push("pup".into());
                        if cancel_prepare {
                            cancel.cancel();
                        }
                        Ok(17)
                    },
                    |_: &Component, prepared| {
                        candle_gen::lock_recover(&events).push("encode".into());
                        Ok(prepared)
                    },
                    encoder_materialization_boundary(|| {
                        candle_gen::lock_recover(&events).push("sync".into());
                        Ok(())
                    }),
                    |_: &Component, encoded, _| {
                        candle_gen::lock_recover(&events).push("render".into());
                        Ok(encoded)
                    },
                );
                let seen = candle_gen::lock_recover(&events).clone();
                let pup = seen.iter().position(|s| s == "pup").unwrap();
                assert_eq!(
                    seen.iter().filter(|s| s.as_str() == "stage").count(),
                    usize::from(stage)
                );
                for component in ["text", "heavy"] {
                    let old_drop = format!("drop-{component}-0");
                    assert_eq!(
                        seen.iter().filter(|s| **s == old_drop).count(),
                        usize::from(stage)
                    );
                    if stage {
                        assert!(
                            seen.iter().position(|s| *s == old_drop).unwrap() < pup,
                            "{seen:?}"
                        );
                    }
                }
                if cancel_prepare {
                    assert!(matches!(result, Err(Error::Canceled)));
                    assert!(
                        !seen
                            .iter()
                            .any(|s| s == "encode" || s == "render" || s.starts_with("load-")),
                        "{seen:?}"
                    );
                } else {
                    assert_eq!(result.unwrap(), 17);
                    assert!(pup < seen.iter().position(|s| s == "encode").unwrap());
                    if stage {
                        assert_eq!(text_count.load(Ordering::SeqCst), 2);
                        assert_eq!(heavy_count.load(Ordering::SeqCst), 2);
                        assert_eq!(
                            seen.iter().filter(|s| s.as_str() == "drop-text-1").count(),
                            1
                        );
                        assert_eq!(
                            seen.iter().filter(|s| s.as_str() == "drop-heavy-1").count(),
                            1
                        );
                    } else {
                        assert_eq!(text_count.load(Ordering::SeqCst), 1);
                        assert_eq!(heavy_count.load(Ordering::SeqCst), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn cancellation_during_decode_stops_the_batch_and_withholds_the_last_result() {
        for count in [1, 8] {
            let cancel = gen_core::CancelFlag::new();
            let mut entered = 0;
            let tokens = vec![17; count];
            let result = decode_batch(&tokens, &cancel, |token| {
                entered += 1; // The decoder has begun, not merely the denoise/progress phase.
                cancel.cancel();
                Ok(*token)
            });
            assert!(matches!(result, Err(Error::Canceled)));
            assert_eq!(
                entered, 1,
                "do not decode the remaining images after cancellation"
            );
        }
    }

    #[test]
    fn cancellation_after_decode_completion_still_prevents_publication() {
        let cancel = gen_core::CancelFlag::new();
        let images = decode_batch(&[17, 23], &cancel, |token| Ok(*token)).unwrap();
        assert_eq!(images, [17, 23]);
        cancel.cancel();
        assert!(matches!(
            checked_completion(&cancel, images),
            Err(Error::Canceled)
        ));
    }

    #[test]
    fn encoder_boundary_syncs_before_drop_on_success_and_error_preserving_primary_error() {
        use std::sync::{Arc, Mutex};
        struct Text(Arc<Mutex<Vec<&'static str>>>);
        impl Drop for Text {
            fn drop(&mut self) {
                candle_gen::lock_recover(&self.0).push("text-drop");
            }
        }
        for (encode_fails, sync_fails) in
            [(false, false), (true, false), (true, true), (false, true)]
        {
            let events = Arc::new(Mutex::new(Vec::new()));
            let load_events = events.clone();
            let heavy_events = events.clone();
            let residency = Residency::from_policy(
                gen_core::OffloadPolicy::Sequential,
                move || {
                    candle_gen::lock_recover(&load_events).push("text-load");
                    Ok(Text(load_events.clone()))
                },
                move |_| {
                    candle_gen::lock_recover(&heavy_events).push("heavy-load");
                    Ok(())
                },
            )
            .unwrap();
            let result = residency.run_request_scoped(
                true,
                false,
                &gen_core::CancelFlag::new(),
                false,
                &mut |_| {},
                |_: &Text| {
                    candle_gen::lock_recover(&events).push("encode");
                    if encode_fails {
                        Err(Error::Msg("primary-encoding-error".into()))
                    } else {
                        Ok(17)
                    }
                },
                encoder_materialization_boundary(|| {
                    candle_gen::lock_recover(&events).push("synchronize");
                    if sync_fails {
                        Err(Error::Msg("synchronization-error".into()))
                    } else {
                        Ok(())
                    }
                }),
                |_: &(), encoded, _| {
                    candle_gen::lock_recover(&events).push("render");
                    Ok(encoded)
                },
            );
            let seen = candle_gen::lock_recover(&events);
            assert_eq!(
                &seen[..4],
                &["text-load", "encode", "synchronize", "text-drop"]
            );
            if encode_fails {
                assert!(
                    matches!(result, Err(Error::Msg(ref message)) if message == "primary-encoding-error")
                );
                assert_eq!(seen.len(), 4);
            } else if sync_fails {
                assert!(
                    matches!(result, Err(Error::Msg(ref message)) if message == "synchronization-error")
                );
                assert_eq!(seen.len(), 4);
            } else {
                assert_eq!(result.unwrap(), 17);
                assert_eq!(&seen[4..], &["heavy-load", "render"]);
            }
        }
    }

    #[test]
    fn descriptor_is_a_flow_map_and_honest() {
        let d = descriptor();
        assert_eq!(d.id, "flux3_klein");
        assert_eq!(d.family, "flux3");
        assert_eq!(d.backend, "candle");
        assert!(!d.capabilities.mac_only);
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

    /// Every unsupported load axis is a typed `Unsupported`, never a stringified message and
    /// never a torch fallback.
    ///
    /// *Mutation that reds this:* dropping any row of `refuse_unsupported_spec`'s table.
    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let sentinel = || LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()));
        let mut quant = sentinel();
        quant.quantize = Some(gen_core::Quant::Nvfp4);
        let mut pid = sentinel();
        pid.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File("/pid-not-on-disk.safetensors".into()),
            gemma: WeightsSource::Dir("/gemma-not-on-disk".into()),
        });
        for (field, spec) in [
            ("quantize", quant),
            ("pid", pid),
            (
                "load_shape",
                sentinel().with_load_shape(gen_core::LoadShape::DeferredMaterialization),
            ),
        ] {
            match load(&spec) {
                Err(gen_core::Error::Unsupported(m)) => assert!(m.contains(field), "{m}"),
                other => panic!("{field}: expected a typed refusal, got {:?}", other.err()),
            }
        }
    }

    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use gen_core::{GenerationMemory, OffloadPolicy};
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
        use gen_core::GenerationMemory;
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
        let err = match load(&spec) {
            Err(err) => err,
            Ok(_) => panic!("an empty package must fail"),
        };
        assert!(err.to_string().contains("config.json"), "{err}");
    }
}
