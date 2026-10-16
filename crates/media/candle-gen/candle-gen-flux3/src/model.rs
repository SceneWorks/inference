//! `Flux3Klein` — the distilled FLUX 3 Image Klein text-to-image [`Generator`] on Candle:
//! descriptor, load, request validation and the staged generate (text encoder → DiT +
//! autoencoder), registered as [`MODEL_ID`]. The request/load surface is the MLX twin's, field for
//! field (both read `gen_core::flux3`).

use std::path::Path;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::flux3::{
    prompt_seed, validate_generation_request, Flux3Config, GenerationParams, GenerationResources,
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
use crate::pipeline::{decode_tokens, denoise_one};
use crate::text_encoder::{Flux3TextEncoder, TextConditioning};

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
            // LoRA loading is sc-26239 (with the base checkpoint).
            supports_lora: false,
            supports_lokr: false,
            // No per-step preview: the packed latent has no fitted RGB projection and the
            // autoencoder decode is the only exact view of a step.
            supports_preview: false,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: MAX_COUNT,
            mac_only: false,
            // Quantized variants are sc-26241.
            supported_quants: &[],
            supports_sequential_offload: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// The heavy half: the DiT and the autoencoder.
pub struct Heavy {
    pub dit: Flux3Dit,
    pub ae: Flux3AutoEncoder,
}

/// A loaded distilled generator.
pub struct Flux3Klein {
    descriptor: ModelDescriptor,
    config: Flux3Config,
    residency: Residency<Flux3TextEncoder, Heavy>,
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

pub(crate) fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_deferred_load_shape(MODEL_ID, spec)?;
    let refusals: [(&str, bool); 8] = [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
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
                 '{TEXT_ENCODER_COMPONENT}' component)"
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
    gen_core::reject_unsupported_adapters(
        MODEL_ID,
        &descriptor().capabilities,
        spec.adapters.len(),
    )?;
    Ok(Box::new(load_inner(spec)?))
}

fn load_inner(spec: &LoadSpec) -> Result<Flux3Klein> {
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
    if !config.dit.flow_map {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: {} is a CFG checkpoint; this route serves the distilled flow map (the \
             base checkpoint is sc-26239)",
            config.model
        )));
    }
    let resources = GenerationResources::from_spec(spec, &config, MODEL_ID)?;
    // Priced from the resources' tensor headers BEFORE any weight is read, so a spec the
    // registration refuses is the same refusal through `load` on every offload policy
    // (`Sequential` defers every component to the first render).
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    let device = candle_gen::default_device()?;
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let te_device = device.clone();
    let text_in = config.dit.text_in_channels;
    let dit_path = resources.dit_weights();
    let ae_path = resources.autoencoder_file.clone();
    let heavy_cfg = config.clone();
    let heavy_device = device.clone();
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || Flux3TextEncoder::load(&te_dir, &te_cfg, text_in, &te_device),
        move |_use_pid| {
            let dit = load_dit(&dit_path, &heavy_cfg, compute, &heavy_device)?;
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
        },
    )?;
    Ok(Flux3Klein {
        descriptor: descriptor(),
        config,
        residency,
        spec: spec.clone(),
        memory_strategy,
    })
}

impl Flux3Klein {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate_impl(req)?;
        // Upstream `generate.py`: an unseeded request is seeded from its caption.
        let params = GenerationParams::resolve(req, prompt_seed(&req.prompt), &self.config)?;
        let total = params.steps * req.count;
        let images = self.residency.run_request_scoped(
            stage_residency_for(&self.spec, req),
            false,
            &req.cancel,
            false,
            on_progress,
            |te: &Flux3TextEncoder| -> Result<TextConditioning> { te.encode(&params.prompt) },
            |_| Ok(()),
            |heavy: &Heavy, text: TextConditioning, on_progress| {
                // Every image's Euler loop first, then ONE `Decoding` and the decodes: the
                // progress contract reports the decode phase once per request.
                let mut tokens = Vec::with_capacity(req.count as usize);
                for c in 0..req.count {
                    let done = c * params.steps;
                    let seed = candle_gen::image_seed(params.seed, c);
                    tokens.push(denoise_one(
                        &heavy.dit,
                        &text,
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
                let mut images = Vec::with_capacity(tokens.len());
                for t in &tokens {
                    images.push(decode_tokens(&heavy.ae, t, h, w)?);
                }
                Ok(images)
            },
        )?;
        Ok(GenerationReport {
            output: Some(GenerationOutput::Images(images)),
            artifacts: None,
            warnings: Vec::new(),
            prefix_kv_cache: None,
        })
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        validate_generation_request(MODEL_ID, req, &self.config)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

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
        assert!(!d.capabilities.supports_lora && !d.capabilities.supports_lokr);
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
        quant.quantize = Some(gen_core::Quant::Q4);
        let mut adapters = sentinel();
        adapters.adapters.push(gen_core::AdapterSpec {
            path: "/lora-not-on-disk".into(),
            scale: 1.0,
            kind: gen_core::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        });
        let mut pid = sentinel();
        pid.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File("/pid-not-on-disk.safetensors".into()),
            gemma: WeightsSource::Dir("/gemma-not-on-disk".into()),
        });
        for (field, spec) in [
            ("quantize", quant),
            ("adapters", adapters),
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
