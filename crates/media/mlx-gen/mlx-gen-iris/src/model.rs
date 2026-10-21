//! `Iris3b` — the Iris-3B text-to-image [`Generator`]: descriptor, load, request validation and the
//! staged generate (text encoder → backbone), registered as [`MODEL_ID`].

use std::path::Path;
use std::sync::{Arc, Mutex};

use gen_core::iris::{
    validate_generation_request, GenerationParams, GenerationResources, IrisConfig, IrisTask,
    FAMILY, GENERATION_MODEL_ID, SAMPLERS,
};
use gen_core::GenerationReport;
use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, gen_core, AdapterApplyReport, AdapterSpec, Capabilities, Error, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Residency, Result, SizeFloor,
};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;

use crate::dit::IrisDiT;
use crate::pipeline::{denoise, encode, noise_batch, to_images, Conditioning};
use crate::text_encoder::IrisTextEncoder;

/// Registry id (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = GENERATION_MODEL_ID;
/// Output sides must be multiples of the patch size (load refuses a patch that does not divide it).
pub const SIZE_MULTIPLE: u32 = gen_core::iris::SIZE_MULTIPLE;
/// Smallest side: one patch.
pub const MIN_SIZE: u32 = 16;
/// Largest side served (the release renders ~1 MP; 2048² is the pixel-space attention ceiling
/// this route admits).
pub const MAX_SIZE: u32 = 2048;
/// Images per request.
pub const MAX_COUNT: u32 = 8;

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // Pixel space: there is no latent and no VAE.
        denoiser_output_latent_space: None,
        control_kinds: None,
        // The Qwen3-VL text encoder is a load-time component (`components["text_encoder"]`);
        // `load` refuses without it.
        required_components: &[gen_core::iris::TEXT_ENCODER_COMPONENT],
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // CFG against the negative prompt run through the same template (default "").
            supports_negative_prompt: true,
            // `guidance` is upstream's `cfg_scale` (default 3.0; 1.0 = CFG off).
            supports_guidance: true,
            supports_true_cfg: false,
            // The FlowDPM-Solver++ in its two upstream orders: `dpmpp_2m` (order 2, the default)
            // and `euler` (order 1 — the first-order DPM-Solver++ step IS the flow Euler step).
            // There is one schedule (the shifted linear grid), so no scheduler name is advertised;
            // `scheduler_shift` is upstream's `shift`.
            samplers: SAMPLERS.to_vec(),
            schedulers: Vec::new(),
            // upstream `cfg_interval`, a prompt list, and `text_encoder.on_caption_overflow`.
            supports_cfg_interval: true,
            supports_prompt_batch: true,
            supports_caption_overflow_policy: true,
            // LoRA / LoKr (and LyCORIS-layout LoHa) residuals on the backbone projections; every
            // file must carry the Iris `family` / `irisTask` stamps.
            supports_lora: true,
            supports_lokr: true,
            // Pixel space: every step's predicted clean image is the preview, decoded exactly.
            supports_preview: true,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: MAX_COUNT,
            mac_only: true,
            supported_quants: &[],
            supports_sequential_offload: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// The backbone half of the generation task.
pub struct Heavy {
    pub dit: IrisDiT,
}

/// A loaded Iris-3B generator.
pub struct Iris3b {
    descriptor: ModelDescriptor,
    config: IrisConfig,
    residency: Residency<IrisTextEncoder, Heavy>,
    /// One report per installed adapter file, from the most recent backbone load.
    adapter_reports: Arc<Mutex<Vec<AdapterApplyReport>>>,
    /// The spec this generator was loaded from: the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against (sc-25690).
    spec: LoadSpec,
    /// The loaded memory contract ([`crate::memory_strategy::generation_contract`] over the
    /// resources on disk), published through [`Generator::memory_strategy_contract`].
    memory_strategy: gen_core::MemoryProviderContract,
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy request scope) OR a `Sequential` load policy. A request can
/// stage a warm-loaded generator, never un-stage a sequential one.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

/// The compute dtype a load spec selects: the release's bf16 autocast by default, upstream's FP32
/// CPU path on an explicit `Precision::Fp32`.
pub fn compute_dtype(spec: &LoadSpec) -> Dtype {
    match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    }
}

/// Load the backbone from its directory (`config.yaml` already parsed) and evaluate every
/// parameter before returning, so no forward ever reads a weight lazily.
pub fn load_backbone(dir: &Path, config: &IrisConfig, compute: Dtype) -> Result<IrisDiT> {
    Ok(load_backbone_with_adapters(dir, config, compute, &[], IrisTask::Generation, MODEL_ID)?.0)
}

/// [`load_backbone`] with `adapters` installed (in order) on its projections for `task` on
/// `base_model` (see [`crate::adapters`]); returns one report per adapter file.
pub fn load_backbone_with_adapters(
    dir: &Path,
    config: &IrisConfig,
    compute: Dtype,
    adapters: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
) -> Result<(IrisDiT, Vec<AdapterApplyReport>)> {
    let path = dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
    let mut weights = Weights::from_file(&path)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    // The release ships FP32. Cast every matrix to the compute dtype ONE TENSOR AT A TIME and
    // drop its source handle, so the load peak is the compute-dtype model plus a single FP32
    // tensor rather than both full copies. Vectors (norm gains, biases) and the text position
    // table stay as stored; the modules decide their own dtype.
    let mut keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    keys.sort();
    for key in keys {
        let Some(array) = weights.remove(&key) else {
            continue;
        };
        // Read the file bytes on MLX's CPU stream first: a GPU cast over an unread `Load` makes the
        // Metal command buffer wait on the disk, which on a cold page cache trips the GPU watchdog
        // (sc-24245).
        mlx_rs::transforms::eval_pending_loads([&array])?;
        let array = if array.ndim() >= 2 && key != "y_pos_embedding" {
            array.as_dtype(compute)?
        } else {
            array
        };
        eval([&array])?;
        weights.insert(key, array);
    }
    let installed = crate::adapters::install(&mut weights, compute, adapters, task, base_model)?;
    let (adapted, reports) = match installed {
        Some((adapted, reports)) => (Some(adapted), reports),
        None => (None, Vec::new()),
    };
    let dit = IrisDiT::from_weights_adapted(&weights, &config.model, compute, adapted.as_ref())?;
    drop(adapted);
    // Every derived parameter is evaluated before any forward can read it.
    let arrays = dit.arrays();
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    drop(weights);
    Ok((dit, reports))
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
                "{MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B generation route \
                 (the text encoder is the '{}' component)",
                gen_core::iris::TEXT_ENCODER_COMPONENT
            )));
        }
    }
    Ok(())
}

/// Construct the generator from its task resources: `spec.weights` = the backbone directory
/// (`config.yaml` + `model.safetensors`), `spec.components["text_encoder"]` = the
/// `Qwen/Qwen3-VL-4B-Instruct` snapshot. `Resident` (default) loads both now; `Sequential` loads
/// the encoder, encodes, drops it, then loads the backbone, per request. A `Resident` generator
/// runs one request the same staged way when its admitted `GenerationMemory::stage_residency` is
/// set and rebuilds the warm pair on the next unstaged request.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    refuse_unsupported_spec(spec)?;
    let resources = GenerationResources::from_spec(spec, MODEL_ID)?;
    let config = IrisConfig::from_dir(&resources.backbone_dir)?;
    config.validate_supported()?;
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let bb_dir = resources.backbone_dir.clone();
    let bb_cfg = config.clone();
    let adapters = spec.adapters.clone();
    let adapter_reports = Arc::new(Mutex::new(Vec::new()));
    let reports_sink = Arc::clone(&adapter_reports);
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || IrisTextEncoder::load(&te_dir, &te_cfg),
        move |_use_pid| {
            let (dit, reports) = load_backbone_with_adapters(
                &bb_dir,
                &bb_cfg,
                compute,
                &adapters,
                IrisTask::Generation,
                MODEL_ID,
            )?;
            if let Ok(mut sink) = reports_sink.lock() {
                *sink = reports;
            }
            Ok(Heavy { dit })
        },
    )?;
    // Priced from the resources' tensor headers after the components loaded, so a broken
    // resource surfaces as its loader's own error.
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    Ok(Box::new(Iris3b {
        descriptor: descriptor(),
        config,
        residency,
        adapter_reports,
        spec: spec.clone(),
        memory_strategy,
    }))
}

impl Iris3b {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate_impl(req)?;
        let params = GenerationParams::resolve(req, default_seed(), &self.config)?;
        let rows = params.prompts.len() as u32;
        let total = (params.steps as u32) * req.count;
        let flow = self.config.flow.clone();
        // Staged residency is a per-request execution decision: the shared worker selects it
        // through the memory-strategy scope, which writes `GenerationMemory` onto the request.
        let (images, warnings) = self.residency.run_request_scoped(
            stage_residency_for(&self.spec, req),
            false,
            &req.cancel,
            false,
            on_progress,
            |te: &IrisTextEncoder| -> Result<Conditioning> { encode(te, &params) },
            |_| Ok(()),
            |heavy: &Heavy, conditioning: Conditioning, on_progress| {
                let channels = heavy.dit.config().in_channels;
                let mut images = Vec::with_capacity((rows * req.count) as usize);
                // One batched solve over the whole prompt batch per `count` repetition; image
                // `k = c·rows + j` draws its noise from `seed + k`.
                for c in 0..req.count {
                    let seeds: Vec<u64> = (0..rows)
                        .map(|j| params.seed.wrapping_add((c * rows + j) as u64))
                        .collect();
                    let z = noise_batch(&seeds, channels, params.width, params.height)?;
                    let done = c * params.steps as u32;
                    let x = denoise(
                        &heavy.dit,
                        &flow,
                        &conditioning,
                        &z,
                        &params,
                        &req.cancel,
                        |step| {
                            on_progress(Progress::Step {
                                current: done + step as u32,
                                total,
                            })
                        },
                        &req.preview,
                    )?;
                    images.push(x);
                }
                // Pixel space: the "decode" is only the [−1, 1] → RGB8 conversion.
                on_progress(Progress::Decoding);
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let mut out = Vec::with_capacity((rows * req.count) as usize);
                for x in &images {
                    out.extend(to_images(x)?);
                }
                mlx_rs::memory::clear_cache();
                Ok((out, conditioning.warnings))
            },
        )?;
        Ok(GenerationReport {
            output: Some(GenerationOutput::Images(images)),
            artifacts: None,
            warnings: warnings
                .into_iter()
                .map(|message| gen_core::GenerationWarning {
                    code: "caption_truncated".into(),
                    message,
                })
                .collect(),
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
/// ignored: there is no decoder to tile (pixel space), no bounded-attention path through the
/// backbone and no block-streaming loader. Staged residency is the one control honoured
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
                 Iris-3B generation route (it stages residency only)"
            )));
        }
    }
    Ok(())
}

impl Generator for Iris3b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.adapter_reports
            .lock()
            .map(|reports| reports.clone())
            .unwrap_or_default()
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
            .ok_or_else(|| gen_core::Error::Msg("iris: the render produced no output".into()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_is_pixel_space_and_honest() {
        let d = descriptor();
        assert_eq!(d.id, "iris_3b");
        assert_eq!(d.family, "iris");
        assert!(d.denoiser_output_latent_space.is_none());
        assert_eq!(d.capabilities.samplers, ["dpmpp_2m", "euler"]);
        assert!(d.capabilities.schedulers.is_empty());
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert!(d.capabilities.supports_preview);
        assert_eq!(d.required_components, ["text_encoder"]);
    }

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Q4);
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }

    /// A request stages when the memory-strategy scope said so OR the load policy is
    /// `Sequential`; an explicit all-off memory block does not un-stage a sequential load.
    ///
    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use mlx_gen::gen_core::{GenerationMemory, OffloadPolicy};
        let resident = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
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
        let unstaged = GenerationRequest {
            memory: Some(GenerationMemory::default()),
            ..Default::default()
        };
        assert!(!stage_residency_for(&resident, &plain));
        assert!(!stage_residency_for(&resident, &unstaged));
        assert!(stage_residency_for(&resident, &staged));
        assert!(stage_residency_for(&sequential, &plain));
        assert!(stage_residency_for(&sequential, &unstaged));
        assert!(stage_residency_for(&sequential, &staged));
    }

    /// The route honours staged residency only; every other memory-ladder control is refused by
    /// name rather than accepted and ignored.
    ///
    /// *Mutation that reds this:* removing any row of `refuse_unhonored_memory`'s table.
    #[test]
    fn unhonoured_memory_controls_are_typed_refusals() {
        use mlx_gen::gen_core::GenerationMemory;
        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        refuse_unhonored_memory(&GenerationRequest::default()).unwrap();
        refuse_unhonored_memory(&with(GenerationMemory::default())).unwrap();
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

    /// No Iris loader has a deferred-materialization load shape; the load refuses one by name
    /// before reading any resource.
    ///
    /// *Mutation that reds this:* dropping `refuse_deferred_load_shape` from
    /// `refuse_unsupported_spec`.
    #[test]
    fn a_deferred_load_shape_is_a_typed_refusal() {
        let spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()))
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("load_shape"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }

    #[test]
    fn a_missing_text_encoder_is_a_load_error() {
        let dir = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let err = load(&spec).err().expect("missing text encoder must fail");
        assert!(err.to_string().contains("text_encoder"), "{err}");
    }
}
