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
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant,
    Residency, Result, SizeFloor,
};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;

use crate::dit::IrisDiT;
use crate::nn::BackboneQuant;
use crate::pipeline::{encode_in, noise_batch, to_images, Conditioning};
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
            // sc-26266: the affine tiers, quantized at load from the dense release — the backbone's
            // projections (the SwiGLU down projections zero-padded to a group multiple) and the
            // Qwen3-VL tower's attention/MLP projections. Embeddings, norms, the text position
            // table and the few sub-group-wide pixel heads stay dense; see `crate::nn` and
            // `crate::text_encoder`.
            supported_quants: &[Quant::Q8, Quant::Q4],
            supports_sequential_offload: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// Label of the backbone's iOS pre-load check.
pub(crate) const BACKBONE_LOAD_QUANTUM: &str = "Iris backbone load";
/// Label of one solved batch's RGB8 conversion quantum.
pub(crate) const IMAGE_QUANTUM: &str = "Iris image conversion";

/// RGB8 conversion of an evaluated `[B, 3, H, W]` solved batch, one row at a time. The
/// denoise finalization quantum has already materialized its clamp. Denormalization, rounding and
/// NHWC layout hold four f32 row copies and four f32 scalars; the host u8 pixels of all rows stay
/// allocated. Count each row buffer at its allocator-rounded size.
pub(crate) fn image_scratch(elements: u64, images: u64) -> u64 {
    let row = mlx_gen::ios_quantum::tensor_bytes(4, &[elements / images.max(1)]);
    4 * crate::retention::allocated(row) + 16 + elements
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
    /// A deferred-materialization load (rung 4): both components stream their stacks by window.
    streamed: bool,
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
    load_backbone_quantized(dir, config, compute, adapters, task, base_model, None)
}

/// [`load_backbone_with_adapters`] at a quantization tier (sc-26266): `quant` packs every
/// projection [`crate::nn::ProjectionLayout::of`] selects to Q4/Q8 at load; `None` is the dense
/// tier. Adapters are installed on the dense projections first and stay forward-time residuals
/// over the packed base. The packed projections are evaluated before this returns and the dense
/// copies are dropped with the source map, so what stays resident is the packed model.
pub fn load_backbone_quantized(
    dir: &Path,
    config: &IrisConfig,
    compute: Dtype,
    adapters: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
    quant: Option<Quant>,
) -> Result<(IrisDiT, Vec<AdapterApplyReport>)> {
    #[cfg(all(test, target_os = "macos"))]
    crate::anchor_tests::record_generation_loader_entry("backbone");
    let requested = quant;
    let quant = BackboneQuant::for_load(quant)?;
    let path = dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
    // A pre-quantized artifact (`crate::packed`) is read packed — no dense transient, no load-time
    // quantization — and only at its own tier, under the bf16 compute its scales are stored at.
    // A dense artifact keeps the release path below unchanged.
    let artifact = crate::packed::artifact_tier(&path)?;
    crate::packed::check_requested("the backbone", artifact, requested)?;
    if artifact.is_some() && compute != Dtype::Bfloat16 {
        return Err(Error::Unsupported(format!(
            "iris: {} is a packed {artifact:?} artifact, which runs under the bf16 compute policy",
            path.display()
        )));
    }
    let quant = if artifact.is_some() { None } else { quant };
    mlx_gen::ios_quantum::before_load(BACKBONE_LOAD_QUANTUM, std::slice::from_ref(&path))?;
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
        let array = read_backbone_tensor(&key, array, compute)?;
        weights.insert(key, array);
    }
    let installed = crate::adapters::install(&mut weights, compute, adapters, task, base_model)?;
    let (adapted, reports) = match installed {
        Some((adapted, reports)) => (Some(adapted), reports),
        None => (None, Vec::new()),
    };
    let dit =
        IrisDiT::from_weights_quantized(&weights, &config.model, compute, adapted.as_ref(), quant)?;
    drop(adapted);
    // Every derived parameter is evaluated before any forward can read it.
    let arrays = dit.arrays();
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    drop(weights);
    Ok((dit, reports))
}

/// Read one backbone tensor the loaders' way: its file bytes on MLX's CPU stream first (a GPU cast
/// over an unread `Load` makes the Metal command buffer wait on the disk, which on a cold page cache
/// trips the GPU watchdog — sc-24245), then every matrix but the text position table and a packed
/// projection's stored parts cast to the compute dtype, evaluated before the next tensor is read.
pub(crate) fn read_backbone_tensor(
    key: &str,
    array: mlx_rs::Array,
    compute: Dtype,
) -> Result<mlx_rs::Array> {
    mlx_rs::transforms::eval_pending_loads([&array])?;
    let array = if array.ndim() >= 2 && key != "y_pos_embedding" && !is_packed_part(key, &array) {
        array.as_dtype(compute)?
    } else {
        array
    };
    eval([&array])?;
    Ok(array)
}

/// The streamed backbone of a deferred-materialization load (rung 4): the skeleton (every tensor
/// but the four block stacks) read and evaluated now, the stacks read per window at forward time
/// (`BlockStream`). `default_window` is the window a forward that names none runs
/// (`None`: the whole stack). A packed artifact streams at its own tier; the dense release at the
/// dense tier only (`refuse_stream_requantization`).
pub fn load_backbone_streamed(
    dir: &Path,
    config: &IrisConfig,
    compute: Dtype,
    quant: Option<Quant>,
    default_window: Option<usize>,
) -> Result<IrisDiT> {
    #[cfg(all(test, target_os = "macos"))]
    crate::anchor_tests::record_generation_loader_entry("backbone");
    let path = dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
    let artifact = crate::packed::artifact_tier(&path)?;
    crate::packed::check_requested("the backbone", artifact, quant)?;
    crate::stream::refuse_stream_requantization(MODEL_ID, "the backbone", artifact, quant)?;
    if artifact.is_some() && compute != Dtype::Bfloat16 {
        return Err(Error::Unsupported(format!(
            "iris: {} is a packed {artifact:?} artifact, which runs under the bf16 compute policy",
            path.display()
        )));
    }
    let skeleton = |name: &str| crate::dit::Stack::of(name).is_none();
    mlx_gen::ios_quantum::before_load_where(
        BACKBONE_LOAD_QUANTUM,
        std::slice::from_ref(&path),
        skeleton,
    )?;
    let mut weights = Weights::from_file(&path)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    let mut keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    keys.sort();
    for key in keys {
        let Some(array) = weights.remove(&key) else {
            continue;
        };
        // A stack tensor's lazy handle is dropped unread: its window reads it.
        if skeleton(&key) {
            let array = read_backbone_tensor(&key, array, compute)?;
            weights.insert(key, array);
        }
    }
    let stream = crate::dit::BlockStream::open(&path, "", &config.model, compute, default_window)?;
    let dit = IrisDiT::from_weights_streamed(
        &weights,
        &config.model,
        compute,
        config.model.in_channels,
        stream,
    )?;
    for group in dit.arrays().chunks(64) {
        eval(group.iter().copied())?;
    }
    drop(weights);
    Ok(dit)
}

/// Whether `key` is part of a packed projection ([`crate::packed`]): its u32 codes or its stored
/// scales / biases, which every loader keeps exactly as stored.
pub(crate) fn is_packed_part(key: &str, array: &mlx_rs::Array) -> bool {
    array.dtype() == Dtype::Uint32 || key.ends_with(".scales") || key.ends_with(".biases")
}

fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_streamed_adapters(MODEL_ID, spec)?;
    crate::memory_strategy::refuse_unloadable_quant(MODEL_ID, spec)?;
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
    Ok(Box::new(load_model(spec)?))
}

/// The concrete loader shared by the public adapter and endpoint-ownership proofs.
pub(crate) fn load_model(spec: &LoadSpec) -> Result<Iris3b> {
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
    // One tier for the whole pipeline: the text tower and the backbone load at the same width.
    let quant = spec.quantize;
    let adapter_reports = Arc::new(Mutex::new(Vec::new()));
    let reports_sink = Arc::clone(&adapter_reports);
    // Rung 4: a deferred-materialization load streams both components' stacks by window; which
    // window each runs is a per-request decision ([`request_rungs`]).
    let streamed = spec.load_shape == gen_core::LoadShape::DeferredMaterialization;
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || {
            let tower = if streamed {
                IrisTextEncoder::load_streamed(&te_dir, &te_cfg, quant)
            } else {
                IrisTextEncoder::load_quantized(&te_dir, &te_cfg, quant)
            }?;
            #[cfg(all(test, target_os = "macos"))]
            crate::anchor_tests::observe_generation_tower(&tower);
            Ok(tower)
        },
        move |_use_pid| {
            if streamed {
                let dit = load_backbone_streamed(&bb_dir, &bb_cfg, compute, quant, None)?;
                #[cfg(all(test, target_os = "macos"))]
                crate::anchor_tests::observe_generation_backbone(&dit);
                return Ok(Heavy { dit });
            }
            let (dit, reports) = load_backbone_quantized(
                &bb_dir,
                &bb_cfg,
                compute,
                &adapters,
                IrisTask::Generation,
                MODEL_ID,
                quant,
            )?;
            if let Ok(mut sink) = reports_sink.lock() {
                *sink = reports;
            }
            #[cfg(all(test, target_os = "macos"))]
            crate::anchor_tests::observe_generation_backbone(&dit);
            Ok(Heavy { dit })
        },
    )?;
    // Priced from the resources' tensor headers after the components loaded, so a broken
    // resource surfaces as its loader's own error.
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    Ok(Iris3b {
        descriptor: descriptor(),
        config,
        residency,
        adapter_reports,
        spec: spec.clone(),
        memory_strategy,
        streamed,
    })
}

impl Iris3b {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn assert_retained_endpoint(&self, owns_components: bool) -> Option<Vec<usize>> {
        let owned = self
            .residency
            .with_resident_parts(|tower, heavy| {
                tower.assert_endpoint_ownership(false);
                heavy.dit.assert_endpoint_ownership(false);
                let mut addresses = tower.endpoint_parameter_addresses();
                addresses.extend(heavy.dit.endpoint_parameter_addresses());
                addresses
            })
            .unwrap();
        assert_eq!(
            owned.is_some(),
            owns_components,
            "endpoint retained-component ownership differs from its execution regime"
        );
        owned
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate_impl(req)?;
        let rungs = request_rungs(req, self.streamed)?;
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let params = GenerationParams::resolve(req, default_seed(), &self.config)?;
        let rows = params.prompts.len() as u32;
        mlx_gen::ios_quantum::diagnostic_metadata(
            MODEL_ID,
            "request",
            &[
                ("stage", u64::from(stage_residency_for(&self.spec, req))),
                ("serial_count", u64::from(req.count)),
                ("prompt_rows", u64::from(rows)),
                ("steps", params.steps as u64),
                ("load_streamed", u64::from(self.streamed)),
            ],
        );
        // Progress is a u32 UI counter; exact execution counts are emitted as u64 metadata.
        let total = (params.steps as u32).saturating_mul(req.count);
        let flow = self.config.flow.clone();
        // Staged residency is a per-request execution decision: the shared worker selects it
        // through the memory-strategy scope, which writes `GenerationMemory` onto the request.
        let (images, warnings) = self.residency.run_request_scoped(
            stage_residency_for(&self.spec, req),
            false,
            &req.cancel,
            false,
            on_progress,
            |te: &IrisTextEncoder| -> Result<Conditioning> {
                encode_in(te, &params, rungs.text_window, &req.cancel)
            },
            |_| Ok(()),
            |heavy: &Heavy, conditioning: Conditioning, on_progress| {
                let channels = heavy.dit.config().in_channels;
                let mut images = Vec::with_capacity((rows * req.count) as usize);
                // One batched solve over the whole prompt batch per `count` repetition; image
                // `k = c·rows + j` draws its noise from `seed + k`.
                for c in 0..req.count {
                    mlx_gen::ios_quantum::diagnostic_metadata(
                        MODEL_ID,
                        "serial",
                        &[
                            ("serial", u64::from(c)),
                            ("completed_batches", images.len() as u64),
                            (
                                "completed_rows",
                                images
                                    .iter()
                                    .map(|image: &mlx_rs::Array| image.shape()[0] as u64)
                                    .sum(),
                            ),
                            ("backbone_streamed", u64::from(heavy.dit.is_streamed())),
                        ],
                    );
                    #[cfg(all(test, target_os = "macos"))]
                    crate::anchor_tests::observe_serial_outputs(c, &images);
                    let seeds: Vec<u64> = (0..rows)
                        .map(|j| params.seed.wrapping_add((c * rows + j) as u64))
                        .collect();
                    let z = noise_batch(&seeds, channels, params.width, params.height)?;
                    let done = c.saturating_mul(params.steps as u32);
                    let x = crate::pipeline::denoise_in(
                        &heavy.dit,
                        &flow,
                        &conditioning,
                        &z,
                        &params,
                        &req.cancel,
                        |step| {
                            on_progress(Progress::Step {
                                current: done.saturating_add(step as u32),
                                total,
                            })
                        },
                        &req.preview,
                        rungs.budget,
                        rungs.dit_window,
                        rungs.pixel_rows,
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
                    mlx_gen::ios_quantum::before(
                        IMAGE_QUANTUM,
                        image_scratch(x.size() as u64, x.shape()[0] as u64),
                    )?;
                    out.extend(to_images(x)?);
                    mlx_gen::ios_quantum::finish(IMAGE_QUANTUM, &[])?;
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
        request_rungs(req, self.streamed)?;
        Ok(())
    }
}

/// What one generation request runs with: rung 3's attention budget and its pixel-row bound
/// (the PiT stage one batch row at a time), and rung 4's windows — `None` for a component the
/// request does not stream, which on a deferred load runs its whole stack in one window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestRungs {
    pub budget: mlx_gen::AttentionBudget,
    pub pixel_rows: bool,
    pub text_window: Option<usize>,
    pub dit_window: Option<usize>,
}

/// Resolve a request's admitted `GenerationMemory` against the load: rung 4 only on a
/// deferred-materialization load (`streamed`), at a published window and component scope.
pub(crate) fn request_rungs(req: &GenerationRequest, streamed: bool) -> Result<RequestRungs> {
    let budget = attention_budget(req)?;
    let Some(memory) = req.memory else {
        return Ok(RequestRungs {
            budget,
            pixel_rows: false,
            text_window: None,
            dit_window: None,
        });
    };
    let pixel_rows = memory.chunk_attention;
    if !memory.stream_transformer_blocks {
        if memory.transformer_window_size.is_some() || memory.transformer_window_component.is_some()
        {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: transformer_window_size / transformer_window_component require \
                 stream_transformer_blocks"
            )));
        }
        return Ok(RequestRungs {
            budget,
            pixel_rows,
            text_window: None,
            dit_window: None,
        });
    }
    if !streamed {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: GenerationMemory::stream_transformer_blocks needs a \
             deferred-materialization load (LoadShape::DeferredMaterialization); this generator \
             holds its stacks resident, so a window would add memory rather than bound it"
        )));
    }
    let window = memory
        .transformer_window_size
        .unwrap_or(crate::stream::TRANSFORMER_WINDOW_SIZE);
    if !crate::stream::TRANSFORMER_WINDOW_SIZES.contains(&window) {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: transformer_window_size {window} is outside the published domain {:?}",
            crate::stream::TRANSFORMER_WINDOW_SIZES
        )));
    }
    let component = memory.transformer_window_component.unwrap_or_default();
    Ok(RequestRungs {
        budget,
        pixel_rows,
        text_window: component.includes_text_encoder().then_some(window as usize),
        dit_window: component.includes_dit().then_some(window as usize),
    })
}

/// Refuse the memory-ladder controls this route has no mechanism for, so none is accepted and
/// ignored: there is no decoder to tile (pixel space). Staged residency, bounded attention and
/// (on a deferred load) the transformer window are request-scoped controls ([`request_rungs`]).
pub(crate) fn refuse_unhonored_memory(req: &GenerationRequest) -> Result<()> {
    let Some(memory) = req.memory else {
        return Ok(());
    };
    attention_budget(req)?;
    let refusals = [("tile_vae_decode", memory.tile_vae_decode)];
    for (control, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: GenerationMemory::{control} is not a memory strategy of the MLX \
                 Iris-3B generation route"
            )));
        }
    }
    Ok(())
}

/// Resolve only the provider's published parameter; never silently accept another budget.
fn attention_budget(req: &GenerationRequest) -> Result<mlx_gen::AttentionBudget> {
    match req.memory {
        Some(m) if m.chunk_attention => {
            let size = m
                .attention_chunk_size
                .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE);
            if size != crate::memory_strategy::ATTENTION_CHUNK_SIZE {
                return Err(Error::Unsupported(format!(
                    "{MODEL_ID}: attention_chunk_size {size} is outside the published domain"
                )));
            }
            Ok(mlx_gen::AttentionBudget::from_score_elements(
                u64::from(size),
                true,
            ))
        }
        Some(m) if m.attention_chunk_size.is_some() => Err(Error::Unsupported(format!(
            "{MODEL_ID}: attention_chunk_size requires chunk_attention"
        ))),
        _ => Ok(mlx_gen::AttentionBudget::UNBOUNDED),
    }
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

    /// The tiers the route cannot load are refused by name before any resource is read: NVFP4
    /// (no affine MLX kernel) and a quantized tier under FP32 compute (the packed scales are bf16).
    /// Q4/Q8 at bf16 get past the spec gate and fail only on the missing resource.
    ///
    /// *Mutation that reds this:* dropping `refuse_unloadable_quant` from
    /// `refuse_unsupported_spec`, or either arm of it.
    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let sentinel = || LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        let refused = |spec: &LoadSpec, field: &str| match load(spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains(field), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        };
        refused(&sentinel().with_quant(Quant::Nvfp4), "quantize");
        let mut fp32 = sentinel().with_quant(Quant::Q4);
        fp32.precision = Precision::Fp32;
        refused(&fp32, "precision");
        for quant in [Quant::Q4, Quant::Q8] {
            let err = load(&sentinel().with_quant(quant))
                .err()
                .expect("nothing is on disk");
            let message = err.to_string();
            assert!(
                !message.contains("quantize") && !message.contains("precision"),
                "{quant:?} must pass the spec gate, got {message}"
            );
        }
        let mut te = sentinel();
        te.text_encoder = Some(WeightsSource::Dir("/te".into()));
        refused(&te, "text_encoder");
    }

    /// The descriptor advertises exactly the tiers the loader accepts.
    #[test]
    fn the_descriptor_advertises_the_affine_tiers() {
        assert_eq!(
            descriptor().capabilities.supported_quants,
            [Quant::Q8, Quant::Q4]
        );
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

    /// The route honours staged residency and bounded attention; unsupported controls are refused by
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
                "attention_chunk_size",
                GenerationMemory {
                    chunk_attention: true,
                    attention_chunk_size: Some(1),
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

    /// Rung 4 resolves per request on a deferred load only: the published window, each component
    /// scope streaming exactly its component(s), the other one at its whole-stack window; on an
    /// eager load, out of the domain, or as a window without the stream flag, a typed refusal.
    /// Rung 3 carries the per-row PiT stage.
    ///
    /// *Mutations that red this:* accepting rung 4 on an eager load; swapping the scopes; dropping
    /// the domain check or the stray-window refusal; decoupling the pixel rows from rung 3.
    #[test]
    fn request_rungs_resolve_the_window_and_scope_on_a_deferred_load() {
        use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        let stream = |component: Option<TransformerComponent>, size: Option<u32>| {
            with(GenerationMemory {
                chunk_attention: true,
                stream_transformer_blocks: true,
                transformer_window_size: size,
                transformer_window_component: component,
                ..Default::default()
            })
        };
        let plain = request_rungs(&GenerationRequest::default(), true).unwrap();
        assert_eq!(
            (plain.text_window, plain.dit_window, plain.pixel_rows),
            (None, None, false)
        );
        for (component, text, dit) in [
            (None, None, Some(1)),
            (Some(TransformerComponent::Dit), None, Some(1)),
            (Some(TransformerComponent::TextEncoder), Some(1), None),
            (Some(TransformerComponent::Both), Some(1), Some(1)),
        ] {
            let rungs = request_rungs(&stream(component, Some(1)), true).unwrap();
            assert_eq!(
                (rungs.text_window, rungs.dit_window),
                (text, dit),
                "{component:?}"
            );
            assert!(rungs.pixel_rows);
        }
        let refused = |req: GenerationRequest, streamed: bool, needle: &str| match request_rungs(
            &req, streamed,
        ) {
            Err(Error::Unsupported(message)) => assert!(message.contains(needle), "{message}"),
            other => panic!("expected a typed refusal naming {needle}, got {other:?}"),
        };
        refused(stream(None, Some(1)), false, "deferred-materialization");
        refused(stream(None, Some(2)), true, "transformer_window_size 2");
        refused(
            with(GenerationMemory {
                transformer_window_size: Some(1),
                ..Default::default()
            }),
            true,
            "require stream_transformer_blocks",
        );
    }

    /// A deferred (streamed) load refuses an adapter stack by name before reading any resource.
    ///
    /// *Mutation that reds this:* dropping `refuse_streamed_adapters` from
    /// `refuse_unsupported_spec`.
    #[test]
    fn a_streamed_load_refuses_adapters() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()))
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        spec.adapters = vec![mlx_gen::AdapterSpec::new(
            "/iris-not-on-disk/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )];
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("adapters"), "{m}"),
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
