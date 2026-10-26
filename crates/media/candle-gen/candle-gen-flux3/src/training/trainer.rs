//! Registered production Candle Trainer: canonical pinned dense base -> frozen
//! native AE/teacher cache -> real LoRA session -> strict state/export lifecycle.
use super::{
    runner::{CachedExample, CandleSession},
    TrainingGraph,
};
use crate::autoencoder::FrozenTrainingEncoder;
use crate::{Flux3Dit, Flux3TextEncoder};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{self, CancelFlag, LoadSpec, Modality, Precision, WeightsSource};
use candle_gen::train::optim::TrainOptimizer;
use candle_gen::{CandleError as Error, Result};
use gen_core::flux3::training::{
    controller::{self, RunPlan},
    identity, lora_targets,
    options::{TrainingOptions, TrainingPrecision},
    preprocess::{self, PixelOptions},
    rng::NativeRng,
    sampler::BatchSampler,
};
use gen_core::flux3::{
    assemble_window, Flux3Config, GenerationResources, AUTOENCODER_COMPONENT, BASE_MODEL_ID,
    DIT_WEIGHTS_FILE, FAMILY, TEXT_ENCODER_COMPONENT,
};
use gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    TrainingTechniques,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 trainer: {detail}"))
}
fn check(flag: &CancelFlag) -> Result<()> {
    if flag.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: BASE_MODEL_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: false,
        max_reference_images: 1,
        techniques: TrainingTechniques::NONE,
    }
}
pub struct Flux3Trainer {
    spec: LoadSpec,
    descriptor: TrainerDescriptor,
}
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    gen_core::train::refuse_trainer_load_overlays(BASE_MODEL_ID, spec).map_err(Error::from)?;
    crate::memory_strategy::refuse_deferred_load_shape(BASE_MODEL_ID, spec).map_err(Error::from)?;
    if !matches!(spec.weights, WeightsSource::Dir(_))
        || spec.quantize.is_some()
        || !spec.adapters.is_empty()
        || spec.pid.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(invalid("requires the dense pinned base directory and no quantization/overlaid adapters/auxiliary graph; no dense or Python substitution"));
    }
    gen_core::control::reject_unknown_components(
        spec,
        &[TEXT_ENCODER_COMPONENT, AUTOENCODER_COMPONENT],
        BASE_MODEL_ID,
    )
    .map_err(Error::from)?;
    // Deliberately lazy: no device context, weights or dataset reads during capability construction.
    Ok(Box::new(Flux3Trainer {
        spec: spec.clone(),
        descriptor: trainer_descriptor(),
    }))
}
candle_gen::register_trainer! {pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer}
impl Trainer for Flux3Trainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        let opts = TrainingOptions::parse(req)?;
        controller::ensure_new_output(req)?;
        if opts.precision == TrainingPrecision::Bf16
            && !cfg!(any(feature = "cuda", feature = "metal"))
        {
            return Err(gen_core::Error::Unsupported("Candle CPU teacher cannot execute BF16; select F32 or a supported native GPU backend".into()));
        }
        if self.spec.precision == Precision::Fp32 && opts.precision != TrainingPrecision::F32 {
            return Err(gen_core::Error::Unsupported(
                "explicit Fp32 LoadSpec and selected training precision disagree".into(),
            ));
        }
        Ok(())
    }
    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        execute(&self.spec, req, on_progress).map_err(Into::into)
    }
}

fn host_pixels(image: &gen_core::Image) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        preprocess::normalized_hwc(image).map_err(Error::from)?,
        (1, image.height as usize, image.width as usize, 3),
        &Device::Cpu,
    )?
    .permute((0, 3, 1, 2))?
    .contiguous()?)
}
/// All source tensors cast on HOST before upload; a typed GPU map is shared by
/// VarBuilder/model, so no full original-dtype GPU map shadows the F32/BF16 base.
fn load_training_dit(
    path: &Path,
    cfg: &gen_core::flux3::DitConfig,
    dtype: DType,
    device: &Device,
    flag: &CancelFlag,
) -> Result<Flux3Dit> {
    let host = candle_gen::mmap_var_builder(&[path.to_path_buf()], DType::F32, &Device::Cpu)?;
    let headers =
        gen_core::weightsmeta::safetensors_path_tensor_headers(path).map_err(Error::from)?;
    let mut typed = HashMap::new();
    for header in headers {
        check(flag)?;
        if !header.is_float() {
            return Err(invalid(
                "nonfloating/packed DiT is not a dense training base",
            ));
        }
        let tensor = host
            .get_unchecked(&header.name)?
            .to_dtype(dtype)?
            .to_device(device)?;
        typed.insert(header.name, tensor);
    }
    let vb = VarBuilder::from_tensors(typed, dtype, device);
    let result = Flux3Dit::from_var_builder(&vb, cfg, dtype, device);
    // Hold the owning map until its actual device queue is drained on success/error.
    let sync = device.synchronize();
    drop(vb);
    match result {
        Err(error) => Err(error),
        Ok(dit) => {
            sync?;
            Ok(dit)
        }
    }
}
/// The dense CFG base this trainer fine-tunes, resolved header-authoritatively (the mode is the
/// DiT checkpoint's `ref_time_in` evidence, never the descriptive `config.json`), refused unless
/// it IS the base route and the pinned release config, and its full dense physical inventory
/// preflighted from the headers — all before any resource, text-encoder, autoencoder, LoRA-target,
/// dataset or device work.
fn resolve_training_base(root: &Path) -> Result<Flux3Config> {
    let cfg = gen_core::flux3::resolve_checkpoint(root).map_err(Error::from)?;
    cfg.validate_route(BASE_MODEL_ID).map_err(Error::from)?;
    identity::validate_release_config(&cfg).map_err(Error::from)?;
    let headers = gen_core::flux3::checkpoint_tensor_headers(root).map_err(Error::from)?;
    crate::dit::validate_checkpoint_headers(
        &cfg.dit,
        gen_core::flux3::artifacts::NativeDitStorage::Publisher(
            gen_core::flux3::quantization::DitFormat::Dense,
        ),
        &headers,
    )?;
    Ok(cfg)
}

fn execute(
    spec: &LoadSpec,
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    check(&req.cancel)?;
    let opts = TrainingOptions::parse(req).map_err(Error::from)?;
    on_progress(TrainingProgress::Preparing);
    check(&req.cancel)?;
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => return Err(invalid("base directory required")),
    };
    let cfg = resolve_training_base(root)?;
    let resources =
        GenerationResources::from_spec(spec, &cfg, BASE_MODEL_ID).map_err(Error::from)?;
    let artifacts =
        identity::observe_pinned_resources(&resources, &req.cancel).map_err(Error::from)?;
    let targets = lora_targets(&cfg.dit, &req.config.lora_target_modules).map_err(Error::from)?;
    let mut pixels = Vec::with_capacity(req.items.len());
    for (index, item) in req.items.iter().enumerate() {
        check(&req.cancel)?;
        pixels.push(
            preprocess::decode_pair(
                &item.image_path,
                item.reference_image_paths.first().map(|p| p.as_path()),
                PixelOptions {
                    width: req.config.resolution,
                    height: req.config.resolution,
                    center_crop: opts.center_crop,
                    random_flip: opts.random_flip,
                    seed: req.config.seed,
                },
                index as u64,
            )
            .map_err(Error::from)?,
        );
    }
    let geometry = pixels
        .iter()
        .map(|p| p.geometry.clone())
        .collect::<Vec<_>>();
    let bucket_ids = vec![0usize; pixels.len()];
    let sampler = BatchSampler::new(&bucket_ids, req.config.batch_size, req.config.seed)
        .map_err(Error::from)?;
    let plan = RunPlan {
        schedule: opts
            .schedule(&req.config, sampler.len() as u64)
            .map_err(Error::from)?,
        sampler,
        max_grad_norm: opts.max_grad_norm,
    };
    let identity = identity::build(
        req,
        &cfg,
        &opts,
        artifacts,
        &targets,
        &geometry,
        if cfg!(feature = "cuda") {
            "candle-cuda"
        } else if cfg!(feature = "metal") {
            "candle-metal"
        } else {
            "candle-cpu"
        },
    )
    .map_err(Error::from)?;
    if req.config.resume {
        controller::find_resume(
            &req.output_dir,
            controller::output_stem(&req.file_name).map_err(Error::from)?,
            &identity,
            plan.policy(req),
            "flux3-native-training-resume-v1",
            req.config.seed,
            &req.cancel,
        )
        .map_err(Error::from)?
        .ok_or_else(|| {
            invalid("resume requested but no compatible saved state exists; refusing restart")
        })?;
    }
    check(&req.cancel)?;
    on_progress(TrainingProgress::LoadingModel);
    check(&req.cancel)?;
    let device = candle_gen::default_device()?;
    let dtype = match opts.precision {
        TrainingPrecision::F32 => DType::F32,
        TrainingPrecision::Bf16 => DType::BF16,
    };
    if device.is_cpu() && dtype == DType::BF16 {
        return Err(invalid(
            "BF16 teacher is unsupported on Candle CPU; use F32 or native CUDA",
        ));
    }
    let encoder = FrozenTrainingEncoder::load(&resources.autoencoder_file, dtype, &device)?;
    let teacher = match Flux3TextEncoder::load_for_training(
        &resources.text_encoder_dir,
        &cfg.text_encoder,
        cfg.dit.text_in_channels,
        &device,
        dtype,
        opts.max_sequence_length,
    ) {
        Ok(teacher) => teacher,
        Err(error) => {
            let _ = encoder.device().synchronize();
            drop(encoder);
            return Err(error);
        }
    };
    let cached = (|| -> Result<Vec<CachedExample>> {
        let mut windows = req
            .items
            .iter()
            .map(|item| teacher.window(&item.caption))
            .collect::<Result<Vec<_>>>()?;
        if req.config.batch_size > 1 {
            let length = windows
                .iter()
                .map(|w| w.length())
                .max()
                .ok_or_else(|| invalid("empty caption windows"))?;
            for window in &mut windows {
                let truncated = window.truncated_tokens;
                *window = assemble_window(
                    &window.input_ids[..window.real],
                    teacher.pad_id(),
                    length,
                    opts.max_sequence_length,
                )
                .map_err(Error::from)?;
                window.truncated_tokens = truncated;
            }
        }
        let empty = teacher.window("")?;
        let mut empty_by_window: BTreeMap<usize, Tensor> = BTreeMap::new();
        let mut captions: BTreeMap<(String, usize), Tensor> = BTreeMap::new();
        let mut cache = Vec::with_capacity(req.items.len());
        for (index, ((item, pair), window)) in
            req.items.iter().zip(&pixels).zip(windows).enumerate()
        {
            check(&req.cancel)?;
            let (latent, grid) = encoder.encode(&host_pixels(&pair.target)?)?;
            let reference = pair
                .reference
                .as_ref()
                .map(|image| -> Result<Tensor> {
                    let (tokens, reference_grid) = encoder.encode(&host_pixels(image)?)?;
                    if reference_grid != grid {
                        return Err(invalid("paired encoder grids differ"));
                    }
                    Ok(tokens.to_device(&Device::Cpu)?)
                })
                .transpose()?;
            let key = (item.caption.clone(), window.length());
            let text = match captions.get(&key) {
                Some(text) => text.clone(),
                None => {
                    let text = teacher
                        .encode_window(window.clone())?
                        .tokens
                        .to_device(&Device::Cpu)?;
                    captions.insert(key, text.clone());
                    text
                }
            };
            let empty_text = if opts.caption_dropout > 0.0 {
                let length = window.length();
                Some(match empty_by_window.get(&length) {
                    Some(text) => text.clone(),
                    None => {
                        let w = assemble_window(
                            &empty.input_ids[..empty.real],
                            teacher.pad_id(),
                            length,
                            opts.max_sequence_length,
                        )
                        .map_err(Error::from)?;
                        let text = teacher.encode_window(w)?.tokens.to_device(&Device::Cpu)?;
                        empty_by_window.insert(length, text.clone());
                        text
                    }
                })
            } else {
                None
            };
            cache.push(CachedExample {
                latent: latent.to_device(&Device::Cpu)?,
                text,
                text_ids: gen_core::flux3::text_ids(window.length()),
                empty_text,
                reference,
                grid,
            });
            on_progress(TrainingProgress::Caching {
                current: (index + 1) as u32,
                total: req.items.len() as u32,
            });
            check(&req.cancel)?;
        }
        Ok(cache)
    })();
    let encoder_sync = encoder.device().synchronize();
    let teacher_sync = teacher.device().synchronize();
    drop(teacher);
    drop(encoder);
    drop(pixels);
    let examples = match cached {
        Err(error) => return Err(error),
        Ok(cache) => {
            encoder_sync?;
            teacher_sync?;
            cache
        }
    };
    check(&req.cancel)?;
    // Re-read identities before any optimization: mutable input/source replacements
    // during caching must refuse, never continue under an earlier provenance hash.
    if gen_core::train::resume::request_fingerprint(req).map_err(Error::from)? != identity.dataset
        || identity::observe_pinned_resources(&resources, &req.cancel).map_err(Error::from)?
            != identity.artifacts
    {
        return Err(invalid(
            "dataset/base/component bytes changed during frozen caching; refusing training",
        ));
    }
    let dit = load_training_dit(
        &resources.package_dir.join(DIT_WEIGHTS_FILE),
        &cfg.dit,
        dtype,
        &device,
        &req.cancel,
    )?;
    let mut initial = BTreeMap::new();
    for (index, target) in targets.iter().enumerate() {
        check(&req.cancel)?;
        let mut rng = NativeRng::keyed(req.config.seed, "lora-gaussian-init", index as u64);
        let elements = (req.config.rank as usize)
            .checked_mul(target.input)
            .ok_or_else(|| invalid("LoRA initialization size overflow"))?;
        let values = (0..elements)
            .map(|_| rng.normal() / req.config.rank as f32)
            .collect::<Vec<_>>();
        initial.insert(
            target.native.clone(),
            Tensor::from_vec(
                values,
                (req.config.rank as usize, target.input),
                &Device::Cpu,
            )?
            .to_device(&device)?,
        );
    }
    let graph = TrainingGraph::new(
        dit,
        req.config.rank as usize,
        req.config.alpha as f64,
        &req.config.lora_target_modules,
        initial,
    )?;
    let optimizer = TrainOptimizer::from_config(
        "adamw",
        graph.vars(),
        req.config.learning_rate,
        req.config.weight_decay,
    )?;
    let mut session = CandleSession::new(
        graph,
        optimizer,
        examples,
        opts,
        identity,
        req.config.seed,
        req.config.gradient_accumulation,
    )?;
    let result = controller::run(&mut session, req, &plan, on_progress).map_err(Error::from);
    let sync = device.synchronize();
    drop(session);
    match result {
        Err(error) => Err(error),
        Ok(output) => {
            sync?;
            check(&req.cancel)?;
            Ok(output)
        }
    }
}
