//! Native MLX **Iris-3B generation training** (story sc-25685): full training from random init or
//! from weights, and LoRA / LoKr adapter training, on the frozen upstream objective —
//! `iris3b/train/trainer.py` single-device semantics over the backend-neutral contract
//! [`gen_core::iris::train`].
//!
//! Per micro-batch (positional draws keyed on `mix_seed(seed, 0, epoch, position)`): select each
//! sample's caption, encode it with the frozen Qwen3-VL tower (on the fly, or from the bit-identical
//! memo), substitute the CFG null for dropped rows, sample logit-normal / uniform timestep
//! indices and the noise, take the rectified-flow loss (v- or x-prediction) and its gradient of
//! `loss / grad_accum`. Every `grad_accum` micro-batches (counted globally, as accelerate does):
//! clip the global gradient norm, update the EMA from the **pre-step** weights (upstream's order),
//! step AdamW or the Muon/AdamW hybrid at `lr · LambdaLR(step)`, advance the scheduler.
//! Non-finite losses drop the window's gradients (`nan_loss_tolerance`).
//!
//! Durability: checkpoints are published atomically (`gen_core::iris::train::publish_checkpoint`)
//! with the trainable tensors, the EMA, the optimizer slots and the scheduler/data/RNG position, so
//! cancel + resume reproduces the uninterrupted run exactly; retention keeps the newest
//! `keep_last_checkpoints` plus milestones. The run ends by exporting a full model directory or an
//! adapter file (schemas in `gen_core::iris::train`).

pub mod model;
pub mod optim;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use gen_core::iris::train::{
    self as contract, adapter_param_route, backbone_identity, batch_seed, caption_seed,
    check_resume, checkpoint_root, checkpoint_staging_dir, dataset_fingerprint, draw_batch,
    export_config_yaml, file_stem, full_model_metadata, full_param_route, latest_checkpoint,
    linear_paths_from_keys, lr_factor, prune_checkpoints, publish_checkpoint,
    read_checkpoint_state, select_caption, AdapterMetadata, ArtifactPlan, CheckpointState,
    DataWalk, HostRng, InitMode, IrisTrainPlan, MixedPrecision, Prediction, ResumeDataPolicy,
    TextConditioningMode, TrainFlowDefaults, TrainSchedule, WeightsSelect, CKPT_EMA,
    CKPT_OPTIMIZER, CKPT_STATE, CKPT_TRAINABLE, TECHNIQUES, TRAINER_ID,
};
use gen_core::iris::{IrisConfig, TextEncoderConfig, FAMILY, TEXT_ENCODER_COMPONENT};
use gen_core::train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques, Trainer,
    TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, LoadSpec, Modality, Result, WeightsSource};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::dit::{IrisDiT, TextBatch};
use crate::solver::{cfg_combine, sample};
use crate::text_encoder::IrisTextEncoder;
use model::{
    adapter_tensors, init_adapter, loss_and_grads, provider_dtype, random_init,
    save_safetensors_atomic, AdapterKind, AdapterTarget, StepBatch, TrainModel, Trainable,
};
use optim::{clip_grads, ema_update, IrisOptimizer, Params};

/// Identity + capabilities of the MLX Iris generation trainer.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: true,
        max_reference_images: 0,
        techniques: TECHNIQUES,
    }
}

/// The resolved resources of a training load: the backbone directory (its `config.yaml` always;
/// `model.safetensors` unless the run initialises randomly) and the Qwen3-VL snapshot.
pub struct IrisTrainer {
    descriptor: TrainerDescriptor,
    backbone_dir: PathBuf,
    text_encoder_dir: PathBuf,
    config: IrisConfig,
    flow_defaults: TrainFlowDefaults,
}

/// Construct the trainer from `spec.weights` (the backbone directory) and
/// `spec.components["text_encoder"]`. Nothing heavy loads here; `train` loads per request.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    refuse_trainer_load_overlays(TRAINER_ID, spec)?;
    for (field, set) in [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("pid", spec.pid.is_some()),
    ] {
        if set {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: LoadSpec::{field} is not supported (the text encoder is \
                 the '{TEXT_ENCODER_COMPONENT}' component)"
            )));
        }
    }
    gen_core::control::reject_unknown_components(spec, &[TEXT_ENCODER_COMPONENT], TRAINER_ID)?;
    let backbone_dir = match &spec.weights {
        WeightsSource::Dir(d) => d.clone(),
        WeightsSource::File(f) => {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the backbone resource must be a directory with config.yaml \
                 (+ model.safetensors), not the file {}",
                f.display()
            )))
        }
    };
    let text_encoder_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(d)) => d.clone(),
        _ => {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the generation task trains on frozen Qwen3-VL conditioning \
                 — stage the text encoder snapshot directory as the '{TEXT_ENCODER_COMPONENT}' \
                 component"
            )))
        }
    };
    let config = IrisConfig::from_dir(&backbone_dir)?;
    config.validate_supported()?;
    let flow_defaults = TrainFlowDefaults::from_dir(&backbone_dir)?;
    for name in gen_core::iris::TEXT_ENCODER_REQUIRED_FILES {
        let p = text_encoder_dir.join(name);
        if !p.is_file() {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the text encoder resource is incomplete — {} is missing",
                p.display()
            )));
        }
    }
    Ok(Box::new(IrisTrainer {
        descriptor: trainer_descriptor(),
        backbone_dir,
        text_encoder_dir,
        config,
        flow_defaults,
    }))
}

mlx_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

impl IrisTrainer {
    fn floors(&self, req: &TrainingRequest) -> Result<()> {
        let d = &self.descriptor;
        validate_training_techniques(d, req)?;
        validate_control_request(d, req)?;
        validate_full_finetune_request(d, req)?;
        validate_edit_request(d, req)?;
        refuse_reference_control_model_options(TRAINER_ID, req)?;
        if req.items.is_empty() {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the dataset is empty"
            )));
        }
        Ok(())
    }

    fn resolve_plan(
        &self,
        req: &TrainingRequest,
        linear_paths: &[String],
    ) -> Result<IrisTrainPlan> {
        let plan = IrisTrainPlan::resolve(req, &self.config, &self.flow_defaults, linear_paths)?;
        if plan.init == InitMode::Weights {
            let w = self
                .backbone_dir
                .join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
            if !w.is_file() {
                return Err(Error::Msg(format!(
                    "{TRAINER_ID} trainer: weights init needs {} (or init: random for a full run)",
                    w.display()
                )));
            }
        }
        Ok(plan)
    }
}

impl Trainer for IrisTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.floors(req)?;
        self.resolve_plan(req, &[])?;
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.floors(req)?;
        let run = Run::prepare(self, req, on_progress)?;
        run.execute(req, on_progress).map_err(Into::into)
    }
}

// =============================================================================================
// Text conditioning
// =============================================================================================

/// One caption's conditioning, stored at the tower's own precision (bf16 values, lossless).
#[derive(Clone)]
struct Encoded {
    states: Array,
    mask: Vec<i32>,
}

/// The frozen text conditioning of a run: the tower (on the fly) or the memo (cached).
struct TextSource {
    encoder: Option<IrisTextEncoder>,
    memo: HashMap<String, Encoded>,
    policy: String,
    max_length: usize,
}

impl TextSource {
    fn encode(&mut self, caption: &str) -> Result<Encoded> {
        if let Some(e) = self.memo.get(caption) {
            return Ok(e.clone());
        }
        let te = self.encoder.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "iris training: caption {caption:?} is not in the conditioning memo"
            ))
        })?;
        let window = te.window(caption)?;
        if window.truncated_tokens > 0 {
            match self.policy.as_str() {
                "error" => {
                    return Err(Error::Msg(format!(
                        "iris training: caption overflow ({} tokens over the {}-token window, \
                         on_caption_overflow = error): {caption:?}",
                        window.truncated_tokens, self.max_length
                    )))
                }
                "warn" => {
                    if let Some(m) = gen_core::iris::caption_overflow_warning(
                        &window,
                        self.max_length,
                        te.suffix_ids().len(),
                    ) {
                        eprintln!("{m}");
                    }
                }
                _ => {}
            }
        }
        let c = te.encode_window(&window)?;
        let e = Encoded {
            states: c.states.as_dtype(Dtype::Bfloat16)?,
            mask: c.mask,
        };
        eval([&e.states])?;
        Ok(e)
    }

    fn remember(&mut self, caption: &str) -> Result<()> {
        if !self.memo.contains_key(caption) {
            let e = self.encode(caption)?;
            self.memo.insert(caption.to_string(), e);
        }
        Ok(())
    }
}

// =============================================================================================
// The run
// =============================================================================================

struct Run {
    plan: IrisTrainPlan,
    config: IrisConfig,
    adaln_zero_init: bool,
    model: TrainModel,
    params: Params,
    ema: Option<Params>,
    opt: IrisOptimizer,
    text: TextSource,
    null: Encoded,
    preview: Vec<(String, Encoded)>,
    preview_negative: Option<Encoded>,
    walk: DataWalk,
    schedule: TrainSchedule,
    total_steps: u64,
    ckpt_root: PathBuf,
    base_identity: String,
    dataset_fingerprint: String,
    // position / counters
    step: u64,
    sched_step: u64,
    lr: f64,
    nan_count: u32,
    epoch: u32,
    pos: usize,
    last_loss: f64,
}

fn rc_map(m: HashMap<String, Array>) -> Params {
    m.into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect()
}

/// Load a backbone tensor file: f32 masters for a full run, or (`adapter_compute`) each tensor
/// straight to the provider dtype, so an adapter run never holds an f32 copy of the base.
fn load_source(path: &Path, adapter_compute: Option<Dtype>) -> Result<HashMap<String, Array>> {
    let mut w = Weights::from_file(path)
        .map_err(|e| Error::Msg(format!("iris training: load {}: {e}", path.display())))?;
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut out = HashMap::with_capacity(keys.len());
    for k in keys {
        let a = w.remove(&k).expect("listed key");
        let a = match adapter_compute {
            Some(c) => provider_dtype(&k, &a, c)?,
            None => a.as_dtype(Dtype::Float32)?,
        };
        eval([&a])?;
        out.insert(k, a);
    }
    Ok(out)
}

/// Every key of `got` must be exactly the architecture's, with the architecture's shapes.
fn check_backbone_keys(
    cfg: &gen_core::iris::ModelConfig,
    got: &HashMap<String, Array>,
    what: &str,
) -> Result<()> {
    let want = contract::backbone_tensor_shapes(cfg);
    if want.len() != got.len() {
        return Err(Error::Msg(format!(
            "iris training: {what} carries {} tensors, the configured architecture {}",
            got.len(),
            want.len()
        )));
    }
    for (k, shape) in want {
        let a = got
            .get(&k)
            .ok_or_else(|| Error::Msg(format!("iris training: {what} lacks {k}")))?;
        let s: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        if s != shape {
            return Err(Error::Msg(format!(
                "iris training: {what} {k} is {s:?}, the config implies {shape:?}"
            )));
        }
    }
    Ok(())
}

impl Run {
    fn prepare(
        trainer: &IrisTrainer,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let config = trainer.config.clone();
        // The plan without targets first: init mode decides whether weights are read at all.
        let pre = trainer.resolve_plan(req, &[])?;
        let compute = match pre.mixed_precision {
            MixedPrecision::Bf16 => Dtype::Bfloat16,
            MixedPrecision::Fp32 => Dtype::Float32,
        };
        on_progress(TrainingProgress::LoadingModel);
        let weights_path = trainer
            .backbone_dir
            .join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
        let adapter_compute = (!req.config.full_finetune).then_some(compute);
        let source: Option<HashMap<String, Array>> = match &pre.init {
            InitMode::Weights => Some(load_source(&weights_path, adapter_compute)?),
            InitMode::Random => None,
            InitMode::LoadFrom(p) => {
                let file = if p.join(CKPT_STATE).is_file() {
                    p.join(CKPT_TRAINABLE)
                } else if p.is_dir() {
                    p.join(gen_core::iris::BACKBONE_WEIGHTS_FILE)
                } else {
                    p.clone()
                };
                Some(load_source(&file, adapter_compute)?)
            }
        };
        if let Some(src) = &source {
            check_backbone_keys(&config.model, src, "the source weights")?;
        }
        let shapes = contract::backbone_tensor_shapes(&config.model);
        let linear_paths =
            linear_paths_from_keys(shapes.iter().map(|(k, s)| (k.as_str(), s.len())));
        let plan = trainer.resolve_plan(req, &linear_paths)?;
        let base_identity = backbone_identity(&trainer.backbone_dir, source.is_some())?;
        let adaln_zero_init = trainer.flow_defaults.adaln_zero_init;

        let (model, params) = match &plan.artifact {
            ArtifactPlan::Full => {
                let params = match source {
                    Some(src) => rc_map(src),
                    None => random_init(&config.model, plan.seed, adaln_zero_init)?,
                };
                (
                    TrainModel {
                        cfg: config.model.clone(),
                        compute,
                        trainable: Trainable::Full,
                    },
                    params,
                )
            }
            art => {
                let src = source.ok_or_else(|| {
                    Error::Msg("iris training: adapter runs need backbone weights".into())
                })?;
                let (kind, targets_paths) = match art {
                    ArtifactPlan::Lora {
                        rank,
                        alpha,
                        targets,
                    } => (
                        AdapterKind::Lora {
                            rank: *rank,
                            alpha: *alpha,
                        },
                        targets,
                    ),
                    ArtifactPlan::Lokr {
                        rank,
                        alpha,
                        decompose_factor,
                        targets,
                    } => (
                        AdapterKind::Lokr {
                            rank: *rank,
                            alpha: *alpha,
                            factor: *decompose_factor,
                        },
                        targets,
                    ),
                    ArtifactPlan::Full => unreachable!(),
                };
                let mut targets = Vec::with_capacity(targets_paths.len());
                for p in targets_paths {
                    let w = &src[&format!("{p}.weight")];
                    targets.push(AdapterTarget {
                        path: p.clone(),
                        out_f: w.shape()[0],
                        in_f: w.shape()[1],
                    });
                }
                let mut base = HashMap::with_capacity(src.len());
                for (k, a) in src {
                    let a = provider_dtype(&k, &a, compute)?;
                    eval([&a])?;
                    base.insert(k, a);
                }
                let params = init_adapter(kind, &targets, plan.seed)?;
                (
                    TrainModel {
                        cfg: config.model.clone(),
                        compute,
                        trainable: Trainable::Adapter {
                            kind,
                            targets,
                            base,
                        },
                    },
                    params,
                )
            }
        };

        let cfg_model = config.model.clone();
        let kind = plan.optimizer.kind;
        let adapter_targets: HashMap<String, String> = match &model.trainable {
            Trainable::Adapter {
                kind: ak, targets, ..
            } => targets
                .iter()
                .flat_map(|t| t.factor_keys(*ak).into_iter().map(|k| (k, t.path.clone())))
                .collect(),
            Trainable::Full => HashMap::new(),
        };
        let mut opt = IrisOptimizer::new(&plan.optimizer, &params, |k, nd| match adapter_targets
            .get(k)
        {
            Some(target) => adapter_param_route(target, nd, kind),
            None => full_param_route(k, nd, kind, &cfg_model),
        })?;
        let mut ema = plan.ema.map(|_| params.clone());

        // ---- text conditioning -------------------------------------------------------------
        let mut te_cfg: TextEncoderConfig = config.text_encoder.clone();
        te_cfg.on_caption_overflow = plan.on_caption_overflow.clone();
        let encoder = IrisTextEncoder::load(&trainer.text_encoder_dir, &te_cfg)?;
        let mut text = TextSource {
            encoder: Some(encoder),
            memo: HashMap::new(),
            policy: plan.on_caption_overflow.clone(),
            max_length: config.text_encoder.max_length,
        };
        let null = text.encode("")?;
        let mut preview = Vec::new();
        let mut preview_negative = None;
        if plan.preview.every > 0 {
            for p in &plan.preview.prompts {
                preview.push((p.clone(), text.encode(p)?));
            }
            if plan.preview.cfg_scale != 1.0 {
                preview_negative = Some(text.encode(&plan.preview.negative_prompt)?);
            }
        }
        // The dataset pass (one `Caching` event per item): every image must decode as an image,
        // every caption variant must be selectable, and — with `text_conditioning: cached` — every
        // distinct caption is encoded once into the memo (an `on_caption_overflow: error` caption
        // fails here, before any step, instead of at its first batch).
        let total = req.items.len() as u32;
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            image::image_dimensions(&item.image_path).map_err(|e| {
                Error::Msg(format!(
                    "iris training: item {} is not a readable image: {e}",
                    item.image_path.display()
                ))
            })?;
            let variants =
                contract::item_caption_variants(item, &plan.caption_field, &plan.caption_fields);
            if variants.is_empty() {
                return Err(Error::Msg(format!(
                    "iris training: item {} carries no caption field {:?}",
                    item.image_path.display(),
                    plan.caption_field
                )));
            }
            if plan.text_conditioning == TextConditioningMode::Cached {
                for c in &variants {
                    text.remember(c)?;
                }
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
        }
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if plan.text_conditioning == TextConditioningMode::Cached {
            text.encoder = None;
            mlx_rs::memory::clear_cache();
        }

        let walk = DataWalk::new(req.items.len(), plan.batch_size)?;
        let total_steps = contract::total_optimizer_steps(
            walk.batches_per_epoch(),
            plan.grad_accum,
            plan.num_epochs,
            plan.max_steps,
        );
        let schedule = TrainSchedule::new(plan.flow.num_train_timesteps, plan.flow.shift);
        let ckpt_root = checkpoint_root(&req.output_dir, &req.file_name);
        let dataset_fingerprint = dataset_fingerprint(req)?;

        let mut run_params = params;
        let (mut step, mut sched_step, mut lr, mut nan_count, mut epoch, mut pos, mut last_loss) =
            (0u64, 0u64, plan.optimizer.lr, 0u32, 1u32, 0usize, f64::NAN);
        let resume_dir = match (&plan.resume_from, plan.resume) {
            (Some(p), _) => {
                let dir = if p.join(CKPT_STATE).is_file() {
                    p.clone()
                } else {
                    latest_checkpoint(p).ok_or_else(|| {
                        Error::Msg(format!(
                            "iris resume: {} holds no complete checkpoint",
                            p.display()
                        ))
                    })?
                };
                Some(dir)
            }
            (None, true) => latest_checkpoint(&ckpt_root),
            (None, false) => None,
        };
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let restored = rc_map(load_tensors(&dir.join(CKPT_TRAINABLE))?);
            if restored.len() != run_params.len()
                || run_params.keys().any(|k| !restored.contains_key(k))
            {
                return Err(Error::Msg(
                    "iris resume: the checkpoint's trainable tensors do not match this run".into(),
                ));
            }
            run_params = restored;
            if let Some(e) = ema.as_mut() {
                *e = rc_map(load_tensors(&dir.join(CKPT_EMA))?);
            }
            opt.load_state(&load_tensors(&dir.join(CKPT_OPTIMIZER))?)?;
            step = saved.step;
            sched_step = saved.scheduler_step;
            lr = if plan.override_lr_on_resume {
                plan.optimizer.lr
            } else {
                saved.lr
            };
            nan_count = saved.nan_count;
            last_loss = saved.last_loss;
            if plan.resume_data_policy == ResumeDataPolicy::Exact {
                epoch = saved.epoch;
                pos = saved.batches_consumed;
            }
        }
        Ok(Self {
            plan,
            config,
            adaln_zero_init,
            model,
            params: run_params,
            ema,
            opt,
            text,
            null,
            preview,
            preview_negative,
            walk,
            schedule,
            total_steps,
            ckpt_root,
            base_identity,
            dataset_fingerprint,
            step,
            sched_step,
            lr,
            nan_count,
            epoch,
            pos,
            last_loss,
        })
    }

    /// Assemble micro-batch `pos` of `epoch`.
    fn batch(&mut self, req: &TrainingRequest, epoch: u32, pos: usize) -> Result<StepBatch> {
        let idx = self.walk.batch(pos);
        let b = idx.len();
        let size = self.plan.image_size;
        let numel = 3 * size * size;
        let draws = draw_batch(
            batch_seed(self.plan.seed, epoch, pos),
            b,
            numel,
            self.plan.text_dropout,
            self.plan.flow.sampler,
            self.plan.flow.num_train_timesteps,
        );
        let mut pixels = Vec::with_capacity(b * numel);
        let mut states = Vec::with_capacity(b);
        let mut masks = Vec::with_capacity(b);
        for (row, i) in idx.enumerate() {
            let item = &req.items[i];
            let img = contract::preprocess_image(&item.image_path, size)?;
            pixels.extend_from_slice(&img.chw);
            let mut rng = HostRng::new(caption_seed(self.plan.seed, epoch, i));
            let caption = select_caption(
                item,
                &self.plan.caption_field,
                &self.plan.caption_fields,
                &mut rng,
            )?;
            let enc = if draws.drop[row] {
                // The dropped row still encodes upstream (skip_dropped_text is a perf flag); the
                // encoder consumes no randomness, so substituting the null here is identical.
                self.null.clone()
            } else {
                self.text.encode(&caption)?
            };
            states.push(enc.states.as_dtype(Dtype::Float32)?);
            masks.push(enc.mask);
        }
        let shape = [b as i32, 3, size as i32, size as i32];
        let sigmas: Vec<f32> = draws
            .timestep_idx
            .iter()
            .map(|&t| self.schedule.sigmas[t])
            .collect();
        let times: Vec<f32> = draws
            .timestep_idx
            .iter()
            .map(|&t| self.schedule.model_times[t])
            .collect();
        Ok(StepBatch {
            x0: Array::from_slice(&pixels, &shape),
            noise: Array::from_slice(&draws.noise, &shape),
            sigma: Array::from_slice(&sigmas, &[b as i32, 1, 1, 1]),
            t: Array::from_slice(&times, &[b as i32]),
            states: concatenate_axis(&states, 0)?,
            masks,
        })
    }

    fn execute(
        mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let plan = self.plan.clone();
        let accum = plan.grad_accum;
        let bpe = self.walk.batches_per_epoch();
        let max_epoch = plan.num_epochs.unwrap_or(u32::MAX);
        let first_step = self.step;
        let mut last_saved = if self.step > 0 { Some(self.step) } else { None };
        let mut window = Window::new(accum);
        let mut micro: u64 = 0;
        let mut boundary = (self.epoch, self.pos);
        let mut cancelled = false;
        let mut finished = self.step >= plan.max_steps as u64;
        let mut epoch = self.epoch;
        let mut start_pos = self.pos;
        while !finished && !cancelled && epoch <= max_epoch {
            for pos in start_pos..bpe {
                if req.cancel.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let batch = self.batch(req, epoch, pos)?;
                let samples = batch.masks.len();
                micro += 1;
                let sync = micro.is_multiple_of(accum as u64);
                let loss = window.micro(&self.model, &self.params, &batch, &plan.flow, samples)?;
                drop(batch);
                if !loss.is_finite() {
                    self.nan_count += 1;
                    eprintln!(
                        "iris training: non-finite loss at step {} ({}/{})",
                        self.step, self.nan_count, plan.nan_loss_tolerance
                    );
                    if self.nan_count > plan.nan_loss_tolerance {
                        return Err(Error::Msg(format!(
                            "iris training: loss was non-finite {} times; aborting",
                            self.nan_count
                        )));
                    }
                    window.discard();
                    continue;
                }
                if !sync {
                    continue;
                }
                let lr_now = self.lr
                    * lr_factor(
                        plan.schedule,
                        self.sched_step,
                        plan.warmup_steps,
                        self.total_steps,
                    );
                let mean_loss = window.mean_loss();
                let ema = match (self.ema.as_mut(), plan.ema) {
                    (Some(e), Some(d)) => Some((e, d)),
                    _ => None,
                };
                let Some(_norm) = window.update(
                    &mut self.params,
                    ema,
                    &mut self.opt,
                    plan.gradient_clip,
                    lr_now,
                )?
                else {
                    continue;
                };
                self.sched_step += 1;
                self.step += 1;
                self.last_loss = mean_loss;
                boundary = (epoch, pos + 1);
                on_progress(TrainingProgress::Training {
                    step: self.step as u32,
                    total: self.total_steps as u32,
                    loss: self.last_loss as f32,
                });
                if plan.save_every > 0 && self.step.is_multiple_of(plan.save_every as u64) {
                    self.save_checkpoint(boundary)?;
                    last_saved = Some(self.step);
                    on_progress(TrainingProgress::Checkpoint {
                        step: self.step as u32,
                    });
                }
                if plan.preview.every > 0
                    && (self.step.is_multiple_of(plan.preview.every as u64) || self.step == 1)
                {
                    self.render_previews(req, on_progress)?;
                }
                if self.step >= plan.max_steps as u64 {
                    finished = true;
                    break;
                }
            }
            if finished || cancelled {
                break;
            }
            epoch += 1;
            start_pos = 0;
        }
        if cancelled && self.step == first_step {
            // No optimizer step ran in this invocation: the typed cancellation, nothing written.
            return Err(Error::Canceled);
        }
        if last_saved != Some(self.step) && self.step > 0 {
            self.save_checkpoint(boundary)?;
            on_progress(TrainingProgress::Checkpoint {
                step: self.step as u32,
            });
        }
        on_progress(TrainingProgress::Saving);
        let path = self.export(req)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: self.step as u32,
            final_loss: self.last_loss as f32,
        })
    }

    fn state(&self, boundary: (u32, usize)) -> CheckpointState {
        CheckpointState {
            step: self.step,
            epoch: boundary.0,
            batches_consumed: boundary.1,
            scheduler_step: self.sched_step,
            lr: self.lr,
            nan_count: self.nan_count,
            last_loss: if self.last_loss.is_finite() {
                self.last_loss
            } else {
                0.0
            },
            artifact: self.plan.artifact.kind().into(),
            optimizer: self.plan.optimizer.kind.as_str().into(),
            ema_decay: self.plan.ema,
            seed: self.plan.seed,
            backend: "mlx".into(),
            state_identity: self.plan.state_identity(),
            data_identity: self.plan.data_identity(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            base_identity: self.base_identity.clone(),
        }
    }

    /// Publish a checkpoint of the state at the last optimizer-step boundary.
    fn save_checkpoint(&self, boundary: (u32, usize)) -> Result<()> {
        std::fs::create_dir_all(&self.ckpt_root).map_err(|e| {
            Error::Msg(format!(
                "iris checkpoint: create {}: {e}",
                self.ckpt_root.display()
            ))
        })?;
        let staging = checkpoint_staging_dir(&self.ckpt_root, self.step);
        if staging.exists() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        std::fs::create_dir_all(&staging).map_err(|e| {
            Error::Msg(format!(
                "iris checkpoint: create {}: {e}",
                staging.display()
            ))
        })?;
        save_params(&staging.join(CKPT_TRAINABLE), &self.params)?;
        if let Some(ema) = &self.ema {
            save_params(&staging.join(CKPT_EMA), ema)?;
        }
        self.opt.save(&staging.join(CKPT_OPTIMIZER))?;
        contract::atomic_write(
            &staging.join(CKPT_STATE),
            serde_json::to_string_pretty(&self.state(boundary).to_json())
                .map_err(|e| Error::Msg(e.to_string()))?
                .as_bytes(),
        )?;
        publish_checkpoint(&self.ckpt_root, &staging, self.step)?;
        prune_checkpoints(
            &self.ckpt_root,
            self.plan.keep_last_checkpoints,
            &self.plan.milestone_steps,
        )?;
        Ok(())
    }

    fn selected(&self, which: WeightsSelect) -> &Params {
        match (which, &self.ema) {
            (WeightsSelect::Ema, Some(e)) => e,
            _ => &self.params,
        }
    }

    /// Write the final artifact (full model directory or adapter file); returns its tensor file.
    fn export(&self, req: &TrainingRequest) -> Result<PathBuf> {
        std::fs::create_dir_all(&req.output_dir)
            .map_err(|e| Error::Msg(format!("iris export: {}: {e}", req.output_dir.display())))?;
        let which = self.plan.export_weights;
        let tensors = self.selected(which);
        match &self.model.trainable {
            Trainable::Full => {
                let dir = req.output_dir.join(file_stem(&req.file_name));
                std::fs::create_dir_all(&dir)
                    .map_err(|e| Error::Msg(format!("iris export: {}: {e}", dir.display())))?;
                let dtype = if self.plan.export_bf16 {
                    Dtype::Bfloat16
                } else {
                    Dtype::Float32
                };
                let mut list: Vec<(String, Array)> = Vec::with_capacity(tensors.len());
                for (k, v) in tensors {
                    list.push((k.to_string(), v.as_dtype(dtype)?));
                }
                list.sort_by(|a, b| a.0.cmp(&b.0));
                contract::atomic_write(
                    &dir.join(gen_core::iris::BACKBONE_CONFIG_FILE),
                    export_config_yaml(&self.config, &self.plan.flow, self.adaln_zero_init)
                        .as_bytes(),
                )?;
                let path = dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
                save_safetensors_atomic(
                    &path,
                    &list,
                    &full_model_metadata(which, self.step, &self.base_identity),
                )?;
                Ok(path)
            }
            Trainable::Adapter { kind, targets, .. } => {
                let path = req.output_dir.join(&req.file_name);
                let meta = AdapterMetadata {
                    network_type: match kind {
                        AdapterKind::Lora { .. } => "lora".into(),
                        AdapterKind::Lokr { .. } => "lokr".into(),
                    },
                    rank: kind.rank(),
                    alpha: kind.alpha(),
                    decompose_factor: match kind {
                        AdapterKind::Lokr { factor, .. } => Some(*factor),
                        AdapterKind::Lora { .. } => None,
                    },
                    weights: which,
                    steps: self.step,
                    base_identity: self.base_identity.clone(),
                    prediction: self.plan.flow.prediction,
                    shift: self.plan.flow.shift,
                    targets: targets.iter().map(|t| t.path.clone()).collect(),
                };
                let mut map = meta.to_map();
                if !req.trigger_words.is_empty() {
                    map.insert(
                        "triggerWords".into(),
                        serde_json::Value::from(req.trigger_words.clone()).to_string(),
                    );
                }
                save_safetensors_atomic(&path, &adapter_tensors(*kind, targets, tensors)?, &map)?;
                Ok(path)
            }
        }
    }

    /// Render the preview prompts from the in-progress state (raw or EMA weights).
    fn render_previews(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pv = &self.plan.preview;
        let dit = self.model.dit(self.selected(pv.weights))?;
        let total = self.preview.len() as u32;
        for (i, (prompt, enc)) in self.preview.iter().enumerate() {
            if req.cancel.is_cancelled() {
                return Ok(());
            }
            let image = render_preview(
                &dit,
                &self.plan,
                enc,
                self.preview_negative.as_ref(),
                pv.seed.wrapping_add(i as u64),
            )?;
            on_progress(TrainingProgress::Sample {
                step: self.step as u32,
                index: i as u32 + 1,
                total,
                prompt: prompt.clone(),
                image,
            });
        }
        mlx_rs::memory::clear_cache();
        Ok(())
    }
}

/// One gradient-accumulation window of `Trainer.run` (accelerate's `accumulate`): each micro-batch
/// contributes the gradient of `loss / grad_accum`; closing the window clips the global gradient
/// norm, updates the EMA from the **pre-step** weights, then steps the optimizer — upstream's
/// order. The training loop and the parity tests drive this one implementation.
pub struct Window {
    accum: usize,
    grads: Option<Params>,
    loss_sum: f64,
    samples: usize,
}

impl Window {
    pub fn new(accum: usize) -> Self {
        Self {
            accum: accum.max(1),
            grads: None,
            loss_sum: 0.0,
            samples: 0,
        }
    }

    /// One micro-batch of `samples` rows: returns its (unscaled) loss. A non-finite loss leaves the
    /// window untouched; the caller decides (upstream drops the window's gradients).
    pub fn micro(
        &mut self,
        model: &TrainModel,
        params: &Params,
        batch: &StepBatch,
        obj: &contract::FlowObjective,
        samples: usize,
    ) -> Result<f32> {
        let (loss, grads) = loss_and_grads(model, params, batch, obj, 1.0 / self.accum as f32)?;
        if !loss.is_finite() {
            return Ok(loss);
        }
        self.grads = Some(match self.grads.take() {
            None => grads,
            Some(mut acc) => {
                for (k, g) in grads {
                    let s = acc[&k].add(&g)?;
                    acc.insert(k, s);
                }
                acc
            }
        });
        self.loss_sum += loss as f64 * samples as f64;
        self.samples += samples;
        Ok(loss)
    }

    /// `optimizer.zero_grad()` + clearing the loss log (the non-finite path).
    pub fn discard(&mut self) {
        self.grads = None;
        self.loss_sum = 0.0;
        self.samples = 0;
    }

    /// The sample-weighted mean loss of the window so far.
    pub fn mean_loss(&self) -> f64 {
        self.loss_sum / self.samples.max(1) as f64
    }

    /// Close the window at learning rate `lr`: clip → EMA → step. Returns the pre-clip global
    /// gradient norm, or `None` when the window holds no gradient (nothing steps).
    pub fn update(
        &mut self,
        params: &mut Params,
        ema: Option<(&mut Params, f64)>,
        opt: &mut IrisOptimizer,
        clip: f64,
        lr: f64,
    ) -> Result<Option<f64>> {
        let Some(grads) = self.grads.take() else {
            self.discard();
            return Ok(None);
        };
        self.discard();
        let grads: Params = grads
            .into_iter()
            .filter(|(k, _)| opt.is_trained(k))
            .collect();
        let (norm, grads) = clip_grads(grads, clip)?;
        if let Some((ema, decay)) = ema {
            ema_update(ema, params, decay)?;
        }
        opt.step(params, &grads, lr)?;
        Ok(Some(norm))
    }
}

fn save_params(path: &Path, params: &Params) -> Result<()> {
    let mut list: Vec<(&str, &Array)> = params.iter().map(|(k, v)| (k.as_ref(), v)).collect();
    list.sort_by(|a, b| a.0.cmp(b.0));
    Array::save_safetensors(list, None::<&HashMap<String, String>>, path)?;
    Ok(())
}

fn load_tensors(path: &Path) -> Result<HashMap<String, Array>> {
    Array::load_safetensors(path)
        .map_err(|e| Error::Msg(format!("iris checkpoint: read {}: {e}", path.display())))
}

/// One preview image: the release sampler (FlowDPM-Solver++ order 2, the run's resolved shift,
/// CFG against the negative prompt) from `seed`'s noise, through the provider's prediction-aware
/// solver (`x0 = x − s·v` for v, `x0 = out` for x — upstream's `_pred_x0`).
#[allow(clippy::too_many_arguments)]
pub fn render_preview_with(
    dit: &IrisDiT,
    steps: usize,
    cfg_scale: f32,
    shift: f64,
    num_timesteps: usize,
    prediction: Prediction,
    size: usize,
    cond: (&Array, &[i32]),
    uncond: Option<(&Array, &[i32])>,
    seed: u64,
) -> Result<mlx_gen::Image> {
    let plan = gen_core::iris::dpm_solver_plan(steps, gen_core::iris::DEFAULT_SOLVER_ORDER, shift)?;
    let channels = dit.config().in_channels;
    let z = crate::pipeline::noise(seed, channels, size as u32, size as u32)?;
    let cond_mask = vec![cond.1.to_vec()];
    let cond_batch = TextBatch {
        states: cond.0,
        mask: &cond_mask,
    };
    let cfg_states;
    let cfg_mask;
    let cfg_batch = match uncond {
        Some((u, um)) if cfg_scale != 1.0 => {
            cfg_states = concatenate_axis(&[u, cond.0], 0)?;
            cfg_mask = vec![um.to_vec(), cond.1.to_vec()];
            Some(TextBatch {
                states: &cfg_states,
                mask: &cfg_mask,
            })
        }
        _ => None,
    };
    let never = mlx_gen::CancelFlag::default();
    let x = sample(
        &z,
        &plan,
        prediction,
        &never,
        |x, step| {
            let t = step.model_time(num_timesteps);
            match &cfg_batch {
                Some(batch)
                    if gen_core::iris::cfg_active(
                        cfg_scale,
                        step.s,
                        gen_core::iris::DEFAULT_CFG_INTERVAL,
                    ) =>
                {
                    let xb = concatenate_axis(&[x, x], 0)?;
                    let tb = Array::from_slice(&[t, t], &[2]);
                    let out = dit.forward(&xb, &tb, batch)?;
                    let halves = out.split(2, 0)?;
                    cfg_combine(&halves[0], &halves[1], cfg_scale)
                }
                _ => dit.forward(x, &Array::from_slice(&[t], &[1]), &cond_batch),
            }
        },
        |_, _| {},
    )?;
    let x = mlx_rs::ops::clip(&x, (-1.0f32, 1.0f32))?;
    crate::pipeline::to_image(&x)
}

fn render_preview(
    dit: &IrisDiT,
    plan: &IrisTrainPlan,
    cond: &Encoded,
    negative: Option<&Encoded>,
    seed: u64,
) -> Result<mlx_gen::Image> {
    let pv = &plan.preview;
    let c = cond.states.as_dtype(Dtype::Float32)?;
    let u = match negative {
        Some(n) => Some(n.states.as_dtype(Dtype::Float32)?),
        None => None,
    };
    render_preview_with(
        dit,
        pv.steps,
        pv.cfg_scale,
        plan.flow.shift,
        plan.flow.num_train_timesteps,
        plan.flow.prediction,
        pv.size,
        (&c, &cond.mask),
        u.as_ref()
            .zip(negative)
            .map(|(s, n)| (s, n.mask.as_slice())),
        seed,
    )
}
