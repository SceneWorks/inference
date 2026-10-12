//! Native MLX **Iris-3B generation training** (stories sc-25685, sc-25689): full training from
//! random init or from weights, and LoRA / LoKr adapter training, on the frozen upstream objective —
//! `iris3b/train/trainer.py` semantics over the backend-neutral contract [`gen_core::iris::train`],
//! single-device or data-parallel.
//!
//! Per micro-batch (positional draws keyed on `mix_seed(seed, rank, epoch, position)`): walk this
//! rank's slice of the data plan (the fixed-square ranged walk, or shape-homogeneous `bucket` /
//! `area` batches), select each sample's caption, encode it with the frozen Qwen3-VL tower (on the
//! fly, or from the bit-identical memo), substitute the CFG null for dropped rows, sample
//! logit-normal / uniform timestep indices and the noise, take the rectified-flow loss (v- or
//! x-prediction) — plus, under `repa.weight > 0`, the REPA / iREPA alignment of block
//! `repa_layer`'s tokens to the frozen DINOv2 teacher — and the gradient of
//! `total / grad_accum` (through MLX's checkpoint transform for the activation-checkpointing
//! regions). Every `grad_accum` micro-batches (counted globally, as accelerate does): reduce the
//! gradients across ranks (DDP mean, or FSDP reduce-scatter to this rank's rows), clip the global
//! gradient norm, update the EMA from the **pre-step** weights (upstream's order), step AdamW or the
//! Muon/AdamW hybrid at `lr · LambdaLR(step)`, advance the scheduler. Non-finite steps drop the
//! window (`nan_loss_tolerance`), collectively.
//!
//! Durability: checkpoints are published atomically (`gen_core::iris::train::publish_checkpoint`)
//! with the trainable tensors (consolidated, or per-rank shards under `checkpoint_io: dcp`), the
//! EMA, the optimizer slots, the REPA projector and the scheduler / data-cursor / loss-scaler
//! position, so cancel + resume reproduces the uninterrupted run exactly and a consolidated
//! checkpoint reshards to another rank count; retention keeps the newest `keep_last_checkpoints`
//! plus milestones. Holdout validation appends to `<checkpoints>/metrics.jsonl`. The run ends by
//! exporting a full model directory or an adapter file (schemas in `gen_core::iris::train`).

pub mod critic;
pub mod depth;
pub mod dist;
pub mod lpips;
pub mod model;
pub mod optim;
pub mod restore;
pub mod teacher;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use gen_core::iris::train::dist::{
    dist_error, Collective, DistError, Partition, Precision, ReduceOp, World,
};
use gen_core::iris::train::holdout::{self as holdout_contract, ValSums};
use gen_core::iris::train::lineage::{
    agree_resume, elect_checkpoint_writer, publish_export_dir, retire_foreign_checkpoints,
};
use gen_core::iris::train::pipeline::DataPipeline;
use gen_core::iris::train::repa::{RepaPlan, PROJECTOR_PREFIX, TEACHER_COMPONENT};
use gen_core::iris::train::shapes::preprocess_file_to_shape;
use gen_core::iris::train::{
    self as contract, adapter_param_route, backbone_identity, batch_seed_rank, caption_seed,
    check_resume, checkpoint_root, checkpoint_staging_dir, dataset_fingerprint, draw_batch,
    export_config_yaml, file_stem, full_model_metadata, full_param_route, latest_checkpoint,
    linear_paths_from_keys, prune_checkpoints, publish_checkpoint, read_checkpoint_state,
    select_caption, AdapterMetadata, ArtifactPlan, CheckpointState, GradScaler, HostRng, InitMode,
    IrisTrainPlan, MixedPrecision, ParamRoute, Prediction, ResumeDataPolicy, TextConditioningMode,
    TrainFlowDefaults, TrainSchedule, WeightsSelect, CKPT_EMA, CKPT_OPTIMIZER, CKPT_STATE,
    CKPT_TRAINABLE, TECHNIQUES, TRAINER_ID,
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
use serde_json::{json, Value as JsonValue};

use crate::dit::{IrisDiT, TextBatch};
use crate::solver::{cfg_combine, sample};
use crate::text_encoder::IrisTextEncoder;

/// A scalar's value as f32, converted on the task-local stream first: `Array::item` converts a
/// non-f32 scalar with an `astype` on the **default device** (the GPU on a Mac), whatever the
/// task-local stream says.
pub(crate) fn scalar_f32(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.item::<f32>())
}
use dist::{make_collective, to_host, Topology};
use model::{
    adapter_tensors, flow_loss_per_sample, init_adapter, init_projector, loss_and_grads,
    loss_and_grads_ext, provider_dtype, random_init, save_safetensors_atomic, AdapterKind,
    AdapterTarget, RepaTargets, StepBatch, TrainExtras, TrainModel, Trainable,
};
use optim::{clip_grads, ema_update, IrisOptimizer, MuonShards, Params};
use teacher::{spatial_normalize, Teacher};

/// The REPA projector's checkpoint member.
pub const CKPT_REPA: &str = "repa_projector.safetensors";

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
/// `model.safetensors` unless the run initialises randomly), the Qwen3-VL snapshot, and (for
/// REPA) the frozen teacher resource.
pub struct IrisTrainer {
    descriptor: TrainerDescriptor,
    backbone_dir: PathBuf,
    text_encoder_dir: PathBuf,
    teacher_dir: Option<PathBuf>,
    config: IrisConfig,
    flow_defaults: TrainFlowDefaults,
}

/// Construct the trainer from `spec.weights` (the backbone directory),
/// `spec.components["text_encoder"]` and, for REPA runs, `spec.components["repa_teacher"]`.
/// Nothing heavy loads here; `train` loads per request.
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
    gen_core::control::reject_unknown_components(
        spec,
        &[TEXT_ENCODER_COMPONENT, TEACHER_COMPONENT],
        TRAINER_ID,
    )?;
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
    let teacher_dir = match spec.components.get(TEACHER_COMPONENT) {
        None => None,
        Some(WeightsSource::Dir(d)) => Some(d.clone()),
        Some(WeightsSource::File(f)) => Some(f.clone()),
    };
    gen_core::iris::downstream::refuse_task_export(&backbone_dir, TRAINER_ID)?;
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
        teacher_dir,
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

    /// The plan, at the contract level (typed distributed refusals stay typed).
    fn plan(
        &self,
        req: &TrainingRequest,
        linear_paths: &[String],
    ) -> gen_core::Result<IrisTrainPlan> {
        IrisTrainPlan::resolve(req, &self.config, &self.flow_defaults, linear_paths)
    }

    fn resolve_plan(
        &self,
        req: &TrainingRequest,
        linear_paths: &[String],
    ) -> Result<IrisTrainPlan> {
        let plan = self.plan(req, linear_paths)?;
        self.check_resources(&plan)?;
        Ok(plan)
    }

    /// The resources the plan reads exist (weights for a weights start; the REPA teacher).
    fn check_resources(&self, plan: &IrisTrainPlan) -> Result<()> {
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
        if let InitMode::LoadFrom(p) = &plan.init {
            if p.extension().is_some_and(|e| e == "pth" || e == "pt") {
                return Err(Error::Unsupported(format!(
                    "{TRAINER_ID} trainer: load_from {} is a torch pickle; native training reads \
                     a native checkpoint or an exported model directory (export it with \
                     scripts/export_checkpoint.py)",
                    p.display()
                )));
            }
        }
        if let Some(r) = &plan.repa {
            teacher_path(&self.teacher_dir, r)?;
        }
        Ok(())
    }
}

/// The REPA teacher resource of a run: `repa.teacher_weights` when set, else the
/// `repa_teacher` load component — refused when neither names an existing resource, never fetched.
fn teacher_path(component: &Option<PathBuf>, repa: &RepaPlan) -> Result<PathBuf> {
    let path = match (&repa.teacher_weights, component) {
        (Some(w), _) if w.contains("://") => {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: repa.teacher_weights {w:?} is a URL; the teacher is a \
                 provisioned resource (nothing is downloaded during training)"
            )))
        }
        (Some(w), Some(c)) if Path::new(w) != c.as_path() => {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: repa.teacher_weights {w:?} and the '{TEACHER_COMPONENT}' \
                 component {} name different teachers",
                c.display()
            )))
        }
        (Some(w), _) => PathBuf::from(w),
        (None, Some(c)) => c.clone(),
        (None, None) => {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: repa.weight > 0 needs the frozen {} teacher — stage it as \
                 the '{TEACHER_COMPONENT}' component (facebook/dinov2-base @ {} for the release \
                 teacher dinov2_vitb14)",
                repa.teacher.entrypoint,
                gen_core::iris::train::repa::DINOV2_BASE_HF_REVISION
            )))
        }
    };
    let file = teacher::teacher_file(&path);
    if !file.is_file() {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the REPA teacher resource {} is missing",
            file.display()
        )));
    }
    Ok(path)
}

impl Trainer for IrisTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.floors(req)?;
        let plan = self.plan(req, &[])?;
        self.check_resources(&plan)?;
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.floors(req)?;
        // Typed refusals (invalid mesh, global batch, split world) before anything loads.
        self.plan(req, &[])?;
        let failure: Rc<RefCell<Option<DistError>>> = Rc::new(RefCell::new(None));
        let result = Run::prepare(self, req, on_progress, failure.clone())
            .and_then(|run| run.execute(req, on_progress));
        match result {
            Ok(out) => Ok(out),
            Err(e) => match failure.borrow_mut().take() {
                Some(d) => Err(d.into_error()),
                None => Err(e.into()),
            },
        }
    }
}

/// A [`Collective`] that records the first typed distributed failure it surfaces, so the trainer
/// can return it typed across the backend error boundary.
struct Recording {
    inner: Box<dyn Collective>,
    slot: Rc<RefCell<Option<DistError>>>,
}

impl Recording {
    fn note<T>(&self, r: gen_core::Result<T>) -> gen_core::Result<T> {
        if let Err(e) = &r {
            if let Some(d) = dist_error(e) {
                let mut s = self.slot.borrow_mut();
                if s.is_none() {
                    *s = Some(d.clone());
                }
            }
        }
        r
    }
}

impl Collective for Recording {
    fn world(&self) -> &World {
        self.inner.world()
    }
    fn all_reduce(
        &mut self,
        data: &mut [f32],
        op: ReduceOp,
        p: Precision,
        g: &Partition,
    ) -> gen_core::Result<()> {
        let r = self.inner.all_reduce(data, op, p, g);
        self.note(r)
    }
    fn all_gather(&mut self, data: &[f32], g: &Partition) -> gen_core::Result<Vec<Vec<f32>>> {
        let r = self.inner.all_gather(data, g);
        self.note(r)
    }
    fn broadcast(&mut self, data: &mut Vec<u8>, root: usize) -> gen_core::Result<()> {
        let r = self.inner.broadcast(data, root);
        self.note(r)
    }
    fn barrier(&mut self, tag: &str) -> gen_core::Result<()> {
        let r = self.inner.barrier(tag);
        self.note(r)
    }
    fn all_reduce_f64(&mut self, data: &mut [f64], op: ReduceOp) -> gen_core::Result<()> {
        let r = self.inner.all_reduce_f64(data, op);
        self.note(r)
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
    extras: TrainExtras,
    /// The trained tensors this rank holds: the whole map, or (sharded) its rows — backbone or
    /// adapter factors plus the REPA projector.
    params: Params,
    /// The gathered full parameters of a sharded run, when held.
    full: Option<Params>,
    /// EMA of the core tensors (never the projector), rows under sharding.
    ema: Option<Params>,
    opt: IrisOptimizer,
    text: TextSource,
    null: Encoded,
    preview: Vec<(String, Encoded)>,
    preview_negative: Option<Encoded>,
    pipeline: DataPipeline,
    holdout_sizes: Vec<(usize, usize)>,
    schedule: TrainSchedule,
    total_steps: u64,
    ckpt_root: PathBuf,
    base_identity: String,
    dataset_fingerprint: String,
    teacher: Option<Teacher>,
    topo: Topology,
    scaler: Option<GradScaler>,
    // position / counters
    step: u64,
    sched_step: u64,
    lr: f64,
    nan_count: u32,
    last_loss: f64,
    first_epoch: u32,
    resume_position: Option<JsonValue>,
    /// Whether this rank publishes consolidated checkpoints (the elected writer of its storage,
    /// sc-25690).
    writes_checkpoints: bool,
    /// The logging window (`log_every` steps of global mean losses).
    log_window: Vec<f64>,
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

/// The Muon seam of a sharded step: gather a matrix's rows from the shard group, keep this rank's.
struct ShardMuon<'a> {
    topo: &'a mut Topology,
}

impl MuonShards for ShardMuon<'_> {
    fn gather(&mut self, key: &str, rows: &Array) -> Result<Array> {
        self.topo.gather_full(key, rows)
    }
    fn own_rows(&self, key: &str, full: &Array) -> Result<Array> {
        self.topo.shard_of(key, full)
    }
}

/// The checkpoint members of a run, keyed as written.
struct CkptPayload {
    trainable: HashMap<String, Array>,
    projector: Option<HashMap<String, Array>>,
    ema: Option<HashMap<String, Array>>,
    optimizer: HashMap<String, Array>,
}

/// Read a checkpoint's members: the consolidated files, or (`checkpoint_io: dcp`) every rank
/// shard reassembled into full tensors in shard order.
fn read_checkpoint_members(dir: &Path, saved: &CheckpointState) -> Result<CkptPayload> {
    let shard_files = dcp_shards(dir);
    if shard_files.is_empty() {
        let projector = dir
            .join(CKPT_REPA)
            .is_file()
            .then(|| load_tensors(&dir.join(CKPT_REPA)))
            .transpose()?;
        let ema = dir
            .join(CKPT_EMA)
            .is_file()
            .then(|| load_tensors(&dir.join(CKPT_EMA)))
            .transpose()?;
        return Ok(CkptPayload {
            trainable: load_tensors(&dir.join(CKPT_TRAINABLE))?,
            projector,
            ema,
            optimizer: load_tensors(&dir.join(CKPT_OPTIMIZER))?,
        });
    }
    let n = saved
        .dist
        .as_ref()
        .and_then(|d| d.get("dp_shard"))
        .and_then(JsonValue::as_u64)
        .unwrap_or(shard_files.len() as u64) as usize;
    if shard_files.len() != n {
        return Err(Error::Msg(format!(
            "iris resume: the dcp checkpoint {} holds {} shard files, its topology wrote {n}",
            dir.display(),
            shard_files.len()
        )));
    }
    let parts: Vec<HashMap<String, Array>> = shard_files
        .iter()
        .map(|p| load_tensors(p))
        .collect::<Result<_>>()?;
    let mut merged: HashMap<String, Array> = HashMap::new();
    let keys: std::collections::BTreeSet<&String> = parts.iter().flat_map(|p| p.keys()).collect();
    for key in keys {
        if key == "o::__t" {
            merged.insert(key.clone(), parts[0][key].clone());
            continue;
        }
        // A shard holding zero rows of a tensor (more shards than rows) writes no entry for it;
        // the reassembled shape is checked against the run's when it is restored.
        let nonempty: Vec<Array> = parts
            .iter()
            .filter_map(|p| p.get(key))
            .filter(|a| a.ndim() == 0 || a.shape()[0] > 0)
            .cloned()
            .collect();
        merged.insert(key.clone(), concatenate_axis(&nonempty, 0)?);
    }
    let take = |prefix: &str| -> HashMap<String, Array> {
        merged
            .iter()
            .filter_map(|(k, v)| k.strip_prefix(prefix).map(|s| (s.to_string(), v.clone())))
            .collect()
    };
    let trainable = take("p::");
    let projector: HashMap<String, Array> = trainable
        .iter()
        .filter(|(k, _)| k.starts_with(PROJECTOR_PREFIX))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let ema = take("e::");
    Ok(CkptPayload {
        trainable: trainable
            .into_iter()
            .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
            .collect(),
        projector: (!projector.is_empty()).then_some(projector),
        ema: (!ema.is_empty()).then_some(ema),
        optimizer: take("o::"),
    })
}

fn dcp_shards(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".safetensors"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// A boundary of an ended pass and the data position captured when that pass ended.
struct Settled {
    boundary: (u32, usize, usize, usize),
    position: JsonValue,
}

impl Run {
    fn prepare(
        trainer: &IrisTrainer,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        failure: Rc<RefCell<Option<DistError>>>,
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let mut config = trainer.config.clone();
        // The plan without targets first: init mode decides whether weights are read at all.
        let pre = trainer.resolve_plan(req, &[])?;
        // The image RoPE law the run trains and exports (stage 1: square).
        config.model.rope_aspect = pre.rope_aspect.clone();
        let compute = match pre.mixed_precision {
            MixedPrecision::Bf16 => Dtype::Bfloat16,
            MixedPrecision::Fp32 => Dtype::Float32,
            MixedPrecision::Fp16 => Dtype::Float16,
        };

        // ---- the topology (rendezvous before anything heavy loads) ----------------------
        let coll = match make_collective(&pre.dist.world, &req.cancel) {
            Ok(c) => c,
            Err(e) => {
                if let Some(d) = dist_error(&e) {
                    *failure.borrow_mut() = Some(d.clone());
                }
                return Err(e.into());
            }
        };
        let coll: Box<dyn Collective> = Box::new(Recording {
            inner: coll,
            slot: failure.clone(),
        });
        if pre.dist.world.world_size > 1 {
            eprintln!(
                "[world] rank={}/{} local_rank={}/{} node={}/{} transport={} | {}",
                pre.dist.world.rank,
                pre.dist.world.world_size,
                pre.dist.world.local_rank,
                pre.dist.world.local_world_size,
                pre.dist.world.node_rank,
                pre.dist.world.num_nodes,
                pre.dist.world.transport,
                pre.dist.summary()
            );
        }

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

        let (model, mut params) = match &plan.artifact {
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
        if let Some(r) = &plan.repa {
            params.extend(init_projector(r, plan.seed)?);
        }
        let extras = TrainExtras {
            repa: plan.repa.clone(),
            ac: plan.ac,
        };
        let full_shapes: HashMap<Rc<str>, Vec<i32>> = params
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().to_vec()))
            .collect();
        let mut topo = Topology {
            plan: plan.dist.clone(),
            coll,
            shapes: full_shapes.clone(),
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
        let route = |k: &str, nd: usize| -> ParamRoute {
            if k.starts_with(PROJECTOR_PREFIX) {
                // `build_muon_param_groups`' `aux` group (and plain AdamW's single group).
                return ParamRoute::AdamW;
            }
            match adapter_targets.get(k) {
                Some(target) => adapter_param_route(target, nd, kind),
                None => full_param_route(k, nd, kind, &cfg_model),
            }
        };
        // A sharded run persists only its rows of every tensor.
        let sharded = plan.dist.sharded();
        let mut rows = Params::with_capacity(params.len());
        let mut full_rows: HashMap<Rc<str>, usize> = HashMap::new();
        for (k, v) in &params {
            rows.insert(k.clone(), topo.shard_of(k, v)?);
            if sharded && v.ndim() >= 1 {
                full_rows.insert(k.clone(), v.shape()[0] as usize);
            }
        }
        let mut opt = IrisOptimizer::new_sharded(&plan.optimizer, &rows, &full_rows, route)?;
        let core_ema = |m: &Params| -> Params {
            m.iter()
                .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        let mut ema = plan.ema.map(|_| core_ema(&rows));

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
        if plan.preview.every > 0 && topo.is_main() {
            for p in &plan.preview.prompts {
                preview.push((p.clone(), text.encode(p)?));
            }
            if plan.preview.cfg_scale != 1.0 {
                preview_negative = Some(text.encode(&plan.preview.negative_prompt)?);
            }
        }
        // The dataset pass (one `Caching` event per item): every image must decode as an image,
        // every caption variant must be selectable, and — with `text_conditioning: cached` — every
        // distinct caption is encoded once into the memo.
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

        // ---- the REPA teacher (frozen, fp32) --------------------------------------------------
        let teacher = match &plan.repa {
            Some(r) => Some(Teacher::load(
                &teacher_path(&trainer.teacher_dir, r)?,
                r.teacher.clone(),
            )?),
            None => None,
        };

        // ---- the data plan of this rank -------------------------------------------------------
        let world = plan.dist.world.clone();
        let mut pipeline = DataPipeline::new(
            &req.items,
            plan.shape.clone(),
            plan.batch_size,
            world.rank,
            world.world_size,
        )?;
        if let Some(warning) = pipeline.unused_tail_warning() {
            eprintln!("{warning}");
        }
        // `_agree_batch_count`: every rank walks the global minimum of shape batches.
        let mut local = [pipeline.epoch_batches() as f64];
        topo.reduce_scalars(&mut local, ReduceOp::Min)?;
        pipeline.set_epoch_batches(local[0] as usize)?;
        let holdout_sizes = pipeline.holdout_sizes(&req.items)?;
        let total_steps = contract::total_optimizer_steps(
            pipeline.epoch_batches(),
            plan.grad_accum,
            plan.num_epochs,
            plan.max_steps,
        );
        let schedule = TrainSchedule::new(plan.flow.num_train_timesteps, plan.flow.shift);
        let ckpt_root = checkpoint_root(&req.output_dir, &req.file_name);
        let dataset_fingerprint = dataset_fingerprint(req)?;

        // ---- resume -------------------------------------------------------------------------
        let mut run_rows = rows;
        let (mut step, mut sched_step, mut lr, mut nan_count, mut last_loss) =
            (0u64, 0u64, plan.optimizer.lr, 0u32, f64::NAN);
        let mut first_epoch = 1u32;
        let mut resume_position = None;
        let mut scaler = (plan.mixed_precision == MixedPrecision::Fp16).then(GradScaler::default);
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
        // sc-25690: every rank resumes from one agreed checkpoint; one writer per storage; the
        // checkpoints that are not this run's lineage move aside before its first publish.
        let search_root = match (&plan.resume_from, plan.resume) {
            (Some(p), _) if !p.join(CKPT_STATE).is_file() => Some(p.clone()),
            (None, true) => Some(ckpt_root.clone()),
            _ => None,
        };
        let resume_dir = match agree_resume(topo.coll.as_mut(), resume_dir, search_root.as_deref())
        {
            Ok(d) => d,
            Err(e) => {
                // typed at `Trainer::train`, like every other distributed failure
                if let Some(d) = dist_error(&e) {
                    failure.borrow_mut().get_or_insert(d.clone());
                }
                return Err(e.into());
            }
        };
        let writes_checkpoints =
            elect_checkpoint_writer(topo.coll.as_mut(), plan.dist.checkpoint_io, &ckpt_root)?;
        if writes_checkpoints {
            retire_foreign_checkpoints(&ckpt_root, resume_dir.as_deref())?;
        }
        if plan.dist.world.world_size > 1 {
            topo.coll.barrier("checkpoint lineage")?;
        }
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let payload = read_checkpoint_members(&dir, &saved)?;
            let core_keys: Vec<Rc<str>> = run_rows
                .keys()
                .filter(|k| !k.starts_with(PROJECTOR_PREFIX))
                .cloned()
                .collect();
            if payload.trainable.len() != core_keys.len()
                || core_keys
                    .iter()
                    .any(|k| !payload.trainable.contains_key(k.as_ref()))
            {
                return Err(Error::Msg(
                    "iris resume: the checkpoint's trainable tensors do not match this run".into(),
                ));
            }
            for (k, v) in &payload.trainable {
                let key: Rc<str> = Rc::from(k.as_str());
                let r = topo.shard_of(&key, v)?;
                run_rows.insert(key, r);
            }
            // The projector carries over when it matches; `resume_optimizer: core` starts a new
            // one cold (its fresh init stays).
            let projector_carries = plan.repa.as_ref().is_some_and(|r| {
                saved.repa_projector.as_deref() == Some(r.projector_identity().as_str())
            });
            if projector_carries {
                let pj = payload.projector.as_ref().ok_or_else(|| {
                    Error::Msg("iris resume: the checkpoint names a projector it lacks".into())
                })?;
                for (k, v) in pj {
                    let key: Rc<str> = Rc::from(k.as_str());
                    let r = topo.shard_of(&key, v)?;
                    run_rows.insert(key, r);
                }
            }
            if let Some(e) = ema.as_mut() {
                let saved_ema = payload.ema.as_ref().ok_or_else(|| {
                    Error::Msg("iris resume: the checkpoint carries no EMA".into())
                })?;
                for (k, v) in saved_ema {
                    let key: Rc<str> = Rc::from(k.as_str());
                    let r = topo.shard_of(&key, v)?;
                    e.insert(key, r);
                }
            }
            // Optimizer slots: full tensors sliced to this rank's rows; under `core` only the
            // backbone's survive (a projector's moments restart).
            let mut slots: HashMap<String, Array> = HashMap::new();
            for (name, v) in &payload.optimizer {
                if name == "__t" {
                    slots.insert(name.clone(), v.clone());
                    continue;
                }
                let key = name.split("::").next().unwrap_or(name);
                if key.starts_with(PROJECTOR_PREFIX) && !projector_carries {
                    continue;
                }
                slots.insert(name.clone(), topo.shard_of(key, v)?);
            }
            if !projector_carries {
                // fill the fresh projector's zero slots from the live optimizer
                for (name, v) in opt.state_tensors() {
                    if name.starts_with(PROJECTOR_PREFIX) {
                        slots.insert(name, v);
                    }
                }
            }
            opt.load_state(&slots)?;
            step = saved.step;
            sched_step = saved.scheduler_step;
            lr = if plan.override_lr_on_resume {
                plan.optimizer.lr
            } else {
                saved.lr
            };
            nan_count = saved.nan_count;
            last_loss = saved.last_loss;
            if let (Some(s), Some(g)) = (scaler.as_mut(), saved.grad_scaler.as_ref()) {
                *s = g.clone();
            }
            // `_resolve_resume_data_position`
            let position = match plan.resume_data_policy {
                ResumeDataPolicy::NewPhase => {
                    eprintln!(
                        "iris training: resume_data_policy=new_phase: retained checkpoint step {} \
                         and model/EMA/optimizer/scheduler state; starting the current dataset at \
                         epoch 1 with zero consumed batches",
                        saved.step
                    );
                    pipeline.fresh_position()?
                }
                ResumeDataPolicy::Exact => match &saved.data_position {
                    Some(p) => p.clone(),
                    None => {
                        let (batch, _) = saved.legacy_batch().ok_or_else(|| {
                            Error::Msg(
                                "iris resume: the checkpoint has neither a data_position nor a \
                                 legacy batch identity; resume_data_policy new_phase starts the \
                                 current dataset fresh"
                                    .into(),
                            )
                        })?;
                        pipeline.legacy_position(saved.epoch, saved.batches_consumed, batch)?
                    }
                },
            };
            first_epoch = position
                .get("epoch")
                .and_then(JsonValue::as_u64)
                .unwrap_or(1) as u32;
            resume_position = Some(position);
        }
        let run = Self {
            plan,
            config,
            adaln_zero_init,
            model,
            extras,
            params: run_rows,
            full: None,
            ema,
            opt,
            text,
            null,
            preview,
            preview_negative,
            pipeline,
            holdout_sizes,
            schedule,
            total_steps,
            ckpt_root,
            base_identity,
            dataset_fingerprint,
            teacher,
            topo,
            scaler,
            step,
            sched_step,
            lr,
            nan_count,
            last_loss,
            first_epoch,
            resume_position,
            writes_checkpoints,
            log_window: Vec::new(),
        };
        if run.topo.is_main() {
            run.write_metrics_header()?;
        }
        Ok(run)
    }

    fn write_metrics_header(&self) -> Result<()> {
        holdout_contract::append_metrics(
            &self.ckpt_root,
            &json!({
                "kind": "run",
                "trainer": TRAINER_ID,
                "backend": "mlx",
                "run_name": self.plan.run_name,
                "tracking": self.plan.tracking,
                "stage": self.plan.stage,
                "parent_stage": self.plan.parent_stage,
                "shape_policy": self.plan.shape_policy_id,
                "activation_checkpointing": self.plan.ac.name(),
                "repa": self.plan.repa.as_ref().map(|r| json!({
                    "variant": r.variant.as_str(), "weight": r.weight, "layer": r.layer,
                    "teacher": r.teacher.entrypoint,
                })),
                "dist": self.plan.dist.identity(),
                "perf": self.plan.perf,
                "start_step": self.step,
            }),
        )?;
        Ok(())
    }

    /// The gathered (unsharded) parameters at `dist.param_dtype` — FSDP2's
    /// `MixedPrecisionPolicy(param_dtype)`: the all-gathered copy the forward and backward read is
    /// rounded to that precision (the f32 shards stay the optimizer's masters; the compute dtype
    /// stays `mixed_precision`'s).
    fn gather_forward(&mut self) -> Result<Params> {
        let full = self.topo.gather_map(&self.params)?;
        let dtype = match self.plan.dist.param_dtype {
            gen_core::iris::train::dist::Precision::Fp32 => return Ok(full),
            gen_core::iris::train::dist::Precision::Bf16 => Dtype::Bfloat16,
            gen_core::iris::train::dist::Precision::Fp16 => Dtype::Float16,
        };
        full.into_iter()
            .map(|(k, v)| Ok((k, v.as_dtype(dtype)?.as_dtype(Dtype::Float32)?)))
            .collect()
    }

    /// The full parameters for a forward (gathered from the shard group under sharding; kept for
    /// the window unless `reshard_after_forward`, kept across steps under `grad_op`).
    fn forward_params(&mut self) -> Result<Params> {
        if !self.plan.dist.sharded() {
            return Ok(self.params.clone());
        }
        if let Some(f) = &self.full {
            return Ok(f.clone());
        }
        let full = self.gather_forward()?;
        let keep =
            !self.plan.dist.sharding.shards_params() || !self.plan.dist.reshard_after_forward;
        if keep {
            self.full = Some(full.clone());
        }
        Ok(full)
    }

    /// Assemble micro-batch `position` of `epoch` from training positions `positions`, plus its
    /// REPA targets.
    fn batch(
        &mut self,
        req: &TrainingRequest,
        epoch: u32,
        position: usize,
        positions: &[usize],
    ) -> Result<(StepBatch, Option<RepaTargets>)> {
        let b = positions.len();
        let (h0, w0) = self.pipeline.sizes[positions[0]];
        let shape = self.pipeline.policy.target(h0, w0);
        let (hh, ww) = (shape.height, shape.width);
        let numel = 3 * hh * ww;
        let draws = draw_batch(
            batch_seed_rank(self.plan.seed, self.plan.dist.world.rank, epoch, position),
            b,
            numel,
            self.plan.text_dropout,
            self.plan.flow.sampler,
            self.plan.flow.num_train_timesteps,
        );
        let mut pixels = Vec::with_capacity(b * numel);
        let mut states = Vec::with_capacity(b);
        let mut masks = Vec::with_capacity(b);
        for (row, &pos) in positions.iter().enumerate() {
            let item = &req.items[self.pipeline.train[pos]];
            let img = preprocess_file_to_shape(&item.image_path, &self.pipeline.policy)?;
            if (img.height, img.width) != (hh, ww) {
                return Err(Error::Msg(format!(
                    "iris training: a shape batch mixed {}x{} into {hh}x{ww}",
                    img.height, img.width
                )));
            }
            pixels.extend_from_slice(&img.chw);
            let mut rng = HostRng::new(caption_seed(self.plan.seed, epoch, pos));
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
        let shape4 = [b as i32, 3, hh as i32, ww as i32];
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
        let batch = StepBatch {
            x0: Array::from_slice(&pixels, &shape4),
            noise: Array::from_slice(&draws.noise, &shape4),
            sigma: Array::from_slice(&sigmas, &[b as i32, 1, 1, 1]),
            t: Array::from_slice(&times, &[b as i32]),
            states: concatenate_axis(&states, 0)?,
            masks,
        };
        let targets = self.repa_targets(&batch.x0, (hh, ww))?;
        Ok((batch, targets))
    }

    /// The frozen teacher's targets for clean images `x0` of `size` (fp32, outside the trace).
    fn repa_targets(&self, x0: &Array, size: (usize, usize)) -> Result<Option<RepaTargets>> {
        let (Some(plan), Some(teacher)) = (&self.plan.repa, &self.teacher) else {
            return Ok(None);
        };
        let p = self.config.model.patch_size;
        let student_grid = (size.0 / p, size.1 / p);
        let teacher_grid = plan.teacher_grid(student_grid);
        let mut t = teacher.tokens(x0, teacher_grid)?;
        if plan.variant == gen_core::iris::train::repa::RepaVariant::Irepa {
            t = spatial_normalize(&t, plan.spatial_norm_gamma)?;
        }
        eval([&t])?;
        Ok(Some(RepaTargets {
            teacher: t,
            student_grid,
            teacher_grid,
        }))
    }

    fn world(&self) -> usize {
        self.plan.dist.world.world_size
    }

    /// One collective flag reduction (any rank set → every rank sees it).
    fn any(&mut self, flags: &mut [f64]) -> Result<()> {
        if self.world() > 1 {
            self.topo.reduce_scalars(flags, ReduceOp::Max)?;
        }
        Ok(())
    }

    fn execute(
        mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let plan = self.plan.clone();
        let accum = plan.grad_accum;
        let max_epoch = plan.num_epochs.unwrap_or(u32::MAX);
        let first_step = self.step;
        let mut last_saved = if self.step > 0 { Some(self.step) } else { None };
        let mut window = Window::new(accum);
        let mut micro: u64 = 0;
        let mut boundary: Option<(u32, usize, usize, usize)> = None;
        let mut cancelled = false;
        let reached = |step: u64| plan.max_steps > 0 && step >= plan.max_steps as u64;
        let mut finished = reached(self.step);
        let mut epoch = self.first_epoch;
        let mut resume = self.resume_position.take();
        // sc-25690: the data position of a boundary is the walker's state of the pass that set
        // it. `begin_epoch` resets the walker, so a boundary carried into the next pass (a cancel
        // before its first step) saves the position captured when its pass ended.
        let mut walker_epoch = 0u32;
        let mut settled: Option<Settled> = None;
        while !finished && !cancelled && epoch <= max_epoch {
            let resumed_pass = resume.is_some();
            let skip = self.pipeline.begin_epoch(epoch, resume.take().as_ref())?;
            walker_epoch = epoch;
            if resumed_pass && self.pipeline.regroups() {
                let mut v = [self.pipeline.remaining_batches() as f64];
                self.topo.reduce_scalars(&mut v, ReduceOp::Min)?;
                self.pipeline.limit_remaining(v[0] as usize);
            }
            let batches = self.pipeline.batches();
            let mut plan_samples = 0usize;
            for (local_step, positions) in batches.iter().enumerate() {
                let mut cancel = [if req.cancel.is_cancelled() { 1.0 } else { 0.0 }];
                self.any(&mut cancel)?;
                if cancel[0] > 0.0 {
                    cancelled = true;
                    break;
                }
                let position = skip + local_step;
                plan_samples += positions.len();
                let (batch, targets) = self.batch(req, epoch, position, positions)?;
                let samples = batch.masks.len();
                micro += 1;
                let sync = micro.is_multiple_of(accum as u64);
                let scale = self.scaler.as_ref().map(|s| s.scale).unwrap_or(1.0) as f32;
                let fwd = self.forward_params()?;
                let (flow, repa, grads) = loss_and_grads_ext(
                    &self.model,
                    &fwd,
                    &batch,
                    &plan.flow,
                    &self.extras,
                    targets.as_ref(),
                    scale / accum as f32,
                )?;
                drop(fwd);
                if self.plan.dist.sharded()
                    && self.plan.dist.sharding.shards_params()
                    && self.plan.dist.reshard_after_forward
                {
                    self.full = None;
                }
                drop(batch);
                let total = flow as f64
                    + plan.repa.as_ref().map(|r| r.weight).unwrap_or(0.0)
                        * repa.unwrap_or(0.0) as f64;
                // The pre-backward guard (`lazy_sync = false`) — collective, so every rank drops
                // the same window.
                let mut bad = [if !plan.lazy_sync && !total.is_finite() {
                    1.0
                } else {
                    0.0
                }];
                self.any(&mut bad)?;
                if bad[0] > 0.0 {
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
                window.add(grads, flow, repa, samples)?;
                if !sync {
                    continue;
                }
                let lr_now = self.lr * plan.lr_factor_at(self.sched_step, self.total_steps);
                match self.close_window(&mut window, lr_now)? {
                    WindowOutcome::Empty => continue,
                    WindowOutcome::NonFinite => {
                        self.nan_count += 1;
                        eprintln!(
                            "iris training: non-finite grad norm at step {} ({}/{})",
                            self.step, self.nan_count, plan.nan_loss_tolerance
                        );
                        if self.nan_count > plan.nan_loss_tolerance {
                            return Err(Error::Msg(format!(
                                "iris training: grads were non-finite {} times; aborting",
                                self.nan_count
                            )));
                        }
                        continue;
                    }
                    WindowOutcome::Stepped {
                        norm,
                        flow,
                        repa,
                        skipped_by_scaler,
                    } => {
                        if !skipped_by_scaler {
                            self.sched_step += plan.scheduler_world();
                        }
                        self.step += 1;
                        self.last_loss = flow;
                        boundary = Some((epoch, position + 1, local_step + 1, plan_samples));
                        self.log_step(flow, repa, norm, lr_now, epoch)?;
                    }
                }
                on_progress(TrainingProgress::Training {
                    step: self.micro_step(),
                    total: (self.total_steps * accum as u64) as u32,
                    loss: self.last_loss as f32,
                });
                if plan.save_every > 0 && self.step.is_multiple_of(plan.save_every as u64) {
                    let b = boundary.expect("stepped");
                    let position = self.position_at(b, walker_epoch, &settled)?;
                    self.save_checkpoint(b, position)?;
                    last_saved = Some(self.step);
                    on_progress(TrainingProgress::Checkpoint {
                        step: self.micro_step(),
                    });
                }
                let preview_due = plan.preview.every > 0
                    && (self.step.is_multiple_of(plan.preview.every as u64)
                        || (plan.preview_at_step_one && self.step == 1));
                if preview_due {
                    self.render_previews(req, on_progress)?;
                }
                if plan.holdout.every > 0
                    && !self.pipeline.holdout.is_empty()
                    && self.step.is_multiple_of(plan.holdout.every as u64)
                {
                    self.validate_holdout(req)?;
                }
                if reached(self.step) {
                    finished = true;
                    break;
                }
            }
            if finished || cancelled {
                break;
            }
            if let Some(b) = boundary.filter(|b| b.0 == walker_epoch) {
                let position = self.pipeline.data_position(b.0, b.1, b.2, b.3)?;
                settled = Some(Settled {
                    boundary: b,
                    position,
                });
            }
            // `save_every_epochs` (and the final epoch), on a step not yet saved.
            if plan.save_every_epochs > 0
                && (epoch.is_multiple_of(plan.save_every_epochs) || Some(epoch) == plan.num_epochs)
                && last_saved != Some(self.step)
            {
                if let Some(b) = boundary {
                    let position = self.position_at(b, walker_epoch, &settled)?;
                    self.save_checkpoint(b, position)?;
                    last_saved = Some(self.step);
                    on_progress(TrainingProgress::Checkpoint {
                        step: self.micro_step(),
                    });
                }
            }
            epoch += 1;
        }
        if cancelled && self.step == first_step {
            // No optimizer step ran in this invocation: the typed cancellation, nothing written.
            return Err(Error::Canceled);
        }
        if last_saved != Some(self.step) && self.step > 0 {
            if let Some(b) = boundary {
                let position = self.position_at(b, walker_epoch, &settled)?;
                self.save_checkpoint(b, position)?;
                on_progress(TrainingProgress::Checkpoint {
                    step: self.micro_step(),
                });
            }
        }
        on_progress(TrainingProgress::Saving);
        let path = self.export(req)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: self.micro_step(),
            final_loss: self.last_loss as f32,
        })
    }

    /// Close one accumulation window: reduce across ranks, unscale (fp16), clip, the lazy-sync
    /// guard, EMA (pre-step), the optimizer step, re-gather (sharded).
    fn close_window(&mut self, window: &mut Window, lr: f64) -> Result<WindowOutcome> {
        let plan = self.plan.clone();
        let Some((grads, flow_sum, repa_sum, samples)) = window.take() else {
            return Ok(WindowOutcome::Empty);
        };
        // The global sample-weighted mean losses (`_LossLogAccumulator`).
        let mut logs = [flow_sum, repa_sum, samples as f64];
        if self.world() > 1 {
            self.topo.reduce_scalars(&mut logs, ReduceOp::Sum)?;
        }
        let flow = logs[0] / logs[2].max(1.0);
        let repa = plan.repa.as_ref().map(|_| logs[1] / logs[2].max(1.0));
        let grads: Params = grads
            .into_iter()
            .filter(|(k, _)| self.opt.is_trained(k))
            .collect();
        let mut grads = self.topo.reduce_grads(grads)?;
        if let Some(s) = &self.scaler {
            let inv = Array::from_f32((1.0 / s.scale) as f32);
            for g in grads.values_mut() {
                *g = g.multiply(&inv)?;
            }
        }
        if plan.log_block_grad_norms && self.topo.is_main() {
            self.log_block_norms(&grads)?;
        }
        let (local_norm, _) = clip_grads(grads.clone(), plan.gradient_clip)?;
        let sq = self.topo.global_sq_norm(local_norm * local_norm)?;
        let norm = sq.sqrt();
        if plan.lazy_sync && !norm.is_finite() {
            return Ok(WindowOutcome::NonFinite);
        }
        let found_inf = !norm.is_finite();
        if found_inf && self.scaler.is_none() {
            // bf16 / fp32 without lazy_sync: torch steps with non-finite grads; upstream's guard
            // already dropped any non-finite loss, so this is a finite-loss / inf-grad window.
            eprintln!(
                "iris training: non-finite gradient norm at step {} with a finite loss",
                self.step
            );
        }
        let coef = contract::clip_coefficient(plan.gradient_clip, norm);
        let c = Array::from_f32(coef as f32);
        for g in grads.values_mut() {
            *g = g.as_dtype(Dtype::Float32)?.multiply(&c)?;
        }
        if let (Some(ema), Some(decay)) = (self.ema.as_mut(), plan.ema) {
            ema_update(ema, &self.params, decay)?;
        }
        let skipped = found_inf && self.scaler.is_some();
        if !skipped {
            if plan.dist.sharded() {
                let Self {
                    opt, params, topo, ..
                } = self;
                let mut seam = ShardMuon { topo };
                opt.step_sharded(params, &grads, lr, Some(&mut seam))?;
            } else {
                self.opt.step(&mut self.params, &grads, lr)?;
            }
        }
        if let Some(s) = self.scaler.as_mut() {
            s.update(found_inf);
        }
        self.full = None;
        if plan.dist.sharded() && !plan.dist.sharding.shards_params() {
            // grad_op: the parameters stay gathered between steps.
            let full = self.gather_forward()?;
            self.full = Some(full);
        }
        Ok(WindowOutcome::Stepped {
            norm,
            flow,
            repa,
            skipped_by_scaler: skipped,
        })
    }

    fn log_step(
        &mut self,
        flow: f64,
        repa: Option<f64>,
        norm: f64,
        lr: f64,
        epoch: u32,
    ) -> Result<()> {
        self.log_window.push(flow);
        let every = self.plan.log_every.max(1) as u64;
        if !self.topo.is_main() || !self.step.is_multiple_of(every) {
            return Ok(());
        }
        let mean = self.log_window.iter().sum::<f64>() / self.log_window.len().max(1) as f64;
        self.log_window.clear();
        let mut rec = json!({
            "kind": "step", "step": self.step, "epoch": epoch, "loss": mean,
            "loss/flow": flow, "grad_norm": norm, "lr": lr,
        });
        if let (Some(r), Some(plan)) = (repa, &self.plan.repa) {
            let o = rec.as_object_mut().expect("object");
            o.insert("loss/repa_raw".into(), json!(r));
            o.insert("loss/repa_weighted".into(), json!(plan.weight * r));
            o.insert("loss/total".into(), json!(flow + plan.weight * r));
        }
        if let Some(s) = &self.scaler {
            rec.as_object_mut()
                .expect("object")
                .insert("loss_scale".into(), json!(s.scale));
        }
        eprintln!(
            "iris training: epoch {epoch} | step {}/{} | loss {mean:.4} | grad {norm:.3} | lr {lr:.3e}",
            self.step, self.total_steps
        );
        Ok(holdout_contract::append_metrics(&self.ckpt_root, &rec)?)
    }

    /// `log_block_grad_norms`: per-module pre-clip, post-reduce gradient norms.
    fn log_block_norms(&self, grads: &Params) -> Result<()> {
        let mut groups: std::collections::BTreeMap<String, f64> = Default::default();
        for (k, g) in grads {
            let name = block_group(k);
            let v = to_host(g)?;
            *groups.entry(name).or_insert(0.0) +=
                v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        }
        let mut rec = serde_json::Map::new();
        rec.insert("kind".into(), json!("grad_norms"));
        rec.insert("step".into(), json!(self.step + 1));
        for (k, v) in groups {
            rec.insert(format!("grad/{k}"), json!(v.sqrt()));
        }
        Ok(holdout_contract::append_metrics(
            &self.ckpt_root,
            &JsonValue::Object(rec),
        )?)
    }

    /// Frozen-grid holdout validation on every rank; the main rank records it.
    fn validate_holdout(&mut self, req: &TrainingRequest) -> Result<()> {
        let plan = self.plan.clone();
        let batches = self
            .pipeline
            .holdout_batches(&self.holdout_sizes, plan.holdout.samples)?;
        let params = self.forward_params()?;
        let dit = self.model.dit(&params)?;
        let n = plan.flow.num_train_timesteps;
        let mut sums = ValSums::default();
        for positions in batches {
            let (h0, w0) = self.holdout_sizes[positions[0]];
            let shape = self.pipeline.policy.target(h0, w0);
            let (hh, ww) = (shape.height, shape.width);
            let numel = 3 * hh * ww;
            let b = positions.len();
            let mut pixels = Vec::with_capacity(b * numel);
            let mut noise = Vec::with_capacity(b * numel);
            let mut states = Vec::with_capacity(b);
            let mut masks = Vec::with_capacity(b);
            let mut t_idx = Vec::with_capacity(b);
            for &k in &positions {
                let item = &req.items[self.pipeline.holdout[k]];
                let img = preprocess_file_to_shape(&item.image_path, &self.pipeline.policy)?;
                pixels.extend_from_slice(&img.chw);
                noise.extend(holdout_contract::grid_noise(k, numel));
                t_idx.push(holdout_contract::grid_timestep(k, n));
                // captions pinned at (seed 0, epoch 0)
                let mut rng = HostRng::new(caption_seed(0, 0, k));
                let caption =
                    select_caption(item, &plan.caption_field, &plan.caption_fields, &mut rng)?;
                let enc = self.text.encode(&caption)?;
                states.push(enc.states.as_dtype(Dtype::Float32)?);
                masks.push(enc.mask);
            }
            let shape4 = [b as i32, 3, hh as i32, ww as i32];
            let sig: Vec<f32> = t_idx.iter().map(|&t| self.schedule.sigmas[t]).collect();
            let tim: Vec<f32> = t_idx
                .iter()
                .map(|&t| self.schedule.model_times[t])
                .collect();
            let batch = StepBatch {
                x0: Array::from_slice(&pixels, &shape4),
                noise: Array::from_slice(&noise, &shape4),
                sigma: Array::from_slice(&sig, &[b as i32, 1, 1, 1]),
                t: Array::from_slice(&tim, &[b as i32]),
                states: concatenate_axis(&states, 0)?,
                masks,
            };
            let losses = to_host(&flow_loss_per_sample(&dit, &batch, &plan.flow)?)?;
            for (l, t) in losses.iter().zip(&t_idx) {
                sums.add(*l as f64, *t, n);
            }
        }
        drop(dit);
        if self.plan.dist.sharded()
            && self.plan.dist.sharding.shards_params()
            && self.plan.dist.reshard_after_forward
        {
            self.full = None;
        }
        let mut flat: Vec<f64> = sums
            .sums
            .iter()
            .chain(sums.counts.iter())
            .copied()
            .collect();
        if self.world() > 1 {
            self.topo.reduce_scalars(&mut flat, ReduceOp::Sum)?;
        }
        let total = ValSums {
            sums: flat[..5].try_into().expect("5"),
            counts: flat[5..].try_into().expect("5"),
        };
        if self.topo.is_main() {
            let logs = total.logs();
            eprintln!(
                "iris training: val @ {}: {}",
                self.step,
                logs.iter()
                    .map(|(k, v)| format!("{}={v:.4}", k.rsplit('/').next().unwrap_or(k)))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            holdout_contract::append_metrics(&self.ckpt_root, &total.record(self.step))?;
        }
        Ok(())
    }

    /// The shared contract's step unit: micro-steps through the last completed optimizer step.
    fn micro_step(&self) -> u32 {
        (self.step * self.plan.grad_accum as u64) as u32
    }

    /// The data position of boundary `b` (sc-25690): read from the walker while it still walks
    /// `b`'s pass (`walker_epoch`), else the position `settled` when that pass ended.
    fn position_at(
        &mut self,
        b: (u32, usize, usize, usize),
        walker_epoch: u32,
        settled: &Option<Settled>,
    ) -> Result<JsonValue> {
        if b.0 == walker_epoch {
            return Ok(self.pipeline.data_position(b.0, b.1, b.2, b.3)?);
        }
        match settled {
            Some(s) if s.boundary == b => Ok(s.position.clone()),
            _ => Err(Error::Msg(format!(
                "iris checkpoint: no data position was captured for the boundary {b:?} of an \
                 ended pass"
            ))),
        }
    }

    fn state(&self, position: JsonValue, boundary: (u32, usize)) -> CheckpointState {
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
            task: gen_core::iris::IrisTask::Generation,
            stage: self.plan.stage.clone(),
            parent_stage: self.plan.parent_stage.clone(),
            data_position: Some(position),
            dist: Some(self.plan.dist.identity()),
            repa_projector: self.plan.repa.as_ref().map(RepaPlan::projector_identity),
            grad_scaler: self.scaler.clone(),
        }
    }

    /// Publish a checkpoint of the state at an optimizer-step boundary
    /// `(epoch, batches_consumed, plan_batches, plan_samples)` — collectively: consolidated by the
    /// elected writer(s), or (`dcp`) one shard file per shard rank.
    fn save_checkpoint(
        &mut self,
        b: (u32, usize, usize, usize),
        position: JsonValue,
    ) -> Result<()> {
        let state = self.state(position, (b.0, b.1));
        let io = self.plan.dist.checkpoint_io;
        let world = self.plan.dist.world.clone();
        let mkdir = |p: &Path| {
            std::fs::create_dir_all(p)
                .map_err(|e| Error::Msg(format!("iris checkpoint: create {}: {e}", p.display())))
        };
        if io == gen_core::iris::train::dist::CheckpointIo::Dcp {
            let staging = self.ckpt_root.join(format!(
                ".{}.dcp",
                gen_core::iris::train::checkpoint_dir_name(self.step)
            ));
            if world.rank == 0 {
                mkdir(&self.ckpt_root)?;
                if staging.exists() {
                    let _ = std::fs::remove_dir_all(&staging);
                }
                mkdir(&staging)?;
            }
            self.topo.coll.barrier("dcp staging")?;
            if world.rank < self.plan.dist.dp_shard.max(1) || !self.plan.dist.sharded() {
                let mut list: Vec<(String, Array)> = Vec::new();
                for (k, v) in &self.params {
                    list.push((format!("p::{k}"), v.clone()));
                }
                if let Some(e) = &self.ema {
                    for (k, v) in e {
                        list.push((format!("e::{k}"), v.clone()));
                    }
                }
                for (k, v) in self.opt.state_tensors() {
                    list.push((format!("o::{k}"), v));
                }
                // Zero-row shards (more shards than rows) have nothing to write.
                list.retain(|(_, a)| a.size() > 0);
                list.sort_by(|a, c| a.0.cmp(&c.0));
                let path = staging.join(format!(
                    "shard_{:05}.safetensors",
                    self.plan.dist.shard_index()
                ));
                save_safetensors_atomic(&path, &list, &Default::default())?;
            }
            self.topo.coll.barrier("dcp shards")?;
            if world.rank == 0 {
                contract::atomic_write(
                    &staging.join(CKPT_STATE),
                    serde_json::to_string_pretty(&state.to_json())
                        .map_err(|e| Error::Msg(e.to_string()))?
                        .as_bytes(),
                )?;
                publish_checkpoint(&self.ckpt_root, &staging, self.step)?;
                prune_checkpoints(
                    &self.ckpt_root,
                    self.plan.keep_last_checkpoints,
                    &self.plan.milestone_steps,
                )?;
            }
            self.topo.coll.barrier("dcp publish")?;
            return Ok(());
        }
        // Consolidated: every rank takes part in the gathers; the writer(s) publish.
        let params = if self.plan.dist.sharded() {
            self.topo.gather_map(&self.params)?
        } else {
            self.params.clone()
        };
        let ema = match &self.ema {
            Some(e) if self.plan.dist.sharded() => Some(self.topo.gather_map(e)?),
            other => other.clone(),
        };
        let mut opt_state: Vec<(String, Array)> = Vec::new();
        for (name, v) in self.opt.state_tensors() {
            if name == "__t" || !self.plan.dist.sharded() {
                opt_state.push((name, v));
                continue;
            }
            let key = name.split("::").next().unwrap_or(&name).to_string();
            let full = self.topo.gather_full(&key, &v)?;
            opt_state.push((name, full));
        }
        if self.writes_checkpoints {
            mkdir(&self.ckpt_root)?;
            let mut staging = checkpoint_staging_dir(&self.ckpt_root, self.step);
            if io == gen_core::iris::train::dist::CheckpointIo::PerNode {
                staging = staging.with_extension(format!("n{}", world.node_rank));
            }
            if staging.exists() {
                let _ = std::fs::remove_dir_all(&staging);
            }
            mkdir(&staging)?;
            let core: Params = params
                .iter()
                .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            save_params(&staging.join(CKPT_TRAINABLE), &core)?;
            let projector: Params = params
                .iter()
                .filter(|(k, _)| k.starts_with(PROJECTOR_PREFIX))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if !projector.is_empty() {
                save_params(&staging.join(CKPT_REPA), &projector)?;
            }
            if let Some(e) = &ema {
                save_params(&staging.join(CKPT_EMA), e)?;
            }
            opt_state.sort_by(|a, c| a.0.cmp(&c.0));
            Array::save_safetensors(
                opt_state.iter().map(|(k, v)| (k.as_str(), v)),
                None::<&HashMap<String, String>>,
                staging.join(CKPT_OPTIMIZER),
            )?;
            contract::atomic_write(
                &staging.join(CKPT_STATE),
                serde_json::to_string_pretty(&state.to_json())
                    .map_err(|e| Error::Msg(e.to_string()))?
                    .as_bytes(),
            )?;
            publish_checkpoint(&self.ckpt_root, &staging, self.step)?;
            prune_checkpoints(
                &self.ckpt_root,
                self.plan.keep_last_checkpoints,
                &self.plan.milestone_steps,
            )?;
        }
        if self.world() > 1 {
            self.topo.coll.barrier("checkpoint")?;
        }
        Ok(())
    }

    /// The full core tensors an export / preview reads (`which`), gathered under sharding.
    fn selected_full(&mut self, which: WeightsSelect) -> Result<Params> {
        let src = match (which, &self.ema) {
            (WeightsSelect::Ema, Some(e)) => e.clone(),
            _ => self.params.clone(),
        };
        let full = if self.plan.dist.sharded() {
            self.topo.gather_map(&src)?
        } else {
            src
        };
        Ok(full
            .into_iter()
            .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
            .collect())
    }

    /// Write the final artifact (full model directory or adapter file); returns its tensor file.
    /// The main rank writes; every rank returns the same path.
    fn export(&mut self, req: &TrainingRequest) -> Result<PathBuf> {
        let which = self.plan.export_weights;
        let tensors = self.selected_full(which)?;
        let path = match &self.model.trainable {
            Trainable::Full => req
                .output_dir
                .join(file_stem(&req.file_name))
                .join(gen_core::iris::BACKBONE_WEIGHTS_FILE),
            Trainable::Adapter { .. } => req.output_dir.join(&req.file_name),
        };
        if !self.topo.is_main() {
            self.topo.coll.barrier("export")?;
            return Ok(path);
        }
        std::fs::create_dir_all(&req.output_dir)
            .map_err(|e| Error::Msg(format!("iris export: {}: {e}", req.output_dir.display())))?;
        match &self.model.trainable {
            Trainable::Full => {
                let dir = req.output_dir.join(file_stem(&req.file_name));
                let dtype = if self.plan.export_bf16 {
                    Dtype::Bfloat16
                } else {
                    Dtype::Float32
                };
                let mut list: Vec<(String, Array)> = Vec::with_capacity(tensors.len());
                for (k, v) in &tensors {
                    list.push((k.to_string(), v.as_dtype(dtype)?));
                }
                list.sort_by(|a, b| a.0.cmp(&b.0));
                let yaml = export_config_yaml(&self.config, &self.plan.flow, self.adaln_zero_init);
                let meta = full_model_metadata(which, self.step, &self.base_identity);
                // sc-25690: config + weights appear together or not at all.
                publish_export_dir(&dir, |stage: &Path| -> Result<()> {
                    contract::atomic_write(
                        &stage.join(gen_core::iris::BACKBONE_CONFIG_FILE),
                        yaml.as_bytes(),
                    )?;
                    save_safetensors_atomic(
                        &stage.join(gen_core::iris::BACKBONE_WEIGHTS_FILE),
                        &list,
                        &meta,
                    )
                })?;
            }
            Trainable::Adapter { kind, targets, .. } => {
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
                save_safetensors_atomic(&path, &adapter_tensors(*kind, targets, &tensors)?, &map)?;
            }
        }
        if self.world() > 1 {
            self.topo.coll.barrier("export")?;
        }
        Ok(path)
    }

    /// Render the preview prompts from the in-progress state (raw or EMA weights) on the main
    /// rank (every rank takes part in a sharded gather).
    fn render_previews(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pv = self.plan.preview.clone();
        let weights = self.selected_full(pv.weights)?;
        if !self.topo.is_main() {
            return Ok(());
        }
        let dit = self.model.dit(&weights)?;
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
                step: self.micro_step(),
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

/// The gradient-norm group of a trained tensor (`_grad_groups`).
fn block_group(key: &str) -> String {
    if key.starts_with(PROJECTOR_PREFIX) {
        return "repa_projector".into();
    }
    let mut parts = key.split('.');
    let head = parts.next().unwrap_or(key);
    match head {
        "blocks" => format!(
            "blocks.{:02}",
            parts
                .next()
                .and_then(|i| i.parse::<usize>().ok())
                .unwrap_or(0)
        ),
        "pixel_blocks" => format!("pixel_blocks.{}", parts.next().unwrap_or("0")),
        other => other.to_string(),
    }
}

/// What closing a window did.
enum WindowOutcome {
    /// The window held no gradient (every micro-batch dropped).
    Empty,
    /// `lazy_sync`: the clipped gradient norm was non-finite; the window is dropped.
    NonFinite,
    Stepped {
        norm: f64,
        flow: f64,
        repa: Option<f64>,
        /// fp16: the loss scaler skipped the optimizer and scheduler step.
        skipped_by_scaler: bool,
    },
}

/// One gradient-accumulation window of `Trainer.run` (accelerate's `accumulate`): each micro-batch
/// contributes the gradient of `loss / grad_accum`; closing the window clips the global gradient
/// norm, updates the EMA from the **pre-step** weights, then steps the optimizer — upstream's
/// order. The training loop and the parity tests drive this one implementation.
pub struct Window {
    accum: usize,
    grads: Option<Params>,
    loss_sum: f64,
    repa_sum: f64,
    samples: usize,
}

impl Window {
    pub fn new(accum: usize) -> Self {
        Self {
            accum: accum.max(1),
            grads: None,
            loss_sum: 0.0,
            repa_sum: 0.0,
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
        self.add(grads, loss, None, samples)?;
        Ok(loss)
    }

    /// Accumulate one micro-batch's gradients and losses.
    pub fn add(
        &mut self,
        grads: Params,
        flow: f32,
        repa: Option<f32>,
        samples: usize,
    ) -> Result<()> {
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
        self.loss_sum += flow as f64 * samples as f64;
        self.repa_sum += repa.unwrap_or(0.0) as f64 * samples as f64;
        self.samples += samples;
        Ok(())
    }

    /// `optimizer.zero_grad()` + clearing the loss log (the non-finite path).
    pub fn discard(&mut self) {
        self.grads = None;
        self.loss_sum = 0.0;
        self.repa_sum = 0.0;
        self.samples = 0;
    }

    /// The sample-weighted mean loss of the window so far.
    pub fn mean_loss(&self) -> f64 {
        self.loss_sum / self.samples.max(1) as f64
    }

    /// Take the window's gradients and loss sums `(grads, flow_sum, repa_sum, samples)`.
    fn take(&mut self) -> Option<(Params, f64, f64, usize)> {
        let g = self.grads.take()?;
        let out = (g, self.loss_sum, self.repa_sum, self.samples);
        self.discard();
        Some(out)
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
    render_preview_with_order(
        dit,
        steps,
        gen_core::iris::DEFAULT_SOLVER_ORDER,
        cfg_scale,
        shift,
        num_timesteps,
        prediction,
        size,
        cond,
        uncond,
        seed,
    )
}

/// [`render_preview_with`] at solver `order` (`sample.order`; 1 = Euler, ≥ 2 = DPM-Solver++ 2M).
#[allow(clippy::too_many_arguments)]
pub fn render_preview_with_order(
    dit: &IrisDiT,
    steps: usize,
    order: usize,
    cfg_scale: f32,
    shift: f64,
    num_timesteps: usize,
    prediction: Prediction,
    size: usize,
    cond: (&Array, &[i32]),
    uncond: Option<(&Array, &[i32])>,
    seed: u64,
) -> Result<mlx_gen::Image> {
    let plan = gen_core::iris::dpm_solver_plan(steps, order.min(2), shift)?;
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
    render_preview_with_order(
        dit,
        pv.steps,
        plan.preview_order,
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
