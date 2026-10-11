//! Native Candle **Iris-3B generation training** (stories sc-25686, sc-25689) — the CUDA/CPU twin
//! of `mlx_gen_iris::train`: full training from random init or from weights, and LoRA / LoKr
//! adapter training, on the frozen upstream objective — `iris3b/train/trainer.py` semantics over
//! the one backend-neutral contract [`gen_core::iris::train`] (request surface, plan, positional
//! randomness, schedule, data pipeline, routing, artifact schemas, checkpoint layout — shared
//! verbatim with the MLX trainer, never forked), single-device or data-parallel.
//!
//! Per micro-batch (positional draws keyed on `mix_seed(seed, rank, epoch, position)`): walk this
//! rank's slice of the data plan (fixed-square ranged walk, or `bucket` / `area` shape batches),
//! select each sample's caption, encode it with the frozen Qwen3-VL tower (on the fly, or from the
//! memo), substitute the CFG null for dropped rows, sample the timestep indices and the noise, take
//! the rectified-flow loss — plus, under `repa.weight > 0`, the REPA / iREPA alignment to the
//! frozen DINOv2 teacher — and the gradient of `total / grad_accum` (candle autograd over f32
//! `Var` masters; activation-checkpointing regions run off the tape and are replayed on it by a
//! segmented backward). Every `grad_accum` micro-batches: reduce across ranks (DDP mean, or FSDP
//! rows), clip the global norm, update the EMA from the **pre-step** weights, step AdamW or the
//! Muon/AdamW hybrid at `lr · LambdaLR(step)`, advance the scheduler. Non-finite steps drop the
//! window collectively (`nan_loss_tolerance`).
//!
//! Durability: checkpoints are staged and published atomically ([`publish_checkpoint`]) in the
//! MLX twin's layout key for key (consolidated, or per-rank shards under `checkpoint_io: dcp`), so
//! cancel + resume reproduces the uninterrupted run exactly and a consolidated checkpoint reshards
//! to another rank count; retention keeps the newest `keep_last_checkpoints` plus milestones;
//! holdout validation appends to `<checkpoints>/metrics.jsonl`. The run ends by exporting a full
//! model directory or an adapter file that loads through the Candle and the MLX inference routes.
//!
//! [`gen_core::iris::train`]: candle_gen::gen_core::iris::train

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

use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::iris::train::dist::{
    dist_error, Collective, DistError, Partition, Precision, ReduceOp, World,
};
use candle_gen::gen_core::iris::train::holdout::{self as holdout_contract, ValSums};
use candle_gen::gen_core::iris::train::pipeline::DataPipeline;
use candle_gen::gen_core::iris::train::repa::{RepaPlan, PROJECTOR_PREFIX, TEACHER_COMPONENT};
use candle_gen::gen_core::iris::train::shapes::preprocess_file_to_shape;
use candle_gen::gen_core::iris::train::{
    self as contract, adapter_param_route, backbone_identity, batch_seed_rank, caption_seed,
    check_resume, checkpoint_root, checkpoint_staging_dir, dataset_fingerprint, draw_batch,
    export_config_yaml, file_stem, full_model_metadata, full_param_route, latest_checkpoint,
    linear_paths_from_keys, prune_checkpoints, publish_checkpoint, read_checkpoint_state,
    select_caption, AdapterMetadata, ArtifactPlan, CheckpointState, GradScaler, HostRng, InitMode,
    IrisTrainPlan, MixedPrecision, ParamRoute, Prediction, ResumeDataPolicy, TextConditioningMode,
    TrainFlowDefaults, TrainSchedule, WeightsSelect, CKPT_EMA, CKPT_OPTIMIZER, CKPT_STATE,
    CKPT_TRAINABLE, TECHNIQUES, TRAINER_ID,
};
use candle_gen::gen_core::iris::{IrisConfig, TextEncoderConfig, FAMILY, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques, Trainer,
    TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, CancelFlag, Image, LoadSpec, Modality, WeightsSource};
use candle_gen::{CandleError as Error, Result};
use serde_json::{json, Value as JsonValue};

use crate::dit::{IrisDiT, TextBatch};
use crate::nn::safetensors_shapes;
use crate::solver::{cfg_combine, sample};
use crate::text_encoder::IrisTextEncoder;
use dist::{make_collective, to_host, Topology};
use model::{
    adapter_tensors, flow_loss_per_sample, init_adapter, init_projector, loss_and_grads,
    loss_and_grads_ext, provider_dtype, random_init, save_safetensors_atomic, AdapterKind,
    AdapterTarget, RepaTargets, StepBatch, TrainExtras, TrainModel, Trainable,
};
use optim::{
    clip_grads, ema_update, mul_s, save_tensors, snapshot, IrisOptimizer, MuonShards, Params,
    Tensors,
};
use teacher::{spatial_normalize, Teacher};

/// The backend this trainer stamps into its checkpoints.
const BACKEND: &str = "candle";

/// The REPA projector's checkpoint member (the MLX twin's name).
pub const CKPT_REPA: &str = "repa_projector.safetensors";

/// Identity + capabilities of the Candle Iris generation trainer (the MLX trainer's, field for
/// field, on `backend = "candle"`).
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: FAMILY,
        backend: BACKEND,
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: true,
        max_reference_images: 0,
        techniques: TECHNIQUES,
    }
}

/// The resolved resources of a training load: the backbone directory, the Qwen3-VL snapshot and
/// (REPA) the frozen teacher resource.
pub struct IrisTrainer {
    descriptor: TrainerDescriptor,
    backbone_dir: PathBuf,
    text_encoder_dir: PathBuf,
    teacher_dir: Option<PathBuf>,
    config: IrisConfig,
    flow_defaults: TrainFlowDefaults,
    device: Device,
}

/// Construct the trainer from `spec.weights` (the backbone directory),
/// `spec.components["text_encoder"]` and, for REPA runs, `spec.components["repa_teacher"]`, on the
/// build's device ([`candle_gen::default_device`]: `cuda:0` on a CUDA build). Nothing heavy loads
/// here; `train` loads per request.
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
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
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

    /// The resources the plan reads exist; the device can run its precision.
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
        if plan.mixed_precision == MixedPrecision::Fp16 && self.device.is_cpu() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: fp16 mixed precision needs a CUDA device on the Candle \
                 backend (candle's CPU backend has no half-precision GEMM)"
            )));
        }
        if let Some(r) = &plan.repa {
            teacher_path(&self.teacher_dir, r)?;
        }
        Ok(())
    }
}

/// The REPA teacher resource of a run: `repa.teacher_weights` when set, else the `repa_teacher`
/// load component — refused when neither names an existing resource, never fetched.
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

/// A [`Collective`] that records the first typed distributed failure it surfaces.
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

/// One caption's conditioning (`[1, T, L, Dt]` f32 on the training device + its mask).
#[derive(Clone)]
struct Encoded {
    states: Tensor,
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
        Ok(Encoded {
            states: c.states.to_dtype(DType::F32)?,
            mask: c.mask,
        })
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
    /// The trained `Var`s this rank holds: the whole map, or (sharded) its rows — backbone or
    /// adapter factors plus the REPA projector.
    params: Params,
    /// The gathered full parameters of a sharded run (fresh `Var`s), when held.
    full: Option<Params>,
    /// EMA of the core tensors (never the projector), rows under sharding.
    ema: Option<Tensors>,
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
    step: u64,
    sched_step: u64,
    lr: f64,
    nan_count: u32,
    last_loss: f64,
    first_epoch: u32,
    resume_position: Option<JsonValue>,
    log_window: Vec<f64>,
}

fn vars(m: HashMap<String, Tensor>) -> Result<Params> {
    m.into_iter()
        .map(|(k, v)| Ok((k, Var::from_tensor(&v)?)))
        .collect()
}

/// Load a backbone tensor file onto `device`: f32 masters for a full run, or (`adapter_compute`)
/// each tensor straight to the provider dtype.
fn load_source(
    path: &Path,
    adapter_compute: Option<DType>,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let shapes = safetensors_shapes(path)?;
    let vb = candle_gen::mmap_var_builder(&[path.to_path_buf()], DType::F32, device)
        .map_err(|e| Error::Msg(format!("iris training: load {}: {e}", path.display())))?;
    let mut out = HashMap::with_capacity(shapes.len());
    for k in shapes.into_keys() {
        let a = vb.get_unchecked(&k)?;
        let a = match adapter_compute {
            Some(c) => provider_dtype(&k, &a, c)?,
            None => a.to_dtype(DType::F32)?,
        };
        out.insert(k, a);
    }
    Ok(out)
}

/// Every key of `got` must be exactly the architecture's, with the architecture's shapes.
fn check_backbone_keys(
    cfg: &gen_core::iris::ModelConfig,
    got: &HashMap<String, Tensor>,
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
        if a.dims() != shape.as_slice() {
            return Err(Error::Msg(format!(
                "iris training: {what} {k} is {:?}, the config implies {shape:?}",
                a.dims()
            )));
        }
    }
    Ok(())
}

/// Read a checkpoint member onto `device`.
fn load_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    candle_gen::candle_core::safetensors::load(path, device)
        .map_err(|e| Error::Msg(format!("iris checkpoint: read {}: {e}", path.display())))
}

/// The Muon seam of a sharded step.
struct ShardMuon<'a> {
    topo: &'a mut Topology,
}

impl MuonShards for ShardMuon<'_> {
    fn gather(&mut self, key: &str, rows: &Tensor) -> Result<Tensor> {
        self.topo.gather_full(key, rows)
    }
    fn own_rows(&self, _key: &str, full: &Tensor) -> Result<Tensor> {
        self.topo.shard_of(full)
    }
}

/// The checkpoint members of a run.
struct CkptPayload {
    trainable: HashMap<String, Tensor>,
    projector: Option<HashMap<String, Tensor>>,
    ema: Option<HashMap<String, Tensor>>,
    optimizer: HashMap<String, Tensor>,
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

/// Read a checkpoint's members: the consolidated files, or (`dcp`) every shard reassembled.
fn read_checkpoint_members(
    dir: &Path,
    saved: &CheckpointState,
    device: &Device,
) -> Result<CkptPayload> {
    let shard_files = dcp_shards(dir);
    if shard_files.is_empty() {
        let opt_file = |name: &str| -> Result<Option<HashMap<String, Tensor>>> {
            let p = dir.join(name);
            p.is_file().then(|| load_tensors(&p, device)).transpose()
        };
        return Ok(CkptPayload {
            trainable: load_tensors(&dir.join(CKPT_TRAINABLE), device)?,
            projector: opt_file(CKPT_REPA)?,
            ema: opt_file(CKPT_EMA)?,
            optimizer: load_tensors(&dir.join(CKPT_OPTIMIZER), device)?,
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
    let parts: Vec<HashMap<String, Tensor>> = shard_files
        .iter()
        .map(|p| load_tensors(p, device))
        .collect::<Result<_>>()?;
    let mut merged: HashMap<String, Tensor> = HashMap::new();
    let keys: std::collections::BTreeSet<&String> = parts.iter().flat_map(|p| p.keys()).collect();
    for key in keys {
        if key == "o::__t" {
            merged.insert(key.clone(), parts[0][key].clone());
            continue;
        }
        // A shard holding zero rows of a tensor (more shards than rows) writes no entry for it;
        // the reassembled shape is checked against the run's when it is restored.
        let rows: Vec<Tensor> = parts
            .iter()
            .filter_map(|p| p.get(key).cloned())
            .filter(|t| t.rank() == 0 || t.dim(0).unwrap_or(0) > 0)
            .collect();
        merged.insert(key.clone(), Tensor::cat(&rows, 0)?);
    }
    let take = |prefix: &str| -> HashMap<String, Tensor> {
        merged
            .iter()
            .filter_map(|(k, v)| k.strip_prefix(prefix).map(|s| (s.to_string(), v.clone())))
            .collect()
    };
    let trainable = take("p::");
    let projector: HashMap<String, Tensor> = trainable
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

/// What closing a window did.
enum WindowOutcome {
    Empty,
    NonFinite,
    Stepped {
        norm: f64,
        flow: f64,
        repa: Option<f64>,
        skipped_by_scaler: bool,
    },
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

fn mkdir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p)
        .map_err(|e| Error::Msg(format!("iris checkpoint: create {}: {e}", p.display())))
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
        let device = trainer.device.clone();
        let pre = trainer.resolve_plan(req, &[])?;
        config.model.rope_aspect = pre.rope_aspect.clone();
        let compute = match pre.mixed_precision {
            MixedPrecision::Bf16 => DType::BF16,
            MixedPrecision::Fp32 => DType::F32,
            MixedPrecision::Fp16 => DType::F16,
        };
        let coll = match make_collective(&pre.dist.world) {
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
            slot: failure,
        });
        if pre.dist.world.world_size > 1 {
            eprintln!(
                "[world] rank={}/{} local_rank={}/{} node={}/{} transport={} device={:?} | {}",
                pre.dist.world.rank,
                pre.dist.world.world_size,
                pre.dist.world.local_rank,
                pre.dist.world.local_world_size,
                pre.dist.world.node_rank,
                pre.dist.world.num_nodes,
                pre.dist.world.transport,
                device.location(),
                pre.dist.summary()
            );
        }

        on_progress(TrainingProgress::LoadingModel);
        let weights_path = trainer
            .backbone_dir
            .join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
        let adapter_compute = (!req.config.full_finetune).then_some(compute);
        let source: Option<HashMap<String, Tensor>> = match &pre.init {
            InitMode::Weights => Some(load_source(&weights_path, adapter_compute, &device)?),
            InitMode::Random => None,
            InitMode::LoadFrom(p) => {
                let file = if p.join(CKPT_STATE).is_file() {
                    p.join(CKPT_TRAINABLE)
                } else if p.is_dir() {
                    p.join(gen_core::iris::BACKBONE_WEIGHTS_FILE)
                } else {
                    p.clone()
                };
                Some(load_source(&file, adapter_compute, &device)?)
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

        let (model, mut full_params) = match &plan.artifact {
            ArtifactPlan::Full => {
                let params = match source {
                    Some(src) => vars(src)?,
                    None => random_init(&config.model, plan.seed, adaln_zero_init, &device)?,
                };
                (
                    TrainModel {
                        cfg: config.model.clone(),
                        compute,
                        device: device.clone(),
                        trainable: Trainable::Full,
                    },
                    params,
                )
            }
            art => {
                let base = source.ok_or_else(|| {
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
                    let (out_f, in_f) = base[&format!("{p}.weight")].dims2()?;
                    targets.push(AdapterTarget {
                        path: p.clone(),
                        out_f,
                        in_f,
                    });
                }
                let params = init_adapter(kind, &targets, plan.seed, &device)?;
                (
                    TrainModel {
                        cfg: config.model.clone(),
                        compute,
                        device: device.clone(),
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
            full_params.extend(init_projector(r, plan.seed, &device)?);
        }
        let extras = TrainExtras {
            repa: plan.repa.clone(),
            ac: plan.ac,
        };
        let mut topo = Topology {
            plan: plan.dist.clone(),
            coll,
            shapes: full_params
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().dims().to_vec()))
                .collect(),
            device: device.clone(),
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
                return ParamRoute::AdamW;
            }
            match adapter_targets.get(k) {
                Some(target) => adapter_param_route(target, nd, kind),
                None => full_param_route(k, nd, kind, &cfg_model),
            }
        };
        let sharded = plan.dist.sharded();
        let params: Params = if sharded {
            full_params
                .iter()
                .map(|(k, v)| Ok((k.clone(), Var::from_tensor(&topo.shard_of(v.as_tensor())?)?)))
                .collect::<Result<_>>()?
        } else {
            full_params
        };
        let full_rows: HashMap<String, usize> = if sharded {
            topo.shapes
                .iter()
                .filter(|(_, s)| !s.is_empty())
                .map(|(k, s)| (k.clone(), s[0]))
                .collect()
        } else {
            HashMap::new()
        };
        let mut opt = IrisOptimizer::new_sharded(&plan.optimizer, &params, &full_rows, route)?;
        let core_ema = |m: &Params| -> Result<Tensors> {
            m.iter()
                .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
                .map(|(k, v)| Ok((k.clone(), v.as_tensor().detach().copy()?)))
                .collect()
        };
        let mut ema = match plan.ema {
            Some(_) => Some(core_ema(&params)?),
            None => None,
        };

        // ---- text conditioning -------------------------------------------------------------
        let mut te_cfg: TextEncoderConfig = config.text_encoder.clone();
        te_cfg.on_caption_overflow = plan.on_caption_overflow.clone();
        let encoder = IrisTextEncoder::load(&trainer.text_encoder_dir, &te_cfg, &device)?;
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
        }

        let teacher = match &plan.repa {
            Some(r) => Some(Teacher::load(
                &teacher_path(&trainer.teacher_dir, r)?,
                r.teacher.clone(),
                &device,
            )?),
            None => None,
        };

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
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let payload = read_checkpoint_members(&dir, &saved, &device)?;
            let core_keys: Vec<&String> = params
                .keys()
                .filter(|k| !k.starts_with(PROJECTOR_PREFIX))
                .collect();
            if payload.trainable.len() != core_keys.len()
                || core_keys
                    .iter()
                    .any(|k| !payload.trainable.contains_key(*k))
            {
                return Err(Error::Msg(
                    "iris resume: the checkpoint's trainable tensors do not match this run".into(),
                ));
            }
            let set = |var: &Var, full: &Tensor, topo: &Topology| -> Result<()> {
                let t = topo.shard_of(&full.to_dtype(DType::F32)?)?;
                if t.dims() != var.as_tensor().dims() {
                    return Err(Error::Msg(format!(
                        "iris resume: checkpoint tensor is {:?}, this run's is {:?}",
                        t.dims(),
                        var.as_tensor().dims()
                    )));
                }
                if t.elem_count() > 0 {
                    var.set(&t)?;
                }
                Ok(())
            };
            for (k, v) in &payload.trainable {
                set(&params[k], v, &topo)?;
            }
            let projector_carries = plan.repa.as_ref().is_some_and(|r| {
                saved.repa_projector.as_deref() == Some(r.projector_identity().as_str())
            });
            if projector_carries {
                let pj = payload.projector.as_ref().ok_or_else(|| {
                    Error::Msg("iris resume: the checkpoint names a projector it lacks".into())
                })?;
                for (k, v) in pj {
                    set(&params[k], v, &topo)?;
                }
            }
            if let Some(e) = ema.as_mut() {
                let saved_ema = payload.ema.as_ref().ok_or_else(|| {
                    Error::Msg("iris resume: the checkpoint carries no EMA".into())
                })?;
                for (k, v) in e.iter_mut() {
                    let s = saved_ema
                        .get(k)
                        .ok_or_else(|| Error::Msg(format!("iris resume: the EMA lacks {k}")))?;
                    *v = topo.shard_of(&s.to_dtype(DType::F32)?)?;
                }
            }
            let mut slots: HashMap<String, Tensor> = HashMap::new();
            for (name, v) in &payload.optimizer {
                if name == "__t" {
                    slots.insert(name.clone(), v.clone());
                    continue;
                }
                let key = name.split("::").next().unwrap_or(name);
                if key.starts_with(PROJECTOR_PREFIX) && !projector_carries {
                    continue;
                }
                slots.insert(name.clone(), topo.shard_of(v)?);
            }
            if !projector_carries {
                for (name, v) in opt.state_tensors()? {
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
                    None => pipeline.legacy_position(saved.epoch, saved.batches_consumed)?,
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
            params,
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
            log_window: Vec::new(),
        };
        if run.topo.is_main() {
            run.write_metrics_header()?;
        }
        Ok(run)
    }

    fn write_metrics_header(&self) -> Result<()> {
        Ok(holdout_contract::append_metrics(
            &self.ckpt_root,
            &json!({
                "kind": "run",
                "trainer": TRAINER_ID,
                "backend": BACKEND,
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
        )?)
    }

    fn world(&self) -> usize {
        self.plan.dist.world.world_size
    }

    fn any(&mut self, flags: &mut [f64]) -> Result<()> {
        if self.world() > 1 {
            self.topo.reduce_scalars(flags, ReduceOp::Max)?;
        }
        Ok(())
    }

    /// The gathered (unsharded) parameters at `dist.param_dtype` (FSDP2's
    /// `MixedPrecisionPolicy(param_dtype)`; see the MLX twin).
    fn gather_forward(&mut self) -> Result<Tensors> {
        let full = self.topo.gather_map(&snapshot(&self.params))?;
        let dtype = match self.plan.dist.param_dtype {
            Precision::Fp32 => return Ok(full),
            Precision::Bf16 => DType::BF16,
            Precision::Fp16 => DType::F16,
        };
        full.into_iter()
            .map(|(k, v)| Ok((k, v.to_dtype(dtype)?.to_dtype(DType::F32)?)))
            .collect()
    }

    /// The forward's `Var`s: this rank's own (replicated), or the gathered full parameters of a
    /// sharded run (fresh `Var`s; kept for the window unless `reshard_after_forward`, kept across
    /// steps under `grad_op`).
    fn forward_params(&mut self) -> Result<Params> {
        if !self.plan.dist.sharded() {
            return Ok(self.params.clone());
        }
        if let Some(f) = &self.full {
            return Ok(f.clone());
        }
        let full = self.gather_forward()?;
        let full = vars(full.into_iter().collect())?;
        let keep =
            !self.plan.dist.sharding.shards_params() || !self.plan.dist.reshard_after_forward;
        if keep {
            self.full = Some(full.clone());
        }
        Ok(full)
    }

    /// The full current values (detached copies) of `which` core tensors, gathered under sharding.
    fn selected_full(&mut self, which: WeightsSelect) -> Result<Tensors> {
        let src: Tensors = match (which, &self.ema) {
            (WeightsSelect::Ema, Some(e)) => e.clone(),
            _ => self
                .params
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
                .collect(),
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

    fn batch(
        &mut self,
        req: &TrainingRequest,
        epoch: u32,
        position: usize,
        positions: &[usize],
    ) -> Result<(StepBatch, Option<RepaTargets>)> {
        let device = self.model.device.clone();
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
                self.null.clone()
            } else {
                self.text.encode(&caption)?
            };
            states.push(enc.states.to_device(&device)?);
            masks.push(enc.mask);
        }
        let shape4 = (b, 3, hh, ww);
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
            x0: Tensor::from_vec(pixels, shape4, &device)?,
            noise: Tensor::from_vec(draws.noise, shape4, &device)?,
            sigma: Tensor::from_vec(sigmas, (b, 1, 1, 1), &device)?,
            t: Tensor::from_vec(times, b, &device)?,
            states: Tensor::cat(&states, 0)?,
            masks,
        };
        let targets = self.repa_targets(&batch.x0, (hh, ww))?;
        Ok((batch, targets))
    }

    fn repa_targets(&self, x0: &Tensor, size: (usize, usize)) -> Result<Option<RepaTargets>> {
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
        Ok(Some(RepaTargets {
            teacher: t.detach(),
            student_grid,
            teacher_grid,
        }))
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
        while !finished && !cancelled && epoch <= max_epoch {
            let resumed_pass = resume.is_some();
            let skip = self.pipeline.begin_epoch(epoch, resume.take().as_ref())?;
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
                    self.save_checkpoint(boundary.expect("stepped"))?;
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
            if plan.save_every_epochs > 0
                && (epoch.is_multiple_of(plan.save_every_epochs) || Some(epoch) == plan.num_epochs)
                && last_saved != Some(self.step)
            {
                if let Some(b) = boundary {
                    self.save_checkpoint(b)?;
                    last_saved = Some(self.step);
                    on_progress(TrainingProgress::Checkpoint {
                        step: self.micro_step(),
                    });
                }
            }
            epoch += 1;
        }
        if cancelled && self.step == first_step {
            return Err(Error::Canceled);
        }
        if last_saved != Some(self.step) && self.step > 0 {
            if let Some(b) = boundary {
                self.save_checkpoint(b)?;
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

    fn close_window(&mut self, window: &mut Window, lr: f64) -> Result<WindowOutcome> {
        let plan = self.plan.clone();
        let Some((grads, flow_sum, repa_sum, samples)) = window.take() else {
            return Ok(WindowOutcome::Empty);
        };
        let mut logs = [flow_sum, repa_sum, samples as f64];
        if self.world() > 1 {
            self.topo.reduce_scalars(&mut logs, ReduceOp::Sum)?;
        }
        let flow = logs[0] / logs[2].max(1.0);
        let repa = plan.repa.as_ref().map(|_| logs[1] / logs[2].max(1.0));
        let grads: Tensors = grads
            .into_iter()
            .filter(|(k, _)| self.opt.is_trained(k))
            .collect();
        let mut grads = self.topo.reduce_grads(grads)?;
        if let Some(s) = &self.scaler {
            let inv = 1.0 / s.scale;
            for g in grads.values_mut() {
                *g = mul_s(&g.to_dtype(DType::F32)?, inv)?;
            }
        }
        if plan.log_block_grad_norms && self.topo.is_main() {
            self.log_block_norms(&grads)?;
        }
        let (local_norm, _) = clip_grads(grads.clone(), plan.gradient_clip)?;
        let norm = self.topo.global_sq_norm(local_norm * local_norm)?.sqrt();
        if plan.lazy_sync && !norm.is_finite() {
            return Ok(WindowOutcome::NonFinite);
        }
        let found_inf = !norm.is_finite();
        if found_inf && self.scaler.is_none() {
            eprintln!(
                "iris training: non-finite gradient norm at step {} with a finite loss",
                self.step
            );
        }
        let coef = contract::clip_coefficient(plan.gradient_clip, norm);
        for g in grads.values_mut() {
            *g = mul_s(&g.to_dtype(DType::F32)?, coef)?;
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
                self.opt.step(&self.params, &grads, lr)?;
            }
        }
        if let Some(s) = self.scaler.as_mut() {
            s.update(found_inf);
        }
        self.full = None;
        if plan.dist.sharded() && !plan.dist.sharding.shards_params() {
            let full = self.gather_forward()?;
            self.full = Some(vars(full.into_iter().collect())?);
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

    fn log_block_norms(&self, grads: &Tensors) -> Result<()> {
        let mut groups: std::collections::BTreeMap<String, f64> = Default::default();
        for (k, g) in grads {
            let v = to_host(g)?;
            *groups.entry(block_group(k)).or_insert(0.0) +=
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

    fn validate_holdout(&mut self, req: &TrainingRequest) -> Result<()> {
        let plan = self.plan.clone();
        let device = self.model.device.clone();
        let batches = self
            .pipeline
            .holdout_batches(&self.holdout_sizes, plan.holdout.samples)?;
        let weights = self.selected_full(WeightsSelect::Raw)?;
        let dit = self.model.dit(&weights)?;
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
                let mut rng = HostRng::new(caption_seed(0, 0, k));
                let caption =
                    select_caption(item, &plan.caption_field, &plan.caption_fields, &mut rng)?;
                let enc = self.text.encode(&caption)?;
                states.push(enc.states.to_device(&device)?);
                masks.push(enc.mask);
            }
            let shape4 = (b, 3, hh, ww);
            let sig: Vec<f32> = t_idx.iter().map(|&t| self.schedule.sigmas[t]).collect();
            let tim: Vec<f32> = t_idx
                .iter()
                .map(|&t| self.schedule.model_times[t])
                .collect();
            let batch = StepBatch {
                x0: Tensor::from_vec(pixels, shape4, &device)?,
                noise: Tensor::from_vec(noise, shape4, &device)?,
                sigma: Tensor::from_vec(sig, (b, 1, 1, 1), &device)?,
                t: Tensor::from_vec(tim, b, &device)?,
                states: Tensor::cat(&states, 0)?,
                masks,
            };
            let losses = to_host(&flow_loss_per_sample(&dit, &batch, &plan.flow)?)?;
            for (l, t) in losses.iter().zip(&t_idx) {
                sums.add(*l as f64, *t, n);
            }
        }
        drop(dit);
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

    fn micro_step(&self) -> u32 {
        (self.step * self.plan.grad_accum as u64) as u32
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
            backend: BACKEND.into(),
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

    /// Publish a checkpoint at an optimizer-step boundary, collectively (see the MLX twin).
    fn save_checkpoint(&mut self, b: (u32, usize, usize, usize)) -> Result<()> {
        let position = self.pipeline.data_position(b.0, b.1, b.2, b.3)?;
        let state = self.state(position, (b.0, b.1));
        let io = self.plan.dist.checkpoint_io;
        let world = self.plan.dist.world.clone();
        let state_json = serde_json::to_string_pretty(&state.to_json())
            .map_err(|e| Error::Msg(e.to_string()))?;
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
                let mut list: Vec<(String, Tensor)> = Vec::new();
                for (k, v) in &self.params {
                    list.push((format!("p::{k}"), v.as_tensor().detach()));
                }
                if let Some(e) = &self.ema {
                    for (k, v) in e {
                        list.push((format!("e::{k}"), v.clone()));
                    }
                }
                for (k, v) in self.opt.state_tensors()? {
                    list.push((format!("o::{k}"), v));
                }
                // Zero-row shards (more shards than rows) have nothing to write.
                list.retain(|(_, t)| t.elem_count() > 0);
                let path = staging.join(format!(
                    "shard_{:05}.safetensors",
                    self.plan.dist.shard_index()
                ));
                save_tensors(&path, &list)?;
            }
            self.topo.coll.barrier("dcp shards")?;
            if world.rank == 0 {
                contract::atomic_write(&staging.join(CKPT_STATE), state_json.as_bytes())?;
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
        let raw: Tensors = self
            .params
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
            .collect();
        let params = if self.plan.dist.sharded() {
            self.topo.gather_map(&raw)?
        } else {
            raw
        };
        let ema = match &self.ema {
            Some(e) if self.plan.dist.sharded() => Some(self.topo.gather_map(e)?),
            other => other.clone(),
        };
        let mut opt_state: Vec<(String, Tensor)> = Vec::new();
        for (name, v) in self.opt.state_tensors()? {
            if name == "__t" || !self.plan.dist.sharded() {
                opt_state.push((name, v));
                continue;
            }
            let key = name.split("::").next().unwrap_or(&name).to_string();
            let full = self.topo.gather_full(&key, &v)?;
            opt_state.push((name, full));
        }
        if io.writes(&world) {
            mkdir(&self.ckpt_root)?;
            let mut staging = checkpoint_staging_dir(&self.ckpt_root, self.step);
            if io == gen_core::iris::train::dist::CheckpointIo::PerNode {
                staging = staging.with_extension(format!("n{}", world.node_rank));
            }
            if staging.exists() {
                let _ = std::fs::remove_dir_all(&staging);
            }
            mkdir(&staging)?;
            let list = |pred: &dyn Fn(&str) -> bool| -> Vec<(String, Tensor)> {
                params
                    .iter()
                    .filter(|(k, _)| pred(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            };
            save_tensors(
                &staging.join(CKPT_TRAINABLE),
                &list(&|k| !k.starts_with(PROJECTOR_PREFIX)),
            )?;
            let projector = list(&|k| k.starts_with(PROJECTOR_PREFIX));
            if !projector.is_empty() {
                save_tensors(&staging.join(CKPT_REPA), &projector)?;
            }
            if let Some(e) = &ema {
                let l: Vec<(String, Tensor)> =
                    e.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                save_tensors(&staging.join(CKPT_EMA), &l)?;
            }
            save_tensors(&staging.join(CKPT_OPTIMIZER), &opt_state)?;
            contract::atomic_write(&staging.join(CKPT_STATE), state_json.as_bytes())?;
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

    /// Write the final artifact; the main rank writes, every rank returns the same path.
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
                std::fs::create_dir_all(&dir)
                    .map_err(|e| Error::Msg(format!("iris export: {}: {e}", dir.display())))?;
                let dtype = if self.plan.export_bf16 {
                    DType::BF16
                } else {
                    DType::F32
                };
                let mut list: Vec<(String, Tensor)> = Vec::with_capacity(tensors.len());
                for (k, v) in &tensors {
                    list.push((k.clone(), v.to_dtype(dtype)?));
                }
                contract::atomic_write(
                    &dir.join(gen_core::iris::BACKBONE_CONFIG_FILE),
                    export_config_yaml(&self.config, &self.plan.flow, self.adaln_zero_init)
                        .as_bytes(),
                )?;
                save_safetensors_atomic(
                    &path,
                    &list,
                    &full_model_metadata(which, self.step, &self.base_identity),
                )?;
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
        Ok(())
    }
}

/// One gradient-accumulation window of `Trainer.run` (accelerate's `accumulate`).
pub struct Window {
    accum: usize,
    grads: Option<Tensors>,
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

    /// One micro-batch of `samples` rows: returns its (unscaled) loss.
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
        grads: Tensors,
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

    pub fn discard(&mut self) {
        self.grads = None;
        self.loss_sum = 0.0;
        self.repa_sum = 0.0;
        self.samples = 0;
    }

    pub fn mean_loss(&self) -> f64 {
        self.loss_sum / self.samples.max(1) as f64
    }

    fn take(&mut self) -> Option<(Tensors, f64, f64, usize)> {
        let g = self.grads.take()?;
        let out = (g, self.loss_sum, self.repa_sum, self.samples);
        self.discard();
        Some(out)
    }

    /// Close the window at learning rate `lr`: clip → EMA → step.
    pub fn update(
        &mut self,
        params: &Params,
        ema: Option<(&mut Tensors, f64)>,
        opt: &mut IrisOptimizer,
        clip: f64,
        lr: f64,
    ) -> Result<Option<f64>> {
        let Some(grads) = self.grads.take() else {
            self.discard();
            return Ok(None);
        };
        self.discard();
        let grads: Tensors = grads
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

/// One preview image (release sampler, order 2); see [`render_preview_with_order`].
#[allow(clippy::too_many_arguments)]
pub fn render_preview_with(
    dit: &IrisDiT,
    steps: usize,
    cfg_scale: f32,
    shift: f64,
    num_timesteps: usize,
    prediction: Prediction,
    size: usize,
    cond: (&Tensor, &[i32]),
    uncond: Option<(&Tensor, &[i32])>,
    seed: u64,
) -> Result<Image> {
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

/// One preview image at solver `order` (`sample.order`; 1 = Euler, ≥ 2 = DPM-Solver++ 2M) from
/// `seed`'s noise (the provider's launch-portable CPU `StdRng`), through the provider's
/// prediction-aware solver.
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
    cond: (&Tensor, &[i32]),
    uncond: Option<(&Tensor, &[i32])>,
    seed: u64,
) -> Result<Image> {
    let plan = gen_core::iris::dpm_solver_plan(steps, order.min(2), shift)?;
    let device = dit.device().clone();
    let channels = dit.config().in_channels;
    let z = crate::pipeline::noise(seed, channels, size as u32, size as u32, &device)?;
    let cond_states = cond.0.to_device(&device)?;
    let cond_mask = vec![cond.1.to_vec()];
    let cond_batch = TextBatch {
        states: &cond_states,
        mask: &cond_mask,
    };
    let cfg_states;
    let cfg_mask;
    let cfg_batch = match uncond {
        Some((u, um)) if cfg_scale != 1.0 => {
            cfg_states = Tensor::cat(&[&u.to_device(&device)?, &cond_states], 0)?;
            cfg_mask = vec![um.to_vec(), cond.1.to_vec()];
            Some(TextBatch {
                states: &cfg_states,
                mask: &cfg_mask,
            })
        }
        _ => None,
    };
    let never = CancelFlag::default();
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
                    let xb = Tensor::cat(&[x, x], 0)?;
                    let tb = Tensor::new(&[t, t], &device)?;
                    let out = dit.forward(&xb, &tb, batch)?;
                    cfg_combine(&out.narrow(0, 0, 1)?, &out.narrow(0, 1, 1)?, cfg_scale)
                }
                _ => dit.forward(x, &Tensor::new(&[t], &device)?, &cond_batch),
            }
        },
        |_, _| {},
    )?;
    crate::pipeline::to_image(&x.clamp(-1f32, 1f32)?)
}

fn render_preview(
    dit: &IrisDiT,
    plan: &IrisTrainPlan,
    cond: &Encoded,
    negative: Option<&Encoded>,
    seed: u64,
) -> Result<Image> {
    let pv = &plan.preview;
    render_preview_with_order(
        dit,
        pv.steps,
        plan.preview_order,
        pv.cfg_scale,
        plan.flow.shift,
        plan.flow.num_train_timesteps,
        plan.flow.prediction,
        pv.size,
        (&cond.states, &cond.mask),
        negative.map(|n| (&n.states, n.mask.as_slice())),
        seed,
    )
}
