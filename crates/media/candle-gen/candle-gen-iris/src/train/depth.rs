//! Native Candle **Iris-3B depth training** (story sc-25687) — the twin of
//! `mlx_gen_iris::train::depth` over the one backend-neutral contract
//! [`gen_core_iris::train::depth`]: the supervised direct-regression fine-tune of the depth task
//! (paper §5.2), registered as the [`DEPTH_TRAINER_ID`] (`iris_3b_depth`) trainer in the Candle
//! catalog.
//!
//! Every step runs **the inference forward** ([`crate::depth::depth_forward`]) built from the f32
//! `Var` masters through [`IrisDiT::from_checkpoint_widened`] (an in-memory [`Checkpoint`], cast to
//! the compute dtype as the estimator casts a loaded export, the casts on the autograd tape), the
//! reducer as the estimator's `[1, C]` [`Linear`]. The loss is Eq. 5 on tensors (the contract's
//! host reference is `depth_loss_host`), its gradient of `loss / grad_accum`; each optimizer step
//! clips, updates the optional EMA from the pre-step weights and steps AdamW / Muon — the
//! generation trainer's optimizer and checkpoint code ([`super::optim`], `gen_core_iris::train`).
//! Checkpoints use the MLX twin's layout and identities, so a run resumes across backends.
//!
//! Candle's `abs` backward is `+1` at exactly 0 where torch's is 0 — reachable only when a valid
//! pixel's error (or a valid pair's difference error) is exactly zero, which a continuous
//! regression does not produce; masked pixels are multiplied by 0 either way.
//!
//! [`gen_core_iris::train::depth`]: gen_core_iris::train::depth

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris::depth::{colorize_inferno, DepthMap, DEPTH_EXTRA_INPUT_CHANNELS};
use gen_core_iris::downstream::{parse_task_settings, EmptyPrompt, TaskExport, EMPTY_PROMPT_FILE};
use gen_core_iris::train::ac::AcPolicy;
use gen_core_iris::train::depth::{
    assemble_batch, check_depth_resume, depth_dataset_fingerprint, depth_dataset_pass,
    depth_export_config_yaml, depth_export_metadata, depth_tensor_shapes, preview_row,
    reducer_init_values, validate_depth_items, widened_projections, DepthBatch, DepthInit,
    DepthLossPlan, DepthTrainPlan, DEPTH_TRAINER_ID, EMPTY_PROMPT_COMPONENT,
    EXPORT_BACKBONE_PREFIX, REDUCER_BIAS, REDUCER_WEIGHT,
};
use gen_core_iris::train::lineage::{publish_export_dir, retire_foreign_checkpoints};
use gen_core_iris::{
    train as contract,
    train::{
        backbone_identity, checkpoint_root, checkpoint_staging_dir, file_stem, full_param_route,
        latest_checkpoint, lr_factor, prune_checkpoints, publish_checkpoint, read_checkpoint_state,
        CheckpointState, DataWalk, MixedPrecision, ResumeDataPolicy, WeightsSelect, CKPT_EMA,
        CKPT_OPTIMIZER, CKPT_STATE, CKPT_TRAINABLE, TECHNIQUES,
    },
};
use gen_core_iris::{
    IrisConfig, IrisTask, BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE, FAMILY,
    TEXT_ENCODER_COMPONENT,
};
use gen_core_train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques,
};

use super::model::{grads_under_ac, save_safetensors_atomic};
use super::optim::{
    clip_grads, detached_copy, ema_update, mul_s, save_tensors, snapshot, IrisOptimizer, Params,
    Tensors,
};
use crate::depth::{depth_forward_hooked, DepthParts};
use crate::dit::{ForwardHooks, IrisDiT, NoHooks};
use crate::nn::{Checkpoint, Linear};
use crate::text_encoder::IrisTextEncoder;

const BACKEND: &str = "candle";

/// Identity + capabilities of the Candle Iris depth trainer: a full fine-tune, no adapters.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: DEPTH_TRAINER_ID,
        family: FAMILY,
        backend: BACKEND,
        modality: Modality::Image,
        supports_lora: false,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: true,
        max_reference_images: 0,
        techniques: TECHNIQUES,
    }
}

/// Where a run's empty-prompt conditioning comes from.
#[derive(Clone, Debug)]
enum PromptSource {
    Export(PathBuf),
    File(PathBuf),
    Encoder(PathBuf),
}

/// The resolved resources of a depth-training load.
pub struct IrisDepthTrainer {
    descriptor: TrainerDescriptor,
    source_dir: PathBuf,
    init: DepthInit,
    config: IrisConfig,
    prompt: PromptSource,
    device: Device,
}

/// Construct the depth trainer on the build's device: `spec.weights` is a `depth/` export (no other
/// resource) or the generation backbone plus its empty-prompt conditioning (the
/// [`EMPTY_PROMPT_COMPONENT`] file, or the `text_encoder` component to encode it once).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let (source_dir, init, config, prompt) = resolve_resources(spec)?;
    Ok(Box::new(IrisDepthTrainer {
        descriptor: trainer_descriptor(),
        source_dir,
        init,
        config,
        prompt,
        device: candle_gen::default_device()?,
    }))
}

fn resolve_resources(spec: &LoadSpec) -> Result<(PathBuf, DepthInit, IrisConfig, PromptSource)> {
    let id = DEPTH_TRAINER_ID;
    refuse_trainer_load_overlays(id, spec)?;
    for (field, set) in [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("pid", spec.pid.is_some()),
    ] {
        if set {
            return Err(Error::Unsupported(format!(
                "{id} trainer: LoadSpec::{field} is not supported"
            )));
        }
    }
    gen_core::control::reject_unknown_components(
        spec,
        &[TEXT_ENCODER_COMPONENT, EMPTY_PROMPT_COMPONENT],
        id,
    )?;
    let dir = match &spec.weights {
        WeightsSource::Dir(d) => d.clone(),
        WeightsSource::File(f) => {
            return Err(Error::Msg(format!(
                "{id} trainer: the resource must be a depth export or generation backbone \
                 directory, not the file {}",
                f.display()
            )))
        }
    };
    let config_path = dir.join(BACKBONE_CONFIG_FILE);
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Msg(format!("{id} trainer: read {}: {e}", config_path.display())))?;
    if parse_task_settings(&text)?.is_some() {
        let export = TaskExport::from_dir(&dir, IrisTask::Depth, id)?;
        if let Some(key) = spec.components.keys().next() {
            return Err(Error::Unsupported(format!(
                "{id} trainer: a depth export carries its own {EMPTY_PROMPT_FILE}; drop the '{key}' \
                 component (the depth task never loads a text encoder)"
            )));
        }
        let prompt = PromptSource::Export(export.empty_prompt_path());
        return Ok((dir, DepthInit::Export, export.config, prompt));
    }
    let config = IrisConfig::parse(&text)?;
    config.validate_supported()?;
    if !dir.join(BACKBONE_WEIGHTS_FILE).is_file() {
        return Err(Error::Msg(format!(
            "{id} trainer: the generation backbone {} has no {BACKBONE_WEIGHTS_FILE}",
            dir.display()
        )));
    }
    let prompt = match (
        spec.components.get(EMPTY_PROMPT_COMPONENT),
        spec.components.get(TEXT_ENCODER_COMPONENT),
    ) {
        (Some(WeightsSource::File(f)), None) => PromptSource::File(f.clone()),
        (None, Some(WeightsSource::Dir(d))) => PromptSource::Encoder(d.clone()),
        (Some(_), Some(_)) => {
            return Err(Error::Msg(format!(
                "{id} trainer: stage the empty-prompt conditioning once — either the \
                 '{EMPTY_PROMPT_COMPONENT}' file or the '{TEXT_ENCODER_COMPONENT}' snapshot"
            )))
        }
        _ => {
            return Err(Error::Msg(format!(
                "{id} trainer: a run from the generation backbone needs its empty-prompt \
                 conditioning — the '{EMPTY_PROMPT_COMPONENT}' component ({EMPTY_PROMPT_FILE} \
                 file) or the '{TEXT_ENCODER_COMPONENT}' component (Qwen3-VL snapshot directory)"
            )))
        }
    };
    candle_gen::gen_core::__model_input_result!(
        Ok((dir, DepthInit::Parent, config, prompt)),
        "candle-gen-iris::train::depth::resolve_resources/generation-configuration",
        Configuration,
        &[&config_path]
    )
}

candle_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

impl IrisDepthTrainer {
    fn floors(&self, req: &TrainingRequest) -> Result<DepthTrainPlan> {
        let d = &self.descriptor;
        validate_training_techniques(d, req)?;
        validate_control_request(d, req)?;
        validate_full_finetune_request(d, req)?;
        validate_edit_request(d, req)?;
        refuse_reference_control_model_options(DEPTH_TRAINER_ID, req)?;
        let plan = DepthTrainPlan::resolve(req, &self.config)?;
        if req.items.is_empty() {
            return Err(Error::Msg(format!(
                "{DEPTH_TRAINER_ID} trainer: the dataset is empty"
            )));
        }
        validate_depth_items(req)?;
        Ok(plan)
    }
}

impl Trainer for IrisDepthTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.floors(req)?;
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        let plan = self.floors(req)?;
        let run = Run::prepare(self, plan, req, on_progress)?;
        run.execute(req, on_progress).map_err(Into::into)
    }
}

// =============================================================================================
// The model
// =============================================================================================

/// The trainable depth model: the frozen empty-prompt conditioning, the compute dtype, the device.
pub struct DepthModel {
    pub geometry: crate::dit::TrainingGeometry,
    pub config: IrisConfig,
    pub compute: DType,
    pub device: Device,
    /// `[1, T, L, D]` f32 on the device and its `[T]` mask.
    pub embeddings: Tensor,
    pub mask: Vec<i32>,
    /// Activation checkpointing of the training forward ([`loss_and_grads`]); none by default.
    pub ac: AcPolicy,
}

impl DepthModel {
    pub fn new(
        config: IrisConfig,
        compute: DType,
        prompt: &EmptyPrompt,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            geometry: Default::default(),
            config,
            compute,
            device: device.clone(),
            embeddings: Tensor::from_vec(prompt.embeddings.clone(), prompt.shape.to_vec(), device)?,
            mask: prompt.mask.clone(),
            ac: AcPolicy::None,
        })
    }

    /// `[B, 3, H, W]` RGB → `[B, 1, H, W]` f32 prediction from `tensors` (`Var`-backed snapshots
    /// under autograd, or plain values), through the estimator's own forward.
    pub fn predict(&self, tensors: &Tensors, rgb: &Tensor) -> Result<Tensor> {
        let (dit, _) = self.backbones(tensors, false)?;
        self.predict_with(&dit, tensors, rgb, &mut NoHooks)
    }

    /// The widened backbone built from `tensors` and, when asked for, the same backbone off the
    /// tape holding the first one's casts (an activation-checkpointing region's forward).
    fn backbones(&self, tensors: &Tensors, tape_free: bool) -> Result<(IrisDiT, Option<IrisDiT>)> {
        let cfg = &self.config.model;
        let backbone: HashMap<String, Tensor> = tensors
            .iter()
            .filter(|(k, _)| k.as_str() != REDUCER_WEIGHT && k.as_str() != REDUCER_BIAS)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let channels = cfg.in_channels + DEPTH_EXTRA_INPUT_CHANNELS;
        let build = |ckpt: &Checkpoint| {
            IrisDiT::from_checkpoint_widened(ckpt, cfg, self.compute, &self.device, channels)
                .map(|dit| dit.with_training_geometry(&self.geometry))
        };
        let ckpt = Checkpoint::from_tensors(backbone, &self.device);
        let dit = build(&ckpt)?;
        let free = if tape_free {
            let shared = ckpt
                .loaded()
                .into_iter()
                .map(|(k, v)| (k, v.detach()))
                .collect();
            Some(build(&Checkpoint::from_tensors(shared, &self.device))?)
        } else {
            None
        };
        Ok((dit, free))
    }

    /// The estimator's forward through an already-built backbone, `hooks` around its blocks.
    fn predict_with(
        &self,
        dit: &IrisDiT,
        tensors: &Tensors,
        rgb: &Tensor,
        hooks: &mut dyn ForwardHooks,
    ) -> Result<Tensor> {
        let cfg = &self.config.model;
        let get = |k: &str| {
            tensors
                .get(k)
                .ok_or_else(|| Error::Msg(format!("iris depth: {k} is not trained")))
        };
        let reducer = Linear::from_parts(
            get(REDUCER_WEIGHT)?
                .reshape((1, cfg.in_channels))?
                .to_dtype(self.compute)?,
            Some(get(REDUCER_BIAS)?.to_dtype(self.compute)?),
        );
        depth_forward_hooked(
            &DepthParts {
                dit,
                reducer: &reducer,
                embeddings: &self.embeddings,
                mask: &self.mask,
                num_train_timesteps: self.config.flow.num_train_timesteps,
            },
            rgb,
            hooks,
        )
    }
}

/// One micro-batch on the device.
pub struct DepthStep {
    pub rgb: Tensor,
    pub target: Tensor,
    pub valid: Tensor,
    /// `[B]` f32 row weights ([`DepthBatch::row_weights`]).
    pub row_weights: Tensor,
    pub rows: usize,
}

impl DepthStep {
    pub fn new(b: &DepthBatch, loss: &DepthLossPlan, device: &Device) -> Result<Self> {
        let (n, h, w) = (b.batch, b.height, b.width);
        Ok(Self {
            rgb: Tensor::from_slice(&b.rgb, (n, 3, h, w), device)?,
            target: Tensor::from_slice(&b.target, (n, 1, h, w), device)?,
            valid: Tensor::from_slice(&b.valid, (n, 1, h, w), device)?,
            row_weights: Tensor::from_vec(b.row_weights(loss.reduction), n, device)?,
            rows: n,
        })
    }
}

/// Eq. 5 on tensors: per row `Σ_{p∈M} |e| + λ(Σ |Δx e| + Σ |Δy e|)` with `e = d̂ − d` and the
/// both-valid pair masks, weighted by the row weights and summed (a scalar f32 tensor).
pub fn depth_loss(pred: &Tensor, step: &DepthStep, gradient_weight: f64) -> Result<Tensor> {
    let e = pred.sub(&step.target)?;
    let v = &step.valid;
    let (_, _, h, w) = e.dims4()?;
    let rows = |t: Tensor| -> Result<Tensor> { Ok(t.sum((1, 2, 3))?) };
    let value = rows(e.abs()?.mul(v)?)?;
    let diff = |t: &Tensor, dim: usize, len: usize| -> Result<Tensor> {
        Ok(t.narrow(dim, 1, len - 1)?
            .sub(&t.narrow(dim, 0, len - 1)?)?)
    };
    let pair = |t: &Tensor, dim: usize, len: usize| -> Result<Tensor> {
        Ok(t.narrow(dim, 1, len - 1)?
            .mul(&t.narrow(dim, 0, len - 1)?)?)
    };
    let gx = rows(diff(&e, 3, w)?.abs()?.mul(&pair(v, 3, w)?)?)?;
    let gy = rows(diff(&e, 2, h)?.abs()?.mul(&pair(v, 2, h)?)?)?;
    let per_row = value.add(&mul_s(&gx.add(&gy)?, gradient_weight)?)?;
    Ok(per_row.mul(&step.row_weights)?.sum_all()?)
}

/// The loss of one micro-batch and its gradient over every trained tensor (zero for a tensor the
/// graph never reaches); the gradient is of `loss · grad_scale`, the returned loss unscaled.
pub fn loss_and_grads(
    model: &DepthModel,
    params: &Params,
    step: &DepthStep,
    loss: &DepthLossPlan,
    grad_scale: f32,
) -> Result<(f32, Tensors)> {
    let snap = snapshot(params);
    let (dit, free) = model.backbones(&snap, model.ac.active())?;
    let mut unscaled = None;
    let grads = grads_under_ac(&dit, free.as_ref(), model.ac, params, |hooks| {
        let pred = model.predict_with(&dit, &snap, &step.rgb, hooks)?;
        let l = depth_loss(&pred, step, loss.gradient_weight)?;
        let scaled = mul_s(&l, grad_scale as f64)?;
        unscaled = Some(l.to_dtype(DType::F32)?.to_vec0::<f32>()?);
        Ok(scaled)
    })?;
    Ok((unscaled.expect("the loss closure ran"), grads))
}

/// One accumulation window (the generation trainer's `Window` over the depth loss).
pub struct DepthWindow {
    accum: usize,
    grads: Option<Tensors>,
    loss_sum: f64,
    samples: usize,
}

impl DepthWindow {
    pub fn new(accum: usize) -> Self {
        Self {
            accum: accum.max(1),
            grads: None,
            loss_sum: 0.0,
            samples: 0,
        }
    }

    /// One micro-batch: returns its (unscaled) loss; a non-finite loss leaves the window untouched.
    pub fn micro(
        &mut self,
        model: &DepthModel,
        params: &Params,
        step: &DepthStep,
        loss: &DepthLossPlan,
    ) -> Result<f32> {
        let (l, grads) = loss_and_grads(model, params, step, loss, 1.0 / self.accum as f32)?;
        if !l.is_finite() {
            return Ok(l);
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
        self.loss_sum += l as f64 * step.rows as f64;
        self.samples += step.rows;
        Ok(l)
    }

    pub fn discard(&mut self) {
        self.grads = None;
        self.loss_sum = 0.0;
        self.samples = 0;
    }

    pub fn mean_loss(&self) -> f64 {
        self.loss_sum / self.samples.max(1) as f64
    }

    /// Close the window: clip → EMA (pre-step weights) → step. Returns the pre-clip norm.
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
        let cohort_loss = self.mean_loss() as f32;
        self.discard();
        let grads: Tensors = grads
            .into_iter()
            .filter(|(k, _)| opt.is_trained(k))
            .collect();
        let (norm, grads) = clip_grads(grads, clip)?;
        if let Some((ema, decay)) = ema {
            ema_update(ema, params, decay)?;
        }
        let cohort_vars = super::cohort_before_named_update(params, &grads, opt)?;
        opt.step(params, &grads, lr)?;
        candle_gen::train::profile::cohort_after_update();
        if let Some(var) = cohort_vars.first() {
            candle_gen::train::profile::cohort_end_step(
                var.device(),
                cohort_loss,
                &cohort_vars,
                "iris-depth",
            )?;
        }
        Ok(Some(norm))
    }
}

// =============================================================================================
// Init and loading
// =============================================================================================

/// Every tensor of a safetensors file as f32 on `device`.
fn load_f32(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let raw = candle_gen::candle_core::safetensors::load(path, device)
        .map_err(|e| Error::Msg(format!("iris depth training: load {}: {e}", path.display())))?;
    raw.into_iter()
        .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
        .collect()
}

/// The trained tensors of a `depth/` export (backbone keys without the `pixel.` prefix).
pub fn load_export_tensors(dir: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let raw = load_f32(&dir.join(BACKBONE_WEIGHTS_FILE), device)?;
    let mut out = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let key = match k.strip_prefix(EXPORT_BACKBONE_PREFIX) {
            Some(name) => name.to_string(),
            None if k == REDUCER_WEIGHT || k == REDUCER_BIAS => k,
            None => {
                return Err(Error::Msg(format!(
                "iris depth training: the export {} carries `{k}`, neither a `pixel.*` backbone \
                     key nor the depth reducer",
                dir.display()
            )))
            }
        };
        out.insert(key, v);
    }
    Ok(out)
}

fn check_depth_keys(cfg: &IrisConfig, got: &HashMap<String, Tensor>, what: &str) -> Result<()> {
    let want = depth_tensor_shapes(&cfg.model);
    if want.len() != got.len() {
        return Err(Error::Msg(format!(
            "iris depth training: {what} carries {} tensors, the depth model {}",
            got.len(),
            want.len()
        )));
    }
    for (k, shape) in want {
        let a = got
            .get(&k)
            .ok_or_else(|| Error::Msg(format!("iris depth training: {what} lacks {k}")))?;
        if a.dims() != shape.as_slice() {
            return Err(Error::Msg(format!(
                "iris depth training: {what} {k} is {:?}, the depth model's is {shape:?}",
                a.dims()
            )));
        }
    }
    Ok(())
}

/// A parent init: the generation backbone with its input projections widened by zero columns for
/// the always-zero 4th channel, plus the reducer.
pub fn parent_tensors(
    dir: &Path,
    cfg: &IrisConfig,
    plan: &DepthTrainPlan,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut t = load_f32(&dir.join(BACKBONE_WEIGHTS_FILE), device)?;
    let want = contract::backbone_tensor_shapes(&cfg.model);
    if want.len() != t.len() || want.iter().any(|(k, _)| !t.contains_key(k)) {
        return Err(Error::Msg(format!(
            "iris depth training: {} is not the configured generation backbone",
            dir.display()
        )));
    }
    for (key, cols) in widened_projections(&cfg.model) {
        let w = t.remove(key).expect("checked");
        let rows = w.dim(0)?;
        let z = Tensor::zeros((rows, cols), DType::F32, device)?;
        t.insert(key.to_string(), Tensor::cat(&[&w, &z], 1)?);
    }
    let c = cfg.model.in_channels;
    let (w, b) = reducer_init_values(plan.reducer_init, c, plan.seed);
    t.insert(
        REDUCER_WEIGHT.into(),
        Tensor::from_vec(w, (1, c, 1, 1), device)?,
    );
    t.insert(REDUCER_BIAS.into(), Tensor::from_vec(b, 1, device)?);
    Ok(t)
}

fn vars(m: HashMap<String, Tensor>) -> Result<Params> {
    m.into_iter()
        .map(|(k, v)| Ok((k, Var::from_tensor(&v.to_dtype(DType::F32)?)?)))
        .collect()
}

fn read_prompt(source: &PromptSource, config: &IrisConfig, device: &Device) -> Result<EmptyPrompt> {
    match source {
        PromptSource::Export(p) | PromptSource::File(p) => Ok(EmptyPrompt::read(p, config)?),
        PromptSource::Encoder(dir) => {
            let te = IrisTextEncoder::load(dir, &config.text_encoder, device)?;
            let window = te.window("")?;
            let c = te.encode_window(&window)?;
            let states = c.states.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
            let (a, b, l, d) = states.dims4()?;
            Ok(EmptyPrompt {
                embeddings: states.flatten_all()?.to_vec1::<f32>()?,
                shape: [a, b, l, d],
                mask: c.mask,
            })
        }
    }
}

// =============================================================================================
// The run
// =============================================================================================

struct Run {
    plan: DepthTrainPlan,
    source_config_text: String,
    model: DepthModel,
    prompt: EmptyPrompt,
    params: Params,
    ema: Option<Tensors>,
    opt: IrisOptimizer,
    kept: Vec<usize>,
    walk: DataWalk,
    total_steps: u64,
    ckpt_root: PathBuf,
    base_identity: String,
    dataset_fingerprint: String,
    step: u64,
    sched_step: u64,
    lr: f64,
    nan_count: u32,
    epoch: u32,
    pos: usize,
    last_loss: f64,
}

fn load_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    candle_gen::candle_core::safetensors::load(path, device)
        .map_err(|e| Error::Msg(format!("iris checkpoint: read {}: {e}", path.display())))
}

impl Run {
    fn prepare(
        trainer: &IrisDepthTrainer,
        plan: DepthTrainPlan,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let device = trainer.device.clone();
        let config = trainer.config.clone();
        let compute = match plan.mixed_precision {
            MixedPrecision::Bf16 => DType::BF16,
            MixedPrecision::Fp32 => DType::F32,
            // Only the generation plan resolves fp16 (sc-25689); this task's plan never does.
            MixedPrecision::Fp16 => {
                return Err(Error::Unsupported(
                    "iris training: fp16 mixed precision is the generation trainer's".into(),
                ))
            }
        };
        let total = req.items.len() as u32;
        let kept = depth_dataset_pass(req, &plan, |i| {
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            })
        })?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        on_progress(TrainingProgress::LoadingModel);
        let source_config_text =
            std::fs::read_to_string(trainer.source_dir.join(BACKBONE_CONFIG_FILE))
                .map_err(|e| Error::Msg(format!("iris depth training: read config.yaml: {e}")))?;
        let tensors = match (&plan.load_from, &trainer.init) {
            (Some(p), _) => {
                if p.join(CKPT_STATE).is_file() {
                    let state = read_checkpoint_state(p)?;
                    if state.task != IrisTask::Depth {
                        return Err(Error::Msg(format!(
                            "iris depth training: load_from {} is a {} checkpoint",
                            p.display(),
                            state.task.name()
                        )));
                    }
                    load_f32(&p.join(CKPT_TRAINABLE), &device)?
                } else {
                    TaskExport::from_dir(p, IrisTask::Depth, DEPTH_TRAINER_ID)?;
                    load_export_tensors(p, &device)?
                }
            }
            (None, DepthInit::Export) => load_export_tensors(&trainer.source_dir, &device)?,
            (None, DepthInit::Parent) => {
                parent_tensors(&trainer.source_dir, &config, &plan, &device)?
            }
        };
        check_depth_keys(&config, &tensors, "the initial weights")?;
        let prompt = read_prompt(&trainer.prompt, &config, &device)?;
        let base_identity = backbone_identity(&trainer.source_dir, true)?;
        let params = vars(tensors)?;
        let cfg_model = config.model.clone();
        let kind = plan.optimizer.kind;
        let mut opt = IrisOptimizer::new(&plan.optimizer, &params, |k, nd| {
            full_param_route(k, nd, kind, &cfg_model)
        })?;
        let mut ema = match plan.ema {
            Some(_) => Some(detached_copy(&params)?),
            None => None,
        };

        let walk = DataWalk::new(kept.len(), plan.batch_size)?;
        if let Some(warning) = walk.unused_tail_warning() {
            eprintln!("{warning}");
        }
        let total_steps = contract::total_optimizer_steps(
            walk.batches_per_epoch(),
            plan.grad_accum,
            plan.num_epochs,
            plan.max_steps,
        );
        let ckpt_root = checkpoint_root(&req.output_dir, &req.file_name);
        let dataset_fingerprint = depth_dataset_fingerprint(req)?;

        let (mut step, mut sched_step, mut lr, mut nan_count, mut epoch, mut pos, mut last_loss) =
            (0u64, 0u64, plan.optimizer.lr, 0u32, 1u32, 0usize, f64::NAN);
        let resume_dir = match (&plan.resume_from, plan.resume) {
            (Some(p), _) => Some(if p.join(CKPT_STATE).is_file() {
                p.clone()
            } else {
                latest_checkpoint(p).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris depth resume: {} holds no complete checkpoint",
                        p.display()
                    ))
                })?
            }),
            (None, true) => latest_checkpoint(&ckpt_root),
            (None, false) => None,
        };
        let resume_src = resume_dir.clone();
        if let Some(dir) = resume_dir {
            let saved = read_checkpoint_state(&dir)?;
            check_depth_resume(&saved, &plan, &dataset_fingerprint, &base_identity)?;
            let restored = load_tensors(&dir.join(CKPT_TRAINABLE), &device)?;
            if restored.len() != params.len() || params.keys().any(|k| !restored.contains_key(k)) {
                return Err(Error::Msg(
                    "iris depth resume: the checkpoint's tensors do not match this run".into(),
                ));
            }
            for (k, var) in &params {
                let t = &restored[k];
                if t.dims() != var.as_tensor().dims() {
                    return Err(Error::Msg(format!(
                        "iris depth resume: checkpoint tensor {k} is {:?}, this run's is {:?}",
                        t.dims(),
                        var.as_tensor().dims()
                    )));
                }
                var.set(&t.to_dtype(DType::F32)?)?;
            }
            if let Some(e) = ema.as_mut() {
                let saved_ema = load_tensors(&dir.join(CKPT_EMA), &device)?;
                for (k, v) in e.iter_mut() {
                    *v = saved_ema
                        .get(k)
                        .ok_or_else(|| Error::Msg(format!("iris depth resume: the EMA lacks {k}")))?
                        .to_dtype(DType::F32)?;
                }
            }
            opt.load_state(&load_tensors(&dir.join(CKPT_OPTIMIZER), &device)?)?;
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
        // sc-25690: the checkpoints under the root that are not this run's lineage move aside
        // before its first publish (retention and `latest` see only this run's) — once the resume
        // source was validated, so a refused resume leaves the root as it was.
        retire_foreign_checkpoints(&ckpt_root, resume_src.as_deref())?;
        Ok(Self {
            model: DepthModel {
                geometry: Default::default(),
                ac: plan.ac,
                ..DepthModel::new(config, compute, &prompt, &device)?
            },
            plan,
            source_config_text,
            prompt,
            params,
            ema,
            opt,
            kept,
            walk,
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
        let mut last_saved = (self.step > 0).then_some(self.step);
        let mut window = DepthWindow::new(accum);
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
                candle_gen::train::profile::cohort_begin_step(&self.model.device)?;
                let batch = assemble_batch(req, &plan, &self.kept, &self.walk, epoch, pos)?;
                let step = DepthStep::new(&batch, &plan.loss, &self.model.device)?;
                drop(batch);
                micro += 1;
                let sync = micro.is_multiple_of(accum as u64);
                let loss = window.micro(&self.model, &self.params, &step, &plan.loss)?;
                drop(step);
                if !loss.is_finite() {
                    self.nan_count += 1;
                    eprintln!(
                        "iris depth training: non-finite loss at step {} ({}/{})",
                        self.step, self.nan_count, plan.nan_loss_tolerance
                    );
                    if self.nan_count > plan.nan_loss_tolerance {
                        return Err(Error::Msg(format!(
                            "iris depth training: loss was non-finite {} times; aborting",
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
                let Some(_norm) =
                    window.update(&self.params, ema, &mut self.opt, plan.gradient_clip, lr_now)?
                else {
                    continue;
                };
                self.sched_step += 1;
                self.step += 1;
                self.last_loss = mean_loss;
                boundary = (epoch, pos + 1);
                on_progress(TrainingProgress::Training {
                    step: self.micro_step(),
                    total: (self.total_steps * accum as u64) as u32,
                    loss: self.last_loss as f32,
                });
                if plan.save_every > 0 && self.step.is_multiple_of(plan.save_every as u64) {
                    self.save_checkpoint(boundary)?;
                    last_saved = Some(self.step);
                    on_progress(TrainingProgress::Checkpoint {
                        step: self.micro_step(),
                    });
                }
                if plan.preview.every > 0 && self.step.is_multiple_of(plan.preview.every as u64) {
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
            return Err(Error::Canceled);
        }
        if last_saved != Some(self.step) && self.step > 0 {
            self.save_checkpoint(boundary)?;
            on_progress(TrainingProgress::Checkpoint {
                step: self.micro_step(),
            });
        }
        on_progress(TrainingProgress::Saving);
        let path = self.export(req)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: self.micro_step(),
            final_loss: self.last_loss as f32,
        })
    }

    fn micro_step(&self) -> u32 {
        (self.step * self.plan.grad_accum as u64) as u32
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
            artifact: "full".into(),
            optimizer: self.plan.optimizer.kind.as_str().into(),
            ema_decay: self.plan.ema,
            seed: self.plan.seed,
            backend: BACKEND.into(),
            state_identity: self.plan.state_identity(),
            data_identity: self.plan.data_identity(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            base_identity: self.base_identity.clone(),
            task: IrisTask::Depth,
            stage: None,
            parent_stage: None,
            data_position: None,
            dist: None,
            repa_projector: None,
            grad_scaler: None,
        }
    }

    fn save_checkpoint(&self, boundary: (u32, usize)) -> Result<()> {
        let io = |p: &Path, e: std::io::Error| {
            Error::Msg(format!("iris checkpoint: create {}: {e}", p.display()))
        };
        std::fs::create_dir_all(&self.ckpt_root).map_err(|e| io(&self.ckpt_root, e))?;
        let staging = checkpoint_staging_dir(&self.ckpt_root, self.step);
        if staging.exists() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        std::fs::create_dir_all(&staging).map_err(|e| io(&staging, e))?;
        let list = |m: &Tensors| -> Vec<(String, Tensor)> {
            m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        save_tensors(
            &staging.join(CKPT_TRAINABLE),
            &list(&snapshot(&self.params)),
        )?;
        if let Some(ema) = &self.ema {
            save_tensors(&staging.join(CKPT_EMA), &list(ema))?;
        }
        save_tensors(&staging.join(CKPT_OPTIMIZER), &self.opt.state_tensors()?)?;
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

    /// The raw (detached) or EMA tensors.
    fn selected(&self, which: WeightsSelect) -> Tensors {
        match (which, &self.ema) {
            (WeightsSelect::Ema, Some(e)) => e.clone(),
            _ => self
                .params
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
                .collect(),
        }
    }

    /// Write the `depth/` export directory (`<output_dir>/<stem>/`) — see the MLX twin.
    fn export(&self, req: &TrainingRequest) -> Result<PathBuf> {
        let dir = req.output_dir.join(file_stem(&req.file_name));
        let which = self.plan.export_weights;
        let mut list: Vec<(String, Tensor)> = Vec::with_capacity(self.params.len());
        for (k, v) in self.selected(which) {
            let key = if k == REDUCER_WEIGHT || k == REDUCER_BIAS {
                k
            } else {
                format!("{EXPORT_BACKBONE_PREFIX}{k}")
            };
            list.push((key, v.to_dtype(DType::F32)?));
        }
        list.sort_by(|a, b| a.0.cmp(&b.0));
        let yaml = depth_export_config_yaml(&self.source_config_text)?;
        let meta = depth_export_metadata(which, self.step, &self.base_identity)
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        // sc-25690: config, empty prompt and weights appear together or not at all.
        publish_export_dir(&dir, |stage: &Path| -> Result<()> {
            contract::atomic_write(&stage.join(BACKBONE_CONFIG_FILE), yaml.as_bytes())?;
            self.prompt.write(&stage.join(EMPTY_PROMPT_FILE))?;
            save_safetensors_atomic(&stage.join(BACKBONE_WEIGHTS_FILE), &list, &meta)
        })?;
        Ok(dir.join(BACKBONE_WEIGHTS_FILE))
    }

    fn render_previews(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pv = &self.plan.preview;
        let items: Vec<usize> = self.kept.iter().copied().take(pv.items).collect();
        let total = items.len() as u32;
        let (h, w) = (self.plan.augment.height, self.plan.augment.width);
        let tensors = self.selected(pv.weights);
        for (n, &i) in items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                return Ok(());
            }
            let row = preview_row(&req.items[i], &self.plan)?;
            let rgb = Tensor::from_vec(row.rgb_chw, (1, 3, h, w), &self.model.device)?;
            let pred = self.model.predict(&tensors, &rgb)?;
            let map = DepthMap {
                width: w as u32,
                height: h as u32,
                values: pred
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?,
            };
            on_progress(TrainingProgress::Sample {
                step: self.micro_step(),
                index: n as u32 + 1,
                total,
                prompt: req.items[i].image_path.display().to_string(),
                image: colorize_inferno(&map),
            });
        }
        Ok(())
    }
}
