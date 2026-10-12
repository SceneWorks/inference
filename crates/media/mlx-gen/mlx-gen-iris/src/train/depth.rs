//! Native MLX **Iris-3B depth training** (story sc-25687): the supervised direct-regression fine-tune
//! of the depth task (paper §5.2) over the backend-neutral contract
//! [`gen_core::iris::train::depth`], registered as the [`DEPTH_TRAINER_ID`] (`iris_3b_depth`)
//! trainer.
//!
//! Every step runs **the inference forward** ([`crate::depth::depth_forward`]): the widened backbone
//! is rebuilt from the traced f32 masters exactly as the estimator loads an export
//! ([`IrisDiT::from_weights_widened`], the provider's autocast policy), the zero 4th channel is
//! concatenated, one forward at `t = flow.num_train_timesteps` on the empty-prompt conditioning, the
//! 1×1 reducer in the compute dtype. The loss is Eq. 5 on tensors (the contract's
//! [`depth_loss_host`](gen_core::iris::train::depth::depth_loss_host) is its reference), its gradient
//! of `loss / grad_accum` over every trained tensor; each optimizer step clips the global norm,
//! updates the optional EMA from the pre-step weights and steps AdamW (or the Muon hybrid) — the
//! generation trainer's optimizer, clipping, EMA and checkpoint code ([`super::optim`],
//! `gen_core::iris::train`).
//!
//! A run starts from a `depth/` export (`LoadSpec::weights`; no other resource) or from the
//! **generation** backbone (the paper's parent; input projections widened with zero columns, the
//! reducer per `reducer_init`), whose empty-prompt conditioning comes from the
//! [`EMPTY_PROMPT_COMPONENT`] file or is encoded once by the `text_encoder` component (then
//! dropped). It ends with a `depth/` export directory in exactly `scripts/export_downstream.py`'s
//! layout that [`crate::depth::load`] runs unchanged.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use gen_core::iris::depth::{colorize_inferno, DepthMap};
use gen_core::iris::downstream::{parse_task_settings, EmptyPrompt, TaskExport, EMPTY_PROMPT_FILE};
use gen_core::iris::train::depth::{
    assemble_batch, check_depth_resume, depth_dataset_fingerprint, depth_dataset_pass,
    depth_export_config_yaml, depth_export_metadata, depth_tensor_shapes, preview_row,
    reducer_init_values, validate_depth_items, widened_projections, DepthBatch, DepthInit,
    DepthLossPlan, DepthTrainPlan, DEPTH_TRAINER_ID, EMPTY_PROMPT_COMPONENT,
    EXPORT_BACKBONE_PREFIX, REDUCER_BIAS, REDUCER_WEIGHT,
};
use gen_core::iris::train::lineage::{publish_export_dir, retire_foreign_checkpoints};
use gen_core::iris::train::{
    self as contract, backbone_identity, checkpoint_root, checkpoint_staging_dir, file_stem,
    full_param_route, latest_checkpoint, lr_factor, prune_checkpoints, publish_checkpoint,
    read_checkpoint_state, CheckpointState, DataWalk, MixedPrecision, ResumeDataPolicy,
    WeightsSelect, CKPT_EMA, CKPT_OPTIMIZER, CKPT_STATE, CKPT_TRAINABLE, TECHNIQUES,
};
use gen_core::iris::{
    IrisConfig, IrisTask, BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE, FAMILY,
    TEXT_ENCODER_COMPONENT,
};
use gen_core::train::{
    refuse_reference_control_model_options, refuse_trainer_load_overlays, validate_control_request,
    validate_edit_request, validate_full_finetune_request, validate_training_techniques, Trainer,
    TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, LoadSpec, Modality, Result, WeightsSource};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{abs, concatenate_axis, sum_axes, zeros};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{Array, Dtype};

use super::model::save_safetensors_atomic;
use super::optim::{clip_grads, ema_update, IrisOptimizer, Params};
use crate::depth::{depth_forward, DepthParts};
use crate::dit::IrisDiT;
use crate::text_encoder::IrisTextEncoder;

/// Identity + capabilities of the MLX Iris depth trainer: a full fine-tune, no adapters.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: DEPTH_TRAINER_ID,
        family: FAMILY,
        backend: "mlx",
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
    /// The `depth/` export's own `empty_prompt.safetensors`.
    Export(PathBuf),
    /// An `empty_prompt.safetensors` staged as the [`EMPTY_PROMPT_COMPONENT`].
    File(PathBuf),
    /// Encode `""` once with the staged Qwen3-VL snapshot.
    Encoder(PathBuf),
}

/// The resolved resources of a depth-training load.
pub struct IrisDepthTrainer {
    descriptor: TrainerDescriptor,
    /// The `depth/` export or the generation backbone directory.
    source_dir: PathBuf,
    init: DepthInit,
    config: IrisConfig,
    prompt: PromptSource,
}

/// Construct the depth trainer: `spec.weights` is a `depth/` export (no other resource may be
/// staged — the task needs no text encoder) or the generation backbone directory plus its
/// empty-prompt conditioning (the [`EMPTY_PROMPT_COMPONENT`] file, or the `text_encoder`
/// component to encode it once). Nothing heavy loads here.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let (source_dir, init, config, prompt) = resolve_resources(spec)?;
    Ok(Box::new(IrisDepthTrainer {
        descriptor: trainer_descriptor(),
        source_dir,
        init,
        config,
        prompt,
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
    let text = std::fs::read_to_string(dir.join(BACKBONE_CONFIG_FILE)).map_err(|e| {
        Error::Msg(format!(
            "{id} trainer: read {}: {e}",
            dir.join(BACKBONE_CONFIG_FILE).display()
        ))
    })?;
    if parse_task_settings(&text)?.is_some() {
        // A task export: it must be the depth export, and it carries its own conditioning.
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
    Ok((dir, DepthInit::Parent, config, prompt))
}

mlx_gen::register_trainer! {
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

/// The trainable depth model: every tensor of [`depth_tensor_shapes`] (f32 masters), the frozen
/// empty-prompt conditioning, the compute dtype.
pub struct DepthModel {
    pub config: IrisConfig,
    pub compute: Dtype,
    /// `[1, T, L, D]` f32 and its `[T]` mask.
    pub embeddings: Array,
    pub mask: Vec<i32>,
}

impl DepthModel {
    pub fn new(config: IrisConfig, compute: Dtype, prompt: &EmptyPrompt) -> Self {
        let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
        Self {
            config,
            compute,
            embeddings: Array::from_slice(&prompt.embeddings, &shape),
            mask: prompt.mask.clone(),
        }
    }

    /// `[B, 3, H, W]` RGB → `[B, 1, H, W]` f32 prediction from `params` (traced or plain), through
    /// the estimator's own forward.
    pub fn predict(&self, params: &HashMap<Rc<str>, Array>, rgb: &Array) -> Result<Array> {
        let cfg = &self.config.model;
        let mut backbone = HashMap::with_capacity(params.len());
        for (k, v) in params {
            if k.as_ref() != REDUCER_WEIGHT && k.as_ref() != REDUCER_BIAS {
                backbone.insert(k.to_string(), v.clone());
            }
        }
        let dit = IrisDiT::from_weights_widened(
            &Weights::from_map(backbone),
            cfg,
            self.compute,
            cfg.in_channels + gen_core::iris::depth::DEPTH_EXTRA_INPUT_CHANNELS,
        )?;
        let get = |k: &str| {
            params
                .get(k)
                .ok_or_else(|| Error::Msg(format!("iris depth: {k} is not trained")))
        };
        let weight = get(REDUCER_WEIGHT)?
            .reshape(&[1, cfg.in_channels as i32])?
            .as_dtype(self.compute)?;
        let bias = get(REDUCER_BIAS)?.as_dtype(self.compute)?;
        depth_forward(
            &DepthParts {
                dit: &dit,
                reducer_weight: &weight,
                reducer_bias: &bias,
                embeddings: &self.embeddings,
                mask: &self.mask,
                num_train_timesteps: self.config.flow.num_train_timesteps,
            },
            rgb,
        )
    }
}

/// One micro-batch on the device.
pub struct DepthStep {
    pub rgb: Array,
    /// `[B, 1, H, W]` f32 target and 0/1 validity.
    pub target: Array,
    pub valid: Array,
    /// `[B]` f32 row weights ([`DepthBatch::row_weights`]).
    pub row_weights: Array,
    pub rows: usize,
}

impl DepthStep {
    pub fn new(b: &DepthBatch, loss: &DepthLossPlan) -> Self {
        let (n, h, w) = (b.batch as i32, b.height as i32, b.width as i32);
        Self {
            rgb: Array::from_slice(&b.rgb, &[n, 3, h, w]),
            target: Array::from_slice(&b.target, &[n, 1, h, w]),
            valid: Array::from_slice(&b.valid, &[n, 1, h, w]),
            row_weights: Array::from_slice(&b.row_weights(loss.reduction), &[n]),
            rows: b.batch,
        }
    }
}

/// Eq. 5 on tensors: per row `Σ_{p∈M} |e| + λ(Σ |Δx e| + Σ |Δy e|)` with `e = d̂ − d` (so
/// `Δe = Δd̂ − Δd`) and the both-valid pair masks, weighted by the row weights and summed (f32).
pub fn depth_loss(pred: &Array, step: &DepthStep, gradient_weight: f64) -> Result<Array> {
    let e = pred.subtract(&step.target)?;
    let v = &step.valid;
    let value = sum_axes(&abs(&e)?.multiply(v)?, &[1, 2, 3], false)?;
    // Explicit task-local stream: `IndexOp::index` targets the default device (the GPU on a Mac)
    // whatever the task-local stream says.
    let st = mlx_rs::Stream::task_local_or_default();
    let dx = e
        .index_device((.., .., .., 1..), &st)
        .subtract(e.index_device((.., .., .., ..-1), &st))?;
    let vx = v
        .index_device((.., .., .., 1..), &st)
        .multiply(v.index_device((.., .., .., ..-1), &st))?;
    let dy = e
        .index_device((.., .., 1.., ..), &st)
        .subtract(e.index_device((.., .., ..-1, ..), &st))?;
    let vy = v
        .index_device((.., .., 1.., ..), &st)
        .multiply(v.index_device((.., .., ..-1, ..), &st))?;
    let grad = sum_axes(&abs(&dx)?.multiply(&vx)?, &[1, 2, 3], false)?.add(&sum_axes(
        &abs(&dy)?.multiply(&vy)?,
        &[1, 2, 3],
        false,
    )?)?;
    let rows = value.add(&grad.multiply(Array::from_f32(gradient_weight as f32))?)?;
    Ok(rows.multiply(&step.row_weights)?.sum(None)?)
}

fn to_exc(e: impl std::fmt::Display) -> Exception {
    Exception::custom(e.to_string())
}

/// The loss of one micro-batch and its gradient over every trained tensor; the gradient is of
/// `loss · grad_scale`, the returned loss unscaled.
pub fn loss_and_grads(
    model: &DepthModel,
    params: &Params,
    step: &DepthStep,
    loss: &DepthLossPlan,
    grad_scale: f32,
) -> Result<(f32, Params)> {
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        let pred = model.predict(&p, &step.rgb).map_err(to_exc)?;
        let l = depth_loss(&pred, step, loss.gradient_weight).map_err(to_exc)?;
        let scaled = l.multiply(Array::from_f32(grad_scale))?;
        Ok(vec![scaled, l])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (vals, grads) =
        vg(params.clone(), 0).map_err(|e| Error::Msg(format!("iris depth train step: {e}")))?;
    let l = vals[1].clone();
    let mut refs: Vec<&Array> = vec![&l];
    refs.extend(grads.values());
    for chunk in refs.chunks(256) {
        eval(chunk.iter().copied())?;
    }
    Ok((super::scalar_f32(&l)?, grads))
}

/// One accumulation window (the generation trainer's `Window` over the depth loss): micro-batches
/// add the gradient of `loss / grad_accum`; closing it clips, updates the EMA from the pre-step
/// weights, steps the optimizer.
pub struct DepthWindow {
    accum: usize,
    grads: Option<Params>,
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

// =============================================================================================
// Init and loading
// =============================================================================================

/// Every tensor of a safetensors file as f32 (each read on the CPU before any cast — sc-24245).
fn load_f32(path: &Path) -> Result<HashMap<String, Array>> {
    let mut w = Weights::from_file(path)
        .map_err(|e| Error::Msg(format!("iris depth training: load {}: {e}", path.display())))?;
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut out = HashMap::with_capacity(keys.len());
    for k in keys {
        let a = w.remove(&k).expect("listed key");
        mlx_rs::transforms::eval_pending_loads([&a])?;
        let a = a.as_dtype(Dtype::Float32)?;
        eval([&a])?;
        out.insert(k, a);
    }
    Ok(out)
}

/// The trained tensors of a `depth/` export (backbone keys without the `pixel.` prefix).
pub fn load_export_tensors(dir: &Path) -> Result<HashMap<String, Array>> {
    let raw = load_f32(&dir.join(BACKBONE_WEIGHTS_FILE))?;
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

/// Every key of `got` must be exactly the depth model's, with its shapes.
fn check_depth_keys(cfg: &IrisConfig, got: &HashMap<String, Array>, what: &str) -> Result<()> {
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
        let s: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        if s != shape {
            return Err(Error::Msg(format!(
                "iris depth training: {what} {k} is {s:?}, the depth model's is {shape:?}"
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
) -> Result<HashMap<String, Array>> {
    let mut t = load_f32(&dir.join(BACKBONE_WEIGHTS_FILE))?;
    let want = contract::backbone_tensor_shapes(&cfg.model);
    if want.len() != t.len() || want.iter().any(|(k, _)| !t.contains_key(k)) {
        return Err(Error::Msg(format!(
            "iris depth training: {} is not the configured generation backbone",
            dir.display()
        )));
    }
    for (key, cols) in widened_projections(&cfg.model) {
        let w = t.remove(key).expect("checked");
        let rows = w.shape()[0];
        let z = zeros::<f32>(&[rows, cols as i32])?;
        let wide = concatenate_axis(&[&w, &z], 1)?;
        eval([&wide])?;
        t.insert(key.to_string(), wide);
    }
    let c = cfg.model.in_channels;
    let (w, b) = reducer_init_values(plan.reducer_init, c, plan.seed);
    t.insert(
        REDUCER_WEIGHT.into(),
        Array::from_slice(&w, &[1, c as i32, 1, 1]),
    );
    t.insert(REDUCER_BIAS.into(), Array::from_slice(&b, &[1]));
    Ok(t)
}

fn rc_map(m: HashMap<String, Array>) -> Params {
    m.into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect()
}

fn read_prompt(source: &PromptSource, config: &IrisConfig) -> Result<EmptyPrompt> {
    match source {
        PromptSource::Export(p) | PromptSource::File(p) => Ok(EmptyPrompt::read(p, config)?),
        PromptSource::Encoder(dir) => {
            let te = IrisTextEncoder::load(dir, &config.text_encoder)?;
            let window = te.window("")?;
            let c = te.encode_window(&window)?;
            let states = c.states.as_dtype(Dtype::Float32)?;
            eval([&states])?;
            let shape: Vec<usize> = states.shape().iter().map(|&d| d as usize).collect();
            let prompt = EmptyPrompt {
                embeddings: states.as_slice::<f32>().to_vec(),
                shape: [shape[0], shape[1], shape[2], shape[3]],
                mask: c.mask,
            };
            drop(te);
            mlx_rs::memory::clear_cache();
            Ok(prompt)
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
    ema: Option<Params>,
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

impl Run {
    fn prepare(
        trainer: &IrisDepthTrainer,
        plan: DepthTrainPlan,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Self> {
        on_progress(TrainingProgress::Preparing);
        let config = trainer.config.clone();
        let compute = match plan.mixed_precision {
            MixedPrecision::Bf16 => Dtype::Bfloat16,
            MixedPrecision::Fp32 => Dtype::Float32,
            // Only the generation plan resolves fp16 (sc-25689); this task's plan never does.
            MixedPrecision::Fp16 => {
                return Err(Error::Unsupported(
                    "iris training: fp16 mixed precision is the generation trainer's".into(),
                ))
            }
        };
        // The dataset pass first (one `Caching` event per item): every example decodes, aligns and
        // is valid (or is skipped under `invalid_examples: skip`) before any weight is read.
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
                    load_f32(&p.join(CKPT_TRAINABLE))?
                } else {
                    TaskExport::from_dir(p, IrisTask::Depth, DEPTH_TRAINER_ID)?;
                    load_export_tensors(p)?
                }
            }
            (None, DepthInit::Export) => load_export_tensors(&trainer.source_dir)?,
            (None, DepthInit::Parent) => parent_tensors(&trainer.source_dir, &config, &plan)?,
        };
        check_depth_keys(&config, &tensors, "the initial weights")?;
        let prompt = read_prompt(&trainer.prompt, &config)?;
        let base_identity = backbone_identity(&trainer.source_dir, true)?;
        let params = rc_map(tensors);
        let cfg_model = config.model.clone();
        let kind = plan.optimizer.kind;
        let mut opt = IrisOptimizer::new(&plan.optimizer, &params, |k, nd| {
            full_param_route(k, nd, kind, &cfg_model)
        })?;
        let mut ema = plan.ema.map(|_| params.clone());

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

        let mut run_params = params;
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
            let restored = load_tensors(&dir.join(CKPT_TRAINABLE))?;
            if restored.len() != run_params.len()
                || run_params
                    .keys()
                    .any(|k| !restored.contains_key(k.as_ref()))
            {
                return Err(Error::Msg(
                    "iris depth resume: the checkpoint's tensors do not match this run".into(),
                ));
            }
            run_params = rc_map(restored);
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
        // sc-25690: the checkpoints under the root that are not this run's lineage move aside
        // before its first publish (retention and `latest` see only this run's) — once the resume
        // source was validated, so a refused resume leaves the root as it was.
        retire_foreign_checkpoints(&ckpt_root, resume_src.as_deref())?;
        Ok(Self {
            model: DepthModel::new(config, compute, &prompt),
            plan,
            source_config_text,
            prompt,
            params: run_params,
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
                let batch = assemble_batch(req, &plan, &self.kept, &self.walk, epoch, pos)?;
                let step = DepthStep::new(&batch, &plan.loss);
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
            backend: "mlx".into(),
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

    /// Write the `depth/` export directory (`<output_dir>/<stem>/`): `config.yaml` (the source's
    /// inference sections + `task: depth`), `model.safetensors` (FP32, `pixel.*` + reducer, the
    /// task-identity stamps), `empty_prompt.safetensors`. Returns the weights file.
    fn export(&self, req: &TrainingRequest) -> Result<PathBuf> {
        let dir = req.output_dir.join(file_stem(&req.file_name));
        let which = self.plan.export_weights;
        let mut list: Vec<(String, Array)> = Vec::with_capacity(self.params.len());
        for (k, v) in self.selected(which) {
            let key = if k.as_ref() == REDUCER_WEIGHT || k.as_ref() == REDUCER_BIAS {
                k.to_string()
            } else {
                format!("{EXPORT_BACKBONE_PREFIX}{k}")
            };
            list.push((key, v.as_dtype(Dtype::Float32)?));
        }
        list.sort_by(|a, b| a.0.cmp(&b.0));
        let yaml = depth_export_config_yaml(&self.source_config_text)?;
        let meta = depth_export_metadata(which, self.step, &self.base_identity);
        // sc-25690: config, empty prompt and weights appear together or not at all.
        publish_export_dir(&dir, |stage: &Path| -> Result<()> {
            contract::atomic_write(&stage.join(BACKBONE_CONFIG_FILE), yaml.as_bytes())?;
            self.prompt.write(&stage.join(EMPTY_PROMPT_FILE))?;
            save_safetensors_atomic(&stage.join(BACKBONE_WEIGHTS_FILE), &list, &meta)
        })?;
        Ok(dir.join(BACKBONE_WEIGHTS_FILE))
    }

    /// Predict the first `preview.items` dataset items (center crop, no flip) from the in-progress
    /// weights and report each as upstream's `colorize` (near = bright).
    fn render_previews(
        &self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        let pv = &self.plan.preview;
        let items: Vec<usize> = self.kept.iter().copied().take(pv.items).collect();
        let total = items.len() as u32;
        let (h, w) = (self.plan.augment.height, self.plan.augment.width);
        for (n, &i) in items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                return Ok(());
            }
            let row = preview_row(&req.items[i], &self.plan)?;
            let rgb = Array::from_slice(&row.rgb_chw, &[1, 3, h as i32, w as i32]);
            let pred = self.model.predict(self.selected(pv.weights), &rgb)?;
            eval([&pred])?;
            let map = DepthMap {
                width: w as u32,
                height: h as u32,
                values: pred.reshape(&[-1])?.as_slice::<f32>().to_vec(),
            };
            on_progress(TrainingProgress::Sample {
                step: self.micro_step(),
                index: n as u32 + 1,
                total,
                prompt: req.items[i].image_path.display().to_string(),
                image: colorize_inferno(&map),
            });
        }
        mlx_rs::memory::clear_cache();
        Ok(())
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
